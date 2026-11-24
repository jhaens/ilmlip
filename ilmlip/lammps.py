"""ilmlip in LAMMPS, through ML-IAP (``pair_style mliap unified``), evaluated in JAX.

Export a checkpoint once,

    python -m ilmlip.lammps model.pt model.mliap [--dtype auto] [--kernels auto]
                                                        [--charge 0 --spin 1]

and use the file in LAMMPS built with KOKKOS (CUDA), ML-IAP, ML-SNAP, PYTHON and MLIAP_ENABLE_PYTHON,
with this package, JAX and CuPy importable by its Python:

    newton on
    pair_style mliap unified model.mliap 0
    pair_coeff * * Na Cl
    # lmp -k on g 1 -sf kk -pk kokkos newton on neigh half -in in.lmp

Every step LAMMPS hands over the vectors r_ij of all pairs with a local atom and takes back the energy
and dE/dr_ij per pair; it assembles forces and virial itself.  Between the layers, messages reach the
ghost atoms through LAMMPS's forward exchange, and their gradients return through the reverse exchange.

Kernels: "fused" (cuEquivariance), "xla" (plain XLA), "fused-reference" (the fused polynomials in plain
JAX, for tests) or "auto" (fused when cuEquivariance runs here), all on a dense neighbour table, as the
JAX steps.  dtype: "auto" (the model's training precision: float64 or tf32), "tf32" (float32, the
network's matrix products in TensorFloat-32 on GPUs that have it), "float32" or "float64".
"""
import os
import pickle
import warnings

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")      # Kokkos shares the GPU
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from .jax_md import (_cast, _core_switch, _energy_fns, _method, _pair_tables, _zbl_pairs,  # noqa: E402
                     fused_available, tables)
from .model import DTYPES  # noqa: E402
from .verlet import check_energy, padded_size  # noqa: E402
from . import vcut  # noqa: E402

try:
    from lammps.mliap.mliap_unified_abc import MLIAPUnified
except ImportError:                          # exporting without LAMMPS
    MLIAPUnified = object

KERNELS = ("auto", "fused", "xla", "fused-reference")


def _table(rij, pi, pj, n_pairs, nlocal, n_rows, width, rcut):
    """The local atoms' pairs as a dense (n_rows, width) table of vectors and neighbour
    indices.  Empty slots point to the atom itself at 2 rcut, where they contribute nothing.
    Also returns the largest neighbour count."""
    P = rij.shape[0]
    local = (jnp.arange(P) < n_pairs) & (pi < nlocal)
    key = jnp.where(local, pi, n_rows)
    order = jnp.argsort(key, stable=True)
    key = key[order]
    cnt = jax.ops.segment_sum(local.astype(jnp.int32), jnp.where(local, pi, 0), num_segments=n_rows)
    slot = jnp.arange(P) - (jnp.cumsum(cnt) - cnt)[jnp.minimum(key, n_rows - 1)]
    ok = (key < n_rows) & (slot < width)
    rows, cols = jnp.where(ok, key, n_rows), jnp.where(ok, slot, 0)
    vec = jnp.broadcast_to(jnp.asarray([2.0 * rcut, 0.0, 0.0], rij.dtype), (n_rows, width, 3))
    vec = vec.at[rows, cols].set(rij[order], mode="drop")
    nbr = jnp.broadcast_to(jnp.arange(n_rows, dtype=pj.dtype)[:, None], (n_rows, width))
    nbr = nbr.at[rows, cols].set(pj[order], mode="drop")
    return vec, nbr, jnp.max(cnt)


def _stages(T, dt, method):
    """The model on a dense table with either kernels: the parts of _energy_fns between the exchanges.
    The back-flow and the per-pair weights live on the bond, so only the messages cross to the ghosts;
    with bound, the first messages carry the sender's gate f0 as one more column (the back-flow's factor)."""
    p, rcut, n_messages = _energy_fns(T, dt, method)[3], T["rcut"], T["n_layers"] - 1
    gated, zc = bool(T.get("bound")), T.get("zbl_switch")

    def stages(n_rows, width):
        blocks = ((n_rows, width),)
        ind_i = jnp.repeat(jnp.arange(n_rows, dtype=jnp.int32), width)

        def layer(t, geo, aux, sp, I_all):
            (_, g, Y), (nbr, pair) = geo, aux[:2]
            f0 = None
            if gated and t == 0:
                I_all, f0 = I_all[:, :-1], I_all[:, -1]
            return p["receive"](t, I_all, g, Y, p["bond_radials"](g, pair, sp["H"], t), blocks, ind_i, nbr, f0)

        def front(rij, pi, pj, npairs, nlocal, sp):
            mu, H = sp["mu"], sp["H"]
            ls = sp["lookup"][mu]
            pi = vcut.lammps_centres(T, rij, pi, pj, mu, n_rows)
            vec, nbr, kmax = _table(rij, pi, pj, npairs, nlocal, n_rows, width, rcut)
            nbr = nbr.reshape(-1)
            r, g, Y = p["geometry"](vec.reshape(-1, 3).astype(dt), mu[ind_i], mu[nbr])
            pair = ls[ind_i] * H["present"].shape[0] + ls[nbr]
            F, f0 = p["first_layer"](g, Y, pair, ls[:n_rows], H, blocks, ind_i)
            msg = p["send"](F, 0)
            if gated:
                msg = jnp.concatenate([msg, f0[:, None]], axis=1)
            return (msg, (r, g, Y), p["read"](None, F)), (nbr, pair, kmax)

        def mid(t):
            def stage(geo, aux, sp, I_all, h, nlocal):
                _, F = layer(t - 1, geo, aux, sp, I_all)
                return p["send"](F, t), h + p["read"](t - 1, F)
            return stage

        def back(geo, aux, sp, I_all, h, nlocal):
            mu, nbr = sp["mu"], aux[0]
            last, _ = layer(n_messages - 1, geo, aux, sp, I_all)
            e = p["finish"](h, last, mu, n_rows)
            if T["zbl"] is not None:
                z = jnp.sum(_zbl_pairs(T["zbl"], geo[0], mu[ind_i], mu[nbr]).reshape(n_rows, width), axis=1)
                e = (e * _core_switch(z, zc) if zc else e) + z
            e = e * ((jnp.arange(n_rows) < nlocal) & (mu[:n_rows] >= 0)).astype(dt)
            return jnp.sum(e), e

        return front, [mid(t) for t in range(1, n_messages)], back
    return stages


def _most_neighbours(tables, data, xp, nlocal):
    """The largest number of pairs around a local atom: the table's width on the first step."""
    n = int(data.npairs)
    pi = xp.asarray(data.pair_i)[:n]
    keep = pi < nlocal
    v = tables.get("vcut")
    if v is not None:
        e, rij = xp.asarray(data.elems), xp.asarray(data.rij)[:n]
        keep &= (rij * rij).sum(1) < xp.asarray(v["rc2"])[e[pi], e[xp.asarray(data.pair_j)[:n]]]
    return int(xp.bincount(pi[keep], minlength=1).max())


def _xp(a):
    """cupy for LAMMPS's device arrays (KOKKOS), numpy otherwise."""
    if type(a).__module__.startswith("cupy"):
        import cupy
        return cupy
    return np


class ILMlipMLIAP(MLIAPUnified):
    """The ML-IAP unified model: tables of a checkpoint, compiled on first use."""

    def __init__(self, tables, dtype="tf32", kernels="auto"):
        super().__init__()
        if kernels not in KERNELS:
            raise ValueError(f"kernels must be one of {KERNELS}, got {kernels!r}")
        if str(dtype) not in DTYPES:
            raise ValueError(f"dtype must be one of {'|'.join(DTYPES)}, got {dtype!r}")
        self.precision = "default" if str(dtype) == "tf32" else "highest"
        self.tables, self.kernels = tables, kernels
        self.dtype = "float32" if str(dtype) == "tf32" else str(dtype)
        self.element_types = list(tables["species"])
        self.ndescriptors = self.nparams = 1
        self.rcutfac = 0.5 * tables["rcut"]        # LAMMPS uses twice this as the pair cutoff
        self._run = None

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_run"] = None                     # compiled functions are rebuilt where loaded
        return state

    def __setstate__(self, state):
        tables = state["tables"]
        if "arch" not in tables:
            raise RuntimeError("a model file this ilmlip cannot read: write it again with "
                               "python -m ilmlip.lammps model.pt model.mliap")
        self.__dict__.update(state)

    def compute_descriptors(self, data):
        pass

    def compute_gradients(self, data):
        pass

    def _build(self, kernels):
        if self.dtype == "float64":
            jax.config.update("jax_enable_x64", True)
        dt = jnp.dtype(self.dtype)
        T = _cast(self.tables, np.dtype(self.dtype))
        T["readout"] = [(jnp.asarray(w, dt), jnp.asarray(b, dt), act) for w, b, act in self.tables["readout"]]
        if kernels == "auto":
            kernels = "fused" if fused_available() else "xla"
        if T["n_layers"] < 2:
            raise ValueError("the LAMMPS interface needs a model with two or more layers")
        stages = _stages(T, dt, None if kernels == "xla" else _method(kernels))
        precision = getattr(self, "precision", "highest")
        self._run = dict(stages=stages, kernels=kernels, precision=precision, fns={}, sizes={},
                         width=None, ok=False, T=T)
        print(f"ilmlip: {kernels} kernels, {self.dtype}, matmul precision {precision}", flush=True)

    def _fns(self, n_rows, width):
        """The compiled stages of one shape: front and middle stages forward, with their backward
        as a closure; the last stage forward and backward in one call."""
        fns = self._run["fns"]
        if (n_rows, width) not in fns:
            front, mids, back = self._run["stages"](n_rows, width)

            def last(geo, aux, mu_all, I_all, h, nlocal):
                (E, e), f = jax.vjp(lambda g_, I_, h_: back(g_, aux, mu_all, I_, h_, nlocal), geo, I_all, h)
                return E, e, f((jnp.ones((), E.dtype), jnp.zeros_like(e)))

            def acc(f, ct, d_geo):
                dg, dI, dh = f(ct)
                return jax.tree_util.tree_map(jnp.add, dg, d_geo), dI, dh

            fns[(n_rows, width)] = dict(
                front=jax.jit(lambda rij, pi, pj, npairs, nlocal, mu_all: jax.vjp(
                    lambda x: front(x, pi, pj, npairs, nlocal, mu_all), rij, has_aux=True)),
                mids=[jax.jit(lambda geo, aux, mu_all, I, h, nlocal, s=s: jax.vjp(
                    lambda g_, I_, h_: s(g_, aux, mu_all, I_, h_, nlocal), geo, I, h)) for s in mids],
                last=jax.jit(last), acc=jax.jit(acc), grad=jax.jit(lambda f, ct: f(ct)[0]))
        return fns[(n_rows, width)]

    def _species(self, mu_all, data, xp):
        """mu_all with _pair_tables of the species seen so far and each species' index among them
        (atoms that migrate bring new species to this rank; a NULL type, -1, reads the last entry)."""
        run, S = self._run, len(self.element_types)
        elems = xp.asarray(data.elems)
        if "seen" not in run:
            run["seen"] = xp.zeros(S + 1, bool)
            run["seen"][S] = True
        if not bool(run["seen"][elems].all()):
            run["seen"][elems] = True
            seen = xp.asnumpy(run["seen"]) if xp is not np else run["seen"]
            present = np.flatnonzero(seen[:S]).astype(np.int32)
            padded = np.full(1 << int(np.ceil(np.log2(len(present)))), S, np.int32)
            padded[:len(present)] = present
            lookup = np.zeros(S + 1, np.int32)
            lookup[present] = np.arange(len(present))
            run["lookup"], run["pairs"] = jnp.asarray(lookup), _pair_tables(run["T"], jnp.asarray(padded))
        return dict(mu=mu_all, lookup=run["lookup"], H=run["pairs"])

    def compute_forces(self, data):
        if self._run is None:
            self._build(getattr(self, "kernels", "auto"))
        run = self._run
        nlocal, ntotal, npairs = int(data.nlocal), int(data.ntotal), int(data.npairs)
        if not hasattr(data, "forward_exchange"):
            raise RuntimeError("ilmlip passes messages between ghost atoms: run LAMMPS with "
                               "KOKKOS (-k on g 1 -sf kk), whose ML-IAP has forward_exchange")
        if nlocal == 0:              # no atoms here, but every rank must exchange equally often
            data.energy = 0.0
            xp, V, n = _xp(data.elems), self.tables["arch"], self.tables["n_layers"] - 1
            widths = [(V["lmax_ind"] + 1) ** 2 * V["n_message"]] * n
            widths[0] += bool(self.tables.get("bound"))             # the first message carries f0
            for exchange, ws in ((data.forward_exchange, widths), (data.reverse_exchange, widths[::-1])):
                for w in ws:
                    buf = xp.zeros((ntotal, w), self.dtype)
                    exchange(buf, xp.empty_like(buf), w)
            return
        xp = _xp(data.rij)
        vcut.lammps_check(self, data, xp)
        # LAMMPS (Kokkos) and JAX use different CUDA streams, and LAMMPS does not fence here: sync both ways
        sync = xp.cuda.runtime.deviceSynchronize if xp is not np else (lambda: None)
        to_jax = (lambda a: jax.dlpack.from_dlpack(a)) if xp is not np else jnp.asarray
        to_xp = (lambda a: xp.from_dlpack(a.block_until_ready())) if xp is not np else np.array
        sync()
        # sizes seen by JAX only grow (padded_size): few compiled shapes; padding and cutting run on CuPy
        sizes = run["sizes"]

        def grown(key, n, fit):
            if n > sizes.get(key, 0):
                sizes[key] = fit(n)
            return sizes[key]

        n_rows = grown("rows", nlocal, lambda n: padded_size(n, per_octave=16, granule=8))
        P = grown("pairs", max(npairs, 1), lambda n: padded_size(1.03 * n))
        G = max(grown("atoms", ntotal, lambda n: padded_size(1.03 * n)), n_rows)

        def padded(a, n, fill, dtype):
            buf = xp.full((n,) + tuple(a.shape[1:]), fill, dtype)
            buf[:a.shape[0]] = a
            return to_jax(buf)

        rij = padded(data.rij, P, 0.0, np.dtype(self.dtype))
        pi = padded(data.pair_i, P, n_rows, np.int32)
        pj = padded(data.pair_j, P, 0, np.int32)
        mu_all = padded(data.elems, G, 0, np.int32)
        mu_all = self._species(mu_all, data, xp)

        def exchange(forward, x):
            """x (rows, ...) through LAMMPS: ghosts get their owners' rows (forward; returns G rows),
            or the ghosts' rows are added to their owners' (reverse; returns the n_rows local ones)."""
            F, m = int(np.prod(x.shape[1:])), min(ntotal, x.shape[0])
            src_buf = xp.zeros((ntotal, F), x.dtype)
            src_buf[:m] = to_xp(x).reshape(x.shape[0], F)[:m]
            dst_buf = xp.empty_like(src_buf)
            (data.forward_exchange if forward else data.reverse_exchange)(src_buf, dst_buf, F)
            sync()
            rows = G if forward else n_rows
            out = xp.zeros((rows, F), x.dtype)
            out[:min(rows, ntotal)] = dst_buf[:rows]
            return to_jax(out).reshape((rows,) + x.shape[1:])

        try:
            with jax.default_matmul_precision(run["precision"]):
                if run["width"] is None:
                    run["width"] = padded_size(_most_neighbours(self.tables, data, xp, nlocal) + 2, granule=2)
                while True:                  # an overflowing table is redone before any exchange
                    f = self._fns(n_rows, run["width"])
                    (msg, geo, h), vjp_front, aux = f["front"](rij, pi, pj, npairs, nlocal, mu_all)
                    kmax = int(aux[-1])
                    if kmax <= run["width"]:
                        break
                    run["width"] = padded_size(kmax + 1, granule=4)
                vjps = []
                for f_mid in f["mids"]:
                    (msg, h), vjp_mid = f_mid(geo, aux, mu_all, exchange(True, msg), h, nlocal)
                    vjps.append(vjp_mid)
                _, e_site, (d_geo, d_I, d_h) = f["last"](geo, aux, mu_all, exchange(True, msg), h, nlocal)
                for vjp_mid in reversed(vjps):        # owners get their ghosts' share back
                    d_geo, d_I, d_h = f["acc"](vjp_mid, (exchange(False, d_I), d_h), d_geo)
                g_rij = f["grad"](vjp_front, (exchange(False, d_I), d_geo, d_h))
        except Exception as exc:
            if run["ok"] or run["kernels"] == "xla":
                raise
            warnings.warn(f"ilmlip: {run['kernels']} kernels failed ({type(exc).__name__}: "
                          f"{str(exc)[:200]}); using xla")
            self._build("xla")
            return self.compute_forces(data)
        run["ok"] = True
        # the E0s and the energy in float64: the E0s dwarf the network's part of the energy
        if "e0" not in run:
            run["e0"] = xp.asarray(np.asarray(self.tables["atomic_energies"], np.float64))
        mu = xp.asarray(data.elems)[:nlocal]
        e_site = to_xp(e_site)[:nlocal].astype(xp.float64) + xp.where(mu >= 0, run["e0"][mu], 0.0)
        energy = float(e_site.sum())
        check_energy(energy)
        data.energy = energy                     # write-only in LAMMPS
        eatoms = xp.asarray(data.eatoms) if xp is not np else data.eatoms
        if eatoms is not None and len(eatoms) >= nlocal:
            eatoms[:nlocal] = e_site
        out = xp.ascontiguousarray(to_xp(g_rij)[:npairs], dtype=xp.float64)   # LAMMPS reads doubles
        (getattr(data, "update_pair_forces_gpu", None) or data.update_pair_forces)(out)
        sync()                                   # LAMMPS has read `out` before JAX may reuse it


def export(path, out, dtype="auto", charge=None, spin=None, kernels="auto"):
    """Write the ML-IAP model file of a checkpoint (a pickle that LAMMPS loads)."""
    import torch
    from .model import NO_STRESS, fold_charge_spin, load, trained_dtype
    m, ck = load(path, "cpu")
    dtype = trained_dtype(ck) if dtype == "auto" else dtype
    m = fold_charge_spin(m, charge, spin)
    if not m.stress:
        warnings.warn(f"{path}: {NO_STRESS}; LAMMPS computes a pressure from the pair forces "
                      f"anyway, do not use it (no NPT, no box relaxation)", stacklevel=2)
    with torch.no_grad():
        model = ILMlipMLIAP(tables(m), dtype, kernels)
    with open(out, "wb") as fh:
        pickle.dump(model, fh)
    return model


if __name__ == "__main__":
    import argparse
    from . import runinfo
    from .lammps import export     # the pickle refers to ilmlip.lammps, which LAMMPS can import
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint")
    p.add_argument("out")
    p.add_argument("--dtype", choices=["auto", "tf32", "float32", "float64"], default="auto")
    p.add_argument("--kernels", choices=KERNELS, default="auto")
    p.add_argument("--charge", type=int, help="total charge, OMOL models only (default 0)")
    p.add_argument("--spin", type=int, help="multiplicity, OMOL models only (default 1)")
    p.add_argument("--elements", nargs="+", help="only these elements: LAMMPS's cutoff becomes theirs")
    a = p.parse_args()
    runinfo.show()
    m = export(a.checkpoint, a.out, a.dtype, a.charge, a.spin, a.kernels)
    if a.elements:
        vcut.lammps_elements(m, a.elements, a.out)
    print(f"{a.out}: elements {' '.join(m.element_types)}, cutoff {2 * m.rcutfac:g} A, {m.dtype}, "
          f"matmul precision {m.precision}, kernels {a.kernels}")
