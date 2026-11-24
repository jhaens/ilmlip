"""Many structures in one call of the JAX model: one graph of structures that do not see each
other, each with its own cell, Verlet list and strain.

    from ilmlip.batched import BatchedModel
    model = BatchedModel("model.pt")
    out = model.compute([atoms_1, atoms_2, ...])      # one dict per structure

Each result has energy (eV, the E0s added in float64), forces (eV/A), stress (ASE's sign and
Voigt order, eV/A^3) and virial (-dE/dstrain, 3x3, eV); stress and virial are zeros without a
periodic cell or with compute_stress=False (the default for models not trained on stresses).
A structure keeps its neighbour list in its slot (slots=: one key per structure, default 0, 1,
...) until an atom has moved more than skin/2. Sizes are padded, so batches of similar size share
one compiled step. Out of device memory, the structures go in calls of half as many from then on.

submit(structures, slots) dispatches the step and returns a handle whose result() gives what
compute() gives. While one group is in flight, the host can serve another with other slots; the
structures must not change between submit and result.

Batching pays for small structures; from a few thousand atoms on, one structure fills the GPU.
One BatchedModel serves one call at a time; calls from several threads wait for each other.
"""
import functools
import itertools
import threading
import warnings
from contextlib import suppress
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from . import vcut
from .jax_md import _XLA_OPTIONS, JaxCalculator, _build_dense, _build_pairs, _out_of_memory, _plan_key, load
from .model import refuse_charge_spin
from .verlet import check_energy, check_finite, padded_size

_STATIC = ("offs", "width", "n_struct", "want_stress")
_MAX_PACKED = 256            # tuples of slots kept: beads called one at a time, chunks


def _step(T, dt, energy):
    """step(buf, seg, cand_i, cand_j, cand_s, cand_ok, ref, mu, amask, ls, present) -> [most
    neighbours per structure, forces, site energies, dE/dstrain per structure, largest squared
    displacement from ref per structure], flat.  buf: positions, then the cells; seg: each atom's
    structure; the candidates and ref as _join makes them."""
    rcut, hi = T["rcut"], jax.lax.Precision.HIGHEST
    far = jnp.asarray([2.0 * rcut, 0.0, 0.0], dt)

    def step(buf, seg, cand_i, cand_j, cand_s, cand_ok, ref, mu, amask, ls, present, offs, width, n_struct,
             want_stress):
        n_atoms = offs[-1]
        positions = buf[:3 * n_atoms].reshape(n_atoms, 3)
        cells = buf[3 * n_atoms:].reshape(n_struct, 3, 3)
        h = cells[seg[cand_i]]
        cshift = cand_s[:, :1] * h[:, 0] + cand_s[:, 1:2] * h[:, 1] + cand_s[:, 2:] * h[:, 2]
        vec = positions[cand_j] - positions[cand_i] + cshift
        inside = cand_ok & (jnp.sum(vec * vec, axis=1) < rcut * rcut)
        inside = vcut.inside(T, inside, vec, mu[cand_i], mu[cand_j])
        ins = inside.astype(jnp.int32)
        cnt = jax.ops.segment_sum(ins, cand_i, num_segments=n_atoms, indices_are_sorted=True)
        slot = jnp.cumsum(ins) - 1 - (jnp.cumsum(cnt) - cnt)[cand_i]
        ok = inside & (slot < width)
        row, slot = jnp.where(ok, cand_i, n_atoms), jnp.where(ok, slot, 0)
        # empty slots: the atom itself at 2 rcut, zero in value and gradient
        nbr = jnp.broadcast_to(jnp.arange(n_atoms, dtype=jnp.int32)[:, None],
                               (n_atoms, width)).at[row, slot].set(cand_j, mode="drop")
        shift = jnp.broadcast_to(far, (n_atoms, width, 3)).at[row, slot].set(cshift, mode="drop")

        def total(pos, strain):
            sh = shift
            if strain is not None:
                sym = 0.5 * (strain + jnp.swapaxes(strain, 1, 2))[seg]
                pos = pos + jnp.einsum("ax,axy->ay", pos, sym, precision=hi)
                sh = sh + jnp.einsum("akx,axy->aky", sh, sym, precision=hi)
            return energy(pos, sh, nbr, mu, amask, ls, present)

        strain = jnp.zeros((n_struct, 3, 3), dt) if want_stress else None
        (_, e_atom), grads = jax.value_and_grad(total, argnums=(0, 1) if want_stress else 0,
                                                has_aux=True)(positions, strain)
        gpos, gstr = grads if want_stress else (grads, jnp.zeros((n_struct, 3, 3), dt))
        need = jax.ops.segment_max(cnt, seg, num_segments=n_struct).astype(dt)
        moved = jax.ops.segment_max(jnp.sum((positions - ref) ** 2, axis=1), seg, num_segments=n_struct)
        return jnp.concatenate([need, -gpos.reshape(-1), e_atom, gstr.reshape(-1), moved.astype(dt)])
    return step


@functools.partial(jax.jit, static_argnames=("offs", "total"))
def _join(cands, offs, total):
    """The candidates and reference positions of all structures as one set (structure k from atom
    offs[k] on), the candidates padded to total.  Built once per rebuild: a call with an array per
    structure costs dispatch time; a coarse total keeps the compiled step as the structures grow."""
    last, n = offs[-1] - 1, sum(len(c[0]) for c in cands)
    pad = total - n
    return (jnp.concatenate([c[0] + o for c, o in zip(cands, offs)] + [jnp.full(pad, last, jnp.int32)]),
            jnp.concatenate([c[1] + o for c, o in zip(cands, offs)] + [jnp.full(pad, last, jnp.int32)]),
            jnp.concatenate([c[2] for c in cands] + [jnp.zeros((pad, 3), cands[0][2].dtype)]),
            jnp.concatenate([c[3] for c in cands] + [jnp.zeros(pad, bool)]),
            jnp.concatenate([c[4] for c in cands]))


class _Handle:
    """What BatchedModel.submit returns: result() waits for the step and gives the results."""

    def __init__(self, model, finish):
        self._model, self._finish, self._done = model, finish, None

    def result(self):
        with self._model._lock:
            if self._done is None:
                self._done = self._finish()
        return self._done


class BatchedModel:
    """model: checkpoint, pretrained name or JaxModel; kernels, compute_stress as for JaxCalculator;
    dtype, charge, spin, device as for jax_md.load.
    skin (A): wider than JaxCalculator's, since one rebuilt list holds up the call of all structures.
    width_step: table widths are multiples of it (larger: fewer compiled shapes).
    neighbour_list: "host", "device" or "auto" (device on a GPU, where host lists compete with the
    calls for the processor); device lists fall back to host ones for good when they fail."""

    def __init__(self, model, kernels="auto", skin=2.0, compute_stress=None, dtype="tf32",
                 charge=None, spin=None, device=None, width_step=2, neighbour_list="auto"):
        if neighbour_list not in ("auto", "host", "device"):
            raise ValueError(f"neighbour_list must be auto|host|device, got {neighbour_list!r}")
        if isinstance(model, (str, Path)):
            from . import pretrained
            model = load(pretrained.path(model), dtype, charge, spin, device)
        else:
            refuse_charge_spin(model.charge_spin_state, charge, spin)
        if compute_stress is None:
            compute_stress = model.stress
        elif compute_stress and not model.stress:
            from .model import NO_STRESS
            raise ValueError(f"{NO_STRESS}: use compute_stress=False (zero stress and virial)")
        # only for its neighbour lists and species tables
        self._calc = JaxCalculator(model, compute_stress=compute_stress, skin=skin, kernels=kernels,
                                   background_compile=False, spatial_order=False)
        self.model, self.skin = model, float(skin)
        self.compute_stress, self.width_step = bool(compute_stress), int(width_step)
        self._jit(self._calc.kernels)
        on_gpu = model.device.platform == "gpu"
        self.neighbour_list = ("device" if on_gpu else "host") if neighbour_list == "auto" else neighbour_list
        self.n_device_builds, self._builders, self._ahead = 0, {}, set()
        self._compile_ahead = on_gpu     # next table width in a thread, as JaxCalculator's background_compile
        self._skin_max, self._calls = 2 * self.skin, 0
        self._slots = {}                 # per slot: its list, the candidates on the device
        self._next = {}                  # per slot: (thread, [list]) of a background rebuild
        self._tuples = {}                # per tuple of slots: the per-atom arrays on the device
        self.n_rebuilds = 0
        self._ids, self._lock = itertools.count(), threading.RLock()
        self.max_structures = None       # per call; set when the device ran out of memory

    def _jit(self, kernels):
        dt = jnp.dtype(self.model.dtype)
        energy = _build_dense(self.model.T, dt, kernels).energy
        self.kernels = kernels
        self._run = jax.jit(_step(self.model.T, dt, energy), static_argnames=_STATIC,
                            compiler_options=_XLA_OPTIONS)

    def _check_state(self, atoms):
        c = self._calc
        got = c._state_of(atoms)
        if got is not None and got != tuple(c.charge_spin_state):
            raise ValueError(f"atoms.info has charge {got[0]}, spin {got[1]}, but this model is for "
                             f"charge {c.charge_spin_state[0]}, spin {c.charge_spin_state[1]}")

    def _collect(self, key):
        """Install the slot's background list once it has finished."""
        th, out = self._next.get(key, (None, None))
        if th is not None and not th.is_alive():
            del self._next[key]
            if out:
                self._install(key, out[0], self._slots[key]["sid"])

    def _list(self, key, atoms):
        """The slot's list for atoms, checked on the host: rebuilt in the background from skin/4
        on, built at once beyond skin/2 (as VerletCalculator._moved)."""
        c, st = self._calc, self._slots.get(key)
        if st is None or not c._same_structure(st, atoms):
            self._next.pop(key, None)
            return self._install(key, self._prepare(atoms.positions.copy(), atoms, self.skin, st))
        self._collect(key)
        st = self._slots[key]
        d2 = c._moved2(st, atoms)
        if d2 > (0.5 * st["skin"]) ** 2:
            self._rebuild(key, atoms, d2)
        elif key not in self._next and d2 > (0.25 * st["skin"]) ** 2:
            self._start(key, atoms)
        return self._slots[key]

    def _rebuild(self, key, atoms, d2):
        """The slot's list is invalid beyond skin/2 (d2: squared displacement): wait for the
        background one if there is one, else build a new one."""
        st = self._slots[key]
        if self._calls - st["built"] < 20:          # the host cannot keep up: a wider skin
            self.skin = min(1.5 * self.skin, self._skin_max)
        th, out = self._next.pop(key, (None, None))
        if th is not None:
            th.join()
            if out:
                st = self._install(key, out[0], st["sid"])
                d2 = self._calc._moved2(st, atoms)
        if d2 > (0.5 * st["skin"]) ** 2:
            self._install(key, self._prepare(atoms.positions.copy(), atoms, self.skin, st), st["sid"])

    def _start(self, key, atoms):
        out = []
        th = threading.Thread(target=self._prepare_into, daemon=False,
                              args=(out, atoms.positions.copy(), atoms.copy(), self.skin, self._slots[key]))
        self._next[key] = th, out
        th.start()

    def _prepare_into(self, out, *args):
        with suppress(Exception):
            out.append(self._prepare(*args))

    def _prepare(self, pos, atoms, skin, old):
        """The host list and its candidates on the device (a thread may run this)."""
        c, dt = self._calc, self.model.dtype
        nl = c._host_list(pos, atoms, skin)
        n_c, n_pad = len(nl["i"]), nl["n_pad"]
        cap = c._cand_cap(nl, old)
        # spare candidates point at the last atom, which keeps cand_i sorted
        ci, cj = np.full(cap, n_pad - 1, np.int32), np.full(cap, n_pad - 1, np.int32)
        ci[:n_c], cj[:n_c] = nl["i"], nl["j"]
        cs = np.zeros((cap, 3), dt)
        cs[:n_c] = nl["S"]
        ref = np.zeros((n_pad, 3), dt)
        ref[:nl["n"]] = pos
        dev = tuple(jax.device_put(x, self.model.device) for x in (ci, cj, cs, np.arange(cap) < n_c, ref))
        plan = None
        if self.neighbour_list == "device":
            plan = _plan_key(pos, nl["cell"], nl["pbc"], nl["n"], n_pad, cap, nl["rcut"], skin, False, (0, 0))
        return dict(n=nl["n"], n_pad=n_pad, mu=nl["mu"], numbers=atoms.numbers.copy(), pbc=atoms.pbc.copy(),
                    plan=plan, cell=nl["cell"], h0=nl["cell"], ref=pos, skin=skin, rcut=nl["rcut"],
                    e0=c._e0[nl["mu"]], stress=nl["stress"], need=nl["need"], cand_cap=cap, dev=dev)

    def _install(self, key, st, sid=None):
        """sid: of the structure in the slot, kept while it is the same one; stamp: of this list."""
        st["stamp"] = next(self._ids)
        st["sid"], st["built"] = st["stamp"] if sid is None else sid, self._calls
        self._slots[key] = st
        self.n_rebuilds += 1
        return st

    def _width(self, need):
        w = self._calc._fit(int(need))
        return -(-w // self.width_step) * self.width_step

    def _layout(self, lists):
        """Per-atom arrays of a tuple of structures, on the device; each structure's atoms are
        padded to a multiple of 8."""
        dt, n_s = self.model.dtype, len(lists)
        n_pad = np.array([st["n_pad"] for st in lists])
        off = np.concatenate([[0], np.cumsum(n_pad)])
        real = np.concatenate([np.arange(o, o + st["n"]) for st, o in zip(lists, off)])
        mu, ls = np.zeros(off[-1], np.int32), np.zeros(off[-1], np.int32)
        mu[real] = mu_real = np.concatenate([st["mu"] for st in lists])
        ls_real, present = self._calc._state_extra(mu_real, len(mu_real))[0][:2]
        ls[real] = ls_real
        amask = np.zeros(off[-1], dt)
        amask[real] = 1
        e0 = np.zeros(off[-1])
        e0[real] = np.concatenate([st["e0"] for st in lists])
        has = np.array([st["stress"] for st in lists])
        put = lambda x: jax.device_put(x, self.model.device)
        return dict(sids=tuple(st["sid"] for st in lists), offs=tuple(int(o) for o in off),
                    n=np.array([st["n"] for st in lists]), B=1 << max(n_s - 1, 0).bit_length(),
                    stress=self.compute_stress and bool(has.any()), has=has, e0=e0, width=0, real=real,
                    numbers=np.concatenate([st["numbers"] for st in lists]),
                    pbc=np.array([st["pbc"] for st in lists]), stamps=None,
                    dev=[put(x) for x in (np.repeat(np.arange(n_s), n_pad), mu, amask, ls, present)])

    def _joined(self, key, lists):
        """The tuple's layout with the candidates and reference positions of the lists in it."""
        lay = self._tuples.get(key)
        if lay is None or lay["sids"] != tuple(st["sid"] for st in lists):
            if len(self._tuples) >= _MAX_PACKED:
                self._tuples.clear()
            lay = self._tuples[key] = self._layout(lists)
        stamps = tuple(st["stamp"] for st in lists)
        if lay["stamps"] != stamps:                 # MD: a list was rebuilt
            half2 = np.array([(0.5 * st["skin"]) ** 2 for st in lists])
            total = padded_size(sum(len(st["dev"][0]) for st in lists))
            lay.update(stamps=stamps, cands=_join(tuple(st["dev"] for st in lists), offs=lay["offs"], total=total),
                       h0=np.array([st["h0"] for st in lists]), half2=half2, quarter2=half2 / 4,
                       third2=half2 * 0.75 ** 2, planned=np.array([st["plan"] is not None for st in lists]))
            lay["width"] = max(lay["width"], self._width(max(st["need"] for st in lists)))
        return lay

    @staticmethod
    def _same(lay, lists, structures, pos):
        """Whether the structures are the ones in the layout (the slot's structure ids)."""
        return (lay is not None and lay["sids"] == tuple(st and st["sid"] for st in lists)
                and np.array_equal([len(p) for p in pos], lay["n"])
                and np.array_equal(np.concatenate([a.numbers for a in structures]), lay["numbers"])
                and np.array_equal([a.pbc for a in structures], lay["pbc"]))

    def compute(self, structures, slots=None):
        """One dict per structure (ase.Atoms), in their order."""
        structures, slots = self._normalise(structures, slots)
        with self._lock:
            try:
                return self.submit(structures, slots).result() if structures else []
            except Exception as exc:
                if not _out_of_memory(exc):
                    raise
            self._tuples.clear()
            self._slots.clear()
            self._next.clear()
            jax.clear_caches()           # every compiled shape holds device memory
            size = min(len(structures), self.max_structures or len(structures))
            if size == 1:
                return self.submit(structures, slots).result()
            self.max_structures = size // 2
            warnings.warn(f"out of device memory: at most {self.max_structures} structures per call",
                          RuntimeWarning)
            return self.compute(structures, slots)

    @staticmethod
    def _normalise(structures, slots):
        structures = list(structures)
        slots = list(range(len(structures)) if slots is None else slots)
        if len(set(slots)) != len(slots) or len(slots) != len(structures):
            raise ValueError("one slot per structure, each slot once")
        return structures, slots

    def submit(self, structures, slots=None):
        """Dispatch the step for structures; the handle's result() waits for it and gives what
        compute() gives.  Groups in flight at once must not share slots."""
        structures, slots = self._normalise(structures, slots)
        with self._lock:
            k = self.max_structures or len(structures)
            if len(structures) > k:         # the chunks after the first go one at a time
                first = self.submit(structures[:k], slots[:k])
                return _Handle(self, lambda: first.result() + [
                    r for c in range(k, len(structures), k)
                    for r in self.submit(structures[c:c + k], slots[c:c + k]).result()])
            if not structures:
                return _Handle(self, list)
            self._calls += 1
            p = self._dispatch(structures, slots)
            return _Handle(self, lambda: self._finish(p))

    def _dispatch(self, structures, slots):
        """Check the structures, bring their lists up to date and start the step."""
        key, n_s, dt = tuple(slots), len(structures), self.model.dtype
        pos = [a.positions for a in structures]
        allpos = np.concatenate(pos)
        if not np.isfinite(allpos).all():
            for q in pos:
                check_finite(q)
        for a in structures:
            self._check_state(a)
        for k in [k for k in self._next if k in key]:      # only this group's slots: others are in flight
            self._collect(k)
        lists, lay = [self._slots.get(k) for k in slots], self._tuples.get(key)
        if not self._same(lay, lists, structures, pos):
            lists = [self._list(k, a) for k, a in zip(slots, structures)]
        cells64 = np.array([a.cell.array for a in structures])
        lay = self._joined(key, lists)
        fixed = (cells64 == lay["h0"]).all(axis=(1, 2))
        if not fixed.all():                         # a changed cell: the host measures the displacement
            for i in np.flatnonzero(~fixed):
                lists[i] = self._list(slots[i], structures[i])
            lay = self._joined(key, lists)
            fixed = (cells64 == lay["h0"]).all(axis=(1, 2))
        offs, B = lay["offs"], lay["B"]
        buf = np.zeros(3 * offs[-1] + 9 * B, dt)       # positions, cells: one transfer
        buf[:3 * offs[-1]].reshape(-1, 3)[lay["real"]] = allpos
        cells = buf[3 * offs[-1]:].reshape(B, 3, 3)
        cells[:] = np.eye(3)
        cells[:n_s] = cells64
        p = SimpleNamespace(structures=structures, slots=slots, lay=lay, fixed=fixed, cells64=cells64,
                            allpos=allpos, offs=offs, B=B, stress=lay["stress"], width=lay["width"],
                            args=(jax.device_put(buf, self.model.device), lay["dev"][0], *lay["cands"],
                                  *lay["dev"][1:]))
        self._launch(p)
        return p

    def _launch(self, p):
        while True:
            try:
                with jax.default_matmul_precision(self.model.precision):
                    p.out = self._run(*p.args, offs=p.offs, width=p.width, n_struct=p.B, want_stress=p.stress)
                return
            except Exception as exc:
                if not self._to_xla(exc):
                    raise

    def _to_xla(self, exc):
        """Other kernels after a failing fused step, as in JaxCalculator (not against out of memory)."""
        if not self.kernels.startswith("fused") or _out_of_memory(exc):
            return False
        warnings.warn(f"fused kernels failed ({type(exc).__name__}: {str(exc)[:200]}); using XLA",
                      RuntimeWarning)
        self._jit("xla")
        return True

    def _finish(self, p):
        """Wait for the step; redo it wider or after a list expired, start the next lists, split
        the results."""
        lay, offs, B, n_s = p.lay, p.offs, p.B, len(p.structures)
        while True:
            try:
                out = np.asarray(p.out)
            except Exception as exc:
                if not self._to_xla(exc):
                    raise
                self._launch(p)
                continue
            need = out[:n_s].astype(int).max(initial=0)
            if need <= p.width:
                break
            p.width = self._width(need)         # the table dropped neighbours: redo it wider
            self._launch(p)
        lay["width"] = max(lay["width"], p.width)    # widths only grow until the structures change
        if self._compile_ahead and need >= p.width - 1:
            self._follow(p)
        d2 = out[B + 4 * offs[-1] + 9 * B:][:n_s]
        expired = np.flatnonzero(p.fixed & (d2 > lay["half2"]))
        if len(expired):
            for i in expired:
                self._rebuild(p.slots[i], p.structures[i], d2[i])
            return self._finish(self._dispatch(p.structures, p.slots))
        device = self.neighbour_list == "device"
        trigger = np.where(lay["planned"], lay["third2"], lay["quarter2"]) if device else lay["quarter2"]
        due = np.flatnonzero(p.fixed & (d2 > trigger))
        if device:
            self._device_rebuilds(p, [i for i in due if lay["planned"][i]])
        for i in due:
            if not (device and lay["planned"][i]) and p.slots[i] not in self._next:
                self._start(p.slots[i], p.structures[i])
        forces = out[B:B + 3 * offs[-1]].reshape(-1, 3).astype(float)
        e_atom = out[B + 3 * offs[-1]:B + 4 * offs[-1]].astype(float) + lay["e0"]
        energies = np.add.reduceat(e_atom, offs[:-1])  # padded atoms have no energy and no E0
        if not np.isfinite(energies).all():
            check_energy(energies[~np.isfinite(energies)][0])
        # no strain derivative without a periodic cell, whatever else is in the batch
        g = out[B + 4 * offs[-1]:][:9 * B].reshape(B, 3, 3)[:n_s].astype(float)
        g = 0.5 * (g + g.transpose(0, 2, 1)) * (p.stress & lay["has"])[:, None, None]
        vol = np.abs(np.linalg.det(p.cells64))
        sig = (g / np.where(vol > 0, vol, 1.0)[:, None, None])[:, [0, 1, 2, 1, 0, 0], [0, 1, 2, 2, 2, 1]]
        return [dict(energy=float(energies[k]), forces=forces[o:o + m], virial=-g[k], stress=sig[k])
                for k, (o, m) in enumerate(zip(offs, lay["n"]))]

    def _follow(self, p):
        """Compile the step for the next table width in a thread (it only grows during MD;
        a compile in the call stalls every structure)."""
        width = self._width(p.width + 1)
        key = (p.offs, width, p.B, p.stress, p.args[2].shape)
        if key in self._ahead:
            return
        self._ahead.add(key)

        def compile_step():
            with suppress(Exception), jax.default_matmul_precision(self.model.precision):
                self._run(*p.args, offs=p.offs, width=width, n_struct=p.B, want_stress=p.stress).block_until_ready()
        threading.Thread(target=compile_step, daemon=False).start()

    def _builder(self, plan):
        """_build_pairs for a plan, on structure k of the packed buffer (atom offset o)."""
        run = self._builders.get(plan)
        if run is None:
            pairs, n_pad = _build_pairs(self.model.T, jnp.dtype(self.model.dtype), *plan), plan[0]
            cls = np.zeros(len(self.model.species), np.int32)

            @jax.jit
            def run(buf, o, amask, mu, cell, rcut, skin):
                at = lambda x, w=1: jax.lax.dynamic_slice_in_dim(x, w * o, w * n_pad)
                m = at(mu)
                return pairs(at(buf, 3).reshape(n_pad, 3), cell, at(amask), m, m, cls, rcut, skin)
            self._builders[plan] = run
        return run

    def _device_rebuilds(self, p, idx):
        """New lists for structures idx from the call's positions, built on the device; the host
        builds a list where sizes are outgrown, and all of them once the device failed."""
        dt, lay, runs = self.model.dtype, p.lay, []
        for i in idx:
            st = self._slots[p.slots[i]]
            try:
                runs.append((i, st, *self._builder(st["plan"])(
                    p.args[0], np.int32(p.offs[i]), lay["dev"][2], lay["dev"][1], p.cells64[i].astype(dt),
                    np.asarray(st["rcut"], dt), np.asarray(self.skin, dt))))
            except Exception as exc:
                self._device_failed(exc)
                runs.append((i, st, None, None))
        first = np.concatenate([[0], np.cumsum(lay["n"])])
        for i, st, out, stats in runs:
            n_pairs = None
            if out is not None:
                try:
                    n_pairs, most, bins, images, need = (int(x) for x in np.asarray(stats))
                except Exception as exc:
                    self._device_failed(exc)
            key, plan = p.slots[i], st["plan"]
            if n_pairs is None or n_pairs > plan[1] or most > plan[5] or bins > plan[4] or not images:
                atoms = p.structures[i]
                self._install(key, self._prepare(atoms.positions.copy(), atoms, self.skin, st), st["sid"])
                continue
            self.n_device_builds += 1
            self._install(key, dict(st, dev=out[:5], need=need, skin=self.skin,
                                    ref=p.allpos[first[i]:first[i + 1]].copy()), st["sid"])

    def _device_failed(self, exc):
        warnings.warn(f"the device neighbour list failed ({type(exc).__name__}: {str(exc)[:200]}); "
                      f"host lists from now on", RuntimeWarning)
        self.neighbour_list = "host"
