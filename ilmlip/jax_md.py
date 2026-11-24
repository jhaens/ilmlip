"""Molecular dynamics with ilmlip models in JAX.

A PyTorch checkpoint is converted once into constant tables. Each MD step is one compiled call on
a dense (atoms, K) neighbour table, returning energy, forces and stress in one array.
The couplings run unrolled in XLA or as fused cuEquivariance kernels ("fused"; "fused-reference"
is the same polynomials in plain JAX, for tests). Very large systems run in chunks with a halo,
with the same result.
"""
import gc
import math
import os
import time
import warnings
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import torch
from ase.calculators.calculator import all_changes

from .model import (COULOMB, DTYPES, angular, fold_charge_spin, load as load_torch, pair_repulsion, radial_basis,
                    zbl_rmax)
from .verlet import VerletCalculator, _spread3, padded_size
from . import vcut

_MD_STATIC = ("n_atoms", "width", "want_stress")
_XLA_OPTIONS = {"xla_gpu_dot_merger_threshold_mb": 0}  # readout-gradient dots copied apart per layer
_PAIR_MATMUL = 16            # up to this many species pairs: one matrix product
_MAX_PAIR_TABLES = 256       # sets of species whose pair tables a model keeps on the device
_EXPAND_PATHS = 512          # c2 is written into a product stack of at most this many paths
_MERGE_CHANNELS = 4          # channel classes of layer 0 share one polynomial from this common divisor on
# "out": the packed result of a step goes to pinned host memory, "io": the positions are read from there
# too (the copies are part of the compiled program).  Empty: "out" on a GPU, off elsewhere; "off" or "0": off
_HOST_MEMORY = os.environ.get("ILMLIP_JAX_HOST_MEMORY", "")
if _HOST_MEMORY not in ("", "off", "0", "out", "io"):
    raise ValueError(f"ILMLIP_JAX_HOST_MEMORY must be empty, off, 0, out or io, not {_HOST_MEMORY!r}")
# the host thread polls for the end of a step instead of sleeping until the runtime wakes it
_SPIN = os.environ.get("ILMLIP_JAX_SPIN", "") == "1"
# bonds of a row per element of the fused bond sum of the message layers (1: one per element); table
# widths are multiples of 4.  Unset: 4 up to 16 message channels, 2 above
_EDGE_GROUP = os.environ.get("ILMLIP_JAX_EDGE_GROUP", "")
if _EDGE_GROUP and (not _EDGE_GROUP.isdigit() or int(_EDGE_GROUP) < 1):
    raise ValueError(f"ILMLIP_JAX_EDGE_GROUP must be a positive integer, not {_EDGE_GROUP!r}")
_EDGE_GROUP = int(_EDGE_GROUP) if _EDGE_GROUP else 0


def _message_group(u):
    """Bonds per element of the fused message sum for u message channels."""
    return _EDGE_GROUP or (4 if u <= 16 else 2)


def _exact(a, b):
    """a @ b in full precision: lattice translations and strains, whatever the network uses."""
    return jnp.matmul(a, b, precision=jax.lax.Precision.HIGHEST)


def enable_x64():
    """JAX float64; call before creating any array."""
    jax.config.update("jax_enable_x64", True)


def tables(m):
    """Everything the JAX step needs, as numpy, from an ILMlip module."""
    a = lambda t: np.asarray(t.detach().cpu().numpy())
    coupling = lambda c: dict(rows_a=a(c.rows_a).astype(np.int32), rows_b=a(c.rows_b).astype(np.int32),
                              factors=a(c.factors), dest=a(c.dest).astype(np.int32), n_out=int(c.n_out))
    products = lambda ps: {k: coupling(c) for k, c in ps.couplings.items()}

    def reduce(red):
        out = dict(n_out=red.n_out, n_lm=red.n_lm, orders=list(red.orders))
        for o in red.orders:
            rows, slot, w = red.per_function(o)
            out[f"rows{o}"], out[f"slot{o}"], out[f"w{o}"] = a(rows).astype(np.int32), a(slot).astype(np.int32), a(w)
        return out

    mods = list(m.readout)
    common = dict(
        lmax=m.lmax, n_lm=(m.lmax + 1) ** 2, lmax_feat=m.lmax_feat, n_lm_feat=(m.lmax_feat + 1) ** 2,
        n_layers=m.n_layers, rcut=float(m.gbasis.rcut), n_basis=m.gbasis.size, cheb_p=m.gbasis.p,
        species=list(m.species), energy_scale=float(a(m.energy_scale)), atomic_energies=a(m.atomic_energies),
        feat_mean=a(m.feat_mean), feat_std=a(m.feat_std), n_features=m.n_features,
        zbl=dict(pair=_zbl_pair_table(m), n=len(m.species)) if m.zbl else None,
        readout=[(a(l.weight), a(l.bias), k + 1 < len(mods) and isinstance(mods[k + 1], torch.nn.SiLU))
                 for k, l in enumerate(mods) if hasattr(l, "weight")],
        vcut=vcut.jax_tables(m),
        bound=m.bound, bond_scale=m.bond_scale, feat_bound=m.feat_bound, zbl_switch=m.zbl_switch,
        bond_rms=a(m.bond_rms) if m.bound else None,
        elem_bias=a(m.elem_bias), head=a(m.head), stress=m.stress)
    return dict(common, arch=_arch_tables(m, a, coupling, products, reduce))


def _backflow_paths(m, ycouple):
    """(degree l1, lm of Y, degree l2, output row, c) of the first message layer's back-flow: the
    coupling of R_l1 Y_l1 and (-1)^l2 Q_l2 Y_l2, both of one direction, is -c R_l1 Q_l2 Y_LM."""
    a = lambda t: t.detach().cpu().numpy()
    u = np.random.default_rng(0).normal(size=(32, 3))
    Y = angular(u / np.linalg.norm(u, axis=1, keepdims=True), m.lmax_feat, np)
    h, tol = np.zeros((len(u), ycouple.n_out)), 1e3 * np.finfo(a(ycouple.factors).dtype).eps
    np.add.at(h.T, a(ycouple.dest), (a(ycouple.factors) * Y[:, a(ycouple.rows_a)] * Y[:, a(ycouple.rows_b)]).T)
    paths = []
    for o, (L, M, (_, l1, _, l2)) in enumerate(ycouple.meta):
        y = Y[:, L * L + L + M]
        scale = h[:, o] @ y / (y @ y)
        # up to the rounding of the factors: a float32 model holds them in float32
        assert np.abs(h[:, o] - scale * y).max() < tol, "a coupling of one direction is not its harmonic"
        paths.append((int(l1), L * L + L + M, int(l2), o, -float(scale) * (-1.0) ** l2))
    return paths


def _arch_tables(m, a, coupling, products, reduce):
    """The model's own tables, for _energy_fns."""
    wide = lambda x: np.asarray(x, np.float64)
    tapered = m.taper is not None
    z = wide(a(m._species_z()))
    inv0 = [{} for _ in m.groups]
    if tapered:
        first = [(int(b), int(o)) for b, o in m.group_keys]
        for b, o in first:
            inv0[b][o] = a(getattr(m, f"inv0g{b}_{o}")).astype(np.int32)
    else:
        first = [(0, int(o)) for t, o in m.inv_keys if t == 0]
        for _, o in first:
            inv0[0][o] = a(getattr(m, f"inv0_{o}")).astype(np.int32)
    later = [(int(t), int(o)) for t, o in m.inv_keys if t > 0]
    layers = []
    for t in range(m.n_layers - 1):
        red = m.reduces[t]
        layers.append(dict(reduce=[reduce(r) for r in (red if isinstance(red, torch.nn.ModuleList) else [red])],
                           ycouple=coupling(m.ycouple[t]), prod=products(m.prods[t]) if t < len(m.prods) else None))
    degrees = []
    for l in range(m.lmax + 1):
        degrees.append(dict(w=a(m.edge_weights[l]), film=a(m.film_weights[l])))
        if l <= m.lmax_ind:
            degrees[-1]["order1"] = a(m._order1_mix(l))
    out = dict(
        n_species=m.n_species, chan=[int(c) for c in m.chan], groups=[tuple(int(x) for x in g) for g in m.groups],
        degrees=degrees, chem0=z @ wide(a(m.chem0.weight)).T, film=a(m._film_table()),
        prod0=[products(P) for P in (m.prod0 if tapered else [m.prod0])], inv0=inv0, features0=first,
        inv={(t, o): a(getattr(m, f"inv{t}_{o}")).astype(np.int32) for t, o in later}, features=later,
        layers=layers, n_message=m.n_message, lmax_ind=m.lmax_ind, backflow=_backflow_paths(m, m.ycouple[0]),
        gram=dict(rank=int(m.gram_proj[0].shape[1]), bispec_rank=int(m.bispec_proj[0].shape[1]),
                  products=products(m.bispec), n_bispec=len(m.bispec.meta[3]),
                  degrees=[dict(rows=a(getattr(m, f"gram_rows{l}")).astype(np.int32),
                                proj=a(m.gram_proj[l]), bispec=a(m.bispec_proj[l]))
                           for l in range(m.lmax_feat + 1)]))
    return out


def _cast(t, dt, device=None):
    """Float tables become device arrays of dtype dt, index tables stay numpy constants."""
    out = {}
    for k, v in t.items():
        if isinstance(v, np.ndarray):
            out[k] = jax.device_put(v.astype(dt), device) if v.dtype.kind == "f" else v
        elif isinstance(v, dict):
            out[k] = _cast(v, dt, device)
        elif isinstance(v, list) and v and isinstance(v[0], dict):
            out[k] = [_cast(x, dt, device) for x in v]
        else:
            out[k] = v
    return out


def jax_device(spec=None):
    """A JAX device: None (JAX's default), a jax.Device, "cpu", "gpu"/"cuda" or "cuda:1"."""
    if spec is None:
        return jax.devices()[0]
    if not isinstance(spec, str):
        return spec
    kind, _, index = spec.lower().partition(":")
    kind = {"cuda": "gpu"}.get(kind, kind)
    return jax.devices(kind)[int(index or 0)]


def _coupling_terms(c, outputs=None):
    """(left, right, cg, out) of a coupling, restricted to the output rows `outputs`
    (renumbered in that order; None keeps all)."""
    L, R = np.asarray(c["rows_a"]), np.asarray(c["rows_b"])
    CG, SEG = np.asarray(c["factors"], np.float64), np.asarray(c["dest"])
    outputs = list(range(int(c["n_out"]))) if outputs is None else [int(o) for o in outputs]
    pos = {o: k for k, o in enumerate(outputs)}
    keep = np.isin(SEG, outputs)
    terms = [(int(l), int(r), float(cg), pos[int(s)])
             for l, r, cg, s in zip(L[keep], R[keep], CG[keep], SEG[keep])]
    return terms, len(outputs)


def _row_sums(groups, x, y):
    out = []
    for terms in groups:
        acc = None
        for i, j, cg in terms:
            v = cg * (x[i] * y[j])
            acc = v if acc is None else acc + v
        out.append(jnp.zeros_like(x[0]) if acc is None else acc)
    return jnp.stack(out)


def _unrolled_coupling(c, n_left, n_right, outputs=None):
    """Coupling as explicit sums of row products, with a hand-written derivative: no
    (terms, atoms, channels) intermediates and no scatters in either direction."""
    terms, n_out = _coupling_terms(c, outputs)
    by_out = [[(l, r, cg) for l, r, cg, o in terms if o == k] for k in range(n_out)]
    by_left = [[(o, r, cg) for l, r, cg, o in terms if l == k] for k in range(n_left)]
    by_right = [[(o, l, cg) for l, r, cg, o in terms if r == k] for k in range(n_right)]

    @jax.custom_vjp
    def couple(left, right):
        return _row_sums(by_out, left, right)

    def fwd(left, right):
        return couple(left, right), (left, right)

    def bwd(res, g):
        left, right = res
        return _row_sums(by_left, g, right), _row_sums(by_right, g, left)

    couple.defvjp(fwd, bwd)
    return couple


def _row_select(rows, n_rows):
    """x[rows] along axis 0, with a stack rather than a scatter as derivative."""
    rows = [int(r) for r in rows]
    pos = {r: k for k, r in enumerate(rows)}

    @jax.custom_vjp
    def select(x):
        return jnp.stack([x[r] for r in rows])

    def fwd(x):
        return select(x), jnp.zeros_like(x[0])

    def bwd(zero, g):
        return (jnp.stack([g[pos[r]] if r in pos else zero for r in range(n_rows)]),)

    select.defvjp(fwd, bwd)
    return select


def _col_select(cols, width):
    """x[:, cols] with a gather as derivative: the cotangent is not scattered into zeros."""
    pos = np.zeros(width, np.int32)
    pos[cols] = np.arange(len(cols))
    keep = np.isin(np.arange(width), cols)

    @jax.custom_vjp
    def select(x):
        return x[:, cols]

    def fwd(x):
        return select(x), None

    def bwd(_, g):
        return (jnp.where(keep, g[:, pos], 0),)

    select.defvjp(fwd, bwd)
    return select


def _reduce_weights(red, rowmap):
    """The message reduction as one weight per order, W[pos, n, lm_out, k]; rowmap[o]
    is the original row at each position of a (pruned) product block."""
    W = {}
    for o in red["orders"]:
        rows, slot = np.asarray(red[f"rows{o}"]), np.asarray(red[f"slot{o}"])
        w = np.asarray(red[f"w{o}"])                       # (row, n, k)
        at = {r: q for q, r in enumerate(rowmap[o])}
        Wo = np.zeros((len(rowmap[o]), w.shape[1], int(red["n_lm"]), w.shape[2]), w.dtype)
        for f, (row, s) in enumerate(zip(rows, slot)):
            Wo[at[int(row)], :, int(s), :] += w[f]
        W[o] = Wo
    return W


def _first_layer(T):
    """The first readout layer as float64 numpy, with the whitening folded in unless the features are
    squashed (feat_bound): W (hidden, features) acting on (f - mean) / std, and its bias."""
    w, b, act = T["readout"][0]
    if T.get("feat_bound"):
        return np.asarray(w, np.float64), np.asarray(b, np.float64), act
    w = np.asarray(w, np.float64) / np.asarray(T["feat_std"], np.float64)[None, :]
    b = np.asarray(b, np.float64) - w @ np.asarray(T["feat_mean"], np.float64)
    return w, b, act


def _zbl_pair_table(m):
    """The constants of the ZBL repulsion of every pair of species, row i * n_species + j, as float64:
    e^2 z_i z_j, z_i^0.23 + z_j^0.23 and the cutoff rmax."""
    z, rcov = (np.asarray(t.detach().cpu().numpy(), np.float64) for t in (m.zbl_z, m.zbl_rcov))
    rmax = zbl_rmax(rcov[:, None], rcov[None, :], m.zbl_factor, float(m.gbasis.rcut), xp=np)
    return np.stack([COULOMB * z[:, None] * z[None, :], z[:, None] ** 0.23 + z[None, :] ** 0.23, rmax],
                    axis=-1).reshape(-1, 3)


def _zbl_pairs(Z, r, mi, mj):
    """Half the ZBL energy of each listed pair (every pair is listed twice), from the constants of
    the pair of species (mi, mj); scalars with one species present."""
    coulomb, powers, rmax = jnp.moveaxis(Z["pair"][mi * Z["n"] + mj], -1, 0)
    return 0.5 * pair_repulsion(r, coulomb, powers, rmax, xp=jnp)


def _species_of(mu, present, i):
    """mu[i]; the one species present: a scalar, which turns the gathers per bond into constants."""
    return present["present"][0] if present["present"].shape[0] == 1 else mu[i]


def _degrees(lmax):
    """The degree l of every component lm up to lmax."""
    return np.concatenate([np.full(2 * l + 1, l) for l in range(lmax + 1)])


def _xla_stack(ps, n1, want):
    """Product stack computing only rows someone reads; rowmap[o] = original rows."""
    ops, rowmap = {}, {1: list(range(n1))}
    higher = [o for o, key in ((3, "c3"), (4, "c4")) if key in ps and want.get(o)]
    if "c2" in ps and (higher or want.get(2)):
        sel = None if higher else want[2]
        ops[2] = _unrolled_coupling(ps["c2"], n1, n1, sel)
        rowmap[2] = list(range(int(ps["c2"]["n_out"]))) if sel is None else list(sel)
    for o in higher:
        key, a, b = {3: ("c3", 2, 1), 4: ("c4", 2, 2)}[o]
        ops[o] = _unrolled_coupling(ps[key], len(rowmap[a]), len(rowmap[b]), want[o])
        rowmap[o] = list(want[o])

    def apply(A):
        out = {1: A}
        if 2 in ops:
            out[2] = ops[2](A, A)
        if 3 in ops:
            out[3] = ops[3](out[2], A)
        if 4 in ops:
            out[4] = ops[4](out[2], out[2])
        return out
    return apply, rowmap


def _mlp_rest(T, x, act1):
    """The readout after its first linear layer, up to the head."""
    if act1:
        x = x * jax.nn.sigmoid(x)
    for w, b, act in T["readout"][1:]:
        x = x @ w.T + b
        if act:
            x = x * jax.nn.sigmoid(x)
    return x


def _build_dense(T, dt, kernels="xla"):
    """The step on a dense neighbour table: md(positions, cell, cand_i, cand_j, cand_s, row_start,
    mu, amask, volume, ref, ls, present, n_atoms, width, want_stress) -> one packed array (_pack).
    row_start (n_atoms + 1): each atom's first candidate (cand_i is sorted), the last the count;
    ref: the positions the candidates were found for.  Counts (most neighbours, edges): most > width
    means the table overflowed and the step must be redone.  ls, present: each atom's species index
    and the species present (_pair_tables).  Energies exclude the E0s, which the host adds in float64
    (VerletCalculator)."""
    dense, _, blocks = _energies(T, dt, kernels)
    return _dense_step(T, dt, dense, blocks)


def _dense_step(T, dt, energy, energy_blocks):
    """md of _build_dense on these energies; energy_blocks: on table rows of several widths."""
    rcut = T["rcut"]
    far = jnp.asarray([2.0 * rcut, 0.0, 0.0], dt)

    def md(positions, cell, cand_i, cand_j, cand_s, row_start, mu, amask, volume, ref, ls, present, n_atoms,
           width, want_stress, wide=(0, 0), rank=None):
        # wide = (n, extra): the first n rows are wider by extra (JaxCalculator._row_classes);
        # elementwise, in full precision: fuses into the bond vectors
        n_cand = cand_i.shape[0]
        iota = jnp.arange(n_cand, dtype=jnp.int32)
        cshift = cand_s[:, :1] * cell[0] + cand_s[:, 1:2] * cell[1] + cand_s[:, 2:] * cell[2]
        vec = positions[cand_j] - positions[cand_i] + cshift
        inside = (iota < row_start[-1]) & (jnp.sum(vec * vec, axis=1) < rcut * rcut)
        inside = vcut.inside(T, inside, vec, _species_of(mu, present, cand_i), _species_of(mu, present, cand_j))
        ins = inside.astype(jnp.int32)
        # one prefix sum gives the counts and the slots
        csum = jnp.cumsum(ins)
        before = jnp.where(row_start > 0, csum[jnp.maximum(row_start - 1, 0)], 0)
        cnt = before[1:] - before[:-1]
        slot = csum - 1 - before[cand_i]
        n_w, extra = wide
        need = jnp.maximum(jnp.max(cnt[:n_w]) - extra, jnp.max(cnt[n_w:])) if n_w else jnp.max(cnt)

        def table(at, fill_nbr, fill_shift):
            """The neighbour and shift tables with candidate k at flat slot at[k] (out of range:
            left out): one scatter of k, then gathers."""
            n_slots = fill_nbr.shape[0]
            k = jnp.full(n_slots, n_cand, jnp.int32).at[at].set(iota, mode="drop")
            has, k = k < n_cand, jnp.minimum(k, n_cand - 1)
            return jnp.where(has, cand_j[k], fill_nbr), jnp.where(has[:, None], cshift[k], fill_shift)

        if n_w:
            # one bond list sorted by centre: n_w rows of width + extra, then rows of width;
            # empty slots: the atom itself at 2 rcut, zero in value and gradient
            wider, n_e = cand_i < n_w, n_w * (width + extra)
            ok = inside & (slot < width + jnp.where(wider, extra, 0))
            ind_i = jnp.concatenate([jnp.repeat(jnp.arange(n_w, dtype=jnp.int32), width + extra),
                                     jnp.repeat(jnp.arange(n_w, n_atoms, dtype=jnp.int32), width)])
            e = jnp.where(wider, cand_i * (width + extra), n_e + (cand_i - n_w) * width) + slot
            e = jnp.where(ok, e, ind_i.shape[0])
            nbr, shift = table(e, ind_i, jnp.broadcast_to(far, (ind_i.shape[0], 3)))
            blocks = ((n_w, width + extra), (n_atoms - n_w, width))
            energy_of = lambda pos, sh: energy_blocks(pos, sh, ind_i, nbr, mu, amask, ls, present, blocks)
        else:
            K = width + extra
            ok = inside & (slot < K)
            # empty slots: the atom itself at 2 rcut, zero in value and gradient
            self_ = jnp.broadcast_to(jnp.arange(n_atoms, dtype=jnp.int32)[:, None], (n_atoms, K)).reshape(-1)
            nbr, shift = table(jnp.where(ok, cand_i * K + slot, n_atoms * K), self_,
                               jnp.broadcast_to(far, (n_atoms * K, 3)))
            nbr, shift = nbr.reshape(n_atoms, K), shift.reshape(n_atoms, K, 3)
            energy_of = lambda pos, sh: energy(pos, sh, nbr, mu, amask, ls, present)

        def total(pos, sh, strain):
            if strain is not None:
                sym = 0.5 * (strain + strain.T)
                pos = pos + _exact(pos, sym)
                sh = sh + _exact(sh, sym)
            return energy_of(pos, sh)
        (E, e_atom), grads = jax.value_and_grad(total, argnums=(0, 2) if want_stress else 0,
                                                has_aux=True)(
            positions, shift, jnp.zeros((3, 3), dt) if want_stress else None)
        gpos, stress = (grads[0], _voigt(grads[1], volume)) if want_stress else (grads, jnp.zeros(6, dt))
        return _pack(E, stress, positions, ref, jnp.stack([need, jnp.sum(cnt)]), -gpos, e_atom, rank)

    md.energy = energy               # (E, site energies), used by batched.py
    return md


def _voigt(gstr, volume):
    sig = gstr / jnp.maximum(volume[0], 1e-12)
    sig = 0.5 * (sig + sig.T)
    return jnp.stack([sig[0, 0], sig[1, 1], sig[2, 2], sig[1, 2], sig[0, 2], sig[0, 1]])


def _pack(E, stress, positions, ref, counts, forces, energies, rank=None):
    """Everything the host reads, as one array (one transfer): energy, stress, the largest
    squared displacement since the candidates were found, two counts as base-4096 digits
    (exact in float32), forces, site energies; the energies leave out the E0s.  rank: the
    forces and energies leave in this order (the inverse of the one the atoms were put in)."""
    if rank is not None:
        forces, energies = forces[rank], energies[rank]
    d = positions - ref
    c = counts.astype(jnp.int32)
    digits = jnp.stack([c // 4096, c % 4096], axis=1).reshape(-1)
    head = jnp.concatenate([E[None], stress, jnp.max(jnp.sum(d * d, axis=1))[None]])
    return jnp.concatenate([head, digits.astype(forces.dtype), forces.reshape(-1), energies])


def _unpack(out, n_pad):
    """(head = [E, stress], (counts), forces, energies, displacement^2) of a packed step."""
    a, b, c, d = out[8:12].tolist()
    return (out[:7], (int(a * 4096 + b), int(c * 4096 + d)),
            out[12:12 + 3 * n_pad].reshape(n_pad, 3), out[12 + 3 * n_pad:], float(out[7]))


def _in_order(md):
    """md on the atoms rearranged by its last two arguments (order: the caller's atom at
    each place, rank: its inverse); positions in, forces and site energies out in the
    caller's order."""
    def run(positions, *args, **static):
        *args, order, rank = args
        return md(positions[order], *args, rank=rank, **static)
    return run


_HOST_DEFAULT_FAILED = set()      # devices on which the default host-memory step did not compile or run


def _host_mode(device):
    """The host-memory mode of a step on this device, "out", "io" or "" (off); see _HOST_MEMORY."""
    if _HOST_MEMORY:
        return "" if _HOST_MEMORY in ("off", "0") else _HOST_MEMORY
    return "out" if device.platform == "gpu" and device not in _HOST_DEFAULT_FAILED else ""


def _host_memory(md, inputs):
    """md with its result in host memory, and with the positions in host memory when inputs."""
    def run(positions, *args, **static):
        if inputs:
            positions = jax.device_put(positions, jax.memory.Space.Device)
        return jax.device_put(md(positions, *args, **static), jax.memory.Space.Host)
    return run


def _out_of_memory(exc):
    text = f"{type(exc).__name__}: {exc}"
    return "RESOURCE_EXHAUSTED" in text or "out of memory" in text.lower()


def _order_args(order, n_pad):
    """(order, rank) of the padded atoms for _in_order; () without an order."""
    if order is None:
        return ()
    o = np.arange(n_pad, dtype=np.int32)
    o[:len(order)] = order
    r = np.empty_like(o)
    r[o] = np.arange(n_pad, dtype=np.int32)
    return o, r


def _method(kernels):
    return "naive" if kernels == "fused-reference" else "uniform_1d"


def _start_backend():
    # cuEquivariance's library links against the CUDA libraries JAX loads when its
    # backend starts: import it after this
    return jax.default_backend()


def fused_available():
    """True when the fused CUDA kernels can run here."""
    try:
        if _start_backend() != "gpu":
            return False
        import cuequivariance_jax  # noqa: F401
        import cuequivariance_ops_jax  # noqa: F401
    except Exception:
        return False
    return True


def _cuex_call(dt, method):
    """call(polynomial, inputs, output shape, indices, name) of cuEquivariance's kernels."""
    _start_backend()
    import cuequivariance_jax as cuex

    def call(poly, inputs, shape, indices, name):
        return cuex.segmented_polynomial(poly, inputs, [jax.ShapeDtypeStruct(shape, dt)],
                                         indices=indices, method=method, name=name)[0]
    return call


def _cuex_stack(call, groups, u, limit=_EXPAND_PATHS, order=None):
    """apply(A, a) -> F of the products of channel classes laid out one after the other
    (fused.stack_plan), and each class's F rows per order.  groups: (couplings, rows of A, rows
    wanted per order, channels) of each class; u: a divisor of all the channels; c2 is part of the
    stack when that has at most `limit` paths.  order: per class a key {(order, row): key} of its rows, by
    which the rows of F follow, which puts F's columns in another order: apply.perm, None if that is
    the same (W[perm] for a matrix W on the columns of F in the order of the classes)."""
    from .fused import stack_plan
    classes = [(c // u, [k for k in ("c2", "c3", "c4") if k in ps], n1, want,
                lambda k, sel, ps=ps: _coupling_terms(ps[k], sel), *([order[i]] if order else []))
               for i, (ps, n1, want, c) in enumerate(groups)]
    c2_poly, f_poly, names, rows, apply_perm = stack_plan(classes, u, limit)

    def apply(A, a):
        blocks = {"A": A}
        if c2_poly is not None:
            blocks["c2"] = call(c2_poly, [A, A], (a, c2_poly.outputs[0].size), None, "vv_c2")
        return call(f_poly, [blocks[x] for x in names], (a, f_poly.outputs[0].size), None, "vv_stack")
    apply.perm = apply_perm
    return apply, rows


def _xla_edge(sources, n_rad, n_out, u):
    """A bond sum in plain XLA: op(radial, Y, nodes, ind_i, blocks, a) -> (a, n_out * u), with
    out[i, o] = sum over the bonds of row i and the paths (s, lm, [q,] o, c) of c radial[s] Y[lm]
    [node[q]].  sources: (paths, segments of the node) each; nodes: (values, row per bond or None);
    sources beyond the nodes take the radial operand as their node (message_poly).  Operands are
    (bonds or rows, segments * u)."""
    by_out = [[(k, p) for k, (paths, _) in enumerate(sources) for p in paths if p[-2] == o] for o in range(n_out)]

    def op(rad, Y, nodes, ind_i, blocks, a):
        E = rad.shape[0]
        R = rad.reshape(E, n_rad, u)
        N = [(x if i is None else x[i]).reshape(E, n, u) if n else None for (x, i), (_, n) in zip(nodes, sources)]
        N += [R] * (len(sources) - len(nodes))
        out = []
        for terms in by_out:
            acc = jnp.zeros((E, u), rad.dtype)
            for k, p in terms:
                v = (p[-1] * Y[:, p[1], None]) * R[:, p[0]]
                acc = acc + (v if N[k] is None else v * N[k][:, p[2]])
            out.append(acc)
        x = jnp.stack(out, 1)
        ends = np.cumsum([0] + [n * K for n, K in blocks])
        return jnp.concatenate([x[e0:e1].reshape(n, K, n_out, u).sum(1)
                                for e0, e1, (n, K) in zip(ends[:-1], ends[1:], blocks)]).reshape(a, n_out * u)
    return op


def _edge_group(K, limit):
    """Bonds of a row per kernel element for rows of K slots: the largest divisor of K up to limit."""
    return max(g for g in range(1, min(K, limit) + 1) if K % g == 0)


def _cuex_edge(call, name, n_y, sources, n_rad, n_out, u, group=1, n_flow=0):
    """_xla_edge as one fused kernel.  group: most bonds of a row per kernel element, for sums with a node
    (message layers); every block of rows takes the largest divisor of its width up to it.  n_flow: the
    back-flow's Y factors are not columns of Y but the operand `flow` (bonds, n_flow) of op."""
    from .fused import edge_poly, message_poly, radial_poly
    polys = {}

    def poly(G):
        if G not in polys:
            if len(sources) == 2:
                polys[G] = message_poly(sources, n_rad, n_y, n_out, u, G, n_flow)
            elif sources[0][1]:
                polys[G] = edge_poly(sources[0][0], n_rad, n_y, sources[0][1], n_out, u, G)
            else:
                polys[G] = radial_poly(sources[0][0], n_rad, n_y, n_out, u)
        return polys[G]

    def op(rad, Y, nodes, ind_i, blocks, a, flow=None):
        used = [(x, i) for x, i in nodes if x is not None]
        groups = [_edge_group(K, group) if used else 1 for _, K in blocks]
        if max(groups) == 1:
            return call(poly(1), [rad, Y] + [x for x, _ in used] if flow is None else [rad, Y, flow] + [x for x, _ in used],
                        (a, n_out * u), [None] * (2 + (flow is not None)) + [i for _, i in used] + [ind_i], name)
        (x, nbr), = used
        ends, out = np.cumsum([0] + [n * K for n, K in blocks]), []
        for e0, e1, (n, K), G in zip(ends[:-1], ends[1:], blocks, groups):
            if not n:
                continue
            q = n * K // G
            nb = nbr[e0:e1].reshape(q, G)
            rows = None if G == K else jnp.repeat(jnp.arange(n, dtype=nbr.dtype), K // G)
            each = [rad, Y] + ([] if flow is None else [flow])
            out.append(call(poly(G), [v[e0:e1].reshape(q, -1) for v in each] + [x] * G, (n, n_out * u),
                            [None] * len(each) + [nb[:, g] for g in range(G)] + [rows], name if G == 1 else f"{name}_g{G}"))
        return out[0] if len(out) == 1 else jnp.concatenate(out)
    return op


def _xla_class(ps, n1, want, u):
    """One class of _cuex_stack in plain XLA."""
    products, rowmap = _xla_stack(ps, n1, want)
    orders = [o for o in (1, 2, 3, 4) if want.get(o)]
    rows = {o: [int(r) for r in want[o]] for o in orders}
    pick = {o: None if rowmap[o] == rows[o] else _row_select([rowmap[o].index(r) for r in rows[o]], len(rowmap[o]))
            for o in orders}

    def apply(A, a):
        blocks = products(jnp.transpose(A.reshape(a, n1, u), (1, 0, 2)))
        F = jnp.concatenate([blocks[o] if pick[o] is None else pick[o](blocks[o]) for o in orders])
        return jnp.transpose(F, (1, 0, 2)).reshape(a, -1)
    return apply, rows


def _xla_rows(groups, u, order=None):
    """_cuex_stack in plain XLA: each class on its own, F in the order of the classes."""
    parts = [_xla_class(*g) for g in groups]
    ends = np.cumsum([0] + [n1 * c for _, n1, _, c in groups])

    def apply(A, a):
        return jnp.concatenate([p(A[:, e0:e1], a) for (p, _), e0, e1 in zip(parts, ends[:-1], ends[1:])], axis=1)
    apply.perm = None
    return apply, [r for _, r in parts]


def _wanted(red, inv):
    """Rows of each order of a layer's products read by a reduction (or None) and the readout."""
    want = {}
    if red is not None:
        for o in red["orders"]:
            want.setdefault(o, set()).update(int(x) for x in np.asarray(red[f"rows{o}"]))
    for o, rows in inv.items():
        want.setdefault(o, set()).update(int(x) for x in rows)
    return {o: sorted(r) for o, r in want.items()}


def _reduction(reds, rows, widths):
    """The message reductions of a layer's product blocks as one matrix on their concatenation."""
    out = []
    for red, rws, u in zip(reds, rows, widths):
        W = _reduce_weights(red, rws)
        for o in sorted(rws):
            shape = (len(rws[o]) * u, int(red["n_lm"]) * int(red["n_out"]))
            out.append(W[o].reshape(shape) if o in W else np.zeros(shape))
    return np.concatenate(out)


def _readout_rows(w1, rows, u, at):
    """The first readout layer (w1: hidden, features) on an F of `rows` with u channels; at: order ->
    (its first feature, the rows read), a feature per channel and read row, channels outermost."""
    start, s = {}, 0
    for o in sorted(rows):
        start[o], s = s, s + len(rows[o])
    W = np.zeros((s * u, w1.shape[0]))
    for o, (col, inv) in at.items():
        for k, r in enumerate(int(x) for x in inv):
            q = start[o] + rows[o].index(r)
            W[q * u:(q + 1) * u] = w1[:, col + np.arange(u) * len(inv) + k].T
    return W


def _gram_tables(V, n_out, w1, col, widths=None):
    """The Gram readout on the last layer's A (a, n_out * k) from feature `col` on: W maps the
    scalars to the first readout layer's hidden columns, the projections P[l, m, rank] and the
    bispectrum's base [lm, bispec rank]; the other features start at the returned column.
    widths: the columns of each (l, m) of P and of the base, at least their ranks (zeros after)."""
    G, k, h = V["gram"], V["n_message"], w1.shape[0]
    r, rb = G["rank"], G["bispec_rank"]
    pw, qw = widths or (r, rb)
    n_m = sum(2 * l + 1 for l in range(len(G["degrees"])))
    W = np.zeros((n_out * k, h + n_m * (pw + qw)))
    rows0 = np.asarray(G["degrees"][0]["rows"])
    for f, o in enumerate(int(x) for x in rows0):
        W[o * k:(o + 1) * k, :h] = w1[:, col + np.arange(k) * len(rows0) + f].T
    p, b = h, h + n_m * pw
    for l, d in enumerate(G["degrees"]):
        proj, bispec = (np.asarray(d[x], np.float64).reshape(-1, k, d[x].shape[1]) for x in ("proj", "bispec"))
        for f, rows in enumerate(np.asarray(d["rows"]).reshape(-1, 2 * l + 1)):
            for m, o in enumerate(int(x) for x in rows):
                W[o * k:(o + 1) * k, p + m * pw:p + m * pw + r] = proj[f]
                W[o * k:(o + 1) * k, b + m * qw:b + m * qw + rb] = bispec[f]
        p, b = p + (2 * l + 1) * pw, b + (2 * l + 1) * qw
    return W, col + k * len(rows0)


def _pair_radial(g, pair, H, n_pairs):
    """sum_k g[e, k] H[pair[e], k, n] of a table H (pairs * basis, n), in full precision like the sum of
    the compiled torch model.  A few pairs in one matrix product, in which the other pairs' zeros are
    exact; many in one fused sum."""
    hi = jax.lax.Precision.HIGHEST
    if n_pairs == 1:
        return jnp.matmul(g, H, precision=hi)
    if n_pairs <= _PAIR_MATMUL:
        hot = (pair[:, None] == jnp.arange(n_pairs, dtype=pair.dtype)).astype(g.dtype)
        return jnp.matmul((hot[:, :, None] * g[:, None, :]).reshape(g.shape[0], -1), H, precision=hi)
    return jnp.sum(g[:, :, None] * H.reshape(n_pairs, g.shape[1], -1)[pair], axis=1)


def _triangle(r):
    """f(P) = the upper triangle, row by row, of sum_m P[a, m, i] P[a, m, j], rank r: elementwise in full
    precision, as the model's sum, with a gather and not a scatter in the derivative."""
    i, j = np.triu_indices(r)
    flat = i * r + j
    place = np.zeros((r, r), np.int32)
    place[i, j] = place[j, i] = np.arange(len(i))
    twice = 1.0 + np.eye(r)

    @jax.custom_vjp
    def f(P):
        return jnp.sum(P[:, :, :, None] * P[:, :, None, :], axis=1).reshape(P.shape[0], -1)[:, flat]

    def fwd(P):
        return f(P), P

    def bwd(P, g):
        S = g[:, place] * twice.astype(g.dtype)
        return (jnp.sum(S[:, None, :, :] * P[:, :, None, :], axis=3),)

    f.defvjp(fwd, bwd)
    return f


def _gram2(G, lmax_feat, lead=0):
    """The Gram features as the paths of fused.gram2_poly.  x holds P[l, m, i] and the bispectrum's base
    Q[lm, k] as segments of u = max(rank, bispec rank) channels (zeros after the ranks), P first.
    out[l * rank + i, j] = sum_m P[l, m, i] P[l, m, j] (the square, whose lower triangle the readout
    ignores), out[n_sq + q, k] the bispectrum (c3 of c2 expanded); segments after those are filler (a
    segment must be written, the readout gives them no weight), as many as make `lead` + the late
    features a multiple of 4 (aligned GEMM) where that is possible.
    Returns (u, n_out, quad, cubic, rows), rows: the feature, segment * u + channel, of each of
    the readout's late features (the triangle by degree, row by row; the bispectrum by output, k)."""
    r, rb, n_b = G["rank"], G["bispec_rank"], G["n_bispec"]
    n_m, u = (lmax_feat + 1) ** 2, max(r, rb)
    n_sq = (lmax_feat + 1) * r
    n_out = n_sq + n_b
    n_out += next((f for f in range(4) if (lead + (n_out + f) * u) % 4 == 0), 0)
    quad = [(l * l * u + m * u + i, l * l + m, l * r + i, 1.0)
            for l in range(lmax_feat + 1) for m in range(2 * l + 1) for i in range(r)]
    quad += [(0, 0, o, 1.0) for o in range(n_sq + n_b, n_out)]
    t2, t3 = (_coupling_terms(G["products"]["c2"])[0],
              _coupling_terms(G["products"]["c3"], range(n_b))[0])
    by_out = {}
    for l, r2, c, o in t2:
        by_out.setdefault(o, []).append((l, r2, c))
    cubic = [(n_m + l, n_m + r2, n_m + rr, n_sq + q, c3 * c2)
             for l2, rr, c3, q in t3 for l, r2, c2 in by_out.get(l2, ())]
    i, j = np.triu_indices(r)
    rows = np.concatenate([(l * r + i) * u + j for l in range(lmax_feat + 1)]
                          + [((n_sq + np.arange(n_b)[:, None]) * u + np.arange(rb)).reshape(-1)])
    return u, n_out, quad, cubic, rows


def _head_first(want, head):
    """The order of _cuex_stack that puts the rows head ({order: rows}) before the others of want."""
    return {(o, x): int(x not in head.get(o, ())) for o, rows in want.items() for x in rows}


def _row_sum(x, blocks):
    """The sum over each row's K bonds of x, for the blocks (rows, K) of a table, bonds row by row."""
    ends = np.cumsum([0] + [n * K for n, K in blocks])
    return jnp.concatenate([x[e0:e1].reshape(n, K).sum(1) for e0, e1, (n, K) in zip(ends[:-1], ends[1:], blocks)])


def _core_switch(z, zc):
    """model.core_switch."""
    u = jnp.minimum(jnp.maximum(z / zc, 0.0), 1.0)       # not jnp.clip: its float64 bounds would enter float32
    return 1.0 - u ** 3 * (10.0 - 15.0 * u + 6.0 * u * u)


def _read_columns(W):
    """(x -> the columns of x that W's non-zero rows read, those rows): up to a multiple of 4 columns
    (16-byte aligned GEMM rows); a slice when they are the first ones."""
    u = np.flatnonzero(np.abs(W).sum(1))
    u = np.union1d(u, np.setdiff1d(np.arange(W.shape[0]), u)[:(-len(u)) % 4])
    if len(u) == W.shape[0]:
        return (lambda x: x), W
    if u[-1] == len(u) - 1:
        return (lambda x: x[:, :len(u)]), W[u]
    return _col_select(u, W.shape[0]), W[u]


def _shifted(c, left, right, out):
    """Coupling c with its rows moved by these offsets."""
    return dict(c, rows_a=np.asarray(c["rows_a"]) + left, rows_b=np.asarray(c["rows_b"]) + right,
                dest=np.asarray(c["dest"]) + out)


def _layer0_plan(T):
    """Layer 0 as (classes, columns of A): taper groups of one width run as one
    block-diagonal base.  A class: u, members, radial segments (degree, channels), paths, n_out rows
    of A, and `first` (per member and order: first row, couplings, the first message layer's
    reduction).  Columns of A: [class | class ...], each one's component and its place among all
    degrees' channels."""
    V, deg = T["arch"], _degrees(T["lmax"])
    classes = {}
    for b, (_, lo, hi) in enumerate(V["groups"]):
        classes.setdefault(hi - lo, []).append(b)
    plan, col_lm, col_at = [], [], []
    start = np.cumsum([0] + list(V["chan"]))
    for u, members in classes.items():
        c = dict(u=u, members=members, segments=[], paths=[], first={}, products={}, sizes=[])
        n = {1: 0, 2: 0, 3: 0, 4: 0}
        for b in members:
            gmax, lo, hi = V["groups"][b]
            n1, ps, first = (gmax + 1) ** 2, V["prod0"][b], dict(n)
            c["sizes"].append(n1)
            c["paths"] += [(len(c["segments"]) + int(deg[lm]), lm, n[1] + lm, 1.0) for lm in range(n1)]
            c["segments"] += [(l, lo, hi) for l in range(gmax + 1)]
            col_lm += [np.repeat(np.arange(n1), u)]
            col_at += [start[deg[np.repeat(np.arange(n1), u)]] + np.tile(np.arange(lo, hi), n1)]
            n[1] += n1
            for o, key, left, right in ((2, "c2", 1, 1), (3, "c3", 2, 1), (4, "c4", 2, 2)):
                if key in ps:
                    x = _shifted(ps[key], first[left], first[right], first[o])
                    y = c["products"].get(key)
                    c["products"][key] = x if y is None else dict(
                        {k: np.concatenate([y[k], x[k]]) for k in ("rows_a", "rows_b", "dest")},
                        factors=np.concatenate([np.asarray(y["factors"]), np.asarray(ps[key]["factors"])]))
                    n[o] += int(ps[key]["n_out"])
                    c["products"][key]["n_out"] = n[o]
            c["first"][b] = first
        c["n_out"] = n[1]
        reds = [V["layers"][0]["reduce"][b] for b in members]
        red = dict(reds[0], orders=sorted({o for r in reds for o in r["orders"]}))
        for o in red["orders"]:
            has = [(r, b) for r, b in zip(reds, members) if o in r["orders"]]
            red[f"rows{o}"] = np.concatenate([np.asarray(r[f"rows{o}"]) + c["first"][b][o] for r, b in has])
            red[f"slot{o}"] = np.concatenate([np.asarray(r[f"slot{o}"]) for r, _ in has])
            red[f"w{o}"] = np.concatenate([np.asarray(r[f"w{o}"]) for r, _ in has])
        c["reduce"] = red
        plan.append(c)
    return plan, (np.concatenate(col_lm), np.concatenate(col_at))


def _pair_tables(T, present):
    """Per-pair weights of the species present (padded), an argument of the steps: layer 0's
    (with up to _PAIR_MATMUL pairs the matrix of layer0_species, for one species its non-zero part
    (basis, columns); else (pairs * basis, columns) per bond sum), the message's and the back-flow's
    radial functions (the back-flow via the order-1 map of the reverse pair's weights), side by side
    as "radial"."""
    V, n, dt = T["arch"], present.shape[0], T["head"].dtype
    S, k, n_basis, n_lm, full = V["n_species"], V["n_message"], T["n_basis"], T["n_lm"], jax.lax.Precision.HIGHEST
    plan, (col_lm, col_at) = _layer0_plan(T)
    sp = jnp.minimum(present, S - 1)                     # padded entries: never a bond's pair
    centre, nbr = jnp.repeat(sp, n), jnp.tile(sp, n)
    F, off, total = [], 0, sum(V["chan"])
    for d, c in zip(V["degrees"], V["chan"]):
        row = V["film"][centre * S + nbr]
        w = d["w"][None] * (1.0 + row[:, None, off:off + c]) + d["film"][None] * row[:, None, total + off:total + off + c]
        F.append(w * V["chem0"][nbr][:, None, :c])
        off += c
    flat = lambda parts: jnp.concatenate(parts, axis=2).reshape(n * n * n_basis, -1)
    out = dict(present=present)
    if n * n <= _PAIR_MATMUL and n == 1:
        out["species"] = jnp.concatenate(F, axis=2)[0][:, col_at]
    elif n * n <= _PAIR_MATMUL:
        hot = jnp.asarray(col_lm[:, None] == np.arange(n_lm), dt)
        X = jnp.concatenate(F, axis=2)[:, :, col_at, None] * hot             # (pair, basis, column, lm)
        X = jnp.transpose(X.reshape(n, n, n_basis, -1, n_lm), (1, 2, 4, 0, 3))
        out["species"] = X.reshape(n * n_basis * n_lm, -1)
    else:
        out["layer0"] = [flat([F[l][:, :, lo:hi] for l, lo, hi in c["segments"]]) for c in plan]
    rev = (jnp.arange(n * n) % n) * n + jnp.arange(n * n) // n
    Q = [jnp.matmul(F[l][rev], V["degrees"][l]["order1"], precision=full) for l in range(V["lmax_ind"] + 1)]
    # [message | back-flow]: one product per bond
    out["radial"] = jnp.concatenate([flat([F[l][:, :, :k] for l in range(T["lmax_feat"] + 1)]), flat(Q)], axis=1)
    return out


def _energy_fns(T, dt, method):
    """Energies: (dense table, one chunk's own atoms, table rows of several widths), each taking ls
    and `present` last; method None: plain XLA.  Also the parts for the LAMMPS stages, which call
    between the exchanges of the messages.

    Layer 0 needs no array per bond and channel with at most _PAIR_MATMUL pairs (layer0_species).
    The back-flow is one more operand of the message's bond sum (_backflow_paths).

    Bounds (model.py): the gate of each bond sum is one factor per row, taken before its products;
    tau squashes the whitened features, so the first readout layer reads them per part; the core
    switch scales the site energy by c(z / z_c) of the row's ZBL energy."""
    V = T["arch"]
    lmax_feat, lmax_ind, k = T["lmax_feat"], V["lmax_ind"], V["n_message"]
    rcut, n_lm, scale = T["rcut"], T["n_lm"], T["energy_scale"]
    deg, n_lmI = _degrees(T["lmax"]), (lmax_ind + 1) ** 2
    full = jax.lax.Precision.HIGHEST
    if method is None:
        edge = lambda name, *args, n_y=n_lm, group=1, n_flow=0: _xla_edge(*args)
        stack = _xla_rows
    else:
        call = _cuex_call(dt, method)
        edge = lambda name, *args, n_y=n_lm, group=1, n_flow=0: _cuex_edge(call, name, n_y, *args, group, n_flow)
        stack = lambda *args, **kw: _cuex_stack(call, *args, **kw)

    def stack1(ps, n1, want, u, head=None):
        """(apply, rows) of one class; head: {order: rows} that F starts with."""
        apply, (rows,) = stack([(ps, n1, want, u)], u, order=[_head_first(want, head)] if head else None)
        return apply, rows

    plan, _ = _layer0_plan(T)
    runs = [(n1, c["u"]) for c in plan for n1 in c["sizes"]]
    w1, b1, act1 = _first_layer(T)
    kappa, a0, tau, zc = T.get("bound"), T.get("bond_scale"), T.get("feat_bound"), T.get("zbl_switch")
    fold = bool(kappa) and method is not None and _message_group(k) > 1      # gated back-flow Y as a fused operand
    if tau:                              # last row of w1: each readout row's feature + 1 (_readout_rows)
        w1 = np.vstack([w1, np.arange(1, T["n_features"] + 1)])

    def whiten(idx):
        """x -> the features of the columns x (idx: the feature of each, -1: none) in units of their
        spread, squashed by tau."""
        mean, std = (jnp.asarray(np.append(np.asarray(T[name], np.float64), pad)[idx], dt)
                     for name, pad in (("feat_mean", 0.0), ("feat_std", 1.0)))

        def squash(x):
            x = (x - mean) / std
            return x * jax.lax.rsqrt(1.0 + (x / tau) ** 2)
        return squash

    def reader(W):
        """F -> the first readout layer's input from the columns of F that W reads; with feat_bound W's
        last column is the feature of each row (+ 1, 0: none), which whiten takes."""
        select, W = _read_columns(W)
        if not tau:
            W = jnp.asarray(W, dt)
            return lambda F: select(F) @ W
        squash, W = whiten(W[:, -1].astype(int) - 1), jnp.asarray(W[:, :-1], dt)
        return lambda F: squash(select(F)) @ W

    def gate(t, parts):
        """The per-row gate factor f = a0 / sigma_t (1 + rho^2 / kappa^2)^(-1/2) of layer t, with
        rho^2 = mean(A^2) / sigma_t^2 over all components of its bond sums (a list of arrays)."""
        sigma = T["bond_rms"][t]
        m = sum(jnp.sum(a * a, axis=1) for a in parts) / sum(a.shape[1] for a in parts)
        return (a0 / sigma) * jax.lax.rsqrt(1.0 + m / (sigma * kappa) ** 2)

    read = {}                            # per class: order -> (first feature, rows read) per group
    col = 0
    for b, o in V["features0"]:
        c = next(c for c in plan if b in c["members"])
        inv = np.asarray(V["inv0"][b][o]) + c["first"][b][o]
        read.setdefault(c["u"], []).append((o, col, inv))
        col += c["u"] * len(inv)
    wants, heads, bounds = [], [], [0]
    for c in plan:
        c["op"] = edge("vv_edge0", [(c["paths"], 0)], len(c["segments"]), c["n_out"], c["u"])
        inv = {}
        for o, _, r in read.get(c["u"], []):
            inv.setdefault(o, []).extend(int(x) for x in r)
        wants.append(_wanted(c["reduce"], inv))
        heads.append({o: sorted({int(x) for x in r}) for o, r in inv.items()})
        bounds.append(bounds[-1] + c["n_out"] * c["u"])
    # the classes are one polynomial when their channels have a common divisor of some size
    cuts = [0, len(plan)] if math.gcd(*(c["u"] for c in plan)) >= _MERGE_CHANNELS else range(len(plan) + 1)
    orders = [_head_first(w, h) for w, h in zip(wants, heads)]
    prod0, rows, perm = [], [], None
    for i, j in zip(cuts[:-1], cuts[1:]):
        apply, r = stack([(c["products"], c["n_out"], w, c["u"]) for c, w in zip(plan[i:j], wants[i:j])],
                         math.gcd(*(c["u"] for c in plan[i:j])), order=orders[i:j] if len(cuts) == 2 else None)
        prod0.append((apply, bounds[i], bounds[j]))
        rows += r
        perm = apply.perm
    W1 = np.concatenate([sum(_readout_rows(w1, r, c["u"], {o: (at, x)}) for o, at, x in read.get(c["u"], []))
                         for c, r in zip(plan, rows)])
    read0 = reader(W1 if perm is None else W1[perm])

    layers, u_prev, reds = [], [c["u"] for c in plan], [c["reduce"] for c in plan]
    for t, lay in enumerate(V["layers"]):
        terms, n_out = _coupling_terms(lay["ycouple"])
        sources = [([(int(deg[l]), l, r, o, cg) for l, r, cg, o in terms], n_lmI)]
        n_rad, n_y, flow = lmax_feat + 1, n_lm, None
        if t == 0:
            # the back-flow's radials follow the message's in one operand (also its node);
            # with the gate, its Y columns scaled by the sender's factor follow Y or, fused, are an operand
            cols = sorted({lm for _, lm, _, _, _ in V["backflow"]})
            flow = slice(cols[0], cols[-1] + 1) if cols == list(range(cols[0], cols[-1] + 1)) else cols
            n_rad, n_y = n_rad + lmax_ind + 1, n_lm + (len(cols) if kappa and not fold else 0)
            at = lambda lm: cols.index(lm) + (0 if fold else n_lm) if kappa else lm
            sources.append(([(l, at(lm), lmax_feat + 1 + r, o, c) for l, lm, r, o, c in V["backflow"]], n_rad))
        n_flow = len(cols) if fold and flow is not None else 0
        W = _reduction(reds, rows, u_prev)
        out = dict(W=jnp.asarray(W if perm is None else W[perm], dt), n_out=n_out, prod=None, flow=flow, fold=bool(n_flow),
                   op=edge("vv_edge", sources, n_rad, n_out, k, n_y=n_y, group=_message_group(k), n_flow=n_flow))
        if lay["prod"] is not None:
            inv = {o: r for (tt, o), r in V["inv"].items() if tt == t + 1}
            red = V["layers"][t + 1]["reduce"][0] if t + 1 < len(V["layers"]) else None
            out["prod"], r = stack1(lay["prod"], n_out, _wanted(red, inv), k, {o: sorted({int(i) for i in x}) for o, x in inv.items()})
            perm = out["prod"].perm
            x = {}
            for tt, o in V["features"]:
                if tt == t + 1:
                    x[o] = col, inv[o]
                    col += k * len(inv[o])
            Wr = _readout_rows(w1, r, k, x)
            out["read"] = reader(Wr if perm is None else Wr[perm])
            rows, u_prev = [r], [k]
            reds = [red] if red is not None else []
        else:
            perm = None
        layers.append(out)

    G = V["gram"]
    h, r, rb = 0 if tau else w1.shape[0], G["rank"], G["bispec_rank"]     # tau: the scalars are late features
    n_m = (lmax_feat + 1) ** 2
    col0, fused = col, method is not None
    Wg, col = _gram_tables(V, layers[-1]["n_out"], w1[:h], col, (max(r, rb),) * 2 if fused else None)
    n_tri = (lmax_feat + 1) * r * (r + 1) // 2
    r3 = {3: list(range(G["n_bispec"]))}
    # GEMM dimensions that are multiples of 4 (16-byte aligned rows), as _read_columns
    n_y = Wg.shape[1]
    Wg = jnp.asarray(np.pad(Wg, ((0, -Wg.shape[0] % 4), (0, -n_y % 4))), dt)
    late = [np.concatenate([w1[:, col:col + n_tri].T,
                            _readout_rows(w1, r3, rb, {3: (col + n_tri, np.arange(G["n_bispec"]))})])]
    lead = 0
    if tau:
        rows0, nf = np.asarray(G["degrees"][0]["rows"]), len(G["degrees"][0]["rows"])
        lead = nf * k
        scalars = _col_select((rows0[:, None] * k + np.arange(k)).reshape(-1), layers[-1]["n_out"] * k)
    if fused:
        u, n_seg, quad, cubic, at = _gram2(G, lmax_feat, lead)
        from .fused import gram2_poly
        gram = gram2_poly(2 * n_m, u, n_seg, quad, cubic)
        W_gram = np.zeros((n_seg * u, late[0].shape[1]))
        W_gram[at] = late[0]
        late = [W_gram]
    else:
        triangle, (bispec, _) = _triangle(r), stack1(G["products"], n_m, r3, rb)
    if tau:
        late.insert(0, w1[:, col0 + (np.arange(nf)[:, None] + np.arange(k) * nf).reshape(-1)].T)
    W_late = np.concatenate(late)
    W_late = np.pad(W_late, ((0, -W_late.shape[0] % 4), (0, 0)))
    if tau:
        squash_late, W_late = whiten(W_late[:, -1].astype(int) - 1), W_late[:, :-1]
    W_late = jnp.asarray(W_late, dt)
    n_late = W_late.shape[0] - lead
    col += n_tri + rb * G["n_bispec"]
    assert col == T["n_features"], (col, T["n_features"])
    b1 = jnp.asarray(b1, dt)

    def geometry(vec, mu_c, mu_n):
        """(r, g, Y) of the bond vectors."""
        r = jnp.maximum(jnp.sqrt(jnp.sum(vec * vec, axis=-1)), 1e-8)
        rhat = vec / r[:, None]
        u = vcut.scaled(T, r, mu_c, mu_n)
        gY = jnp.concatenate([radial_basis(u, rcut, T["n_basis"], T["cheb_p"], jnp), angular(rhat, T["lmax"], jnp)], axis=1)
        return r, gY[:, :T["n_basis"]], gY[:, T["n_basis"]:]

    def layer0_species(g, Y, sp_n, sp_r, blocks, X, n):
        """All groups' A, [group | group ...]: the sums of g Y per neighbour species, in full precision
        (geometry), times X; sp_n, sp_r: the species of the neighbours (bonds) and the rows.  One
        species: X holds each column's weights per basis function, and both products are sums of
        elementwise products."""
        ends = np.cumsum([0] + [c * K for c, K in blocks])
        if n == 1:
            G = jnp.concatenate([(g[e0:e1].reshape(c, K, -1, 1) * Y[e0:e1].reshape(c, K, 1, -1)).sum(1)
                                 for e0, e1, (c, K) in zip(ends[:-1], ends[1:], blocks)])
            at, A = 0, []
            for n1, u in runs:
                A.append((G[:, :, :n1, None] * X[:, at:at + n1 * u].reshape(1, -1, n1, u)).sum(1).reshape(G.shape[0], -1))
                at += n1 * u
            return jnp.concatenate(A, axis=1)
        hot = (sp_n[:, None] == jnp.arange(n, dtype=sp_n.dtype)).astype(g.dtype)
        g = (hot[:, :, None] * g[:, None, :]).reshape(g.shape[0], -1)
        G = jnp.concatenate([jnp.einsum("aKj,aKm->ajm", g[e0:e1].reshape(c, K, -1), Y[e0:e1].reshape(c, K, -1), precision=full)
                             for e0, e1, (c, K) in zip(ends[:-1], ends[1:], blocks)])
        A = jnp.matmul(G.reshape(G.shape[0], -1), X, precision=full)
        hot = (sp_r[:, None] == jnp.arange(n, dtype=sp_r.dtype)).astype(g.dtype)
        return jnp.sum(A.reshape(A.shape[0], n, -1) * hot[:, :, None], axis=1)

    def bond_radials(g, pair, H, t):
        """Each bond's radial functions of the message and, for message layer t = 0, of the back-flow,
        side by side in one product."""
        table = H["radial"] if layers[t]["flow"] is not None else H["radial"][:, :(lmax_feat + 1) * k]
        return _pair_radial(g, pair, table, H["present"].shape[0] ** 2)

    def first_layer(g, Y, pair, sp_r, H, blocks, ind_i):
        """(the products of layer 0, its gate factors f or None); H: _pair_tables; pair, sp_r: each bond's pair
        of species and each row's species, among H's."""
        n = H["present"].shape[0]
        R = sum(c for c, _ in blocks)
        if "species" in H:
            A = layer0_species(g, Y, pair % n, sp_r, blocks, H["species"], n)
        else:
            A = jnp.concatenate([c["op"](_pair_radial(g, pair, h0, n * n), Y, [(None, None)], ind_i, blocks, R)
                                 for c, h0 in zip(plan, H["layer0"])], axis=1)
        f = None
        if kappa:
            f = gate(0, [A])
            A = A * f[:, None]
        F = [p(A[:, c0:c1], R) for p, c0, c1 in prod0]
        return (F[0] if len(F) == 1 else jnp.concatenate(F, axis=1)), f

    def send(F, t):
        """The messages of message layer t, from the products of the layer before."""
        return F @ layers[t]["W"]

    def receive(t, It, g, Y, radials, blocks, ind_i, nbr, f0=None):
        """(bond sum, its products or None) of message layer t; radials: bond_radials(.., t);
        nbr: each bond's row of the messages It; f0: layer 0's gate factors (first_layer), which
        scale the senders' back-flow."""
        lay_t, flow = layers[t], layers[t]["flow"]
        R, E = sum(n for n, _ in blocks), sum(n * K for n, K in blocks)
        rad, Y = radials[:E], Y[:E]
        more = {}
        if flow is not None and kappa:
            Ys = Y[:, flow]
            Ys = Ys if f0 is None else Ys * f0[nbr][:, None]
            if lay_t["fold"]:
                more = dict(flow=Ys)
            else:
                Y = jnp.concatenate([Y, Ys], axis=1)
        last = lay_t["op"](rad, Y, [(It, nbr)], ind_i, blocks, R, **more)
        if kappa:
            last = last * gate(t + 1, [last])[:, None]
        return last, None if lay_t["prod"] is None else lay_t["prod"](last, R)

    def read(t, F):
        """What products F add to the first readout layer's input; t: their message layer, None: layer 0."""
        return read0(F) if t is None else layers[t]["read"](F)

    def finish(x, last, mu, R):
        """The site energies of R rows from the first readout layer's input x and the last bond sum."""
        y = jnp.pad(last, ((0, 0), (0, Wg.shape[0] - last.shape[1]))) @ Wg
        if not fused:
            P, off, late = y[:, h:h + n_m * r], 0, []
            for l in range(lmax_feat + 1):
                late.append(triangle(P[:, off:off + (2 * l + 1) * r].reshape(R, 2 * l + 1, r)))
                off += (2 * l + 1) * r
            late.append(bispec(y[:, h + n_m * r:n_y], R))
            late.append(jnp.zeros((R, n_late - sum(z.shape[1] for z in late)), x.dtype))
        else:
            xg = y[:, h:n_y]
            late = [call(gram, [xg, xg], (R, n_seg * u), None, "vv_gram2")]
            if n_late > n_seg * u:
                late.append(jnp.zeros((R, n_late - n_seg * u), xg.dtype))
        if tau:
            late.insert(0, scalars(last))
        late = jnp.concatenate(late, axis=1)
        x = x + squash_late(late) @ W_late if tau else x + y[:, :h] + late @ W_late
        mu = mu if jnp.ndim(mu) == 0 else mu[:R]
        x = _mlp_rest(T, x + b1 + T["elem_bias"][mu], act1)
        return jnp.sum(x * T["head"][mu], axis=-1) * scale

    def site_energy(g, Y, pair, sp_r, H, lay, mu):
        """lay[t]: (blocks, centres, neighbours as rows) of layer t, whose bonds are the first of g and Y."""
        blocks, ind_i, _ = lay[0]
        F, f0 = first_layer(g, Y, pair, sp_r, H, blocks, ind_i)
        Fs = [(None, F)]
        for t, (blocks, ind_i, nbr) in enumerate(lay[1:]):
            last, F = receive(t, send(Fs[-1][1], t), g, Y, bond_radials(g, pair, H, t), blocks, ind_i, nbr, f0)
            if F is not None:
                Fs.append((t, F))
        R = sum(n for n, _ in blocks)
        x = read(None, Fs[0][1][:R])
        for t, F in Fs[1:]:
            x = x + read(t, F[:R])
        return finish(x, last, mu, R)

    def energy(positions, shift, ind_i, ind_j, mu, amask, ls, present, blocks):
        rows = np.cumsum([0] + [n for n, _ in blocks])
        centres = jnp.concatenate([jnp.repeat(positions[r0:r1], K, axis=0)
                                   for r0, r1, (_, K) in zip(rows[:-1], rows[1:], blocks)])
        mu_c, mu_n = _species_of(mu, present, ind_i), _species_of(mu, present, ind_j)
        r, g, Y = geometry(positions[ind_j] - centres + shift, mu_c, mu_n)
        e = site_energy(g, Y, ls[ind_i] * present["present"].shape[0] + ls[ind_j], ls, present,
                        [(blocks, ind_i, ind_j)] * T["n_layers"], mu if jnp.ndim(mu_c) else mu_c)
        if T["zbl"] is not None:
            z = _row_sum(_zbl_pairs(T["zbl"], r, mu_c, mu_n), blocks)
            e = (e * _core_switch(z, zc) if zc else e) + z
        e = e * amask
        return jnp.sum(e), e

    def energy_dense(positions, shift, nbr, mu, amask, ls, present):
        a, K = nbr.shape
        return energy(positions, shift.reshape(-1, 3), jnp.repeat(jnp.arange(a, dtype=nbr.dtype), K),
                      nbr.reshape(-1), mu, amask, ls, present, ((a, K),))

    def energy_rows(positions, shift, rows, nbr_g, nbr_l, mu, cmask, n_rows, ls, present):
        R0, K = nbr_g.shape
        mu_r, mu_n = mu[rows], mu[nbr_g].reshape(-1)
        mu_c = jnp.repeat(mu_r, K)
        r, g, Y = geometry((positions[nbr_g] - positions[rows][:, None, :] + shift).reshape(-1, 3), mu_c, mu_n)
        pair = jnp.repeat(ls[rows], K) * present["present"].shape[0] + ls[nbr_g].reshape(-1)
        lay = [(((R, K),), jnp.repeat(jnp.arange(R, dtype=jnp.int32), K), nbr_l[:R].reshape(-1)) for R in n_rows]
        e = site_energy(g, Y, pair, ls[rows], present, lay, mu_r)
        if T["zbl"] is not None:
            c = n_rows[-1]
            z = jnp.sum(_zbl_pairs(T["zbl"], r[:c * K], mu_c[:c * K], mu_n[:c * K]).reshape(c, K), axis=1)
            e = (e * _core_switch(z, zc) if zc else e) + z
        e = e * cmask
        return jnp.sum(e), e

    parts = dict(geometry=geometry, bond_radials=bond_radials, first_layer=first_layer, send=send,
                 receive=receive, read=read, finish=finish)
    return energy_dense, energy_rows, energy, parts


def _energies(T, dt, kernels):
    """(dense, rows, blocks) energies."""
    return _energy_fns(T, dt, None if kernels == "xla" else _method(kernels))[:3]


def _build_chunked(T, dt, kernels):
    """md(positions, cell, rows, cr, cj, cjl, cs, cok, cmask, mu, volume, ref, *species, n_atoms, width,
    want_stress, row_counts); per chunk (leading axis) its table rows, candidate pairs
    (row cr, neighbour atom cj and row cjl, image cs, valid cok) and own-atom mask; species: ls,
    present."""
    energy_rows = _energies(T, dt, kernels)[1]
    rcut = T["rcut"]
    far = jnp.asarray([2.0 * rcut, 0.0, 0.0], dt)

    def md(positions, cell, rows, cr, cj, cjl, cs, cok, cmask, mu, volume, ref, *species, n_atoms, width,
           want_stress, row_counts, rank=None):
        n_chunks, R = rows.shape
        c = row_counts[-1]
        take = lambda x, k: jax.lax.dynamic_index_in_dim(x, k, keepdims=False)

        def chunk(k, carry):
            E, gpos, gstr, e_atom, need, n_edges = carry
            rows_k, cr_k, cj_k, cjl_k = take(rows, k), take(cr, k), take(cj, k), take(cjl, k)
            cmask_k = take(cmask, k)
            cshift = _exact(take(cs, k).astype(dt), cell)
            vec = positions[cj_k] - positions[rows_k[cr_k]] + cshift
            inside = take(cok, k) & (jnp.sum(vec * vec, axis=1) < rcut * rcut)
            inside = vcut.inside(T, inside, vec, mu[rows_k[cr_k]], mu[cj_k])
            ins = inside.astype(jnp.int32)
            cnt = jax.ops.segment_sum(ins, cr_k, num_segments=R, indices_are_sorted=True)
            slot = jnp.cumsum(ins) - 1 - (jnp.cumsum(cnt) - cnt)[cr_k]
            ok = inside & (slot < width)
            row = jnp.where(ok, cr_k, R)
            slot = jnp.where(ok, slot, 0)
            nbr_g = jnp.broadcast_to(rows_k[:, None], (R, width)).at[row, slot].set(cj_k, mode="drop")
            nbr_l = jnp.broadcast_to(jnp.arange(R, dtype=jnp.int32)[:, None],
                                     (R, width)).at[row, slot].set(cjl_k, mode="drop")
            shift = jnp.broadcast_to(far, (R, width, 3)).at[row, slot].set(cshift, mode="drop")

            def total(pos, strain):
                sh = shift
                if strain is not None:
                    sym = 0.5 * (strain + strain.T)
                    pos = pos + _exact(pos, sym)
                    sh = sh + _exact(sh, sym)
                return energy_rows(pos, sh, rows_k, nbr_g, nbr_l, mu, cmask_k, row_counts, *species)

            if want_stress:
                (Ek, ek), (gp, gs) = jax.value_and_grad(total, argnums=(0, 1), has_aux=True)(
                    positions, jnp.zeros((3, 3), dt))
                gstr = gstr + gs
            else:
                (Ek, ek), gp = jax.value_and_grad(total, has_aux=True)(positions, None)
            return (E + Ek, gpos + gp, gstr, e_atom.at[rows_k[:c]].add(ek),
                    jnp.maximum(need, jnp.max(cnt)), n_edges + jnp.sum(cnt[:c], dtype=jnp.int32))

        init = (jnp.zeros((), dt), jnp.zeros_like(positions), jnp.zeros((3, 3), dt),
                jnp.zeros(n_atoms, dt), jnp.int32(0), jnp.int32(0))
        E, gpos, gstr, e_atom, need, n_edges = jax.lax.fori_loop(0, n_chunks, chunk, init)
        stress = _voigt(gstr, volume) if want_stress else jnp.zeros(6, dt)
        return _pack(E, stress, positions, ref, jnp.stack([need, n_edges]), -gpos, e_atom, rank)

    return md


def _pair_bins(pos, valid, cell, pbc, cutoff, xp):
    """Cell-list bins at least `cutoff` wide, with xp numpy or jax.numpy.  A periodic axis is binned
    in fractional coordinates after the lattice translation that moves the atom into the cell; an
    open axis in Cartesian coordinates from its lowest valid atom (at most 1024 bins).  Returns each
    atom's bin (n, 3), the translations taken off it (n, 3), the bins per axis (3,) and the images
    per axis a pair can need (3,)."""
    per = np.asarray(pbc, bool)
    if per.any():
        u = xp.stack([xp.cross(cell[1], cell[2]), xp.cross(cell[2], cell[0]), xp.cross(cell[0], cell[1])])
        det = xp.sum(cell[0] * u[0])
        s = (pos[:, :1] * u[:, 0] + pos[:, 1:2] * u[:, 1] + pos[:, 2:] * u[:, 2]) / det
        w = xp.abs(det) / xp.sqrt(xp.sum(u * u, axis=1))                # lattice-plane spacings
    else:
        s, w = pos, xp.ones(3, pos.dtype)
    wrap = xp.where(per, xp.floor(s), 0.0)
    s = s - wrap
    lo = xp.where(per, 0.0, xp.min(xp.where(valid[:, None], s, xp.inf), axis=0))
    hi = xp.where(per, 1.0, xp.max(xp.where(valid[:, None], s, -xp.inf), axis=0))
    # a hair wider than the cutoff: rounding cannot put a pair inside it two bins apart
    c = cutoff * (1 + 1e-4) / w
    n_per = xp.clip(xp.floor(1 / c), 1, 1024)
    width = xp.where(per, 1 / n_per, xp.maximum(c, (hi - lo) / 1023))
    n = xp.where(per, n_per, xp.floor((hi - lo) / width) + 1)
    b = xp.clip(xp.floor((s - lo) / width), 0, n - 1)
    reach = xp.where(per & (c > 1), xp.ceil(c), 1)
    return b.astype(np.int32), wrap.astype(np.int32), n.astype(np.int32), reach.astype(np.int32)


def _plan_key(ref, h, pbc, n, n_pad, cap, rcut, skin, ordered, wide):
    """The static sizes of _build_pairs for a structure, from the bins of its host-built list
    (positions ref, cell h) with room to grow; None for a singular periodic cell."""
    pbc = tuple(bool(p) for p in pbc)
    if any(pbc) and abs(np.linalg.det(h)) < 1e-9:
        return None
    b, _, nb, reach = _pair_bins(ref, np.ones(n, bool), h, pbc, rcut + skin, np)
    most = int(np.bincount(b[:, 0] + nb[0] * (b[:, 1] + nb[1] * b[:, 2])).max())
    return (n_pad, cap, pbc, tuple(int(r) for r in reach), padded_size(1.5 * int(np.prod(nb)), granule=64),
            padded_size(1.25 * most + 2, granule=8), ordered, wide)


def _build_pairs(T, dt, n_pad, cap, pbc, reach, n_bins, occupancy, ordered, wide):
    """Verlet candidates built on the device by a cell list: pairs(positions, cell, amask, mu, ls,
    cls, rcut, skin) -> ((cand_i, cand_j, cand_s, cand_ok, ref) + (mu, ls, order, rank) if ordered
    + (row_start,), stats = [pairs, most atoms in a bin, bins, enough images, most neighbours]).
    Pairs within rcut + skin, in the layout of the host list for _build_dense, at most cap of them.
    positions, mu, ls in the order of the Atoms; ordered: along a Z-order curve of the bins (species
    with cls 0 first).  The bins are a table of n_bins with at most `occupancy` atoms each; stats
    beyond the static sizes mean the list is incomplete."""
    i32, per, rc_model = jnp.int32, np.asarray(pbc, bool), T["rcut"]
    combos = np.stack(np.meshgrid(*[np.arange(-k, k + 1) for k in reach], indexing="ij"), -1).reshape(-1, 3)
    # neighbour bins in groups: no (atoms, bins, occupancy) array beyond ~2^24 entries
    G = int(np.clip((1 << 24) // (n_pad * occupancy), 1, len(combos)))
    n_groups = -(-len(combos) // G)
    offsets = np.zeros((n_groups * G, 3), np.int32)
    offsets[:len(combos)] = combos
    real = np.arange(n_groups * G) < len(combos)
    n_w, extra = wide

    def pairs(positions, cell, amask, mu, ls, cls, rcut, skin):
        valid = amask > 0
        b, wrap, nb, images = _pair_bins(positions, valid, cell, pbc, rcut + skin, jnp)
        pos, out = positions, ()
        if ordered:
            code = _spread3(b[:, 0]) | (_spread3(b[:, 1]) << 1) | (_spread3(b[:, 2]) << 2)
            order = jnp.argsort(jnp.where(valid, cls[mu] + code, (1 << 31) - 1), stable=True).astype(i32)
            rank = jnp.zeros(n_pad, i32).at[order].set(jnp.arange(n_pad, dtype=i32))
            b, wrap, pos, mu, ls = b[order], wrap[order], pos[order], mu[order], ls[order]
            out = (mu, ls, order, rank)
        key = jnp.where(valid, b[:, 0] + nb[0] * (b[:, 1] + nb[1] * b[:, 2]), n_bins)
        by_bin = jnp.argsort(key, stable=True).astype(i32)
        count = jax.ops.segment_sum(valid.astype(i32), key, num_segments=n_bins)
        start = jnp.cumsum(count, dtype=i32) - count
        m, me = jnp.arange(occupancy, dtype=i32), jnp.arange(n_pad, dtype=i32)[:, None, None]

        def group(g):
            """The candidates of every atom in the bins of offset group g: (listed, inside rcut, j, S)."""
            v = b[:, None, :] + jax.lax.dynamic_slice_in_dim(jnp.asarray(offsets), g * G, G)[None]
            img = jnp.floor_divide(v, nb)
            ok = jax.lax.dynamic_slice_in_dim(jnp.asarray(real), g * G, G) & jnp.all(per | (img == 0), axis=-1)
            img = jnp.where(per, img, 0)
            a = v - img * nb
            nkey = jnp.where(ok, a[..., 0] + nb[0] * (a[..., 1] + nb[1] * a[..., 2]), 0)
            j = by_bin[jnp.minimum(start[nkey][..., None] + m, n_pad - 1)]
            S = (img[:, :, None, :] + wrap[:, None, None, :] - wrap[j]).astype(dt)
            # the step's own bond and distance arithmetic, elementwise (full precision)
            vec = pos[j] - pos[:, None, None, :] + S[..., :1] * cell[0] + S[..., 1:2] * cell[1] + S[..., 2:] * cell[2]
            d2 = jnp.sum(vec * vec, axis=-1)
            keep = ((m < jnp.where(ok, count[nkey], 0)[..., None]) & valid[:, None, None]
                    & (d2 < (rcut + skin) ** 2) & ((j != me) | jnp.any(S != 0, axis=-1)))
            keep = vcut.listed(T, keep, vec, mu[:, None, None], mu[j], skin)
            inside = keep & (d2 < rc_model * rc_model)
            inside = vcut.inside(T, inside, vec, mu[:, None, None], mu[j])
            return keep.reshape(n_pad, -1), inside.reshape(n_pad, -1), j.reshape(n_pad, -1), S.reshape(n_pad, -1, 3)

        def counts(g, acc):
            keep, inside = group(g)[:2]
            return acc[0] + jnp.sum(keep, axis=1, dtype=i32), acc[1] + jnp.sum(inside, axis=1, dtype=i32)

        cnt, cin = jax.lax.fori_loop(0, n_groups, counts, (jnp.zeros(n_pad, i32), jnp.zeros(n_pad, i32)))
        first = jnp.cumsum(cnt, dtype=i32) - cnt

        def write(g, acc):
            # each atom's pairs at its offset from one prefix sum: sorted by atom
            cj, cs, done = acc
            keep, _, j, S = group(g)
            at = jnp.where(keep, first[:, None] + done[:, None] + jnp.cumsum(keep, axis=1, dtype=i32) - 1, cap)
            return cj.at[at].set(j, mode="drop"), cs.at[at].set(S, mode="drop"), done + jnp.sum(keep, axis=1, dtype=i32)

        cj, cs, _ = jax.lax.fori_loop(0, n_groups, write, (jnp.full(cap, n_pad - 1, i32), jnp.zeros((cap, 3), dt),
                                                           jnp.zeros(n_pad, i32)))
        total = jnp.sum(cnt)
        # spare slots point at the last atom, which keeps cand_i sorted
        ci = jnp.repeat(jnp.arange(n_pad, dtype=i32), cnt, total_repeat_length=cap)
        need = jnp.maximum(jnp.max(cin[:n_w]) - extra, jnp.max(cin[n_w:])) if n_w else jnp.max(cin)
        stats = jnp.stack([total, jnp.max(count), jnp.prod(nb), jnp.all(images <= np.asarray(reach)).astype(i32),
                           need]).astype(i32)
        row_start = jnp.concatenate([first, total[None].astype(i32)])
        return (ci, cj, cs, jnp.arange(cap) < total, pos) + out + (row_start,), stats

    return pairs


def _split_coords(pos, cell, pbc):
    """Wrapped fractional coordinates times the cell widths (plain positions without a cell)."""
    cell = np.asarray(cell, float)
    vol = abs(np.linalg.det(cell))
    if vol < 1e-9:
        return np.asarray(pos, float)
    f = np.asarray(pos, float) @ np.linalg.inv(cell)
    f[:, np.asarray(pbc, bool)] %= 1.0
    return f * vol / np.linalg.norm(np.cross(cell[[1, 2, 0]], cell[[2, 0, 1]]), axis=1)


def _bisect(q, n_parts):
    """Recursive coordinate bisection of points q into n_parts nearly equal blocks."""
    part = np.zeros(len(q), np.int32)
    todo = [(np.arange(len(q)), n_parts, 0)]
    while todo:
        idx, k, base = todo.pop()
        if k == 1:
            part[idx] = base
            continue
        k1, sub = k // 2, q[idx]
        axis = int(np.argmax(sub.max(axis=0) - sub.min(axis=0)))
        m = len(idx) * k1 // k
        order = np.argpartition(sub[:, axis], m) if 0 < m < len(idx) else np.arange(len(idx))
        todo += [(idx[order[:m]], k1, base), (idx[order[m:]], k - k1, base + k1)]
    return part


def _ranges(starts, counts):
    """Concatenated arange(s, s + c)."""
    counts = np.asarray(counts, np.int64)
    ends = np.cumsum(counts)
    return np.repeat(np.asarray(starts, np.int64) - (ends - counts), counts) + np.arange(int(counts.sum()))


def _chunk_plan(pos, cell, pbc, i, j, S, n, chunk, n_layers, caps=None):
    """Host arrays of the chunked step for chunks of at most `chunk` atoms, from a candidate
    list sorted by i; `caps` from an earlier plan keeps the compiled shape while it fits."""
    n_chunks = -(-n // int(chunk))
    part = _bisect(_split_coords(pos[:n], cell, pbc), n_chunks)
    order = np.argsort(part, kind="stable")
    sizes = np.bincount(part, minlength=n_chunks)
    starts = np.concatenate([[0], np.cumsum(sizes)])
    loc = np.empty(n, np.int64)                        # row within its own chunk
    loc[order] = np.arange(n) - starts[part[order]]
    deg = np.bincount(i, minlength=n)
    off = np.concatenate([[0], np.cumsum(deg)])
    hops = n_layers - 1
    big = int(np.abs(S).max(initial=0)) > 127

    chunks = []
    for k in range(n_chunks):
        groups = [order[starts[k]:starts[k + 1]]]       # own, 1-hop, 2-hop ...
        for _ in range(hops):
            nb = j[_ranges(off[groups[-1]], deg[groups[-1]])]
            new = np.unique(nb[part[nb] != k])
            for g in groups[1:]:
                new = new[~np.isin(new, g, assume_unique=True)]
            groups.append(new)
        chunks.append(groups)

    c_cap = -(-int(sizes.max()) // 8) * 8
    h_need = [max(len(g[h]) for g in chunks) for h in range(1, hops + 1)]
    p_need = max(int(sum(deg[x].sum() for x in g)) for g in chunks)
    fresh = dict(h=[padded_size(1.05 * x, granule=8) if x else 0 for x in h_need], p=padded_size(1.05 * p_need))
    h_cap, p_cap = fresh["h"], fresh["p"]
    if (caps and caps["c"] == c_cap and len(caps["h"]) == hops
            and all(a <= b <= 1.3 * f + 8 for a, b, f in zip(h_need, caps["h"], fresh["h"]))
            and p_need <= caps["p"] <= 1.3 * fresh["p"]):
        h_cap, p_cap = list(caps["h"]), caps["p"]
    base = np.cumsum([0, c_cap] + h_cap)               # first row of each group
    R = int(base[-1])
    row_counts = tuple(int(base[hops + 1 - t]) for t in range(hops + 1))

    rows = np.zeros((n_chunks, R), np.int32)
    cr = np.full((n_chunks, p_cap), R - 1, np.int32)   # padding: last row, still sorted
    cj = np.zeros((n_chunks, p_cap), np.int32)
    cjl = np.full((n_chunks, p_cap), R - 1, np.int32)
    cs = np.zeros((n_chunks, p_cap, 3), np.int32 if big else np.int8)
    cok = np.zeros((n_chunks, p_cap), bool)
    cmask = np.zeros((n_chunks, c_cap))
    for k, groups in enumerate(chunks):
        rows[k] = groups[0][0]                         # padding rows: a real atom
        halo = np.concatenate(groups[1:]) if hops else np.zeros(0, np.int64)
        local = (np.concatenate([base[h] + np.arange(len(g)) for h, g in enumerate(groups[1:], start=1)])
                 if hops else np.zeros(0, np.int64))
        srt = np.argsort(halo)
        halo, local = halo[srt], local[srt]
        p = 0
        for h, g in enumerate(groups):
            rows[k, base[h]:base[h] + len(g)] = g
            idx = _ranges(off[g], deg[g])
            m, jj = len(idx), j[idx]
            cr[k, p:p + m] = base[h] + np.repeat(np.arange(len(g)), deg[g])
            cj[k, p:p + m] = jj
            cs[k, p:p + m] = S[idx]
            cok[k, p:p + m] = True
            if h < hops:                               # a later layer reads these rows
                mine = part[jj] == k
                where = np.searchsorted(halo, jj[~mine])
                assert np.array_equal(halo[np.minimum(where, len(halo) - 1)], jj[~mine])
                lj = np.empty(m, np.int64)
                lj[mine] = loc[jj[mine]]
                lj[~mine] = local[where]
                cjl[k, p:p + m] = lj
            else:
                cjl[k, p:p + m] = cr[k, p:p + m]
            p += m
        cmask[k, :len(groups[0])] = 1.0
    return (rows, cr, cj, cjl, cs, cok, cmask), row_counts, dict(c=c_cap, h=h_cap, p=p_cap)


class JaxModel:
    """Compiled MD steps of one checkpoint (and one charge/spin state) on one device.

    dtype: "tf32" (float32, the network's matrix products in TensorFloat-32 on GPUs that
    have it), "float32" or "float64"; lattice translations and strains always in full
    precision (_exact)."""

    def __init__(self, tables, dtype="tf32", device=None):
        if str(dtype) not in DTYPES:
            raise ValueError(f"dtype must be one of {'|'.join(DTYPES)}, got {dtype!r}")
        self.precision = "default" if str(dtype) == "tf32" else "highest"
        self.dtype = np.dtype("float32" if str(dtype) == "tf32" else dtype)
        if self.dtype == np.float64 and not jax.config.read("jax_enable_x64"):
            raise RuntimeError("float64 needs enable_x64() before the model is built")
        self.device = jax_device(device)
        put = lambda x: jax.device_put(np.asarray(x, self.dtype), self.device)
        self.T = _cast(tables, self.dtype, self.device)
        self.T["readout"] = [(put(w), put(b), act) for w, b, act in tables["readout"]]
        self.species, self.rcut, self.n_layers = tables["species"], tables["rcut"], tables["n_layers"]
        self.e0 = np.asarray(tables["atomic_energies"], np.float64)
        del self.T["atomic_energies"]          # the host adds them in float64 (VerletCalculator)
        self.stress = bool(self.T.pop("stress", True))
        self.pair_rcut = vcut.host_table(tables)
        self.charge_spin_state = None
        self._steps, self._pairs = {}, {}
        self.pair_build = jax.jit(lambda present: _pair_tables(self.T, present))

    def pair_tables(self, present):
        """_pair_tables of the species present (padded), once per set of species."""
        key = present.tobytes()
        if key not in self._pairs:
            if len(self._pairs) >= _MAX_PAIR_TABLES:
                self._pairs.clear()
            self._pairs[key] = self.pair_build(jax.device_put(present, self.device))
        return self._pairs[key]

    def md_step(self, kernels="xla", ordered=False):
        """The jitted step; ordered: with (order, rank) as its last arguments (_in_order).
        Lower it inside jax.default_matmul_precision(self.precision)."""
        mode = _host_mode(self.device)
        key = (kernels, ordered, mode)
        if key not in self._steps:
            md = _build_dense(self.T, jnp.dtype(self.dtype), kernels)
            md = _in_order(md) if ordered else md
            options = _XLA_OPTIONS
            if mode:
                md = _host_memory(md, mode == "io")
                options = dict(options, xla_gpu_enable_host_memory_offloading=True)
            self._steps[key] = jax.jit(md, static_argnames=_MD_STATIC + ("wide",), compiler_options=options)
        return self._steps[key]

    def md_step_chunked(self, kernels="fused", ordered=False):
        key = ("chunked", kernels, ordered)
        if key not in self._steps:
            md = _build_chunked(self.T, jnp.dtype(self.dtype), kernels)
            self._steps[key] = jax.jit(_in_order(md) if ordered else md,
                                       static_argnames=_MD_STATIC + ("row_counts",),
                                       compiler_options=_XLA_OPTIONS)
        return self._steps[key]

    def hessian_step(self, kernels="xla", ordered=False):
        """(linearize, apply), jitted: linearize(positions, *arguments of md_step) -> the pullback of
        the forces at fixed neighbours; apply(pullback, U) = -U dF/dx for cotangents U (m, n_atoms, 3),
        m rows of the Hessian.  Call them inside jax.default_matmul_precision(self.precision)."""
        key = ("hessian", kernels, ordered)
        if key not in self._steps:
            # from the raw step: md_step's jit cannot be nested (compiler options are top-level only)
            md = _build_dense(self.T, jnp.dtype(self.dtype), kernels)
            md = _in_order(md) if ordered else md

            def linearize(positions, *args, **static):
                forces = lambda p: md(p, *args, **static)[12:12 + p.size].reshape(p.shape)
                return jax.vjp(forces, positions)[1]

            apply = jax.vmap(lambda pull, u: -pull(u)[0], in_axes=(None, 0))
            self._steps[key] = (jax.jit(linearize, static_argnames=_MD_STATIC + ("wide",),
                                        compiler_options=_XLA_OPTIONS),
                                jax.jit(apply, compiler_options=_XLA_OPTIONS))
        return self._steps[key]


def load(path, dtype="tf32", charge=None, spin=None, device=None):
    """A JaxModel of a checkpoint; charge/spin models are folded for (charge, spin)."""
    m, _ = load_torch(path, "cpu")
    return from_torch(m, dtype, charge, spin, device)


def from_torch(m, dtype="tf32", charge=None, spin=None, device=None):
    m = fold_charge_spin(m, charge, spin)
    with torch.no_grad():
        out = JaxModel(tables(m), dtype, device)
    out.charge_spin_state = getattr(m, "folded_charge_spin", None)
    return out


class _Unchunk(Exception):
    """Fused kernels failed on a structure in chunks: rebuild it as one XLA call."""


class JaxCalculator(VerletCalculator):
    """ASE calculator with a Verlet list and XLA or fused kernels, run on the model's device.

    kernels: "auto" (fused when cuEquivariance runs here, else "xla"), "xla", "fused" or
    "fused-reference"; a failing fused kernel falls back to XLA with a warning.
    chunk_atoms: "auto" (chunks above 1e5 atoms when one call would not fit), atoms per chunk, or
    None (no chunks).
    background_compile: compile the next table width in a thread (None: on a GPU only, where the
    compiler does not compete with the MD for the CPU).
    spatial_order: the steps see the atoms sorted along a space-filling curve, rebuilt with the
    list; results are in the order of the Atoms.  "auto": off after three changes of structure
    (each structure compiles steps of its own).
    neighbour_list: "host" (vesin, rebuilt in a background thread) or "device" (a compiled cell
    list on the model's device, rebuilt once an atom has moved 3/8 of the skin).  The host builds a
    list for each new structure, when an atom jumps beyond half the skin, when the device list
    outgrows its sizes or runs out of memory, and for chunks.
    """

    _AUTO_CHUNK_MIN = 100_000
    _step_reports_displacement = True
    engine = "jax"

    def __init__(self, model, compute_stress=True, skin=1.5, kernels="auto",
                 background_compile=None, chunk_atoms="auto", spatial_order="auto", neighbour_list="host", **kw):
        if isinstance(model, (str, Path)):
            model = load(model)
        if kernels not in ("auto", "xla", "fused", "fused-reference"):
            raise ValueError(f"kernels must be auto|xla|fused|fused-reference, got {kernels!r}")
        if not (chunk_atoms in (None, "auto") or int(chunk_atoms) >= 0):
            raise ValueError(f"chunk_atoms must be 'auto', None or a count, got {chunk_atoms!r}")
        if spatial_order not in (True, False, "auto"):
            raise ValueError(f"spatial_order must be True, False or 'auto', got {spatial_order!r}")
        if neighbour_list not in ("host", "device"):
            raise ValueError(f"neighbour_list must be 'host' or 'device', got {neighbour_list!r}")
        on_gpu = model.device.platform == "gpu"
        fused = on_gpu and fused_available()
        if kernels == "auto":
            kernels = "fused" if fused else "xla"
        if chunk_atoms not in (None, "auto") and int(chunk_atoms) and not kernels.startswith("fused"):
            warnings.warn(f"chunk_atoms={chunk_atoms} is ignored: only the fused kernels run in chunks",
                          RuntimeWarning)
        if background_compile is None:
            background_compile = on_gpu
        super().__init__(model.species, model.rcut, model.dtype, compute_stress=compute_stress and model.stress,
                         skin=skin, background_compile=background_compile, **kw)
        self.pair_rcut = model.pair_rcut
        self.charge_spin_state = model.charge_spin_state
        self._e0 = model.e0
        self.model, self.kernels, self.chunk_atoms = model, kernels, chunk_atoms
        self.model_stress = model.stress
        self.device = model.device
        self.spatial_order = spatial_order
        self._pre = {}               # one-call executables compiled while sizing chunks
        self._split = None           # (structure, species of the wider rows, extra width)
        self._launched = False       # a fused step has run on this device
        self.neighbour_list, self.n_device_builds = neighbour_list, 0
        self._pairs = {}             # plan key -> jitted device list builder

    @property
    def n_chunks(self):
        return (self._nl or {}).get("n_chunks", 0)

    def _fit(self, need):
        """The table width for `need` neighbours: with the fused kernels a multiple of the bonds per element."""
        g = (_EDGE_GROUP or 4) if self.kernels.startswith("fused") else 1
        return -(-max(need, 2) // g) * g if g > 2 else VerletCalculator._fit(need)

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        while True:
            try:
                return super().calculate(atoms, properties, system_changes)
            except _Unchunk:
                self._nl, self._next = None, None
            except Exception as exc:
                if not self._smaller_chunks(exc, atoms):
                    raise

    def _smaller_chunks(self, exc, atoms):
        """Out of memory with chunk_atoms="auto": halve the chunks and try again."""
        if self.chunk_atoms != "auto" or not self.kernels.startswith("fused") or not _out_of_memory(exc):
            return False
        st = self._nl
        n = st["n"] if st is not None else len(atoms if atoms is not None else self.atoms)
        chunk = ((st.get("chunk_atoms") or n) if st is not None else n) // 2 // 1024 * 1024
        if chunk < 4096:
            return False
        warnings.warn(f"out of GPU memory: {n} atoms in chunks of {chunk}", RuntimeWarning)
        self._chunk_retry = (n, chunk)
        self._nl, self._next = None, None
        return True

    def _put(self, arrays):
        return tuple(jax.device_put(x, self.device) for x in arrays)

    def _state_extra(self, mu, n, order=None):
        """The species arguments of the step (_pair_tables) and the order ones (_order_args), and
        their part of the key."""
        present = np.unique(mu[:n])
        width = 1 << int(np.ceil(np.log2(len(present))))
        padded = np.full(width, len(self.model.species), np.int32)
        padded[:len(present)] = present
        ls = np.zeros(len(mu), np.int32)
        ls[:n] = np.searchsorted(present, mu[:n])
        return (ls, self.model.pair_tables(padded)) + _order_args(order, len(mu)), (width, order is not None)

    def _make_state(self, pos, atoms, nl):
        chunk = self._chunk_size(atoms, nl)
        if not chunk:
            nl = self._row_classes(atoms, nl)
            st, need = super()._make_state(pos, atoms, nl)
            if "wide" in nl:
                st["key_extra"] += (nl["wide"],)
            return st, need
        dt, n, n_pad = self.dtype, nl["n"], nl["n_pad"]
        old = self._nl
        caps = (old.get("chunk_caps") if old is not None and old["n"] == n
                and old.get("chunk_atoms") == chunk else None)
        cap = self._rows_cap(self._fit(nl["need"]), n_pad)
        listed = pos if nl["order"] is None else pos[nl["order"]]
        while True:
            arrays, row_counts, caps = _chunk_plan(listed, nl["cell"], nl["pbc"], nl["i"], nl["j"],
                                                   nl["S"], n, chunk, self.model.n_layers, caps)
            if row_counts[0] <= cap or chunk <= 1024:
                break
            chunk = max(1024, int(0.95 * chunk * cap / row_counts[0]))      # too many rows with the halo
            caps = None
        rows, cr, cj, cjl, cs, cok, cmask = arrays
        mu = np.zeros(n_pad, np.int32)
        mu[:n] = nl["mu"]
        ref = np.zeros((n_pad, 3), dt)
        ref[:n] = listed
        extra, key = self._state_extra(mu, n)
        species, sp_key = extra[:2], key[:1]
        args = self._put((nl["cell"].astype(dt), rows, cr, cj, cjl, cs, cok, cmask.astype(dt), mu,
                          np.array([nl["vol"]], dt), ref) + species + _order_args(nl["order"], n_pad))
        st = dict(n=n, n_pad=n_pad, numbers=atoms.numbers.copy(), pbc=nl["pbc"], cell=nl["cell"],
                  h0=nl["cell"], vol_arg=9, ref=pos, stress=nl["stress"], args=args, pos=np.zeros((n_pad, 3), dt),
                  cand_cap=("chunks", rows.shape[0], row_counts, cr.shape[1], cs.dtype.str),
                  chunk_atoms=chunk, chunk_caps=caps, n_chunks=rows.shape[0],
                  key_extra=(nl["order"] is not None,) + sp_key)
        return st, nl["need"]

    def _row_classes(self, atoms, nl):
        """nl with the atoms of the species that have the most neighbours first, their table
        rows wider than the rest (nl["wide"]: how many, by how much), when that saves a fifth
        of the table; kept for the structure's later lists.  Fused kernels in one call."""
        if not (self.kernels.startswith("fused") and nl["order"] is not None):
            return nl
        n, n_pad, mu, counts = nl["n"], nl["n_pad"], nl["mu"], nl["counts"]
        key = (atoms.numbers.tobytes(), atoms.pbc.tobytes())
        split = self._split if self._split is not None and self._split[0] == key else None
        if split is None:
            present = np.unique(mu)
            need = np.array([counts[mu == s].max() for s in present])
            rank = np.argsort(need)
            best = (n_pad * self._fit(int(need.max())), None, 0)
            for k in range(1, len(present)):
                wide = present[rank[k:]]
                n_w = int(np.isin(mu, wide).sum())
                w_n, w_w = self._fit(int(need[rank[:k]].max())), self._fit(int(need[rank[k:]].max()))
                slots = n_w * w_w + (n_pad - n_w) * w_n
                if slots < best[0]:
                    best = (slots, wide, w_w - w_n)
            total = n_pad * self._fit(int(need.max()))
            split = self._split = (key, best[1] if best[0] < 0.8 * total else None, best[2])
        wide, extra = split[1], split[2]
        if wide is None:
            return nl
        is_w = np.isin(mu, wide)
        perm = np.concatenate([np.flatnonzero(is_w), np.flatnonzero(~is_w)])
        inv = np.empty(n, np.int64)
        inv[perm] = np.arange(n)
        deg = np.bincount(nl["i"], minlength=n)
        off = np.concatenate([[0], np.cumsum(deg)])
        idx = _ranges(off[perm], deg[perm])
        n_w, counts = int(is_w.sum()), counts[perm]
        order = perm if nl["order"] is None else nl["order"][perm]
        return dict(nl, i=np.repeat(np.arange(n, dtype=np.int32), deg[perm]), j=inv[nl["j"][idx]].astype(np.int32),
                    S=nl["S"][idx], mu=mu[perm], counts=counts, order=order, wide=(n_w, extra),
                    need=int(max(counts[:n_w].max() - extra, counts[n_w:].max(initial=0))))

    def _install(self, st, need):
        super()._install(st, need)
        if self.neighbour_list == "device" and "plan" not in st:
            st["plan"] = self._device_plan(st)

    def _moved(self, d2, atoms):
        st = self._nl
        if self.neighbour_list != "device" or st.get("plan") is None or d2 > (0.5 * st["skin"]) ** 2:
            return super()._moved(d2, atoms)
        if d2 > (0.375 * st["skin"]) ** 2:
            self._device_rebuild(atoms)
        return False

    def _device_plan(self, st):
        """The plan of the device-built lists of this structure; None (host lists) for chunks and
        singular periodic cells."""
        if not isinstance(st["cand_cap"], int):
            return None
        wide = st["key_extra"][2] if len(st["key_extra"]) > 2 else (0, 0)
        key = _plan_key(st["ref"], st["cell"], st["pbc"], st["n"], st["n_pad"], st["cand_cap"], st["rcut"],
                        st["skin"], st["ordered"], wide)
        if key is None:
            return None
        cls = np.full(len(self.model.species), (1 << 30) if wide[0] else 0, np.int32)
        if wide[0]:
            cls[self._split[1]] = 0
        args = st["args"]
        own = (lambda x: x[args[12]]) if st["ordered"] else (lambda x: x)      # list -> Atoms order
        return dict(key=key, mu=own(args[5]), ls=own(args[9]), cls=self._put((cls,))[0])

    def _device_rebuild(self, atoms):
        """Install a list built on the device at these positions.  The host builds it (and new
        sizes) when the device list would be incomplete, and every list once the device failed."""
        st, plan = self._nl, self._nl["plan"]
        _, cap, _, _, n_bins, occupancy, ordered, _ = plan["key"]
        if plan["key"] not in self._pairs:
            self._pairs[plan["key"]] = jax.jit(_build_pairs(self.model.T, jnp.dtype(self.dtype), *plan["key"]))
        h = atoms.cell.array
        st["pos"][:st["n"]] = atoms.positions
        try:
            pos, cell = self._put((st["pos"], h.astype(self.dtype)))
            out, stats = self._pairs[plan["key"]](pos, cell, st["args"][6], plan["mu"], plan["ls"], plan["cls"],
                                                  np.asarray(st["rcut"], self.dtype), np.asarray(self.skin, self.dtype))
            pairs, most, bins, images, need = (int(x) for x in np.asarray(stats))
        except Exception as exc:
            warnings.warn(f"the device neighbour list failed ({type(exc).__name__}: {str(exc)[:200]}); "
                          f"host lists from now on", RuntimeWarning)
            self.neighbour_list, pairs = "host", None
        if pairs is None or pairs > cap or most > occupancy or bins > n_bins or not images:
            self._install(*self._prepare(atoms.positions.copy(), atoms))
            return
        args = list(st["args"])
        args[1:4], args[4], args[8] = out[:3], out[-1], out[4]
        if ordered:
            args[5], args[9], args[11], args[12] = out[5:9]
        new = dict(st, args=tuple(args), ref=atoms.positions.copy(), h0=h.copy(), skin=self.skin)
        if not np.array_equal(st["cell"], h):
            self._set_cell(new, h)
        self.n_device_builds += 1
        self._install(new, need)

    def _rows_cap(self, width, n_pad=None):
        """Most table rows one call may hold: no array of the fused step beyond 1.25 x 2^31 elements
        (the kernels' index arithmetic is not documented as 64-bit; the largest size was checked
        against chunked calls).  Element counts per row and edge come from the step's jaxpr."""
        if n_pad is not None and n_pad < 10_000:
            return 1 << 62
        widths = self.model.__dict__.setdefault("_fused_widths", {})
        kernels = self.kernels
        if kernels not in widths:
            dt = jnp.dtype(self.dtype)
            energy = _energies(self.model.T, dt, kernels)[0]
            a, K = 1009, 13                               # sizes no feature axis has
            species = (jnp.zeros(a, jnp.int32), self.model.pair_tables(np.zeros(1, np.int32)))
            jaxpr = jax.make_jaxpr(jax.grad(lambda *x: energy(*x)[0]))(
                jnp.zeros((a, 3), dt), jnp.zeros((a, K, 3), dt), jnp.zeros((a, K), jnp.int32),
                jnp.zeros(a, jnp.int32), jnp.ones(a, dt), *species)
            per = {"row": 1, "edge": 1}

            def walk(jp):
                for eqn in jp.eqns:
                    for v in eqn.outvars:
                        sh = tuple(getattr(v.aval, "shape", ()))
                        if sh[:2] == (a, K) or sh[:1] == (a * K,):
                            rest = sh[2:] if sh[:2] == (a, K) else sh[1:]
                            per["edge"] = max(per["edge"], int(np.prod(rest)))
                        elif sh[:1] == (a,):
                            per["row"] = max(per["row"], int(np.prod(sh[1:])))
                    for p in eqn.params.values():
                        for q in (p if isinstance(p, (tuple, list)) else (p,)):
                            q = getattr(q, "jaxpr", q)
                            if hasattr(q, "eqns"):
                                walk(q)
            walk(jaxpr.jaxpr)
            widths[kernels] = (per["row"], per["edge"])
        per_row, per_edge = widths[kernels]
        return int(1.25 * (2 ** 31 - 1) / max(per_row, width * per_edge))

    def _chunk_size(self, atoms, nl):
        """Atoms per chunk for this structure, 0 for one call."""
        if not self.chunk_atoms or not self.kernels.startswith("fused"):
            return 0
        old = self._nl
        n, n_pad = nl["n"], nl["n_pad"]
        cap = self._rows_cap(self._fit(nl["need"]), n_pad)
        limit = max(1024, int(cap / 1.25)) if n_pad > cap else 0
        if old is not None and self._same_structure(old, atoms):
            chunk = old.get("chunk_atoms", 0)
            if limit:
                chunk = min(chunk or limit, limit)
            return chunk if 0 < chunk < n else 0
        if self.chunk_atoms != "auto":
            chunk = int(self.chunk_atoms) if n > int(self.chunk_atoms) else 0
        else:
            retry = getattr(self, "_chunk_retry", None)
            chunk = (retry[1] if retry is not None and retry[0] == n
                     else self._memory_chunk(nl, rows=limit or None))
        if limit:
            chunk = min(chunk or limit, limit)
        return chunk if 0 < chunk < n else 0

    def _memory_chunk(self, nl, rows=None):
        """0 when XLA's estimate of the one-call step fits in 80 % of the free GPU memory,
        else chunks of ~45 % of it; `rows`: estimate from a one-call step of that size."""
        if nl["n_pad"] < self._AUTO_CHUNK_MIN:
            return 0
        stats = self.device.memory_stats() or {}
        if "bytes_limit" not in stats:
            return 0
        free = stats["bytes_limit"] - stats.get("bytes_in_use", 0)
        n = rows or nl["n"]
        extra = self._state_extra(nl["mu"], nl["n"], nl["order"])[1]
        if rows:
            key = (int(np.ceil(n / 8) * 8), padded_size(1.05 * len(nl["i"]) * n / nl["n"]),
                   self._stressed(nl), self._fit(nl["need"])) + extra
        else:
            key = (nl["n_pad"], self._cand_cap(nl), self._stressed(nl), self._fit(nl["need"])) + extra
        avals, static = self._signature(key)
        with jax.default_matmul_precision(self.model.precision):
            exe = self.model.md_step(self.kernels, bool(extra[1])).lower(*avals, **static).compile()
        try:
            mem = exe.memory_analysis()
            need = mem.temp_size_in_bytes + mem.argument_size_in_bytes + mem.output_size_in_bytes
        except Exception:
            need = 0
        if not rows and need <= 0.8 * free:
            self._pre[key] = exe
            return 0
        if need <= 0:
            return 0
        return max(4096, int(n * 0.45 * free / need) // 1024 * 1024)

    def _aval(self, shape, dtype, memory_kind=None):
        return jax.ShapeDtypeStruct(
            shape, dtype, sharding=jax.sharding.SingleDeviceSharding(self.device, memory_kind=memory_kind))

    def _signature(self, key):
        n_pad, cap, stress, width, n_species, ordered, *wide = key
        dt, S = self.model.dtype, self._aval
        avals = (S((n_pad, 3), dt, "pinned_host" if _host_mode(self.device) == "io" else None),
                 S((3, 3), dt), S((cap,), np.int32), S((cap,), np.int32),
                 S((cap, 3), dt), S((n_pad + 1,), np.int32), S((n_pad,), np.int32), S((n_pad,), dt),
                 S((1,), dt), S((n_pad, 3), dt), S((n_pad,), np.int32), self._present_aval(n_species))
        avals += (S((n_pad,), np.int32),) * (2 * ordered)
        return avals, dict(n_atoms=n_pad, width=width, want_stress=stress, wide=wide[0] if wide else (0, 0))

    def _present_aval(self, n):
        """The aval of a step's `present`, n species."""
        at = self._aval((n,), np.int32)
        return jax.tree.map(lambda x: self._aval(x.shape, x.dtype), jax.eval_shape(self.model.pair_build, at))

    def _zeros(self, avals):
        return [jax.tree.map(lambda a: jax.device_put(np.zeros(a.shape, a.dtype), a.sharding), x) for x in avals]

    def _build(self, key):
        """Ahead-of-time compilation of one shape (also from a background thread)."""
        chunked = isinstance(key[1], tuple)
        if not chunked:
            avals, static = self._signature(key)

        def compile_():
            if chunked:
                return self._compile_chunked(key)
            with jax.default_matmul_precision(self.model.precision):
                return self.model.md_step(self.kernels, bool(key[5])).lower(*avals, **static).compile()
        try:
            exe = None if chunked else self._pre.pop(key, None)
            if exe is None:
                exe = compile_()
            if not chunked and self.kernels.startswith("fused") and (
                    not self._launched or key[0] < self._AUTO_CHUNK_MIN):
                # a kernel can compile and still fail to launch: a dry run on zeros, once, and for every
                # width of a small system (background compiles too)
                jax.block_until_ready(exe(*self._zeros(avals)))
                self._launched = True
        except Exception as exc:
            # out of memory is handled by _smaller_chunks; other kernels would not help, nor would
            # they help when an explicitly requested host memory of a step is what failed
            if _out_of_memory(exc) or _HOST_MEMORY in ("out", "io"):
                raise
            if not chunked and _host_mode(self.device):
                warnings.warn(f"step with its result in host memory failed ({type(exc).__name__}: "
                              f"{str(exc)[:200]}); result in device memory", RuntimeWarning)
                _HOST_DEFAULT_FAILED.add(self.device)
                return self._build(key)
            if not self.kernels.startswith("fused"):
                raise
            warnings.warn(f"fused kernels failed ({type(exc).__name__}: {str(exc)[:200]}); "
                          f"using XLA", RuntimeWarning)
            self.kernels = "xla"
            if chunked:
                raise _Unchunk from exc        # only the fused kernels run in chunks
            exe = compile_()
        # compiled programs live long: keep full collections from walking them
        gc.freeze()

        n_pad = key[0]
        mode = _host_mode(self.device)
        host_out, host_in = bool(mode) and not chunked, mode == "io" and not chunked

        def start(pos, *args):
            if host_in and isinstance(pos, jax.Array):
                pos = np.asarray(pos)
            out = exe(pos, *args)                         # returns before the step is done
            if not host_out:
                out.copy_to_host_async()                  # the copy starts once the step is done
            return out

        def finish(out):
            while _SPIN and not out.is_ready():
                time.sleep(0)                             # lets other Python threads run
            return _unpack(np.asarray(out), n_pad)        # waits for the step

        def run(pos, *args):
            return finish(exe(pos, *args))
        run.start, run.finish = start, finish
        return run

    def _compile_chunked(self, key):
        n_pad, (_, n_chunks, row_counts, n_cand, sdt), stress, width, ordered, *species = key
        dt, S = self.model.dtype, self._aval
        K, R, P, c = n_chunks, row_counts[0], n_cand, row_counts[-1]
        avals = (S((n_pad, 3), dt), S((3, 3), dt), S((K, R), np.int32), S((K, P), np.int32),
                 S((K, P), np.int32), S((K, P), np.int32), S((K, P, 3), np.dtype(sdt)),
                 S((K, P), np.bool_), S((K, c), dt), S((n_pad,), np.int32), S((1,), dt), S((n_pad, 3), dt))
        avals += tuple(x for n in species for x in (S((n_pad,), np.int32), self._present_aval(n)))
        avals += (S((n_pad,), np.int32),) * (2 * ordered)
        with jax.default_matmul_precision(self.model.precision):
            return self.model.md_step_chunked(self.kernels, ordered).lower(
                *avals, n_atoms=n_pad, width=width, want_stress=stress, row_counts=row_counts).compile()

    def get_hessian(self, atoms=None, method="dense", k=6, chunk=16, tol=None):
        """d2E/dx2 in eV/A^2 at fixed cell and neighbour list, with the arguments and results
        of TorchCalculator.get_hessian; atoms: the last ones calculated by default.  Fused
        kernels whose second derivatives fail fall back to XLA with a warning.  Not for
        systems in chunks."""
        if method not in ("dense", "lanczos"):
            raise ValueError(f"get_hessian: method must be 'dense' or 'lanczos', not {method!r}")
        atoms = self.atoms if atoms is None else atoms
        if atoms is None:
            raise ValueError("get_hessian: no atoms given and none calculated yet")
        self.get_potential_energy(atoms)                 # installs a list valid for atoms
        if self.n_chunks:
            raise NotImplementedError("get_hessian: this system is evaluated in chunks; the Hessian needs "
                                      "one call (chunk_atoms=None)")
        n_pad, n3 = self._nl["n_pad"], 3 * self._nl["n"]
        m = chunk if method == "dense" else 1
        kernels = getattr(self, "_hessian_kernels", self.kernels)
        try:
            rows = self._hessian_rows(kernels, m)
        except Exception as exc:
            if not kernels.startswith("fused") or _out_of_memory(exc):
                raise
            warnings.warn(f"second derivatives of the {kernels} kernels failed ({type(exc).__name__}: "
                          f"{str(exc)[:200]}); the Hessian with XLA", RuntimeWarning)
            self._hessian_kernels = "xla"
            rows = self._hessian_rows("xla", m)

        def unit(i):
            """Cotangents of the coordinates i .. i + m - 1, those of padded atoms and beyond 3N zero."""
            U = np.zeros((m, 3 * n_pad), self.dtype)
            c = np.arange(i, min(i + m, n3))
            U[c - i, c] = 1
            return U.reshape(m, n_pad, 3)

        if method == "lanczos":
            from scipy.sparse.linalg import LinearOperator, eigsh

            def hvp(v):
                u = np.zeros(3 * n_pad, self.dtype)
                u[:n3] = np.ravel(v)
                return np.asarray(rows(u.reshape(1, n_pad, 3)), np.float64).reshape(-1)[:n3]

            if tol is None:
                tol = 0 if self.dtype == np.float64 else 1e-5
            return eigsh(LinearOperator((n3, n3), matvec=hvp, dtype=np.float64), k=k, which="SA", tol=tol)
        try:
            blocks = [rows(unit(i)) for i in range(0, n3, m)]
            H = np.concatenate([np.asarray(b, np.float64).reshape(m, -1) for b in blocks])[:n3, :n3]
            return 0.5 * (H + H.T)
        except Exception as exc:
            if m == 1 or not _out_of_memory(exc):
                raise
        # outside the except clause, so that the failed blocks are freed first
        warnings.warn(f"get_hessian: out of device memory with chunk={m}; trying chunk={m // 2}", RuntimeWarning)
        return self.get_hessian(atoms, method, k, m // 2, tol)

    def _hessian_rows(self, kernels, m):
        """rows(U) = m rows of the Hessian at the installed list and positions, for cotangents U
        (m, n_pad, 3); fused kernels compiled and run once here, so that those that fail do so now."""
        st = self._nl
        linearize, apply = self.model.hessian_step(kernels, st["ordered"])
        static = dict(self._signature(self._key(self._width))[1], want_stress=False)
        with jax.default_matmul_precision(self.model.precision):
            pull = linearize(jax.device_put(st["pos"], self.device), *st["args"], **static)
            if kernels != "xla":
                jax.block_until_ready(apply(pull, np.zeros((m,) + st["pos"].shape, self.dtype)))

        def rows(U):
            with jax.default_matmul_precision(self.model.precision):
                return apply(pull, U)
        return rows
