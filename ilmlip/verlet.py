"""An ASE calculator that keeps a Verlet neighbour list on the accelerator.

The list holds the pairs within rcut + skin, built on the host by vesin; a step ships only the
positions and picks the pairs inside rcut itself. The list stays valid while every atom has moved
less than skin/2 since the build, so no pair within rcut can be missing. From skin/4 on, the next
list is built in a background thread; lists that expire within 20 steps widen the skin of the
next ones, up to twice. A changed cell keeps the list while the bound of _moved2 holds.

A backend subclass supplies _put(arrays) and _build(key) -> run(pos, *args), where run returns
(head = [E, stress], (most neighbours, edges), forces, energies, d2): site energies without the
E0s (the host adds them) and d2, the largest squared displacement since the build. Optional
run.start and run.finish split the step into dispatch and wait.
With spatial_order the list is for the atoms in morton_order (st["order"]); positions and forces
stay in the order of the Atoms.
"""
import threading
from contextlib import suppress

import numpy as np
from ase.calculators.calculator import Calculator, PropertyNotImplementedError, all_changes
from . import vcut

_VESIN = {}
_VESIN_LOCK = threading.Lock()     # one vesin NeighborList is not safe in two threads


_SLAB_PAIRS = 1 << 30     # vesin returns wrong atom indices beyond 2**31 pairs: bigger lists in slabs


def _vesin(positions, cell, pbc, cutoff):
    from vesin import NeighborList
    key = round(float(cutoff), 6)
    nl = _VESIN.get(key)
    if nl is None:
        nl = _VESIN[key] = NeighborList(cutoff=float(cutoff), full_list=True, sorted=True)
    with _VESIN_LOCK:
        i, j, S, d = nl.compute(points=positions, box=cell, periodic=[bool(p) for p in pbc],
                                quantities="ijSd")
    return (np.asarray(i, np.int32), np.asarray(j, np.int32), np.asarray(S, np.int32),
            np.asarray(d, float))


def _slab_plan(positions, cell, pbc, cutoff):
    """(axis, slabs) for a list of more than _SLAB_PAIRS pairs, else None."""
    vol = abs(float(np.linalg.det(cell)))
    if vol < 1e-9 or not pbc.any():
        return None
    n = len(positions)
    pairs = n * n / vol * 4.0 / 3.0 * np.pi * cutoff ** 3
    if pairs <= _SLAB_PAIRS:
        return None
    width = vol / np.linalg.norm(np.cross(cell[[1, 2, 0]], cell[[2, 0, 1]]), axis=1)
    axis = int(np.argmax(np.where(pbc, width, 0.0)))
    slabs = min(int(np.ceil(1.3 * pairs / _SLAB_PAIRS)), int(width[axis] / (2.0 * cutoff)))
    return (axis, slabs) if slabs > 1 else None


def _slab_neighbours(positions, cell, pbc, cutoff, axis, slabs):
    """host_neighbours from one vesin list per slab along a periodic cell vector (its atoms
    first, then the images within cutoff of it, that direction not periodic)."""
    f = positions @ np.linalg.inv(cell)[:, axis]
    n0 = np.floor(f)
    t = f - n0
    up = t >= 1.0
    t[up] -= 1.0
    n0[up] += 1.0
    h = cutoff * np.linalg.norm(np.linalg.inv(cell)[:, axis])
    per = pbc.copy()
    per[axis] = False
    parts = []
    for k in range(slabs):
        lo, hi = k / slabs, (k + 1) / slabs
        own = np.nonzero((t >= lo) & (t < hi))[0]
        halo = [own]
        shift = [np.zeros(len(own), np.int64)]
        for s in (-1, 0, 1):
            u = t + s
            sel = np.nonzero((u >= lo - h) & (u < hi + h) & ~((s == 0) & (u >= lo) & (u < hi)))[0]
            halo.append(sel)
            shift.append(np.full(len(sel), s, np.int64))
        idx, img = np.concatenate(halo), np.concatenate(shift) - n0[np.concatenate(halo)].astype(np.int64)
        p, q, S, d = _vesin(positions[idx] + img[:, None] * cell[axis], cell, per, cutoff)
        keep = p < len(own)
        p, q, S, d = p[keep], q[keep], S[keep], d[keep]
        S[:, axis] += (img[q] - img[p]).astype(np.int32)
        parts.append((idx[p].astype(np.int32), idx[q].astype(np.int32), S, d))
    # each atom's pairs come from its own slab, sorted by it there
    counts = np.zeros(len(positions), np.int64)
    for i, *_ in parts:
        counts += np.bincount(i, minlength=len(positions))
    off = np.concatenate([[0], np.cumsum(counts)])
    total = int(off[-1])
    out = (np.empty(total, np.int32), np.empty(total, np.int32), np.empty((total, 3), np.int32),
           np.empty(total, float))
    while parts:
        part = parts.pop(0)
        i = part[0]
        first = np.concatenate([[0], np.nonzero(np.diff(i))[0] + 1])
        run = np.repeat(first, np.diff(np.concatenate([first, [len(i)]])))
        at = off[i] + np.arange(len(i)) - run
        for dst, src in zip(out, part):
            dst[at] = src
    return out


def host_neighbours(positions, cell, pbc, cutoff):
    """(i, j, S, d) of all pairs within cutoff, sorted by i; bond = pos[j] - pos[i] + S @ cell."""
    positions = np.asarray(positions, float)
    cell = np.asarray(cell, float).reshape(3, 3)
    pbc = np.asarray(pbc, bool).reshape(3)
    plan = _slab_plan(positions, cell, pbc, cutoff)
    if plan is not None:
        return _slab_neighbours(positions, cell, pbc, cutoff, *plan)
    return _vesin(positions, cell, pbc, cutoff)


def check_finite(positions):
    # a non-finite atom would silently leave the list; the sum catches it, the exact test settles overflow
    with np.errstate(over="ignore", invalid="ignore"):
        total = np.add.reduce(positions, axis=None)
    if not np.isfinite(total) and not np.isfinite(positions).all():
        raise FloatingPointError("non-finite atomic positions")


def check_energy(e):
    if not np.isfinite(e):
        raise FloatingPointError(f"non-finite energy {e}")


def padded_size(n, per_octave=8, granule=128):
    """Smallest size >= n among per_octave sizes per doubling, a multiple of granule: few
    compiled shapes."""
    n = max(int(np.ceil(n)), granule)
    p = 1 << (n.bit_length() - 1)
    cap = next(p + p * k // per_octave for k in range(per_octave + 1)
               if p + p * k // per_octave >= n)
    return -(-cap // granule) * granule


def _spread3(q):
    """The 10 bits of q on every third bit."""
    q = q & 0x3FF
    q = (q | (q << 16)) & 0x030000FF
    q = (q | (q << 8)) & 0x0300F00F
    q = (q | (q << 4)) & 0x030C30C3
    return (q | (q << 2)) & 0x09249249


def morton_order(positions, cell, pbc):
    """The atoms along a Z-order curve: wrapped fractional coordinates along periodic
    axes, the bounding box along the others (and without a cell), 1024 steps each."""
    pos = np.asarray(positions, float)
    cell = np.asarray(cell, float).reshape(3, 3)
    pbc = np.asarray(pbc, bool).reshape(3)
    if abs(np.linalg.det(cell)) > 1e-9:
        f = np.linalg.solve(cell.T, pos.T).T
        f[:, pbc] %= 1.0
    else:
        f, pbc = pos.copy(), np.zeros(3, bool)
    lo, hi = f.min(axis=0), f.max(axis=0)
    lo[pbc], hi[pbc] = 0.0, 1.0
    q = np.clip((f - lo) / np.maximum(hi - lo, 1e-12) * 1024, 0, 1023).astype(np.int64)
    code = _spread3(q[:, 0]) | (_spread3(q[:, 1]) << 1) | (_spread3(q[:, 2]) << 2)
    return np.argsort(code, kind="stable").astype(np.int32)


def _same(a, b):
    """np.array_equal without its overhead (called for every comparison of a step)."""
    return a.shape == b.shape and bool((a == b).all())


def _same_small(a, b):
    """_same for a cell or pbc: faster as lists."""
    return a.tolist() == b.tolist()


def _whole(out):
    return out


def _equal(a, b):
    """_same; in MD the first rows already differ."""
    return _same(a[:8], b[:8]) and _same(a, b)


class VerletCalculator(Calculator):
    implemented_properties = ["energy", "free_energy", "forces", "energies", "stress"]
    spatial_order = False
    model_stress = True          # False: the model was not trained on stresses
    _ORDER_SWAPS = 3             # spatial_order="auto": off from this change of structure on

    def __init__(self, species, rcut, dtype, compute_stress=True, skin=1.5,
                 background_compile=True, async_rebuild=True, **kw):
        super().__init__(**kw)
        # check_state compares exactly against self._snap; use_cache would copy the Atoms
        self.use_cache = False
        from ase.data import atomic_numbers
        self.rcut, self.dtype = float(rcut), np.dtype(dtype)
        self.pair_rcut = None
        self.compute_stress, self.skin = bool(compute_stress), float(skin)
        self._skin_max, self._calls = 2 * self.skin, 0
        self.background_compile, self.async_rebuild = bool(background_compile), bool(async_rebuild)
        self._z2mu = np.full(max(atomic_numbers.values()) + 1, -1, np.int32)
        for k, s in enumerate(species):
            self._z2mu[atomic_numbers[s]] = k
        self._e0 = np.zeros(len(species))
        self._swaps = 0          # changes of structure (other atoms, not other positions)
        self._nl = None          # the Verlet state in use
        self._next = None        # (thread, [state]) of a background rebuild
        self._snap = None        # the one Atoms copy ASE compares against
        self._width = 0
        self._exe = {}           # (n_pad, cand_cap, stress, width) -> run
        self._bg = None          # (key, thread) of a background compile
        self._stress_on = False
        self.n_rebuilds = self.n_overflows = self.n_edges = self.max_neighbours = 0

    _step_reports_displacement = False

    def _state_of(self, atoms):
        cs = getattr(self, "charge_spin_state", None)
        if cs is None:
            return None
        return int(atoms.info.get("charge", cs[0])), int(atoms.info.get("spin", cs[1]))

    def check_state(self, atoms, tol=1e-15):
        # exact comparisons of what the model reads
        a = self.atoms
        if a is None:
            return list(all_changes)
        changes = []
        got = self._state_of(atoms)
        if got is not None and got != tuple(self.charge_spin_state):
            changes.append("charge_spin")
        if len(a) != len(atoms) or not _same(a.numbers, atoms.numbers):
            changes.append("numbers")
        elif not _equal(a.positions, atoms.positions):
            changes.append("positions")
        if not _same_small(a.cell.array, atoms.cell.array):
            changes.append("cell")
        if not _same_small(a.pbc, atoms.pbc):
            changes.append("pbc")
        return changes

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        self._calls += 1
        got = None if atoms is None else self._state_of(atoms)
        if got is not None and got != tuple(self.charge_spin_state):
            q, s = self.charge_spin_state
            raise ValueError(f"atoms.info has charge {got[0]}, spin {got[1]}, but this "
                             f"calculator is for charge {q}, spin {s}")
        fresh = atoms is not None
        if not fresh:
            atoms = self.atoms
        # ASE derived the changes from the structure it last gave this calculator
        moved_only = fresh and list(system_changes or ()) == ["positions"]
        if "stress" in properties:
            if not self.model_stress:
                from .model import NO_STRESS
                raise PropertyNotImplementedError(NO_STRESS)
            self._stress_on = True
        # the step measures Cartesian displacements, so a changed cell is measured here
        h = atoms.cell.array
        on_device = self._step_reports_displacement and self.async_rebuild
        self._refresh(atoms, measure=not on_device, same_atoms=moved_only)
        on_device = on_device and _same_small(self._nl["h0"], h)
        while True:
            st = self._nl
            if not _same_small(st["cell"], h):
                self._set_cell(st, h)
            st["pos"][:st["n"]] = atoms.positions
            width = self._width
            run = self._executable(width)
            out = getattr(run, "start", run)(st["pos"], *st["args"])
            if fresh:                  # what the step does not need, while it runs
                check_finite(atoms.positions)
                self._snapshot(atoms)
                fresh = False
            head, (need, n_edges), forces, e_atom, d2 = getattr(run, "finish", _whole)(out)
            if on_device and self._moved(d2, atoms):
                continue
            if need <= width:
                break
            # the table dropped neighbours: redo the step with a wider one
            self.n_overflows += 1
            if need > self._width:
                up = self._fit(self._width + 1)
                self._width = up if need <= up else self._fit(need)
        self.n_edges, self.max_neighbours = n_edges, need
        self._follow(need)
        n = st["n"]
        # in float64 on the host: the E0s dwarf the network's part of the energy
        e_atom = e_atom[:n] + st["e0"]
        e = float(e_atom.sum())
        check_energy(e)
        self.results = dict(energy=e, free_energy=e, forces=np.asarray(forces[:n], float),
                            energies=e_atom)
        if self._stressed(st):
            self.results["stress"] = np.asarray(head[1:7], float)

    def _snapshot(self, atoms):
        """self.atoms: the one copy of atoms that check_state compares against."""
        s = self._snap
        if (s is not None and len(s) == len(atoms) and _same(s.numbers, atoms.numbers)
                and _same_small(s.cell.array, atoms.cell.array) and _same_small(s.pbc, atoms.pbc)):
            s.positions[:] = atoms.positions
        else:
            s = self._snap = atoms.copy()
        self.atoms = s

    @staticmethod
    def _fit(need):
        return max(2, need + (need & 1))

    def _stressed(self, st):
        return st["stress"] and self._stress_on

    def _key(self, width):
        st = self._nl
        return (st["n_pad"], st["cand_cap"], self._stressed(st), int(width)) + st.get("key_extra", ())

    def _executable(self, width):
        key = self._key(width)
        run = self._exe.get(key)
        if run is None:
            if self._bg and self._bg[0] == key and self._bg[1].is_alive():
                self._bg[1].join()
                run = self._exe.get(key)
            if run is None:
                run = self._exe[key] = self._build(key)
        return run

    def _follow(self, need):
        """Compile the next width in the background once the count is within one of the
        current one (it only grows during MD: shrinking costs recompiles)."""
        if not (self.background_compile and need >= self._width - 1):
            return
        key = self._key(self._fit(self._width + 1))
        if key in self._exe or (self._bg is not None and self._bg[1].is_alive()):
            return
        # not a daemon thread: one killed at exit inside the compiler aborts the process
        th = threading.Thread(target=self._build_quietly, args=(key,), daemon=False)
        self._bg = (key, th)
        th.start()

    def _build_quietly(self, key):
        with suppress(Exception):                   # the synchronous path reports real errors
            self._exe[key] = self._build(key)

    @staticmethod
    def _same_structure(st, atoms, same_atoms=False):
        """same_atoms: the numbers and pbc are known to be those of the list (positions moved only)."""
        h = atoms.cell.array
        return (st is not None and (same_atoms or (st["n"] == len(atoms)
                                                    and _same(st["numbers"], atoms.numbers)
                                                    and _same_small(st["pbc"], atoms.pbc)))
                # a changed cell is followed (_moved2) between non-singular cells
                and (_same_small(st["cell"], h) or abs(np.linalg.det(st["h0"]) * np.linalg.det(h)) > 0))

    def _moved2(self, st, atoms):
        """The square of max |(s - s0) h| + (rcut + skin) ||a - I||_2 / 2, a = h0^-1 h,
        (s - s0) h = r - r0 a: the displacement for the skin/2 rule."""
        h0, h = st["h0"], atoms.cell.array
        if _same_small(h0, h):
            dr = atoms.positions - st["ref"]
            return float(np.einsum("ij,ij->i", dr, dr).max())
        a = np.linalg.solve(h0, h)
        dr = atoms.positions - st["ref"] @ a
        d = (np.sqrt(np.einsum("ij,ij->i", dr, dr).max())
             + 0.5 * (st["rcut"] + st["skin"]) * np.linalg.norm(a - np.eye(3), 2))
        return float(d * d) if np.isfinite(d) else np.inf       # nan from a near-singular cell: rebuild

    def _set_cell(self, st, h):
        """Follow a changed cell: the step takes the cell and the volume as arguments."""
        vol = abs(float(np.linalg.det(h)))
        cell, v = self._put((h.astype(self.dtype), np.array([vol], self.dtype)))
        args = list(st["args"])
        args[0], args[st["vol_arg"]] = cell, v
        st.update(cell=h.copy(), vol=vol, args=tuple(args))

    def _refresh(self, atoms, measure=True, same_atoms=False):
        """Make the list valid for atoms, rebuilding as late as possible; measure=False
        leaves the displacement to the step (_moved with what it reports)."""
        if not self._same_structure(self._nl, atoms, same_atoms):
            st = self._nl
            if st is not None and (st["n"] != len(atoms) or not np.array_equal(st["numbers"], atoms.numbers)):
                self._swaps += 1
            self._next = None
            self._install(*self._prepare(atoms.positions.copy(), atoms))
            return
        nxt = self._next
        if nxt is not None and not nxt[0].is_alive():
            self._next = None
            if nxt[1]:
                self._install(*nxt[1][0])
        if measure or not _same_small(self._nl["h0"], atoms.cell.array):
            self._moved(self._moved2(self._nl, atoms), atoms)

    def _moved(self, d2, atoms):
        """Act on d2, the largest squared displacement since the list in use was built:
        True when that list was not valid for atoms and a valid one is installed now."""
        st = self._nl
        half2, quarter2 = (0.5 * st["skin"]) ** 2, (0.25 * st["skin"]) ** 2
        if d2 > half2:
            if self._calls - st["built"] < 20:     # the host cannot keep up: a wider skin
                self.skin = min(1.5 * self.skin, self._skin_max)
            if self._next is not None:              # being built: wait for it
                nxt, self._next = self._next, None
                nxt[0].join()
                if nxt[1]:
                    self._install(*nxt[1][0])
                    d2 = self._moved2(self._nl, atoms)
                    half2 = (0.5 * self._nl["skin"]) ** 2
            if d2 > half2:
                self._install(*self._prepare(atoms.positions.copy(), atoms))
            return True
        if self.async_rebuild and d2 > quarter2 and self._next is None:
            out = []
            th = threading.Thread(target=self._prepare_into,
                                  args=(atoms.positions.copy(), atoms.copy(), out), daemon=False)
            self._next = (th, out)
            th.start()
        return False

    def _prepare_into(self, pos, atoms, out):
        with suppress(Exception):
            out.append(self._prepare(pos, atoms))

    def _prepare(self, pos, atoms):
        check_finite(pos)
        skin = self.skin
        nl = self._host_list(pos, atoms, skin)
        st, need = self._make_state(pos, atoms, nl)
        st["skin"], st["rcut"] = skin, nl["rcut"]
        st["e0"] = self._e0[self._z2mu[atoms.numbers]]
        st["ordered"] = nl["order"] is not None
        return st, need

    def _ordered(self):
        """Whether the next list is in morton_order: spatial_order, or "auto" until _ORDER_SWAPS
        changes of structure (each new structure compiles steps of its own)."""
        if self.spatial_order == "auto":
            return self._swaps < self._ORDER_SWAPS
        return bool(self.spatial_order)

    def _host_list(self, pos, atoms, skin):
        n = len(atoms)
        mu = self._z2mu[atoms.numbers]
        if (mu < 0).any():
            missing = sorted(set(np.asarray(atoms.get_chemical_symbols())[mu < 0]))
            raise ValueError(f"elements not in this model: {missing}")
        # padded atoms have no neighbours and are masked out of the energy
        n_pad = int(np.ceil(n / 8) * 8) or 8
        cell = atoms.cell.array.copy()
        pbc = atoms.pbc & cell.any(1)      # periodic along the cell vectors there are
        order = morton_order(pos, cell, pbc) if self._ordered() else None
        if order is not None:
            pos, mu = pos[order], mu[order]
        # a local: a background rebuild runs this while the main thread uses self.rcut
        rcut = vcut.list_cutoff(self.pair_rcut, mu, self.rcut)
        i, j, S, d = host_neighbours(pos, cell, pbc, rcut + skin)
        i, j, S, d = vcut.verlet_keep(self.pair_rcut, mu, skin, i, j, S, d)
        inside = d < rcut
        inside = vcut.verlet_inside(self.pair_rcut, mu, i, j, d, inside)
        counts = np.bincount(i[inside], minlength=n)
        vol = abs(float(np.linalg.det(cell)))
        return dict(n=n, n_pad=n_pad, mu=mu, cell=cell, pbc=pbc, i=i, j=j, S=S, order=order, rcut=rcut,
                    need=int(counts.max(initial=0)), counts=counts, vol=vol,
                    stress=self.compute_stress and vol > 0 and bool(pbc.any()))

    def _cand_cap(self, nl, old=None):
        """Candidate buffer size; that of old (default: the list in use) while it still fits."""
        old, cap = old or self._nl, padded_size(1.05 * len(nl["i"]))
        if (old and old["n_pad"] == nl["n_pad"] and isinstance(old["cand_cap"], int)
                and len(nl["i"]) <= old["cand_cap"] <= 1.3 * cap):
            cap = old["cand_cap"]
        return cap

    def _make_state(self, pos, atoms, nl):
        dt, n, n_pad = self.dtype, nl["n"], nl["n_pad"]
        cap = self._cand_cap(nl)
        # spare slots point at the last atom, which keeps cand_i sorted
        ci = np.full(cap, n_pad - 1, np.int32)
        ci[:len(nl["i"])] = nl["i"]
        cj = np.full(cap, n_pad - 1, np.int32)
        cj[:len(nl["j"])] = nl["j"]
        cs = np.zeros((cap, 3), dt)
        cs[:len(nl["S"])] = nl["S"]
        row_start = np.zeros(n_pad + 1, np.int32)
        row_start[1:] = np.cumsum(np.bincount(nl["i"], minlength=n_pad))
        mu = np.zeros(n_pad, np.int32)
        mu[:n] = nl["mu"]
        amask = np.zeros(n_pad, dt)
        amask[:n] = 1.0
        ref = np.zeros((n_pad, 3), dt)
        ref[:n] = pos if nl["order"] is None else pos[nl["order"]]
        extra, key = self._state_extra(mu, n, nl["order"])
        args = self._put((nl["cell"].astype(dt), ci, cj, cs, row_start, mu, amask,
                          np.array([nl["vol"]], dt), ref) + extra)
        st = dict(n=n, n_pad=n_pad, numbers=atoms.numbers.copy(), pbc=nl["pbc"],
                  cell=nl["cell"], h0=nl["cell"], vol_arg=7, ref=pos, cand_cap=cap, stress=nl["stress"],
                  args=args, pos=np.zeros((n_pad, 3), dt), key_extra=key)
        return st, nl["need"]

    def _install(self, st, need):
        old = self._nl
        same = old is not None and all(np.array_equal(old[k], st[k]) for k in ("numbers", "pbc"))
        self._nl, self.rcut = st, st["rcut"]
        st["built"] = self._calls
        self.n_rebuilds += 1
        # grow now if this would overflow; shrink only for other atoms
        if need > self._width or not self._width or (not same and self._fit(need) * 1.3 < self._width):
            self._width = self._fit(need)
