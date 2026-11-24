"""Packed datasets: flat arrays, batched and neighbour-listed on the GPU.

One directory per split (train/, val/) with .npy arrays and split.json, and a
meta.json next to them (sources, element references, statistics):

    Z (A,) uint8, pos (A, 3) float32, forces (A, 3) float32
    offset (S+1,) int64       structure k = atoms offset[k]:offset[k+1]
    cell (S, 3, 3) float32    rows are lattice vectors, zero if not periodic
    pbc (S, 3) bool, energy (S,) float64 (total), stress (S, 6) float32 (Voigt, NaN if absent)
    charge (S,) int8, spin (S,) uint8 (2S+1), src (S,) uint8, group (S,) int32 (optional)

    python -m ilmlip.data out_dir train.xyz [more files] --val val.xyz
"""
import argparse
import json
import os
import queue
import struct
import threading
from pathlib import Path

import numpy as np
import torch
from . import runinfo
from . import vcut

ATOM_ARRAYS = {"Z": (np.uint8, ()), "pos": (np.float32, (3,)), "forces": (np.float32, (3,))}
STRUCT_ARRAYS = {"cell": (np.float32, (3, 3)), "pbc": (np.bool_, (3,)), "energy": (np.float64, ()),
                 "stress": (np.float32, (6,)), "charge": (np.int8, ()), "spin": (np.uint8, ()),
                 "src": (np.uint8, ()), "group": (np.int32, ())}
OPTIONAL = {"group"}
ZMAX = 100


def composition(Z, n_atoms):
    """(structures, ZMAX + 1) element counts of concatenated structures."""
    n_atoms = np.asarray(n_atoms, np.int64)
    s = np.repeat(np.arange(len(n_atoms)), n_atoms)
    return np.bincount(s * (ZMAX + 1) + np.asarray(Z, np.int64),
                       minlength=len(n_atoms) * (ZMAX + 1)).reshape(-1, ZMAX + 1)


class PackedWriter:
    """Streams structures into the .npy files of one split; shapes are written on close()."""

    HEADER = 128

    def __init__(self, path):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self._fh, self._n, self._offset, self._keys = {}, {}, [np.zeros(1, np.int64)], None
        self.n_struct = self.n_atoms = 0
        self.periodic = self.stress = False
        self.elements = np.zeros(ZMAX + 1, np.int64)

    def _file(self, key):
        if key not in self._fh:
            fh = open(self.path / f"{key}.npy.tmp", "wb")
            fh.write(b"\0" * self.HEADER)
            self._fh[key], self._n[key] = fh, 0
        return self._fh[key]

    def add(self, n_atoms, **arrays):
        n_atoms = np.asarray(n_atoms, np.int64).reshape(-1)
        s, a = len(n_atoms), int(n_atoms.sum())
        if s == 0:
            return
        arrays.setdefault("charge", np.zeros(s, np.int8))
        arrays.setdefault("spin", np.ones(s, np.uint8))
        z, q, m = (np.asarray(arrays.get(k, ())) for k in ("Z", "charge", "spin"))
        if z.size and (z.min() < 1 or z.max() > ZMAX):
            raise ValueError(f"atomic numbers outside 1..{ZMAX}")
        if q.size and (q.min() < -128 or q.max() > 127):
            raise ValueError("charges outside -128..127")
        if m.size and (m.min() < 1 or m.max() > 255):
            raise ValueError("spin is the multiplicity 2S + 1: 1..255")
        if self._keys is None:
            self._keys = set(arrays)
        elif set(arrays) != self._keys:
            raise ValueError(f"arrays {sorted(arrays)} differ from the first block's {sorted(self._keys)}")
        for key, val in arrays.items():
            spec = ATOM_ARRAYS.get(key) or STRUCT_ARRAYS.get(key)
            if spec is None:
                raise ValueError(f"unknown array {key!r}")
            dt, tail = spec
            n = a if key in ATOM_ARRAYS else s
            v = np.ascontiguousarray(np.asarray(val).astype(dt, copy=False)).reshape(n, *tail)
            self._file(key).write(v.tobytes())
            self._n[key] += n
            if key == "pbc":
                self.periodic |= bool(v.any())
            elif key == "stress":
                self.stress |= bool(np.isfinite(v).all(1).any())
            elif key == "Z":
                self.elements += np.bincount(v, minlength=ZMAX + 1)[:ZMAX + 1]
        self._offset.append(self.n_atoms + np.cumsum(n_atoms))
        self.n_struct += s
        self.n_atoms += a

    def close(self):
        if self.n_struct == 0:
            for key in list(ATOM_ARRAYS) + [k for k in STRUCT_ARRAYS if k not in OPTIONAL]:
                self._file(key)
        for key, fh in self._fh.items():
            dt, tail = ATOM_ARRAYS.get(key) or STRUCT_ARRAYS[key]
            fh.seek(0)
            fh.write(_npy_header(dt, (self._n[key], *tail), self.HEADER))
            fh.close()
            (self.path / f"{key}.npy.tmp").replace(self.path / f"{key}.npy")
        np.save(self.path / "offset.npy", np.concatenate(self._offset))
        for f in [*self.path.glob("n_edges_rc*.npy"), *self.path.glob("n_edges_rc*.json"),
                  *(self.path / f"{k}.npy" for k in OPTIONAL - set(self._fh))]:
            f.unlink(missing_ok=True)
        for f in [*self.path.glob("n_edges_vc*.npy"), *self.path.glob("n_edges_vc*.json")]:
            f.unlink(missing_ok=True)
        missing = (set(ATOM_ARRAYS) | set(STRUCT_ARRAYS)) - set(self._fh) - OPTIONAL
        if missing:
            raise ValueError(f"never given {sorted(missing)}")
        info = dict(n_structures=self.n_struct, n_atoms=self.n_atoms, periodic=self.periodic,
                    stress=self.stress,
                    elements={int(z): int(c) for z, c in enumerate(self.elements) if c})
        (self.path / "split.json").write_text(json.dumps(info, indent=1))
        return info


def _npy_header(dtype, shape, total):
    """A version-1.0 .npy header of exactly `total` bytes."""
    d = {"descr": np.lib.format.dtype_to_descr(np.dtype(dtype)), "fortran_order": False,
         "shape": tuple(int(x) for x in shape)}
    h = repr(d).encode("latin1")
    pad = total - 10 - len(h) - 1
    if pad < 0:
        raise ValueError("npy header too long")
    return b"\x93NUMPY\x01\x00" + struct.pack("<H", total - 10) + h + b" " * pad + b"\n"


def lstsq_e0(Z, offset, energy):
    """{Z: E0} by plain least squares of the total energies on the composition."""
    XtX, Xty, n = np.zeros((ZMAX + 1, ZMAX + 1)), np.zeros(ZMAX + 1), np.diff(offset)
    for k in range(0, len(n), 1 << 17):
        comp = composition(Z[offset[k]:offset[min(k + (1 << 17), len(n))]], n[k:k + (1 << 17)]).astype(np.float64)
        XtX += comp.T @ comp
        Xty += comp.T @ np.asarray(energy[k:k + (1 << 17)], float)
    present = np.where(np.diag(XtX) > 0)[0]
    e0 = np.linalg.lstsq(XtX[np.ix_(present, present)], Xty[present], rcond=None)[0]
    return {int(z): float(v) for z, v in zip(present, e0)}


def check_charge_spin(split, indices, max_charge, max_spin, log=print):
    """Report the structures whose total charge or multiplicity a model with these tables clamps to
    its nearest row (a warning, not an error: a long run should not stop for a few); their number."""
    if not len(indices):
        return 0
    q, s = np.asarray(split.charge[indices]), np.asarray(split.spin[indices])
    n = int(((q < -max_charge) | (q > max_charge) | (s < 1) | (s > max_spin)).sum())
    if n:
        log(f"[warn] {split.path}: {n:,} structures with charges {q.min()}..{q.max()} or multiplicities "
            f"{s.min()}..{s.max()} outside the model's tables (charges -{max_charge}..{max_charge}, "
            f"multiplicities 1..{max_spin}): they get the nearest row")
    return n


class PackedSplit:
    """One split, memory-mapped; only `offset` is read up front."""

    def __init__(self, path):
        self.path = Path(path)
        if not (self.path / "offset.npy").exists():
            raise FileNotFoundError(f"{self.path}: no packed split (offset.npy)")
        self.offset = np.load(self.path / "offset.npy")
        self.n_atoms = np.diff(self.offset)
        for k in ATOM_ARRAYS:
            setattr(self, k, np.load(self.path / f"{k}.npy", mmap_mode="r"))
        for k in STRUCT_ARRAYS:
            p = self.path / f"{k}.npy"
            setattr(self, k, np.load(p, mmap_mode="r") if p.exists() else None)
        info, meta = self.path / "split.json", self.path.parent / "meta.json"
        self.info = json.loads(info.read_text()) if info.exists() else {}
        self.meta = json.loads(meta.read_text()) if meta.exists() else {}
        self.sources = list(self.meta.get("sources", []))

    def __len__(self):
        return len(self.n_atoms)

    @property
    def any_periodic(self):
        if "periodic" in self.info:
            return bool(self.info["periodic"])
        return any(np.asarray(self.pbc[k:k + (1 << 22)]).any() for k in range(0, len(self), 1 << 22))

    @property
    def any_stress(self):
        if "stress" in self.info:
            return bool(self.info["stress"])
        return any(np.isfinite(np.asarray(self.stress[k:k + (1 << 22)])).all(1).any()
                   for k in range(0, len(self), 1 << 22))


def image_boxes(cell, pbc, rcut, with_inv=True):
    """Per structure the image translations a pair can need: K_d = floor(2 rcut / h_d) + 1
    consecutive integers along axis d (h_d: lattice-plane spacing), r = rcut / h_d, inv(cell)."""
    cell = np.asarray(cell, np.float64).reshape(-1, 3, 3)
    pbc = np.asarray(pbc, bool).reshape(-1, 3)
    vol = np.abs(np.linalg.det(cell))
    periodic = pbc.any(1)
    bad = periodic & (vol < 1e-9)
    if bad.any():
        raise ValueError(f"{int(bad.sum())} periodic structure(s) with a singular cell "
                         f"(first: {int(np.argmax(bad))})")
    area = np.linalg.norm(np.cross(cell[:, [1, 2, 0]], cell[:, [2, 0, 1]]), axis=2)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(pbc, rcut * area / np.maximum(vol, 1e-300)[:, None], 0.0)
    K = np.where(pbc, np.floor(2.0 * r).astype(np.int64) + 1, 1)
    if not with_inv:
        return K, r, None
    inv = np.zeros_like(cell)
    if periodic.any():
        inv[periodic] = np.linalg.inv(cell[periodic])
    return K, r, inv


def _rowvec_mat(v, M):
    # written out so the rounding is the same in every batch: counts made once must hold
    return v[:, 0:1] * M[:, 0, :] + v[:, 1:2] * M[:, 1, :] + v[:, 2:3] * M[:, 2, :]


def candidate_pairs(pos, n, a0, K, r, inv, cell, pbc, n_cand, rcut, periodic=True):
    """All (i, j, S) of a batch that can lie within rcut, and which do.  n_cand comes from
    the host, so nothing waits for the device."""
    dev = pos.device
    kp = K.prod(1) if periodic else torch.ones_like(n)
    cnt = n * n * kp
    st = torch.repeat_interleave(torch.arange(len(n), device=dev), cnt, output_size=n_cand)
    loc = torch.arange(n_cand, device=dev) - (torch.cumsum(cnt, 0) - cnt)[st]
    kps, ns = kp[st], n[st]
    pair, img = loc // kps, loc % kps
    i, j = a0[st] + pair // ns, a0[st] + pair % ns
    dp = pos[j] - pos[i]
    if not periodic:
        S = torch.zeros_like(dp)
        return i, j, S, ((dp * dp).sum(1) < rcut * rcut) & (i != j), st
    Ks = K[st]
    kb, kc = Ks[:, 1], Ks[:, 2]
    f = _rowvec_mat(dp, inv[st])
    s0 = torch.where(pbc[st], torch.ceil(-f - r[st]), torch.zeros_like(f))
    S = s0 + torch.stack([img // (kb * kc), (img // kc) % kb, img % kc], 1).to(f.dtype)
    vec = dp + _rowvec_mat(S, cell[st])
    keep = ((vec * vec).sum(1) < rcut * rcut) & ((i != j) | (S != 0).any(1))
    return i, j, S, keep, st


class PackedLoader:
    """Batches of a PackedSplit assembled on `device`: one pass per iteration, a new order per epoch
    (a function of seed and epoch).

    indices: the structures used; weights: per structure, sampled w times per epoch on average;
    resident: "device", "host" (memory-mapped, gathered in a background thread) or "auto" (device if
    the per-atom arrays take less than half its free memory).  Host-resident data are shuffled in blocks
    of 1024 structures within windows of 256 blocks."""

    CHUNK = 1 << 20

    def __init__(self, split, rcut, device="cuda", dtype=None, batch_atoms=2048, batch_structs=64,
                 shuffle=True, seed=0, indices=None, weights=None, resident="auto",
                 pair_rcut=None,
                 edge_cache=True, log=print):
        self.split, self.rcut, self.device = split, float(rcut), torch.device(device)
        self.pair_rcut = pair_rcut
        self.dtype = dtype or torch.float32
        self.batch_atoms, self.batch_structs = int(batch_atoms), int(batch_structs)
        self.shuffle, self.seed, self.log = bool(shuffle), int(seed), log
        S = len(split)
        self.indices = np.arange(S) if indices is None else np.asarray(indices, np.int64)
        self.weights = None if weights is None else np.asarray(weights, float)
        self.epoch = self.skip = 0
        self.n_atoms = split.n_atoms.astype(np.int64)
        if len(self.indices) and (self.n_atoms[self.indices] > self.batch_atoms).any():
            raise ValueError(f"a structure has {int(self.n_atoms[self.indices].max())} atoms > "
                             f"batch_atoms {self.batch_atoms}")
        elements = [int(z) for z in split.info.get("elements", {})] or \
            np.unique(np.asarray(split.Z[:1 << 20])).tolist()
        if min(elements, default=1) < 1 or max(elements, default=1) > ZMAX:
            raise ValueError(f"{split.path}: atomic numbers outside 1..{ZMAX}")

        dev = self.device
        nbytes = split.Z.nbytes + split.pos.nbytes + split.forces.nbytes
        if resident == "auto":
            resident = "device"
            if dev.type == "cuda":
                resident = "device" if nbytes < 0.5 * torch.cuda.mem_get_info(dev)[0] else "host"
        if resident not in ("device", "host"):
            raise ValueError(f"resident must be auto|device|host, got {resident!r}")
        self.resident = resident

        self.any_periodic, self.any_stress = split.any_periodic, split.any_stress
        if self.any_periodic:
            self.periodic, self.n_cand = np.zeros(S, bool), np.zeros(S, np.int64)
            for k in range(0, S, self.CHUNK):
                sl = slice(k, k + self.CHUNK)
                pbc = np.asarray(split.pbc[sl])
                K = image_boxes(split.cell[sl], pbc, self.rcut, with_inv=False)[0]
                self.periodic[sl] = pbc.any(1)
                self.n_cand[sl] = self.n_atoms[sl] ** 2 * K.prod(1)
        else:
            self.periodic, self.n_cand = None, self.n_atoms ** 2

        # device-resident: arrays on the device; host-resident: per-structure tables, shipped per batch
        if resident == "device":
            put = lambda x, dt: torch.tensor(np.asarray(x), dtype=dt, device=dev)
            self.s_K = self.s_r = self.s_inv = self.s_cell = self.s_pbc = self.s_vol = None
            if self.any_periodic:
                pbc, cell64 = np.asarray(split.pbc), np.asarray(split.cell, np.float64)
                K, r, inv = image_boxes(cell64, pbc, self.rcut)
                self.s_K, self.s_r = put(K, torch.long), put(r, torch.float64)
                self.s_inv = put(inv, torch.float64)
                self.s_cell, self.s_pbc = put(cell64, torch.float64), put(pbc, torch.bool)
                self.s_vol = put(np.abs(np.linalg.det(cell64)) * self.periodic, torch.float64)
            self.s_n, self.s_off = put(self.n_atoms, torch.int32), put(split.offset[:-1], torch.long)
            self.s_energy = put(split.energy, torch.float64)
            self.s_stress = put(split.stress, torch.float32) if self.any_stress else None
            self.s_charge = put(split.charge, torch.int8)
            self.s_spin, self.s_src = put(split.spin, torch.uint8), put(split.src, torch.uint8)
            self.a_Z, self.a_pos = put(split.Z, torch.uint8), put(split.pos, torch.float32)
            self.a_F = put(split.forces, torch.float32)
        self.block = 1 if resident == "device" else 1024
        self.window = 256
        self.log(f"[packed] {split.path}: {S:,} structures, {int(self.n_atoms.sum()):,} atoms "
                 f"({nbytes / 1e9:.2f} GB per-atom data, {resident}-resident"
                 + (f", shuffled in blocks of {self.block}" if self.block > 1 else "") + ")")
        self.dropped = torch.zeros((), dtype=torch.long, device=dev)    # edges beyond the counts
        self.n_edges = self._edge_counts(edge_cache)

    def _edge_counts(self, cache):
        """Edges within rcut of every structure, counted once with the batches' arithmetic
        and cached next to the data, stamped with the sizes and times of the files counted from."""
        path = self.split.path / f"n_edges_rc{self.rcut:g}.npy"
        path = path if self.pair_rcut is None else self.split.path / vcut.cache_name(self.pair_rcut)
        self.edge_file = path
        files = [self.split.path / f"{k}.npy" for k in ("offset", "Z", "pos", "cell", "pbc")]
        stamp = {f.name: [f.stat().st_size, f.stat().st_mtime_ns] for f in files if f.exists()}
        side = path.with_suffix(".json")
        if cache and path.exists():
            # the stamp before the counts: a writer replaces the counts first
            old = json.loads(side.read_text()) if side.exists() else None
            ne = np.load(path)
            if len(ne) == len(self.split) and old in (None, stamp):
                if old is None:      # counts too small for the files would stop at check_counts
                    self.log(f"[packed] {path.name} has no stamp: taken to fit the files as they are")
                    try:
                        self._stamp(side, stamp)
                    except OSError:
                        pass
                return ne.astype(np.int64)
        import time
        t0, S = time.time(), len(self.split)
        ne = np.zeros(S, np.int64)
        budget = (1 << 22) if self.any_periodic else (1 << 24)
        start, csum = 0, np.cumsum(self.n_cand)
        while start < S:
            base = csum[start - 1] if start else 0
            end = int(np.searchsorted(csum, base + budget, side="right"))
            end = min(max(end, start + 1), start + (1 << 16))
            ne[start:end] = self._assemble(np.arange(start, end), count_only=True)
            start = end
        self.log(f"[packed] counted {int(ne.sum()):,} edges within {self.rcut:g} A in "
                 f"{time.time() - t0:.1f} s (mean {ne.sum() / max(1, self.n_atoms.sum()):.1f} per atom)")
        if cache:
            try:
                tmp = path.with_suffix(f".tmp{os.getpid()}.npy")
                np.save(tmp, ne.astype(np.int32))
                tmp.replace(path)
                self._stamp(side, stamp)
            except OSError as exc:
                self.log(f"[packed] edge counts not cached ({exc})")
        return ne

    @staticmethod
    def _stamp(side, stamp):
        tmp = side.with_suffix(f".tmp{os.getpid()}.json")
        tmp.write_text(json.dumps(stamp))
        tmp.replace(side)

    def check_counts(self):
        """Raise if a batch had more edges than its cached count (the excess was dropped); waits
        for the device."""
        n = int(self.dropped)
        if n:
            raise RuntimeError(f"{n:,} edges dropped: the counts in {self.edge_file} are too small for "
                               f"these data; delete it to count them again")

    def plan(self, epoch):
        """(order, bounds) of this epoch's batches, greedily packed."""
        rng = np.random.default_rng([self.seed, epoch])
        idx = self.indices
        if self.weights is not None:
            w = self.weights[idx]
            idx = np.repeat(idx, np.floor(w).astype(np.int64) + (rng.random(len(idx)) < w - np.floor(w)))
        if not self.shuffle:
            order = idx.copy()
        elif self.block <= 1:
            order = rng.permutation(idx)
        else:
            blocks = [idx[k:k + self.block] for k in range(0, len(idx), self.block)]
            perm = rng.permutation(len(blocks))
            order = np.concatenate([
                rng.permutation(np.concatenate([blocks[b] for b in perm[w:w + self.window]]))
                for w in range(0, len(perm), self.window)]) if blocks else idx.copy()
        csum = np.cumsum(self.n_atoms[order])
        bounds, start, n = [0], 0, len(order)
        while start < n:
            base = csum[start - 1] if start else 0
            end = int(np.searchsorted(csum, base + self.batch_atoms, side="right"))
            end = min(max(end, start + 1), start + self.batch_structs)
            bounds.append(end)
            start = end
        return order, np.asarray(bounds, np.int64)

    def __len__(self):
        return len(self.plan(self.epoch)[1]) - 1

    def __iter__(self):
        order, bounds = self.plan(self.epoch)
        skip, self.skip = self.skip, 0
        self.epoch += 1
        ranges = [(int(bounds[k]), int(bounds[k + 1])) for k in range(skip, len(bounds) - 1)]
        if self.resident == "host":
            yield from self._host_iter(order, ranges)
            return
        order_dev = torch.as_tensor(order, device=self.device)
        for a, b in ranges:
            yield self._assemble(order[a:b], ids=order_dev[a:b])

    def fixed(self, n_structures=None, seed=0, stratify=False):
        """A fixed list of batches (validation).  stratify: drawn evenly from every group."""
        idx = self.indices
        if stratify and self.split.group is None:
            self.log(f"[warn] {self.split.path}: no group.npy, so a plain random validation sample")
        if stratify and n_structures is not None and self.split.group is not None:
            rng = np.random.default_rng(seed)
            g = np.asarray(self.split.group)[idx]
            srt = np.argsort(g, kind="stable")
            _, starts = np.unique(g[srt], return_index=True)
            ends = np.append(starts[1:], len(srt))
            per = -(-n_structures // max(1, len(starts)))
            picks = [idx[srt[a:b]][rng.choice(b - a, min(per, b - a), replace=False)]
                     for a, b in zip(starts, ends)]
            out = np.concatenate(picks) if picks else idx[:0]
            idx = out[rng.permutation(len(out))[:n_structures]]
        elif n_structures is not None and n_structures < len(idx):
            idx = np.sort(np.random.default_rng(seed).choice(idx, n_structures, replace=False))
        saved = self.indices, self.weights, self.shuffle
        self.indices, self.weights, self.shuffle = idx, None, False
        try:
            order, bounds = self.plan(0)
            batches = [self._assemble(order[bounds[k]:bounds[k + 1]]) for k in range(len(bounds) - 1)]
            self.check_counts()
            return batches
        finally:
            self.indices, self.weights, self.shuffle = saved

    def _atoms_of(self, idx):
        n = self.n_atoms[idx]
        first = self.split.offset[idx] - (np.cumsum(n) - n)
        return np.repeat(first, n) + np.arange(int(n.sum()))

    def _structs_dev(self, ids):
        sel = lambda t: None if t is None else t[ids]
        d = dict(n=self.s_n[ids].long(), energy=self.s_energy[ids], charge=self.s_charge[ids].long(),
                 spin=self.s_spin[ids].long(), src=self.s_src[ids].long(),
                 stress=sel(self.s_stress), vol=sel(self.s_vol))
        if self.any_periodic:
            d.update(K=self.s_K[ids], r=self.s_r[ids], inv=self.s_inv[ids], cell=self.s_cell[ids],
                     pbc=self.s_pbc[ids])
        return d

    def _structs_host(self, idx):
        """Per-structure values of host-resident data, one float64 row each (exact for the
        integers and float32 labels): n, energy, charge, spin, src, stress, volume, K, r, inv,
        cell, pbc."""
        sp, B = self.split, len(idx)
        m = np.zeros((B, 39 if self.any_periodic else 12))
        m[:, 0], m[:, 1] = self.n_atoms[idx], sp.energy[idx]
        m[:, 2], m[:, 3], m[:, 4] = sp.charge[idx], sp.spin[idx], sp.src[idx]
        m[:, 5:11] = sp.stress[idx] if self.any_stress else np.nan
        if self.any_periodic:
            cell, pbc = np.asarray(sp.cell[idx], np.float64), np.asarray(sp.pbc[idx])
            K, r, inv = image_boxes(cell, pbc, self.rcut)
            m[:, 11] = np.abs(np.linalg.det(cell)) * pbc.any(1)
            m[:, 12:15], m[:, 15:18] = K, r
            m[:, 18:27], m[:, 27:36], m[:, 36:39] = inv.reshape(B, 9), cell.reshape(B, 9), pbc
        return m

    def _unpack(self, m):
        B = m.shape[0]
        d = dict(n=m[:, 0].long(), energy=m[:, 1], charge=m[:, 2].long(), spin=m[:, 3].long(),
                 src=m[:, 4].long(), stress=m[:, 5:11] if self.any_stress else None,
                 vol=m[:, 11] if self.any_periodic else None)
        if self.any_periodic:
            d.update(K=m[:, 12:15].long(), r=m[:, 15:18], inv=m[:, 18:27].reshape(B, 3, 3),
                     cell=m[:, 27:36].reshape(B, 3, 3), pbc=m[:, 36:39] > 0)
        return d

    def _host_iter(self, order, ranges):
        q, stop = queue.Queue(maxsize=32), threading.Event()
        sp, cuda = self.split, self.device.type == "cuda"

        def put(item):
            while not stop.is_set():
                try:
                    q.put(item, timeout=0.5)
                    return True
                except queue.Full:
                    pass
            return False

        def work():
            try:
                for a, b in ranges:
                    idx = order[a:b]
                    atom = self._atoms_of(idx)
                    arrays = (torch.from_numpy(np.asarray(sp.Z[atom])),
                              torch.from_numpy(np.ascontiguousarray(sp.pos[atom])),
                              torch.from_numpy(np.ascontiguousarray(sp.forces[atom])),
                              torch.from_numpy(self._structs_host(idx)))
                    if cuda:
                        arrays = tuple(x.pin_memory() for x in arrays)
                    if not put((idx, arrays)):
                        return
            except BaseException as exc:
                put(exc)
            put(None)

        threading.Thread(target=work, daemon=True).start()
        try:
            while True:
                item = q.get()
                if item is None:
                    break
                if isinstance(item, BaseException):
                    raise item
                idx, arrays = item
                Z, pos, F, M = (x.to(self.device, non_blocking=True) for x in arrays)
                yield self._assemble(idx, atoms=(Z.long(), pos, F), structs=self._unpack(M))
        finally:
            stop.set()

    def _assemble(self, idx, ids=None, atoms=None, count_only=False, structs=None):
        """The batch of structures idx; ids: the same on the device; atoms: (Z, pos,
        forces) already gathered; structs: their per-structure tables."""
        dev, dt = self.device, self.dtype
        idx = np.asarray(idx, np.int64)
        B, A, C = len(idx), int(self.n_atoms[idx].sum()), int(self.n_cand[idx].sum())
        if structs is None:
            if self.resident == "device":
                structs = self._structs_dev(torch.as_tensor(idx, device=dev) if ids is None else ids)
            else:
                structs = self._unpack(torch.as_tensor(self._structs_host(idx), device=dev))
        n = structs["n"]
        a0 = torch.cumsum(n, 0) - n
        bvec = torch.repeat_interleave(torch.arange(B, device=dev), n, output_size=A)
        if atoms is None:
            if self.resident == "host":
                atom, sp = self._atoms_of(idx), self.split
                pos = torch.as_tensor(np.ascontiguousarray(sp.pos[atom]), device=dev)
                if not count_only:
                    Z = torch.as_tensor(np.asarray(sp.Z[atom], np.int64), device=dev)
                    F = torch.as_tensor(np.ascontiguousarray(sp.forces[atom]), device=dev)
            else:
                ids = torch.as_tensor(idx, device=dev) if ids is None else ids
                atom = self.s_off[ids][bvec] + torch.arange(A, device=dev) - a0[bvec]
                pos = self.a_pos[atom]
                if not count_only:
                    Z, F = self.a_Z[atom].long(), self.a_F[atom]
        else:
            Z, pos, F = atoms
        if count_only and self.pair_rcut is not None:
            Z = vcut.count_species(self, atom)
        periodic = self.any_periodic and bool(self.periodic[idx].any())
        g = structs.get
        i, j, S, keep, st = candidate_pairs(pos.double(), n, a0, g("K"), g("r"), g("inv"), g("cell"),
                                            g("pbc"), C, self.rcut, periodic)
        if self.pair_rcut is not None:
            keep = keep & vcut.within(self, Z, i, j, pos.double(), S, st, structs, periodic)
        if count_only:
            return torch.zeros(B, dtype=torch.long, device=dev).index_add_(0, st, keep.long()).cpu().numpy()

        # compact the kept candidates into E slots without asking the device how many: E is cached
        E = int(self.n_edges[idx].sum())
        rank = torch.cumsum(keep, 0) - 1
        dst = torch.where(keep & (rank < E), rank, torch.full_like(rank, E))
        packed = torch.stack([i, j, S[:, 0].long(), S[:, 1].long(), S[:, 2].long()], 1)
        out = torch.zeros(E + 1, 5, dtype=torch.long, device=dev)
        out.index_copy_(0, dst, packed)
        out = out[:E]
        n_keep = keep.sum()
        valid = torch.arange(E, device=dev) < n_keep           # padding beyond rcut if a count disagrees
        self.dropped += (n_keep - E).clamp(min=0)
        ii, jj = out[:, 0], out[:, 1]
        if periodic:
            shift = _rowvec_mat(out[:, 2:].double(), structs["cell"][bvec[ii]])
        else:
            shift = torch.zeros(E, 3, dtype=torch.float64, device=dev)
        shift = torch.where(valid[:, None], shift, torch.full_like(shift, 2.0 * self.rcut))
        return dict(
            positions=pos.to(dt).requires_grad_(True), mu=Z - 1,       # species are Z = 1..100
            ind_i=torch.where(valid, ii, 0), ind_j=torch.where(valid, jj, 0),
            # energy labels in float64, as the model's energies: absolute totals lose digits in float32
            shift=shift.to(dt), batch=bvec, energy=structs["energy"], forces=F.to(dt),
            n_atoms_per_struct=n.to(dt),
            volume=(structs["vol"].to(dt) if structs["vol"] is not None
                    else torch.zeros(B, dtype=dt, device=dev)),
            stress=(structs["stress"].to(dt) if structs["stress"] is not None
                    else torch.full((B, 6), float("nan"), dtype=dt, device=dev)),
            src=structs["src"], charge=structs["charge"], spin=structs["spin"],
            n_atoms=A, n_struct=B, packed=True)


KEYS = dict(energy=("REF_energy", "energy", "free_energy", "DFT_energy", "total_energy"),
            forces=("REF_forces", "forces", "DFT_forces", "force"),
            stress=("REF_stress", "stress", "DFT_stress"))


def label(atoms, quantity):
    """A label from info, arrays or an attached calculator; None if absent."""
    store = [atoms.info, atoms.arrays, getattr(atoms.calc, "results", None) or {}]
    return next((s[k] for k in KEYS[quantity] for s in store if k in s), None)


def pack(paths, out):
    """Write ASE-readable structure files into one packed split."""
    from ase.io import iread
    w = PackedWriter(out)
    for p in paths:
        for k, at in enumerate(iread(str(p))):
            e, f, s = (label(at, q) for q in ("energy", "forces", "stress"))
            if e is None or f is None:
                raise ValueError(f"{p}, structure {k}: no energy or forces")
            s = np.full(6, np.nan) if s is None else np.asarray(s, float).reshape(-1)
            if s.size == 9:
                s = s.reshape(3, 3)[[0, 1, 2, 1, 0, 0], [0, 1, 2, 2, 2, 1]]
            w.add([len(at)], Z=at.numbers, pos=at.positions, forces=f, cell=np.asarray(at.cell),
                  pbc=at.pbc[None], energy=[e], stress=s[None], charge=[int(at.info.get("charge", 0))],
                  spin=[int(at.info.get("spin", 1))], src=[0])
    return w.close()


def main(argv=None):
    p = argparse.ArgumentParser(description="Pack structure files (extxyz, ...) into a dataset.")
    p.add_argument("out", help="dataset directory (train/, val/, meta.json)")
    p.add_argument("train", nargs="+", help="training structures")
    p.add_argument("--val", nargs="+", required=True, help="validation structures")
    a = p.parse_args(argv)
    runinfo.show()
    out = Path(a.out)
    info = pack(a.train, out / "train")
    pack(a.val, out / "val")
    tr = PackedSplit(out / "train")
    e0 = lstsq_e0(tr.Z, tr.offset, tr.energy)
    per_atom = np.asarray(tr.energy) / tr.n_atoms
    (out / "meta.json").write_text(json.dumps(dict(
        sources=["data"], e0=e0, stats=dict(train=dict(
            e_per_atom_mean=float(per_atom.mean()), e_per_atom_std=float(per_atom.std())))),
        indent=1))
    print(f"{out}: {info['n_structures']:,} training structures, E0 of {len(e0)} elements")


if __name__ == "__main__":
    main()
