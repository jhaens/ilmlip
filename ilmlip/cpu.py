"""ilmlip on the CPU: a compiled engine (C++, OpenMP, SIMD) and an ASE calculator.

    from ilmlip.cpu import CpuCalculator

    atoms.calc = CpuCalculator("model.pt")                  # float32, OpenMP's default thread count
    atoms.calc = CpuCalculator("model.pt", n_threads=1)     # one core (one MD per core)

The descriptor and readout are written out by hand in csrc/cpu_engine.cpp, forward and backward,
one call per MD step.  Same energy, forces and stress as the torch model.

The engine is compiled on first use for the CPU it runs on (g++ -O3 -march=native -fopenmp; CXX picks
another compiler) into a folder next to the package, or ILMLIP_BUILD, under a name made of the source,
the compiler and the CPU, so nodes with different CPUs on one file system each get their own build.
"""
import ctypes
import fcntl
import functools
import hashlib
import os
import platform
import shutil
import subprocess
import threading
from contextlib import suppress
from pathlib import Path

import numpy as np
import torch
from ase.calculators.calculator import PropertyNotImplementedError, all_changes

from .model import NO_STRESS, fold_charge_spin, load as load_torch, refuse_charge_spin, zbl_tables
from .verlet import VerletCalculator, check_energy, check_finite
from . import vcut

SRC = Path(__file__).with_name("csrc") / "cpu_engine.cpp"
FLAGS = ["-O3", "-march=native", "-fopenmp", "-std=gnu++17", "-shared", "-fPIC",
         "-fno-math-errno", "-ffp-contract=fast", "-Wno-psabi"]
# GCC's generic x86 tuning splits unaligned 32-byte loads and stores in halves
SPLIT_OFF = ["-mno-avx256-split-unaligned-load", "-mno-avx256-split-unaligned-store"]
MAGIC, VERSION = 0x67686370, 5
_LIBS = {}

_CPU_KEYS = ("model name", "flags",                                          # x86
             "Features", "CPU implementer", "CPU architecture", "CPU variant", "CPU part")   # ARM


def _cpu_fingerprint():
    """The CPU model and its instruction-set features (-march=native depends on both)."""
    found = {}
    with suppress(OSError):
        for line in open("/proc/cpuinfo"):
            key = line.split(":", 1)[0].strip()
            if key in _CPU_KEYS and key not in found:
                found[key] = line.split(":", 1)[1].strip()
    return "|".join([platform.machine(), platform.processor()] + [found.get(k, "") for k in _CPU_KEYS])


def _compiler():
    cxx = os.environ.get("CXX") or shutil.which("g++") or shutil.which("c++")
    if not cxx:
        raise RuntimeError("the CPU engine needs a C++ compiler (g++ >= 8); set CXX")
    try:
        out = subprocess.run([cxx, "--version"], capture_output=True, text=True)
    except OSError as exc:
        raise RuntimeError(f"the C++ compiler {cxx!r} does not run ({exc}); set CXX to its path") from None
    return cxx, (out.stdout.splitlines() or ["?"])[0]


def build_dir():
    env = os.environ.get("ILMLIP_BUILD")
    if env:
        return Path(env)
    root = Path(__file__).resolve().parent.parent
    cache = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "ilmlip" / "build"
    if root.name in ("site-packages", "dist-packages"):
        return cache
    here = root / "build"
    with suppress(OSError):
        here.mkdir(parents=True, exist_ok=True)
        if os.access(here, os.W_OK):
            return here
    return cache


@functools.lru_cache(maxsize=None)
def _flags(cxx, version):
    """FLAGS, and SPLIT_OFF where the compiler takes them (GCC on x86)."""
    try:
        ok = subprocess.run([cxx, *SPLIT_OFF, "-x", "c++", "-fsyntax-only", "-"], input="",
                            capture_output=True, text=True, timeout=120).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        ok = False
    return FLAGS + (SPLIT_OFF if ok else [])


def library():
    """Path of the engine built for this CPU (built on first use, one process at a time)."""
    cxx, version = _compiler()
    flags = _flags(cxx, version)
    key = hashlib.sha256(b"\0".join([SRC.read_bytes(), version.encode(), " ".join(flags).encode(),
                                     _cpu_fingerprint().encode()])).hexdigest()[:16]
    d = build_dir()
    d.mkdir(parents=True, exist_ok=True)
    so = d / f"cpu_engine_{key}.so"
    if so.exists():
        return so
    with open(d / ".lock", "w") as lock:
        with suppress(OSError):               # file systems without locks: build anyway
            fcntl.flock(lock, fcntl.LOCK_EX)
        if so.exists():
            return so
        tmp = d / f".cpu_engine_{key}.{platform.node()}.{os.getpid()}.so"
        cmd = [cxx, *flags, str(SRC), "-o", str(tmp)]
        p = subprocess.run(cmd, capture_output=True, text=True)
        if p.returncode:
            raise RuntimeError(f"compiling the CPU engine failed ({' '.join(cmd)}):\n{p.stderr[-4000:]}")
        os.replace(tmp, so)
    return so


def _lib():
    so = str(library())
    if so not in _LIBS:
        lib = ctypes.CDLL(so)
        P, I64, I32 = ctypes.c_void_p, ctypes.c_int64, ctypes.c_int
        for t in ("f32", "f64"):
            create, compute, free = (getattr(lib, f"vv_{x}_{t}") for x in ("create", "compute", "free"))
            create.restype, create.argtypes = P, [P, I64, P, I64, ctypes.c_char_p, I32]
            compute.restype = I32
            compute.argtypes = [P, I32] + [P] * 11 + [I32, ctypes.c_char_p, I32]
            free.restype, free.argtypes = None, [P]
        _LIBS[so] = lib
    return _LIBS[so]


def _np(t):
    return t.detach().cpu().double().numpy()


def _species_z(m):
    """The element embeddings, (species, embed_dim)."""
    return _np(m.type_feat) @ _np(m.embed_proj.weight).T


def _rows(P, need):
    """The engine's rows of the Products P: all base rows, then the rows of orders 2, 3, 4 that
    need (order -> rows) asks for or a higher product reads.  ({order: {row: engine row}}, number
    of rows, terms (a, b, out, factor))."""
    need = {o: set(need.get(o, ())) for o in (1, 2, 3, 4)}
    terms = {}
    for o, key, lb, rb in ((4, "c4", 2, 2), (3, "c3", 2, 1), (2, "c2", 1, 1)):
        if key not in P.couplings:
            if need[o]:
                raise ValueError(f"rows of order {o} read but not computed")
            continue
        c = P.couplings[key]
        l_, r_, g_, s_ = (_np(x) if x.is_floating_point() else x.cpu().numpy()
                          for x in (c.rows_a, c.rows_b, c.factors, c.dest))
        keep = np.isin(s_, sorted(need[o]))
        terms[o] = [(int(a), int(b), float(g), int(s)) for a, b, g, s in
                    zip(l_[keep], r_[keep], g_[keep], s_[keep])]
        for a, b, _, _ in terms[o]:
            need[lb].add(a)
            need[rb].add(b)
    nbase = len(P.meta[1])
    where, nrows = {1: {r: r for r in range(nbase)}}, nbase
    for o in (2, 3, 4):
        where[o] = {r: nrows + k for k, r in enumerate(sorted(need[o]))}
        nrows += len(need[o])
    src = {2: (1, 1), 3: (2, 1), 4: (2, 2)}
    return where, nrows, [(where[src[o][0]][a], where[src[o][1]][b], where[o][s], g)
                          for o in (2, 3, 4) for a, b, g, s in terms.get(o, [])]


def _put_block(I, D, C, lo, nbase, nrows, nl, flat, frows):
    I += [C, lo, nbase, nrows, nl, len(flat), len(frows)]
    for a, b, o, _ in flat:
        I += [a, b, o]
    D.append(np.array([g for *_, g in flat]))
    I += frows


def pack(m):
    """(ints, floats) of an ILMlip module without charge/spin inputs (fold them first).

    Products compute only the rows something reads (readout, next layer's reduction, higher product).
    The readout input is reordered to the engine's layout (per block, per read-out row, per channel) by
    permuting the first layer's weights, with the feature whitening folded in.  Layer 0 is one block per
    channel group.  Ordered-pair terms (FiLM, back-flow) go in as factors: the engine makes a pair's
    weights when a structure has the pair.  The bounds go in as numbers (the gate's per-layer bond RMS
    too), the squash's mean and spread in the engine's column order."""
    if m.charge_spin:
        raise ValueError("fold the charge/spin inputs first (model.fold_charge_spin)")
    S, L = m.n_species, m.n_layers
    hidden = [l for l in m.readout if hasattr(l, "weight")]
    acts = [type(l).__name__ for l in m.readout]
    if acts != ["Linear", "SiLU"] * len(hidden):
        raise ValueError(f"unsupported readout {acts}")
    I = [MAGIC, VERSION, S, L, m.lmax, m.lmax_feat, m.gbasis.size, m.gbasis.p, int(m.zbl), len(hidden)]
    D = [[m.rcut, m.zbl_factor, float(m.energy_scale)]]

    z = _species_z(m)
    chem0 = z @ _np(m.chem0.weight).T                  # (S, channels)
    weights = [_np(w) for w in m.edge_weights]         # per degree (n_basis, channels_l)
    prod0 = list(m.prod0) if m.taper else [m.prod0]

    # every layer's blocks: (Products, channels, first channel, degrees, rows read out as (key, rows))
    blocks = [[(P, hi - lo, lo, g + 1,
                [((b, o), getattr(m, f"inv0g{b}_{o}")) for bb, o in m.group_keys if bb == b] if m.taper else
                [((0, o), getattr(m, f"inv0_{o}")) for t, o in m.inv_keys if t == 0])
               for b, (P, (g, lo, hi)) in enumerate(zip(prod0, m.groups))]]
    for t in range(1, L):
        P = m.prods[t - 1] if t <= len(m.prods) else None
        blocks.append([(P, m.n_message, 0, m.lmax_feat + 1,
                        [((t, o), getattr(m, f"inv{t}_{o}")) for tt, o in m.inv_keys if tt == t])])
    reduces = [[m.reduces[t]] if not isinstance(m.reduces[t], torch.nn.ModuleList) else list(m.reduces[t])
               for t in range(L - 1)]

    cols, at = {}, 0
    for b, o in m.group_keys:
        cols[(0, b, o)] = at
        at += blocks[0][b][1] * len(getattr(m, f"inv0g{b}_{o}"))
    for t, o in m.inv_keys:
        cols[(t, 0, o)] = at
        at += (m.n_channels if t == 0 else m.n_message) * len(getattr(m, f"inv{t}_{o}"))
    scalars = m.gram_rows0.tolist()
    gram_col = at + m.n_message * len(scalars)

    rowmaps, feat_cols = [], []
    for t in range(L):
        I.append(1000 + t)
        I.append(len(blocks[t]))
        maps = []
        for b, (P, C, lo, nl, read) in enumerate(blocks[t]):
            need = {}
            for (_, o), rows in read:
                need.setdefault(o, set()).update(int(x) for x in rows)
            if t < L - 1:
                for o in reduces[t][b].orders:
                    need.setdefault(o, set()).update(int(x) for x in reduces[t][b].per_function(o)[0])
            if P is None:                              # the last layer before a Gram readout
                nbase = len(m.ycouple[t - 1].meta)
                where, nrows, flat = {1: {r: r for r in range(nbase)}}, nbase, []
            else:
                nbase = len(P.meta[1])
                where, nrows, flat = _rows(P, need)
            maps.append(where)
            frows = []
            for (a, o), rows in read:
                key = (t, a if t == 0 and m.taper else 0, o)
                for rr, r in enumerate(int(x) for x in rows):
                    frows.append(where[o][r])
                    feat_cols += [cols[key] + c * len(rows) + rr for c in range(C)]
            if P is None:
                for f, r in enumerate(scalars):
                    frows.append(r)
                    feat_cols += [at + c * len(scalars) + f for c in range(C)]
            _put_block(I, D, C, lo, nbase, nrows, nl, flat, frows)
        rowmaps.append(maps)
        if t > 0:
            rows = []
            for b, red in enumerate(reduces[t - 1]):
                for o in red.orders:
                    used, slot, w = red.per_function(o)
                    w = _np(w)                         # (functions, n, k)
                    rows += [(b, rowmaps[t - 1][b][o][r], s, w[f])
                             for f, (r, s) in enumerate(zip(used.tolist(), slot.tolist()))]
            yc = m.ycouple[t - 1]
            I += [reduces[t - 1][0].n_lm, len(rows), len(yc.dest)]
            for b, s_, t_, w in rows:
                I += [b, s_, t_]
                D.append(w)
            for lm, r, g, o in zip(yc.rows_a.tolist(), yc.rows_b.tolist(), _np(yc.factors), yc.dest.tolist()):
                I += [lm, r, o]
                D.append(g)

    r, rb = m.gram_proj[0].shape[1], m.bispec_proj[0].shape[1]
    I += [1400, r, rb]
    for l in range(m.lmax_feat + 1):
        rows = getattr(m, f"gram_rows{l}")
        I += [len(rows) // (2 * l + 1)] + rows.tolist()
        D += [_np(m.gram_proj[l]), _np(m.bispec_proj[l])]
    n3 = len(m.bispec.meta[3])
    where, nrows, flat = _rows(m.bispec, {3: set(range(n3))})
    bis_col = gram_col + (m.lmax_feat + 1) * r * (r + 1) // 2
    feat_cols += list(range(gram_col, bis_col))
    frows = [where[3][k] for k in range(n3)]
    for k in range(n3):
        feat_cols += [bis_col + c * n3 + k for c in range(rb)]
    _put_block(I, D, rb, 0, len(m.bispec.meta[1]), nrows, 0, flat, frows)

    # what the engine needs for the pair tables: FiLM, and the back-flow without the sender's own chemistry
    n_bf = m.lmax_ind + 1
    I += [1500, n_bf] + [int(c) for c in m.chan]
    D += weights + [chem0] + [_np(w) for w in m.film_weights] + [_np(m._film_table())]
    D += [_np(m._order1_mix(l)) for l in range(n_bf)]

    if sorted(feat_cols) != list(range(m.n_features)):
        raise ValueError("feature count mismatch")
    # readout: whitening folded into the first layer unless feat_bound squashes it (then the engine whitens),
    # its columns in the engine's order
    I += [2000, m.n_features]
    w1, b1 = _np(hidden[0].weight), _np(hidden[0].bias)
    if not m.feat_bound:
        w1 = w1 / _np(m.feat_std)[None, :]
        b1 = b1 - w1 @ _np(m.feat_mean)
    w1 = w1[:, np.asarray(feat_cols)]
    bias = b1[None, :] + _np(m.elem_bias)                              # (S, H)
    for h, lin in enumerate(hidden):
        w = w1 if h == 0 else _np(lin.weight)
        I += [w.shape[1], w.shape[0]]
        D += [w, bias if h == 0 else _np(lin.bias)]
    D += [_np(m.head), _np(m.atomic_energies)]
    z_, rcov = zbl_tables(m.species) if m.zbl else (np.zeros(S), np.zeros(S))
    D += [np.asarray(z_, float), np.asarray(rcov, float)]
    I += [2500, int(bool(m.bound)), int(bool(m.feat_bound)), int(bool(m.zbl_switch))]
    D.append(np.array([m.bond_scale, m.bound or 0.0, m.feat_bound or 0.0, m.zbl_switch or 0.0]))
    if m.bound:
        D.append(_np(m.bond_rms))
    if m.feat_bound:
        D += [_np(m.feat_mean)[feat_cols], _np(m.feat_std)[feat_cols]]
    vcut.pack(m, I, D)
    I.append(MAGIC)
    return np.asarray(I, np.int64), np.concatenate([np.asarray(x, np.float64).reshape(-1) for x in D])


_CREATE_LOCK = threading.Lock()     # the engine builds its harmonics tables in a global


class CpuModel:
    """A checkpoint (or ILMlip module) in the engine, float32 or float64."""

    def __init__(self, model, dtype="float32", charge=None, spin=None):
        if isinstance(model, (str, Path)):
            model, _ = load_torch(model, "cpu")
        m = fold_charge_spin(model, charge, spin)
        self.species, self.rcut, self.n_layers = list(m.species), float(m.rcut), int(m.n_layers)
        self.pair_rcut = vcut.table(m)
        self.charge_spin_state = getattr(m, "folded_charge_spin", None)
        self.stress = m.stress
        self.dtype = np.dtype(dtype)
        if self.dtype not in (np.float32, np.float64):
            raise ValueError(f"dtype float32 or float64, not {dtype}")
        self._t = "f32" if self.dtype == np.float32 else "f64"
        ints, floats = pack(m)
        self._lib = _lib()
        self._err = ctypes.create_string_buffer(4096)
        with _CREATE_LOCK:
            self._h = getattr(self._lib, f"vv_create_{self._t}")(ints.ctypes.data, len(ints), floats.ctypes.data,
                                                                 len(floats), self._err, len(self._err))
        if not self._h:
            raise RuntimeError(f"CPU engine: {self._err.value.decode()}")
        self._compute = getattr(self._lib, f"vv_compute_{self._t}")
        self._lock = threading.Lock()          # one call at a time: the engine keeps its scratch

    def __del__(self):
        h, self._h = getattr(self, "_h", None), None
        if h:
            getattr(self._lib, f"vv_free_{self._t}")(h)

    def compute(self, pos, mu, off, cj, csh, rev, n_threads=0):
        """(energy, site energies, forces, virial (3, 3), bonds inside rcut) of one structure:
        positions (n, 3), species index per atom, candidate offsets (n + 1), neighbour, shift
        vector (cand, 3) and reverse per candidate (see check_pairs).  Calls are serialised."""
        pos = np.ascontiguousarray(pos, np.float64)
        n = len(pos)
        mu, off = np.ascontiguousarray(mu, np.int32), np.ascontiguousarray(off, np.int64)
        cj, rev = np.ascontiguousarray(cj, np.int32), np.ascontiguousarray(rev, np.int32)
        csh = np.ascontiguousarray(csh, np.float64)
        nc = len(cj)
        if nc >= 2 ** 31 - 1:
            raise ValueError(f"CPU engine: {nc} candidate pairs, more than the engine's 32-bit indices hold")
        if (pos.shape != (n, 3) or mu.shape != (n,) or off.shape != (n + 1,) or int(off[-1]) != nc
                or rev.shape != (nc,) or csh.shape != (nc, 3)):
            raise ValueError("CPU engine: inconsistent array shapes")
        e, nb = np.zeros(1), np.zeros(1, np.int64)
        e_atom, forces, vir = np.empty(n), np.empty((n, 3)), np.empty(9)
        with self._lock:
            rc = self._compute(self._h, n, pos.ctypes.data, mu.ctypes.data, off.ctypes.data, cj.ctypes.data,
                               csh.ctypes.data, rev.ctypes.data, e.ctypes.data, e_atom.ctypes.data,
                               forces.ctypes.data, vir.ctypes.data, nb.ctypes.data, int(n_threads or 0),
                               self._err, len(self._err))
            if rc:
                raise RuntimeError(f"CPU engine: {self._err.value.decode()}")
        return float(e[0]), e_atom, forces, vir.reshape(3, 3), int(nb[0])

    def check_pairs(self, mu, off, cj, rev):
        """Raise unless species, neighbours and reverses index what they should."""
        n, nc = len(mu), len(cj)
        if nc >= 2 ** 31 - 1:
            raise ValueError(f"{nc} neighbour pairs: the CPU engine indexes them with 32 bits")
        if n and (mu.min() < 0 or mu.max() >= len(self.species)):
            raise ValueError("species index out of range")
        if nc and (cj.min() < 0 or cj.max() >= n or rev.min() < 0 or rev.max() >= nc):
            raise ValueError("neighbour or reverse index out of range")
        if np.any(np.diff(off) < 0) or (n and (off[0] != 0 or off[-1] != nc)):
            raise ValueError("candidate offsets are not monotonic")


def reverse_pairs(i, j, S):
    """The reverse (j, i, -S) of every pair, and which pairs have one (rounding can leave a
    pair on one side of the cutoff only)."""
    S = np.asarray(S, np.int64)
    n, b = int(max(i.max(initial=0), j.max(initial=0))) + 1, 2 * int(np.abs(S).max(initial=0)) + 1
    if n * n * b ** 3 <= 2 ** 63:                       # (i, j, S) fits one int64 key
        def key(a, c, s):
            return (((np.asarray(a, np.int64) * n + c) * b + s[:, 0]) * b + s[:, 1]) * b + s[:, 2] + (b ** 3 - 1) // 2
        kf = key(i, j, S)
        of = np.argsort(kf)
        kf, kr = kf[of], key(j, i, -S)
        at = np.minimum(np.searchsorted(kf, kr), max(len(kf) - 1, 0))
        ok = kf[at] == kr
        return np.where(ok, of[at], -1), ok
    kf = np.stack([i, j, S[:, 0], S[:, 1], S[:, 2]]).astype(np.int64)
    kr = np.stack([j, i, -S[:, 0], -S[:, 1], -S[:, 2]]).astype(np.int64)
    of, orr = np.lexsort(kf[::-1]), np.lexsort(kr[::-1])
    rev = np.full(len(i), -1, np.int64)
    if np.array_equal(kf[:, of], kr[:, orr]):
        rev[orr] = of
        return rev, np.ones(len(i), bool)
    index = {tuple(k): c for c, k in enumerate(kf.T)}
    for c, k in enumerate(kr.T):
        rev[c] = index.get(tuple(k), -1)
    return rev, rev >= 0


class CpuCalculator(VerletCalculator):
    """ASE calculator: the CPU engine on a Verlet list (vesin, rebuilt in the background).

    model: checkpoint path, ILMlip module or CpuModel; dtype: float32 (default; "tf32" is the same,
    CPUs have no TF32) or float64; charge, spin: only for models with charge/spin inputs (default 0, 1);
    n_threads: engine threads (default 0: omp_get_max_threads(), which OMP_NUM_THREADS sets).  A CpuModel
    brings its own dtype and charge/spin state (giving others is an error)."""

    engine, device = "cpp", "cpu"

    def __init__(self, model, compute_stress=True, skin=1.5, n_threads=None, dtype=None,
                 charge=None, spin=None, **kw):
        dtype = "float32" if dtype == "tf32" else dtype
        if isinstance(model, CpuModel):
            if dtype is not None and np.dtype(dtype) != model.dtype:
                raise ValueError(f"the CpuModel has dtype={model.dtype}; build another CpuModel for "
                                 f"dtype={np.dtype(dtype)}")
            refuse_charge_spin(model.charge_spin_state, charge, spin)
        else:
            model = CpuModel(model, dtype or "float32", charge, spin)
        kw.pop("background_compile", None)      # nothing compiles in the background here
        super().__init__(model.species, model.rcut, np.float64, compute_stress=compute_stress and model.stress, skin=skin,
                         background_compile=False, **kw)
        self.pair_rcut = model.pair_rcut
        self.model, self.n_threads, self.model_stress = model, int(n_threads or 0), model.stress
        self.charge_spin_state = model.charge_spin_state
        self.kernels, self.layout = "cpu", "cpu"

    def _make_state(self, pos, atoms, nl):
        n = nl["n"]
        i, j, S = nl["i"], nl["j"], nl["S"]
        if len(i) and np.any(np.diff(i) < 0):
            order = np.argsort(i, kind="stable")
            i, j, S = i[order], j[order], S[order]
        rev, ok = reverse_pairs(i, j, S)
        if not ok.all():                                   # drop pairs without a reverse
            keep = np.flatnonzero(ok)
            new = np.full(len(i), -1, np.int64)
            new[keep] = np.arange(len(keep))
            i, j, S, rev = i[keep], j[keep], S[keep], new[rev[keep]]
        off = np.zeros(n + 1, np.int64)
        np.cumsum(np.bincount(i, minlength=n), out=off[1:])
        args = (np.ascontiguousarray(nl["mu"], np.int32), off, np.ascontiguousarray(j, np.int32),
                np.ascontiguousarray(S @ nl["cell"], np.float64), np.ascontiguousarray(rev, np.int32))
        self.model.check_pairs(args[0], off, args[2], args[4])
        st = dict(n=n, numbers=atoms.numbers.copy(), pbc=nl["pbc"], cell=nl["cell"], h0=nl["cell"], S=S,
                  ref=pos, stress=nl["stress"], vol=nl["vol"], args=args, pos=np.zeros((n, 3), np.float64))
        return st, nl["need"]

    def get_hessian(self, *args, **kwargs):
        raise NotImplementedError("get_hessian: not in the C++ engine; load_calculator(..., engine='jax') or "
                                  "engine='torch'")

    def _set_cell(self, st, h):
        st.update(cell=h.copy(), vol=abs(float(np.linalg.det(h))),
                  args=st["args"][:3] + (st["S"] @ h,) + st["args"][4:])

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        got = None if atoms is None else self._state_of(atoms)
        if got is not None and got != tuple(self.charge_spin_state):
            q, s = self.charge_spin_state
            raise ValueError(f"atoms.info has charge {got[0]}, spin {got[1]}, but this "
                             f"calculator is for charge {q}, spin {s}")
        if "stress" in properties and not self.model_stress:
            raise PropertyNotImplementedError(NO_STRESS)
        self._calls += 1
        if atoms is None:
            atoms = self.atoms
        else:
            check_finite(atoms.positions)
            s = self._snap
            if (s is not None and len(s) == len(atoms) and np.array_equal(s.numbers, atoms.numbers)
                    and np.array_equal(s.cell.array, atoms.cell.array) and np.array_equal(s.pbc, atoms.pbc)):
                s.positions[:] = atoms.positions
            else:
                s = self._snap = atoms.copy()
            self.atoms = s
        self._refresh(atoms)
        st, h = self._nl, atoms.cell.array
        if not np.array_equal(st["cell"], h):
            self._set_cell(st, h)
        st["pos"][:] = atoms.positions
        e, e_atom, forces, vir, nb = self.model.compute(st["pos"], *st["args"], n_threads=self.n_threads)
        check_energy(e)
        self.n_edges = nb
        self.results = dict(energy=e, free_energy=e, forces=forces, energies=e_atom)
        if st["stress"]:
            sig = 0.5 * (vir + vir.T) / st["vol"]
            self.results["stress"] = np.array([sig[0, 0], sig[1, 1], sig[2, 2], sig[1, 2], sig[0, 2], sig[0, 1]])
