"""Equivariant products of spherical tensors: the product basis of the descriptor.

A basis is a list of rows (l, m, path), one per component m = -l..l of a function of degree l.  The
path is () for a base function, or (path_a, l_a, path_b, l_b) for a product of two factors.
`couple` returns the rows of a product basis and its sparse recipe,
out[dest] += factors * a[rows_a] * b[rows_b].
"""
import itertools
from collections import namedtuple
from functools import lru_cache

import numpy as np
from scipy import sparse

Table = namedtuple("Table", "meta rows_a rows_b factors dest")
ZERO = 1e-12            # the quadrature leaves ~1e-16 where the exact coefficient is zero
Function = namedtuple("Function", "l path rows")


@lru_cache(maxsize=None)
def _sphere(degree):
    """Points on the unit sphere and weights summing to the mean over it, exact for polynomials of the
    coordinates up to `degree`: Gauss-Legendre in z times a uniform grid in the azimuth."""
    z, wz = np.polynomial.legendre.leggauss(degree // 2 + 1)
    phi = 2 * np.pi * np.arange(degree + 1) / (degree + 1)
    rho = np.sqrt(1.0 - z * z)[:, None]
    points = np.stack([rho * np.cos(phi), rho * np.sin(phi), np.repeat(z[:, None], len(phi), 1)], -1)
    return points.reshape(-1, 3), np.repeat(wz / (2 * len(phi)), len(phi))


@lru_cache(maxsize=None)
def coupling_tensor(l1, l2, L):
    """C[M + L, m1 + l1, m2 + l2], the equivariant product of a degree-l1 and a degree-l2 tensor into
    degree L (l1 + l2 + L even), scaled so that sum C[M, m1, m2] Y_l1m1(u) Y_l2m2(u) = Y_LM(u).
    The projection of Y_l1m1 Y_l2m2 onto Y_LM over the sphere is such a tensor; its scale follows from
    the same quadrature."""
    from .model import angular
    u, w = _sphere(l1 + l2 + L)
    Y = angular(u, max(l1, l2, L), np)
    y1, y2, yL = (Y[:, l * l:(l + 1) * (l + 1)] for l in (l1, l2, L))
    proj = np.einsum("p,pM,pa,pb->Mab", w, yL, y1, y2)
    image = np.einsum("Mab,pa,pb->pM", proj, y1, y2)              # = scale * Y_L
    scale = np.einsum("p,pM,pM->", w, image, yL) / (2 * L + 1)
    C = proj / scale
    C[np.abs(C) < ZERO] = 0.0
    return C


def uncoupled(lmax):
    """The rows of a base of degrees 0..lmax."""
    return [(l, m, ()) for l in range(lmax + 1) for m in range(-l, l + 1)]


def functions(basis):
    """The functions of a basis in the order they first appear, each with the row of every
    component, rows[m + l]."""
    found = {}
    for row, (l, m, path) in enumerate(basis):
        found.setdefault((l, path), {}).setdefault(m, row)
    out = []
    for (l, path), rows in found.items():
        if sorted(rows) != list(range(-l, l + 1)):
            raise ValueError(f"function {path} of degree {l}: components {sorted(rows)}")
        out.append(Function(l, path, np.array([rows[m] for m in range(-l, l + 1)])))
    return out


def couple(A, B, lmax, Lmax, symmetric=False, drop=None):
    """Couplings of the functions of bases A and B (degree <= lmax) into degrees L <= Lmax of natural
    parity (l_a + l_b + L even), each scaled as coupling_tensor; rows in the order made (pairs of
    functions, then L, then M).

    symmetric: B is A, and each unordered pair of functions is coupled once, the one of higher degree
    (of equal degree, the lower path) as the first factor; a function times itself uses the symmetric
    part of the coefficients.  drop: (path, L) pairs left out, by default duplicates()."""
    drop = duplicates() if drop is None else drop
    fa = [f for f in functions(A) if f.l <= lmax]
    if symmetric:
        first = lambda f: (-f.l, f.path)
        pairs = [tuple(sorted((f, g), key=first)) for f, g in itertools.combinations_with_replacement(fa, 2)]
    else:
        pairs = list(itertools.product(fa, [f for f in functions(B) if f.l <= lmax]))
    meta, parts = [], []
    for f, g in pairs:
        path = (f.path, f.l, g.path, g.l)
        square = symmetric and f is g
        for L in range(abs(f.l - g.l), min(f.l + g.l, Lmax) + 1):
            if (f.l + g.l + L) % 2 or (path, L) in drop:
                continue
            C = coupling_tensor(f.l, g.l, L)
            if square:
                C = np.triu(C + C.transpose(0, 2, 1)) - C * np.eye(len(f.rows))
            for M in range(-L, L + 1):
                a, b = np.nonzero(np.abs(C[M + L]) > ZERO)
                if len(a):
                    parts.append((f.rows[a], g.rows[b], C[M + L][a, b], len(meta)))
                    meta.append((L, M, path))
    if not meta:
        raise ValueError("no coupling survived the filters")
    rows_a, rows_b, factors = (np.concatenate([p[k] for p in parts]) for k in range(3))
    dest = np.concatenate([np.full(len(p[0]), p[3]) for p in parts])
    return Table(meta, rows_a.astype(np.int64), rows_b.astype(np.int64), factors.astype(float),
                 dest.astype(np.int64))


DUP_LMAX = 5    # shortcut: duplicates searched for l <= 5 only; couplings of higher l all stay


PAIR00 = ((), 0, (), 0)     # path of (A_0 A_0)


@lru_cache(maxsize=None)
def duplicates():
    """(path, L) of the couplings of order 3, (A A) A, and 4, (A A)(A A), of one base that
    are a multiple of another coupling of the same order and L.  Of each such group the one
    whose second factor has the lowest degree stays (then the lowest path), except that
    (A_0 A_0)_0 A_L stays rather than (A_L A_0)_L A_0."""
    A = uncoupled(DUP_LMAX)
    aa = couple(A, A, DUP_LMAX, DUP_LMAX, symmetric=True, drop=())
    x = np.random.default_rng(0).normal(size=(16, len(A)))
    xaa = evaluate(aa, x, x)
    out = set()
    for t, y in ((couple(aa.meta, A, DUP_LMAX, DUP_LMAX, drop=()), x),
                 (couple(aa.meta, aa.meta, DUP_LMAX, DUP_LMAX, symmetric=True, drop=()), xaa)):
        v, values = evaluate(t, xaa, y), {}
        for k, (L, M, path) in enumerate(t.meta):
            values.setdefault((L, path), np.zeros((len(x), 2 * L + 1)))[:, M + L] = v[:, k]
        for L in {L for L, _ in values}:
            group = sorted((h for l, h in values if l == L),
                           key=lambda h: (not (h[0] == PAIR00 and h[1] == 0), h[3], h))
            u = np.array([values[L, h].ravel() / np.linalg.norm(values[L, h]) for h in group])
            same = np.abs(u @ u.T) > 1 - 1e-9
            for i, h in enumerate(group):
                if same[i, :i].any():       # a multiple of one that comes first
                    out.add((h, L))
    return frozenset(out)


def evaluate(t, x, y):
    """The functions of table t on the values x[sample, rows_a], y[sample, rows_b] of its two
    factors: (sample, number of rows of the product)."""
    gather = sparse.csr_matrix((t.factors, (np.arange(len(t.dest)), t.dest)), shape=(len(t.dest), len(t.meta)))
    step = max(1, 20_000_000 // len(t.dest))
    return np.concatenate([(x[i:i + step, t.rows_a] * y[i:i + step, t.rows_b]) @ gather
                           for i in range(0, len(x), step)])
