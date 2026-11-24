"""The ilmlip model."""
import copy
import inspect
from math import factorial, sqrt

import numpy as np
import torch
import torch.nn as nn

from .coupling import couple, functions, uncoupled
from . import fused_torch
from .fused_torch import KERNELS

N_GROUPS, N_PERIODS = 32, 7

# Ziegler-Biersack-Littmark screening: e^2/(4 pi eps0) in eV A, length 0.8854 a0 in A, (weight, decay) terms
COULOMB = 14.399645
SCREENING_LENGTH = 0.46850
SCREENING = ((0.18175, 3.19980), (0.50986, 0.94229), (0.28022, 0.40290), (0.02817, 0.20162))
NO_STRESS = "this model was not trained on stresses (hparams stress=False): no stress or virial"


def core_switch(z, zc):
    """1 - 10 u^3 + 15 u^4 - 6 u^5 for u = z / zc < 1, 0 beyond: 1 at 0, 0 at zc, two vanishing derivatives at both."""
    u = (z / zc).clamp(0.0, 1.0)
    return 1.0 - u ** 3 * (10.0 - 15.0 * u + 6.0 * u * u)


def repulsion(r, z1, z2, rmax, xp=torch):
    """Screened nuclear repulsion of a pair of atomic numbers z1, z2 at distance r (numpy, torch or
    jax.numpy), taken to zero at rmax by (1 - u)^3 (1 + 3u + 6u^2 + 10u^3 + 15u^4 + 21u^5), u = r / rmax."""
    return pair_repulsion(r, COULOMB * z1 * z2, z1 ** 0.23 + z2 ** 0.23, rmax, xp)


def pair_repulsion(r, coulomb, powers, rmax, xp=torch):
    """repulsion of a pair from its constants: coulomb = e^2 z1 z2, powers = z1^0.23 + z2^0.23."""
    reduced = r * powers / SCREENING_LENGTH
    screening = 0.0
    for weight, decay in SCREENING:
        screening = screening + weight * xp.exp(-decay * reduced)
    u = r / rmax
    taper = (1.0 - u) ** 3 * (1.0 + u * (3.0 + u * (6.0 + u * (10.0 + u * (15.0 + 21.0 * u)))))
    return coulomb * screening / r * xp.where(u < 1.0, taper, 0.0 * taper)


def strained(x, s):
    """Rows x (n, 3) times (1 + s) for strains s (n, 3, 3), elementwise: a matrix product would
    round the geometry to TF32."""
    return x + (x[:, :, None] * s).sum(1)


def zbl_rmax(rcov_i, rcov_j, factor, rcut, xp=torch):
    m = factor * (rcov_i + rcov_j)
    return m.clamp(max=rcut) if xp is torch else xp.minimum(m, xp.full_like(m, rcut))


def zbl_tables(species):
    """Atomic numbers and covalent radii (ASE) of the species."""
    from ase.data import atomic_numbers, covalent_radii
    z = np.array([atomic_numbers[s] for s in species], dtype=float)
    return z, np.asarray(covalent_radii[z.astype(int)], dtype=float)


def group_period(Z):
    """(group, period) in the 32-column periodic table, where the f-block has its own columns."""
    if Z <= 0 or Z > 118:
        raise ValueError(f"atomic number out of range: {Z}")
    if Z <= 2:
        return (1 if Z == 1 else 32), 1
    if Z <= 10:
        return (Z - 2 if Z <= 4 else Z + 22), 2
    if Z <= 18:
        return (Z - 10 if Z <= 12 else Z + 14), 3
    if Z <= 36:
        return (Z - 18 if Z <= 20 else Z - 4), 4
    if Z <= 54:
        return (Z - 36 if Z <= 38 else Z - 22), 5
    if Z <= 86:
        return Z - 54, 6
    return Z - 86, 7


def type_onehot(species):
    from ase.data import atomic_numbers
    out = np.zeros((len(species), N_GROUPS + N_PERIODS))
    for k, sym in enumerate(species):
        g, p = group_period(atomic_numbers[sym])
        out[k, g - 1] = out[k, N_GROUPS + p - 1] = 1.0
    return out


class Harmonics(nn.Module):
    """Real spherical harmonics of unit vectors, scaled to Y_00 = 1 (see angular)."""

    def __init__(self, lmax):
        super().__init__()
        self.lmax = int(lmax)

    def forward(self, u):
        return angular(u, self.lmax, torch)


def angular(u, lmax, xp):
    """Real spherical harmonics of the unit vectors u (n, 3) -> (n, (lmax + 1)^2), for torch or
    jax.numpy (xp): rows l = 0..lmax, in each m = -l..l, sqrt(4 pi) times the orthonormal real
    harmonics without the Condon-Shortley phase,

        Y_lm = sqrt((2 - [m = 0]) (2l + 1) (l - |m|)! / (l + |m|)!) P_l^(|m|)(z) Re/Im (x + iy)^|m|

    with Re for m >= 0 and Im for m < 0, P_l^(k) the k-th derivative of the Legendre polynomial."""
    x, y, z = u[..., 0], u[..., 1], u[..., 2]
    one = xp.ones_like(z)
    cos_part, sin_part = [one], [xp.zeros_like(z)]          # Re and Im of (x + iy)^k
    for _ in range(lmax):
        c, s = cos_part[-1], sin_part[-1]
        cos_part.append(x * c - y * s)
        sin_part.append(y * c + x * s)
    deriv = {}                                              # (l, k): P_l^(k)(z)
    for k in range(lmax + 1):
        deriv[k, k] = float(np.prod(np.arange(1, 2 * k, 2))) * one            # (2k - 1)!!
        if k < lmax:
            deriv[k + 1, k] = (2 * k + 1) * z * deriv[k, k]
        for l in range(k + 2, lmax + 1):
            deriv[l, k] = ((2 * l - 1) * z * deriv[l - 1, k] - (l + k - 1) * deriv[l - 2, k]) / (l - k)
    rows = []
    for l in range(lmax + 1):
        for m in range(-l, l + 1):
            k = abs(m)
            scale = sqrt((2 - (m == 0)) * (2 * l + 1) * factorial(l - k) / factorial(l + k))
            rows.append(scale * deriv[l, k] * (sin_part[k] if m < 0 else cos_part[k]))
    return xp.stack(rows, -1)


class RadialBasis(nn.Module):
    """Chebyshev polynomials of the scaled distance under a smooth cutoff (see radial_basis)."""

    def __init__(self, size, rcut, p=5):
        super().__init__()
        self.size, self.rcut, self.p = int(size), float(rcut), int(p)

    def forward(self, r):
        return radial_basis(r, self.rcut, self.size, self.p, torch)


def scaled_envelope(r, rcut, p, xp):
    """(t, e(t)) for t = min(r / rcut, 1); e is the envelope of radial_basis."""
    t = r / rcut
    t = xp.where(t < 1.0, t, xp.ones_like(t))
    envelope = xp.zeros_like(t)
    for j in reversed(range(p)):                            # Horner
        envelope = envelope * t + (j + 1) * (j + 2) / 2
    return t, envelope * (1.0 - t) ** 3


def radial_basis(r, rcut, size, p, xp):
    """g_k(r) = T_k(2t - 1) e(t) for k = 1..size, t = min(r / rcut, 1), for torch or jax.numpy:
    Chebyshev polynomials of the first kind under the envelope

        e(t) = (1 - t)^3 sum_{j<p} (j + 1)(j + 2) / 2 t^j,

    the polynomial of degree p + 2 with e(0) = 1 whose value and first two derivatives vanish at t = 1
    (DimeNet's polynomial envelope, Gasteiger et al. 2020), so every g_k is zero beyond the cutoff."""
    t, envelope = scaled_envelope(r, rcut, p, xp)
    s = 2.0 * t - 1.0
    cheb = [xp.ones_like(s), s]
    while len(cheb) <= size:
        cheb.append(2.0 * s * cheb[-1] - cheb[-2])
    return xp.stack(cheb[1:size + 1], -1) * envelope[..., None]


class Coupling(nn.Module):
    """A coupling table as a gather, a product and an index_add."""

    def __init__(self, table):
        super().__init__()
        self.meta, self.n_out = table.meta, len(table.meta)
        self.register_buffer("rows_a", torch.tensor(table.rows_a, dtype=torch.long))
        self.register_buffer("rows_b", torch.tensor(table.rows_b, dtype=torch.long))
        self.register_buffer("factors", torch.tensor(table.factors))
        self.register_buffer("dest", torch.tensor(table.dest, dtype=torch.long))

    def forward(self, a, b):
        # index_select: its backward is index_add, indexing would use index_put
        prod = (a.index_select(-1, self.rows_a)
                * b.index_select(-1, self.rows_b) * self.factors)
        out = torch.zeros(prod.shape[:-1] + (self.n_out,), dtype=prod.dtype, device=prod.device)
        return out.index_add_(-1, self.dest, prod)


class Products(nn.Module):
    """Products of order 2, 3, 4 of a base: (A A), (A A) A and (A A)(A A).

    The pair products take every l <= lmax of the base, the higher orders l <= lmax_feat.
    """

    def __init__(self, base, lmax, lmax_feat, product_order, Lmax):
        super().__init__()
        self.meta, mods = {1: base}, {}
        if product_order >= 2:
            t2 = couple(base, base, lmax, Lmax[2], symmetric=True)
            self.meta[2], mods["c2"] = t2.meta, Coupling(t2)
        if product_order >= 3:
            t3 = couple(self.meta[2], base, lmax_feat, Lmax[3])
            self.meta[3], mods["c3"] = t3.meta, Coupling(t3)
        if product_order >= 4:
            t4 = couple(self.meta[2], self.meta[2], lmax_feat, Lmax[4], symmetric=True)
            self.meta[4], mods["c4"] = t4.meta, Coupling(t4)
        self.couplings = nn.ModuleDict(mods)

    def forward(self, A):
        out, c = {1: A}, self.couplings
        if "c2" in c:
            out[2] = c["c2"](A, A)
        if "c3" in c:
            out[3] = c["c3"](out[2], A)
        if "c4" in c:
            out[4] = c["c4"](out[2], out[2])
        return out


def _radial_weights(lmax, n_basis, n_out):
    """weights[l, k, n] of the radial functions R_ln = sum_k weights[l, k, n] g_k, N(0, 1 / n_basis)."""
    return nn.Parameter(torch.randn(lmax + 1, n_basis, n_out) / sqrt(n_basis))


def _tapered_weights(chan, n_basis):
    """One weights[k, n] of _radial_weights per degree, for chan[l] channels each."""
    return nn.ParameterList([nn.Parameter(torch.randn(n_basis, c) / sqrt(n_basis)) for c in chan])


class Reduce(nn.Module):
    """The messages: for every degree l <= lmax_out, one linear map of all product functions of
    degree l, of every order, with their n_in channels each, onto n_out channels, the same for
    every component m.  mix<l>[function, in, out]; functions in the order of prod.meta.  `partial`:
    a degree without functions gives zeros."""

    def __init__(self, prod, n_in, lmax_out, n_out, partial=False):
        super().__init__()
        self.n_out, self.lmax_out, self.n_lm = int(n_out), int(lmax_out), (lmax_out + 1) ** 2
        self.parts = {}                          # l -> [(order, number of functions)]
        for l in range(self.lmax_out + 1):
            self.parts[l] = []
            for order, rows in prod.meta.items():
                picked = [f.rows for f in functions(rows) if f.l == l]
                if picked:
                    self.register_buffer(f"pick{l}_{order}", torch.tensor(np.stack(picked)).reshape(-1),
                                         persistent=False)
                    self.parts[l].append((order, len(picked)))
            n_fun = sum(n for _, n in self.parts[l])
            if not n_fun:
                if not partial:
                    raise ValueError(f"no product function of degree {l} for the messages")
                continue
            setattr(self, f"mix{l}", nn.Parameter(torch.randn(n_fun, n_in, self.n_out) / sqrt(n_fun * n_in)))
        self.orders = sorted({o for parts in self.parts.values() for o, _ in parts})

    def per_function(self, order):
        """(rows of the product, message slot l^2 + l + m, weights [row, in, out]) of one order."""
        rows, slots, weights = [], [], []
        for l, parts in self.parts.items():
            first = 0
            for o, n in parts:
                if o == order:
                    pick = getattr(self, f"pick{l}_{o}").reshape(n, 2 * l + 1)
                    w = getattr(self, f"mix{l}")[first:first + n]
                    rows.append(pick.reshape(-1))
                    slots.append((l * l + torch.arange(2 * l + 1, device=pick.device)).repeat(n))
                    weights.append(w.repeat_interleave(2 * l + 1, dim=0))
                first += n
        return torch.cat(rows), torch.cat(slots), torch.cat(weights)

    def forward(self, blocks):
        a = blocks[1].shape[0]
        out = []
        for l, parts in self.parts.items():
            y, first = 0, 0
            for o, n in parts:
                x = blocks[o].index_select(2, getattr(self, f"pick{l}_{o}"))      # (atoms, in, n (2l+1))
                x = x.reshape(a, -1, n, 2 * l + 1).permute(0, 3, 2, 1).reshape(a, 2 * l + 1, -1)
                w = getattr(self, f"mix{l}")[first:first + n]
                y = y + torch.matmul(x, w.reshape(-1, self.n_out))
                first += n
            if isinstance(y, int):
                y = blocks[1].new_zeros(a, 2 * l + 1, self.n_out)
            out.append(y.transpose(1, 2))                                        # (atoms, out, 2l+1)
        return torch.cat(out, 2)


class ILMlip(nn.Module):
    """Descriptor and one MLP with a hidden bias and an output vector per element.

    The first layer's harmonics (lmax) can go beyond every other equivariant feature (lmax_feat), as
    ACE lowers l with the body order: only its pair products see l > lmax_feat.  The readout takes the
    invariants of every layer."""

    def __init__(self, species, rcut=5.0, lmax=3, lmax_feat=1, n_basis=8, n_channels=12,
                 product_order=3, lmax_ind=1, n_layers=2, n_message=12, hidden=(64,), embed_dim=8,
                 atomic_energies=None, energy_scale=1.0, zbl=False, zbl_factor=0.65,
                 rcut_elem=None,
                 charge_spin=False, max_charge=10, max_spin=11, taper=None, gram_rank=16, bispec_rank=8, stress=True,
                 bound=None, bond_scale=0.05, feat_bound=None, zbl_switch=None,
                 kernels="torch", dtype=torch.float64):
        super().__init__()
        if kernels not in KERNELS:
            raise ValueError(f"kernels: one of {KERNELS}, not {kernels!r}")
        if not hidden:
            raise ValueError("the readout needs a hidden layer")
        if not lmax_ind <= lmax_feat <= lmax:
            raise ValueError(f"need lmax_ind <= lmax_feat <= lmax, got {lmax_ind}, {lmax_feat}, {lmax}")
        if n_layers < 2:
            raise ValueError(f"n_layers >= 2: the model needs a message layer, got {n_layers}")
        if taper is not None:
            taper = [int(c) for c in taper]
            if len(taper) != lmax + 1 or min(taper) < 1 or any(a < b for a, b in zip(taper, taper[1:])):
                raise ValueError(f"taper: {lmax + 1} non-increasing channel numbers, got {taper}")
            n_channels = taper[0]
        if zbl_switch and not zbl:
            raise ValueError("zbl_switch switches the learned energy by the ZBL energy: it needs zbl=True")
        if any(v is not None and not v > 0 for v in (bound, feat_bound, zbl_switch)) or not bond_scale > 0:
            raise ValueError("bound, bond_scale, feat_bound and zbl_switch are positive numbers (or None)")
        self.chan = taper or [n_channels] * (lmax + 1)         # channels of each degree of the first layer
        if min(self.chan[:lmax_feat + 1]) < n_message:
            raise ValueError(f"the message layer shares the first layer's radial functions: its degrees <= {lmax_feat} "
                             f"need >= {n_message} channels")
        self.species, self.n_species = list(species), len(species)
        self.rcut, self.lmax, self.n_layers = float(rcut), int(lmax), int(n_layers)
        self.lmax_feat, self.n_channels, self.lmax_ind = int(lmax_feat), int(n_channels), int(lmax_ind)
        self.taper, self.n_message = taper, int(n_message)
        # channel group g holds the channels [taper[g + 1], taper[g]) and the degrees <= g
        self.groups = [(lmax, 0, n_channels)] if taper is None else [
            (g, taper[g + 1] if g < lmax else 0, taper[g]) for g in range(lmax + 1)
            if taper[g] > (taper[g + 1] if g < lmax else 0)]
        Lmax = {2: lmax_feat, 3: max(lmax_ind, 1), 4: 0}

        self.gbasis = RadialBasis(n_basis, rcut)
        self.angular = Harmonics(lmax)
        self.edge_weights = (_radial_weights(lmax, n_basis, n_channels) if taper is None
                             else _tapered_weights(taper, n_basis))
        self.register_buffer("type_feat", torch.tensor(type_onehot(self.species), dtype=dtype))
        self.embed_proj = nn.Linear(N_GROUPS + N_PERIODS, embed_dim, bias=False)
        self.chem0 = nn.Linear(embed_dim, n_channels, bias=False)
        self.film_weights = (_radial_weights(lmax, n_basis, n_channels) if taper is None
                             else _tapered_weights(taper, n_basis))
        self.film = nn.Sequential(nn.Linear(2 * embed_dim, embed_dim), nn.SiLU(),
                                  nn.Linear(embed_dim, 2 * sum(self.chan)))
        nn.init.zeros_(self.film[2].weight)
        nn.init.zeros_(self.film[2].bias)
        self.register_buffer("parity", torch.tensor([(-1.0) ** l for l in range(lmax_ind + 1)
                                                     for _ in range(2 * l + 1)]), persistent=False)
        if taper is None:
            self.prod0 = Products(uncoupled(lmax), lmax, lmax_feat, product_order, Lmax)
        else:
            self.prod0 = nn.ModuleList([Products(uncoupled(g), g, lmax_feat, product_order, Lmax)
                                        for g, _, _ in self.groups])

        self.reduces, self.ycouple, self.prods = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        prev, n_prev = self.prod0, n_channels
        for t in range(1, self.n_layers):
            if t == 1 and taper is not None:
                self.reduces.append(nn.ModuleList([Reduce(P, hi - lo, lmax_ind, n_message, partial=True)
                                                   for P, (_, lo, hi) in zip(self.prod0, self.groups)]))
            else:
                self.reduces.append(Reduce(prev, n_prev, lmax_ind, n_message))
            c = couple(uncoupled(lmax_feat), uncoupled(lmax_ind), lmax_feat, lmax_feat)
            self.ycouple.append(Coupling(c))
            if t < self.n_layers - 1:       # the last layer's bond sum goes to the Gram readout
                prev = Products(c.meta, lmax_feat, lmax_feat, product_order, Lmax)
                self.prods.append(prev)
                n_prev = n_message

        # the L = 0 rows of every product, read out
        self.n_features, self.inv_keys, self.group_keys = 0, [], []

        def read(name, keys, key, P, n_ch):
            for o, m in P.meta.items():
                rows = [i for i, r in enumerate(m) if r[0] == 0]
                if rows:
                    self.register_buffer(f"{name}_{o}", torch.tensor(rows, dtype=torch.long))
                    keys.append((key, o))
                    self.n_features += n_ch * len(rows)

        if taper is None:
            read("inv0", self.inv_keys, 0, self.prod0, n_channels)
        else:
            for b, (P, (_, lo, hi)) in enumerate(zip(self.prod0, self.groups)):
                read(f"inv0g{b}", self.group_keys, b, P, hi - lo)
        for t, P in enumerate(self.prods, 1):
            read(f"inv{t}", self.inv_keys, t, P, n_message)
        self._init_gram(c.meta, n_message, gram_rank, bispec_rank)

        layers, d = [], self.n_features
        for h in hidden:
            layers += [nn.Linear(d, h), nn.SiLU()]
            d = h
        self.readout = nn.Sequential(*layers)
        self.elem_bias = nn.Parameter(torch.zeros(self.n_species, hidden[0]))
        self.head = nn.Parameter(torch.zeros(self.n_species, d))
        self.atomic_energies = nn.Parameter(torch.zeros(self.n_species) if atomic_energies is None
                                            else torch.tensor(atomic_energies, dtype=dtype))
        self.register_buffer("energy_scale", torch.tensor(float(energy_scale), dtype=dtype))
        self.register_buffer("feat_mean", torch.zeros(self.n_features, dtype=dtype))
        self.register_buffer("feat_std", torch.ones(self.n_features, dtype=dtype))

        # bounds: kappa (bound), a0 (bond_scale), tau (feat_bound), z_c in eV (zbl_switch); None = off
        self.bound, self.bond_scale, self.feat_bound, self.zbl_switch = bound, float(bond_scale), feat_bound, zbl_switch
        self.kernels = kernels     # how the descriptor runs; not part of the checkpoint
        if bound:
            self.register_buffer("bond_rms", torch.ones(self.n_layers, dtype=dtype))
        self._tap = None
        self.zbl, self.zbl_factor = bool(zbl), float(zbl_factor)
        if self.zbl:
            z, rcov = zbl_tables(self.species)
            self.register_buffer("zbl_z", torch.tensor(z, dtype=dtype), persistent=False)
            self.register_buffer("zbl_rcov", torch.tensor(rcov, dtype=dtype), persistent=False)
        from . import vcut
        vcut.attach(self, rcut_elem)

        # rows: charge + max_charge, multiplicity - 1
        self.charge_spin = bool(charge_spin)
        self.stress = bool(stress)        # False: not trained on stresses, refuses them
        self.max_charge, self.max_spin = int(max_charge), int(max_spin)
        if self.charge_spin:
            nq, h1 = 2 * self.max_charge + 1, self.readout[0].out_features
            self.charge_embed = nn.Parameter(torch.zeros(nq, embed_dim))
            self.spin_embed = nn.Parameter(torch.zeros(self.max_spin, embed_dim))
            self.charge_bias = nn.Parameter(torch.zeros(nq, h1))
            self.spin_bias = nn.Parameter(torch.zeros(self.max_spin, h1))
        self.to(dtype)

    def _init_gram(self, meta, n_message, gram_rank, bispec_rank):
        """The last layer's readout features: its scalars, the Gram matrices of its degrees (as
        projected onto gram_rank channels) and the order-3 invariants of the projected base."""
        r, rb = min(int(gram_rank), n_message), int(bispec_rank)
        self.gram_proj, self.bispec_proj = nn.ParameterList(), nn.ParameterList()
        for l in range(self.lmax_feat + 1):
            rows = np.stack([f.rows for f in functions(meta) if f.l == l])         # (functions, 2l+1)
            self.register_buffer(f"gram_rows{l}", torch.tensor(rows.reshape(-1)), persistent=False)
            n_in = len(rows) * n_message
            self.gram_proj.append(nn.Parameter(torch.randn(n_in, r) / sqrt(n_in)))
            self.bispec_proj.append(nn.Parameter(torch.randn(n_in, rb) / sqrt(n_in)))
            if l == 0:
                self.n_features += n_in
        self.register_buffer("gram_pairs", torch.triu_indices(r, r), persistent=False)
        self.bispec = Products(uncoupled(self.lmax_feat), self.lmax_feat, self.lmax_feat, 3,
                               {2: self.lmax_feat, 3: 0, 4: 0})
        self.n_features += (self.lmax_feat + 1) * r * (r + 1) // 2 + rb * len(self.bispec.meta[3])

    def _gram(self, A):
        a = A.shape[0]
        out, Q = [A.index_select(2, self.gram_rows0).reshape(a, -1)], []
        for l in range(self.lmax_feat + 1):
            x = A.index_select(2, getattr(self, f"gram_rows{l}")).reshape(a, A.shape[1], -1, 2 * l + 1)
            x = x.permute(0, 3, 2, 1).reshape(a, 2 * l + 1, -1)                      # (atoms, m, function * channel)
            P = torch.matmul(x, self.gram_proj[l])
            out.append((P[:, :, self.gram_pairs[0]] * P[:, :, self.gram_pairs[1]]).sum(1))
            Q.append(torch.matmul(x, self.bispec_proj[l]).transpose(1, 2))
        return out + [self.bispec(torch.cat(Q, 2))[3].reshape(a, -1)]

    def _species_z(self):
        return self.embed_proj(self.type_feat)

    def _embed(self, batch):
        z = self.embed_proj(self.type_feat[batch["mu"]])
        if self.charge_spin:
            q, s = self._state_rows(batch)
            z = z + (self.charge_embed[q] + self.spin_embed[s])[batch["batch"]]
        return z

    def _film_table(self):
        """(gamma, beta) of every ordered pair (centre species, neighbour species), concatenated over
        the degrees and channels of the first layer: (n_species^2, 2 sum_l channels_l).  From the
        element embeddings alone, so with charge_spin too it does not depend on the structure
        (fold_charge_spin keeps them in film_z)."""
        z, S = (self.film_z if hasattr(self, "film_z") else self._species_z()), self.n_species
        return self.film(torch.cat([z[:, None].expand(S, S, -1), z[None].expand(S, S, -1)], -1).reshape(S * S, -1))

    @staticmethod
    def _radial(weights, g):
        """R[l] = sum_k weights[l][k, n] g[e, k] -> (edges, n) of every degree l."""
        return torch.matmul(g, weights) if isinstance(weights, torch.Tensor) else [torch.matmul(g, w) for w in weights]

    def _radials(self, g):
        return self._radial(self.edge_weights, g), self._radial(self.film_weights, g)

    def _object(self, R, B, chem, table, pair, degrees=None):
        """The first layer's radial functions of the first `degrees` degrees, (edges, channels_l) each:
        (1 + gamma_ln) R_ln + beta_ln B_ln of the pair's table row, times the neighbour's chemistry."""
        film, total = table.index_select(0, pair), sum(self.chan)
        out, off = [], 0
        for l, c in enumerate(self.chan[:degrees]):
            r = R[l] * (1.0 + film[:, off:off + c]) + film[:, total + off:total + off + c] * B[l]
            out.append(r * chem[:, :c])
            off += c
        return out

    def _folded(self):
        """Whether the first layer's radial functions come from per-pair weights (_keyed_weights): only
        under torch.compile, which fuses the gather into the sum over the basis (eager would hold
        (edges, n_basis, channels)), and without weight gradients (the gradient of a gathered weight
        is a scatter over the edges)."""
        return torch.compiler.is_compiling() and not any(p.requires_grad for p in self.parameters())

    def _keyed_weights(self, chem, table):
        """sum_k g_k F_l[key, k, n] = the first layer's radial function of degree l times the
        neighbour's chemistry (chem, one row per species), per ordered pair (key centre * n_species +
        neighbour).  One (keys, n_basis, channels_l) per degree."""
        out, off, total = [], 0, sum(self.chan)
        chem = chem.repeat(self.n_species, 1)                          # row key -> the neighbour's
        for l, c in enumerate(self.chan):
            W = self.edge_weights[l]
            gamma, beta = table[:, None, off:off + c], table[:, None, total + off:total + off + c]
            W = W * (1.0 + gamma) + self.film_weights[l] * beta
            out.append(W * chem[:, None, :c])
            off += c
        return out

    @staticmethod
    def _keyed(F, g, key):
        return (g[:, :, None] * F.index_select(0, key)).sum(1)

    def _order1_mix(self, l):
        """The order-1 rows of the first message map of degree l, (channels_l, n_message): the rows of
        every channel group that holds degree l, in channel order."""
        red = self.reduces[0]
        if not isinstance(red, nn.ModuleList):
            return getattr(red, f"mix{l}")[0]
        parts = sorted(((lo, getattr(r, f"mix{l}")[0]) for r, (gmax, lo, hi) in zip(red, self.groups) if gmax >= l),
                       key=lambda p: p[0])
        return torch.cat([w for _, w in parts], 0)

    @staticmethod
    def _group(R, Y, lmax, lo, hi):
        """The per-edge object of the degrees <= lmax and the channels [lo, hi): (edges, hi - lo, (lmax + 1)^2)."""
        return torch.cat([R[l][:, lo:hi, None] * Y[:, None, l * l:(l + 1) ** 2] for l in range(lmax + 1)], 2)

    def _messages(self, t, blocks):
        red = self.reduces[t]
        if isinstance(red, nn.ModuleList):
            return sum(r(b) for r, b in zip(red, blocks))
        return red(blocks)

    def _backflow(self, g, Y, R, B, chem, table, rev, F=None):
        """Order-1 part of the first message map on the reverse edge's first-layer object (centre j,
        neighbour i): same r, so same R and B (the cutoff is symmetric), Y(-u) = (-1)^l Y(u), and the
        chemistry of i.  The map acts on the radial functions before they meet Y, or on F with per-pair
        weights.  rev: the table row of the reverse pair."""
        n = self.lmax_ind + 1
        if F is not None:
            Q = [self._keyed(torch.matmul(F[l], self._order1_mix(l)), g, rev) for l in range(n)]
        else:
            Q = [torch.matmul(o, self._order1_mix(l)) for l, o in enumerate(self._object(R, B, chem, table, rev, n))]
        return torch.cat([q[:, :, None] * Y[:, None, l * l:(l + 1) ** 2] for l, q in enumerate(Q)], 2) * self.parity

    def bounded(self):
        """Whether any of the bounds (bound, feat_bound, zbl_switch) is on."""
        return bool(self.bound or self.feat_bound or self.zbl_switch)

    def _gate(self, t, A):
        """B1: the bond sum of layer t (a tensor, or for t = 0 the list of the taper groups) times one
        invariant factor per atom, f = a0 / sigma_t (1 + rho^2 / kappa^2)^(-1/2), rho^2 = mean(A^2) /
        sigma_t^2, so the RMS component of f A stays below a0 kappa at any density.  Returns (f A, f);
        (A, None) off."""
        if not self.bound:
            return A, None
        parts = A if isinstance(A, list) else [A]
        sigma = self.bond_rms[t]
        m = sum((a * a).sum((1, 2)) for a in parts) / sum(a.shape[1] * a.shape[2] for a in parts)
        f = (self.bond_scale / sigma) * torch.rsqrt(1.0 + m / (sigma * self.bound) ** 2)
        if self._tap is not None:
            self._tap.append((t, m.detach(), f.detach()))
        parts = [a * f[:, None, None] for a in parts]
        return (parts if isinstance(A, list) else parts[0]), f

    @torch.no_grad()
    def gate_factors(self, batch):
        """The gate g in (0, 1] of every atom, one tensor per layer ([] without bound)."""
        self._tap = []
        try:
            self.descriptor(batch)
            return [f * self.bond_rms[t] / self.bond_scale for t, _, f in self._tap]
        finally:
            self._tap = None

    def _whiten(self, x):
        """The features in units of their spread; B2: x -> x (1 + (x / tau)^2)^(-1/2) bounds every one by tau."""
        x = (x - self.feat_mean) / self.feat_std
        return x if not self.feat_bound else x * torch.rsqrt(1.0 + (x / self.feat_bound) ** 2)

    @torch.no_grad()
    def whitened(self, batch):
        """x~ of every atom: the features in units of their spread, before the squash."""
        return (self.descriptor(batch) - self.feat_mean) / self.feat_std

    def descriptor(self, batch):
        """Invariant features (atoms, n_features)."""
        pos, ind_i, ind_j = batch["positions"], batch["ind_i"], batch["ind_j"]
        vec = pos[ind_j] - pos[ind_i] + batch["shift"]
        r = vec.norm(dim=1).clamp(min=1e-8)
        rhat = vec / r[:, None]
        r = r if self.rcut_elem is None else r * self.vc_scale[batch["mu"][ind_i], batch["mu"][ind_j]]
        g, Y = self.gbasis(r), self.angular(rhat)
        z, mu, S = self._embed(batch), batch["mu"], self.n_species
        table = self._film_table()
        n, chem0 = batch["n_atoms"], self.chem0(z)

        # per-pair weights need a chemistry per species; charge_spin makes it per structure
        fold = not self.charge_spin and self.kernels == "torch" and self._folded()
        F = self._keyed_weights(self.chem0(self._species_z()), table) if fold else None
        rad = film_rad = None
        if F is not None:
            R = [self._keyed(f, g, mu[ind_i] * S + mu[ind_j]) for f in F]
        else:
            rad, film_rad = self._radials(g)
            R = self._object(rad, film_rad, chem0[ind_j], table, mu[ind_i] * S + mu[ind_j])
        if self.kernels != "torch":
            return fused_torch.descriptor(self, batch, Y, R, rad, film_rad, chem0, table)
        A = [torch.zeros(n, hi - lo, (l + 1) ** 2, dtype=pos.dtype, device=pos.device)
             .index_add_(0, ind_i, self._group(R, Y, l, lo, hi)) for l, lo, hi in self.groups]
        A, f0 = self._gate(0, A)
        layers = [[P(a) for P, a in zip(self.prod0, A)] if self.taper else self.prod0(A[0])]
        Y = Y[:, :(self.lmax_feat + 1) ** 2]
        RY = self._group(R, Y, self.lmax_feat, 0, self.n_message)
        for t in range(self.n_layers - 1):
            I = self._messages(t, layers[-1])[ind_j]
            if t == 0:
                back = self._backflow(g, Y, rad, film_rad, chem0[ind_i], table, mu[ind_j] * S + mu[ind_i], F)
                I = I - (back if f0 is None else back * f0[ind_j][:, None, None])    # the sender's gate
            per_bond = self.ycouple[t](RY, I)
            A = torch.zeros(n, per_bond.shape[1], per_bond.shape[2], dtype=pos.dtype,
                            device=pos.device).index_add_(0, ind_i, per_bond)
            A, _ = self._gate(t + 1, A)
            layers.append(self.prods[t](A) if t < len(self.prods) else A)
        feats = [layers[0][b][o].index_select(2, getattr(self, f"inv0g{b}_{o}")).reshape(n, -1)
                 for b, o in self.group_keys]
        feats += [layers[t][o].index_select(2, getattr(self, f"inv{t}_{o}")).reshape(n, -1)
                  for t, o in self.inv_keys]
        return torch.cat(feats + self._gram(layers[-1]), dim=1)

    def zbl_atoms(self, batch):
        pos, ind_i, ind_j, mu = batch["positions"], batch["ind_i"], batch["ind_j"], batch["mu"]
        r = (pos[ind_j] - pos[ind_i] + batch["shift"]).norm(dim=1).clamp(min=1e-8)
        mi, mj = mu[ind_i], mu[ind_j]
        rmax = zbl_rmax(self.zbl_rcov[mi], self.zbl_rcov[mj], self.zbl_factor, self.rcut)
        v = 0.5 * repulsion(r, self.zbl_z[mi], self.zbl_z[mj], rmax)
        return torch.zeros(batch["n_atoms"], dtype=v.dtype, device=v.device).index_add_(0, ind_i, v)

    def forward(self, batch, compute_forces=True, compute_stress=False):
        """Energies, forces and the stress (the strain derivative of the same energy)."""
        strain = None
        if compute_stress:
            if not self.stress:
                raise ValueError(NO_STRESS)
            strain = torch.zeros(batch["n_struct"], 3, 3, dtype=batch["positions"].dtype,
                                 device=batch["positions"].device, requires_grad=True)
            sym = 0.5 * (strain + strain.transpose(1, 2))
            batch = dict(batch)
            at_struct = batch["batch"]
            batch["positions"] = strained(batch["positions"], sym.index_select(0, at_struct))
            batch["shift"] = strained(batch["shift"],
                                      sym.index_select(0, at_struct.index_select(0, batch["ind_i"])))

        x = self._whiten(self.descriptor(batch))
        e_atom = (self._readout(x, batch) * self.head[batch["mu"]]).sum(-1) * self.energy_scale
        z = self.zbl_atoms(batch) if self.zbl else None
        if self.zbl_switch:
            e_atom = e_atom * core_switch(z, self.zbl_switch)         # B3: no learned energy in the core
        # added and summed in float64: the E0s dwarf the network's part of the energy
        e_atom = e_atom.double() + self.atomic_energies[batch["mu"]].double()
        if self.zbl:
            e_atom = e_atom + z.double()
        E = torch.zeros(batch["n_struct"], dtype=torch.float64,
                        device=e_atom.device).index_add(0, batch["batch"], e_atom)

        out = {"energy": E, "atomic_energy": e_atom}
        if compute_forces or compute_stress:
            wrt = [batch["positions"]] + ([strain] if compute_stress else [])
            grads = torch.autograd.grad(E.sum(), wrt, create_graph=self.training)
            if compute_forces:
                out["forces"] = -grads[0]
            if compute_stress:
                sig = grads[-1] / batch["volume"].clamp(min=1e-12)[:, None, None]
                sig = 0.5 * (sig + sig.transpose(1, 2))
                out["stress"] = torch.stack([sig[:, 0, 0], sig[:, 1, 1], sig[:, 2, 2],
                                             sig[:, 1, 2], sig[:, 0, 2], sig[:, 0, 1]], dim=1)
        return out

    def _state_rows(self, batch):
        q = batch["charge"].long() + self.max_charge
        s = batch["spin"].long() - 1
        return q.clamp(0, 2 * self.max_charge), s.clamp(0, self.max_spin - 1)

    def _readout(self, x, batch):
        h = self.readout[0](x) + self.elem_bias[batch["mu"]]
        if self.charge_spin:
            q, s = self._state_rows(batch)
            h = h + (self.charge_bias[q] + self.spin_bias[s])[batch["batch"]]
        return self.readout[1:](h)

    @torch.no_grad()
    def fit_feature_normalisation(self, batches):
        """With bound, first the RMS component sigma_t of every bond sum (layer by layer, the later ones
        behind the earlier gates), then the mean and standard deviation of every feature over the atoms of
        the batches, summed in float64: the squares of the highest-order features overflow float32."""
        for t in range(self.n_layers if self.bound else 0):
            self.bond_rms[t], total, count = 1.0, 0.0, 0
            for b in batches:
                self._tap = []
                try:
                    self.descriptor(b)
                    m = next(m for k, m, _ in self._tap if k == t)
                finally:
                    self._tap = None
                total, count = total + float(m.double().sum()), count + len(m)
            self.bond_rms[t] = max(total / count, 1e-30) ** 0.5
        f = torch.cat([self.descriptor(b).detach() for b in batches], 0)
        f = f[torch.isfinite(f).all(1)]
        if len(f) < 2:
            raise ValueError("feature normalisation needs at least two atoms with finite features")
        chunks = f.split(4096)
        mean = sum(c.double().sum(0) for c in chunks) / len(f)
        var = sum((c.double() - mean).pow(2).sum(0) for c in chunks) / (len(f) - 1)
        top = torch.finfo(self.feat_std.dtype).max
        self.feat_mean.copy_(mean)
        self.feat_std.copy_(var.sqrt().clamp(min=1e-6, max=top))

    def n_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def refuse_charge_spin(state, charge, spin):
    """Raises unless the charge and spin asked for (None: not asked) fit a model already without
    charge/spin inputs: state = the (charge, spin) it was folded for, None if it never had them."""
    if charge is None and spin is None:
        return
    if state is None:
        raise ValueError("charge and spin are inputs of the OMOL models only; this model has none")
    want = (state[0] if charge is None else int(charge), state[1] if spin is None else int(spin))
    if want != tuple(state):
        raise ValueError(f"this model is for charge {state[0]}, spin {state[1]}, not {want[0]}, {want[1]}")


def fold_charge_spin(model, charge=None, spin=None):
    """A plain model equal to `model` for structures of this total charge and multiplicity (default 0
    and 1).  A model without charge/spin inputs refuses both.

    Exact: every element's one-hot code has one group bit, so a vector added to all chemical embeddings
    moves into the group columns of embed_proj."""
    if not model.charge_spin:
        refuse_charge_spin(getattr(model, "folded_charge_spin", None), charge, spin)
        return model
    charge, spin = (0 if charge is None else charge), (1 if spin is None else spin)
    q, s = int(charge) + model.max_charge, int(spin) - 1
    if not (0 <= q <= 2 * model.max_charge and 0 <= s < model.max_spin):
        raise ValueError(f"charge {charge} / multiplicity {spin} outside the model's tables "
                         f"(|Q| <= {model.max_charge}, 2S+1 <= {model.max_spin})")
    m = copy.deepcopy(model)
    with torch.no_grad():
        m.register_buffer("film_z", m._species_z().detach().clone())
        shift = m.charge_embed[q] + m.spin_embed[s]
        m.embed_proj.weight[:, :N_GROUPS] += shift[:, None]
        m.readout[0].bias += m.charge_bias[q] + m.spin_bias[s]
    for name in ("charge_embed", "spin_embed", "charge_bias", "spin_bias"):
        delattr(m, name)
    m.charge_spin = False
    m.folded_charge_spin = (int(charge), int(spin))
    return m


DTYPES = ("tf32", "float32", "float64")


def trained_dtype(ck):
    """"float64" for a checkpoint (as load returns it) trained in double precision (--dtype float64), else "tf32"."""
    args = ck.get("args") or (ck.get("finetune") or {}).get("args") or {}
    return "float64" if args.get("dtype") == "float64" else "tf32"


def set_dtype(name):
    """The --dtype of train and finetune: switches torch's TF32 matrix products on for "tf32", off
    otherwise, and returns the torch dtype of the weights and geometry."""
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = name == "tf32"
    return torch.float64 if name == "float64" else torch.float32


def load(path, device="cpu", untrained=False, kernels="torch"):
    """(model, checkpoint) of a training or fine-tuning checkpoint, its descriptor run by `kernels`."""
    ck = torch.load(path, map_location=device, weights_only=True)
    sd = ck.get("model", ck)
    if not isinstance(sd, dict) or not any(k.split(".")[0] == "edge_weights" for k in sd):
        raise ValueError(f"{path}: not an ilmlip checkpoint (keys {list(ck)[:8]})")
    known = set(inspect.signature(ILMlip.__init__).parameters) - {"self", "species", "kernels"}
    kw = {k: v for k, v in (ck.get("hparams") or {}).items() if k in known}
    model = ILMlip(ck["species"], kernels=kernels, **kw)
    if not untrained:
        exact = {k: v.clone() for k, v in model.state_dict().items() if k.endswith((".factors", ".rows_a", ".rows_b", ".dest"))}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing or unexpected:
            raise ValueError(f"{path}: missing {missing}, unexpected {unexpected}")
        saved = model.state_dict()
        for k, v in exact.items():
            same = (torch.allclose(saved[k], v, rtol=1e-6, atol=1e-7) if v.is_floating_point()
                    else torch.equal(saved[k], v))
            if not same:
                raise ValueError(f"{path}: {k} is not the table this package builds")
            saved[k].copy_(v)       # a float32 checkpoint holds the tables rounded
    return model.to(device).eval(), ck


_VESIN = {}


def neighbours(positions, cell, pbc, cutoff):
    """(i, j, S) of all pairs within cutoff, i sorted; the bond is pos[j] - pos[i] + S @ cell."""
    from vesin import NeighborList
    nl = _VESIN.get(cutoff)
    if nl is None:
        nl = _VESIN[cutoff] = NeighborList(cutoff=float(cutoff), full_list=True)
    i, j, S = nl.compute(points=np.asarray(positions, float), box=np.asarray(cell, float),
                         periodic=[bool(p) for p in pbc], quantities="ijS")
    order = np.argsort(i, kind="stable")
    return (np.asarray(i, np.int64)[order], np.asarray(j, np.int64)[order],
            np.asarray(S, float)[order])


def graph(atoms, rcut, species):
    """The arrays of one structure; labels NaN where absent."""
    index = {s: k for k, s in enumerate(species)}
    cell = np.asarray(atoms.cell, float).reshape(3, 3)
    pbc = np.asarray(atoms.pbc, bool) & cell.any(1)      # periodic along the cell vectors there are
    symbols = atoms.get_chemical_symbols()
    missing = sorted(set(symbols) - set(index))
    if missing:
        raise ValueError(f"elements not in this model: {missing}")
    i, j, S = neighbours(atoms.get_positions(), cell, pbc, rcut)
    mu = np.array([index[s] for s in symbols])
    return dict(positions=atoms.get_positions(), mu=mu,
                ind_i=i, ind_j=j, shift=S @ cell, n_atoms=len(atoms),
                volume=abs(float(np.linalg.det(cell))) if np.any(pbc) else 0.0,
                energy=np.nan, forces=np.full((len(atoms), 3), np.nan), stress=np.full(6, np.nan))


def collate(graphs, device="cpu", dtype=torch.float64):
    """Several graphs as one batch (a disconnected graph)."""
    cat = lambda key: np.concatenate([g[key] for g in graphs])
    off = np.cumsum([0] + [g["n_atoms"] for g in graphs])[:-1]
    t = lambda x, d=dtype: torch.tensor(np.asarray(x), dtype=d, device=device)
    return dict(
        positions=t(cat("positions")).requires_grad_(True),
        mu=t(cat("mu"), torch.long),
        ind_i=t(np.concatenate([g["ind_i"] + o for g, o in zip(graphs, off)]), torch.long),
        ind_j=t(np.concatenate([g["ind_j"] + o for g, o in zip(graphs, off)]), torch.long),
        shift=t(cat("shift")),
        batch=t(np.concatenate([np.full(g["n_atoms"], k) for k, g in enumerate(graphs)]), torch.long),
        n_atoms=int(sum(g["n_atoms"] for g in graphs)), n_struct=len(graphs),
        energy=t([g["energy"] for g in graphs], torch.float64), forces=t(cat("forces")),
        n_atoms_per_struct=t([g["n_atoms"] for g in graphs]),
        volume=t([g["volume"] for g in graphs]), stress=t([g["stress"] for g in graphs]),
        charge=t([g.get("charge", 0) for g in graphs], torch.long),
        spin=t([g.get("spin", 1) for g in graphs], torch.long))
