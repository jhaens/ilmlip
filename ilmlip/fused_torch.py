"""The descriptor of ILMlip with its bond sums and products as cuEquivariance segmented polynomials.

kernels="fused" runs them as CUDA kernels, "fused-reference" as the same polynomials in plain torch.
The polynomials are those of fused.py: the bond sums, the product stacks and the bispectrum products.
The rest (distances, harmonics, radial functions, FiLM, gates, back-flow, message and Gram matrix
products, readout) runs in torch.  Weights and tables are read at every call, so energies, forces and
their gradients equal the torch path; the kernels have their own derivatives, second ones included.
Operands are (batch, rows * channels), rows outermost (the model's (atoms, channels, rows) transposed).
"""
import weakref

import numpy as np
import torch

from . import fused

KERNELS = ("torch", "fused", "fused-reference")
_ENGINES = weakref.WeakKeyDictionary()


def available():
    """True when the fused CUDA kernels can run here."""
    try:
        import cuequivariance_ops_torch  # noqa: F401
        import cuequivariance_torch  # noqa: F401
    except Exception:
        return False
    return torch.cuda.is_available()


def require(kernels, device):
    """Raises unless these kernels run on the device."""
    if kernels not in KERNELS:
        raise ValueError(f"kernels must be one of {KERNELS}, got {kernels!r}")
    if kernels == "fused" and not (torch.device(device).type == "cuda" and available()):
        raise RuntimeError("kernels='fused' needs a CUDA device and the packages cuequivariance-torch and "
                           "cuequivariance-ops-torch-cu12; kernels='fused-reference' runs the same polynomials "
                           "in plain torch anywhere")
    if kernels == "fused-reference":
        try:
            import cuequivariance_torch  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("kernels='fused-reference' needs the package cuequivariance-torch") from exc


def choose(kernels, device, log=print):
    """kernels, with "auto" resolved: "fused" on a CUDA device with the cuEquivariance packages, else "torch"."""
    if kernels != "auto":
        return kernels
    if torch.device(device).type == "cuda" and available():
        return "fused"
    log("[kernels] torch: fused needs a CUDA device and the packages cuequivariance-torch and cuequivariance-ops-torch-cu12")
    return "torch"


def engine(m):
    """The polynomials of model m (built at the first call, again if the model moves)."""
    key = (m.head.device, m.head.dtype)
    cached = _ENGINES.get(m)
    if cached is None or cached[0] != key:
        require(m.kernels, key[0])
        cached = _ENGINES[m] = (key, Engine(m, "uniform_1d" if m.kernels == "fused" else "naive"))
    return cached[1]


def _terms(c, outputs=None):
    """(left, right, cg, out) of a Coupling module, restricted to the output rows `outputs` (renumbered)."""
    L, R, CG, O = (getattr(c, k).cpu().numpy() for k in ("rows_a", "rows_b", "factors", "dest"))
    outs = list(range(c.n_out)) if outputs is None else [int(o) for o in outputs]
    pos = {o: k for k, o in enumerate(outs)}
    keep = np.isin(O, outs)
    return [(int(l), int(r), float(cg), pos[int(o)]) for l, r, cg, o in zip(L[keep], R[keep], CG[keep], O[keep])], len(outs)


def _wanted(red, invs):
    """{order: sorted rows} of a base's products that its message reduction and the readout read."""
    want = {}
    for l, parts in ({} if red is None else red.parts).items():
        for o, _ in parts:
            want.setdefault(o, set()).update(getattr(red, f"pick{l}_{o}").tolist())
    for o, rows in invs:
        want.setdefault(o, set()).update(rows.tolist())
    return {o: sorted(r) for o, r in want.items()}


class Stack:
    """The products of one base (n1 rows of u channels) that are read, as one array F (atoms, rows, u)."""

    def __init__(self, eng, products, n1, want, u, name):
        has = [k for k in ("c2", "c3", "c4") if k in products.couplings]
        c2, f, self.names, (rows,), _ = fused.stack_plan([(1, has, n1, want, lambda k, sel: _terms(products.couplings[k], sel))], u)
        self.c2 = None if c2 is None else eng.poly(c2, name + "_c2")
        self.f, self.eng = eng.poly(f, name + "_f"), eng
        start, s = {}, 0
        for o in sorted(rows):
            start[o], s = s, s + len(rows[o])
        self.u, self.n_rows = u, s
        self.at = {o: {r: start[o] + q for q, r in enumerate(rows[o])} for o in rows}

    def index(self, order, rows):
        """Positions in F of these rows of the order's product."""
        return torch.tensor([self.at[order][int(r)] for r in rows.tolist()], dtype=torch.long, device=self.eng.device)

    def __call__(self, A):
        """F (atoms, rows, u) of the base A (atoms, n1 * u)."""
        blocks = {"A": A}
        if self.c2 is not None:
            blocks["c2"] = self.eng.run(self.c2, [A, A])
        F = self.eng.run(self.f, [blocks[x] for x in self.names])
        return F.reshape(A.shape[0], self.n_rows, self.u)


class Level:
    """A base of one layer and one channel group: its products, the reduction of its F to the next
    layer's messages and the features read."""

    def __init__(self, eng, base, u, n1, red, invs, name):
        self.u, self.n1, self.red = u, n1, red
        self.stack = None if base is None else Stack(eng, base, n1, _wanted(red, invs), u, name)
        self.gather = {}
        if red is not None and self.stack is not None:
            for l, parts in red.parts.items():
                if parts:
                    self.gather[l] = (torch.cat([self.stack.index(o, getattr(red, f"pick{l}_{o}")) for o, _ in parts]),
                                      sum(n for _, n in parts))

    def F(self, A):
        """The products of the base A (atoms, n1, u), or A itself without products."""
        return A if self.stack is None else self.stack(A.reshape(A.shape[0], -1))

    def messages(self, F):
        """The message reduction of F, (atoms, n_lm_out * n_out): Reduce.forward in this layout."""
        a, red, out = F.shape[0], self.red, []
        for l in range(red.lmax_out + 1):
            if l not in self.gather:
                out.append(F.new_zeros(a, 2 * l + 1, red.n_out))
                continue
            pos, n = self.gather[l]
            x = F.index_select(1, pos).reshape(a, n, 2 * l + 1, self.u).permute(0, 2, 1, 3).reshape(a, 2 * l + 1, -1)
            out.append(torch.matmul(x, getattr(red, f"mix{l}").reshape(-1, red.n_out)))
        return torch.cat(out, 1).reshape(a, -1)


class Engine:
    def __init__(self, m, method):
        import cuequivariance_torch as cuet
        self.cuet, self.method, self.device, self.polys = cuet, method, m.head.device, []
        lmax, lf, k = m.lmax, m.lmax_feat, m.n_message
        deg = np.concatenate([np.full(2 * l + 1, l) for l in range(lmax + 1)])
        taper, n_layers = m.taper is not None, m.n_layers
        red = lambda t, b: None if t >= n_layers - 1 else m.reduces[t][b] if taper and t == 0 else m.reduces[t]
        keys = [(0, b, o) for b, o in m.group_keys] + [(t, 0, o) for t, o in m.inv_keys]
        self.buffer = lambda t, b, o: getattr(m, f"inv0g{b}_{o}" if taper and t == 0 else f"inv{t}_{o}")
        invs = lambda t, b: [(o, self.buffer(t, b, o)) for kt, kb, o in keys if (kt, kb) == (t, b)]

        self.levels = [[]]
        for b, (gmax, lo, hi) in enumerate(m.groups):
            u, n1 = hi - lo, (gmax + 1) ** 2
            lv = Level(self, m.prod0[b] if taper else m.prod0, u, n1, red(0, b), invs(0, b), f"vv_p0_{b}")
            lv.gmax, lv.lo, lv.hi = gmax, lo, hi
            lv.bond = self.poly(fused.radial_poly([(int(deg[lm]), lm, lm, 1.0) for lm in range(n1)],
                                                  gmax + 1, (lmax + 1) ** 2, n1, u), f"vv_bond_{b}")
            self.levels[0].append(lv)
        self.edge = []
        for t in range(1, n_layers):
            c = m.ycouple[t - 1]
            terms, _ = _terms(c)
            lm_deg = np.concatenate([np.full(2 * l + 1, l) for l in range(lf + 1)])
            self.edge.append(self.poly(fused.edge_poly([(int(lm_deg[l]), l, r, o, cg) for l, r, cg, o in terms],
                                                       lf + 1, (lmax + 1) ** 2, (m.lmax_ind + 1) ** 2, c.n_out, k),
                                       f"vv_edge_{t}"))
            base = m.prods[t - 1] if t - 1 < len(m.prods) else None
            self.levels.append([Level(self, base, k, c.n_out, red(t, 0), invs(t, 0), f"vv_p{t}")])
        self.features = [(t, b, o, self.levels[t][b].stack.index(o, self.buffer(t, b, o))) for t, b, o in keys]
        rb = m.bispec_proj[0].shape[1]
        self.bispec = Stack(self, m.bispec, (lf + 1) ** 2, {3: list(range(len(m.bispec.meta[3])))}, rb, "vv_bispec")

    def poly(self, p, name):
        self.polys.append(p)
        sp = self.cuet.SegmentedPolynomial(p, method=self.method, name=name)
        return sp.to(self.device) if self.method == "naive" else sp

    def run(self, sp, inputs, idx=None, n=None):
        """The polynomial's output, per row, or summed over the rows with the same idx into n rows."""
        if idx is None:
            return sp(inputs)[0]
        return sp(inputs, output_indices={0: idx}, output_shapes={0: inputs[0].new_empty(n, 0)})[0]


def _backflow(m, Y, rad, film_rad, chem, table, rev):
    """ILMlip._backflow, (bonds, lm_out * n_message)."""
    Q = [torch.matmul(o, m._order1_mix(l)) for l, o in enumerate(m._object(rad, film_rad, chem, table, rev, m.lmax_ind + 1))]
    back = torch.cat([Y[:, l * l:(l + 1) ** 2, None] * q[:, None, :] for l, q in enumerate(Q)], 1)
    return (back * m.parity[None, :, None]).reshape(back.shape[0], -1)


def _gram(m, e, A):
    """ILMlip._gram on the last layer's base A (atoms, rows, n_message)."""
    a, k = A.shape[0], A.shape[2]
    out, Q = [A.index_select(1, m.gram_rows0).transpose(1, 2).reshape(a, -1)], []
    for l in range(m.lmax_feat + 1):
        x = A.index_select(1, getattr(m, f"gram_rows{l}")).reshape(a, -1, 2 * l + 1, k).permute(0, 2, 1, 3)
        x = x.reshape(a, 2 * l + 1, -1)
        P = torch.matmul(x, m.gram_proj[l])
        out.append((P[:, :, m.gram_pairs[0]] * P[:, :, m.gram_pairs[1]]).sum(1))
        Q.append(torch.matmul(x, m.bispec_proj[l]))
    B = e.bispec(torch.cat(Q, 1).reshape(a, -1))
    return out + [B.transpose(1, 2).reshape(a, -1)]


def descriptor(m, batch, Y, R, rad, film_rad, chem0, table):
    """ILMlip.descriptor from the first layer's radial functions R on: the same features."""
    e = engine(m)
    n, ind_i, ind_j, mu = batch["n_atoms"], batch["ind_i"], batch["ind_j"], batch["mu"]
    S, lf, k = m.n_species, m.lmax_feat, m.n_message
    A = [e.run(lv.bond, [torch.cat([R[l][:, lv.lo:lv.hi] for l in range(lv.gmax + 1)], 1), Y], ind_i, n)
         .reshape(n, lv.n1, lv.u) for lv in e.levels[0]]
    A, f0 = m._gate(0, A)
    layers = [[lv.F(a) for lv, a in zip(e.levels[0], A)]]
    Yf = Y
    Y = Y[:, :(lf + 1) ** 2]
    for t in range(m.n_layers - 1):
        I = sum(lv.messages(F) for lv, F in zip(e.levels[t], layers[-1]))[ind_j]
        if t == 0:
            back = _backflow(m, Y, rad, film_rad, chem0[ind_i], table, mu[ind_j] * S + mu[ind_i])
            I = I - (back if f0 is None else back * f0[ind_j][:, None])
        radial = torch.cat([R[l][:, :k] for l in range(lf + 1)], 1)
        lv = e.levels[t + 1][0]
        A = e.run(e.edge[t], [radial, Yf, I], ind_i, n).reshape(n, lv.n1, k)
        A, _ = m._gate(t + 1, A)
        layers.append([lv.F(A)])
    feats = [layers[t][b].index_select(1, pos).transpose(1, 2).reshape(n, -1) for t, b, o, pos in e.features]
    return torch.cat(feats + _gram(m, e, layers[-1][0]), dim=1)
