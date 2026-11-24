"""Equivariance of the internal features of an ilmlip model, float64, CPU, torch engine.

Invariant energies and covariant forces (invariance_check.py) do not show that every intermediate tensor
transforms correctly: a wrong coupling coefficient or sign could cancel in the invariants.  Here a random
periodic structure and its rotated copy (positions and cell) are passed through the model, and every
degree-l block of
    A1   the gated bond sums of the first layer (every channel group),
    B    every product function of the first layer (orders 1 ... nu_max, all degrees L),
    I    the messages (degrees l <= l_msg),
    A2   the gated bond sums of the later layers,
must satisfy X(R) = X D^l(R) over the m index.  D^l(R) is fitted by least squares from the model's own
real spherical harmonics, Y_l(u R^T) = Y_l(u) D^l, so no convention is assumed; the fit residual and
|D D^T - 1| are printed, and a control shows that the rotation changes the blocks l > 0.  Per-atom quantities only (the edge order of the neighbour list may change).

    python equivariance_check.py [--config XS S M L XL XXL] [--checkpoint model.pt ...]

Without arguments a small random model is checked.  The exit code is the number of failed blocks.
"""
import argparse
import sys

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from shared import random_model, structure
from ilmlip import load_calculator
from ilmlip.coupling import functions
from ilmlip.model import Products, Reduce, angular, collate, graph

TOL = 1e-10


def wigner(R, lmax, rng):
    """D[l] with Y_l(u R^T) = Y_l(u) D[l] for the model's real harmonics; also max |D D^T - 1| and the fit residual."""
    u = rng.normal(size=(400, 3))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    Y = angular(torch.tensor(u), lmax, torch).numpy()
    Yr = angular(torch.tensor(u @ R.T), lmax, torch).numpy()
    D, orth = [], 0.0
    for l in range(lmax + 1):
        s = slice(l * l, (l + 1) ** 2)
        M, *_ = np.linalg.lstsq(Y[:, s], Yr[:, s], rcond=None)
        orth = max(orth, abs(M @ M.T - np.eye(2 * l + 1)).max(), abs(Y[:, s] @ M - Yr[:, s]).max())
        D.append(M)
    return D, orth


def capture(model, atoms):
    """Run the descriptor and record the per-atom equivariant tensors with their (degree, rows) layout."""
    got, hooks = [], []
    gate = model._gate

    def gated(t, A):
        out = gate(t, A)
        for a in (out[0] if isinstance(out[0], list) else [out[0]]):
            if t == 0:            # the first layer's channel groups: standard rows l^2 ... (l+1)^2 - 1
                layout = [(l, list(range(l * l, (l + 1) ** 2))) for l in range(int(a.shape[2] ** 0.5))]
            else:                 # later layers: the rows of their coupling table (several paths per degree)
                layout = [(f.l, list(f.rows)) for f in functions(model.ycouple[t - 1].meta)]
            got.append((f"A{t + 1}", layout, a.detach().clone()))
        return out

    def on_products(mod, _inp, out):
        for order, x in out.items():
            got.append((f"B order {order}", [(f.l, list(f.rows)) for f in functions(mod.meta[order])], x.detach().clone()))

    def on_reduce(mod, _inp, out):
        got.append(("I", [(l, list(range(l * l, (l + 1) ** 2))) for l in range(mod.lmax_out + 1)], out.detach().clone()))

    model._gate = gated
    for m in model.modules():
        if isinstance(m, Products):
            hooks.append(m.register_forward_hook(on_products))
        elif isinstance(m, Reduce):
            hooks.append(m.register_forward_hook(on_reduce))
    try:
        with torch.no_grad():
            model.descriptor(collate([graph(atoms, model.rcut, model.species)]))
    finally:
        model._gate = gate
        for h in hooks:
            h.remove()
    return got


def check(label, model, a, seed=3):
    """Number of (tensor, degree) blocks of the model on structure a that are not equivariant."""
    rng = np.random.default_rng(seed)
    R = Rotation.random(random_state=seed).as_matrix()
    b = a.copy()
    b.set_cell(a.cell[:] @ R.T, scale_atoms=False)
    b.set_positions(a.positions @ R.T)
    lmax = int(model.lmax)
    D, orth = wigner(R, lmax, rng)
    x0, x1 = capture(model, a), capture(model, b)
    assert [k for k, *_ in x0] == [k for k, *_ in x1]
    print(f"\n{label}\n  lmax {lmax}, l_feat {model.lmax_feat}, l_msg {model.lmax_ind}; D^l from the model's harmonics: "
          f"|D D^T - 1| and fit residual <= {orth:.1e}")
    worst, control = {}, 0.0
    for (name, layout, t0), (_, _, t1) in zip(x0, x1):
        for l, rows in layout:
            if l <= lmax:
                X0, X1 = t0[..., rows].numpy(), t1[..., rows].numpy()
                scale = max(abs(X0).max(), 1e-300)
                dev = abs(X0 @ D[l] - X1).max() / scale
                worst[name, l] = max(worst.get((name, l), 0.0), dev)
                if l > 0:
                    control = max(control, abs(X0 - X1).max() / scale)
    print(f"  {'tensor':14s} {'l':>2s}  max |X D^l - X(R)| / max|X|")
    for (name, l), dev in sorted(worst.items()):
        print(f"  {name:14s} {l:2d}  {dev:.1e}{'' if dev < TOL else '   FAIL'}")
    print(f"  control: without D^l the blocks l > 0 differ by up to {control:.1e}{'' if control > 1e-3 else '   FAIL'}")
    return int(sum(dev >= TOL for dev in worst.values()) + (control <= 1e-3))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", nargs="*", default=[], help="release configurations with random weights (XS, S, M, L, XL, XXL)")
    ap.add_argument("--checkpoint", nargs="*", default=[], help="checkpoint files")
    args = ap.parse_args(argv)
    a = structure(np.random.default_rng(3))
    models = [(f"config {c}, random weights", random_model(a, c)) for c in args.config]
    models += [(c, c) for c in args.checkpoint]
    fails = 0
    for label, m in models or [("small random model", random_model(a))]:
        fails += check(label, load_calculator(m, device="cpu", engine="torch", dtype="float64").model, a)
    print(f"\n{'all blocks equivariant' if not fails else f'{fails} blocks FAILED'}")
    return fails


if __name__ == "__main__":
    sys.exit(main())
