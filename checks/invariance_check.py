"""Symmetries and consistency of the ilmlip energy model, float64, CPU, torch engine.

A random periodic structure of five elements; energy, forces and stress of the original and of rotated
(positions and cell), permuted, lattice-translated and rigidly translated copies; forces against the
central-difference gradient of E; stress against the central-difference strain derivative.  A small random
model is used unless a release configuration (--config) or a checkpoint (--checkpoint) is given.

    python invariance_check.py [--config S] [--checkpoint model.pt] [--atoms 40] [--dmin 1.8]

--dmin 1.0 puts atoms inside the core switch.  The exit code is the number of failed rows.
"""
import argparse
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation

from shared import random_model, structure
from ilmlip import load_calculator
from ilmlip.model import collate, core_switch

SYMMETRY_TOL, FD_TOL, CONTROL_MIN = 1e-10, 1e-6, 1e-3


def run(calc, a, seed=7, h=1e-5):
    """Rows (test, quantity, deviation, unit, kind) of the symmetry, control and finite-difference tests."""
    rng = np.random.default_rng(seed)

    def evaluate(b):
        b = b.copy()
        b.calc = calc
        return b.get_potential_energy(), b.get_forces(), b.get_stress(voigt=False)

    def moved(positions=None, cell=None):
        b = a.copy()
        b.set_cell(a.cell[:] if cell is None else cell, scale_atoms=False)
        b.set_positions(a.positions if positions is None else positions)
        return b

    E0, F0, S0 = evaluate(a)
    print(f"{len(a)} atoms {a.get_chemical_formula()}, cell lengths {np.round(a.cell.lengths(), 2)}, "
          f"E = {E0:.6f} eV, max|F| = {abs(F0).max():.3f} eV/A, max|sigma| = {abs(S0).max():.2e} eV/A^3")
    rows = []

    def symmetry(name, b, F_ref, S_ref):
        E, F, S = evaluate(b)
        rows.extend([(name, "dE", abs(E - E0), "eV", "sym"), (name, "|F' - " + F_ref[0] + "|", abs(F - F_ref[1]).max(), "eV/A", "sym"),
                     (name, "|s' - " + S_ref[0] + "|", abs(S - S_ref[1]).max(), "eV/A^3", "sym")])

    R = Rotation.random(random_state=3).as_matrix()       # positions and cell: E invariant, F -> F R^T, sigma -> R sigma R^T
    symmetry("rotation", moved(a.positions @ R.T, a.cell[:] @ R.T), ("F R^T", F0 @ R.T), ("R s R^T", R @ S0 @ R.T))
    p = rng.permutation(len(a))                          # E invariant, F permuted, sigma invariant
    symmetry("permutation", a[p], ("F[p]", F0[p]), ("s", S0))
    n = rng.integers(-2, 3, (len(a), 3))                 # atoms moved by integer combinations of the cell vectors
    symmetry("lattice translation", moved(a.positions + n @ a.cell[:]), ("F", F0), ("s", S0))
    symmetry("rigid translation", moved(a.positions + rng.normal(0, 3, 3)), ("F", F0), ("s", S0))
    rows.append(("control (displaced)", "dE", abs(evaluate(moved(a.positions + rng.normal(0, 0.01, (len(a), 3))))[0] - E0),
                 "eV", "control"))                       # the test can see a change

    t0 = time.time()
    fd = np.zeros_like(F0)
    for i in range(len(a)):
        for k in range(3):
            e = []
            for sgn in (1, -1):
                x = a.positions.copy()
                x[i, k] += sgn * h
                e.append(evaluate(moved(x))[0])
            fd[i, k] = -(e[0] - e[1]) / (2 * h)
    rows.append(("finite difference", "|F + dE/dx|/max|F|", abs(fd - F0).max() / abs(F0).max(), "(rel.)", "fd"))
    print(f"finite-difference forces: {6 * len(a)} evaluations in {time.time() - t0:.0f} s, "
          f"max abs. deviation {abs(fd - F0).max():.1e} eV/A")

    fdS = np.zeros((3, 3))                               # (1/V) dE/d(strain), symmetric strain of positions and cell
    for i in range(3):
        for j in range(i, 3):
            e = []
            for sgn in (1, -1):
                eps = np.zeros((3, 3))
                eps[i, j] = eps[j, i] = sgn * h / (1 if i == j else 2)     # d(eps : sigma) = h sigma_ij either way
                M = np.eye(3) + eps
                e.append(evaluate(moved(a.positions @ M, a.cell[:] @ M))[0])
            fdS[i, j] = fdS[j, i] = (e[0] - e[1]) / (2 * h) / a.get_volume()
    rows.append(("finite difference", "|s - dE/de/V|/max|s|", abs(fdS - S0).max() / abs(S0).max(), "(rel.)", "fd"))
    print(f"stress (autograd, eV/A^3):\n{np.round(S0, 8)}\nstress (finite difference):\n{np.round(fdS, 8)}")
    return rows


def passed(dev, kind):
    return dev > CONTROL_MIN if kind == "control" else dev < (FD_TOL if kind == "fd" else SYMMETRY_TOL)


def report(rows):
    print(f"\n{'test':<22}{'quantity':<22}{'max deviation':>16}  status")
    for name, q, dev, unit, kind in rows:
        tol = f"> {CONTROL_MIN:g}" if kind == "control" else f"< {FD_TOL if kind == 'fd' else SYMMETRY_TOL:g}"
        print(f"{name:<22}{q:<22}{dev:>16.2e} {unit:<7}{'ok' if passed(dev, kind) else 'FAIL'} ({tol})")
    return sum(not passed(dev, kind) for *_, dev, _, kind in rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", help="release configuration with random weights (XS, S, M, L, XL, XXL); default: a small model")
    ap.add_argument("--checkpoint", help="a checkpoint file instead of a random model")
    ap.add_argument("--atoms", type=int, default=40)
    ap.add_argument("--dmin", type=float, default=1.8, help="minimum distance in A; 1.0 reaches the core switch")
    args = ap.parse_args(argv)
    a = structure(np.random.default_rng(7), args.atoms, args.dmin)
    model = args.checkpoint or random_model(a, args.config)
    calc = load_calculator(model, device="cpu", engine="torch", dtype="float64", compute_stress=True)
    if calc.model.zbl_switch:
        z = calc.model.zbl_atoms(collate([calc._graph(a)], calc.device, calc.dtype)).detach()
        s = core_switch(z, calc.model.zbl_switch)
        print(f"ZBL energy per atom up to {z.max():.3g} eV; core switch s in [{s.min():.4f}, {s.max():.4f}], "
              f"{int((s < 1).sum())} atoms inside the switch")
    return report(run(calc, a))


if __name__ == "__main__":
    sys.exit(main())
