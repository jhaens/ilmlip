# Checks

Symbolic (sympy) and numerical checks of the bounds, the core switch, the ZBL taper and the symmetries of
the ILMlip model. They import the installed package (`pip install -e .` in a clone) and run on a laptop CPU
in float64; no checkpoint, GPU or data is needed (random weights, random structures, fixed seeds).

    pip install sympy scipy pytest
    cd checks
    python bounds_sympy.py                       # about 10 s
    python invariance_check.py                   # about 25 s
    python invariance_check.py --dmin 1.0        # about 30 s, atoms inside the core switch
    python equivariance_check.py                 # about 10 s
    pytest -q                                    # the same at smaller sizes, about 20 s

The scripts exit with the number of failed rows or blocks (0 = all passed). `invariance_check.py` and
`equivariance_check.py` take `--config XS|S|M|L|XL|XXL` (the release configuration with random weights) or
`--checkpoint model.pt` instead of the default small model; `shared.py` builds the structure and the model.

## bounds_sympy.py

Each statement is proved with sympy (the LaTeX of the key expressions is printed) and then compared with the
package's own function, so the statement and the code cannot drift apart.

| bound | statement proved | code compared |
|---|---|---|
| B1 gate | f = a0/sigma (1 + m/(sigma kappa)^2)^(-1/2); RMS(fA) = a0 kappa x/sqrt(1+x^2), x = sqrt(m)/(sigma kappa), so RMS < a0 kappa; strictly increasing in sqrt(m); limits a0 kappa (m to infinity) and a0/sigma (m to 0); the gate is 0.99805 at m = sigma^2 for kappa = 16 | `ILMlip._gate`, tensor and list-of-groups input, \|A\| from 1e-6 to 1e9 |
| B2 squash | h(x) = x (1+(x/tau)^2)^(-1/2) is odd, \|h\| < tau, h' = (1+(x/tau)^2)^(-3/2) in (0, 1] (monotone, 1-Lipschitz), h tends to +-tau | `ILMlip._whiten` (stub, feat_bound 3) |
| B3 core switch | s(u) = 1 - 10u^3 + 15u^4 - 6u^5: s(0) = 1, s(1) = 0, s' = s'' = 0 at both ends, s' = -30u^2(1-u)^2 <= 0, s''' = -60 at both ends: exactly C2 after clipping | `core_switch` |
| ZBL taper | T(u) = (1-u)^3 (1+3u+6u^2+10u^3+15u^4+21u^5) = 1 - 28u^6 + 48u^7 - 21u^8: T = 1 with T' ... T^(5) = 0 at 0, T = T' = T'' = 0 and T''' = -336 at 1 (C2), T' <= 0; the pair energy has V = V' = V'' = 0 at rmax and is 0 beyond | `repulsion` (Z = 14, 8; relative deviation 1.5e-14) |

Expected output: every line `[ok]`, last line `all statements verified`.

## invariance_check.py

A random periodic triclinic structure of Li, O, Si, Cl, Fe (40 atoms, minimum distance 1.8 A, cell about 7.6 A, so
periodic images enter the neighbour list). Energy, forces and stress of the original are compared with those of
a rotated copy (positions and cell), permuted atoms, atoms shifted by random lattice vectors and a rigid
translation (tolerance 1e-10); a control displacement of 0.01 A must change the energy (> 1e-3 eV); forces are
compared with central differences of E over all coordinates, the stress with central differences of E under
symmetric strain of positions and cell (h = 1e-5, tolerance 1e-6 relative to the largest force or stress).
`--dmin 1.0` places atoms inside the core switch (ZBL energies up to 30 eV, forces about 300 eV/A), so the
switch and the ZBL taper are differentiated as well. `--atoms` sets the size.

Expected output, small random model, defaults (rows of one test shortened):

```
40 atoms Cl4Fe9Li16O4Si7, cell lengths [7.6  7.44 7.42], E = -0.944619 eV, max|F| = 1.005 eV/A, ...
finite-difference forces: 240 evaluations in 14 s, max abs. deviation 2.8e-10 eV/A
rotation              |F' - F R^T|                  4.94e-15 eV/A   ok (< 1e-10)
...
control (displaced)   dE                            2.10e-02 eV     ok (> 0.001)
finite difference     |F + dE/dx|/max|F|            2.82e-10 (rel.) ok (< 1e-06)
finite difference     |s - dE/de/V|/max|s|          2.30e-09 (rel.) ok (< 1e-06)
```

With `--dmin 1.0`: ZBL energy per atom up to 32.5 eV, 28 atoms inside the switch, symmetry rows below 4e-12,
finite differences 1e-9 relative. The small model has all bounds on (bound 16, feat_bound 8, zbl_switch 10 eV).

## equivariance_check.py

The invariance of the outputs does not show that every internal tensor transforms correctly: a wrong coupling
coefficient or sign could cancel in the invariants. Here every degree-l block of the gated bond sums of every
layer (A1 per channel group, A2), of every product function of the first layer (orders 1 to nu_max, all degrees)
and of the messages I must satisfy X(R) = X D^l(R) over the m index for a random rotation R of positions and
cell. D^l is fitted by least squares from the model's own real harmonics (`ilmlip.model.angular`), so no sign or
ordering convention is assumed; the fit residual and |D D^T - 1| are printed (about 1e-14). The control line
shows that the rotation changes the blocks l > 0 (by O(1)) before D^l is applied. Per-atom tensors only.

Expected output: every block below 1e-10 (about 1e-15 to 1e-14 for the small model and for the random-weight
XS, S, M, L, XL and XXL configurations), then `all blocks equivariant`. The layout of the later layers' bond
sums follows their coupling table (several paths per degree), not the standard (l+1)^2 rows.

## test_checks.py

Thin pytest wrappers: the four parts of `bounds_sympy.py`, `invariance_check.py` at 16 atoms (open and with
atoms inside the core switch) and `equivariance_check.py` for the small model and XS. They are skipped without
sympy or scipy. This folder is not part of `pytest tests`; run `pytest checks`.
