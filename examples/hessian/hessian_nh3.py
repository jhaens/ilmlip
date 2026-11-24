"""Vibrations of ammonia from the analytic Hessian: the pyramidal minimum and the planar
transition state of its inversion (one imaginary mode)."""
import numpy as np
from ase import Atoms
from ase.build import molecule
from ase.optimize import BFGS
from ase.vibrations import VibrationsData

from ilmlip import load_calculator

calc = load_calculator("OMOL-S", engine="torch", dtype="float64")
planar = Atoms("NH3", positions=[[0, 0, 0]] + [[np.cos(t), np.sin(t), 0] for t in np.radians([90, 210, 330])])

energy = {}
for name, atoms in [("pyramidal", molecule("NH3")), ("planar", planar)]:
    atoms.calc = calc
    BFGS(atoms, logfile=None).run(fmax=1e-5)
    energy[name] = atoms.get_potential_energy()
    print(name)
    print(VibrationsData.from_2d(atoms, calc.get_hessian(atoms)).tabulate())
print(f"inversion barrier: {energy['planar'] - energy['pyramidal']:.3f} eV")
