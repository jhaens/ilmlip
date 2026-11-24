"""Five MDs on one GPU: water at 300, 400 and 500 K, and water with one and two KOH at 300 K, all
with Bussi (0.5 fs). Each MD is an ASE run in a process of its own; this process holds the model and
computes the forces of all five in one call per step."""
import numpy as np
from ase import units
from ase.io import read
from ase.md.bussi import Bussi
from ase.md.velocitydistribution import Stationary, thermalize_momenta

from ilmlip.lockstep import imap

RUNS = [                         # structure, temperature (K); the runs may differ in size and species
    ("water.xyz", 300), ("water.xyz", 400), ("water.xyz", 500),
    ("water_1koh.xyz", 300), ("water_2koh.xyz", 300),
]


def md(calc, run):
    path, T = run
    name = f"{path.removesuffix('.xyz')}_{T}K"
    atoms = read(path)
    atoms.calc = calc
    rng = np.random.default_rng(RUNS.index(run))     # one generator per MD
    thermalize_momenta(atoms, temperature_K=T, rng=rng)
    Stationary(atoms)
    Bussi(atoms, 0.5 * units.fs, temperature_K=T, taut=100 * units.fs, rng=rng,
          trajectory=f"{name}.traj", logfile=f"{name}.log", loginterval=100).run(20000)   # 10 ps
    return name


if __name__ == "__main__":             # the processes import this file
    for name in imap("OAM-S-omat", md, RUNS):
        print(name, "done")
