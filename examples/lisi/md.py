import numpy as np
from ase import units
from ase.io import read
from ase.md.nose_hoover_chain import NoseHooverChainNVT
from ase.md.velocitydistribution import thermalize_momenta

from ilmlip import load_calculator

atoms = read("train.xyz")
atoms.calc = load_calculator("OAM-S-omat")     # or a fine-tuned model: "naive/model.pt"

thermalize_momenta(atoms, temperature_K=300, rng=np.random.default_rng())
dyn = NoseHooverChainNVT(
    atoms, 0.5 * units.fs,
    temperature_K=300, tdamp=50 * units.fs,
    trajectory="md.traj", logfile="md.log", loginterval=100,
)
dyn.run(1000)
