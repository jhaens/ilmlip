"""Helpers of the checks: a random periodic structure and a random ilmlip model, either a small one or
one with the hyper-parameters of a release configuration."""
import inspect
import json
from pathlib import Path

import numpy as np
import torch
from ase import Atoms

import ilmlip
from ilmlip.model import ILMlip, collate, graph

ELEMENTS = (3, 8, 14, 17, 26)
SYMBOLS = ["Li", "O", "Si", "Cl", "Fe"]
SMALL = dict(rcut=4.5, lmax=3, lmax_feat=2, n_basis=6, n_channels=8, product_order=3, lmax_ind=2, n_layers=2,
             n_message=4, hidden=(16,), embed_dim=8, zbl=True, taper=[8, 8, 6, 4],
             bound=16.0, bond_scale=0.05, feat_bound=8.0, zbl_switch=10.0)
CONFIGS = Path(ilmlip.__file__).parent / "configs"


def structure(rng, n=40, dmin=1.8, elements=ELEMENTS):
    """Random triclinic cell (about 11 A^3 per atom), atoms added one by one at least dmin apart."""
    cell = 7.6 * (n / 40) ** (1 / 3) * np.eye(3) + rng.normal(0, 0.4, (3, 3))
    a = Atoms([int(rng.choice(elements))], positions=[[0, 0, 0]], cell=cell, pbc=True)
    while len(a) < n:
        b = a + Atoms("H", positions=[rng.random(3) @ cell])
        if b.get_distances(len(a), range(len(a)), mic=True).min() > dmin:
            b.numbers[-1] = rng.choice(elements)
            a = b
    return a


def random_model(atoms, config=None, seed=0):
    """A float64 model with random weights for the five elements, calibrated on atoms (the bounds need
    the spread of the bond sums and features).  config: the name of a release configuration (XS ... XXL)
    or None for the small model."""
    hp = SMALL if config is None else json.loads((CONFIGS / f"{config}.json").read_text())["hparams"]
    known = set(inspect.signature(ILMlip.__init__).parameters)
    torch.manual_seed(seed)
    m = ILMlip(SYMBOLS, **{k: v for k, v in hp.items() if k in known and k != "species"})
    with torch.no_grad():
        for p in m.parameters():
            if p.abs().sum() == 0:
                p.normal_(0, 0.1)
        m.energy_scale.fill_(0.7)
    m.eval()
    m.fit_feature_normalisation([collate([graph(atoms, m.rcut, m.species)])])
    return m
