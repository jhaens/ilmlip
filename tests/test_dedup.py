"""Path deduplication of the product functions."""
import inspect
import json

import numpy as np
import torch
from ase import Atoms
from ase.geometry import get_distances

from conftest import CONFIGS
from ilmlip.coupling import couple, uncoupled
from ilmlip.model import ILMlip, Products, collate, graph

ELEMENTS = ["Na", "Cl", "O", "Si", "H", "Cu"]


def model(config):
    hp = json.load(open(CONFIGS / f"{config}.json"))["hparams"]
    hp = {k: v for k, v in hp.items() if k in inspect.signature(ILMlip.__init__).parameters}
    torch.manual_seed(0)
    return ILMlip(ELEMENTS, **hp)


def structure(rng, n=100):
    """Random periodic structure of n atoms, 2-4 elements, no pair closer than 1.3 A."""
    species = rng.choice(ELEMENTS, rng.integers(2, 5), replace=False)
    cell = (n / rng.uniform(0.04, 0.09)) ** (1 / 3) * (np.eye(3) + rng.normal(0, 0.15, (3, 3)))
    pos = []
    while len(pos) < n:
        p = rng.random(3) @ cell
        if not pos or get_distances(p[None], np.array(pos), cell=cell, pbc=True)[1].min() > 1.3:
            pos.append(p)
    return Atoms(rng.choice(species, n), positions=pos, cell=cell, pbc=True)


def singular_values(m, n_atoms):
    rng, out = np.random.default_rng(1), []
    for _ in range(-(-n_atoms // 800)):
        with torch.no_grad():
            out.append(m.descriptor(collate([graph(structure(rng), m.rcut, m.species) for _ in range(8)])).numpy())
    F = np.concatenate(out)
    return np.linalg.svd(F / np.linalg.norm(F, axis=0), compute_uv=False)


def rank(s, tol=1e-8):
    return int((s > tol * s[0]).sum())


def test_path_tables_are_the_couplings():
    A = uncoupled(2)
    P = Products(A, 2, 1, 3, {2: 1, 3: 1, 4: 0})
    t2 = couple(A, A, 2, 1, symmetric=True)
    t3 = couple(t2.meta, A, 1, 1)
    for t, c in ((t2, P.couplings["c2"]), (t3, P.couplings["c3"])):
        assert np.array_equal(t.rows_a, c.rows_a.numpy()) and np.array_equal(t.dest, c.dest.numpy())
        assert np.array_equal(t.factors, c.factors.numpy())
    assert P.meta[3] == t3.meta
    assert model("S").n_features == 406


def test_path_features_have_full_rank():
    m = model("S")
    assert (m.n_features, rank(singular_values(m, 1600))) == (406, 406)
