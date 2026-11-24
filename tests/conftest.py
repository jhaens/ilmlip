from pathlib import Path

import numpy as np
import pytest
import torch
from ase import Atoms
from ase.build import bulk, fcc111, molecule

import ilmlip
from ilmlip.model import ILMlip, collate, graph

SPECIES = ["H", "C", "O", "Na", "Si", "Cl", "Cu"]
CONFIGS = Path(ilmlip.__file__).parent / "configs"       # the installed package's, also outside a checkout
HP = dict(rcut=4.0, lmax=2, lmax_feat=1, n_basis=5, n_channels=4, product_order=3, lmax_ind=1, n_layers=2,
          n_message=3, hidden=[8], embed_dim=4, zbl=True)
SHAPES = dict(S=dict(lmax=2, lmax_feat=1, product_order=3), XL=dict(lmax=2, lmax_feat=2, product_order=4))
KERNELS = ["xla"]
try:
    import cuequivariance_jax  # noqa: F401
    KERNELS.append("fused-reference")
except ImportError:
    pass


def tiny(seed=0, charge_spin=False, lmax=2, lmax_feat=1, product_order=3, dtype=torch.float64, n_layers=2):
    """A small random model with a non-trivial readout."""
    torch.manual_seed(seed)
    m = ILMlip(SPECIES, rcut=4.0, lmax=lmax, lmax_feat=lmax_feat, n_basis=5, n_channels=4,
                     product_order=product_order, lmax_ind=1, n_layers=n_layers, n_message=3, hidden=(8,),
                     embed_dim=4, zbl=True, charge_spin=charge_spin, dtype=dtype)
    with torch.no_grad():
        for p in m.parameters():
            if p.abs().sum() == 0:
                p.normal_(0, 0.1)
        m.feat_mean.normal_(0, 0.1)
        m.feat_std.uniform_(0.5, 1.5)
        m.energy_scale.fill_(0.7)
    return m.eval()


def structures():
    rng = np.random.default_rng(0)
    nacl = bulk("NaCl", "rocksalt", a=5.64).repeat((2, 2, 1))
    nacl.positions += rng.normal(0, 0.1, nacl.positions.shape)
    tri = Atoms("SiOCH", scaled_positions=rng.random((4, 3)), pbc=True,
                cell=[[4.1, 0.2, 0.1], [0.6, 3.9, 0.3], [0.4, -0.5, 4.4]])
    slab = fcc111("Cu", (2, 2, 2), vacuum=5.0)
    slab.positions += rng.normal(0, 0.05, slab.positions.shape)
    mol = molecule("CH3CH2OH")
    mol.positions += rng.normal(0, 0.05, mol.positions.shape)
    return dict(nacl=nacl, triclinic=tri, slab=slab, molecule=mol)


def evaluate(model, atoms, stress=False, **state):
    g = graph(atoms, model.rcut, model.species)
    g.update(state)
    out = model(collate([g]), compute_stress=stress)
    return {k: v.detach().numpy() for k, v in out.items()}


def results(calc, atoms):
    """Energy, forces, site energies and (periodic) stress of a copy of atoms under calc, in one array."""
    a = atoms.copy()
    a.calc = calc
    out = [a.get_potential_energy(), *a.get_forces().ravel(), *a.get_potential_energies()]
    return np.array(out + (list(a.get_stress()) if a.pbc.any() else []))


def close(a, b, tol=1e-11):
    return np.abs(a - b).max() <= tol * max(1.0, np.abs(b).max())


def same_on_device(a, b, device):
    """Bit-identical on the CPU; on a GPU two programs differ by summation order, so float64 agrees to 1e-11 of the scale."""
    a, b = np.asarray(a), np.asarray(b)
    if device.platform == "cpu":
        return np.array_equal(a, b)
    return a.shape == b.shape and np.abs(a - b).max() <= 1e-11 * np.abs(b).max()


def numerical_derivatives(energy, atoms, n=4, h=1e-5):
    """Central differences of energy(atoms): the forces on the first n atoms and, in a periodic
    cell, the stress dE/d(strain) / V in ASE's Voigt order (None otherwise)."""
    def central(moved):
        return (energy(moved(h)) - energy(moved(-h))) / (2 * h)

    def displaced(i, d):
        def moved(s):
            a = atoms.copy()
            a.positions[i, d] += s
            return a
        return moved

    def strained(i, j):
        def moved(s):
            e = np.zeros((3, 3))
            e[i, j] += s / 2
            e[j, i] += s / 2
            a = atoms.copy()
            a.set_cell(atoms.cell.array @ (np.eye(3) + e), scale_atoms=True)
            return a
        return moved

    forces = np.array([[-central(displaced(i, d)) for d in range(3)] for i in range(min(n, len(atoms)))])
    if not atoms.pbc.any():
        return forces, None
    voigt = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))
    return forces, np.array([central(strained(i, j)) for i, j in voigt]) / atoms.get_volume()


@pytest.fixture
def model():
    return tiny()
