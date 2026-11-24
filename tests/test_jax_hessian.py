"""JaxCalculator.get_hessian against TorchCalculator.get_hessian, in chunks, in both atom orders, with both
kernels and the table layouts of the fused ones, in float32, and its limits (the finite differences of the
forces and the Lanczos spectrum are in test_calculator)."""
import warnings
from functools import lru_cache

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)

from ase import Atoms

from conftest import KERNELS, same_on_device, structures, tiny
from test_switches import ALL, build
from ilmlip import jax_md
from ilmlip.calculator import TorchCalculator
from ilmlip.jax_md import JaxCalculator, from_torch

FUSED = "fused-reference" in KERNELS
needs_fused = pytest.mark.skipif(not FUSED, reason="cuequivariance-jax not installed")


def rel(a, b):
    return np.abs(a - b).max() / np.abs(b).max()


def unsymmetrised(calc, atoms, m=16):
    """The rows of get_hessian before it symmetrises them."""
    calc.get_potential_energy(atoms)
    n3, n_pad = 3 * len(atoms), calc._nl["n_pad"]
    rows = calc._hessian_rows(calc.kernels, m)
    U = np.eye(-(-n3 // m) * m, 3 * n_pad).reshape(-1, m, n_pad, 3)
    return np.concatenate([np.asarray(rows(u)).reshape(m, -1) for u in U])[:n3, :n3]


@lru_cache(maxsize=None)
def shared(dtype="float64"):
    """conftest.tiny() in JAX, one per precision for the module: the tests reuse its compiled programs."""
    return from_torch(tiny(), dtype)


def case(name):
    """(model, in JAX, atoms): the plain model or one at charge 1, spin 2."""
    if name == "charge_spin":
        atoms = structures()["molecule"]
        atoms.info.update(charge=1, spin=2)
        m = tiny(charge_spin=True)
        return m, from_torch(m, "float64", 1, 2), atoms
    return tiny(), shared(), structures()[name]


@pytest.mark.parametrize("name", ["molecule", "nacl", "triclinic", "charge_spin"])
def test_matches_torch(name):
    m, jm, atoms = case(name)
    q, s = atoms.info.get("charge"), atoms.info.get("spin")
    H = JaxCalculator(jm, kernels="xla").get_hessian(atoms)
    ref = TorchCalculator(m, dtype="float64", charge=q, spin=s).get_hessian(atoms)
    assert H.shape == (3 * len(atoms),) * 2 and H.dtype == np.float64
    assert np.array_equal(H, H.T)
    assert rel(H, ref) <= 1e-11


def test_chunks_atom_order_and_layer0_do_not_matter(monkeypatch):
    atoms = structures()["nacl"]
    calc = JaxCalculator(shared(), kernels="xla")
    H = calc.get_hessian(atoms)
    others = [calc.get_hessian(atoms, chunk=c) for c in (1, 5)]
    others.append(JaxCalculator(shared(), kernels="xla", spatial_order=False).get_hessian(atoms))
    monkeypatch.setattr(jax_md, "_PAIR_MATMUL", 1)
    per_bond = JaxCalculator(from_torch(tiny(), "float64"), kernels="xla")
    others.append(per_bond.get_hessian(atoms))
    assert "layer0" in per_bond._nl["args"][10]
    for other in others:
        assert rel(other, H) <= 1e-12


@needs_fused
def test_fused_reference_matches_torch():
    """Table rows of one width, the common layout of the fused kernels (rows of two widths, from
    element cutoffs, have a test of their own)."""
    m, atoms = tiny(), structures()["nacl"]
    calc = JaxCalculator(from_torch(m, "float64"), kernels="fused-reference")
    H = calc.get_hessian(atoms)
    assert sum(n for n, _ in calc._nl["key_extra"][2:]) == 0
    assert rel(H, TorchCalculator(m, dtype="float64").get_hessian(atoms)) <= 1e-11
    raw = unsymmetrised(calc, atoms)
    assert np.abs(raw - raw.T).max() <= 1e-12 * np.abs(raw).max()


def test_float32_is_close_to_float64():
    atoms = structures()["nacl"]
    H = JaxCalculator(shared(), kernels="xla").get_hessian(atoms)
    calc = JaxCalculator(shared("float32"), kernels="xla")
    ev = np.linalg.eigvalsh(H)
    assert rel(calc.get_hessian(atoms), H) <= 1e-4
    assert np.abs(calc.get_hessian(atoms, method="lanczos", k=4)[0] - ev[:4]).max() <= 1e-4 * np.abs(ev).max()


def test_default_atoms_are_the_last_calculated():
    calc = JaxCalculator(shared(), kernels="xla")
    with pytest.raises(ValueError, match="no atoms"):
        calc.get_hessian()
    atoms = structures()["molecule"]
    atoms.calc = calc
    atoms.get_forces()
    assert same_on_device(calc.get_hessian(), calc.get_hessian(atoms), calc.device)


def test_an_isolated_atom_has_a_zero_hessian():
    calc = JaxCalculator(shared(), kernels="xla")
    assert np.array_equal(calc.get_hessian(Atoms("Cu")), np.zeros((3, 3)))


@needs_fused
def test_not_in_chunks():
    calc = JaxCalculator(shared(), kernels="fused-reference", chunk_atoms=4)
    with pytest.raises(NotImplementedError, match="chunks"):
        calc.get_hessian(structures()["nacl"])
    assert calc.n_chunks == 2


@needs_fused
def test_failing_fused_second_derivatives_fall_back_to_xla(monkeypatch):
    jm = shared()
    atoms = structures()["nacl"]
    calc = JaxCalculator(jm, kernels="fused-reference")
    real = jm.hessian_step

    def failing(message):
        def step(kernels, ordered):
            if kernels == "xla":
                return real(kernels, ordered)

            def linearize(*args, **static):
                raise RuntimeError(message)
            return linearize, None
        return step

    monkeypatch.setattr(jm, "hessian_step", failing("RESOURCE_EXHAUSTED: Out of memory"))
    with pytest.raises(RuntimeError, match="RESOURCE_EXHAUSTED"):
        calc.get_hessian(atoms)
    monkeypatch.setattr(jm, "hessian_step", failing("no second derivative"))
    with pytest.warns(RuntimeWarning, match="with XLA"):
        H = calc.get_hessian(atoms)
    assert calc.kernels == "fused-reference"
    assert rel(H, TorchCalculator(tiny(), dtype="float64").get_hessian(atoms)) <= 1e-11
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        assert same_on_device(calc.get_hessian(atoms), H, calc.device)


def test_dense_chunks_halve_when_the_device_runs_out_of_memory(monkeypatch):
    atoms = structures()["nacl"]
    calc = JaxCalculator(shared(), kernels="xla")
    H = calc.get_hessian(atoms)
    real = JaxCalculator._hessian_rows

    def limited(self, kernels, m):
        rows = real(self, kernels, m)

        def run(U):
            if U.shape[0] > 4:
                raise RuntimeError("RESOURCE_EXHAUSTED: out of memory while trying to allocate")
            return rows(U)
        return run

    monkeypatch.setattr(JaxCalculator, "_hessian_rows", limited)
    with pytest.warns(RuntimeWarning, match="chunk=16; trying chunk=8") as got:
        assert rel(calc.get_hessian(atoms), H) <= 1e-12
    assert [str(w.message).rsplit("=", 1)[1] for w in got] == ["8", "4"]


@pytest.mark.parametrize("kernels", ["xla"] + ["fused-reference"] * FUSED)
def test_all_options_hessian_matches_torch(kernels):
    m = build(**ALL)
    atoms = structures()["molecule"]
    H = JaxCalculator(from_torch(m, "float64"), kernels=kernels).get_hessian(atoms)
    assert rel(H, TorchCalculator(m, dtype="float64").get_hessian(atoms)) < 1e-11
