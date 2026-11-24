"""load_calculator(dispersion=...): Grimme's D3 from the dftd3 package on top of a model."""
import sys

import numpy as np
import pytest
import torch
from ase.build import bulk, molecule
from ase.calculators.calculator import Calculator, all_changes
from ase.md.langevin import Langevin
from ase.units import fs

from conftest import HP, SPECIES, close, numerical_derivatives, tiny
from test_calculator import HAVE_CXX, save
from ilmlip import lockstep, load_calculator, pretrained
from ilmlip.calculator import CommitteeCalculator, _functional, with_d3
from ilmlip.lockstep import imap

DFTD3 = pytest.importorskip("dftd3.ase").DFTD3
NAME = "OAM-S-omat"


class Zero(Calculator):
    implemented_properties = ["energy", "free_energy", "forces", "stress"]

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.results = dict(energy=0.0, free_energy=0.0, forces=np.zeros((len(atoms), 3)), stress=np.zeros(6))


@pytest.fixture(scope="module", autouse=True)
def models(tmp_path_factory):
    """The released OAM-S-omat when it is on this machine, else a small random model of that name."""
    if pretrained.find(NAME):
        yield
        return
    folder = tmp_path_factory.mktemp("models")
    save(tiny(), folder / f"{NAME}.pt")
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("ILMLIP_MODELS", str(folder))
        yield


def nacl(n=2):
    at = bulk("NaCl", "rocksalt", a=5.64, cubic=True).repeat(n)
    at.rattle(0.1, seed=0)
    return at


def properties(calc, atoms):
    a = atoms.copy()
    a.calc = calc
    out = [a.get_potential_energy(), *a.get_forces().ravel()]
    return np.array(out + (list(a.get_stress()) if a.pbc.any() else []))


def plain_plus_d3(atoms, **kw):
    return (properties(load_calculator(NAME, device="cpu", **kw), atoms)
            + properties(DFTD3(method="pbe", damping="d3bj"), atoms))


@pytest.mark.parametrize("engine", ["cpp", "jax", "torch"])
def test_wrapper_is_the_model_plus_d3(engine):
    if engine == "cpp" and not HAVE_CXX:
        pytest.skip("the cpp engine needs a C++ compiler")
    if engine == "jax":
        pytest.importorskip("jax")
    c = load_calculator(NAME, device="cpu", engine=engine, dispersion="d3bj")
    if c.mixer.calcs[0].engine != engine:
        pytest.skip(f"the {engine} engine did not load here")
    at = nacl()
    assert len(at) == 64
    assert close(properties(c, at), plain_plus_d3(at, engine=engine), 1e-8)


def test_d3_forces_and_stress_are_the_derivatives_of_its_energy():
    at = nacl(1)
    at.rattle(0.2, seed=1)
    c = with_d3(Zero(), "d3bj", "pbe", realspace_cutoff=dict(width2=10.0, width3=10.0))

    def energy(a):
        a.calc = c
        return a.get_potential_energy()

    f, s = numerical_derivatives(energy, at, n=len(at))
    a = at.copy()
    a.calc = c
    assert np.abs(a.get_forces() - f).max() < 1e-7
    assert np.abs(a.get_stress() - s).max() < 1e-8
    assert np.abs(s).max() > 1e-5


def test_molecule_without_a_cell():
    mol = molecule("CH3CH2OH")
    mol.rattle(0.05, seed=0)
    c = load_calculator(NAME, device="cpu", engine="torch", dispersion="d3bj")
    ref = plain_plus_d3(mol, engine="torch")
    assert close(properties(c, mol), ref, 1e-8)
    assert properties(c, mol).shape == (1 + 3 * len(mol),)


def test_d3_means_d3bj_and_bad_names_raise():
    at = nacl(1)
    a = properties(load_calculator(NAME, device="cpu", engine="torch", dispersion="d3"), at)
    b = properties(load_calculator(NAME, device="cpu", engine="torch", dispersion="d3bj"), at)
    assert close(a, b, 1e-12)
    zero = properties(with_d3(Zero(), "d3zero", "pbe"), at)
    assert not close(zero, properties(with_d3(Zero(), "d3bj", "pbe"), at), 1e-6)
    with pytest.raises(ValueError, match="dispersion must be one of"):
        load_calculator(NAME, device="cpu", engine="torch", dispersion="d4")
    with pytest.raises(ValueError, match="dispersion must be one of"):
        with_d3(Zero(), "grimme")


def test_committee_with_dispersion():
    path = pretrained.path(NAME)
    at = nacl(1)
    c = load_calculator([path, path], device="cpu", engine="torch", dispersion="d3bj")
    ref = (properties(CommitteeCalculator([load_calculator(path, device="cpu", engine="torch")] * 2), at)
           + properties(DFTD3(method="pbe", damping="d3bj"), at))
    assert close(properties(c, at), ref, 1e-8)


def omol_type(folder):
    """A checkpoint with charge and spin inputs, as a fine-tuned OMOL model is."""
    torch.save(dict(model=tiny(charge_spin=True).state_dict(), hparams=dict(HP, charge_spin=True),
                    species=SPECIES), folder / "ft.pt")
    return folder / "ft.pt"


def test_functional_follows_the_training_data():
    assert _functional("OAM-S-omat") == "pbe"
    assert _functional("MATPES-r2SCAN-S") == "r2scan"
    assert _functional("MATPES-r2SCAN-S", "pbe") == "pbe"
    assert _functional(["a.pt", "b.pt"]) == "pbe"
    assert _functional("ft.pt", "r2scan") == "r2scan"


def test_a_checkpoint_with_charge_and_spin_takes_d3(tmp_path):
    path, mol = omol_type(tmp_path), molecule("CH3CH2OH")
    mol.rattle(0.05, seed=0)
    kw = dict(device="cpu", engine="torch", charge=0, spin=1)
    c = load_calculator(path, dispersion="d3bj", **kw)
    ref = properties(load_calculator(path, **kw), mol) + properties(DFTD3(method="pbe", damping="d3bj"), mol)
    assert close(properties(c, mol), ref, 1e-8)


@pytest.mark.parametrize("model, why", [("OMOL-S", "OMOL.*VV10"), ("omol_xl", "OMOL.*VV10"),
                                        ("SOLIDACID-S", "PBE-D3: its models have dispersion already"),
                                        (["OAM-S-omat", "OMOL-S"], "OMOL.*VV10"),
                                        (["SOLIDACID-M", "SOLIDACID-L"], "PBE-D3: its models")])
def test_pretrained_omol_and_solidacid_refuse_d3_by_name_before_loading(model, why, tmp_path, monkeypatch):
    def loaded(*args):
        raise AssertionError("a model was looked up")

    monkeypatch.setenv("ILMLIP_MODELS", str(tmp_path))
    monkeypatch.setattr(pretrained, "fetch", loaded)
    monkeypatch.setattr(lockstep, "_start", loaded)
    with pytest.raises(ValueError, match=why):
        load_calculator(model, dispersion="d3bj", functional="pbe")
    with pytest.raises(ValueError, match=why):
        list(imap(model, task, [(0, nacl(1))], dispersion="d3bj"))


def test_imap_takes_d3_for_a_checkpoint_with_charge_and_spin(tmp_path, monkeypatch):
    def started(*args):
        raise RuntimeError("started")

    monkeypatch.setattr(lockstep, "_start", started)
    with pytest.raises(RuntimeError, match="started"):
        list(imap(omol_type(tmp_path), task, [(0, molecule("H2O"))], dispersion="d3bj", device="cpu"))


def test_missing_dftd3_raises_before_a_model_is_loaded_or_a_process_started(monkeypatch):
    monkeypatch.setitem(sys.modules, "dftd3", None)
    monkeypatch.setitem(sys.modules, "dftd3.ase", None)

    def started(*args):
        raise AssertionError("a process was started")

    monkeypatch.setattr(lockstep, "_start", started)
    with pytest.raises(ImportError, match="pip install dftd3"):
        load_calculator("no-such-model", dispersion="d3")
    with pytest.raises(ImportError, match="pip install dftd3"):
        list(imap("no-such-model", task, [(0, nacl(1))], dispersion="d3"))
    with pytest.raises(ValueError, match="dispersion must be one of"):
        list(imap("no-such-model", task, [(0, nacl(1))], dispersion="d4"))


def task(calc, item):
    k, atoms = item
    atoms.calc = calc
    return k, atoms.get_potential_energy(), atoms.get_forces()


def test_imap_adds_d3_in_the_task_processes():
    items = []
    for k in range(2):
        at = bulk("NaCl", "rocksalt", a=5.64, cubic=True).repeat((2, 1, 1))
        at.rattle(0.1, seed=k)
        items.append((k, at))
    ref = load_calculator(NAME, engine="jax", device="cpu", dispersion="d3bj")
    plain = load_calculator(NAME, engine="jax", device="cpu")
    out = sorted(imap(NAME, task, items, dispersion="d3bj", device="cpu"), key=lambda r: r[0])
    assert len(out) == 2
    for (k, e, f), (_, at) in zip(out, items):
        want = properties(ref, at)
        assert abs(e - want[0]) < 1e-4 * max(1.0, abs(want[0]))
        assert np.abs(f.ravel() - want[1:1 + 3 * len(at)]).max() < 1e-3
        base = properties(plain, at)
        assert abs(e - base[0]) > 10 * 1e-4 * max(1.0, abs(want[0]))
        assert np.abs(f.ravel() - base[1:1 + 3 * len(at)]).max() > 1e-3


def test_langevin_with_d3():
    at = nacl(1)
    at.calc = load_calculator(NAME, device="cpu", engine="torch", dispersion="d3bj")
    Langevin(at, 1 * fs, temperature_K=300, friction=0.02, rng=np.random.default_rng(0)).run(20)
    assert np.isfinite(at.get_potential_energy()) and np.isfinite(at.get_forces()).all()
