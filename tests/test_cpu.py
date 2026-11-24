"""The C++ engine (CpuCalculator) against the torch model."""
import inspect
import json
import shutil
import threading

import numpy as np
import pytest
import torch

if not (shutil.which("g++") or shutil.which("c++")):
    pytest.skip("the CPU engine needs a C++ compiler", allow_module_level=True)

from ase import Atoms, units
from ase.build import bulk
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet

from conftest import CONFIGS, SHAPES, SPECIES, close, results, structures, tiny
from ilmlip.calculator import TorchCalculator, load_calculator
from ilmlip.cpu import CpuCalculator, CpuModel, library, reverse_pairs
from ilmlip.model import ILMlip, collate, graph
from ilmlip.verlet import host_neighbours
from test_bounds import CASES as BOUND_CASES, dense
from test_switches import ALL, CASES, REFERENCE, build, config_model

VARIANTS = dict(CASES, all_xl=dict(ALL, lmax_feat=2, product_order=4, taper=[5, 4, 4]),
                all_higher=dict(ALL, lmax=3, lmax_feat=2, lmax_ind=2, taper=[6, 5, 4, 3]),
                all_cutoffs=dict(ALL, rcut_elem={"H": 3.0, "Na": 3.5, "Cl": 3.8}))


def extra_structures():
    rng = np.random.default_rng(5)
    cu = bulk("Cu", "fcc", a=3.6)                        # one atom: every neighbour is an image
    tight = Atoms("NaCl", positions=[[0, 0, 0], [0.9, 0.1, 0.0]], cell=[6, 6, 6], pbc=True)   # ZBL on
    dense = Atoms("HCONaSiClCu", positions=rng.random((7, 3)) * 3.0, cell=np.eye(3) * 3.2, pbc=True)
    return dict(cu=cu, tight=tight, dense=dense)


@pytest.mark.parametrize("shape", list(SHAPES))
@pytest.mark.parametrize("state", [None, (1, 2)])
def test_matches_torch(shape, state):
    """float64 equal to torch, for S and XL shapes with and without charge and spin."""
    m = tiny(charge_spin=state is not None, **SHAPES[shape])
    q, s = state or (None, None)
    cc = CpuCalculator(m, dtype="float64", charge=q, spin=s)
    tc = TorchCalculator(m, dtype="float64", charge=q, spin=s)
    for at in list(structures().values()) + list(extra_structures().values()):
        at.info.update(charge=q, spin=s)
        assert close(results(cc, at), results(tc, at), 1e-11)


@pytest.mark.parametrize("n_layers", [2, 3, 4])
def test_other_depths(n_layers):
    """Models of 2, 3 and 4 layers equal torch."""
    m = tiny(n_layers=n_layers)
    cc, tc = CpuCalculator(m, dtype="float64"), TorchCalculator(m, dtype="float64")
    for at in structures().values():
        assert close(results(cc, at), results(tc, at), 1e-11)


def test_float32():
    """The float32 engine is within 1e-6 of float64 torch."""
    m = tiny(**SHAPES["XL"])
    cc, tc = CpuCalculator(m, dtype="float32"), TorchCalculator(m, dtype="float64")
    for at in structures().values():
        assert close(results(cc, at), results(tc, at), 1e-6)


@pytest.mark.parametrize("case", VARIANTS)
def test_variants_match_torch(case):
    """One calculator through all structures: their species pairs come one after the other."""
    m = build(**VARIANTS[case])
    cc, tc = CpuCalculator(m, dtype="float64"), TorchCalculator(m, dtype="float64")
    for at in list(structures().values()) + list(extra_structures().values()):
        assert close(results(cc, at), results(tc, at), 1e-11)


def test_folded_charge_spin_of_a_variant():
    m = build(**ALL, charge_spin=True)
    cc = CpuCalculator(m, dtype="float64", charge=1, spin=2)
    tc = TorchCalculator(m, dtype="float64", charge=1, spin=2)
    for at in structures().values():
        at.info.update(charge=1, spin=2)
        assert close(results(cc, at), results(tc, at), 1e-11)


def test_config_shapes():
    """The shipped S: channels above a SIMD vector, element cutoffs, three hidden layers."""
    h = json.loads((CONFIGS / "S.json").read_text())["hparams"]
    keys = set(inspect.signature(ILMlip.__init__).parameters) & set(h)
    m = build(**{k: h[k] for k in keys})
    cc, tc = CpuCalculator(m, dtype="float64"), TorchCalculator(m, dtype="float64")
    for at in structures().values():
        assert close(results(cc, at), results(tc, at), 1e-11)


def test_variant_float32():
    """The float32 engine is within 1e-5 of float64 torch for the all-switches model."""
    m = build(**ALL)
    cc, tc = CpuCalculator(m, dtype="float32"), TorchCalculator(m, dtype="float64")
    for at in structures().values():
        assert close(results(cc, at), results(tc, at), 1e-5)


def test_variant_committee():
    """A committee of all-switches models gives the torch committee's energy and spread."""
    members = [build(seed=s, **ALL) for s in (0, 1)]
    cc = load_calculator(members, engine="cpp", dtype="float64")
    tc = load_calculator(members, engine="torch", dtype="float64")
    a, b = structures()["nacl"], structures()["nacl"]
    a.calc, b.calc = cc, tc
    assert abs(a.get_potential_energy() - b.get_potential_energy()) < 1e-11
    assert abs(cc.results["energy_std"] - tc.results["energy_std"]) < 1e-11


@pytest.mark.parametrize("name", ["S", "XL"])
def test_config_models_keep_the_stored_results(name):
    """The stored results of config_model S and XL (test_switches)."""
    ref, at = REFERENCE[name], structures()["nacl"]
    at.calc = CpuCalculator(config_model(name), dtype="float64")
    np.testing.assert_allclose(at.get_potential_energy(), ref["energy"], rtol=1e-11)
    np.testing.assert_allclose(at.get_forces()[:3], ref["forces"], rtol=1e-9, atol=1e-13)
    np.testing.assert_allclose(at.get_stress(), ref["stress"], rtol=1e-9, atol=1e-13)


def bound_structures():
    """Test structures and compressed ones: the ZBL energy of some atom between 0 and the switch, a dense cell
    where the gate acts."""
    rng = np.random.default_rng(3)
    s = structures()
    return dict(nacl=s["nacl"], triclinic=s["triclinic"], **dense(),
                dimer=Atoms("CO", positions=[[0, 0, 0], [0.8, 0.05, 0.0]], cell=[7, 7, 7], pbc=True),
                cluster=Atoms("NaClCO", positions=[[0, 0, 0], [1.1, 0.2, 0.0], [0.5, 1.3, 0.4], [2.0, 1.6, 0.8]]),
                packed=Atoms("HCONaSiClCu" * 2, positions=rng.random((14, 3)) * 2.6, cell=np.eye(3) * 2.8, pbc=True))


def check_bounds_act(m, ats):
    """The core switch is strictly between 0 and 1 for some atom, and the gate below 0.5 for some atom."""
    switch = gate = 1.0
    for at in ats:
        b = collate([graph(at, m.rcut, m.species)])
        with torch.no_grad():
            if m.zbl_switch:
                u = (m.zbl_atoms(b) / m.zbl_switch).numpy()
                switch = min(switch, np.min(u[(u > 0) & (u < 1)], initial=1.0))
            gate = min([gate] + [float(g.min()) for g in m.gate_factors(b)])
    assert not m.zbl_switch or switch < 1.0
    assert not m.bound or gate < 0.5


@pytest.mark.parametrize("case", BOUND_CASES)
def test_bounds_match_torch(case):
    """Each bound keyword and the combinations equal torch, on structures where the gate and the core switch act."""
    m = build(**BOUND_CASES[case])
    cc, tc = CpuCalculator(m, dtype="float64"), TorchCalculator(m, dtype="float64")
    ats = {**structures(), **bound_structures()}
    check_bounds_act(m, ats.values())
    for at in ats.values():
        assert close(results(cc, at), results(tc, at), 1e-11)


@pytest.mark.parametrize("name", ["S", "L"])
@pytest.mark.parametrize("over", [{}, dict(bound=None, zbl_switch=None), dict(feat_bound=8.0)],
                         ids=["bounds", "no_bounds", "feat_bound"])
def test_shipped_configs_match_torch(name, over):
    """Shipped configurations (bounds on, off, with the squash): float64 equal to torch to 1e-10."""
    m = config_model(name, **over)
    cc, tc = CpuCalculator(m, dtype="float64"), TorchCalculator(m, dtype="float64")
    ats = bound_structures()
    check_bounds_act(m, ats.values())
    for at in ats.values():
        assert close(results(cc, at), results(tc, at), 1e-10)


def test_bounded_float32():
    """The float32 engine is within 1e-5 of float64 torch for a shipped configuration with the squash."""
    m = config_model("S", feat_bound=8.0)
    cc, tc = CpuCalculator(m, dtype="float32"), TorchCalculator(m, dtype="float64")
    for at in bound_structures().values():
        assert close(results(cc, at), results(tc, at), 1e-5)


@pytest.mark.parametrize("make, a", [(tiny, 5.64), (lambda: build(**ALL), 5.64),
                                     (lambda: build(**BOUND_CASES["all_3_layers"]), 4.2)],
                         ids=["tiny", "all", "bounded"])
def test_threads_agree(make, a):
    """One and four threads give the same results."""
    m = CpuModel(make(), "float64")
    at = bulk("NaCl", "rocksalt", a=a).repeat((3, 3, 3))
    at.positions += np.random.default_rng(2).normal(0, 0.1, at.positions.shape)
    one, many = results(CpuCalculator(m, n_threads=1), at), results(CpuCalculator(m, n_threads=4), at)
    assert close(one, many, 1e-13)


def test_background_compile_is_ignored():
    """background_compile does not change the results."""
    m = CpuModel(tiny(), "float64")
    at = bulk("NaCl", "rocksalt", a=5.64).repeat((2, 2, 2))
    assert close(results(CpuCalculator(m, background_compile=True), at), results(CpuCalculator(m), at), 1e-13)


def test_verlet_list_follows_md():
    """Forces along an MD run with Verlet rebuilds equal those of fresh neighbour lists."""
    m = tiny()
    at = bulk("NaCl", "rocksalt", a=5.64).repeat((2, 2, 2))
    MaxwellBoltzmannDistribution(at, temperature_K=1500, rng=np.random.default_rng(1))
    calc = CpuCalculator(m, dtype="float64", skin=0.3)
    at.calc = calc
    ref = TorchCalculator(m, dtype="float64")
    dyn = VelocityVerlet(at, 2 * units.fs)
    for _ in range(12):
        dyn.run(5)
        assert close(results(calc, at), results(ref, at), 1e-11)
    assert calc.n_rebuilds > 1


def test_list_follows_cell():
    """NPT-like steps (the cell and the atoms move a little) give the results of fresh lists
    with few rebuilds, lists built in the background at an older cell among them; a large
    compression rebuilds."""
    m = CpuModel(tiny(), "float64")
    rng = np.random.default_rng(3)
    at = bulk("NaCl", "rocksalt", a=5.64).repeat((2, 2, 2))
    at.positions += rng.normal(0, 0.05, at.positions.shape)
    calc, older = CpuCalculator(m, skin=0.5), 0
    for _ in range(30):
        e = rng.normal(0, 1e-3, (3, 3))
        at.set_cell(at.cell.array @ (np.eye(3) + e + e.T), scale_atoms=True)
        at.positions += rng.normal(0, 0.01, at.positions.shape)
        n = calc.n_rebuilds
        assert close(results(calc, at), results(CpuCalculator(m), at), 1e-12)
        older += calc.n_rebuilds > n and not np.array_equal(calc._nl["h0"], at.cell.array)
    assert calc.n_rebuilds <= 6 and older
    at.set_cell(at.cell.array * 0.85, scale_atoms=True)
    assert close(results(calc, at), results(CpuCalculator(m), at), 1e-12)
    assert np.array_equal(calc._nl["h0"], at.cell.array)


def test_engine_is_cached():
    """library() is the same on every call."""
    assert library() == library()


def test_shared_model_and_inputs():
    """One CpuModel shared by concurrent Python threads, int64 index arrays, mismatched options."""
    m = CpuModel(tiny(), "float64")
    at = bulk("NaCl", "rocksalt", a=5.64).repeat((3, 3, 2))
    at.positions += np.random.default_rng(4).normal(0, 0.1, at.positions.shape)
    calc = CpuCalculator(m)
    ref = results(calc, at)
    mu, off, cj, csh, rev = calc._nl["args"]
    pos = at.get_positions()
    e64 = m.compute(pos, mu.astype(np.int64), off, cj.astype(np.int64), csh, rev.astype(np.int64))
    assert abs(e64[0] - ref[0]) < 1e-12
    out = []
    jobs = [threading.Thread(target=lambda: out.append(m.compute(pos, mu, off, cj, csh, rev)[2]))
            for _ in range(4)]
    for j in jobs:
        j.start()
    for j in jobs:
        j.join()
    assert all(np.array_equal(f, out[0]) for f in out)
    with pytest.raises(ValueError):
        CpuCalculator(m, dtype="float32")
    with pytest.raises(ValueError):
        CpuCalculator(m, charge=1)


def test_host_neighbours_in_threads():
    """The background rebuilds of several calculators call vesin at the same time."""
    systems = [bulk("NaCl", "rocksalt", a=5.64).repeat((3 + k, 3, 3)) for k in range(2)]
    ref = [host_neighbours(a.positions, a.cell.array, a.pbc, 5.5) for a in systems]
    bad = []

    def work(a, want):
        for _ in range(30):
            got = host_neighbours(a.positions, a.cell.array, a.pbc, 5.5)
            if not all(np.array_equal(g, w) for g, w in zip(got, want)):
                bad.append(len(a))

    jobs = [threading.Thread(target=work, args=pair) for pair in zip(systems, ref)]
    for j in jobs:
        j.start()
    for j in jobs:
        j.join()
    assert not bad


def test_reverse_pairs():
    """The reverse of each pair is the one a dict of (i, j, S) finds, also for pairs without
    one and for keys past 64 bits."""
    def want(i, j, S):
        index = {(a, c, *s): k for k, (a, c, s) in enumerate(zip(i.tolist(), j.tolist(), S.tolist()))}
        rev = np.array([index.get((c, a, *(-x for x in s)), -1)
                        for a, c, s in zip(i.tolist(), j.tolist(), S.tolist())], np.int64)
        return rev, rev >= 0

    lists = [host_neighbours(a.positions, a.cell.array, a.pbc, 5.5)[:3]
             for a in [*structures().values(), extra_structures()["cu"]]]
    lists += [(np.zeros(0, np.int32), np.zeros(0, np.int32), np.zeros((0, 3), np.int32)),
              (np.array([0, 1]), np.array([1, 0]), np.array([[10 ** 6, 0, 0], [-10 ** 6, 0, 0]]))]
    for i, j, S in lists:
        for keep in (np.ones(len(i), bool), np.arange(len(i)) != len(i) // 2):
            got, ref = reverse_pairs(i[keep], j[keep], S[keep]), want(i[keep], j[keep], S[keep])
            assert all(np.array_equal(g, r) and g.dtype == r.dtype for g, r in zip(got, ref))


def test_build_folder(tmp_path, monkeypatch):
    """build_dir: next to a source checkout, in the user cache when installed."""
    from ilmlip import cpu
    monkeypatch.delenv("ILMLIP_BUILD", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    checkout = tmp_path / "repo" / "ilmlip"
    checkout.mkdir(parents=True)
    monkeypatch.setattr(cpu, "__file__", str(checkout / "cpu.py"))
    assert cpu.build_dir() == tmp_path / "repo" / "build"
    installed = tmp_path / "lib" / "site-packages" / "ilmlip"
    installed.mkdir(parents=True)
    monkeypatch.setattr(cpu, "__file__", str(installed / "cpu.py"))
    assert cpu.build_dir() == tmp_path / "cache" / "ilmlip" / "build"
    assert not (installed.parent / "build").exists()
