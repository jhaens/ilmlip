"""The torch model: charge and spin, sizes of the configurations, checkpoint layouts, couplings (invariance,
derivatives and the round trip are in test_switches)."""
import copy
import inspect
import json
import os
import pickle

import numpy as np
import pytest
import torch
from ase.build import bulk

from conftest import CONFIGS, HP, SPECIES, structures, tiny
from ilmlip.model import ILMlip, collate, fold_charge_spin, graph, load, repulsion


def test_charge_and_spin_only_for_models_that_take_them():
    plain = tiny()
    assert fold_charge_spin(plain) is plain
    for kw in (dict(charge=-2), dict(spin=1), dict(charge=0, spin=1)):
        with pytest.raises(ValueError, match="OMOL"):
            fold_charge_spin(plain, **kw)
    folded = fold_charge_spin(tiny(charge_spin=True))
    assert folded.folded_charge_spin == (0, 1)
    assert fold_charge_spin(folded, 0, 1) is folded
    with pytest.raises(ValueError, match="charge 0, spin 1"):
        fold_charge_spin(folded, -2)


@pytest.mark.parametrize("name, size", [("XS", (140_456, 306)), ("S", (300_612, 406)), ("M", (849_732, 406)),
                                        ("L", (963_587, 622)), ("XL", (6_953_980, 4_376)),
                                        ("XXL", (27_822_308, 9_152))])
def test_config_sizes(name, size):
    from ase.data import chemical_symbols
    species = [chemical_symbols[z] for z in range(1, 101)]
    h = json.loads((CONFIGS / f"{name}.json").read_text())["hparams"]
    kw = {k: (tuple(v) if k == "hidden" else v) for k, v in h.items() if k in inspect.signature(ILMlip.__init__).parameters}
    m = ILMlip(species, **kw)
    assert (m.n_parameters(), m.n_features) == size


def test_repulsion_and_its_slope_vanish_at_rmax():
    r = torch.linspace(0.5, 2.0, 200, dtype=torch.float64, requires_grad=True)
    e = repulsion(r, torch.tensor(14.0), torch.tensor(8.0), torch.tensor(2.0))
    (g,) = torch.autograd.grad(e.sum(), r)
    assert e[-1].abs() < 1e-12 and g[-1].abs() < 1e-9 and (e[:-1] > 0).all()


def test_torch_calculator_notices_a_new_charge_state():
    from ilmlip.calculator import TorchCalculator
    at = structures()["molecule"]
    at.calc = TorchCalculator(tiny(charge_spin=True), dtype="float64", charge=0, spin=1)
    at.get_potential_energy()
    at.info["charge"] = 1
    with pytest.raises(ValueError, match="charge/spin"):
        at.get_potential_energy()


def test_a_model_file_that_runs_code_is_refused(tmp_path):
    class Payload:
        def __reduce__(self):
            return os.mkdir, (str(tmp_path / "ran"),)

    torch.save(dict(metrics=Payload()), tmp_path / "evil.pt")
    with pytest.raises(pickle.UnpicklingError):
        load(tmp_path / "evil.pt")
    assert not (tmp_path / "ran").exists()


def test_float32_energies_add_the_e0s_in_float64():
    """Absolute E0s (OMol25: up to 7e4 eV): a float32 model is off by the rounding of its network part only."""
    m = tiny()
    with torch.no_grad():
        m.atomic_energies.fill_(-2047.4)
        m.atomic_energies[SPECIES.index("Cl")] = -70040.9
    m32 = copy.deepcopy(m).to(torch.float32)
    m.atomic_energies.data = m32.atomic_energies.double()      # the same E0s in both
    at = bulk("NaCl", "rocksalt", a=5.64, cubic=True).repeat((3, 3, 3))
    at.positions += np.random.default_rng(0).normal(0, 0.05, at.positions.shape)
    g = graph(at, m.rcut, m.species)
    a = m(collate([g]), compute_forces=False)
    b = m32(collate([g], dtype=torch.float32), compute_forces=False)
    assert b["energy"].dtype == b["atomic_energy"].dtype == torch.float64
    assert (b["atomic_energy"] - a["atomic_energy"]).abs().max() < 1e-6
    assert (b["energy"] - a["energy"]).abs().max() < 1e-5


def test_a_float64_load_of_a_float32_checkpoint_has_the_exact_tables(tmp_path):
    saved = tiny(dtype=torch.float32).state_dict()
    torch.save(dict(model=saved, hparams=HP, species=SPECIES), tmp_path / "m.pt")
    loaded = load(tmp_path / "m.pt")[0].state_dict()
    fresh = tiny().state_dict()
    tables = [k for k in fresh if k.endswith(".factors")]
    assert tables and all(torch.equal(loaded[k], fresh[k]) for k in tables)
    saved[tables[0]] = saved[tables[0]] * 1.01
    torch.save(dict(model=saved, hparams=HP, species=SPECIES), tmp_path / "other.pt")
    with pytest.raises(ValueError, match=tables[0]):
        load(tmp_path / "other.pt")


def test_duplicate_couplings():
    """Dropped couplings are multiples of kept ones of the same order and L, the kept ones of none."""
    from ilmlip.coupling import couple, duplicates, uncoupled
    drop = duplicates()
    pair = lambda la, lb, L: (((), la, (), lb), L)
    assert (pair(1, 0, 1) + ((), 0), 1) in drop  # (A1 A0)_1 A0
    assert (pair(0, 0, 0) + ((), 1), 1) not in drop  # (A0 A0)_0 A1: every 2-layer model keeps it
    A = uncoupled(2)
    aa = couple(A, A, 2, 2, symmetric=True, drop=())
    x = np.random.default_rng(1).normal(size=(12, len(A)))

    def evaluate(t, a, b):
        out = np.zeros((len(a), len(t.meta)))
        np.add.at(out.T, t.dest, t.factors[:, None] * (a[:, t.rows_a] * b[:, t.rows_b]).T)
        return out

    xaa = evaluate(aa, x, x)
    for t, y in ((couple(aa.meta, A, 2, 2, drop=()), x),
                 (couple(aa.meta, aa.meta, 2, 2, symmetric=True, drop=()), xaa)):
        v, ch = evaluate(t, xaa, y), {}
        for k, (L, M, hist) in enumerate(t.meta):
            ch.setdefault((hist, L), np.zeros((len(x), 2 * L + 1)))[:, M + L] = v[:, k]
        unit = {k: c.ravel() / np.linalg.norm(c) for k, c in ch.items()}
        kept = [k for k in unit if k not in drop]
        for k in unit:
            hits = [j for j in kept if j != k and j[1] == k[1] and abs(unit[k] @ unit[j]) > 1 - 1e-9]
            assert bool(hits) == (k in drop), k
