"""The bounds: B1 gate of every bond sum, B2 squash of the whitened features, B3 core switch of the learned
energy by the atom's ZBL energy (keywords bound, bond_scale, feat_bound, zbl_switch)."""
import inspect
import json

import numpy as np
import pytest
import torch
from ase import Atoms
from ase.data import chemical_symbols
from scipy.spatial.transform import Rotation

from conftest import CONFIGS, SPECIES, evaluate, numerical_derivatives, structures
from test_switches import ALL, BASE, BOUNDS, build, spy
from ilmlip.model import ILMlip, collate, core_switch, graph, load

CASES = dict(gate=dict(bound=2.0), squash=dict(feat_bound=8.0), core=dict(zbl_switch=10.0),
             bounds=BOUNDS, all=dict(ALL, **BOUNDS), all_3_layers=dict(ALL, n_layers=3, **BOUNDS))
GATED = ["gate", "all", "all_3_layers"]
UNBOUNDED_REFERENCE = json.loads('{"energy": -0.4593170021704349, "forces": [[7.547448187523228e-06, -2.773202708536376e-07, -5.337758284405075e-06], [-3.145979101818462e-05, -0.00017117255103660017, -6.403095355311385e-05], [0.00013618919930392768, -5.5516477139336195e-05, -9.74037591267375e-06]], "fsum": 0.001745041720319147, "stress": [1.559587920173985e-05, 1.4802563078480423e-05, 1.3964649102759692e-05, -5.818192370709538e-08, -4.4721304096791335e-07, 2.596759451974157e-07]}')


def dense():
    """Compressed structures: NaCl at half the lattice constant, ethanol at 0.6 of its bond lengths (the
    ZBL energies of its atoms lie below and above the switch)."""
    nacl = structures()["nacl"]
    nacl.set_cell(nacl.cell.array * 0.5, scale_atoms=True)
    mol = structures()["molecule"]
    mol.positions *= 0.6
    return dict(nacl_half=nacl, squeezed=mol)


def rms(A):
    parts = A if isinstance(A, list) else [A]
    return (sum((a * a).sum((1, 2)) for a in parts) / sum(a.shape[1] * a.shape[2] for a in parts)).sqrt()


def test_keywords_off_is_the_unbounded_model():
    m = build(**ALL)
    assert not m.bounded()
    assert not hasattr(m, "bond_rms") and "bond_rms" not in m.state_dict()
    out = evaluate(m, structures()["nacl"], stress=True)
    ref = UNBOUNDED_REFERENCE
    np.testing.assert_allclose(out["energy"][0], ref["energy"], rtol=1e-11)
    np.testing.assert_allclose(out["forces"][:3], ref["forces"], rtol=1e-10, atol=1e-13)
    np.testing.assert_allclose(out["stress"][0], ref["stress"], rtol=1e-10, atol=1e-13)
    assert abs(np.abs(out["forces"]).sum() - ref["fsum"]) < 1e-11 * ref["fsum"]
    explicit = build(**ALL, bound=None, bond_scale=0.05, feat_bound=None, zbl_switch=None)
    assert list(explicit.state_dict()) == list(m.state_dict())
    again = evaluate(explicit, structures()["nacl"], stress=True)
    assert all(np.array_equal(again[k], out[k]) for k in out)


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("name", ["nacl_half", "squeezed", "triclinic", "molecule"])
def test_invariance(case, name):
    m = build(**CASES[case])
    at = {**structures(), **dense()}[name]
    ref = evaluate(m, at)
    R = Rotation.random(random_state=1).as_matrix()
    perm = np.random.default_rng(2).permutation(len(at))
    b = at[perm]
    b.positions = b.positions @ R.T + [0.3, -1.1, 2.0]
    b.set_cell(b.cell.array @ R.T)
    out = evaluate(m, b)
    assert abs(out["energy"][0] - ref["energy"][0]) < 1e-12 * max(1.0, abs(ref["energy"][0]))
    np.testing.assert_allclose(out["forces"], ref["forces"][perm] @ R.T, rtol=1e-9, atol=1e-10)


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("name", ["nacl_half", "squeezed", "triclinic"])
def test_forces_and_stress_are_derivatives(case, name):
    m = build(**CASES[case])
    at = {**structures(), **dense()}[name]
    periodic = bool(at.pbc.any())
    out = evaluate(m, at, stress=periodic)
    f, s = numerical_derivatives(lambda a: evaluate(m, a)["energy"][0], at)
    assert np.all(np.abs(out["forces"][:len(f)] - f) < 1e-6 * np.maximum(1.0, np.abs(f)))
    if periodic:
        assert np.all(np.abs(out["stress"][0] - s) < 1e-6 * np.maximum(1.0, np.abs(s)))


@pytest.mark.parametrize("case", GATED)
def test_bond_sums_are_calibrated_to_unit_variance(case):
    m = build(**CASES[case])
    g = [graph(a, m.rcut, m.species) for a in structures().values()]
    m._tap = []
    m.descriptor(collate(g))
    tap, m._tap = m._tap, None
    assert [t for t, _, _ in tap] == list(range(m.n_layers))
    for t, ms, _ in tap:
        assert abs(ms.mean() / m.bond_rms[t] ** 2 - 1) < 1e-12
    assert (m.bond_rms != 1).all()


def test_gate_factors():
    """gate_factors: none without the gate, else one factor in (0, 1] per atom and layer."""
    b = collate([graph(structures()["nacl"], 4.0, SPECIES)])
    assert build(**CASES["squash"]).gate_factors(b) == []
    m = build(**CASES["all"])
    gates = m.gate_factors(b)
    assert len(gates) == m.n_layers and m._tap is None
    assert all(g.shape == (b["n_atoms"],) and (g > 0).all() and (g <= 1).all() for g in gates)


@pytest.mark.parametrize("case", ["all", "all_3_layers"])
def test_bounds_hold_on_a_compressed_cell(case):
    m = build(**CASES[case])
    a0, kappa, tau = m.bond_scale, m.bound, m.feat_bound
    seen, nacl = spy(m), structures()["nacl"]
    with torch.no_grad():
        m.descriptor(collate([graph(a, m.rcut, m.species) for a in structures().values()]))
        typical = [rms(A) for A, _ in seen.values()]
    for r in typical:  # calibration atoms: mean(rho^2) = 1, so the scale is below a0
        assert 0.5 * a0 < r.pow(2).mean().sqrt() < a0
    cell = nacl.copy()
    cell.set_cell(cell.cell.array * 0.5, scale_atoms=True)
    b = collate([graph(cell, m.rcut, m.species)])
    with torch.no_grad():
        x = m.descriptor(b)
        for A, _ in seen.values():
            assert rms(A).max() < a0 * kappa
        assert min(float(g.min()) for g in m.gate_factors(b)) < 0.3         # the gate acts
        assert (m._whiten(x).abs() < tau).all()
        assert ((x - m.feat_mean) / m.feat_std).abs().max() > tau          # the squash acts


def test_core_switch():
    """core_switch is 1 at 0, 0.5 at half the switch energy and 0 from it on, with zero slope outside (0, switch)."""
    z = torch.tensor([0.0, 5.0, 10.0, 12.0], dtype=torch.float64, requires_grad=True)
    c = core_switch(z, 10.0)
    torch.testing.assert_close(c[[0, 2, 3]], torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64))
    assert float(c[1].detach()) == pytest.approx(0.5)
    (dc,) = torch.autograd.grad(c.sum(), z)
    assert dc[0] == dc[2] == dc[3] == 0 and dc[1] < 0


@pytest.mark.parametrize("case", ["core", "all"])
def test_learned_energy_vanishes_where_the_zbl_energy_reaches_the_switch(case):
    m = build(**CASES[case])
    with torch.no_grad():
        m.atomic_energies.zero_()
    zc, regimes = m.zbl_switch, set()
    for r in np.linspace(0.45, 0.92, 12):
        b = collate([graph(Atoms("CO", positions=[[0, 0, 0], [r, 0, 0]]), m.rcut, m.species)])
        z = m.zbl_atoms(b).double()
        e = m(b, compute_forces=False)["atomic_energy"]
        m.zbl_switch = None
        learned = m(b, compute_forces=False)["atomic_energy"] - z
        m.zbl_switch = zc
        torch.testing.assert_close(e - z, core_switch(z, zc) * learned, rtol=1e-12, atol=1e-14)
        if (z >= zc).all():
            assert torch.equal(e, z) and learned.abs().max() > 1e-6
            regimes.add("core")
        elif (z > 0).all():
            regimes.add("outside" if (z < 1e-3).all() else "switch")
    assert regimes >= {"core", "switch"}


@pytest.mark.parametrize("case", GATED)
def test_checkpoint_round_trip_keeps_the_gate(case, tmp_path):
    kw = CASES[case]
    m = build(**kw)
    hp = {k: list(v) if isinstance(v, tuple) else v for k, v in {**BASE, **kw}.items()}
    torch.save(dict(model=m.state_dict(), hparams=hp, species=SPECIES), tmp_path / "m.pt")
    m2, _ = load(tmp_path / "m.pt")
    assert m2.bound == m.bound and torch.equal(m2.bond_rms, m.bond_rms) and (m2.bond_rms != 1).all()
    at = structures()["nacl"]
    assert evaluate(m2, at)["energy"][0] == evaluate(m, at)["energy"][0]


def species100():
    return [chemical_symbols[z] for z in range(1, 101)]


CONFIG_HP = {name: json.loads((CONFIGS / f"{name}.json").read_text())["hparams"] for name in ("XS", "S", "M", "L", "XL", "XXL")}
CONFIG_HP["3_layers_order_3"] = dict(CONFIG_HP["S"], lmax_feat=2, lmax_ind=2, n_layers=3, n_message=12,
                                     taper=[24, 12, 12], gram_rank=12)
CONFIG_HP["3_layers_order_4"] = dict(CONFIG_HP["3_layers_order_3"], product_order=4)


def finite_under_compression(name, **extra):
    known = set(inspect.signature(ILMlip.__init__).parameters)
    hp = {k: (tuple(v) if k == "hidden" else v) for k, v in {**CONFIG_HP[name], **extra}.items() if k in known}
    torch.manual_seed(0)
    m = ILMlip(species100(), dtype=torch.float32, **hp)
    with torch.no_grad():
        m.head.normal_(0, 0.1)
        m.elem_bias.normal_(0, 0.1)
    m.eval()

    def batch(s):
        at = structures()["nacl"]
        at.set_cell(at.cell.array * s, scale_atoms=True)
        return collate([graph(at, m.rcut, m.species)], dtype=torch.float32)

    m.fit_feature_normalisation([batch(1.0)])
    for s in (1.0, 0.7, 0.5):
        b = batch(s)
        x = m.descriptor(b)
        out = m(b, compute_stress=True)
        assert all(torch.isfinite(v).all() for v in (x, *out.values())), (name, s)
        assert not m.feat_bound or (m._whiten(x).abs() <= m.feat_bound * (1 + 1e-6)).all()          # saturates at tau in float32


@pytest.mark.parametrize("name", CONFIG_HP)
def test_float32_is_finite_under_compression(name):
    """Features, energies, forces and stresses at s = 1, 0.7, 0.5 of the lattice, calibrated at s = 1."""
    finite_under_compression(name)


def lattice(s):
    at = structures()["nacl"]
    at.set_cell(at.cell.array * s, scale_atoms=True)
    return at


def batch_of(m, at):
    return collate([graph(at, m.rcut, m.species)])


def test_feature_saturation_statistics():
    """train.feat_stats: the largest whitened feature and the fraction above tau / 2."""
    from ilmlip import train
    m = build(**CASES["all"])
    bs = [batch_of(m, a) for a in (structures()["nacl"], lattice(0.5))]
    out = train.feat_stats(m, bs)
    x = torch.cat([(m.descriptor(b) - m.feat_mean) / m.feat_std for b in bs]).abs()
    assert out["feat_max"] == x.max().item() and out["feat_sat"] == (x > m.feat_bound / 2).double().mean().item()
    assert out["feat_max"] > m.feat_bound / 2 > 0 and 0 < out["feat_sat"] < 1


def test_training_logs_the_feature_saturation(tmp_path, capsys):
    from test_train import HP, history, make_data, recipe_args
    from ilmlip import train
    make_data(tmp_path / "data")
    (tmp_path / "hp.json").write_text(json.dumps(dict(hparams=dict(HP, bound=2.0, feat_bound=8.0))))
    train.main(recipe_args(tmp_path, tmp_path / "run", "--epochs", "1"))
    out = capsys.readouterr().out
    row = history(tmp_path / "run")[0]
    assert row["feat_max"] > 0 and 0 <= row["feat_sat"] <= 1 and "gate_mean/0" in row
    assert "gate factors (mean/min) L0:" in out and "|x~| max" in out
