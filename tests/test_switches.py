"""Architecture switches (taper, ranks, per-pair weights, charge/spin fold): invariance, derivatives, exact folds, checkpoint loading."""
import inspect
import json

import numpy as np
import pytest
import torch
from ase import Atoms
from scipy.spatial.transform import Rotation

from conftest import CONFIGS, HP, SPECIES, evaluate, numerical_derivatives, structures, tiny
from ilmlip.model import ILMlip, collate, fold_charge_spin, graph, load

BASE = dict(rcut=4.0, lmax=2, lmax_feat=1, n_basis=5, n_channels=5, product_order=3, lmax_ind=1, n_layers=2,
            n_message=3, hidden=(8,), embed_dim=4, zbl=True, gram_rank=4, bispec_rank=2)
FEATURES = dict(
    taper=dict(taper=[5, 4, 3]),
    ranks=dict(gram_rank=2, bispec_rank=4),
)
ALL = {k: v for f in FEATURES.values() for k, v in f.items()}
CASES = {"default": {}, **FEATURES, "all": ALL, "all_3_layers": dict(ALL, n_layers=3)}
BOUNDS = dict(bound=2.0, bond_scale=0.05, feat_bound=8.0, zbl_switch=10.0)


def calibrate(m):
    g = [graph(a, m.rcut, m.species) for a in structures().values()]
    m.fit_feature_normalisation([collate(g, dtype=m.feat_std.dtype)])


def build(seed=0, **kw):
    """A random model of BASE with the keywords kw; a bounded one is calibrated on the test structures."""
    torch.manual_seed(seed)
    m = ILMlip(SPECIES, **{**BASE, **kw})
    with torch.no_grad():
        for p in m.parameters():
            if p.abs().sum() == 0:
                p.normal_(0, 0.1)
        m.feat_mean.normal_(0, 0.1)
        m.feat_std.uniform_(0.5, 1.5)
        m.energy_scale.fill_(0.7)
    m.eval()
    if m.bounded():
        calibrate(m)
    return m


def make(case, **kw):
    """The model of CASES[case], or conftest's tiny one."""
    return tiny(**kw) if case == "tiny" else build(**CASES[case], **kw)


def spy(m):
    """{layer: (gated bond sum, factor)} of the last descriptor call."""
    seen, gate = {}, m._gate

    def wrapped(t, A):
        out = gate(t, A)
        seen[t] = out
        return out
    m._gate = wrapped
    return seen


# stored NaCl energy, first three forces and stress of config_model S and XL
REFERENCE = json.loads('{"S": {"energy": 0.39635121962562736, "forces": [[-0.02087923580198407, 0.024192246050485137, 0.05394517802740526], [0.0038041104566188273, -0.07644249224656918, 0.01733901755613732], [0.08652925082724222, 0.061141457332811774, -0.06317441976965055]], "fsum": 0.8671452004784066, "stress": [-0.014581654035790958, -0.01476903049894033, -0.015129618448481902, 0.0001055260959859581, -0.00018769530471904938, -0.00023661666690712443], "n": 20279}, "XL": {"energy": 1.950654176332779, "forces": [[0.23083591214703955, -0.15350586032528374, 1.2687217308941319], [-0.06528523262678923, -0.28409108149635387, 0.5334292784661762], [1.4779336256318292, 0.432944897343725, -1.4967316250625735]], "fsum": 15.438793217918345, "stress": [-0.0016303311757634598, 0.00400768898025293, 0.0014863875530592948, 0.0011181617183662533, -0.0014385178934191447, -0.0036323266377769553], "n": 238095}}')


def config_model(name, **over):
    """A shipped configuration on conftest's species with one hidden layer of 16 and random head and E0;
    a small bond RMS makes the gate act."""
    h = json.loads((CONFIGS / f"{name}.json").read_text())["hparams"]
    keys = set(inspect.signature(ILMlip.__init__).parameters) & set(h) - {"hidden"}
    torch.manual_seed(0)
    m = ILMlip(SPECIES, hidden=(16,), **{**{k: h[k] for k in keys}, **over})
    with torch.no_grad():
        for p in m.parameters():
            if p.abs().sum() == 0:
                p.normal_(0, 0.1)
        m.feat_mean.normal_(0, 0.1)
        m.feat_std.uniform_(0.5, 1.5)
        if m.bound:
            m.bond_rms.uniform_(0.005, 0.03)
    return m.eval()


@pytest.mark.parametrize("name", ["S", "XL"])
def test_stored_results_of_the_config_models(name):
    m = config_model(name)
    ref, out = REFERENCE[name], evaluate(m, structures()["nacl"], stress=True)
    assert m.n_parameters() == ref["n"]
    np.testing.assert_allclose(out["energy"][0], ref["energy"], rtol=1e-11)
    np.testing.assert_allclose(out["forces"][:3], ref["forces"], rtol=1e-10, atol=1e-13)
    np.testing.assert_allclose(out["stress"][0], ref["stress"], rtol=1e-10, atol=1e-13)
    assert abs(np.abs(out["forces"]).sum() - ref["fsum"]) < 1e-11 * ref["fsum"]


@pytest.mark.parametrize("case", [*CASES, "tiny"])
@pytest.mark.parametrize("name", list(structures()))
def test_invariance(case, name):
    """The energy is invariant and the forces rotate under rotation, translation and permutation."""
    m = make(case)
    at = structures()[name]
    ref = evaluate(m, at)
    R = Rotation.random(random_state=1).as_matrix()
    perm = np.random.default_rng(2).permutation(len(at))
    b = at[perm]
    b.positions = b.positions @ R.T + [0.3, -1.1, 2.0]
    b.set_cell(b.cell.array @ R.T)
    out = evaluate(m, b)
    assert abs(out["energy"][0] - ref["energy"][0]) < 1e-12 * max(1.0, abs(ref["energy"][0]))
    np.testing.assert_allclose(out["forces"], ref["forces"][perm] @ R.T, rtol=1e-9, atol=1e-10)


@pytest.mark.parametrize("case", [*CASES, "tiny"])
@pytest.mark.parametrize("name", ["nacl", "triclinic", "molecule"])
def test_forces_and_stress_are_derivatives(case, name):
    """Forces and stress equal central differences of the energy."""
    m = make(case)
    at = structures()[name]
    periodic = bool(at.pbc.any())
    out = evaluate(m, at, stress=periodic)
    f, s = numerical_derivatives(lambda a: evaluate(m, a)["energy"][0], at)
    assert np.all(np.abs(out["forces"][:len(f)] - f) < 1e-6 * np.maximum(1.0, np.abs(f)))
    if periodic:
        assert np.all(np.abs(out["stress"][0] - s) < 1e-6 * np.maximum(1.0, np.abs(s)))


@pytest.mark.parametrize("case", CASES)
def test_smooth_at_the_cutoff(case):
    """The energy is smooth across the cutoff: a pair inside matters, the jump at rc is below 1e-9."""
    m = build(**CASES[case])

    def energy(d):
        at = Atoms("CHO", positions=[[0, 0, 0], [d, 0, 0], [-1.5, 1.6, 0.4]])
        return evaluate(m, at)["energy"][0]

    rc = m.rcut
    assert abs(energy(rc + 0.2) - energy(rc - 1.5)) > 1e-5         # the pair matters inside
    assert abs(energy(rc + 1e-3) - energy(rc - 1e-3)) < 1e-9
    assert abs(energy(rc + 1e-4) - energy(rc - 1e-4)) < 1e-12


@pytest.mark.parametrize("kw", [dict(), dict(taper=[5, 4, 3]), dict(taper=[5, 3, 2], lmax_ind=1)])
@pytest.mark.parametrize("gate", [{}, dict(bound=2.0, bond_scale=0.5)], ids=["ungated", "gated"])
def test_backflow_cancels_the_senders_order_1_term(kw, gate):
    """The removed term is the order-1 part of j's message for the bond (i, j): f_j (the gate; 1 without
    one) times the reverse edge's object (exact at fixed f)."""
    m = build(**gate, **kw)
    at = structures()["triclinic"]
    batch = collate([graph(at, m.rcut, m.species)])
    blocks, seen, gates = [], {}, spy(m)
    prods = m.prod0 if isinstance(m.prod0, torch.nn.ModuleList) else [m.prod0]
    for P in prods:
        P.register_forward_hook(lambda mod, inp, out: blocks.append(out))
    m.ycouple[0].register_forward_hook(lambda mod, inp, out: seen.update(RY=inp[0], I=inp[1], out=out))
    with torch.no_grad():
        m.descriptor(batch)
        I = m._messages(0, blocks if m.taper else blocks[0])

        ind_i, ind_j = batch["ind_i"], batch["ind_j"]
        shift = batch["shift"].numpy()
        edges = {(int(i), int(j), *np.round(s, 8)): e
                 for e, (i, j, s) in enumerate(zip(ind_i.tolist(), ind_j.tolist(), shift))}
        rev = np.array([edges[int(j), int(i), *np.round(-s, 8)]
                        for i, j, s in zip(ind_i.tolist(), ind_j.tolist(), shift)])

        vec = batch["positions"][ind_j] - batch["positions"][ind_i] + batch["shift"]
        r = vec.norm(dim=1)
        g, Y = m.gbasis(r), m.angular(vec / r[:, None])
        mu, S = batch["mu"], m.n_species
        table = m._film_table()
        z = m._embed(batch)
        R = m._object(*m._radials(g), m.chem0(z)[ind_j], table, mu[ind_i] * S + mu[ind_j])  # of every edge, as centre i
        reds = m.reduces[0] if isinstance(m.reduces[0], torch.nn.ModuleList) else [m.reduces[0]]
        D = torch.zeros_like(I[ind_j])
        for red, (gmax, lo, hi) in zip(reds, m.groups):
            for l in range(min(gmax, m.lmax_ind) + 1):
                if not red.parts[l] or red.parts[l][0][0] != 1:
                    continue
                phi = R[l][rev][:, lo:hi, None] * Y[rev][:, None, l * l:(l + 1) ** 2]    # the reverse edge's object
                D[:, :, l * l:(l + 1) ** 2] += torch.einsum("eim,io->eom", phi, getattr(red, f"mix{l}")[0])
        if gate:
            D = D * gates[0][1][ind_j][:, None, None]
        I_ref = I[ind_j] - D
        assert D.abs().max() > 1e-3 * I[ind_j].abs().max()
        assert (I_ref - seen["I"]).abs().max() < 1e-12
        assert (m.ycouple[0](seen["RY"], I_ref) - seen["out"]).abs().max() < 1e-12


FOLDED = {k: CASES[k] for k in ("default", "taper", "all", "all_3_layers")}
FOLDED["cutoffs"] = dict(rcut_elem={"H": 3.0, "Na": 3.5, "Cl": 3.8})


@pytest.mark.parametrize("bounds", [{}, BOUNDS], ids=["unbounded", "bounds"])
@pytest.mark.parametrize("case", FOLDED)
def test_per_pair_weights_are_exact(case, bounds, monkeypatch):
    m = build(**FOLDED[case], **bounds)
    at = structures()["nacl"]
    ref = evaluate(m, at, stress=True)
    monkeypatch.setattr(ILMlip, "_folded", lambda self: True)
    out = evaluate(m, at, stress=True)
    assert abs(out["energy"][0] - ref["energy"][0]) < 1e-12 * max(1.0, abs(ref["energy"][0]))
    np.testing.assert_allclose(out["forces"], ref["forces"], rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(out["stress"], ref["stress"], rtol=1e-10, atol=1e-12)


def test_per_pair_weights_only_compiled_and_frozen():
    from torch.fx.experimental.proxy_tensor import make_fx
    m = build(**CASES["all"])
    assert not m._folded()
    assert not torch.compile(lambda: m._folded(), backend="eager")()
    seen = []
    make_fx(lambda x: (seen.append(m._folded()), x)[1])(torch.zeros(1))
    assert seen == [False]
    m.requires_grad_(False)
    torch._dynamo.reset()
    assert torch.compile(lambda: m._folded(), backend="eager")()


@pytest.mark.parametrize("bounds", [{}, BOUNDS], ids=["unbounded", "bounds"])
@pytest.mark.parametrize("case", ["all", "default"])
def test_compiled_frozen_model_is_the_eager_one(case, bounds):
    m = build(**FOLDED[case], **bounds).requires_grad_(False)
    b = collate([graph(structures()["triclinic"], m.rcut, m.species)])
    ref = m(b)
    torch._dynamo.reset()
    out = torch.compile(m, backend="aot_eager")(b)
    assert abs(out["energy"].item() - ref["energy"].item()) < 1e-12 * max(1.0, abs(ref["energy"].item()))
    np.testing.assert_allclose(out["forces"].numpy(), ref["forces"].numpy(), rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("case, bounds", [("default", {}), ("all", {}), ("tiny", {}), ("default", BOUNDS), ("all", BOUNDS)],
                         ids=["default", "all", "tiny", "default-bounds", "all-bounds"])
def test_charge_spin_fold_is_exact(case, bounds):
    m = make(case, **bounds, charge_spin=True)
    at = structures()["molecule"]
    for q, s in ((0, 1), (1, 2), (-2, 3)):
        a = evaluate(m, at, charge=q, spin=s)
        b = evaluate(fold_charge_spin(m, q, s), at)
        assert abs(a["energy"][0] - b["energy"][0]) < 1e-12 * max(1, abs(a["energy"][0]))
        np.testing.assert_allclose(a["forces"], b["forces"], atol=1e-12)
    with pytest.raises(ValueError):
        fold_charge_spin(m, 11, 1)


def test_folded_charge_spin_model_compiled_and_frozen_is_the_eager_one():
    m = fold_charge_spin(build(**ALL, charge_spin=True), 1, 2).requires_grad_(False)
    b = collate([graph(structures()["molecule"], m.rcut, m.species)])
    ref = m(b)
    torch._dynamo.reset()
    out = torch.compile(m, backend="aot_eager")(b)
    assert abs(out["energy"].item() - ref["energy"].item()) < 1e-12 * max(1.0, abs(ref["energy"].item()))
    np.testing.assert_allclose(out["forces"].numpy(), ref["forces"].numpy(), rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("bad", [dict(taper=[5, 6, 3]), dict(taper=[5, 4]), dict(n_message=7), dict(taper=[5, 2, 2]),
                                 dict(n_layers=1), dict(n_layers=0), dict(zbl=False, zbl_switch=10.0), dict(bound=0.0),
                                 dict(bound=-1.0), dict(feat_bound=-2.0), dict(bond_scale=0.0)])
def test_bad_hyper_parameters(bad):
    with pytest.raises(ValueError):
        ILMlip(SPECIES, **{**BASE, **bad})


@pytest.mark.parametrize("case", [*CASES, "tiny"])
def test_checkpoint_round_trip(case, tmp_path):
    """A saved model loads with the same energy."""
    m = make(case)
    hp = HP if case == "tiny" else {**BASE, **CASES[case]}
    hp = {k: list(v) if isinstance(v, tuple) else v for k, v in hp.items()}
    torch.save(dict(model=m.state_dict(), hparams=hp, species=SPECIES), tmp_path / "m.pt")
    m2, _ = load(tmp_path / "m.pt")
    at = structures()["nacl"]
    assert evaluate(m2, at)["energy"][0] == evaluate(m, at)["energy"][0]


def saved(tmp_path, hp):
    m = build()
    torch.save(dict(model=m.state_dict(), hparams={**BASE, "hidden": [8], **hp}, species=SPECIES), tmp_path / "m.pt")
    return tmp_path / "m.pt"