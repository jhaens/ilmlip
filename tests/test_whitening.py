"""fit_feature_normalisation with overflowing and non-finite features; the average of a model with an infinite buffer."""
import torch

from conftest import SPECIES, structures, tiny
from ilmlip import train
from ilmlip.model import collate, graph


def batch(model, dtype):
    return collate([graph(a, model.rcut, model.species) for a in structures().values()], dtype=dtype)


def test_whitening_matches_mean_and_std():
    m = tiny()
    b = batch(m, torch.float64)
    m.fit_feature_normalisation([b, b])
    f = torch.cat([m.descriptor(b), m.descriptor(b)]).detach()
    assert torch.allclose(m.feat_mean, f.mean(0), rtol=1e-12, atol=1e-14)
    assert torch.allclose(m.feat_std, f.std(0).clamp(min=1e-6), rtol=1e-12, atol=1e-14)


def test_whitening_of_features_whose_squares_overflow_float32(monkeypatch):
    m = tiny(dtype=torch.float32)
    b = batch(m, torch.float32)
    big = m.descriptor(b).detach()
    big[0, 3] = 1e25
    monkeypatch.setattr(m, "descriptor", lambda _: big)
    assert not torch.isfinite((big - big.mean(0)).pow(2).sum(0)).all()      # what a GPU's std sums
    m.fit_feature_normalisation([b])
    assert torch.isfinite(m.feat_mean).all() and torch.isfinite(m.feat_std).all()
    assert m.feat_std[3] > 1e22


def test_whitening_leaves_out_atoms_with_non_finite_features(monkeypatch):
    m = tiny()
    b = batch(m, torch.float64)
    f = m.descriptor(b).detach()
    bad = f.clone()
    bad[1, 0] = float("inf")
    monkeypatch.setattr(m, "descriptor", lambda _: bad)
    m.fit_feature_normalisation([b])
    keep = torch.arange(len(f)) != 1
    assert torch.allclose(m.feat_mean, f[keep].mean(0), rtol=1e-12, atol=1e-14)


def test_average_of_a_model_with_an_infinite_buffer():
    m = tiny()
    with torch.no_grad():
        m.feat_std[2] = float("inf")
    ema = train.EMA(m, 0.999)
    ema.update(m, 1)
    assert torch.isnan(ema.shadow["feat_std"][2])
    ck = dict(model=m.state_dict(), ema=ema.shadow, species=SPECIES)
    weights = train.ema_weights(ck, m, log=lambda *_: None)
    assert set(weights) == {n for n, _ in m.named_parameters()}
    assert all(torch.isfinite(v).all() for v in weights.values())
