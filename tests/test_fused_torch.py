"""kernels="fused" of the torch model: the descriptor's bond sums and products as segmented polynomials.
On the CPU in float64 they run as "fused-reference" (the same polynomials in plain torch)."""
import copy
import inspect
import json

import pytest
import torch

pytest.importorskip("cuequivariance_torch")

from conftest import CONFIGS, SPECIES, structures, tiny
from test_switches import ALL, BASE, CASES as CASES_SWITCHES, build
from test_bounds import CASES as CASES_BOUNDS
from test_finetune import setup  # noqa: F401  (fixture)
from test_train import HP, make_data
from ilmlip import finetune, fused_torch, train
from ilmlip.model import ILMlip, collate, graph, load

REF = "fused-reference"
TOL = 1e-12


def twin(m):
    """(m, its copy run by the polynomials)."""
    f = copy.deepcopy(m)
    f.kernels = REF
    return m, f


def rel(a, b):
    a, b = a.detach(), b.detach()
    return float((a - b).abs().max() / a.abs().max().clamp(min=1e-300))


def force_loss(m, b):
    """(outputs, gradients of a loss on energy, forces and stress with respect to every parameter): the
    gradient passes the forces' derivative a second time."""
    m.train()
    out = m(b, compute_stress=m.stress)
    loss = out["energy"].pow(2).sum() + out["forces"].pow(2).sum() + (out["stress"].pow(2).sum() if m.stress else 0)
    params = {n: p for n, p in m.named_parameters() if p.requires_grad}
    grads = torch.autograd.grad(loss, list(params.values()), allow_unused=True)
    return out, dict(zip(params, grads))


def assert_same(m, f, b, tol=TOL):
    (a, ga), (c, gc) = force_loss(m, b), force_loss(f, b)
    for k in a:
        assert rel(a[k], c[k]) < tol, k
    assert ga.keys() == gc.keys()
    for k in ga:
        assert (ga[k] is None) == (gc[k] is None), k
        if ga[k] is not None and ga[k].abs().max() > 0:
            assert rel(ga[k], gc[k]) < tol, k


def batch_of(m):
    return collate([graph(a, m.rcut, m.species) for a in structures().values()], dtype=m.head.dtype)


CASES = {**CASES_SWITCHES, **CASES_BOUNDS}
CASES["4_layers_gram"] = dict(ALL, n_layers=4)


@pytest.mark.parametrize("case", CASES)
def test_features_equal_the_torch_path(case):
    m, f = twin(build(**CASES[case]))
    b = batch_of(m)
    assert torch.allclose(m.descriptor(b), f.descriptor(b), rtol=TOL, atol=TOL * 1e-3)
    assert fused_torch.engine(f).polys
    assert_same(m, f, b)


@pytest.mark.parametrize("kw", [dict(), dict(product_order=4), dict(product_order=2), dict(n_layers=3),
                                dict(lmax=3, lmax_feat=2), dict(charge_spin=True)])
def test_untapered_model_equals_the_torch_path(kw):
    """Without taper groups, one bond sum for all channels."""
    m, f = twin(tiny(**kw))
    b = batch_of(m)
    if m.charge_spin:
        b["charge"], b["spin"] = b["charge"] + 1, b["spin"] + 1
    assert_same(m, f, b)


@pytest.mark.parametrize("name", ["S", "L", "XL"])
def test_configs_equal_the_torch_path(name):
    """The configurations (taper groups, pair FiLM, shared pair objects, no back-flow, Gram readout,
    gate, squash and core switch), random weights calibrated by fit_feature_normalisation."""
    h = json.loads((CONFIGS / f"{name}.json").read_text())["hparams"]
    known = set(inspect.signature(ILMlip.__init__).parameters)
    torch.manual_seed(0)
    m = ILMlip(SPECIES, **{k: (tuple(v) if k == "hidden" else v) for k, v in h.items() if k in known})
    with torch.no_grad():
        for p in m.parameters():
            if p.abs().sum() == 0:
                p.normal_(0, 0.1)
        m.head.normal_(0, 0.3)
        m.energy_scale.fill_(0.7)
    m.eval()
    b = batch_of(m)
    m.fit_feature_normalisation([b])
    m, f = twin(m)
    assert m.bounded()
    assert_same(m, f, b)


def test_calibration_agrees():
    m, f = twin(build(**CASES["all"]))
    b = batch_of(m)
    m.fit_feature_normalisation([b])
    f.fit_feature_normalisation([b])
    for k in ("feat_mean", "feat_std", "bond_rms"):
        assert rel(getattr(m, k), getattr(f, k)) < TOL, k


def test_compiled_step_equals_the_torch_path(monkeypatch):
    """CompiledStep traces the position derivative with make_fx and compiles the graph; the gradient
    with respect to the weights then passes the polynomials' second derivative."""
    real = torch.compile
    monkeypatch.setattr(torch, "compile", lambda gm, **kw: real(gm, backend="aot_eager", dynamic=False))
    m, f = twin(build(**CASES["all"]))
    b = batch_of(m)
    res = {}
    for tag, mod in (("torch", m), ("fused", f)):
        e, fo, st = train.CompiledStep(mod, 64, len(structures()), with_stress=True, log=lambda *a: None)(b)
        loss = e.pow(2).sum() + fo.pow(2).sum() + st.pow(2).sum()
        names = [n for n, p in mod.named_parameters()]
        res[tag] = (e, fo, st, dict(zip(names, torch.autograd.grad(loss, list(mod.parameters()), allow_unused=True))))
    for a, c in zip(res["torch"][:3], res["fused"][:3]):
        assert rel(a, c) < TOL
    for k, g in res["torch"][3].items():
        if g is not None and g.abs().max() > 0:
            assert rel(g, res["fused"][3][k]) < TOL, k
    out = m(b, compute_stress=True)
    assert rel(out["forces"].detach(), res["fused"][1].detach()) < TOL      # and the eager torch model's


def test_polynomials_are_those_of_the_cuda_method(monkeypatch):
    """Every polynomial passes the checks of the "uniform_1d" CUDA method (one uniform mode, uniform
    segments, no coefficient modes); its kernel is replaced by a stub that is never called."""
    from cuequivariance_torch.primitives import segmented_polynomial_uniform_1d as mod
    for name, value in (("uniform_1d", lambda *a: None), ("BATCH_DIM_AUTO", -2), ("BATCH_DIM_SHARED", 0),
                        ("BATCH_DIM_BATCHED", 1), ("BATCH_DIM_INDEXED", -1)):
        monkeypatch.setattr(mod, name, value, raising=False)
    for name in ("S", "XL"):
        h = json.loads((CONFIGS / f"{name}.json").read_text())["hparams"]
        known = set(inspect.signature(ILMlip.__init__).parameters)
        m = ILMlip(SPECIES, kernels=REF, **{k: (tuple(v) if k == "hidden" else v) for k, v in h.items() if k in known})
        polys = fused_torch.engine(m).polys
        assert len(polys) > 10
        for p in polys:
            mod.SegmentedPolynomialFromUniform1dJit(p)


def test_kernels_are_not_part_of_the_checkpoint(tmp_path):
    assert ILMlip(SPECIES, rcut=4.0).kernels == "torch"
    assert "kernels" not in train.ARCH_KEYS
    with pytest.raises(ValueError, match="kernels"):
        ILMlip(SPECIES, rcut=4.0, kernels="cueq")
    m, f = twin(build(**CASES["all"]))
    assert list(m.state_dict()) == list(f.state_dict())
    assert all(torch.equal(v, f.state_dict()[k]) for k, v in m.state_dict().items())
    path, known = tmp_path / "ck.pt", inspect.signature(ILMlip.__init__).parameters
    hp = {k: v for k, v in {**BASE, **CASES["all"]}.items() if k in known}
    torch.save(dict(model=f.state_dict(), hparams=hp, species=f.species), path)
    g, _ = load(str(path), kernels=REF)
    assert g.kernels == REF and load(str(path))[0].kernels == "torch"


def test_fused_needs_cuda_and_the_kernels(tmp_path):
    m = build(**CASES["all"])
    m.kernels = "fused"
    if not fused_torch.available():
        with pytest.raises(RuntimeError, match="CUDA"):
            m.descriptor(batch_of(m))
        with pytest.raises(SystemExit, match="CUDA"):
            train.main(["--packed", "x", "--hparams", "S", "--out", str(tmp_path), "--kernels", "fused", "--device", "cpu"])


def test_train_cli_gives_the_same_weights(tmp_path, monkeypatch):
    """--kernels fused-reference trains the same model as torch, with checkpoints of the same keys and hparams."""
    make_data(tmp_path / "data")
    hp = dict(HP, taper=[4, 3, 2], gram_rank=3, bispec_rank=2, bound=2.0, feat_bound=8.0)
    (tmp_path / "hp.json").write_text(json.dumps(dict(hparams=hp)))
    cks, calls, real = {}, [], fused_torch.descriptor
    monkeypatch.setattr(fused_torch, "descriptor", lambda *a: calls.append(1) or real(*a))
    for kernels in ("torch", REF):
        train.main(["--packed", str(tmp_path / "data"), "--hparams", str(tmp_path / "hp.json"), "--out",
                    str(tmp_path / kernels), "--device", "cpu", "--dtype", "float64", "--zbl", "--loss", "huber",
                    "--w-stress", "30", "--steps", "6", "--batch-atoms", "24", "--batch-structs", "4",
                    "--warmup", "2", "--val-every", "3", "--val-batches", "2", "--ckpt-every", "6",
                    "--log-every", "2", "--kernels", kernels])
        cks[kernels] = torch.load(tmp_path / kernels / "checkpoint.pt", weights_only=False)
        assert bool(calls) == (kernels == REF)
    a, b = cks["torch"], cks[REF]
    assert a["hparams"] == b["hparams"] and "kernels" not in a["hparams"] and a["model"].keys() == b["model"].keys()
    for k, v in a["model"].items():
        assert torch.allclose(v, b["model"][k], rtol=1e-8, atol=1e-10), k


@pytest.mark.parametrize("mode", ["naive", "lora"])
def test_finetune_gives_the_same_weights(setup, mode, monkeypatch):
    calls, real = [], fused_torch.descriptor
    monkeypatch.setattr(fused_torch, "descriptor", lambda *a: calls.append(1) or real(*a))
    for kernels in ("torch", REF):
        finetune.main(["--foundation", str(setup / "found.pt"), "--train", str(setup / "train.xyz"),
                                      "--mode", mode, "--epochs", "2", "--batch-size", "3", "--device", "cpu",
                                      "--dtype", "float64", "--out", str(setup / kernels), "--kernels", kernels])
        assert bool(calls) == (kernels == REF)
    a, b = (torch.load(setup / k / "model.pt", weights_only=False)["model"] for k in ("torch", REF))
    assert a.keys() == b.keys()
    for k, v in a.items():
        assert torch.allclose(v, b[k], rtol=1e-8, atol=1e-10), k
