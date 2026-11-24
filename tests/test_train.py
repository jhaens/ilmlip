import io
import json
import os
import pickle
import shutil
import shlex
import signal
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from ase import Atoms
from ase.build import bulk

from conftest import structures, tiny
from ilmlip import evaluate, finetune, fused_torch, runinfo, tracking, train
from ilmlip.data import PackedWriter, composition, lstsq_e0, PackedSplit
from ilmlip.model import ILMlip, collate, graph, load, set_dtype

HP = dict(rcut=4.0, lmax=2, lmax_feat=1, n_basis=4, n_channels=4, product_order=3, lmax_ind=1,
          n_layers=2, n_message=3, embed_dim=4, hidden=[8], lr=5e-3)


def make_data(root, n_train=18, unpaired=False):
    m, rng = tiny(), np.random.default_rng(0)
    for split, n in (("train", n_train), ("val", 6)):
        w = PackedWriter(root / split)
        for k in range(n):
            at = bulk("NaCl", "rocksalt", a=5.64) if k % 2 else bulk("Si", "diamond", a=5.43)
            if unpaired and k % 5 == 0 and k % 2:
                at.numbers[0] = 17                  # a Cl2 cell, so Na and Cl do not always come in pairs
            at.positions += rng.normal(0, 0.1, at.positions.shape)
            o = m(collate([graph(at, 4.0, m.species)]), compute_stress=True)
            w.add([len(at)], Z=at.numbers, pos=at.positions, forces=o["forces"].detach().numpy(),
                  cell=np.asarray(at.cell), pbc=at.pbc[None], energy=[float(o["energy"][0])],
                  stress=o["stress"].detach().numpy(), src=[0])
        w.close()
    tr = PackedSplit(root / "train")
    per = np.asarray(tr.energy) / tr.n_atoms
    (root / "meta.json").write_text(json.dumps(dict(
        sources=["data"], e0=lstsq_e0(tr.Z, tr.offset, tr.energy),
        stats=dict(train=dict(e_per_atom_mean=float(per.mean()), e_per_atom_std=float(per.std()))))))


def history(run):
    """The lines of history.jsonl without their times."""
    rows = [json.loads(line) for line in (run / "history.jsonl").read_text().splitlines()]
    return [{k: v for k, v in r.items() if k != "seconds"} for r in rows]


def test_defaults_are_the_production_recipe():
    a = train.parse_args(["--packed", "x", "--hparams", "S", "--out", "o"])
    assert (a.optimizer, a.lr_adamw, a.muon_rms, a.loss, a.w_energy, a.w_forces, a.w_stress) == (
        "muon", 3e-3, 0.2, "mae", 1, 1, 3)
    assert (a.weight_decay, a.grad_clip, a.batch_atoms, a.batch_structs, a.dtype, a.kernels) == (
        1e-3, 5, 8192, 512, "tf32", "auto")
    assert train.parse_args(["--packed", "x", "--hparams", "S", "--out", "o", "--optimizer", "adamw"]).lr_adamw is None
    for path in (Path(train.__file__).parent / "configs").glob("*.json"):
        h = json.loads(path.read_text())["hparams"]
        assert h["lr"] == 2.5e-3 and not {"weight_decay", "force_frac", "grad_clip"} & set(h), path.name
    lines = []
    assert fused_torch.choose("auto", "cpu", lines.append) == "torch" and len(lines) == 1
    assert fused_torch.choose("fused-reference", "cpu") == "fused-reference"


def test_train_resumes_exactly(tmp_path, monkeypatch):
    make_data(tmp_path / "data")
    (tmp_path / "hp.json").write_text(json.dumps(dict(hparams=HP)))
    args = lambda out: ["--packed", str(tmp_path / "data"), "--hparams", str(tmp_path / "hp.json"),
                        "--out", str(out), "--device", "cpu", "--dtype", "float64", "--zbl",
                        "--loss", "huber", "--w-stress", "30", "--epochs", "2", "--batch-atoms", "24",
                        "--batch-structs", "4", "--warmup", "2", "--val-every", "4", "--val-batches", "2",
                        "--ckpt-every", "3", "--log-every", "2"]
    train.main(args(tmp_path / "a"))
    real = train.lr_at

    def stop_at_5(step, *a, **k):
        if step == 5:
            train._STOP = True
        return real(step, *a, **k)

    monkeypatch.setattr(train, "lr_at", stop_at_5)
    train.main(args(tmp_path / "b"))
    steps = []
    monkeypatch.setattr(train, "lr_at", lambda step, *a, **k: steps.append(step) or real(step, *a, **k))
    train._STOP = False
    train.main(args(tmp_path / "b"))
    a = torch.load(tmp_path / "a" / "checkpoint.pt", weights_only=False)
    b = torch.load(tmp_path / "b" / "checkpoint.pt", weights_only=False)
    assert a["step"] == b["step"] > 5
    assert steps == list(range(6, b["step"]))                  # continues at step 6
    for k in a["model"]:
        assert torch.equal(a["model"][k], b["model"][k])
    # the average goes on with the decay of step 6; a resume takes the reference energies of the weights
    assert all(torch.equal(a["ema"][k], b["ema"][k]) for k in a["ema"] if k != "atomic_energies")
    assert history(tmp_path / "b") == history(tmp_path / "a")
    assert all(np.isfinite(h["e_mae"]) for h in history(tmp_path / "a"))


def test_stop_after_then_resume(tmp_path):
    make_data(tmp_path / "data")
    (tmp_path / "hp.json").write_text(json.dumps(dict(hparams=HP)))
    args = lambda out: ["--packed", str(tmp_path / "data"), "--hparams", str(tmp_path / "hp.json"),
                        "--out", str(out), "--device", "cpu", "--dtype", "float64",
                        "--epochs", "2", "--batch-atoms", "24", "--batch-structs", "4", "--warmup", "2",
                        "--val-every", "4", "--val-batches", "2", "--ckpt-every", "100", "--log-every", "2"]
    train.main(args(tmp_path / "a"))
    train.main(args(tmp_path / "b") + ["--stop-after", "4"])
    ck = torch.load(tmp_path / "b" / "checkpoint.pt", weights_only=False)
    assert ck["step"] == 4 and ck["ema"]
    train.main(args(tmp_path / "b"))
    a = torch.load(tmp_path / "a" / "checkpoint.pt", weights_only=False)
    b = torch.load(tmp_path / "b" / "checkpoint.pt", weights_only=False)
    assert a["step"] == b["step"] > 4
    assert all(torch.equal(a["model"][k], b["model"][k]) for k in a["model"])
    assert history(tmp_path / "b") == history(tmp_path / "a")     # a restart from step 0 logs step 4 twice


def test_resume_after_init_keeps_the_architecture(tmp_path):
    """On resume the --init checkpoint's architecture and ZBL win over --hparams and --zbl."""
    make_data(tmp_path / "data")
    (tmp_path / "hp.json").write_text(json.dumps(dict(hparams=HP)))
    (tmp_path / "other.json").write_text(json.dumps(dict(hparams=dict(HP, n_message=5))))
    args = lambda out, hp, *x: ["--packed", str(tmp_path / "data"), "--hparams", str(tmp_path / hp),
                                "--out", str(out), "--device", "cpu", "--dtype", "float64",
                                "--epochs", "2", "--batch-atoms", "24", "--batch-structs", "4",
                                "--warmup", "2", "--val-every", "4", "--val-batches", "2",
                                "--ckpt-every", "100", "--log-every", "2", *x]
    train.main(args(tmp_path / "base", "hp.json", "--zbl"))
    init = lambda out, *x: args(out, "other.json", "--init", str(tmp_path / "base" / "best.pt"), *x)
    train.main(init(tmp_path / "a"))
    train.main(init(tmp_path / "b", "--stop-after", "4"))
    train.main(init(tmp_path / "b"))
    a = torch.load(tmp_path / "a" / "checkpoint.pt", weights_only=False)
    b = torch.load(tmp_path / "b" / "checkpoint.pt", weights_only=False)
    for ck in (a, b):
        assert ck["hparams"]["zbl"] and ck["hparams"]["n_message"] == 3
    assert a["step"] == b["step"] > 4
    assert all(torch.equal(a["model"][k], b["model"][k]) for k in a["model"])


OFFSETS = {11: 0.3, 14: 0.1, 17: -0.2}        # eV per atom, the second source's energies


def make_two_source_data(root):
    """Structures of a source 'ref' and of a source 'other' whose energies are shifted per element."""
    m, rng = tiny(), np.random.default_rng(0)
    for split, n in (("train", 24), ("val", 8)):
        w = PackedWriter(root / split)
        for k in range(n):
            at = bulk("NaCl", "rocksalt", a=5.64) if k % 2 else bulk("Si", "diamond", a=5.43)
            at.positions += rng.normal(0, 0.1, at.positions.shape)
            o = m(collate([graph(at, 4.0, m.species)]), compute_stress=True)
            src = (k // 2) % 2
            w.add([len(at)], Z=at.numbers, pos=at.positions, forces=o["forces"].detach().numpy(),
                  cell=np.asarray(at.cell), pbc=at.pbc[None],
                  energy=[float(o["energy"][0]) + src * sum(OFFSETS[z] for z in at.numbers)],
                  stress=o["stress"].detach().numpy(), src=[src])
        w.close()
    tr = PackedSplit(root / "train")
    comp, energy, src = composition(tr.Z, tr.n_atoms).astype(float), np.asarray(tr.energy), np.asarray(tr.src)

    def fit(rows):
        cols = np.where(comp[rows].sum(0) > 0)[0]
        e0 = np.linalg.lstsq(comp[rows][:, cols], energy[rows], rcond=None)[0]
        return {int(z): float(v) for z, v in zip(cols, e0)}

    per = energy / tr.n_atoms
    (root / "meta.json").write_text(json.dumps(dict(
        sources=["ref", "other"], e0=fit(slice(None)),
        e0_groups=dict(ref=fit(src == 0), other=fit(src == 1), ref_sources=["ref"]),
        stats=dict(train=dict(e_per_atom_mean=float(per.mean()), e_per_atom_std=float(per.std()))))))


def two_source_args(tmp_path, out, *extra):
    return ["--packed", str(tmp_path / "data"), "--hparams", str(tmp_path / "hp.json"), "--out", str(out),
            "--device", "cpu", "--dtype", "float64", "--batch-atoms", "24", "--batch-structs", "4",
            "--warmup", "2", "--val-every", "4", "--val-batches", "2", "--ckpt-every", "100",
            "--log-every", "2", *extra]


def test_fold_shift_starts_on_the_other_scale(tmp_path):
    """--fold-shift reproduces the reference model on the other sources; a resume does not shift again."""
    make_two_source_data(tmp_path / "data")
    (tmp_path / "hp.json").write_text(json.dumps(dict(hparams=HP)))
    base = tmp_path / "base"
    train.main(two_source_args(tmp_path, base, "--ref-sources", "ref", "--zbl", "--epochs", "2", "--w-stress", "1"))
    ck = torch.load(base / "best.pt", weights_only=False)
    assert ck["src_shift"].abs().max() > 0
    fold = lambda out, *x: two_source_args(tmp_path, out, "--init", str(base / "best.pt"),
                                           "--sources", "other", "--fold-shift", *x)
    train.main(fold(tmp_path / "start", "--steps", "0"))
    start = torch.load(tmp_path / "start" / "checkpoint.pt", weights_only=False)
    assert torch.equal(start["model"]["atomic_energies"],
                       ck["model"]["atomic_energies"] + ck["src_shift"].double())
    ev = lambda *x: evaluate.main([*x, str(tmp_path / "data"), "--sources", "other", "--device", "cpu",
                                   "--dtype", "float64"])["metrics"]["all"]
    shifted, folded = ev(str(base / "best.pt"), "--ema"), ev(str(tmp_path / "start" / "checkpoint.pt"))
    assert folded["n_structures"] == 4
    for k in ("e_mae", "f_mae", "s_mae"):
        assert abs(shifted[k] - folded[k]) < 1e-9 * max(1.0, shifted[k])

    train.main(fold(tmp_path / "a", "--epochs", "2"))
    train.main(fold(tmp_path / "b", "--epochs", "2", "--stop-after", "3"))
    train.main(fold(tmp_path / "b", "--epochs", "2"))
    a = torch.load(tmp_path / "a" / "checkpoint.pt", weights_only=False)
    b = torch.load(tmp_path / "b" / "checkpoint.pt", weights_only=False)
    assert a["step"] == b["step"] > 3
    assert all(torch.equal(a["model"][k], b["model"][k]) for k in a["model"])


def test_fold_shift_refuses_mixed_scales(tmp_path):
    make_two_source_data(tmp_path / "data")
    (tmp_path / "hp.json").write_text(json.dumps(dict(hparams=HP)))
    train.main(two_source_args(tmp_path, tmp_path / "base", "--ref-sources", "ref", "--steps", "2"))
    train.main(two_source_args(tmp_path, tmp_path / "plain", "--steps", "2"))
    init = lambda name: ["--init", str(tmp_path / name / "checkpoint.pt"), "--steps", "0", "--fold-shift"]
    for k, extra in enumerate((init("base"),                                    # no --sources
                               init("base") + ["--sources", "ref,other"],       # a reference source
                               init("base") + ["--sources", "other", "--ref-sources", "ref"],
                               init("plain") + ["--sources", "other"])):        # no offsets
        with pytest.raises(SystemExit):
            train.main(two_source_args(tmp_path, tmp_path / f"bad{k}", *extra))


def test_source_offsets_are_summed_in_float64():
    shift = train.SourceShift(3, [True, False], init=[0.1, 0.2, 0.3])
    b = dict(mu=torch.arange(3).repeat(1000), src=torch.tensor([1]), batch=torch.zeros(3000, dtype=torch.long),
             n_struct=1)
    got = shift(b)
    assert got.dtype == torch.float64
    assert float(got[0]) == pytest.approx(1000 * np.float32([0.1, 0.2, 0.3]).astype(float).sum(), rel=1e-12)


@pytest.mark.parametrize("dtype, tf32, torch_dtype", [("tf32", True, torch.float32), ("float32", False, torch.float32),
                                                       ("float64", False, torch.float64)])
def test_dtype_sets_the_torch_precision_of_train_and_finetune(dtype, tf32, torch_dtype):
    assert train.parse_args(["--packed", "x", "--hparams", "S", "--out", "o"]).dtype == "tf32"
    assert train.parse_args(["--packed", "x", "--hparams", "S", "--out", "o", "--dtype", dtype]).dtype == dtype
    assert finetune.parser().parse_args(["--foundation", "f", "--train", "t", "--dtype", dtype]).dtype == dtype
    # torch's cuBLAS module cannot delete allow_tf32, as monkeypatch's undo may try to
    old = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = not tf32
        assert set_dtype(dtype) == torch_dtype
        assert torch.backends.cuda.matmul.allow_tf32 == torch.backends.cudnn.allow_tf32 == tf32
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = old


def test_compiled_strain_has_no_matrix_product(monkeypatch):
    """The compiled step strains positions and shifts elementwise, as model.forward: a matrix product
    would run in TF32 under --dtype tf32."""
    graphs, real = [], torch.compile
    monkeypatch.setattr(torch, "compile", lambda gm, **kw: graphs.append(gm) or real(gm, backend="aot_eager",
                                                                                     dynamic=False))
    m = tiny()
    b = collate([graph(a, m.rcut, m.species) for a in structures().values() if a.pbc.any()])
    stress = train.CompiledStep(m, 64, 4, with_stress=True, log=lambda *a: None)(b)[2]
    torch.testing.assert_close(stress, m(b, compute_stress=True)["stress"], rtol=1e-10, atol=1e-14)
    aten = torch.ops.aten
    products = [n for n in graphs[0].graph.nodes if n.target in (aten.mm.default, aten.bmm.default)]
    assert products and all(tuple(x.meta["val"].shape[-2:]) != (3, 3) for n in products for x in n.args)


def make_random_data(root, n_train):
    """Small structures with random labels, enough of them for several shuffle blocks."""
    rng = np.random.default_rng(0)
    for split, n in (("train", n_train), ("val", 8)):
        na = rng.integers(2, 5, n)
        A = int(na.sum())
        w = PackedWriter(root / split)
        w.add(na, Z=rng.choice([1, 6, 8], A), pos=rng.uniform(0, 3.0, (A, 3)),
              forces=rng.normal(0, 1, (A, 3)), cell=np.zeros((n, 3, 3)), pbc=np.zeros((n, 3), bool),
              energy=-3.0 * na + rng.normal(0, 0.1, n), stress=np.full((n, 6), np.nan),
              src=np.zeros(n, np.uint8))
        w.close()
    tr = PackedSplit(root / "train")
    per = np.asarray(tr.energy) / tr.n_atoms
    (root / "meta.json").write_text(json.dumps(dict(
        sources=["data"], e0=lstsq_e0(tr.Z, tr.offset, tr.energy),
        stats=dict(train=dict(e_per_atom_mean=float(per.mean()), e_per_atom_std=float(per.std()))))))


def edit_meta(root, **fields):
    """Set fields of meta.json; None removes one."""
    meta = {**json.loads((root / "meta.json").read_text()), **fields}
    (root / "meta.json").write_text(json.dumps({k: v for k, v in meta.items() if v is not None}))


def test_resumed_init_run_keeps_its_start(tmp_path):
    """A stage-2 run stopped before its first step resumes without fitting the whitening again,
    and with --ref-sources without needing the E0 fits in meta.json."""
    make_two_source_data(tmp_path / "data")
    (tmp_path / "hp.json").write_text(json.dumps(dict(hparams=HP)))
    base = tmp_path / "base"
    train.main(two_source_args(tmp_path, base, "--ref-sources", "ref", "--ema", "0", "--steps", "8"))
    edit_meta(tmp_path / "data", e0_groups=None, stats=dict(train=dict(e_per_atom_std=1.0)))
    command = two_source_args(tmp_path, tmp_path / "b", "--init", str(base / "checkpoint.pt"),
                              "--ref-sources", "ref", "--steps", "0")
    train.main(command)
    train.main(command)
    stage1 = torch.load(base / "checkpoint.pt", weights_only=False)["model"]
    stage2 = torch.load(tmp_path / "b" / "checkpoint.pt", weights_only=False)["model"]
    for k in ("feat_mean", "feat_std", "energy_scale"):
        assert torch.equal(stage2[k], stage1[k])


def test_resume_keeps_the_shuffle_order_when_the_residency_changes(tmp_path):
    """Host-resident data are shuffled in blocks, device-resident data are not: a resume
    on the other residency continues with the order of the checkpoint."""
    make_random_data(tmp_path / "data", 1100)                  # more than one block of 1024
    (tmp_path / "hp.json").write_text(json.dumps(dict(hparams=HP)))
    run = lambda out, resident, *x: train.main(two_source_args(tmp_path, out, "--resident", resident,
                                                               "--steps", "12", *x))
    run(tmp_path / "a", "host")
    run(tmp_path / "b", "host", "--stop-after", "6")
    run(tmp_path / "b", "device")
    a = torch.load(tmp_path / "a" / "checkpoint.pt", weights_only=False)
    b = torch.load(tmp_path / "b" / "checkpoint.pt", weights_only=False)
    assert a["step"] == b["step"] == 12
    assert all(torch.equal(a["model"][k], b["model"][k]) for k in a["model"])


def test_evaluate_leaves_out_omol_subsets_without_structures(tmp_path):
    make_two_source_data(tmp_path / "data")
    edit_meta(tmp_path / "data", sources=["ani2x", "biomolecules"])
    (tmp_path / "hp.json").write_text(json.dumps(dict(hparams=HP)))
    train.main(two_source_args(tmp_path, tmp_path / "m", "--steps", "2"))
    evaluate.main([str(tmp_path / "m" / "checkpoint.pt"), str(tmp_path / "data"), "--sources", "ani2x",
                   "--device", "cpu", "--out", str(tmp_path / "e.json")])
    assert set(json.loads((tmp_path / "e.json").read_text())["omol_leaderboard"]) == {"all", "neutral_organics"}


def test_refit_e0_finds_the_energy_offsets(tmp_path):
    """--refit-e0 moves every element's reference energy by the offset the new energies carry;
    the checkpoint's e0 are the refitted ones, e0_fit those of meta.json."""
    make_data(tmp_path / "data", 120, unpaired=True)
    (tmp_path / "hp.json").write_text(json.dumps(dict(hparams=HP)))
    train.main(two_source_args(tmp_path, tmp_path / "base", "--ema", "0", "--steps", "40"))
    shutil.copytree(tmp_path / "data", tmp_path / "shifted")
    sp = PackedSplit(tmp_path / "shifted" / "train")
    offset = np.zeros(101)
    offset[list(OFFSETS)] = list(OFFSETS.values())
    np.save(tmp_path / "shifted" / "train" / "energy.npy", sp.energy + composition(sp.Z, sp.n_atoms) @ offset)
    train.main(two_source_args(tmp_path, tmp_path / "ft", "--init", str(tmp_path / "base" / "checkpoint.pt"),
                               "--refit-e0", "1000", "--steps", "0", "--packed", str(tmp_path / "shifted")))
    ck = lambda run: torch.load(tmp_path / run / "checkpoint.pt", weights_only=False)
    delta = ck("ft")["model"]["atomic_energies"] - ck("base")["model"]["atomic_energies"]
    for z, o in OFFSETS.items():
        assert abs(delta[z - 1] - o) < 0.01
    assert ck("ft")["e0"] == ck("ft")["model"]["atomic_energies"].tolist()
    fit = json.loads((tmp_path / "shifted" / "meta.json").read_text())["e0"]
    assert ck("ft")["e0_fit"] == [fit.get(str(z), 0.0) for z in range(1, 101)]


def test_committee_evaluation_matches_a_hand_computation(tmp_path):
    make_data(tmp_path / "data")
    (tmp_path / "hp.json").write_text(json.dumps(dict(hparams=HP)))
    cks = []
    for seed in ("0", "1"):
        train.main(two_source_args(tmp_path, tmp_path / seed, "--zbl", "--steps", "6", "--seed", seed))
        cks.append(str(tmp_path / seed / "checkpoint.pt"))
    res = evaluate.main([*cks, str(tmp_path / "data"), "--device", "cpu", "--dtype", "float64",
                         "--dump", str(tmp_path / "d.npz")])
    models = [load(c)[0].double() for c in cks]
    sp = PackedSplit(tmp_path / "data" / "val")
    n, E, F, ref = np.asarray(sp.n_atoms), [], [], []
    for k in range(len(sp)):
        lo, hi = sp.offset[k], sp.offset[k + 1]
        at = Atoms(numbers=sp.Z[lo:hi], positions=np.asarray(sp.pos[lo:hi], float),
                   cell=np.asarray(sp.cell[k], float), pbc=sp.pbc[k])
        outs = [m(collate([graph(at, m.rcut, m.species)])) for m in models]
        E.append([float(o["energy"][0]) for o in outs])
        F.append(np.stack([o["forces"].detach().numpy() for o in outs]))
        ref.append(np.asarray(sp.forces[lo:hi], float))
    E, cat = np.array(E), np.concatenate
    e_err, e_std = np.abs(E.mean(1) - sp.energy) / n, E.std(1) / n
    f_diff = [f.mean(0) - r for f, r in zip(F, ref)]
    f_err = [np.linalg.norm(d, axis=1) for d in f_diff]
    f_dev = [np.sqrt(((f - f.mean(0)) ** 2).sum(-1).mean(0)) for f in F]
    approx = lambda x: pytest.approx(x, rel=1e-6)
    assert res["metrics"]["all"]["e_mae"] == approx(1e3 * e_err.mean())
    assert res["metrics"]["all"]["f_mae"] == approx(1e3 * np.abs(cat(f_diff)).mean())
    for member, e in zip(res["members"], E.T):
        assert member["e_mae"] == approx(1e3 * (np.abs(e - sp.energy) / n).mean())
    for key, dev, err in (("e", e_std, e_err), ("f", cat(f_dev), cat(f_err))):
        assert res["spread"][key]["mean_spread"] == approx(1e3 * dev.mean())
        assert res["spread"][key]["mean_error"] == approx(1e3 * err.mean())
        assert res["spread"][key]["pearson_r"] == approx(np.corrcoef(dev, err)[0, 1])
    d = np.load(tmp_path / "d.npz")
    for key, want in (("e_err", e_err), ("e_std", e_std), ("f_dev_max", [x.max() for x in f_dev])):
        np.testing.assert_allclose(d[key], want, rtol=1e-6)


def test_train_refuses_missing_energy_statistics(tmp_path):
    """Without stats.train.e_per_atom_std the default energy scale would be NaN."""
    make_data(tmp_path / "data")
    edit_meta(tmp_path / "data", stats=None)
    (tmp_path / "hp.json").write_text(json.dumps(dict(hparams=HP)))
    with pytest.raises(SystemExit, match="e_per_atom_std"):
        train.main(two_source_args(tmp_path, tmp_path / "a", "--epochs", "1"))
    train.main(two_source_args(tmp_path, tmp_path / "b", "--epochs", "1", "--scale-from", "forces"))
    assert torch.load(tmp_path / "b" / "checkpoint.pt", weights_only=False)["model"]["energy_scale"] > 0


def test_ema_starts_from_the_whitened_model(tmp_path):
    make_data(tmp_path / "data")
    (tmp_path / "hp.json").write_text(json.dumps(dict(hparams=HP)))
    train.main(two_source_args(tmp_path, tmp_path / "a", "--steps", "6"))
    ck = torch.load(tmp_path / "a" / "checkpoint.pt", weights_only=False)
    for k in ("feat_mean", "feat_std", "energy_scale"):
        torch.testing.assert_close(ck["ema"][k].double(), ck["model"][k], rtol=1e-6, atol=0)


def test_ema_is_float64_and_warms_up():
    """The decay after step n is min(decay, (1 + n) / (10 + n)); a float32 model gets a float64
    average, whose steps of 1e-3 x 1.4 eV do not round away at a reference energy of -7e4 eV."""
    m = tiny()
    ema = train.EMA(m, 0.999)
    start = {k: v.clone() for k, v in ema.shadow.items()}
    with torch.no_grad():
        for p in m.parameters():
            p.add_(1.0)
    ema.update(m, 1)
    d = 2 / 11
    for k, v in m.state_dict().items():
        if v.is_floating_point():
            torch.testing.assert_close(ema.shadow[k], d * start[k] + (1 - d) * v, rtol=1e-14, atol=1e-14)
    m = tiny(dtype=torch.float32)
    with torch.no_grad():
        m.atomic_energies.fill_(-70042.32)
        ema = train.EMA(m, 0.999)
        m.atomic_energies.fill_(-70040.94)
    for n in range(10 ** 7, 10 ** 7 + 1000):
        ema.update(m, n)
    assert {v.dtype for v in ema.shadow.values()} == {torch.float64}
    a, b = float(np.float32(-70042.32)), float(np.float32(-70040.94))
    torch.testing.assert_close(ema.shadow["atomic_energies"],
                               torch.full((len(m.species),), b + (a - b) * 0.999 ** 1000, dtype=torch.float64),
                               rtol=0, atol=1e-6)


def test_an_average_is_read_with_the_reference_energies_of_the_weights(tmp_path, capsys):
    """Reference energies that the average never moved (float32 checkpoints): evaluate --ema,
    --init and a resume take those of the weights, and say so."""
    make_data(tmp_path / "data")
    (tmp_path / "hp.json").write_text(json.dumps(dict(hparams=HP)))
    run = lambda out, *x: train.main(two_source_args(tmp_path, out, *x))
    run(tmp_path / "a", "--steps", "8", "--stop-after", "4")
    path = tmp_path / "a" / "checkpoint.pt"
    ev = lambda: evaluate.main([str(path), str(tmp_path / "data"), "--ema", "--device", "cpu",
                                "--dtype", "float64"])["metrics"]["all"]
    before = ev()
    ck = torch.load(path, weights_only=True)
    raw = ck["model"]["atomic_energies"]
    ck["ema"]["atomic_energies"] = raw.clone()
    ck["ema"]["atomic_energies"][10] -= 1.5                   # Na
    torch.save(ck, path)
    capsys.readouterr()
    assert ev() == before
    run(tmp_path / "init", "--init", str(path), "--steps", "0")
    assert torch.equal(torch.load(tmp_path / "init" / "checkpoint.pt", weights_only=True)["model"]["atomic_energies"],
                       raw)
    run(tmp_path / "a", "--steps", "4")
    assert torch.equal(torch.load(path, weights_only=True)["ema"]["atomic_energies"], raw)
    assert capsys.readouterr().out.count("differ by up to 1.5 eV, Na") == 3


def test_ema_of_a_resumed_run_without_one_starts_from_its_weights(tmp_path):
    make_data(tmp_path / "data")
    (tmp_path / "hp.json").write_text(json.dumps(dict(hparams=HP)))
    run = lambda *x: train.main(two_source_args(tmp_path, tmp_path / "a", *x))
    run("--ema", "0", "--steps", "8", "--stop-after", "4")
    run("--steps", "4")
    ck = torch.load(tmp_path / "a" / "checkpoint.pt", weights_only=True)
    assert all(torch.equal(ck["ema"][k], v) for k, v in ck["model"].items() if v.is_floating_point())


def test_resume_after_a_kill_logs_every_step_once(tmp_path, monkeypatch):
    """Killed after a validation that no checkpoint holds: that line goes and the step is logged
    again; the times go on from the checkpoint's, or from the last line kept if it has none."""
    make_data(tmp_path / "data")
    (tmp_path / "hp.json").write_text(json.dumps(dict(hparams=HP)))
    command = lambda out: two_source_args(tmp_path, out, "--steps", "8", "--val-every", "2", "--ckpt-every", "3")
    real = train.lr_at

    def killed_at_5(step, *a, **k):
        if step == 5:
            raise RuntimeError("killed")
        return real(step, *a, **k)

    monkeypatch.setattr(train, "lr_at", killed_at_5)
    with pytest.raises(RuntimeError, match="killed"):
        train.main(command(tmp_path / "a"))
    monkeypatch.setattr(train, "lr_at", real)
    assert [h["step"] for h in history(tmp_path / "a")] == [2, 4]
    shutil.copytree(tmp_path / "a", tmp_path / "old")
    ck = torch.load(tmp_path / "a" / "checkpoint.pt", weights_only=True)
    assert ck["step"] == 3
    torch.save(dict(ck, seconds=1e6), tmp_path / "a" / "checkpoint.pt")
    torch.save({k: v for k, v in ck.items() if k != "seconds"}, tmp_path / "old" / "checkpoint.pt")
    rows = (tmp_path / "old" / "history.jsonl").read_text().splitlines()
    (tmp_path / "old" / "history.jsonl").write_text("".join(
        json.dumps(dict(json.loads(r), seconds=s)) + "\n" for r, s in zip(rows, (5e5, 9e5))))
    for run, start in (("a", 1e6), ("old", 5e5)):
        train.main(command(tmp_path / run))
        rows = [json.loads(line) for line in (tmp_path / run / "history.jsonl").read_text().splitlines()]
        assert [r["step"] for r in rows] == [2, 4, 6, 8]
        assert all(start < r["seconds"] < start + 1e3 for r in rows[1:])
        assert start < torch.load(tmp_path / run / "checkpoint.pt", weights_only=True)["seconds"] < start + 1e3


def test_resume_keeps_the_weight_decay_and_takes_the_optimiser_of_this_device(tmp_path, capsys):
    """A checkpoint of a GPU run (fused) resumes on the CPU; the weight decay stays the checkpoint's
    when --weight-decay changes (the offsets' group keeps none)."""
    make_two_source_data(tmp_path / "data")
    (tmp_path / "hp.json").write_text(json.dumps(dict(hparams=HP)))
    command = two_source_args(tmp_path, tmp_path / "a", "--ref-sources", "ref", "--steps", "8",
                              "--optimizer", "adamw", "--weight-decay", "1e-9")
    train.main(command + ["--stop-after", "4"])
    path = tmp_path / "a" / "checkpoint.pt"
    ck = torch.load(path, weights_only=True)
    for g in ck["optimizer"]["param_groups"]:
        g["fused"] = True
    torch.save(ck, path)
    capsys.readouterr()
    train.main(command[:-1] + ["1e-3"])
    groups = torch.load(path, weights_only=True)["optimizer"]["param_groups"]
    assert [(g["fused"], g["weight_decay"]) for g in groups] == [(None, 1e-9), (None, 0.0)]
    assert "weight decay 1e-09 of the checkpoint kept" in capsys.readouterr().out


def test_sigusr2_stops_like_sigterm_unless_ignored(tmp_path, monkeypatch):
    """LSF's run-limit warning finishes the step and writes a checkpoint, but stays ignored where
    the process starts with it ignored (a launcher that relays SIGTERM); main() puts the
    handlers back, also when it fails."""
    make_data(tmp_path / "data")
    (tmp_path / "hp.json").write_text(json.dumps(dict(hparams=HP)))
    real = train.lr_at

    def warned_at_5(step, *a, **k):
        if step == 5:
            signal.raise_signal(signal.SIGUSR2)
        return real(step, *a, **k)

    monkeypatch.setattr(train, "lr_at", warned_at_5)
    caught = []
    mine = lambda *a: caught.append(a)
    term, usr2 = signal.getsignal(signal.SIGTERM), signal.signal(signal.SIGUSR2, mine)
    try:
        train.main(two_source_args(tmp_path, tmp_path / "a", "--steps", "12"))
        assert signal.getsignal(signal.SIGUSR2) is mine and signal.getsignal(signal.SIGTERM) is term
        signal.signal(signal.SIGUSR2, signal.SIG_IGN)
        train.main(two_source_args(tmp_path, tmp_path / "b", "--steps", "12"))
        assert signal.getsignal(signal.SIGUSR2) is signal.SIG_IGN
        with pytest.raises(SystemExit):
            train.main(two_source_args(tmp_path, tmp_path / "c", "--sources", "nothing"))
        assert signal.getsignal(signal.SIGTERM) is term
    finally:
        signal.signal(signal.SIGUSR2, usr2)
    assert not caught
    for run, step in (("a", 6), ("b", 12)):
        assert torch.load(tmp_path / run / "checkpoint.pt", weights_only=True)["step"] == step


class Payload:
    """An object in a pickle: loading a checkpoint must not run code."""


def test_checkpoints_load_as_weights_only(tmp_path):
    """Every file train.py writes loads with weights_only=True; a file that unpickles an object is
    refused as --init and as the run's own checkpoint."""
    make_data(tmp_path / "data")
    (tmp_path / "hp.json").write_text(json.dumps(dict(hparams=HP)))
    base = tmp_path / "base"
    train.main(two_source_args(tmp_path, base, "--zbl", "--epochs", "1"))
    for f in (base / "checkpoint.pt", base / "best.pt", *(base / "epochs").glob("*.pt")):
        torch.load(f, weights_only=True)
    bad = dict(torch.load(base / "best.pt", weights_only=True), note=Payload())
    torch.save(bad, tmp_path / "bad.pt")
    with pytest.raises(pickle.UnpicklingError):
        train.main(two_source_args(tmp_path, tmp_path / "x", "--init", str(tmp_path / "bad.pt"), "--steps", "0"))
    (tmp_path / "y").mkdir()
    torch.save(bad, tmp_path / "y" / "checkpoint.pt")
    with pytest.raises(pickle.UnpicklingError):
        train.main(two_source_args(tmp_path, tmp_path / "y", "--steps", "0"))


def test_charges_and_spins_outside_the_model_tables_are_reported(tmp_path, capsys):
    """A model with charge and spin tables clamps them to its nearest row: training and evaluation say so."""
    make_data(tmp_path / "data")
    (tmp_path / "hp.json").write_text(json.dumps(dict(hparams=HP)))
    train.main(two_source_args(tmp_path, tmp_path / "m", "--charge-spin", "--steps", "2"))
    charge = np.zeros(18, np.int8)
    charge[3] = -11                                       # the default tables: |charge| <= 10, 2S+1 <= 11
    np.save(tmp_path / "data" / "train" / "charge.npy", charge)
    spin = np.ones(6, np.uint8)
    spin[2] = 12
    np.save(tmp_path / "data" / "val" / "spin.npy", spin)
    capsys.readouterr()
    train.main(two_source_args(tmp_path, tmp_path / "n", "--charge-spin", "--steps", "2"))
    assert "train: 1 structures with charges -11..0 or multiplicities 1..1 outside" in capsys.readouterr().out
    evaluate.main([str(tmp_path / "m" / "checkpoint.pt"), str(tmp_path / "data"), "--device", "cpu"])
    assert "val: 1 structures with charges 0..0 or multiplicities 1..12 outside" in capsys.readouterr().out


def test_edge_counts_too_small_are_an_error(tmp_path):
    """Counts below the edges of a batch (a cache from before stamps, taken as it is) would drop
    edges: training, its validation sample and evaluate stop, naming the file."""
    make_data(tmp_path / "data")
    (tmp_path / "hp.json").write_text(json.dumps(dict(hparams=HP)))
    train.main(two_source_args(tmp_path, tmp_path / "a", "--steps", "2"))
    for split in ("train", "val"):
        cache = tmp_path / "data" / split / "n_edges_rc4.npy"
        np.save(cache, np.load(cache) - 1)
        cache.with_suffix(".json").unlink()
        with pytest.raises(RuntimeError, match=f"{split}/n_edges_rc4.npy"):
            train.main(two_source_args(tmp_path, tmp_path / split, "--steps", "4"))
    with pytest.raises(RuntimeError, match="val/n_edges_rc4.npy"):
        evaluate.main([str(tmp_path / "a" / "checkpoint.pt"), str(tmp_path / "data"), "--device", "cpu"])


def test_dropped_edges_are_found_before_a_checkpoint_without_a_log_step(tmp_path):
    make_data(tmp_path / "data")
    (tmp_path / "hp.json").write_text(json.dumps(dict(hparams=HP)))
    train.main(two_source_args(tmp_path, tmp_path / "a", "--steps", "2"))
    cache = tmp_path / "data" / "train" / "n_edges_rc4.npy"
    np.save(cache, np.load(cache) - 1)
    cache.with_suffix(".json").unlink()
    with pytest.raises(RuntimeError, match="train/n_edges_rc4.npy"):
        train.main(two_source_args(tmp_path, tmp_path / "b", "--steps", "3", "--log-every", "100", "--val-every", "100"))
    assert not (tmp_path / "b" / "checkpoint.pt").exists()


def test_a_run_that_died_before_its_first_checkpoint_starts_its_history_again(tmp_path, monkeypatch):
    make_data(tmp_path / "data")
    (tmp_path / "hp.json").write_text(json.dumps(dict(hparams=HP)))
    command = two_source_args(tmp_path, tmp_path / "a", "--steps", "8", "--val-every", "2", "--ckpt-every", "100")
    real = train.lr_at

    def killed_at_5(step, *a, **k):
        if step == 5:
            raise RuntimeError("killed")
        return real(step, *a, **k)

    monkeypatch.setattr(train, "lr_at", killed_at_5)
    with pytest.raises(RuntimeError, match="killed"):
        train.main(command)
    monkeypatch.setattr(train, "lr_at", real)
    assert not (tmp_path / "a" / "checkpoint.pt").exists()
    train.main(command)
    assert [h["step"] for h in history(tmp_path / "a")] == [2, 4, 6, 8]


def tracked_args(tmp_path, out, *extra):
    return ["--packed", str(tmp_path / "data"), "--hparams", str(tmp_path / "hp.json"), "--out", str(out),
            "--device", "cpu", "--dtype", "float64", "--epochs", "2", "--batch-atoms", "24",
            "--batch-structs", "4", "--warmup", "2", "--val-every", "4", "--val-batches", "2",
            "--ckpt-every", "100", "--log-every", "2", *extra]


def test_tracking_that_cannot_run_changes_nothing(tmp_path, monkeypatch, capsys):
    """Not installed, or failing on the first call: the run is the run without --track."""
    make_data(tmp_path / "data")
    (tmp_path / "hp.json").write_text(json.dumps(dict(hparams=HP)))
    train.main(tracked_args(tmp_path, tmp_path / "a"))
    monkeypatch.setattr(tracking, "available", lambda backend: False)
    train.main(tracked_args(tmp_path, tmp_path / "b", "--track", "wandb"))
    assert "wandb is not installed" in capsys.readouterr().out

    class Broken:
        def __getattr__(self, name):
            raise ConnectionError("offline")

    def start(self, *a):
        self.run, self.id = Broken(), "x"

    monkeypatch.setattr(tracking, "available", lambda backend: True)
    monkeypatch.setattr(tracking.Tracker, "_start_wandb", start)
    train.main(tracked_args(tmp_path, tmp_path / "c", "--track", "wandb"))
    assert "no live tracking from here on" in capsys.readouterr().out
    a = torch.load(tmp_path / "a" / "checkpoint.pt", weights_only=False)
    for out in "bc":
        b = torch.load(tmp_path / out / "checkpoint.pt", weights_only=False)
        assert a["step"] == b["step"] and all(torch.equal(a["model"][k], b["model"][k]) for k in a["model"])
        assert (tmp_path / out / "history.jsonl").read_text().count("\n") == \
            (tmp_path / "a" / "history.jsonl").read_text().count("\n")
    assert not (tmp_path / "b" / "tracking.json").exists()


def test_mlflow_continues_the_run_on_resume(tmp_path, monkeypatch):
    pytest.importorskip("mlflow")
    from mlflow.tracking import MlflowClient
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    make_data(tmp_path / "data")
    (tmp_path / "hp.json").write_text(json.dumps(dict(hparams=HP)))
    run = lambda *x: train.main(tracked_args(tmp_path, tmp_path / "a", "--track", "mlflow",
                                             "--track-project", "test", *x))
    run("--stop-after", "4")
    run()
    c = MlflowClient()
    runs = c.search_runs([c.get_experiment_by_name("test").experiment_id])
    assert len(runs) == 1 and runs[0].info.status == "FINISHED"
    rid = runs[0].info.run_id
    assert json.loads((tmp_path / "a" / "tracking.json").read_text()) == dict(backend="mlflow", id=rid)
    last = torch.load(tmp_path / "a" / "checkpoint.pt", weights_only=False)["step"]
    assert max(m.step for m in c.get_metric_history(rid, "train/loss")) == last - last % 2
    hist = [json.loads(l) for l in (tmp_path / "a" / "history.jsonl").read_text().splitlines()]
    f_mae = c.get_metric_history(rid, "val/f_mae")
    assert sorted(m.value for m in f_mae) == pytest.approx(sorted(h["f_mae"] for h in hist))
    assert runs[0].data.params["args/hparams"] == str(tmp_path / "hp.json")
    assert "best/f_rmse" in runs[0].data.metrics


def test_wandb_offline(tmp_path, monkeypatch):
    pytest.importorskip("wandb")
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("WANDB_SILENT", "true")
    make_data(tmp_path / "data")
    (tmp_path / "hp.json").write_text(json.dumps(dict(hparams=HP)))
    train.main(tracked_args(tmp_path, tmp_path / "a", "--track", "wandb", "--stop-after", "4"))
    train.main(tracked_args(tmp_path, tmp_path / "a", "--track", "wandb"))
    rid = json.loads((tmp_path / "a" / "tracking.json").read_text())["id"]
    runs = sorted((tmp_path / "a" / "wandb").glob("offline-run-*"))
    assert len(runs) == 2 and all(r.name.split("-")[3] == rid for r in runs)    # "-1": two runs in one second


def recipe_args(tmp_path, out, *extra):
    return ["--packed", str(tmp_path / "data"), "--hparams", str(tmp_path / "hp.json"), "--out", str(out),
            "--device", "cpu", "--dtype", "float64", "--w-stress", "5", "--epochs", "2",
            "--batch-atoms", "24", "--batch-structs", "4", "--warmup", "2", "--val-every", "4",
            "--val-batches", "2", "--ckpt-every", "3", "--log-every", "2", *extra]


def test_muon_takes_the_weight_matrices_and_resumes_exactly(tmp_path, capsys):
    make_data(tmp_path / "data")
    (tmp_path / "hp.json").write_text(json.dumps(dict(hparams=HP)))
    cmd = recipe_args(tmp_path, tmp_path / "b", "--optimizer", "muon", "--loss", "mae")
    train.main(recipe_args(tmp_path, tmp_path / "a", "--optimizer", "muon", "--loss", "mae"))
    train.main(cmd + ["--stop-after", "5"])
    log = capsys.readouterr().out
    line = next(x for x in log.splitlines() if x.startswith("[optim] Muon"))
    for name in ("edge_weights", "chem0.weight", "reduces.0.mix0", "film_weights", "gram_proj.0", "readout.0.weight"):
        assert f" {name} (" in line
    for name in ("embed_proj", "elem_bias", "head", "atomic_energies", "readout.0.bias"):
        assert f" {name}" not in line
    ck = torch.load(tmp_path / "b" / "checkpoint.pt", weights_only=True)
    assert ck["step"] == 5 and ck["optimizer"]["muon"]["state"]
    train.main(cmd)
    a = torch.load(tmp_path / "a" / "checkpoint.pt", weights_only=False)
    b = torch.load(tmp_path / "b" / "checkpoint.pt", weights_only=False)
    assert a["step"] == b["step"] > 5
    assert all(torch.equal(a["model"][k], b["model"][k]) for k in a["model"])
    assert all(np.isfinite(r["e_mae"]) for r in history(tmp_path / "a"))
    with pytest.raises(SystemExit, match="not of --optimizer adamw"):
        train.main(recipe_args(tmp_path, tmp_path / "b", "--optimizer", "adamw"))


def test_muon_update_is_orthogonal_per_matrix():
    w = torch.randn(3, 5, 7, dtype=torch.float64, requires_grad=True)
    opt = train.Muon([w], lr=1.0, momentum=0.0, rms=1.0)
    w.grad = torch.randn_like(w)
    before = w.detach().clone()
    opt.step()
    d = (before - w.detach()) / 7 ** 0.5
    s = torch.linalg.svdvals(d)
    assert s.min() > 0.5 and s.max() < 1.5


def test_muon_graph_is_the_per_tensor_step(tmp_path):
    shapes = [(28, 8), (8, 32), (3, 16, 14), (2, 9, 5), (200, 130), (130, 200), (256, 406), (406, 256)]
    torch.manual_seed(0)
    start = [torch.randn(*s, dtype=torch.float64) for s in shapes]
    ps = {g: [torch.nn.Parameter(w.clone()) for w in start] for g in (False, True)}
    opts = {g: train.Muon(ps[g], lr=0.01, weight_decay=1e-3, graph=g) for g in (False, True)}
    for step in range(4):
        grads = [torch.randn_like(w) for w in start]
        if step == 2:
            grads[1][0, 0] = float("nan")
        for g, opt in opts.items():
            for p, d in zip(ps[g], grads):
                p.grad = d.clone()
            opt.found_inf = torch.tensor(float(step == 2), dtype=torch.float64)
            opt.param_groups[0]["lr"] = 0.01 * (step + 1)
            opt.step()
    assert all(torch.allclose(a, b, rtol=0, atol=1e-13) for a, b in zip(ps[False], ps[True]))
    mg = opts[True]._graphs[0]
    assert [tuple(b.shape) for b, _ in mg.buckets] == [(7, 14, 32), (2, 130, 200), (2, 256, 406)]
    buf = io.BytesIO()
    torch.save(opts[True].state_dict(), buf)          # as in a checkpoint: no shared tensors
    buf.seek(0)
    sd = torch.load(buf, weights_only=True)
    again = [torch.nn.Parameter(p.detach().clone()) for p in ps[True]]
    resumed = train.Muon(again, lr=0.01, weight_decay=1e-3, graph=True)
    resumed.load_state_dict(sd)
    for p, q in zip(ps[True], again):
        p.grad = q.grad = torch.ones_like(p)
    opts[True].step()
    resumed.step()
    assert all(torch.equal(p, q) for p, q in zip(ps[True], again))


def test_muon_graph_trains_like_the_per_tensor_muon(tmp_path):
    make_data(tmp_path / "data")
    (tmp_path / "hp.json").write_text(json.dumps(dict(hparams=HP)))
    for g in ("muon-graph", "no-muon-graph"):
        train.main(recipe_args(tmp_path, tmp_path / g, "--optimizer", "muon", "--loss", "mae", f"--{g}"))
    a = torch.load(tmp_path / "muon-graph" / "checkpoint.pt", weights_only=True)
    b = torch.load(tmp_path / "no-muon-graph" / "checkpoint.pt", weights_only=True)
    assert all(torch.allclose(a["model"][k], b["model"][k], rtol=1e-10, atol=1e-12) for k in a["model"])


def muon_state(order, shapes):
    torch.manual_seed(0)
    ps = {n: torch.nn.Parameter(torch.randn(*shapes[n], dtype=torch.float64)) for n in shapes}
    opt = train.Muon([ps[n] for n in order], lr=0.01, names=list(order))
    for n in order:
        ps[n].grad = torch.randn_like(ps[n])
    opt.step()
    buf = io.BytesIO()
    torch.save(opt.state_dict(), buf)
    buf.seek(0)
    return {n: opt.state[ps[n]]["momentum_buffer"].clone() for n in order}, torch.load(buf, weights_only=True)


SHAPES = dict(a=(8, 8), b=(28, 8), c=(8, 8), d=(3, 16, 14), e=(256, 256))


def test_muon_state_loads_by_name_in_another_order():
    before, sd = muon_state("abcde", SHAPES)
    ps = {n: torch.nn.Parameter(torch.zeros(*SHAPES[n], dtype=torch.float64)) for n in SHAPES}
    opt = train.Muon([ps[n] for n in "ecbda"], lr=0.01, names=list("ecbda"))
    opt.load_state_dict(sd)
    assert all(torch.equal(opt.state[ps[n]]["momentum_buffer"], before[n]) for n in SHAPES)


def test_muon_resumes_in_a_process_with_another_hash_seed(tmp_path):
    import subprocess
    import sys
    make_data(tmp_path / "data")
    (tmp_path / "hp.json").write_text(json.dumps(dict(hparams=HP)))
    run = lambda seed, *more: subprocess.run(
        [sys.executable, "-c", "import sys; from ilmlip import train; train.main(sys.argv[1:])",
         *recipe_args(tmp_path, tmp_path / "b", "--optimizer", "muon", "--loss", "mae", *more)],
        env=dict(os.environ, PYTHONHASHSEED=str(seed)), check=True, capture_output=True, text=True)
    run(1, "--stop-after", "5")
    out = run(2).stdout
    assert "[resume] step 5" in out


def test_wsd_schedule_and_loss_options(tmp_path):
    lr = lambda s: train.lr_at(s, 1.0, 10, 100, schedule="wsd", decay_frac=0.2)
    assert lr(9) == 1.0 and lr(10) == lr(79) == 1.0 and lr(100) == pytest.approx(0.02)
    assert lr(90) == pytest.approx(0.5 + 0.01)
    assert train.lr_at(50, 1.0, 10, 100) == train.lr_at(50, 1.0, 10, 100, schedule="cosine")
    make_data(tmp_path / "data")
    (tmp_path / "hp.json").write_text(json.dumps(dict(hparams=HP)))
    train.main(recipe_args(tmp_path, tmp_path / "a", "--schedule", "wsd", "--optimizer", "adamw",
                           "--weight-decay", "1e-4", "--grad-clip", "0.5"))
    ck = torch.load(tmp_path / "a" / "checkpoint.pt", weights_only=True)
    assert ck["hparams"]["weight_decay"] == 1e-4 and ck["hparams"]["grad_clip"] == 0.5
    assert ck["optimizer"]["param_groups"][0]["weight_decay"] == 1e-4
    assert np.isfinite(history(tmp_path / "a")[-1]["f_mae"])


def test_architecture_keys_pass_through_to_the_model(tmp_path, monkeypatch):
    seen = {}
    real = train.ILMlip
    monkeypatch.setattr(train, "ILMlip", lambda species, **kw: seen.update(kw) or real(species, **kw))
    make_data(tmp_path / "data")
    (tmp_path / "hp.json").write_text(json.dumps(dict(hparams=dict(HP, gram_rank=2, bispec_rank=3))))
    train.main(recipe_args(tmp_path, tmp_path / "a", "--stop-after", "1"))
    assert (seen["gram_rank"], seen["bispec_rank"]) == (2, 3)
    hp = json.loads((tmp_path / "a" / "hparams.json").read_text())
    ck = torch.load(tmp_path / "a" / "checkpoint.pt", weights_only=True)["hparams"]
    assert (hp["gram_rank"], hp["bispec_rank"]) == (ck["gram_rank"], ck["bispec_rank"]) == (2, 3)


@pytest.mark.parametrize("kind", ["adamw", "muon"])
def test_no_e0_decay_keeps_the_reference_energies(kind):
    m = tiny()
    m.atomic_energies.data.fill_(-1000.0)
    h = dict(lr=0.1, weight_decay=0.5)
    log = lambda *a: None
    for flag, moved in ((False, True), (True, False)):
        before = m.atomic_energies.detach().clone()
        opt = train.make_optimizer(m, None, h, kind, 0.95, 0.2, torch.device("cpu"), log, e0_decay=not flag)
        for p in m.parameters():
            p.grad = torch.zeros_like(p)
        opt.step()
        assert (not torch.equal(m.atomic_energies, before)) == moved


def test_a_loss_per_term_is_the_loss_when_all_three_agree(tmp_path, capsys):
    make_data(tmp_path / "data")
    (tmp_path / "hp.json").write_text(json.dumps(dict(hparams=HP)))
    one = ["--loss", "huber"]
    each = one + sum(([f"--loss-{q}", "huber"] for q in ("energy", "forces", "stress")), [])
    train.main(recipe_args(tmp_path, tmp_path / "a", *one))
    train.main(recipe_args(tmp_path, tmp_path / "b", *each))
    train.main(recipe_args(tmp_path, tmp_path / "c", *one, "--loss-forces", "mae"))
    a, b, c = (torch.load(tmp_path / x / "checkpoint.pt", weights_only=True) for x in "abc")
    assert all(torch.equal(a["model"][k], b["model"][k]) for k in a["model"])
    assert history(tmp_path / "a") == history(tmp_path / "b")
    assert not all(torch.equal(a["model"][k], c["model"][k]) for k in a["model"])
    assert a["args"]["loss_forces"] is None and c["args"]["loss_forces"] == "mae"
    assert "energy huber, forces mae, stress huber" in capsys.readouterr().out


def test_gate_factors_are_logged_at_validation(tmp_path, monkeypatch, capsys):
    gates = lambda self, b: [torch.full((b["n_atoms"],), 0.5), torch.linspace(0.25, 1.0, b["n_atoms"])]
    monkeypatch.setattr(ILMlip, "gate_factors", gates, raising=False)
    make_data(tmp_path / "data")
    (tmp_path / "hp.json").write_text(json.dumps(dict(hparams=HP)))
    train.main(recipe_args(tmp_path, tmp_path / "a", "--epochs", "1"))
    assert "gate factors (mean/min) L0: 0.500/0.500  L1: " in capsys.readouterr().out
    row = history(tmp_path / "a")[0]
    assert row["gate_mean/0"] == 0.5 and row["gate_min/1"] == 0.25 and 0.25 < row["gate_mean/1"] < 1.0


def test_the_log_starts_with_the_command(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sys, "orig_argv", ["python", "-m", "ilmlip.train", "--out", "my run", "--note", "it's"])
    runinfo.called("ilmlip.train", None)
    head = runinfo.header()
    assert head[0] == "command    " + shlex.join(sys.orig_argv) and "'my run'" in head[0]
    assert "torch " + torch.__version__ in head[1] and "ilmlip " in head[1]
    make_data(tmp_path / "data")
    (tmp_path / "hp.json").write_text(json.dumps(dict(hparams=HP)))
    argv = recipe_args(tmp_path, tmp_path / "a", "--stop-after", "1")
    train.main(argv)
    out = capsys.readouterr().out.splitlines()
    assert out[0] == "command    " + shlex.join([sys.executable, "-m", "ilmlip.train", *map(str, argv)])
    assert out[1] == head[1]


def test_lr_adamw_sets_only_the_adamw_part(tmp_path, capsys, monkeypatch):
    make_data(tmp_path / "data")
    (tmp_path / "hp.json").write_text(json.dumps(dict(hparams=HP)))
    seen = []
    step = train.MuonAdamW.step

    def spy(self, *a, **k):
        seen.append(({g["lr"] for g in self.adamw.param_groups}, {g["lr"] for g in self.muon.param_groups}))
        return step(self, *a, **k)
    monkeypatch.setattr(train.MuonAdamW, "step", spy)
    train.main(recipe_args(tmp_path, tmp_path / "a", "--optimizer", "muon", "--lr", "1e-3", "--lr-adamw", "3e-4",
                           "--warmup", "1", "--stop-after", "3"))
    (aw,), (mu,) = seen[-1]
    assert aw == pytest.approx(0.3 * mu) and "AdamW part at peak lr" in capsys.readouterr().out
    with pytest.raises(SystemExit, match="lr-adamw"):
        train.main(recipe_args(tmp_path, tmp_path / "b", "--optimizer", "adamw", "--lr-adamw", "3e-4",
                               "--stop-after", "1"))
