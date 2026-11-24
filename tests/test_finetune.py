import csv
import pickle
from pathlib import Path

import numpy as np
import pytest
import torch
from ase.build import bulk
from ase.io import read, write

from conftest import SPECIES, tiny
from ilmlip import finetune
from ilmlip.calculator import TorchCalculator
from ilmlip.model import ILMlip, collate, graph, load
from ilmlip.replay import prepare_replay

HP = dict(rcut=4.0, lmax=2, lmax_feat=1, n_basis=5, n_channels=4, product_order=3, lmax_ind=1, n_layers=2, n_message=3,
          hidden=[8], embed_dim=4, zbl=True, lr=1e-3)


@pytest.fixture
def setup(tmp_path):
    m, rng = tiny(), np.random.default_rng(0)
    torch.save(dict(model=m.state_dict(), hparams=HP, species=SPECIES), tmp_path / "found.pt")
    frames = []
    for k in range(8):
        at = bulk("NaCl", "rocksalt", a=5.64).repeat((1, 1, 2))
        at.positions += rng.normal(0, 0.1, at.positions.shape)
        o = m(collate([graph(at, 4.0, SPECIES)]), compute_stress=False)
        at.info["energy"] = float(o["energy"][0]) + 0.3
        at.arrays["forces"] = o["forces"].detach().numpy() * 1.1
        frames.append(at)
    write(tmp_path / "train.xyz", frames, format="extxyz")
    replay = [bulk("Si", "diamond", a=5.43) for _ in range(10)]
    for a in replay:
        a.positions += rng.normal(0, 0.03, a.positions.shape)
    write(tmp_path / finetune.REPLAY_FILES[0], replay[1:], format="extxyz")
    write(tmp_path / finetune.REPLAY_FILES[1], replay[:1], format="extxyz")
    return tmp_path


@pytest.mark.parametrize("mode", ["naive", "lora", "multihead", "scratch"])
def test_modes(setup, mode):
    out = setup / mode
    s = finetune.main(["--foundation", str(setup / "found.pt"), "--train", str(setup / "train.xyz"),
                       "--mode", mode, "--epochs", "2", "--batch-size", "3", "--device", "cpu",
                       "--dtype", "float64", "--out", str(out), "--replay", str(setup)])
    assert np.isfinite(s["fine_tuned"]["train"]["f_mae"])
    m, ck = load(out / "model.pt")
    assert ck["finetune"]["mode"] == mode
    # the saved model (LoRA updates merged) reproduces the training error
    calc = TorchCalculator(out / "model.pt", dtype="float64")
    err = []
    for k, at in enumerate(read(setup / "train.xyz", ":")):
        if k in s["split_valid"]:
            continue
        f = at.get_forces().copy()       # extxyz labels come back as a calculator
        at.calc = calc
        err.append(np.abs(at.get_forces() - f).ravel())
    assert abs(np.concatenate(err).mean() * 1000 - s["fine_tuned"]["train"]["f_mae"]) < 1e-6


def wise_args(setup, out, *more, epochs=2, dtype="float64"):
    return ["--foundation", str(setup / "found.pt"), "--train", str(setup / "train.xyz"), "--epochs", str(epochs),
            "--batch-size", "3", "--device", "cpu", "--dtype", dtype, "--replay", str(setup),
            "--out", str(setup / out), *more]


def weights(path):
    return torch.load(path, weights_only=False)["model"]


def test_wise(setup):
    finetune.main(wise_args(setup, "plain"))
    s = finetune.main(wise_args(setup, "wise", "--wise", "2"))
    m0, m1, avg = (weights(setup / "wise" / f) for f in ("member_00/model.pt", "member_01/model.pt", "model.pt"))
    # member 0 is the fine-tune without --wise; member 1 sees the batches in another order
    plain = weights(setup / "plain" / "model.pt")
    assert all(torch.equal(v, plain[k]) for k, v in m0.items())
    assert not all(torch.equal(v, m1[k]) for k, v in m0.items())
    assert all(torch.allclose(v, (m0[k] + m1[k]) / 2, rtol=1e-15, atol=0) for k, v in avg.items()
               if v.is_floating_point())
    assert s["wise"] == [str((setup / "wise" / f"member_0{k}").resolve()) for k in range(2)]
    assert np.isfinite(s["fine_tuned"]["valid"]["f_mae"]) and s["drift"]
    # the saved average reproduces the validation error of the summary
    calc = TorchCalculator(setup / "wise" / "model.pt", dtype="float64")
    err = []
    for k in s["split_valid"]:
        at = read(setup / "train.xyz", k)
        f = at.get_forces().copy()
        at.calc = calc
        err.append(np.abs(at.get_forces() - f).ravel())
    assert abs(np.concatenate(err).mean() * 1000 - s["fine_tuned"]["valid"]["f_mae"]) < 1e-6

    # rerun: only the unfinished member trains
    (setup / "wise" / "model.pt").unlink()
    (setup / "wise" / "member_01" / "summary.json").unlink()
    t0 = (setup / "wise" / "member_00" / "model.pt").stat().st_mtime_ns
    again = finetune.main(wise_args(setup, "wise", "--wise", "2"))
    assert (setup / "wise" / "member_00" / "model.pt").stat().st_mtime_ns == t0
    assert (setup / "wise" / "member_01.incomplete").is_dir()
    assert again["fine_tuned"]["valid"] == s["fine_tuned"]["valid"]
    (setup / "wise" / "model.pt").unlink()
    with pytest.raises(SystemExit, match="other settings"):
        finetune.main(wise_args(setup, "wise", "--wise", "2", epochs=3))
    with pytest.raises(SystemExit, match="naive"):
        finetune.main(wise_args(setup, "lora", "--wise", "2", "--mode", "lora"))


def test_wise_parallel(setup):
    seq = finetune.main(wise_args(setup, "seq", "--wise", "3"))
    par = finetune.main(wise_args(setup, "par", "--wise", "3", "--wise-parallel", "3"))
    a, b = weights(setup / "seq" / "model.pt"), weights(setup / "par" / "model.pt")
    assert all(torch.allclose(v, b[k], rtol=1e-12, atol=1e-14) for k, v in a.items() if v.is_floating_point())
    assert abs(seq["fine_tuned"]["valid"]["f_mae"] - par["fine_tuned"]["valid"]["f_mae"]) < 1e-9
    assert all((setup / "par" / f"member_0{k}" / "summary.json").is_file() for k in range(3))


def test_wise_diverged_member_is_not_trained_again(setup):
    argv = wise_args(setup, "div", "--wise", "2", "--lr", "1e8", epochs=30, dtype="float32")
    with pytest.raises(SystemExit, match="finite loss"):
        finetune.main(argv)
    t0 = (setup / "div" / "member_00" / "summary.json").stat().st_mtime_ns
    with pytest.raises(SystemExit, match="finite loss"):
        finetune.main(argv)
    assert (setup / "div" / "member_00" / "summary.json").stat().st_mtime_ns == t0
    assert not (setup / "div" / "model.pt").exists()


def test_init_starts_another_cycle_from_a_fine_tune(setup):
    first = finetune.main(wise_args(setup, "first"))
    init = str(setup / "first" / "model.pt")
    # a vanishing learning rate keeps the weights, so the second cycle ends where it started
    s = finetune.main(wise_args(setup, "second", "--init", init, "--lr", "1e-30", epochs=1))
    assert s["zero_shot"]["valid"]["f_mae"] == pytest.approx(first["fine_tuned"]["valid"]["f_mae"], rel=1e-12)
    assert s["e0"] == first["e0"]
    # the drift is measured against the foundation
    assert s["drift"]["f_mae"] > 0
    assert s["drift"]["f_mae"] == pytest.approx(first["drift"]["f_mae"], rel=1e-9)
    with pytest.raises(SystemExit, match="model_pt_head.pt"):
        finetune.main(wise_args(setup, "multi", "--init", init, "--mode", "multihead"))


@pytest.mark.parametrize("mode", ["multihead", "scratch"])
def test_init_continues_multihead_and_scratch(setup, mode):
    first = finetune.main(wise_args(setup, f"{mode}_1", "--mode", mode))
    init = str(setup / f"{mode}_1" / "model.pt")
    s = finetune.main(wise_args(setup, f"{mode}_2", "--mode", mode, "--init", init, "--lr", "1e-30", epochs=1))
    assert s["zero_shot"]["valid"]["f_mae"] == pytest.approx(first["fine_tuned"]["valid"]["f_mae"], rel=1e-12)
    if mode == "multihead":
        # the replay head continues the first cycle's
        a, b = (weights(setup / f"{mode}_{k}" / "model_pt_head.pt") for k in (1, 2))
        assert all(torch.allclose(v, b[k], rtol=1e-12, atol=1e-14) for k, v in a.items() if v.is_floating_point())


def test_batches_on_the_device_are_the_collated_batches(setup):
    _, graphs, _ = finetune.read_data([str(setup / "train.xyz")], {}, 4.0, SPECIES)
    _, more, _ = finetune.read_data([str(setup / finetune.REPLAY_FILES[0])], {}, 4.0, SPECIES)
    graphs += more                                   # other sizes, no labels
    graphs[1]["energy"] = np.nan
    views = finetune.on_device(graphs, "cpu", torch.float64)
    for group in ([0], [3, 9, 1], list(range(len(graphs)))):
        a, b = finetune.join([views[k] for k in group]), collate([graphs[k] for k in group])
        assert a.keys() == b.keys()
        for k, v in a.items():
            if torch.is_tensor(v):
                torch.testing.assert_close(v, b[k], rtol=0, atol=0, equal_nan=True)
            else:
                assert v == b[k]


def head_files(setup):
    """Three training files as from other DFT settings (energy offsets, force scales): NaCl twice, then Si
    labelled by the foundation."""
    calc = TorchCalculator(setup / "found.pt", dtype="float64")
    si = [a for f in finetune.REPLAY_FILES for a in read(setup / f, ":")]
    paths = []
    for name, frames, de, sf in (("a", read(setup / "train.xyz", ":"), 0.0, 1.0),
                                 ("b", read(setup / "train.xyz", ":"), -2.0, 0.8), ("c", si, 1.5, 1.2)):
        for at in frames:
            at.calc = at.calc or calc
            e, f = at.get_potential_energy(), at.get_forces()
            at.calc = None
            at.info["energy"], at.arrays["forces"] = e + de, f * sf
        write(setup / f"{name}.xyz", frames, format="extxyz")
        paths.append(str(setup / f"{name}.xyz"))
    return paths


HEADS = (("a", "model.pt"), ("b", "model_b.pt"), ("c", "model_c.pt"))


@pytest.mark.parametrize("replay", ["folder", "none", "weight0"])
def test_multihead_trains_a_head_per_file(setup, replay):
    out = setup / f"heads_{replay}"
    more = {"folder": [str(setup)], "none": ["none"], "weight0": [str(setup), "--replay-weight", "0"]}[replay]
    s = finetune.main(["--foundation", str(setup / "found.pt"), "--train", *head_files(setup), "--mode", "multihead",
                       "--epochs", "2", "--batch-size", "3", "--device", "cpu", "--dtype", "float64",
                       "--out", str(out), "--replay", *more])
    assert (out / "model_pt_head.pt").is_file() == (replay == "folder")
    assert bool(s["drift"]) == (replay != "none")
    assert s["steps"] == 2 * 3 * 3                  # an epoch: every file's 7 or 9 training structures, batches of 3
    sds = {h: weights(out / f) for h, f in HEADS}
    # the heads share all but the readout and the E0s
    for k, v in sds["a"].items():
        assert all(torch.equal(v, sds[h][k]) for h in "bc") != finetune.is_head(k), k
    # a head changes the per-element rows of its own file's elements only
    found = weights(setup / "found.pt")
    for h, own in (("a", ("Na", "Cl")), ("b", ("Na", "Cl")), ("c", ("Si",))):
        for k in ("elem_bias", "head", "atomic_energies"):
            for el in ("Na", "Cl", "Si"):
                i = SPECIES.index(el)
                assert torch.equal(sds[h][k][i].double(), found[k][i].double()) != (el in own), (h, k, el)
    # each file is its head: it reproduces that head's training errors
    for h, f in HEADS:
        calc = TorchCalculator(out / f, dtype="float64")
        de, df = [], []
        for k, at in enumerate(read(setup / f"{h}.xyz", ":")):
            if k in s["split_valid"][h]:
                continue
            e, frc = at.get_potential_energy(), at.get_forces().copy()
            at.calc = calc
            de.append(abs(at.get_potential_energy() - e) / len(at))
            df.append(np.abs(at.get_forces() - frc).ravel())
        assert abs(np.mean(de) * 1000 - s["fine_tuned"][f"train {h}"]["e_mae"]) < 1e-6
        assert abs(np.concatenate(df).mean() * 1000 - s["fine_tuned"][f"train {h}"]["f_mae"]) < 1e-6


def test_foundation_by_name(setup, monkeypatch):
    models = setup / "models"
    models.mkdir()
    (setup / "found.pt").rename(models / "OAM-S-omat.pt")
    monkeypatch.setenv("ILMLIP_MODELS", str(models))
    out = setup / "by_name"
    finetune.main(["--foundation", "oam_s_omat", "--train", str(setup / "train.xyz"), "--epochs", "1",
                   "--device", "cpu", "--dtype", "float64", "--out", str(out), "--replay", "none"])
    assert load(out / "model.pt")[1]["finetune"]["foundation"] == str(models / "OAM-S-omat.pt")
    with pytest.raises(SystemExit, match="neither a checkpoint file nor a pretrained model"):
        finetune.main(["--foundation", "OAM-XXL", "--train", str(setup / "train.xyz"), "--device", "cpu",
                       "--out", str(setup / "x")])


def test_multihead_argument_errors(setup):
    a, b, _ = head_files(setup)
    base = ["--foundation", str(setup / "found.pt"), "--mode", "multihead", "--device", "cpu",
            "--out", str(setup / "x")]
    for more, why in ((["--train", a, "--replay", "none"], "second head"),
                      (["--train", a, "--replay", str(setup), "--replay-weight", "0"], "second head"),
                      (["--train", a, a, "--replay", "none"], "must differ"),
                      (["--train", a, b, "--valid", a], "one file per --train file")):
        with pytest.raises(SystemExit, match=why):
            finetune.main(base + more)


def test_init_continues_every_head(setup):
    files = head_files(setup)
    run = lambda out, *more, epochs="2": finetune.main(
        ["--foundation", str(setup / "found.pt"), "--train", *files, "--mode", "multihead", "--epochs", epochs,
         "--batch-size", "3", "--device", "cpu", "--dtype", "float64", "--replay", str(setup),
         "--out", str(setup / out), *more])
    first = run("h1")
    s = run("h2", "--init", str(setup / "h1" / "model.pt"), "--lr", "1e-30", epochs="1")
    for h, _ in HEADS:
        for q in ("e_mae", "f_mae"):
            assert s["zero_shot"][f"valid {h}"][q] == pytest.approx(first["fine_tuned"][f"valid {h}"][q], rel=1e-12)
    for f in ("model_c.pt", "model_pt_head.pt"):
        a, b = weights(setup / "h1" / f), weights(setup / "h2" / f)
        assert all(torch.allclose(v, b[k], rtol=1e-12, atol=1e-14) for k, v in a.items() if v.is_floating_point())
    (setup / "h1" / "model_b.pt").unlink()
    with pytest.raises(SystemExit, match="model_b.pt"):
        run("h3", "--init", str(setup / "h1" / "model.pt"))


@pytest.mark.parametrize("mode", ["naive", "lora", "multihead", "heads"])
def test_the_compiled_step_is_the_eager_step(setup, mode, monkeypatch):
    real = torch.compile
    # the compiled graph without code generation, so that the CPU test stays fast
    monkeypatch.setattr(torch, "compile", lambda gm, **kw: real(gm, backend="aot_eager", dynamic=kw.get("dynamic")))
    more = ["--mode", mode] if mode != "heads" else ["--mode", "multihead", "--train", *head_files(setup)[:2]]
    s = {c: finetune.main(wise_args(setup, f"{mode}_{c}", *more, f"--{c}")) for c in ("compile", "no-compile")}
    assert "training step compiled" in (setup / f"{mode}_compile" / "log.txt").read_text()
    for split, errs in s["compile"]["fine_tuned"].items():
        for k, v in errs.items():
            assert v == pytest.approx(s["no-compile"]["fine_tuned"][split][k], rel=1e-9)


def test_a_step_without_labels_is_skipped(setup):
    frames = read(setup / "train.xyz", ":")
    frames[2].calc = None                            # extxyz labels come back as a calculator
    write(setup / "partly.xyz", frames, format="extxyz")
    s = finetune.main(["--foundation", str(setup / "found.pt"), "--train", str(setup / "partly.xyz"),
                       "--epochs", "2", "--batch-size", "1", "--valid-fraction", "0", "--device", "cpu",
                       "--dtype", "float64", "--out", str(setup / "partly"), "--replay", "none"])
    assert (setup / "partly" / "log.txt").read_text().count("1 of 8 steps skipped") == 2
    assert s["epochs_run"] == 2 and np.isfinite(s["fine_tuned"]["train"]["f_mae"])


def test_a_diverged_run_stops(setup):
    s = finetune.main(["--foundation", str(setup / "found.pt"), "--train", str(setup / "train.xyz"),
                       "--epochs", "30", "--batch-size", "3", "--lr", "1e8", "--device", "cpu",
                       "--dtype", "float32", "--out", str(setup / "diverged"), "--replay", "none"])
    assert s["epochs_run"] < 30 and (setup / "diverged" / "model.pt").is_file()


def test_mlflow_tracks_epochs_and_wise_groups(setup, monkeypatch):
    pytest.importorskip("mlflow")
    from mlflow.tracking import MlflowClient
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{setup / 'mlflow.db'}")
    s = finetune.main(wise_args(setup, "wise", "--wise", "2", "--track", "mlflow", "--track-project", "ft"))
    c = MlflowClient()
    runs = {r.info.run_name: r for r in c.search_runs([c.get_experiment_by_name("ft").experiment_id])}
    assert sorted(runs) == ["wise/average", "wise/member_00", "wise/member_01"]
    assert all(r.data.tags["group"] == "wise" and r.info.status == "FINISHED" for r in runs.values())
    member = runs["wise/member_00"]
    assert sorted(m.step for m in c.get_metric_history(member.info.run_id, "valid/f_mae")) == [0, 1, 2]
    assert sorted(m.step for m in c.get_metric_history(member.info.run_id, "train/loss")) == [1, 2]
    assert runs["wise/average"].data.metrics["fine_tuned/valid/f_mae"] == pytest.approx(s["fine_tuned"]["valid"]["f_mae"])
    # tracking does not change the members: the finished ones are kept
    (setup / "wise" / "model.pt").unlink()
    assert finetune.main(wise_args(setup, "wise", "--wise", "2"))["fine_tuned"] == s["fine_tuned"]


def test_replay_needs_ten_structures(tmp_path):
    write(tmp_path / "few.xyz", [bulk("Cu", "fcc", a=3.6)] * 5, format="extxyz")
    with pytest.raises(SystemExit, match="at least 10"):
        prepare_replay(tmp_path / "few.xyz", 5, 0, tmp_path / "out")


def learned(s):
    """The fine-tune's validation force error is clearly below the zero-shot error."""
    return s["fine_tuned"]["valid"]["f_mae"] < 0.5 * s["zero_shot"]["valid"]["f_mae"]


@pytest.mark.parametrize("mode, lr, learns", [("naive", "1e-3", True), ("multihead", None, True), ("naive", "0", False)])
def test_fine_tuning_learns(setup, mode, lr, learns):
    # the training forces are the foundation's x 1.1; a run that does not train must fail the check
    s = finetune.main(wise_args(setup, mode, "--mode", mode, *(["--lr", lr] if lr else []), epochs=20))
    assert learned(s) == learns


def test_muon_fine_tunes_the_weight_matrices(setup):
    s = {g: finetune.main(wise_args(setup, f"muon_{g}", "--optimizer", "muon", "--lr", "3e-4", f"--{g}", epochs=20))
         for g in ("muon-graph", "no-muon-graph")}
    log = (setup / "muon_muon-graph" / "log.txt").read_text()
    assert "Muon on " in log and learned(s["muon-graph"])
    for split, errs in s["muon-graph"]["fine_tuned"].items():
        for k, v in errs.items():
            assert v == pytest.approx(s["no-muon-graph"]["fine_tuned"][split][k], rel=1e-9)
    finetune.main(wise_args(setup, "muon_lora", "--optimizer", "muon", "--mode", "lora"))
    assert "Adam alone" in (setup / "muon_lora" / "log.txt").read_text()


def test_lora_trains_low_rank_updates_of_the_frozen_foundation(setup):
    s = finetune.main(wise_args(setup, "lora", "--mode", "lora", "--lora-rank", "2", epochs=20))
    assert learned(s)
    found, tuned = weights(setup / "found.pt"), weights(setup / "lora" / "model.pt")
    adapted = {n for n, p in load(setup / "found.pt")[0].named_parameters() if p.dim() >= 2}
    ranks = {}
    for k, v in found.items():
        if k in adapted:
            d = (tuned[k] - v).reshape(v.shape[0], -1)
            # merging W + scale B A rounds at the ulp of W, far below atol
            ranks[k] = int(torch.linalg.matrix_rank(d, atol=1e-12, rtol=0))
        elif k == "atomic_energies":
            assert tuned[k].tolist() == [s["e0"][el] for el in SPECIES]
        else:
            assert torch.equal(tuned[k], v), k
    assert ranks and all(1 <= r <= 2 for r in ranks.values()), ranks


def mixed(path, n_si=20, n_fe=60):
    """Si structures (the foundation's elements) among Fe structures (an element it lacks)."""
    rng = np.random.default_rng(1)
    frames = [bulk("Si", "diamond", a=5.43) for _ in range(n_si)] + [bulk("Fe", "bcc", a=2.87) for _ in range(n_fe)]
    for a in frames:
        a.positions += rng.normal(0, 0.03, a.positions.shape)
    write(path, [frames[i] for i in rng.permutation(len(frames))])
    return path


def drawn(folder):
    """The replay structures in folder: their number and elements."""
    frames = [a for f in finetune.REPLAY_FILES for a in read(folder / f, ":")]
    return len(frames), {el for a in frames for el in a.get_chemical_symbols()}


@pytest.mark.parametrize("suffix", [".xyz", ".traj"])
def test_replay_is_drawn_among_the_foundations_elements(tmp_path, suffix):
    src = mixed(tmp_path / f"mixed{suffix}")
    prepare_replay(src, 15, 0, tmp_path / "out", species=SPECIES)
    assert drawn(tmp_path / "out") == (15, {"Si"})
    # when every structure qualifies, the draw is the one without species
    prepare_replay(src, 15, 0, tmp_path / "any")
    prepare_replay(src, 15, 0, tmp_path / "all", species=SPECIES + ["Fe"])
    for f in finetune.REPLAY_FILES:
        a, b = read(tmp_path / "any" / f, ":"), read(tmp_path / "all" / f, ":")
        assert len(a) == len(b) and all(np.array_equal(x.positions, y.positions) for x, y in zip(a, b))
    with pytest.raises(SystemExit, match="5 structures with only the foundation's elements"):
        prepare_replay(mixed(tmp_path / f"few{suffix}", n_si=5), 15, 0, tmp_path / "few", species=SPECIES)


def test_fine_tunes_draw_replay_among_the_foundations_elements(setup):
    src = str(mixed(setup / "mixed.xyz"))
    run = lambda out, *more: finetune.main(
        ["--foundation", str(setup / "found.pt"), "--train", str(setup / "train.xyz"), "--epochs", "1",
         "--device", "cpu", "--dtype", "float64", "--replay", src, "--replay-n", "15", "--out", str(setup / out), *more])
    run("plain")
    assert drawn(setup / "plain") == (15, {"Si"})
    run("wise", "--wise", "2")
    assert drawn(setup / "wise") == (15, {"Si"})
    finetune.main(["prepare-replay", "--source", src, "--n", "15", "--foundation", str(setup / "found.pt"),
                   "--out", str(setup / "cli")])
    assert drawn(setup / "cli") == (15, {"Si"})


def test_replay_takes_only_the_elements_the_foundation_was_trained_on(setup):
    from ilmlip.replay import species_of
    ck = torch.load(setup / "found.pt", weights_only=True)
    ck["model"]["atomic_energies"][SPECIES.index("Si")] = 0.0      # listed, but without a reference energy
    torch.save(ck, setup / "no_si.pt")
    assert species_of(str(setup / "no_si.pt")) == [s for s in SPECIES if s != "Si"]
    with pytest.raises(SystemExit, match="0 structures with only the foundation's elements"):
        finetune.main(["--foundation", str(setup / "no_si.pt"), "--train", str(setup / "train.xyz"),
                       "--epochs", "1", "--device", "cpu", "--dtype", "float64", "--replay",
                       str(mixed(setup / "mixed.xyz")), "--replay-n", "15", "--out", str(setup / "out")])


def test_replay_frames_of_untrained_elements_are_left_out(setup):
    ck = torch.load(setup / "found.pt", weights_only=True)
    ck["model"]["atomic_energies"][SPECIES.index("Si")] = 0.0
    torch.save(ck, setup / "no_si.pt")
    rng = np.random.default_rng(3)
    frames = [bulk("Si", "diamond", a=5.43) for _ in range(12)] + [bulk("NaCl", "rocksalt", a=5.64) for _ in range(12)]
    for a in frames:
        a.positions += rng.normal(0, 0.03, a.positions.shape)
    write(setup / "both.xyz", frames)
    prepare_replay(setup / "both.xyz", 24, 0, setup / "replay")        # a folder drawn for another foundation
    finetune.main(["--foundation", str(setup / "no_si.pt"), "--train", str(setup / "train.xyz"), "--mode", "multihead",
                   "--epochs", "1", "--device", "cpu", "--dtype", "float64", "--replay", str(setup / "replay"),
                   "--out", str(setup / "out")])
    log = (setup / "out" / "log.txt").read_text()
    assert "structures have elements the foundation was not trained on, left out" in log
    assert (setup / "out" / "model_pt_head.pt").exists()


def test_scratch_whitens_a_random_sample(setup, monkeypatch):
    rng, frames = np.random.default_rng(2), []
    for name, cell, a in (("NaCl", "rocksalt", 5.64), ("Si", "diamond", 5.43)):     # sorted by system
        for _ in range(70):
            at = bulk(name, cell, a=a)
            at.positions += rng.normal(0, 0.05, at.positions.shape)
            at.info["energy"], at.arrays["forces"] = 0.0, np.zeros((len(at), 3))
            frames.append(at)
    write(setup / "sorted.xyz", frames, format="extxyz")

    class Whitened(Exception):
        pass

    def whiten(self, batches):
        raise Whitened(batches)
    monkeypatch.setattr(ILMlip, "fit_feature_normalisation", whiten)
    samples = []
    for k in range(2):
        with pytest.raises(Whitened) as got:
            finetune.main(["--foundation", str(setup / "found.pt"), "--train", str(setup / "sorted.xyz"),
                           "--mode", "scratch", "--valid-fraction", "0", "--device", "cpu", "--dtype", "float64",
                           "--out", str(setup / f"scratch_{k}")])
        samples.append(got.value.args[0])
    assert sum(b["n_struct"] for b in samples[0]) == 64
    assert {SPECIES[i] for b in samples[0] for i in b["mu"].tolist()} == {"Na", "Cl", "Si"}
    # seeded: the same sample again
    assert all(torch.equal(a["positions"], b["positions"]) for a, b in zip(*samples))


def test_a_validation_set_without_labels_is_no_validation_set(setup):
    frames = read(setup / "train.xyz", ":")
    for at in frames:
        at.calc = None                               # extxyz labels come back as a calculator
    write(setup / "unlabelled.xyz", frames, format="extxyz")
    s = finetune.main(["--foundation", str(setup / "found.pt"), "--train", str(setup / "train.xyz"),
                       "--valid", str(setup / "unlabelled.xyz"), "--epochs", "3", "--batch-size", "3",
                       "--device", "cpu", "--dtype", "float64", "--out", str(setup / "unlabelled"), "--replay", "none"])
    assert s["best_epoch"] == 3
    assert (setup / "unlabelled" / "log.txt").read_text().count("validation structures have no labels") == 1


def test_energies_and_pseudolabels_are_batched_by_atoms(model, monkeypatch):
    graphs = [graph(bulk("NaCl", "rocksalt", a=5.64).repeat((1, 1, n)), 4.0, SPECIES) for n in (1, 3, 1, 1, 2, 4, 1)]
    sizes, real = [], finetune.collate
    monkeypatch.setattr(finetune, "collate", lambda gs, **kw: sizes.append([g["n_atoms"] for g in gs]) or real(gs, **kw))
    one = finetune.energies(model, graphs, 1, "cpu", torch.float64)
    assert all(len(s) == 1 for s in sizes)
    sizes.clear()
    np.testing.assert_allclose(finetune.energies(model, graphs, 6, "cpu", torch.float64), one, rtol=1e-12)
    # up to 6 atoms per batch; a larger structure alone
    assert all(sum(s) <= 6 or len(s) == 1 for s in sizes) and any(len(s) > 1 for s in sizes)
    labels = []
    for budget in (1, 6):
        gs = [dict(g) for g in graphs]
        sizes.clear()
        finetune.pseudolabel(model, gs, budget, "cpu", torch.float64)
        assert all(sum(s) <= budget or len(s) == 1 for s in sizes)
        labels.append(gs)
    for a, b in zip(*labels):
        assert a["energy"] == pytest.approx(b["energy"], rel=1e-12)
        np.testing.assert_allclose(a["forces"], b["forces"], rtol=1e-10, atol=1e-14)


def test_outputs_load_with_weights_only(setup):
    args = finetune.parser().parse_args(["--foundation", str(setup / "found.pt"), "--train", str(setup / "train.xyz"),
                                         "--epochs", "1", "--device", "cpu", "--dtype", "float64", "--replay", "none"])
    args.out, args.train = setup / "paths", [setup / "train.xyz"]      # as a script may pass them
    finetune.fine_tune(args)
    assert load(setup / "paths" / "model.pt")[1]["finetune"]["args"]["out"] == str(setup / "paths")


class Touch:
    """Unpickling it creates a file: code that loading a model must not run."""

    def __init__(self, path):
        self.path = path

    def __reduce__(self):
        return Path.touch, (self.path,)


def test_model_files_that_run_code_are_refused(setup):
    finetune.main(wise_args(setup, "multi", "--mode", "multihead", epochs=1))
    ran = setup / "ran"
    for f in ("model.pt", "model_pt_head.pt"):
        bad = setup / f"bad_{f.split('.')[0]}"
        bad.mkdir()
        for g in ("model.pt", "model_pt_head.pt"):
            ck = torch.load(setup / "multi" / g, weights_only=True)
            torch.save(dict(ck, extra=Touch(ran)) if g == f else ck, bad / g)
        with pytest.raises(pickle.UnpicklingError):
            finetune.main(wise_args(setup, f"again_{f}", "--mode", "multihead", "--init", str(bad / "model.pt")))
        with pytest.raises(pickle.UnpicklingError):
            finetune.average([setup / "multi" / f, bad / f])
    assert not ran.exists()


def test_charge_and_spin_outside_the_foundations_tables(setup):
    torch.save(dict(model=tiny(charge_spin=True).state_dict(), hparams=dict(HP, charge_spin=True), species=SPECIES),
               setup / "cs.pt")
    model = load(setup / "cs.pt")[0]
    frames = read(setup / "train.xyz", ":")
    for at, (q, m) in zip(frames, [(0, 1), (10, 11), (-10, 1), (1, 2), (0, 1), (0, 1), (0, 1), (0, 1)]):
        at.info.update(charge=q, spin=m)
    write(setup / "good.xyz", frames, format="extxyz")
    finetune.read_data([str(setup / "good.xyz")], {}, 4.0, SPECIES, model)
    for q, m in ((11, 1), (-11, 1), (0, 0), (0, 12)):
        frames[3].info.update(charge=q, spin=m)
        write(setup / "bad.xyz", frames, format="extxyz")
        with pytest.raises(SystemExit, match=f"bad.xyz, structure 3: charge {q} / multiplicity {m} outside"):
            finetune.read_data([str(setup / "good.xyz"), str(setup / "bad.xyz")], {}, 4.0, SPECIES, model)
    for opt in ("--train", "--valid", "--test"):
        files = {o: str(setup / ("bad.xyz" if o == opt else "good.xyz")) for o in ("--train", "--valid", "--test")}
        with pytest.raises(SystemExit, match="bad.xyz, structure 3"):
            finetune.main(["--foundation", str(setup / "cs.pt"), *(x for o, f in files.items() for x in (o, f)),
                           "--device", "cpu", "--out", str(setup / "cs"), "--replay", "none"])


@pytest.mark.parametrize("mode", ["naive", "multihead"])
def test_save_every_and_the_losses_of_every_epoch(setup, mode):
    s = finetune.main(wise_args(setup, mode, "--mode", mode, "--save-every", "2", epochs=3))
    out = setup / mode
    names = sorted(f.name for f in out.glob("*_ep*.pt"))
    assert names == ["model_ep002.pt", "model_ep003.pt"] + (["model_pt_head_ep002.pt", "model_pt_head_ep003.pt"]
                                                          if mode == "multihead" else [])
    m, ck = load(out / "model_ep002.pt")
    assert ck["finetune"]["epoch"] == 2 and set(ck["model"]) == set(load(out / "model.pt")[1]["model"])
    assert not (out / "model_ep001.pt").exists()
    finetune.main(wise_args(setup, mode + "1", "--mode", mode, "--save-every", "1", epochs=2))
    assert len(list((setup / (mode + "1")).glob("model_ep*.pt"))) == 2
    rows = list(csv.DictReader(open(out / "history.csv")))
    f = lambda r, k: float(r[k])
    for r in rows:
        assert f(r, "train_loss") == pytest.approx(sum(f(r, f"train_loss_{q}") for q in "efs"), rel=1e-12)
        assert f(r, "valid_loss") == pytest.approx(f(r, "valid_loss_e") + f(r, "valid_loss_f"), rel=1e-12)
        assert (f(r, "train_loss_replay") > 0) == (mode == "multihead")
    text = (out / "log.txt").read_text()
    assert "command    " in text.split("\n")[1] and "train loss  valid loss" in text
    assert "lr 3.00e-03" in text              # LR["adam"] for naive and multihead
    assert s["drift"] and "drift      from the foundation on 1 structures: force MAE" in text
