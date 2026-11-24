"""A model not trained on stresses (hparams stress=False) gives energies and forces but no stress."""
import json

import numpy as np
import pytest
import torch
from ase.build import bulk, molecule
from ase.calculators.calculator import PropertyNotImplementedError
from ase.io import write

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)

from conftest import SPECIES, structures, tiny
from test_calculator import HP as CALC_HP, needs_cxx
import test_train
from test_train import HP, make_data, recipe_args
from ilmlip import evaluate, finetune, train
from ilmlip.batched import BatchedModel
from ilmlip.calculator import TorchCalculator
from ilmlip.data import PackedSplit, PackedWriter
from ilmlip.jax_md import JaxCalculator, from_torch
from ilmlip.model import collate, graph, load


def saved(tmp_path, stress, name="m.pt"):
    torch.save(dict(model=tiny().state_dict(), hparams=dict(CALC_HP, stress=stress), species=SPECIES),
               tmp_path / name)
    return tmp_path / name


class NoStress(PackedWriter):
    def add(self, *args, stress=None, **kw):
        return super().add(*args, stress=np.full((len(kw["energy"]), 6), np.nan), **kw)


def make_data_without_stress(root, monkeypatch):
    with monkeypatch.context() as mp:
        mp.setattr(test_train, "PackedWriter", NoStress)
        make_data(root)
    assert not PackedSplit(root / "train").any_stress


def test_the_flag_comes_from_the_checkpoint(tmp_path):
    m, _ = load(saved(tmp_path, False))
    b = collate([graph(bulk("NaCl", "rocksalt", a=5.64), 4.0, SPECIES)])
    assert not m.stress and torch.isfinite(m(b)["forces"]).all()
    with pytest.raises(ValueError, match="not trained on stresses"):
        m(b, compute_stress=True)
    assert load(saved(tmp_path, True, "s.pt"))[0].stress
    torch.save(dict(model=tiny().state_dict(), hparams=CALC_HP, species=SPECIES), tmp_path / "old.pt")
    assert load(tmp_path / "old.pt")[0].stress                  # a checkpoint without the stress hyper-parameter


@pytest.mark.parametrize("engine", ["torch", "jax", pytest.param("cpp", marks=needs_cxx)])
def test_calculators_give_energy_and_forces_but_refuse_stress(tmp_path, engine):
    from ilmlip.cpu import CpuCalculator
    make = dict(torch=lambda p: TorchCalculator(p, dtype="float64"),
                jax=lambda p: JaxCalculator(from_torch(load(p)[0], "float64"), spatial_order=False),
                cpp=lambda p: CpuCalculator(p, dtype="float64"))[engine]
    off, on = make(saved(tmp_path, False)), make(saved(tmp_path, True, "s.pt"))
    for at in structures().values():
        a, b = at.copy(), at.copy()
        a.calc, b.calc = off, on
        assert abs(a.get_potential_energy() - b.get_potential_energy()) < 1e-10 * len(at)
        assert np.abs(a.get_forces() - b.get_forces()).max() < 1e-10
        if at.pbc.any():
            b.get_stress()
            with pytest.raises(PropertyNotImplementedError, match="not trained on stresses"):
                a.get_stress()


def forces_then_stress(calc, at):
    at.calc = calc
    at.get_forces()
    try:
        at.get_stress()
    except PropertyNotImplementedError:
        return "refused"


def test_batched_and_lockstep(tmp_path):
    from ilmlip.lockstep import imap
    jm = from_torch(load(saved(tmp_path, False))[0], "float64")
    with pytest.raises(ValueError, match="compute_stress=False"):
        BatchedModel(jm, compute_stress=True)
    bm = BatchedModel(jm)
    assert not bm.compute_stress
    nacl = bulk("NaCl", "rocksalt", a=5.64)
    assert np.isfinite(bm.compute([nacl])[0]["forces"]).all()
    assert list(imap(bm, forces_then_stress, [nacl.copy(), molecule("H2O")])) == ["refused"] * 2


def test_lammps_export_warns(tmp_path):
    from ilmlip.lammps import export
    with pytest.warns(UserWarning, match="no NPT"):
        export(saved(tmp_path, False), tmp_path / "m.pkl", dtype="float64")


def test_training_without_stress_labels(tmp_path, capsys, monkeypatch):
    make_data_without_stress(tmp_path / "data", monkeypatch)
    (tmp_path / "hp.json").write_text(json.dumps(dict(hparams=HP)))
    train.main(recipe_args(tmp_path, tmp_path / "a", "--stop-after", "3"))
    assert "--w-stress 5 ignored" in capsys.readouterr().out
    assert torch.load(tmp_path / "a" / "checkpoint.pt", weights_only=True)["hparams"]["stress"] is False
    train.main(recipe_args(tmp_path, tmp_path / "a"))                       # resumed
    assert not load(tmp_path / "a" / "best.pt")[0].stress
    assert "s_mae" not in evaluate.main([str(tmp_path / "a" / "best.pt"), str(tmp_path / "data")])["metrics"]["all"]


def test_a_stage_with_stress_labels_gains_the_flag_and_one_without_keeps_it(tmp_path, monkeypatch):
    make_data(tmp_path / "stressed")
    make_data_without_stress(tmp_path / "data", monkeypatch)
    (tmp_path / "hp.json").write_text(json.dumps(dict(hparams=HP)))
    train.main(recipe_args(tmp_path, tmp_path / "plain", "--stop-after", "3"))
    gain = recipe_args(tmp_path, tmp_path / "gain", "--stop-after", "3", "--init", str(tmp_path / "plain" / "checkpoint.pt"))
    gain[gain.index("--packed") + 1] = str(tmp_path / "stressed")
    train.main(gain)
    assert load(tmp_path / "gain" / "checkpoint.pt")[0].stress
    train.main(recipe_args(tmp_path, tmp_path / "keep", "--stop-after", "3", "--init", str(tmp_path / "gain" / "checkpoint.pt")))
    assert load(tmp_path / "keep" / "checkpoint.pt")[0].stress
    m = evaluate.main([str(tmp_path / "gain" / "checkpoint.pt"), str(tmp_path / "stressed")])
    assert "s_mae" in m["metrics"]["all"]
    zero = recipe_args(tmp_path, tmp_path / "zero", "--stop-after", "3", "--w-stress", "0")
    zero[zero.index("--packed") + 1] = str(tmp_path / "stressed")
    train.main(zero)
    assert not load(tmp_path / "zero" / "checkpoint.pt")[0].stress     # labels, but no stress term


@pytest.mark.parametrize("labels", [False, True])
def test_fine_tuning_a_model_without_stress(tmp_path, labels):
    path, rng = saved(tmp_path, False, "found.pt"), np.random.default_rng(0)
    m = load(path)[0]
    frames = []
    for _ in range(6):
        at = bulk("NaCl", "rocksalt", a=5.64).repeat((1, 1, 2))
        at.positions += rng.normal(0, 0.1, at.positions.shape)
        o = m(collate([graph(at, 4.0, SPECIES)]))
        at.info["energy"] = float(o["energy"][0])
        at.arrays["forces"] = o["forces"].detach().numpy()
        if labels:
            at.info["stress"] = rng.normal(0, 0.01, 6)
        frames.append(at)
    write(tmp_path / "train.xyz", frames, format="extxyz")
    replay = [bulk("Si", "diamond", a=5.43) for _ in range(11)]
    write(tmp_path / finetune.REPLAY_FILES[0], replay[1:], format="extxyz")
    write(tmp_path / finetune.REPLAY_FILES[1], replay[:1], format="extxyz")
    finetune.main(["--foundation", str(path), "--train", str(tmp_path / "train.xyz"), "--epochs", "1",
                   "--batch-size", "3", "--device", "cpu", "--dtype", "float64", "--out", str(tmp_path / "ft"),
                   "--replay", str(tmp_path), "--mode", "multihead", "--w-stress", "1"])
    assert load(tmp_path / "ft" / "model.pt")[0].stress is labels
