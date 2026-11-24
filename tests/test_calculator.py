"""load_calculator and its fallbacks, pretrained model lookup, export, Hessians, edge cases of the calculators."""
import json
import runpy
import shutil
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest
import torch

from conftest import HP, SPECIES, results, structures, tiny
from ilmlip import load_calculator, pretrained
from ilmlip import calculator as calc_module
from ilmlip.calculator import CommitteeCalculator, TorchCalculator

HAVE_CXX = bool(shutil.which("g++") or shutil.which("c++"))
needs_cxx = pytest.mark.skipif(not HAVE_CXX, reason="the cpp engine needs a C++ compiler")


def same(calc, ref, tol=1e-10):
    for at in structures().values():
        a, b = results(calc, at), results(ref, at)
        assert np.abs(a - b).max() <= tol * max(1.0, np.abs(b).max())


def quiet(**kw):
    """load_calculator, failing on any warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        return load_calculator(**kw)


def save(m, path, args=None):
    torch.save(dict(model=m.state_dict(), hparams=HP, species=SPECIES, args=args or {}), path)
    return path


@needs_cxx
def test_default_on_a_cpu_machine_is_the_cpp_engine(monkeypatch):
    from types import SimpleNamespace
    from ilmlip import calculator
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(calculator, "_jax_devices", lambda: [SimpleNamespace(platform="cpu")])
    m = tiny()
    c = quiet(model=m, dtype="float64")
    assert (c.engine, c.device) == ("cpp", "cpu")
    same(c, TorchCalculator(m, dtype="float64"))


@pytest.mark.parametrize("engine", ["jax", "torch"] + (["cpp"] if HAVE_CXX else []))
def test_each_engine_on_the_cpu(engine):
    """Each engine loads on the CPU with charge and spin and equals torch."""
    if engine == "jax":
        pytest.importorskip("jax")
    m = tiny(charge_spin=True)
    c = quiet(model=m, device="cpu", engine=engine, dtype="float64", charge=1, spin=2)
    assert c.engine == engine and str(c.device).startswith("cpu")
    assert tuple(c.charge_spin_state) == (1, 2)
    same(c, TorchCalculator(m, dtype="float64", charge=1, spin=2))
    if engine == "jax":
        assert c.kernels == "xla" and not c.background_compile


@pytest.mark.parametrize("engine", ["jax", "torch"] + (["cpp"] if HAVE_CXX else []))
def test_charge_and_spin_refused_by_models_without_them(engine):
    if engine == "jax":
        pytest.importorskip("jax")
    for kw in (dict(charge=-2), dict(spin=1)):
        with pytest.raises(ValueError, match="OMOL"):
            load_calculator(tiny(), device="cpu", engine=engine, dtype="float64", **kw)


def test_missing_gpu_falls_back_with_a_warning(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(calc_module, "_jax_devices", lambda: [type("D", (), {"platform": "cpu"})()])
    with pytest.warns(RuntimeWarning, match="sees no GPU"):
        c = load_calculator(tiny(), device="cuda", engine="torch")
    assert (c.engine, str(c.device)) == ("torch", "cpu")


@needs_cxx
def test_missing_compiler_falls_back_to_jax(monkeypatch):
    pytest.importorskip("jax")
    monkeypatch.setenv("CXX", "/nonexistent/g++")
    with pytest.warns(RuntimeWarning, match="C\\+\\+ engine cannot be built"):
        c = load_calculator(tiny(), device="cpu")
    assert c.engine == "jax"


def test_missing_jax_falls_back(monkeypatch):
    monkeypatch.setattr(calc_module, "_jax_devices", lambda: [])
    monkeypatch.setenv("CXX", "/nonexistent/g++")
    with pytest.warns(RuntimeWarning, match="JAX is not installed"):
        c = load_calculator(tiny(), engine="jax", device="cpu")
    assert c.engine == "torch"


def test_fused_kernels_on_the_cpu_warn():
    pytest.importorskip("jax")
    with pytest.warns(RuntimeWarning, match="GPU only"):
        c = load_calculator(tiny(), engine="jax", device="cpu", kernels="fused")
    assert c.kernels == "xla"


@needs_cxx
def test_options_of_another_engine_are_dropped():
    with pytest.warns(RuntimeWarning, match="chunk_atoms"):
        c = load_calculator(tiny(), engine="cpp", chunk_atoms=None, skin=1.0, compute_stress=False)
    assert c.skin == 1.0 and not c.compute_stress


def test_contradictions_raise():
    """Engine, device and kernels that contradict each other raise ValueError."""
    with pytest.raises(ValueError):
        load_calculator(tiny(), engine="cpp", device="cuda")
    with pytest.raises(ValueError):
        load_calculator(tiny(), engine="numpy")
    with pytest.raises(ValueError):
        load_calculator(tiny(), kernels="cuda")


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    """No bundled model, an empty cache in tmp_path/home, no model repository."""
    monkeypatch.setattr(pretrained, "BUNDLED", tmp_path / "bundled")
    monkeypatch.setattr(pretrained, "REPO", "")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("ILMLIP_MODELS", raising=False)
    monkeypatch.delenv("ILMLIP_REPO", raising=False)


def export(monkeypatch, *argv):
    """umlip/export.py on the command line."""
    monkeypatch.setattr(sys, "argv", ["export.py", *map(str, argv)])
    runpy.run_path(str(Path(__file__).resolve().parents[1] / "umlip" / "export.py"), run_name="__main__")


def test_pretrained_names():
    """The pretrained names, their canonical spellings and the errors of path()."""
    sizes = ["XS", "S", "M", "L", "XL", "XXL"]
    assert pretrained.names() == [
        "OMOL-XS", "OMOL-S", "OMOL-M", "OMOL-L", "OMOL-XL", "SOLIDACID-S", "SOLIDACID-M", "SOLIDACID-L",
        *(f"MATPES-PBE-{z}" for z in sizes), *(f"MATPES-r2SCAN-{z}" for z in sizes),
        *(f"OAM-{z}-mpa" for z in sizes), *(f"OAM-{z}-omat" for z in sizes),
        "OAM-L-mpa-WiSE10", "OAM-L-omat-WiSE10", "OAM-XXL-mpa-WiSE10", "OAM-S30-mpa",
        "OAM-L-mpa-committee", "OAM-L-omat-committee"]
    assert pretrained.canonical(" oam_l_OMAT-committee ") == "OAM-L-omat-committee"
    assert pretrained.canonical("OAM_L_omat_seed3") == "OAM-L-omat-seed3"
    assert pretrained.canonical("matpes_r2scan_xl") == "MATPES-r2SCAN-XL"
    assert pretrained.canonical("oam_s30_MPA") == "OAM-S30-mpa"
    for size, aliases in pretrained.oam_omat_model_aliases.items():
        assert {pretrained.canonical(f" {a.upper()} ") for a in aliases} == {f"OAM-{size}-omat"}
    assert not set(pretrained.names()) & {a for v in pretrained.oam_omat_model_aliases.values() for a in v}
    assert not any(map(pretrained.canonical, ["OAM-S", "MPtrj-S", "OMol25-S", "SOLID-ACID-0-S", "MATPES-S"]))
    with pytest.raises(ValueError, match="OAM-XL-omat"):
        pretrained.path("OAM-XXL")
    with pytest.raises(ValueError, match="one file per member"):
        pretrained.path("OAM-L-omat-committee")


def test_path_searches_the_model_folders(isolated, tmp_path, monkeypatch):
    mine = tmp_path / "mine"
    for d in (mine, pretrained.BUNDLED):
        d.mkdir()
    monkeypatch.setenv("ILMLIP_MODELS", str(mine))
    m = tiny()
    save(m, pretrained.BUNDLED / "OAM-S-omat.pt")
    assert pretrained.path("oam_s_omat") == pretrained.BUNDLED / "OAM-S-omat.pt"
    save(m, mine / "OAM-S-omat.pt")
    assert pretrained.path("oam_s_omat") == pretrained.path("Frodo") == mine / "OAM-S-omat.pt"
    same(quiet(model="OAM-S-omat", engine="torch", device="cpu", dtype="float64"),
         TorchCalculator(m, dtype="float64"))
    with pytest.raises(FileNotFoundError, match="ILMLIP_MODELS.*ILMLIP_REPO"):
        pretrained.path("OAM-XL-mpa")


@pytest.mark.parametrize("url", [False, True], ids=["folder", "url"])
def test_missing_model_is_downloaded_into_the_cache(isolated, tmp_path, monkeypatch, url):
    repo, cache = tmp_path / "repo", tmp_path / "home" / ".cache" / "ilmlip" / "models"
    m = tiny()
    for name in ("OAM-XL-omat", "OAM-L-omat-seed2"):
        export(monkeypatch, save(m, tmp_path / "run.pt"), name, repo)
    monkeypatch.setenv("ILMLIP_REPO", repo.as_uri() if url else str(repo))
    assert pretrained.path("oam_xl_omat") == cache / "OAM-XL-omat.pt"
    (repo / "OAM-XL-omat.pt").unlink()
    same(quiet(model="OAM-XL-omat", engine="torch", device="cpu", dtype="float64"),
         TorchCalculator(m, dtype="float64"))

    index = json.loads((repo / "models.json").read_text())
    index["OAM-L-omat-seed2"]["sha256"] = "0" * 64
    (repo / "models.json").write_text(json.dumps(index))
    with pytest.raises(ValueError, match="sha256"):
        pretrained.path("OAM-L-omat-seed2")
    assert [f.name for f in cache.iterdir()] == ["OAM-XL-omat.pt"]
    with pytest.raises(FileNotFoundError, match="models.json"):
        pretrained.path("OAM-S-mpa")


def test_committees(isolated, tmp_path, monkeypatch):
    monkeypatch.setenv("ILMLIP_MODELS", str(tmp_path))
    assert pretrained.members("oam-l-mpa-committee") == ["OAM-L-mpa"] + [f"OAM-L-mpa-seed{k}" for k in range(1, 5)]
    assert pretrained.members("OAM-S-omat") is None
    members = pretrained.members("OAM-L-omat-committee")
    files = [save(tiny(seed=k), tmp_path / f"{n}.pt") for k, n in enumerate(members)]
    c = quiet(model="OAM-L-omat-committee", engine="torch", device="cpu")
    assert isinstance(c, CommitteeCalculator) and len(c.members) == 5
    c2 = quiet(model=files[:2], engine="torch", device="cpu", dtype="float64")
    at = structures()["nacl"]
    at.calc = c2
    e = [results(TorchCalculator(f, dtype="float64"), at)[0] for f in files[:2]]
    assert abs(at.get_potential_energy() - np.mean(e)) < 1e-10
    assert at.calc.results["energy_std"] > 0


def fine_tune_float64(tmp_path):
    """The model.pt of a float64 fine-tune of a tiny foundation."""
    from ase.build import bulk
    from ase.io import write
    from ilmlip import finetune
    from ilmlip.model import collate, graph
    m, rng = tiny(), np.random.default_rng(0)
    frames = []
    for _ in range(6):
        at = bulk("NaCl", "rocksalt", a=5.64).repeat((1, 1, 2))
        at.positions += rng.normal(0, 0.1, at.positions.shape)
        out = m(collate([graph(at, HP["rcut"], SPECIES)]), compute_stress=False)
        at.info["energy"], at.arrays["forces"] = float(out["energy"][0]), out["forces"].detach().numpy()
        frames.append(at)
    write(tmp_path / "train.xyz", frames, format="extxyz")
    finetune.main(["--foundation", str(save(m, tmp_path / "found.pt")), "--train", str(tmp_path / "train.xyz"),
                   "--epochs", "1", "--batch-size", "3", "--lr", "1e-3", "--device", "cpu", "--dtype", "float64",
                   "--replay", "none", "--out", str(tmp_path / "ft")])
    return tmp_path / "ft" / "model.pt"


def test_dtype_auto_follows_the_training_precision(tmp_path):
    m = tiny()
    trained = {"tf32": dict(dtype="tf32"), "float32": dict(dtype="float32"), "float64": dict(dtype="float64")}
    files = {name: save(m, tmp_path / f"{name}.pt", args) for name, args in trained.items()}
    files["finetuned64"] = fine_tune_float64(tmp_path)
    for name, path in files.items():
        c = quiet(model=path, engine="jax", device="cpu")
        want = "float64" if "64" in name else "float32"
        assert c.model.dtype == np.dtype(want)
        assert c.model.precision == ("highest" if want == "float64" else "default")


@needs_cxx
def test_tf32_is_float32_on_the_cpu():
    c = quiet(model=tiny(), engine="cpp", dtype="tf32")
    assert c.model.dtype == np.float32


def test_torch_tf32_switch_is_restored():
    before = torch.backends.cuda.matmul.allow_tf32
    c = TorchCalculator(tiny(), dtype="tf32")
    assert c.tf32 and c.dtype == torch.float32
    results(c, structures()["nacl"])
    assert torch.backends.cuda.matmul.allow_tf32 == before
    with pytest.raises(ValueError):
        TorchCalculator(tiny(), dtype="bfloat16")


def test_a_slab_cell_without_a_vacuum_vector_is_periodic_in_plane():
    from ase.build import fcc111
    m = tiny()
    slab = fcc111("Cu", (2, 2, 3))               # a rank-2 cell: periodic along its two vectors only
    slab.positions += np.random.default_rng(0).normal(0, 0.05, slab.positions.shape)
    padded = slab.copy()
    padded.cell[2] = [0, 0, 30.0]
    calc = TorchCalculator(m, dtype="float64")
    slab.calc = calc
    energy = slab.get_potential_energy()
    assert "stress" not in calc.results
    padded.calc = TorchCalculator(m, dtype="float64")
    assert abs(padded.get_potential_energy() - energy) < 1e-10
    if HAVE_CXX:
        cpp = slab.copy()
        cpp.calc = quiet(model=m, engine="cpp", device="cpu", dtype="float64")
        assert abs(cpp.get_potential_energy() - energy) < 1e-10


def test_pretrained_command_lists_and_downloads_committees(isolated, tmp_path, monkeypatch, capsys):
    members = pretrained.members("OAM-L-omat-committee")
    ck = save(tiny(), tmp_path / "run.pt")
    for n in members:
        export(monkeypatch, ck, n, tmp_path / "repo")
    monkeypatch.setenv("ILMLIP_REPO", str(tmp_path / "repo"))
    monkeypatch.setattr(sys, "argv", ["pretrained", "oam_l_omat_committee"])
    pretrained.main()
    cache = tmp_path / "home" / ".cache" / "ilmlip" / "models"
    assert sorted(f.name for f in cache.iterdir()) == sorted(f"{n}.pt" for n in members)
    c = quiet(model="OAM-L-omat-committee", engine="torch", device="cpu")
    assert isinstance(c, CommitteeCalculator) and len(c.members) == 5
    capsys.readouterr()
    monkeypatch.setattr(sys, "argv", ["pretrained"])
    pretrained.main()
    listed = capsys.readouterr().out.splitlines()
    assert [l.split()[0] for l in listed if l.endswith("present")] == ["OAM-L-omat", "OAM-L-omat-committee"]
    assert len(listed) == len(pretrained.names())


def test_export_refuses_names_pretrained_does_not_know(tmp_path, monkeypatch):
    run = save(tiny(), tmp_path / "run.pt")
    for name in ("OAM-S", "OAM-L-omat-committee"):
        with pytest.raises(SystemExit):
            export(monkeypatch, run, name, tmp_path / "repo")
    assert not (tmp_path / "repo").exists()
    export(monkeypatch, run, "oam_l_omat_seed2", tmp_path / "repo")
    assert list(json.loads((tmp_path / "repo" / "models.json").read_text())) == ["OAM-L-omat-seed2"]
    assert (tmp_path / "repo" / "OAM-L-omat-seed2.pt").is_file()


def test_export_keeps_the_training_precision(tmp_path, monkeypatch):
    from ilmlip.model import load, trained_dtype
    for name, args, want in [("OAM-S-omat", dict(dtype="float64"), "float64"),
                             ("OAM-M-omat", dict(dtype="tf32"), "tf32"),
                             ("OAM-L-omat", dict(dtype="float64"), "float64")]:
        export(monkeypatch, save(tiny(), tmp_path / "run.pt", args), name, tmp_path)
        assert trained_dtype(load(tmp_path / f"{name}.pt")[1]) == want
    assert quiet(model=str(tmp_path / "OAM-S-omat.pt"), engine="torch", device="cpu").dtype == torch.float64


def fd_hessian(calc, atoms, h=1e-4):
    a = atoms.copy()
    a.calc = calc
    x = a.get_positions()
    H = np.zeros((x.size, x.size))
    for k in range(x.size):
        f = []
        for s in (h, -h):
            y = x.ravel().copy()
            y[k] += s
            a.set_positions(y.reshape(x.shape))
            f.append(a.get_forces().ravel())
        H[:, k] = (f[1] - f[0]) / (2 * h)
    return H


def hessian_calculator(engine):
    """conftest's tiny model, float64, on the torch or the JAX engine."""
    if engine == "jax":
        pytest.importorskip("jax")
        from ilmlip import jax_md
        jax_md.enable_x64()
        return jax_md.JaxCalculator(jax_md.from_torch(tiny(), "float64"), kernels="xla")
    return TorchCalculator(tiny(), dtype="float64")


@pytest.mark.parametrize("engine", ["torch", "jax"])
@pytest.mark.parametrize("name", ["molecule", "nacl"])
def test_hessian_is_the_finite_difference_of_the_forces(engine, name):
    calc = hessian_calculator(engine)
    atoms = structures()[name]
    H = calc.get_hessian(atoms)
    ref = fd_hessian(calc, atoms)
    assert H.shape == (3 * len(atoms),) * 2 and H.dtype == np.float64
    assert np.abs(H - H.T).max() <= 1e-10 * np.abs(H).max()
    assert np.abs(H - ref).max() <= 1e-6 * np.abs(ref).max()


def test_hessian_in_chunks_and_by_rows(monkeypatch):
    calc = TorchCalculator(tiny(), dtype="float64")
    with pytest.raises(ValueError, match="no atoms"):
        calc.get_hessian()
    atoms = structures()["molecule"]
    atoms.calc = calc
    atoms.get_forces()
    H = calc.get_hessian()
    chunked = calc.get_hessian(atoms, chunk=4)
    grad = torch.autograd.grad

    def unbatched(*a, is_grads_batched=False, **kw):
        if is_grads_batched:
            raise RuntimeError("no batching rule")
        return grad(*a, **kw)

    monkeypatch.setattr(torch.autograd, "grad", unbatched)
    by_rows = calc.get_hessian(atoms)
    for other in (chunked, by_rows):
        assert np.abs(other - H).max() <= 1e-12 * np.abs(H).max()


@pytest.mark.parametrize("engine", ["torch", "jax"])
@pytest.mark.parametrize("name", ["molecule", "nacl"])
def test_lanczos_hessian_is_the_lowest_end_of_the_dense_spectrum(engine, name):
    calc = hessian_calculator(engine)
    atoms = structures()[name]
    H = calc.get_hessian(atoms)
    ev = np.linalg.eigvalsh(H)
    lam, vec = calc.get_hessian(atoms, method="lanczos", k=4)
    assert vec.shape == (len(H), 4)
    assert np.abs(lam - ev[:4]).max() <= 1e-8 * np.abs(ev).max()
    assert (np.linalg.norm(H @ vec - lam * vec, axis=0) <= 1e-6 * np.abs(ev).max()).all()
    with pytest.raises(ValueError, match="method"):
        calc.get_hessian(atoms, method="power")


ENGINES = ["jax", "torch"] + (["cpp"] if HAVE_CXX else [])


def engine(name, m, **kw):
    if name == "jax":
        pytest.importorskip("jax")
    return load_calculator(model=m, engine=name, device="cpu", dtype="float64", **kw)


def energy_forces(calc, atoms):
    a = atoms.copy()
    a.calc = calc
    return np.array([a.get_potential_energy(), *a.get_forces().ravel()])


@pytest.mark.parametrize("name", ENGINES)
def test_atoms_without_neighbours(name):
    from ase import Atoms
    m = tiny()
    for at in (Atoms("Cu"), Atoms("CuO", positions=[[0, 0, 0], [9, 0, 0]]),
               Atoms("Na", cell=np.eye(3) * 10, pbc=True)):
        a, b = energy_forces(engine(name, m), at), energy_forces(TorchCalculator(m, dtype="float64"), at)
        assert np.abs(a - b).max() <= 1e-10 * max(1.0, np.abs(b).max())


@pytest.mark.parametrize("name", ENGINES)
@pytest.mark.parametrize("cell", [np.zeros((3, 3)), np.diag([12.0, 0.0, 0.0])], ids=["none", "one-vector"])
def test_periodic_flag_without_a_cell_vector(name, cell):
    mol = structures()["molecule"]
    mol.cell, mol.pbc = cell, True
    a, b = energy_forces(engine(name, tiny()), mol), energy_forces(TorchCalculator(tiny(), dtype="float64"), mol)
    assert np.abs(a - b).max() <= 1e-10 * max(1.0, np.abs(b).max())


@pytest.mark.parametrize("name", ENGINES)
def test_no_stress_without_periodicity(name):
    from ase.calculators.calculator import PropertyNotImplementedError
    mol = structures()["molecule"]
    mol.cell, mol.pbc = np.eye(3) * 12.0, False
    mol.calc = engine(name, tiny())
    mol.get_potential_energy()
    with pytest.raises(PropertyNotImplementedError):
        mol.get_stress()


@pytest.mark.parametrize("name", [e for e in ENGINES if e != "torch"])
def test_non_finite_positions_raise(name):
    at = structures()["nacl"]
    at.calc = engine(name, tiny())
    at.get_potential_energy()
    at.positions[3, 0] = np.nan
    with pytest.raises(FloatingPointError):
        at.get_potential_energy()


@pytest.mark.parametrize("name", ENGINES)
def test_non_finite_energy_raises(name):
    m = tiny()
    with torch.no_grad():
        m.energy_scale.fill_(float("nan"))
    at = structures()["nacl"]
    at.calc = engine(name, m)
    with pytest.raises(FloatingPointError):
        at.get_potential_energy()


@pytest.mark.parametrize("name", ENGINES)
def test_float32_with_absolute_e0s(name):
    """E0s of 2e3 and 7e4 eV (OMol25) are added and summed in float64: float32 energies are off by the
    rounding of the network's part, far below the float32 ulp of the total."""
    from ase.build import bulk
    if name == "jax":
        pytest.importorskip("jax")
    at = bulk("NaCl", "rocksalt", a=5.64, cubic=True).repeat((3, 3, 3))
    at.positions += np.random.default_rng(0).normal(0, 0.05, at.positions.shape)

    def energies(calc):
        a = at.copy()
        a.calc = calc
        return a.get_potential_energy(), a.get_potential_energies()

    m = tiny()
    with torch.no_grad():
        m.atomic_energies.zero_()
    net = np.abs(energies(TorchCalculator(m, dtype="float64"))[1]).max()
    with torch.no_grad():
        m.energy_scale.div_(net)                                    # a network part of ~1 eV per site
        m.atomic_energies.fill_(-2047.4)
        m.atomic_energies[SPECIES.index("Cl")] = -70040.9
    e, site = energies(load_calculator(model=m, engine=name, device="cpu", dtype="float32"))
    ref, ref_site = energies(TorchCalculator(m, dtype="float64"))
    assert np.abs(site - ref_site).max() < 1e-5 and abs(e - ref) < 1e-4 and abs(site.sum() - e) < 1e-6


@pytest.mark.parametrize("name", ENGINES)
def test_elements_swapped_at_the_same_counts(name):
    m = tiny()
    with torch.no_grad():
        m.atomic_energies.fill_(-2047.4)
        m.atomic_energies[SPECIES.index("Cl")] = -70040.9
    at = structures()["nacl"]
    at.calc = engine(name, m)
    at.get_potential_energy()
    at.numbers[[0, 1]] = at.numbers[[1, 0]]          # a Na and a Cl change places
    ref = at.copy()
    ref.calc = TorchCalculator(m, dtype="float64")
    assert abs(at.get_potential_energy() - ref.get_potential_energy()) < 1e-8
    assert np.abs(at.get_potential_energies() - ref.get_potential_energies()).max() < 1e-8


@pytest.mark.parametrize("name", [e for e in ENGINES if e != "jax"] + ["committee"])
def test_engines_without_a_hessian_say_so(name):
    m = tiny()
    calc = (CommitteeCalculator([TorchCalculator(m, dtype="float64")]) if name == "committee"
            else engine(name, m))
    if name == "torch":
        assert calc.get_hessian(structures()["molecule"]).shape == (27, 27)
        return
    with pytest.raises(NotImplementedError, match="engine='jax'|member"):
        calc.get_hessian(structures()["molecule"])


@needs_cxx
def test_cpp_skin_stays_while_rebuilds_are_rare():
    from ilmlip.cpu import CpuCalculator, CpuModel
    calc = CpuCalculator(CpuModel(tiny(), "float64"), skin=0.3, async_rebuild=False)
    at = structures()["nacl"]
    at.calc = calc
    step = np.random.default_rng(1).normal(0, 1, at.positions.shape)
    step *= 0.004 / np.abs(step).max()
    for _ in range(120):
        at.positions += step
        at.get_potential_energy()
    assert calc.n_rebuilds > 1 and calc.skin == 0.3


def test_unknown_elements_are_named():
    from ase import Atoms
    from ilmlip.model import graph
    with pytest.raises(ValueError, match="Xe"):
        graph(Atoms("CuXe", positions=[[0, 0, 0], [2, 0, 0]]), 4.0, SPECIES)
