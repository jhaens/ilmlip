"""Element-specific cutoffs (ilmlip.vcut): the model, every engine, the data path, training,
and their removal."""
import json
import shutil

import numpy as np
import pytest
import torch
from ase import Atoms
from ase.build import bulk
from scipy.spatial.transform import Rotation

from conftest import KERNELS, SHAPES, SPECIES, close, numerical_derivatives, results, structures
from ilmlip import evaluate, train, vcut
from ilmlip.calculator import TorchCalculator
from ilmlip.data import PackedLoader, PackedSplit, PackedWriter, lstsq_e0
from ilmlip.model import ILMlip, collate, fold_charge_spin, graph, load

RC = {"H": 3.5, "C": 4.0, "O": 3.8, "Na": 5.0, "Si": 4.2, "Cl": 4.5, "Cu": 4.6}


def tiny(rcut_elem=RC, rcut=5.0, seed=0, **kw):
    """A small random model with element cutoffs (as conftest.tiny)."""
    torch.manual_seed(seed)
    shape = dict(lmax=2, lmax_feat=1, product_order=3)
    shape.update(kw)
    m = ILMlip(SPECIES, rcut=rcut, n_basis=5, n_channels=4, lmax_ind=1, n_message=3, hidden=(8,),
                    embed_dim=4, zbl=True, rcut_elem=rcut_elem, dtype=torch.float64, **shape)
    with torch.no_grad():
        for p in m.parameters():
            if p.abs().sum() == 0:
                p.normal_(0, 0.1)
        m.feat_mean.normal_(0, 0.1)
        m.feat_std.uniform_(0.5, 1.5)
        m.energy_scale.fill_(0.7)
    return m.eval()


def evaluate_model(model, atoms, stress=False, prune=True, **state):
    g = graph(atoms, model.rcut, model.species)
    g = vcut.prune(g, model) if prune else g
    g.update(state)
    out = model(collate([g]), compute_stress=stress)
    return {k: v.detach().numpy() for k, v in out.items()}


def dimer(a, b, r):
    return Atoms(a + b, positions=[[0, 0, 0], [r, 0, 0]])


def test_pair_cutoff_is_the_larger_one():
    rc = vcut.table(tiny())
    k = {s: n for n, s in enumerate(SPECIES)}
    assert rc[k["H"], k["Na"]] == rc[k["Na"], k["H"]] == 5.0
    assert rc[k["H"], k["H"]] == 3.5 and rc[k["O"], k["Cl"]] == 4.5
    assert vcut.table(tiny(None)) is None
    with pytest.raises(ValueError):
        tiny({"Na": 5.5})                                   # above rcut
    with pytest.raises(ValueError):
        tiny({"Cu": 1.0})                                   # inside the ZBL repulsion of Cu-Cu
    with pytest.raises(ValueError):
        tiny({"na": 4.5})


def test_equal_cutoffs_are_the_plain_model():
    plain, same = tiny(None), tiny({s: 5.0 for s in SPECIES})
    for at in structures().values():
        a, b = evaluate_model(plain, at, stress=bool(at.pbc.any())), evaluate_model(same, at, stress=bool(at.pbc.any()))
        for key in a:
            assert np.array_equal(a[key], b[key])


def test_pruning_changes_nothing():
    m = tiny()
    for at in structures().values():
        a, b = evaluate_model(m, at), evaluate_model(m, at, prune=False)
        assert np.array_equal(a["energy"], b["energy"])
        np.testing.assert_allclose(a["forces"], b["forces"], atol=1e-14)


def test_pairs_beyond_their_cutoff_do_nothing():
    m = tiny()
    alone = lambda s: evaluate_model(m, Atoms(s, positions=[[0, 0, 0]]))["energy"][0]
    far = evaluate_model(m, dimer("H", "H", 3.6), prune=False)                 # H-H: 3.5 A
    assert far["energy"][0] == pytest.approx(2 * alone("H"), abs=1e-13)
    assert np.abs(far["forces"]).max() < 1e-13
    near = evaluate_model(m, dimer("H", "Na", 4.0))                           # H-Na: 5 A
    assert abs(near["energy"][0] - alone("H") - alone("Na")) > 1e-6


def test_smooth_at_the_pair_cutoff():
    m = tiny()
    e = lambda r: evaluate_model(m, dimer("O", "Cl", r), prune=False)        # O-Cl: 4.5 A
    for h in (1e-3, 1e-4):
        lo, hi = e(4.5 - h), e(4.5 + h)
        assert abs(lo["energy"][0] - hi["energy"][0]) < 1e3 * h ** 3
        assert np.abs(lo["forces"]).max() < 1e3 * h ** 2


@pytest.mark.parametrize("name", list(structures()))
def test_invariance(name):
    m = tiny()
    at = structures()[name]
    ref = evaluate_model(m, at)
    R = Rotation.random(random_state=3).as_matrix()
    perm = np.random.default_rng(4).permutation(len(at))
    b = at[perm]
    b.positions = b.positions @ R.T + [0.7, -0.4, 1.3]
    b.set_cell(b.cell.array @ R.T)
    out = evaluate_model(m, b)
    assert abs(out["energy"][0] - ref["energy"][0]) < 1e-10
    np.testing.assert_allclose(out["forces"], ref["forces"][perm] @ R.T, atol=1e-10)


@pytest.mark.parametrize("name", ["nacl", "triclinic", "molecule"])
def test_forces_and_stress_are_derivatives(name):
    m = tiny()
    at = structures()[name]
    periodic = bool(at.pbc.any())
    out = evaluate_model(m, at, stress=periodic, prune=False)
    f, s = numerical_derivatives(lambda a: evaluate_model(m, a, prune=False)["energy"][0], at)
    assert np.all(np.abs(out["forces"][:len(f)] - f) < 1e-6 * np.maximum(1.0, np.abs(f)))
    if periodic:
        assert np.all(np.abs(out["stress"][0] - s) < 1e-6 * np.maximum(1.0, np.abs(s)))

def test_checkpoint_keeps_the_cutoffs(tmp_path):
    m = tiny(charge_spin=True)
    hp = dict(rcut=5.0, lmax=2, lmax_feat=1, n_basis=5, n_channels=4, product_order=3, lmax_ind=1, n_layers=2,
              n_message=3, hidden=[8], embed_dim=4, zbl=True, charge_spin=True, rcut_elem=RC)
    torch.save(dict(model=m.state_dict(), hparams=hp, species=SPECIES), tmp_path / "m.pt")
    m2, _ = load(tmp_path / "m.pt")
    assert m2.rcut_elem == RC and "vc_scale" not in m2.state_dict()
    assert m2.state_dict()["vc_cutoffs"].tolist() == [round(RC[s] * 1000) for s in SPECIES]
    at = structures()["molecule"]
    a, b = evaluate_model(m, at, charge=1, spin=2), evaluate_model(m2, at, charge=1, spin=2)
    assert np.array_equal(a["energy"], b["energy"])
    folded = fold_charge_spin(m2, 1, 2)
    assert np.abs(evaluate_model(folded, at)["energy"] - b["energy"]).max() < 1e-12     # sums in another order


try:
    import jax
    jax.config.update("jax_enable_x64", True)
    from ilmlip.jax_md import JaxCalculator, from_torch
except ImportError:
    jax = None
needs_jax = pytest.mark.skipif(jax is None, reason="jax not installed")



def more_structures():
    rng = np.random.default_rng(5)
    mixed = Atoms("HCONaSiClCu" * 2, positions=rng.random((14, 3)) * 6.0, cell=np.eye(3) * 6.5, pbc=True)
    return dict(structures(), mixed=mixed, cu=bulk("Cu", "fcc", a=3.6))


@needs_jax
@pytest.mark.parametrize("kernels", KERNELS)
@pytest.mark.parametrize("shape", list(SHAPES))
@pytest.mark.parametrize("state", [None, (1, 2)])
def test_jax_matches_torch(kernels, shape, state):
    m = tiny(charge_spin=state is not None, **SHAPES[shape])
    q, s = state or (None, None)
    jc = JaxCalculator(from_torch(m, "float64", q, s), kernels=kernels)
    tc = TorchCalculator(m, dtype="float64", charge=q, spin=s)
    for at in more_structures().values():
        at.info.update(charge=q, spin=s)
        assert close(results(jc, at), results(tc, at), 1e-11)


def md_against_fresh_lists(calc, m):
    from ase import units
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
    from ase.md.verlet import VelocityVerlet
    at = bulk("NaCl", "rocksalt", a=5.64).repeat((2, 2, 2))
    at.numbers[::5] = 1                                    # H among them: pairs with three cutoffs
    MaxwellBoltzmannDistribution(at, temperature_K=1500, rng=np.random.default_rng(1))
    at.calc = calc
    ref = TorchCalculator(m, dtype="float64")
    dyn = VelocityVerlet(at, 1 * units.fs)
    for _ in range(6):
        dyn.run(5)
        b = at.copy()
        b.calc = ref
        np.testing.assert_allclose(at.get_forces(), b.get_forces(), atol=1e-10)
    assert calc.n_rebuilds > 1


@needs_jax
def test_jax_verlet_list_follows_md():
    m = tiny()
    md_against_fresh_lists(JaxCalculator(from_torch(m, "float64"), kernels="xla", skin=0.3), m)


@needs_jax
def test_jax_device_list_has_the_pairs_of_their_cutoffs():
    from ilmlip.verlet import host_neighbours
    m = tiny()
    calc = JaxCalculator(from_torch(m, "float64"), kernels="xla", neighbour_list="device")
    for k, at in enumerate(more_structures().values(), start=1):
        results(calc, at)
        calc._device_rebuild(at)
        st = calc._nl
        ci, cj, cs, row_start = (np.asarray(x) for x in st["args"][1:5])
        ok = np.arange(len(ci)) < row_start[-1]
        if st["ordered"]:
            order = np.asarray(st["args"][11])
            ci, cj = order[ci], order[cj]
        i, j, S, d = host_neighbours(at.positions, at.cell.array, at.pbc & at.cell.array.any(1), st["rcut"] + calc.skin)
        i, j, S, _ = vcut.verlet_keep(calc.pair_rcut, calc._z2mu[at.numbers], calc.skin, i, j, S, d)
        assert ({(a, b, *s) for a, b, s in zip(ci[ok], cj[ok], np.rint(cs[ok]).astype(int))}
                == {(a, b, *s) for a, b, s in zip(i, j, S)})
        assert calc.n_device_builds == k


@needs_jax
@pytest.mark.skipif("fused-reference" not in KERNELS, reason="cuequivariance-jax not installed")
@pytest.mark.parametrize("neighbour_list", ["host", "device"])
def test_jax_row_classes(neighbour_list):
    """The two Na (cutoff 5 A, 32 neighbours) get wider table rows than H and O (18-20): the
    same numbers, also along MD with list rebuilds (on the device: the wide rows stay first)."""
    from ase import units
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
    from ase.md.verlet import VelocityVerlet
    m = tiny()
    at = Atoms("H" * 40 + "O" * 22 + "Na" * 2, positions=np.indices((4, 4, 4)).reshape(3, -1).T * 2.25,
               cell=np.eye(3) * 9.0, pbc=True)
    at.positions += np.random.default_rng(2).normal(0, 0.1, at.positions.shape)
    MaxwellBoltzmannDistribution(at, temperature_K=1500, rng=np.random.default_rng(1))
    calc = JaxCalculator(from_torch(m, "float64"), kernels="fused-reference", skin=0.3, neighbour_list=neighbour_list)
    ref = TorchCalculator(m, dtype="float64")
    assert close(results(calc, at), results(ref, at), 1e-11)
    n_w, extra = calc._nl["key_extra"][-1]
    assert n_w == 2 and extra >= 10
    at.calc = calc
    dyn = VelocityVerlet(at, 1 * units.fs)
    for _ in range(4):
        dyn.run(5)
        b = at.copy()
        b.calc = ref
        np.testing.assert_allclose(at.get_forces(), b.get_forces(), atol=1e-10)
    if neighbour_list == "host":
        assert calc.n_rebuilds > 1 and calc.skin > 0.3          # fast-expiring lists widen the skin
    else:
        assert calc.n_device_builds > 1 and calc.n_rebuilds == calc.n_device_builds + 1
        assert np.array_equal(np.asarray(calc._nl["args"][5])[:2], [3, 3])     # Na first


@needs_jax
@pytest.mark.parametrize("n_read", [1, 2, 3, 4, 5, 8, 11])
def test_jax_readout_gemms_are_aligned(n_read):
    """The columns of F that a readout GEMM reads come in a multiple of 4 floats (16-byte aligned rows), with
    the same product as the whole of F."""
    from ilmlip.jax_md import _read_columns
    rng = np.random.default_rng(n_read)
    W = np.zeros((19, 5))
    W[rng.choice(19, n_read, replace=False)] = rng.normal(size=(n_read, 5))
    select, Wr = _read_columns(W)
    F = rng.normal(size=(7, 19))
    assert Wr.shape[0] % 4 == 0 and np.abs(np.asarray(select(F)) @ Wr - F @ W).max() < 1e-12


cpp =shutil.which("g++") or shutil.which("c++")


@pytest.mark.skipif(not cpp, reason="the CPU engine needs a C++ compiler")
@pytest.mark.parametrize("shape", list(SHAPES))
@pytest.mark.parametrize("state", [None, (1, 2)])
def test_cpu_engine_matches_torch(shape, state):
    from ilmlip.cpu import CpuCalculator
    m = tiny(charge_spin=state is not None, **SHAPES[shape])
    q, s = state or (None, None)
    cc = CpuCalculator(m, dtype="float64", charge=q, spin=s)
    tc = TorchCalculator(m, dtype="float64", charge=q, spin=s)
    for at in more_structures().values():
        at.info.update(charge=q, spin=s)
        assert close(results(cc, at), results(tc, at), 1e-11)
    f32 = CpuCalculator(m, dtype="float32", charge=q, spin=s)
    at = more_structures()["mixed"]
    at.info.update(charge=q, spin=s)
    assert close(results(f32, at), results(tc, at), 1e-6)


@pytest.mark.skipif(not cpp, reason="the CPU engine needs a C++ compiler")
def test_cpu_verlet_list_follows_md():
    from ilmlip.cpu import CpuCalculator
    m = tiny()
    md_against_fresh_lists(CpuCalculator(m, dtype="float64", skin=0.3), m)


@needs_jax
@pytest.mark.parametrize("kernels", KERNELS)
@pytest.mark.parametrize("name", list(more_structures()))
def test_lammps_matches_torch(kernels, name):
    from test_lammps import Data
    from ilmlip.jax_md import tables
    from ilmlip.lammps import ILMlipMLIAP
    m = tiny()
    at = more_structures()[name]
    periodic = bool(at.pbc.any())
    ref = m(collate([vcut.prune(graph(at, m.rcut, m.species), m)]), compute_stress=periodic)
    model = ILMlipMLIAP(tables(m), "float64", kernels)
    d = Data(at, m.rcut)
    model.compute_forces(d)
    assert d.total == pytest.approx(float(ref["energy"][0]), abs=1e-10 * max(1.0, abs(float(ref["energy"][0]))))
    np.testing.assert_allclose(d.eatoms, ref["atomic_energy"].detach().numpy(), atol=1e-10)
    np.testing.assert_allclose(d.forces(), ref["forces"].detach().numpy(), atol=1e-9)
    if periodic:
        w = np.einsum("pa,pb->ab", d.rij, d.fij) / at.get_volume()
        np.testing.assert_allclose([w[0, 0], w[1, 1], w[2, 2], w[1, 2], w[0, 2], w[0, 1]],
                                   ref["stress"][0].detach().numpy(), atol=1e-9)


@needs_jax
def test_lammps_table_is_as_wide_as_the_pair_cutoffs_need():
    from ase.build import molecule
    from test_lammps import Data
    from ilmlip.jax_md import tables
    from ilmlip.lammps import ILMlipMLIAP
    from ilmlip.verlet import padded_size
    rng, water = np.random.default_rng(0), Atoms(cell=np.eye(3) * 9.86, pbc=True)
    for _ in range(32):
        w = molecule("H2O")
        w.rotate(rng.random() * 360, rng.normal(size=3))
        w.translate(rng.random(3) * 9.86)
        water += w
    m = tiny()
    model = ILMlipMLIAP(tables(m), "float64", "xla")
    d = Data(water, m.rcut)
    model.compute_forces(d)
    assert model._run["width"] < padded_size(int(1.2 * d.npairs / d.nlocal) + 2, granule=4)
    ref = m(collate([vcut.prune(graph(water, m.rcut, m.species), m)]))
    np.testing.assert_allclose(d.forces(), ref["forces"].detach().numpy(), atol=1e-9)


@needs_jax
def test_host_list_has_the_cutoff_of_the_elements_present():
    m = tiny()
    jm = from_torch(m, "float64")
    cu = bulk("Cu", "fcc", a=3.6).repeat(3)
    cu.rattle(0.05, seed=1)
    nacl = structures()["nacl"]
    for at, want in ((cu, RC["Cu"]), (nacl, max(RC["Na"], RC["Cl"]))):
        jc = JaxCalculator(jm, kernels="xla")
        assert close(results(jc, at), results(TorchCalculator(m, dtype="float64"), at), 1e-11)
        assert jc.rcut == want


@needs_jax
def test_lammps_export_for_some_elements(tmp_path):
    from test_lammps import Data
    from ilmlip.jax_md import tables
    from ilmlip.lammps import ILMlipMLIAP
    m = tiny()
    model = ILMlipMLIAP(tables(m), "float64", "xla")
    vcut.lammps_elements(model, ["Cu"], tmp_path / "cu.mliap")
    assert 2 * model.rcutfac == pytest.approx(RC["Cu"])
    with open(tmp_path / "cu.mliap", "rb") as fh:
        model = __import__("pickle").load(fh)
    cu = bulk("Cu", "fcc", a=3.6).repeat(3)
    cu.rattle(0.05, seed=1)
    d = Data(cu, 2 * model.rcutfac)
    model.compute_forces(d)
    ref = m(collate([vcut.prune(graph(cu, m.rcut, m.species), m)]))
    np.testing.assert_allclose(d.forces(), ref["forces"].detach().numpy(), atol=1e-9)
    other = ILMlipMLIAP(tables(m), "float64", "xla")
    vcut.lammps_elements(other, ["Cu"], tmp_path / "cu2.mliap")
    with pytest.raises(RuntimeError):
        other.compute_forces(Data(structures()["nacl"], 2 * other.rcutfac))


def pack_structures(path, model):
    w = PackedWriter(path)
    for at in more_structures().values():
        o = model(collate([vcut.prune(graph(at, model.rcut, model.species), model)]),
                  compute_stress=bool(at.pbc.any()))
        w.add([len(at)], Z=at.numbers, pos=at.positions, forces=o["forces"].detach().numpy(),
              cell=np.asarray(at.cell) if at.pbc.any() else np.zeros((3, 3)), pbc=at.pbc[None],
              energy=[float(o["energy"][0])],
              stress=o["stress"].detach().numpy() if at.pbc.any() else np.full((1, 6), np.nan), src=[0])
    w.close()


@needs_jax
def test_jax_hessian_matches_torch():
    m = tiny()
    at = more_structures()["mixed"]
    H = JaxCalculator(from_torch(m, "float64"), kernels="xla").get_hessian(at)
    ref = TorchCalculator(m, dtype="float64").get_hessian(at)
    assert np.abs(H - ref).max() <= 1e-11 * np.abs(ref).max()


@needs_jax
@pytest.mark.skipif("fused-reference" not in KERNELS, reason="cuequivariance-jax not installed")
def test_jax_hessian_on_rows_of_two_widths():
    """Two Na (cutoff 5 A) among H and O (3.5, 3.8 A): wider rows for the Na, evaluated as blocks of their own."""
    m = tiny()
    at = Atoms("H" * 10 + "O" * 6 + "Na" * 2, positions=np.random.default_rng(2).random((18, 3)) * 7.0,
               cell=np.eye(3) * 7.0, pbc=True)
    jm = from_torch(m, "float64")
    calc = JaxCalculator(jm, kernels="fused-reference")
    H = calc.get_hessian(at)
    assert sum(n for n, _ in calc._nl["key_extra"][2:]) == 2
    ref = TorchCalculator(m, dtype="float64").get_hessian(at)
    assert np.abs(H - ref).max() <= 1e-11 * np.abs(ref).max()
    assert np.abs(H - JaxCalculator(jm, kernels="xla").get_hessian(at)).max() <= 1e-12 * np.abs(ref).max()


@needs_jax
@pytest.mark.skipif("fused-reference" not in KERNELS, reason="cuequivariance-jax not installed")
def test_row_classes_end_with_the_automatic_spatial_order():
    """spatial_order="auto": the row classes of the fused kernels go with the order once the calculator
    has changed structure three times; the energies and forces stay."""
    m = tiny()
    at = Atoms("H" * 10 + "O" * 6 + "Na" * 2, positions=np.random.default_rng(2).random((18, 3)) * 7.0,
               cell=np.eye(3) * 7.0, pbc=True)
    ref = results(TorchCalculator(m, dtype="float64"), at)
    calc = JaxCalculator(from_torch(m, "float64"), kernels="fused-reference")
    wide = []
    for x in (at, structures()["nacl"], structures()["molecule"], structures()["slab"], at):
        out = results(calc, x)
        wide.append(len(calc._nl["key_extra"]) > 2)
    assert wide[0] and not wide[-1] and not calc._nl["ordered"]
    assert close(out, ref, 1e-11)


@pytest.mark.parametrize("resident", ["device", "host"])
def test_packed_loader_has_the_pairs_of_their_cutoffs(tmp_path, resident):
    m = tiny()
    pack_structures(tmp_path / "train", m)
    sp = PackedSplit(tmp_path / "train")
    loader = PackedLoader(sp, m.rcut, device="cpu", dtype=torch.float64, batch_atoms=64, batch_structs=1,
                          shuffle=False, resident=resident, pair_rcut=vcut.by_z(m), log=lambda *a: None)
    assert (tmp_path / "train" / vcut.cache_name(vcut.by_z(m))).exists()
    for b, at in zip(loader, more_structures().values()):
        g = vcut.prune(graph(at, m.rcut, m.species), m)
        key = lambda i, j, s: sorted(zip(i.tolist(), j.tolist(), np.round(s, 6).tolist()))
        shift = b["shift"].detach().numpy()
        assert key(b["ind_i"].numpy(), b["ind_j"].numpy(), shift) == key(g["ind_i"], g["ind_j"], g["shift"])
    again = PackedLoader(sp, m.rcut, device="cpu", resident=resident, pair_rcut=vcut.by_z(m), log=lambda *a: None)
    assert np.array_equal(again.n_edges, loader.n_edges)
    plain = PackedLoader(sp, m.rcut, device="cpu", resident=resident, log=lambda *a: None)
    assert plain.n_edges.sum() > loader.n_edges.sum()
    pack_structures(tmp_path / "train", m)                 # repacking drops the counts of both kinds
    assert not list((tmp_path / "train").glob("n_edges_*.npy"))


def make_data(root, model):
    rng = np.random.default_rng(0)
    for split, n in (("train", 12), ("val", 4)):
        w = PackedWriter(root / split)
        for k in range(n):
            at = bulk("NaCl", "rocksalt", a=5.64) if k % 2 else bulk("Si", "diamond", a=5.43)
            at.positions += rng.normal(0, 0.1, at.positions.shape)
            o = model(collate([graph(at, 5.0, model.species)]), compute_stress=True)
            w.add([len(at)], Z=at.numbers, pos=at.positions, forces=o["forces"].detach().numpy(),
                  cell=np.asarray(at.cell), pbc=at.pbc[None], energy=[float(o["energy"][0])],
                  stress=o["stress"].detach().numpy(), src=[0])
        w.close()
    tr = PackedSplit(root / "train")
    per = np.asarray(tr.energy) / tr.n_atoms
    (root / "meta.json").write_text(json.dumps(dict(
        sources=["data"], e0=lstsq_e0(tr.Z, tr.offset, tr.energy),
        stats=dict(train=dict(e_per_atom_mean=float(per.mean()), e_per_atom_std=float(per.std()))))))


HP = dict(rcut=5.0, lmax=2, lmax_feat=1, n_basis=4, n_channels=4, product_order=3, lmax_ind=1, n_layers=2,
          n_message=3, embed_dim=4, hidden=[8], lr=5e-3)


def test_train_resume_init_and_evaluate(tmp_path):
    make_data(tmp_path / "data", tiny(None))
    (tmp_path / "vc.json").write_text(json.dumps(dict(hparams=dict(HP, rcut_elem={"Na": 4.5, "Si": 4.0, "Cl": 3.8}))))
    (tmp_path / "plain.json").write_text(json.dumps(dict(hparams=dict(HP, rcut=4.0))))
    args = lambda out, hp, *x: ["--packed", str(tmp_path / "data"), "--hparams", str(tmp_path / hp), "--out", str(out),
                                "--device", "cpu", "--dtype", "float64", "--zbl", "--epochs", "2",
                                "--batch-atoms", "24", "--batch-structs", "4", "--warmup", "2", "--val-every", "4",
                                "--val-batches", "2", "--ckpt-every", "100", "--log-every", "2", *x]
    train.main(args(tmp_path / "a", "vc.json"))
    train.main(args(tmp_path / "b", "vc.json", "--stop-after", "3"))
    train.main(args(tmp_path / "b", "vc.json"))
    a = torch.load(tmp_path / "a" / "checkpoint.pt", weights_only=False)
    b = torch.load(tmp_path / "b" / "checkpoint.pt", weights_only=False)
    assert a["step"] == b["step"] > 3 and a["hparams"]["rcut_elem"] == {"Na": 4.5, "Si": 4.0, "Cl": 3.8}
    assert all(torch.equal(a["model"][k], b["model"][k]) for k in a["model"])
    m, _ = load(tmp_path / "a" / "best.pt")
    assert m.rcut_elem["Na"] == 4.5 and m.rcut_elem.get("H") is None and m.rcut == 5.0
    assert list((tmp_path / "data" / "train").glob("n_edges_vc*.npy"))
    # the architecture is the --init checkpoint's: a plain one stays plain
    train.main(args(tmp_path / "base", "plain.json"))
    train.main(args(tmp_path / "c", "vc.json", "--init", str(tmp_path / "base" / "best.pt")))
    c = torch.load(tmp_path / "c" / "checkpoint.pt", weights_only=False)
    assert "rcut_elem" not in c["hparams"] and c["hparams"]["rcut"] == 4.0
    r = evaluate.main([str(tmp_path / "a" / "best.pt"), str(tmp_path / "data"), "--device", "cpu",
                       "--dtype", "float64"])
    assert np.isfinite(r["metrics"]["all"]["f_mae"])


def test_finetune_from_a_model_with_element_cutoffs(tmp_path):
    from ase.io import write
    from ilmlip import finetune
    m = tiny()
    hp = dict(rcut=5.0, lmax=2, lmax_feat=1, n_basis=5, n_channels=4, product_order=3, lmax_ind=1, n_layers=2,
              n_message=3, hidden=[8], embed_dim=4, zbl=True, lr=1e-3, rcut_elem=RC)
    torch.save(dict(model=m.state_dict(), hparams=hp, species=SPECIES), tmp_path / "found.pt")
    rng, frames = np.random.default_rng(0), []
    for _ in range(6):
        at = bulk("NaCl", "rocksalt", a=5.64).repeat((1, 1, 2))
        at.positions += rng.normal(0, 0.1, at.positions.shape)
        o = m(collate([graph(at, 5.0, SPECIES)]))
        at.info["energy"] = float(o["energy"][0]) + 0.2
        at.arrays["forces"] = o["forces"].detach().numpy() * 1.1
        frames.append(at)
    write(tmp_path / "train.xyz", frames, format="extxyz")
    s = finetune.main(["--foundation", str(tmp_path / "found.pt"), "--train", str(tmp_path / "train.xyz"),
                       "--epochs", "2", "--batch-size", "3", "--device", "cpu", "--dtype", "float64",
                       "--out", str(tmp_path / "ft"), "--replay", "none"])
    assert np.isfinite(s["fine_tuned"]["train"]["f_mae"])
    ft, _ = load(tmp_path / "ft" / "model.pt")
    assert ft.rcut_elem == RC


def test_rule():
    rule = vcut.rule()
    assert rule["Cs"] == 6.0 and rule["K"] == 5.38 and rule["Na"] == 4.64
    assert rule["H"] == rule["O"] == rule["F"] == rule["Fe"] == 4.0 and min(rule.values()) == 4.0
    assert rule["P"] == rule["As"] == 6.0 and rule["Se"] == 5.66 and rule["S"] == 5.26
    assert vcut.rule(anions=())["P"] == 4.0
    assert len(rule) == 100


def test_committee_needs_one_neighbour_list():
    with pytest.raises(SystemExit):
        vcut.same([tiny(), tiny(None)])
    vcut.same([tiny(), tiny(seed=1)])
