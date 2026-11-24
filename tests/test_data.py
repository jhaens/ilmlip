"""The packed data format: writer, split, loader and the edge-count cache."""
import numpy as np
import pytest
import torch
from ase.io import write

from conftest import SPECIES, structures, tiny
from ilmlip.data import PackedLoader, PackedSplit, PackedWriter, pack
from ilmlip.model import collate, graph


def labelled(model, atoms):
    g = graph(atoms, model.rcut, model.species)
    o = model(collate([g]), compute_stress=bool(atoms.pbc.any()))
    a = atoms.copy()
    a.info["energy"] = float(o["energy"][0])
    a.arrays["forces"] = o["forces"].detach().numpy()
    if atoms.pbc.any():
        a.info["stress"] = o["stress"][0].detach().numpy()
    return a


@pytest.fixture
def dataset(tmp_path):
    m = tiny()
    frames = [labelled(m, a) for a in structures().values()]
    write(tmp_path / "s.xyz", frames, format="extxyz")
    pack([tmp_path / "s.xyz"], tmp_path / "data" / "train")
    return tmp_path / "data" / "train", frames


def test_pack_round_trip(dataset):
    path, frames = dataset
    sp = PackedSplit(path)
    assert len(sp) == len(frames)
    for k, a in enumerate(frames):
        lo, hi = sp.offset[k], sp.offset[k + 1]
        np.testing.assert_array_equal(sp.Z[lo:hi], a.numbers)
        np.testing.assert_allclose(sp.pos[lo:hi], a.positions, atol=1e-5)
        assert abs(sp.energy[k] - a.info["energy"]) < 1e-9


@pytest.mark.parametrize("resident", ["device", "host"])
def test_neighbour_lists_match_vesin(dataset, resident):
    path, frames = dataset
    loader = PackedLoader(PackedSplit(path), 4.0, device="cpu", dtype=torch.float64, batch_atoms=64,
                          batch_structs=1, shuffle=False, resident=resident, edge_cache=False,
                          log=lambda *a: None)
    sp = PackedSplit(path)
    for k, (a, b) in enumerate(zip(frames, loader)):
        a = a.copy()              # the stored (float32) geometry
        a.set_cell(np.asarray(sp.cell[k], float))
        a.positions = np.asarray(sp.pos[sp.offset[k]:sp.offset[k + 1]], float)
        ref = graph(a, 4.0, ["H", "C", "O", "Na", "Si", "Cl", "Cu"])
        mine = sorted(zip(b["ind_i"].tolist(), b["ind_j"].tolist(),
                          *np.round(b["shift"].detach().numpy(), 5).T))
        theirs = sorted(zip(ref["ind_i"].tolist(), ref["ind_j"].tolist(), *np.round(ref["shift"], 5).T))
        assert mine == theirs


@pytest.mark.parametrize("resident", ["device", "host"])
def test_energy_labels_stay_float64(dataset, resident):
    """Total energies of 1e5 eV (OMol25 is not referenced) are not rounded to float32 in float32 batches:
    the packed loader's, collate's and those fine-tuning keeps on the device."""
    from ilmlip import finetune
    path, frames = dataset
    energy = np.asarray(PackedSplit(path).energy) - 123456.789
    np.save(path / "energy.npy", energy)
    loader = PackedLoader(PackedSplit(path), 4.0, device="cpu", dtype=torch.float32, batch_atoms=64,
                          batch_structs=2, shuffle=False, resident=resident, edge_cache=False,
                          log=lambda *a: None)
    graphs = [{**graph(a, 4.0, SPECIES), "energy": e} for a, e in zip(frames, energy)]
    for got in (torch.cat([b["energy"] for b in loader]), collate(graphs, dtype=torch.float32)["energy"],
                finetune.join(finetune.on_device(graphs, "cpu", torch.float32))["energy"]):
        assert got.dtype == torch.float64 and np.array_equal(got.numpy(), energy)


def test_writer_refuses_unknown_arrays(tmp_path):
    w = PackedWriter(tmp_path / "x")
    with pytest.raises(ValueError):
        w.add([1], Z=[1], pos=[[0, 0, 0]], forces=[[0, 0, 0]], velocity=[[0, 0, 0]])


def test_repacking_drops_the_stale_edge_cache(dataset):
    path, frames = dataset
    PackedLoader(PackedSplit(path), 4.0, device="cpu", log=lambda *a: None)       # caches the edge counts
    np.save(path / "group.npy", np.zeros(len(frames), np.int32))
    denser = []
    for a in frames:
        a = a.copy()
        a.set_cell(0.9 * np.asarray(a.cell))
        a.positions *= 0.9
        denser.append(a)
    write(path.parent.parent / "denser.xyz", denser, format="extxyz")
    pack([path.parent.parent / "denser.xyz"], path)
    sp = PackedSplit(path)
    assert sp.group is None
    loader = PackedLoader(sp, 4.0, device="cpu", log=lambda *a: None)
    for k, a in enumerate(denser):
        a.set_cell(np.asarray(sp.cell[k], float))
        a.positions = np.asarray(sp.pos[sp.offset[k]:sp.offset[k + 1]], float)
        assert loader.n_edges[k] == len(graph(a, 4.0, SPECIES)["ind_i"])


def test_positions_rewritten_in_place_are_counted_again(dataset):
    """Arrays rewritten without PackedWriter (which drops the cache): the counts follow the files;
    a cache without a stamp is taken as it is."""
    path, frames = dataset
    quiet = lambda *a: None
    PackedLoader(PackedSplit(path), 4.0, device="cpu", log=quiet)                  # counts and caches
    sp = PackedSplit(path)
    pos, cell = 0.9 * np.asarray(sp.pos), 0.9 * np.asarray(sp.cell)
    np.save(path / "pos.npy", pos)
    np.save(path / "cell.npy", cell)
    loader = PackedLoader(PackedSplit(path), 4.0, device="cpu", log=quiet)
    for k, a in enumerate(frames):
        a = a.copy()
        a.set_cell(cell[k].astype(float))
        a.positions = pos[sp.offset[k]:sp.offset[k + 1]].astype(float)
        assert loader.n_edges[k] == len(graph(a, 4.0, SPECIES)["ind_i"])
    (path / "n_edges_rc4.json").unlink()
    lines = []
    again = PackedLoader(PackedSplit(path), 4.0, device="cpu", log=lines.append)
    assert any("no stamp" in line for line in lines) and (path / "n_edges_rc4.json").exists()
    assert not any(line.startswith("[packed] counted") for line in lines)
    np.testing.assert_array_equal(again.n_edges, loader.n_edges)


@pytest.mark.parametrize("bad", [dict(Z=[101]), dict(charge=[200]), dict(spin=[0])], ids=["Z", "charge", "spin"])
def test_writer_refuses_values_out_of_range(tmp_path, bad):
    arrays = dict(Z=[1], pos=np.zeros((1, 3)), forces=np.zeros((1, 3)), cell=np.eye(3)[None],
                  pbc=np.ones((1, 3), bool), energy=[0.0], stress=np.full((1, 6), np.nan), src=[0])
    with pytest.raises(ValueError):
        PackedWriter(tmp_path).add([1], **{**arrays, **bad})
