"""Device-side Verlet lists (JaxCalculator(neighbour_list="device")): the pairs of vesin,
MD equal to host lists, the fallbacks to the host and the growth of the static sizes."""
import functools

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)

from ase import Atoms, units
from ase.build import bulk
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet

from conftest import close, results, structures, tiny
from ilmlip import jax_md
from ilmlip.calculator import TorchCalculator
from ilmlip.jax_md import JaxCalculator, from_torch
from ilmlip.verlet import host_neighbours

N_PAD, CAP = 128, 1 << 17


def random_structure(kind, rng):
    if kind == "triclinic":                      # positions outside the cell, any order
        at = Atoms("H" * int(rng.integers(30, 110)), pbc=True,
                   cell=np.eye(3) * rng.uniform(8, 14, 3) + rng.normal(0, 1.0, (3, 3)))
        at.set_scaled_positions(rng.random((len(at), 3)) * 3 - 1)
    elif kind == "small":                        # atoms see their own images
        at = Atoms("H" * int(rng.integers(1, 4)), pbc=True, cell=np.eye(3) * rng.uniform(1.5, 4, 3)
                   + rng.normal(0, 0.4, (3, 3)))
        at.set_scaled_positions(rng.random((len(at), 3)))
    elif kind == "open":
        at = Atoms("H" * 60, positions=rng.normal(0, 4, (60, 3)), cell=np.eye(3) * 3.0)
    elif kind == "slab":
        cell = np.diag([rng.uniform(5, 12), rng.uniform(5, 12), 30.0])
        cell[1, 0] = rng.uniform(-2, 2)
        at = Atoms("H" * 70, cell=cell, pbc=[True, True, False])
        at.set_scaled_positions(np.c_[rng.random((70, 2)) * 2 - 0.5, rng.uniform(0.3, 0.6, 70)])
    elif kind == "wire":                         # periodic along a cell vector shorter than the cutoff
        at = Atoms("H" * 20, cell=np.diag([2.0 + rng.random(), 30.0, 30.0]), pbc=[True, False, False])
        at.set_scaled_positions(np.c_[rng.random(20), rng.uniform(0.4, 0.6, (20, 2))])
    else:
        at = bulk("Cu", "fcc", a=3.6).repeat((3, 3, 3))
        at.rattle(0.2, seed=int(rng.integers(1000)))
        at = at[rng.permutation(len(at))]
    return at


@functools.lru_cache(maxsize=None)
def builder(dt, pbc, reach, ordered):
    return jax.jit(jax_md._build_pairs(from_torch(tiny(), "float64").T, np.dtype(dt), N_PAD, CAP, pbc, reach,
                                       4096, N_PAD, ordered, (0, 0)))


def device_pairs(at, cutoff, dt=np.float64, ordered=False, cls=None):
    """The device list of at as {(i, j, S)} in the atoms' numbering, and the species in the list's order."""
    pbc = tuple(bool(p) for p in at.pbc & at.cell.array.any(1))
    n = len(at)
    reach = jax_md._pair_bins(at.positions, np.ones(n, bool), at.cell.array, pbc, cutoff, np)[3]
    pos, mu = np.zeros((N_PAD, 3), dt), np.zeros(N_PAD, np.int32)
    pos[:n] = at.positions
    mu[:n] = np.arange(n) % 2
    cls = np.zeros(2, np.int32) if cls is None else np.asarray(cls, np.int32)
    out, stats = builder(np.dtype(dt).name, pbc, tuple(int(r) for r in reach), ordered)(
        pos, at.cell.array.astype(dt), (np.arange(N_PAD) < n).astype(dt), mu, mu, cls, dt(cutoff - 1), dt(1))
    pairs, most, bins, images, _ = np.asarray(stats)
    assert pairs <= CAP and most <= N_PAD and bins <= 4096 and images
    ci, cj, cs, ok = (np.asarray(x) for x in out[:4])
    assert np.all(np.diff(ci) >= 0) and ok.sum() == pairs and not ok[pairs:].any()
    if ordered:
        mu_l, order, rank = (np.asarray(x) for x in (out[5], out[7], out[8]))
        assert np.array_equal(order[rank], np.arange(N_PAD)) and np.array_equal(mu_l, mu[order])
        assert np.array_equal(np.asarray(out[4]), pos[order])
        ci, cj, mu = order[ci], order[cj], mu_l
    return {(a, b, *s) for a, b, s in zip(ci[ok], cj[ok], np.rint(cs[ok]).astype(int))}, mu[:n]


def vesin_pairs(at, cutoff):
    i, j, S, _ = host_neighbours(at.positions, at.cell.array, at.pbc & at.cell.array.any(1), cutoff)
    return {(a, b, *s) for a, b, s in zip(i, j, S)}


KINDS = ["triclinic", "small", "open", "slab", "wire", "fcc"]


@pytest.mark.parametrize("ordered", [False, True])
@pytest.mark.parametrize("kind", KINDS)
def test_device_list_has_the_pairs_of_vesin(kind, ordered):
    rng = np.random.default_rng(KINDS.index(kind))
    for _ in range(4):
        at, cutoff = random_structure(kind, rng), rng.uniform(3.0, 7.0)
        assert device_pairs(at, cutoff, ordered=ordered)[0] == vesin_pairs(at, cutoff)


def test_device_list_in_float32_differs_only_at_the_cutoff():
    rng = np.random.default_rng(7)
    for kind in KINDS:
        at, cutoff = random_structure(kind, rng), 5.5
        for i, j, *S in device_pairs(at, cutoff, np.float32)[0] ^ vesin_pairs(at, cutoff):
            r = np.linalg.norm(at.positions[j] - at.positions[i] + np.array(S) @ at.cell.array)
            assert abs(r - cutoff) < 1e-4


def test_device_list_puts_the_wide_rows_first():
    at = random_structure("fcc", np.random.default_rng(3))
    got, mu = device_pairs(at, 5.0, ordered=True, cls=[1 << 30, 0])
    assert got == vesin_pairs(at, 5.0)
    assert np.array_equal(mu, np.sort(mu)[::-1])       # species 1 (cls 0) first


def md(calc, steps=6):
    at = bulk("NaCl", "rocksalt", a=5.64).repeat((2, 2, 2))
    at = at[np.random.default_rng(0).permutation(len(at))]
    MaxwellBoltzmannDistribution(at, temperature_K=1500, rng=np.random.default_rng(1))
    at.calc = calc
    dyn, out = VelocityVerlet(at, 2 * units.fs), []
    for _ in range(steps):
        dyn.run(5)
        out.append(np.concatenate([at.positions.ravel(), at.get_forces().ravel()]))
    return at, np.array(out)


@pytest.mark.parametrize("spatial_order", [True, False])
def test_md_with_device_lists_equals_md_with_host_lists(spatial_order):
    m = tiny()
    jm = from_torch(m, "float64")
    calc = JaxCalculator(jm, kernels="xla", skin=0.3, neighbour_list="device", spatial_order=spatial_order)
    at, dev = md(calc)
    _, host = md(JaxCalculator(jm, kernels="xla", skin=0.3, spatial_order=spatial_order))
    assert np.abs(dev - host).max() < 1e-10
    assert calc.n_device_builds > 2 and calc.n_rebuilds == calc.n_device_builds + 1
    assert close(results(calc, at), results(TorchCalculator(m, dtype="float64"), at), 1e-10)


def test_device_list_follows_the_cell():
    """NPT-like steps rebuild on the device at the changed cell; a large compression on the host."""
    m = tiny()
    calc = JaxCalculator(from_torch(m, "float64"), kernels="xla", skin=0.5, neighbour_list="device")
    ref, rng = TorchCalculator(m, dtype="float64"), np.random.default_rng(3)
    at = bulk("NaCl", "rocksalt", a=5.64).repeat((2, 2, 2))
    at.positions += rng.normal(0, 0.05, at.positions.shape)
    for _ in range(20):
        e = rng.normal(0, 1e-3, (3, 3))
        at.set_cell(at.cell.array @ (np.eye(3) + e + e.T), scale_atoms=True)
        at.positions += rng.normal(0, 0.03, at.positions.shape)
        n = calc.n_device_builds
        assert close(results(calc, at), results(ref, at), 1e-10)
        assert calc.n_device_builds == n or np.array_equal(calc._nl["h0"], at.cell.array)
    assert calc.n_device_builds > 1 and calc.n_rebuilds == calc.n_device_builds + 1
    n = calc.n_device_builds
    at.set_cell(at.cell.array * 0.85, scale_atoms=True)
    assert close(results(calc, at), results(ref, at), 1e-10)
    assert calc.n_device_builds == n


@pytest.mark.parametrize("size", ["cap", "occupancy", "bins", "images"])
def test_outgrown_sizes_rebuild_on_the_host_and_grow(size):
    m = tiny()
    calc = JaxCalculator(from_torch(m, "float64"), kernels="xla", skin=1.0, neighbour_list="device")
    ref, at = TorchCalculator(m, dtype="float64"), bulk("NaCl", "rocksalt", a=5.64).repeat((4, 4, 1))
    at.positions += np.random.default_rng(0).normal(0, 0.05, at.positions.shape)
    results(calc, at)
    plan = calc._nl["plan"]
    assert plan["key"][3] == (1, 1, 2)           # the short cell vector needs two images, 2 x 2 x 1 bins
    k = ["cap", "occupancy", "bins", "images"].index(size)
    small = list(plan["key"])
    small[(1, 5, 4, 3)[k]] = (128, 1, 1, (1, 1, 1))[k]
    plan["key"] = tuple(small)
    at.positions[0] += [0.4, 0.0, 0.0]           # past 3/8 of the skin: rebuilt after this step
    assert close(results(calc, at), results(ref, at))
    assert calc.n_device_builds == 0 and calc.n_rebuilds == 2 and calc._nl["plan"]["key"] != tuple(small)
    at.positions[1] += [0.4, 0.0, 0.0]
    assert close(results(calc, at), results(ref, at))
    assert calc.n_device_builds == 1 and calc.n_rebuilds == 3


def test_a_table_overflow_in_the_step_before_a_device_rebuild_redoes_it():
    """The rebuild after the step widens the table for its list: the overflowed step is still redone."""
    m = tiny()
    calc = JaxCalculator(from_torch(m, "float64"), kernels="xla", skin=1.0, neighbour_list="device")
    at = structures()["nacl"]
    results(calc, at)
    calc._width -= 4
    at.positions[0] += [0.4, 0.0, 0.0]
    assert close(results(calc, at), results(TorchCalculator(m, dtype="float64"), at))
    assert calc.n_device_builds == 1 and calc.n_overflows == 1


def test_a_jump_beyond_half_the_skin_rebuilds_on_the_host():
    m = tiny()
    calc = JaxCalculator(from_torch(m, "float64"), kernels="xla", skin=1.0, neighbour_list="device")
    at = structures()["nacl"]
    results(calc, at)
    at.positions[3] += [0.7, 0.0, 0.0]
    assert close(results(calc, at), results(TorchCalculator(m, dtype="float64"), at))
    assert calc.n_rebuilds == 2 and calc.n_device_builds == 0


def test_out_of_memory_leaves_the_lists_to_the_host(monkeypatch):
    m = tiny()
    calc = JaxCalculator(from_torch(m, "float64"), kernels="xla", skin=1.0, neighbour_list="device")
    at = structures()["nacl"]
    results(calc, at)

    def oom(*args):
        raise RuntimeError("RESOURCE_EXHAUSTED: Out of memory while trying to allocate 9.99GiB")
    monkeypatch.setattr(jax_md, "_build_pairs", lambda *a: oom)
    at.positions[0] += [0.4, 0.0, 0.0]
    with pytest.warns(RuntimeWarning, match="host lists from now on"):
        a = results(calc, at)
    assert close(a, results(TorchCalculator(m, dtype="float64"), at))
    assert calc.neighbour_list == "host" and calc.n_device_builds == 0 and calc.n_rebuilds == 2
    assert "plan" not in calc._nl


def test_chunks_keep_host_lists():
    pytest.importorskip("cuequivariance_jax")
    m = tiny()
    jm = from_torch(m, "float64")
    at = bulk("NaCl", "rocksalt", a=5.64).repeat((4, 4, 3))
    at.positions += np.random.default_rng(0).normal(0, 0.05, at.positions.shape)
    calc = JaxCalculator(jm, kernels="fused-reference", chunk_atoms=40, skin=1.0, neighbour_list="device")
    one = results(calc, at)
    assert calc.n_chunks > 1 and calc._nl["plan"] is None
    at.positions[0] += [0.4, 0.0, 0.0]
    assert close(results(calc, at), results(JaxCalculator(jm, kernels="xla"), at))
    assert calc.n_device_builds == 0 and one.shape == results(calc, at).shape


def test_neighbour_list_option():
    jm = from_torch(tiny(), "float64")
    with pytest.raises(ValueError, match="neighbour_list"):
        JaxCalculator(jm, neighbour_list="gpu")
    calc = JaxCalculator(jm, kernels="xla")
    results(calc, structures()["nacl"])
    assert calc.neighbour_list == "host" and "plan" not in calc._nl
