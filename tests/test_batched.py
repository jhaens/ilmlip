"""BatchedModel: many structures in one call, equal to one structure at a time; list rebuilds, device lists, the i-PI driver."""
import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)

from ase.build import bulk

from conftest import KERNELS, structures, tiny
from test_switches import ALL, build
from ilmlip.batched import BatchedModel
from ilmlip.calculator import TorchCalculator
from ilmlip.jax_md import JaxCalculator, from_torch



def reference(calc, atoms):
    a = atoms.copy()
    a.calc = calc
    return a.get_potential_energy(), a.get_forces(), a.get_stress() if a.pbc.any() else np.zeros(6)


@pytest.mark.parametrize("kernels", KERNELS)
def test_one_call_equals_one_structure_at_a_time(kernels):
    """Energies, forces, stresses and virials of a batch (slots in any order, drifting positions) equal single JaxCalculator calls."""
    model = from_torch(tiny(), "float64")
    calc = JaxCalculator(model, kernels=kernels, spatial_order=False)
    batched = BatchedModel(model, kernels=kernels)
    rng = np.random.default_rng(1)
    structs = list(structures().values())
    structs.append(structs[0].copy())                 # five: the structure count is padded to eight
    structs[-1].positions += np.random.default_rng(3).normal(0, 0.05, structs[-1].positions.shape)
    for call in range(3):
        slots = list(rng.permutation(len(structs))) if call == 2 else None
        for a, got in zip(structs, batched.compute(structs, slots)):
            e, f, s = reference(calc, a)
            assert abs(got["energy"] - e) <= 1e-10 * max(1.0, abs(e))
            assert np.abs(got["forces"] - f).max() <= 1e-10
            assert np.abs(got["stress"] - s).max() <= 1e-12
            if a.pbc.any():
                assert np.allclose(got["virial"], -got["stress"][[[0, 5, 4], [5, 1, 3], [4, 3, 2]]] * a.get_volume(),
                                   atol=1e-12)
        for a in structs:
            a.positions += rng.normal(0, 0.01, a.positions.shape)
        if call == 1:                # same slots: no list rebuilt
            assert batched.n_rebuilds == len(structs)


def test_virial_of_a_molecule_does_not_depend_on_the_batch():
    batched = BatchedModel(from_torch(tiny(), "float64"))
    structs = structures()
    mol = next(a for a in structs.values() if not a.pbc.any())
    crystal = next(a for a in structs.values() if a.pbc.all())
    alone = batched.compute([mol], ["m"])[0]
    together = batched.compute([mol, crystal], ["m", "c"])[0]
    assert not alone["virial"].any() and not together["virial"].any()
    assert np.abs(alone["forces"] - together["forces"]).max() <= 1e-12


def _driver(tmp_path, atoms):
    from ilmlip.ipi import Driver
    atoms.write(tmp_path / "t.xyz")
    return Driver(template=str(tmp_path / "t.xyz"), model=BatchedModel(from_torch(tiny(), "float64")))


def _beads(atoms, n, rng):
    from ipi.utils.units import unit_to_internal
    pos = [unit_to_internal("length", "angstrom", atoms.positions + rng.normal(0, 0.1, atoms.positions.shape))
           for _ in range(n)]
    return pos, [unit_to_internal("length", "angstrom", atoms.cell.array.T)] * n


def test_ipi_driver_follows_beads_in_any_order(tmp_path):
    pytest.importorskip("ipi")
    nacl = bulk("NaCl", "rocksalt", a=5.64).repeat((2, 2, 2))
    driver = _driver(tmp_path, nacl)
    pos, cells = _beads(nacl, 6, np.random.default_rng(2))
    first = driver(cells, pos)
    assert driver.model.n_rebuilds == 6
    order = [5, 2, 0, 3, 1, 4]                            # over a socket: any order
    again = driver([cells[k] for k in order], [pos[k] for k in order])
    assert driver.model.n_rebuilds == 6                   # every bead found its list
    for k, out in zip(order, again):
        assert out[0] == pytest.approx(first[k][0], abs=1e-12)
        assert np.abs(out[1] - first[k][1]).max() < 1e-12
    part = driver(cells[3:5], [pos[4], pos[3]])           # a partial batch keeps the other lists
    assert driver.model.n_rebuilds == 6 and part[0][0] == pytest.approx(first[4][0], abs=1e-12)
    padded = driver(cells + cells[:1], pos + pos[5:])     # i-PI pads a socket batch with copies
    assert driver.model.n_rebuilds == 6 and padded[6][0] == padded[5][0]
    single = driver(cells[0], pos[0])
    assert single[0] == pytest.approx(first[0][0], abs=1e-12) and single[3] == ""
    with pytest.raises(ValueError, match="template has 16"):
        driver(cells[0], pos[0][:5])


def test_ipi_driver_error_in_a_thread_ends_the_process(tmp_path):
    pytest.importorskip("ipi")
    import subprocess
    import sys
    from pathlib import Path
    tests = Path(__file__).parent
    bulk("NaCl", "rocksalt", a=5.64).repeat((2, 1, 1)).write(tmp_path / "t.xyz")
    script = f"""
import sys, threading
import numpy as np
sys.path.insert(0, {str(tests)!r})
from conftest import tiny
from ilmlip.batched import BatchedModel
from ilmlip.jax_md import from_torch
from ilmlip.ipi import Driver
driver = Driver(template={str(tmp_path / "t.xyz")!r}, model=BatchedModel(from_torch(tiny(), "float32")))
t = threading.Thread(target=driver, args=([np.eye(3) * 20], [np.zeros((5, 3))]))   # not 4 atoms
t.start(); t.join()
print("still running")
"""
    run = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=600)
    assert run.returncode == 1 and "still running" not in run.stdout
    assert "but the template has 4" in run.stderr


def test_md_moves_start_background_rebuilds_and_stay_exact():
    model = from_torch(tiny(), "float64")
    calc = JaxCalculator(model, spatial_order=False)
    batched = BatchedModel(model, skin=0.2, neighbour_list="host")       # the background threads
    structs = [structures()[k] for k in ("nacl", "slab", "molecule")]
    vel = [np.random.default_rng(k).normal(0, 0.02, a.positions.shape) for k, a in enumerate(structs)]
    background = 0
    for call in range(14):
        if call == 12:
            structs[0].positions += 0.15 * vel[0] / np.abs(vel[0]).max()      # beyond skin/2 at once
        for a, got in zip(structs, batched.compute(structs)):
            e, f, s = reference(calc, a)
            assert abs(got["energy"] - e) <= 1e-10 * max(1.0, abs(e))
            assert np.abs(got["forces"] - f).max() <= 1e-10
            assert np.abs(got["stress"] - s).max() <= 1e-12
        if batched._next:
            background += 1
            for th, _ in list(batched._next.values()):
                th.join()                                    # installed in the next call
        for a, v in zip(structs, vel):
            a.positions += v
    assert background and batched.n_rebuilds > len(structs)
    assert all(st["built"] > 1 for st in batched._slots.values())


def test_one_rebuild_keeps_the_other_candidates_and_compiled_shapes():
    model = from_torch(tiny(), "float64")
    batched = BatchedModel(model, skin=0.3)
    structs = [structures()[k] for k in ("nacl", "slab", "molecule")]
    for _ in range(21):                                      # a younger list would widen the skin
        batched.compute(structs)
    dev = {k: st["dev"] for k, st in batched._slots.items()}
    caps = {k: st["cand_cap"] for k, st in batched._slots.items()}
    shapes, rebuilds = batched._run._cache_size(), batched.n_rebuilds
    structs[1].positions += 0.2                              # beyond skin/2: a synchronous rebuild
    batched.compute(structs)
    assert batched.n_rebuilds == rebuilds + 1
    assert all(batched._slots[k]["dev"] is dev[k] for k in (0, 2))
    assert batched._slots[1]["dev"] is not dev[1] and batched._slots[1]["cand_cap"] == caps[1]
    assert batched._run._cache_size() == shapes


def _equal_to_reference(calc, batched, structs):
    for a, got in zip(structs, batched.compute(structs)):
        e, f, s = reference(calc, a)
        assert abs(got["energy"] - e) <= 1e-10 * max(1.0, abs(e))
        assert np.abs(got["forces"] - f).max() <= 1e-10
        assert np.abs(got["stress"] - s).max() <= 1e-12


def test_a_jump_beyond_half_the_skin_is_caught_on_the_device():
    model = from_torch(tiny(), "float64")
    calc = JaxCalculator(model, spatial_order=False)
    batched = BatchedModel(model, skin=0.3)
    structs = [structures()[k] for k in ("nacl", "slab", "molecule")]
    moved2 = []
    real = batched._calc._moved2
    batched._calc._moved2 = lambda *a: moved2.append(1) or real(*a)
    for _ in range(21):                                      # a younger list would widen the skin
        _equal_to_reference(calc, batched, structs)
    rebuilds = batched.n_rebuilds
    structs[2].positions[3] += [0.1, 0.1, 0.1]               # 0.17 > skin/2 for one atom
    _equal_to_reference(calc, batched, structs)
    assert batched.n_rebuilds == rebuilds + 1 and not moved2  # fixed cells: no host measurement


@pytest.mark.parametrize("neighbour_list", ["host", "device"])
def test_a_changing_cell_follows_the_host_check_and_stays_exact(neighbour_list):
    model = from_torch(tiny(), "float64")
    calc = JaxCalculator(model, spatial_order=False)
    batched = BatchedModel(model, skin=0.3, neighbour_list=neighbour_list)
    structs = [structures()[k] for k in ("nacl", "slab")]
    for call in range(10):
        strain = np.eye(3) + 0.015 * np.array([[1, 0.2, 0], [0.2, -0.5, 0], [0, 0, 1]])
        structs[0].set_cell(structs[0].cell.array @ strain, scale_atoms=True)
        _equal_to_reference(calc, batched, structs)
    assert batched.n_rebuilds > len(structs)


def test_device_lists_are_exact_and_replace_the_host_threads():
    model = from_torch(tiny(), "float64")
    calc = JaxCalculator(model, spatial_order=False)
    batched = BatchedModel(model, skin=0.2, neighbour_list="device")
    structs = [structures()[k] for k in ("nacl", "slab", "molecule")]
    vel = [np.random.default_rng(k).normal(0, 0.02, a.positions.shape) for k, a in enumerate(structs)]
    for call in range(14):
        _equal_to_reference(calc, batched, structs)
        for a, v in zip(structs, vel):
            a.positions += v
    assert batched.n_device_builds >= 3 and batched.n_rebuilds > len(structs) and not batched._next
    assert batched.neighbour_list == "device"


def test_an_outgrown_plan_falls_back_to_the_host_list():
    model = from_torch(tiny(), "float64")
    calc = JaxCalculator(model, spatial_order=False)
    batched = BatchedModel(model, skin=0.2, neighbour_list="device")
    structs = [structures()[k] for k in ("nacl", "slab")]
    _equal_to_reference(calc, batched, structs)
    plan = batched._slots[0]["plan"]
    batched._slots[0]["plan"] = plan[:5] + (1,) + plan[6:]        # bins that hold one atom
    built = batched.n_device_builds
    for a in structs:
        a.positions += 0.05                                       # 0.087: beyond 3/8 of the skin, below 1/2
    _equal_to_reference(calc, batched, structs)
    assert batched.n_device_builds == built + 1                   # the slab only
    assert batched._slots[0]["plan"] == plan and batched.neighbour_list == "device"


def test_two_groups_in_flight_give_what_compute_gives():
    model = from_torch(tiny(), "float64")
    calc = JaxCalculator(model, spatial_order=False)
    batched = BatchedModel(model, skin=0.2, neighbour_list="host")
    structs = list(structures().values())
    vel = [np.random.default_rng(k).normal(0, 0.02, a.positions.shape) for k, a in enumerate(structs)]
    for call in range(8):
        first = batched.submit(structs[:2], [0, 1])
        second = batched.submit(structs[2:], [2, 3])
        for a, got in zip(structs, first.result() + second.result()):
            e, f, s = reference(calc, a)
            assert abs(got["energy"] - e) <= 1e-10 * max(1.0, abs(e))
            assert np.abs(got["forces"] - f).max() <= 1e-10
        assert first.result() is first.result()
        for a, v in zip(structs, vel):
            a.positions += v
    assert batched.n_rebuilds > len(structs)
    single = batched.compute(structs[:1], [0])[0]
    assert abs(single["energy"] - reference(calc, structs[0])[0]) <= 1e-10


@pytest.mark.parametrize("kernels", KERNELS)
def test_all_options_in_one_call(kernels):
    """A model with all switches in a batch equals the torch model."""
    m = build(**ALL)
    batched = BatchedModel(from_torch(m, "float64"), kernels=kernels)
    ref = TorchCalculator(m, dtype="float64")
    structs = list(structures().values())
    for a, got in zip(structs, batched.compute(structs)):
        e, f, s = reference(ref, a)
        assert abs(got["energy"] - e) <= 1e-11 * max(1.0, abs(e))
        assert np.abs(got["forces"] - f).max() <= 1e-11
        assert np.abs(got["stress"] - s).max() <= 1e-12
