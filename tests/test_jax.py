"""The JAX engine (JaxCalculator) against the torch model: kernels, neighbour lists, precision, bounds."""
import inspect
import json

import numpy as np
import pytest
import torch

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)

from ase import Atoms, units
from ase.build import bulk
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet

import test_bounds
from conftest import CONFIGS, KERNELS, SHAPES, SPECIES, results, same_on_device, structures, tiny
from test_switches import ALL, CASES, build, calibrate
from test_vcut import tiny as tiny_vcut
from ilmlip import jax_md
from ilmlip.calculator import TorchCalculator
from ilmlip.jax_md import JaxCalculator, from_torch
from ilmlip.model import ILMlip
from ilmlip.verlet import check_finite, host_neighbours, morton_order

pytestmark = pytest.mark.filterwarnings("error:fused kernels failed:RuntimeWarning")    # no silent fall back to the XLA kernels


@pytest.mark.parametrize("kernels", KERNELS)
@pytest.mark.parametrize("shape", list(SHAPES))
@pytest.mark.parametrize("state", [None, (1, 2)])
def test_matches_torch(kernels, shape, state):
    """float64 equal to torch, for S and XL shapes with and without charge and spin."""
    m = tiny(charge_spin=state is not None, **SHAPES[shape])
    q, s = state or (None, None)
    jc = JaxCalculator(from_torch(m, "float64", q, s), kernels=kernels)
    tc = TorchCalculator(m, dtype="float64", charge=q, spin=s)
    for at in structures().values():
        at.info.update(charge=q, spin=s)
        a, b = results(jc, at), results(tc, at)
        assert np.abs(a - b).max() <= 1e-11 * max(1.0, np.abs(b).max())


def test_verlet_list_follows_md():
    """Forces along an MD run with Verlet rebuilds equal those of fresh neighbour lists."""
    m = tiny()
    at = bulk("NaCl", "rocksalt", a=5.64).repeat((2, 2, 2))
    MaxwellBoltzmannDistribution(at, temperature_K=1500, rng=np.random.default_rng(1))
    calc = JaxCalculator(from_torch(m, "float64"), kernels="xla", skin=0.3)
    at.calc = calc
    ref = TorchCalculator(m, dtype="float64")
    dyn = VelocityVerlet(at, 2 * units.fs)
    for _ in range(6):
        dyn.run(5)
        b = at.copy()
        b.calc = ref
        np.testing.assert_allclose(at.get_forces(), b.get_forces(), atol=1e-10)
    assert calc.n_rebuilds > 1


def check_list_follows_cell(calc, m, at):
    """NPT-like steps (the cell and the atoms move a little) give the results of fresh lists
    with few rebuilds, lists built in the background at an older cell among them; a large
    compression rebuilds."""
    ref, rng, older = TorchCalculator(m, dtype="float64"), np.random.default_rng(3), 0
    at.positions += rng.normal(0, 0.05, at.positions.shape)
    at.calc = calc
    at.get_stress()                    # stress from the start: one compile per shape
    for _ in range(20):
        e = rng.normal(0, 1e-3, (3, 3))
        at.set_cell(at.cell.array @ (np.eye(3) + e + e.T), scale_atoms=True)
        at.positions += rng.normal(0, 0.01, at.positions.shape)
        n = calc.n_rebuilds
        a, b = results(calc, at), results(ref, at)
        assert np.abs(a - b).max() <= 1e-10 * max(1.0, np.abs(b).max())
        older += calc.n_rebuilds > n and not np.array_equal(calc._nl["h0"], at.cell.array)
    assert calc.n_rebuilds <= 5 and older
    at.set_cell(at.cell.array * 0.85, scale_atoms=True)
    a, b = results(calc, at), results(ref, at)
    assert np.abs(a - b).max() <= 1e-10 * max(1.0, np.abs(b).max())
    assert np.array_equal(calc._nl["h0"], at.cell.array)


@pytest.mark.parametrize("spatial_order", [True, False])
def test_list_follows_cell(spatial_order):
    m = tiny()
    calc = JaxCalculator(from_torch(m, "float64"), kernels="xla", skin=0.5, background_compile=False,
                         spatial_order=spatial_order)
    check_list_follows_cell(calc, m, bulk("NaCl", "rocksalt", a=5.64).repeat((2, 2, 2)))


@pytest.mark.skipif("fused-reference" not in KERNELS, reason="cuequivariance-jax not installed")
def test_list_follows_cell_in_chunks():
    m = tiny()
    calc = JaxCalculator(from_torch(m, "float64"), kernels="fused-reference", chunk_atoms=40, skin=0.5,
                         background_compile=False, spatial_order=True)
    check_list_follows_cell(calc, m, bulk("NaCl", "rocksalt", a=5.64).repeat((4, 4, 3)))
    assert calc.n_chunks > 1


@pytest.mark.parametrize("kernels", KERNELS)
def test_a_jump_beyond_half_the_skin_rebuilds_the_list(kernels):
    m = tiny()
    jc = JaxCalculator(from_torch(m, "float64"), kernels=kernels, skin=1.0)
    tc = TorchCalculator(m, dtype="float64")
    at = structures()["nacl"]
    at.calc = jc
    at.get_forces()
    n0 = jc.n_rebuilds
    at.positions[3] += [0.7, 0.0, 0.0]
    a, b = results(jc, at), results(tc, at)
    assert jc.n_rebuilds > n0
    assert np.abs(a - b).max() <= 1e-11 * max(1.0, np.abs(b).max())


def test_a_cell_shrinking_under_the_atoms_rebuilds_the_list():
    """A cell shrinking under fixed atoms rebuilds the list (the step's displacement check sees only the atoms)."""
    m = tiny()
    jc = JaxCalculator(from_torch(m, "float64"), kernels="xla", skin=1.0)
    tc = TorchCalculator(m, dtype="float64")
    at = bulk("NaCl", "rocksalt", a=5.64).repeat((2, 2, 2))
    at.calc = jc
    at.get_forces()
    n0 = jc.n_rebuilds
    at.set_cell(at.cell.array * 0.85)
    a, b = results(jc, at), results(tc, at)
    assert jc.n_rebuilds > n0
    assert np.abs(a - b).max() <= 1e-10 * max(1.0, np.abs(b).max())


def test_stress_from_the_first_request_on():
    """Stress is computed from its first request on and kept in the results afterwards."""
    jc = JaxCalculator(from_torch(tiny(), "float64"), kernels="xla")
    at = structures()["nacl"]
    at.calc = jc
    at.get_forces()
    assert "stress" not in jc.results
    s = at.get_stress()
    assert np.abs(s).max() > 0
    at.positions[0] += 0.01
    at.get_forces()
    assert "stress" in jc.results


def test_packed_counts_are_exact_in_float32():
    """Integer counts packed into the float32 result vector survive exactly."""
    import jax.numpy as jnp
    from ilmlip.jax_md import _pack, _unpack
    f32 = jnp.float32
    pos = jnp.ones((8, 3), f32)
    out = np.asarray(_pack(jnp.asarray(1.5, f32), jnp.zeros(6, f32), pos, pos - 0.5,
                           jnp.asarray([19, 2 ** 31 - 7], jnp.int32), jnp.full((8, 3), 2.0, f32),
                           jnp.full(8, 3.0, f32)))
    assert out.dtype == np.float32
    head, counts, forces, energies, d2 = _unpack(out, 8)
    assert counts == (19, 2 ** 31 - 7) and head[0] == 1.5 and d2 == 0.75
    assert forces.shape == (8, 3) and (forces == 2.0).all() and (energies == 3.0).all()


@pytest.mark.parametrize("n_present", [1, 2, 5])
def test_layer0_pair_matrix_equals_per_bond(n_present, monkeypatch):
    """Layer 0 as one pair matrix or per bond gives the same results for 1, 2 and 5 species present."""
    m = tiny()
    rng = np.random.default_rng(3)
    at = bulk("NaCl", "rocksalt", a=5.64).repeat((2, 2, 2))
    at.positions += rng.normal(0, 0.1, at.positions.shape)
    at.symbols[:n_present] = ["H", "C", "O", "Si", "Cu"][:n_present]
    if n_present == 1:
        at.symbols[:] = "Cu"
    out = {}
    for matrix in (True, False):
        monkeypatch.setattr(jax_md, "_PAIR_MATMUL", 64 if matrix else 0)
        for kernels in KERNELS:
            calc = JaxCalculator(from_torch(m, "float64"), kernels=kernels)
            out[matrix, kernels] = results(calc, at)
            assert ("species" in calc._nl["args"][10]) == matrix and ("layer0" in calc._nl["args"][10]) != matrix
    ref = results(TorchCalculator(m, dtype="float64"), at)
    for v in out.values():
        assert np.abs(v - ref).max() <= 1e-11 * max(1.0, np.abs(ref).max())


def test_morton_order_keeps_neighbours_close():
    """morton_order is a permutation that puts neighbours close together."""
    at = bulk("NaCl", "rocksalt", a=5.64).repeat((10, 10, 10))
    at = at[np.random.default_rng(0).permutation(len(at))]
    order = morton_order(at.positions, at.cell.array, at.pbc)
    assert sorted(order) == list(range(len(at)))
    rank = np.empty_like(order)
    rank[order] = np.arange(len(order))
    i, j = host_neighbours(at.positions, at.cell.array, at.pbc, 4.0)[:2]
    assert np.median(np.abs(rank[i] - rank[j])) < 0.1 * np.median(np.abs(i - j))


def test_spatial_order_returns_the_atoms_order():
    """Results come back in the caller's atom order with the spatial order on and off."""
    m = tiny()
    at = bulk("NaCl", "rocksalt", a=5.64).repeat((3, 3, 2))
    at = at[np.random.default_rng(2).permutation(len(at))]
    at.positions += np.random.default_rng(3).normal(0, 0.1, at.positions.shape)
    ref = results(TorchCalculator(m, dtype="float64"), at)
    for on in (True, False):
        calc = JaxCalculator(from_torch(m, "float64"), kernels="xla", spatial_order=on)
        out = results(calc, at)
        assert (calc._nl["key_extra"][-1]) == on
        assert np.abs(out - ref).max() <= 1e-11 * max(1.0, np.abs(ref).max())


@pytest.mark.skipif("fused-reference" not in KERNELS, reason="cuequivariance-jax not installed")
def test_spatial_order_auto_ends_after_three_changes_of_structure():
    """MD of one structure keeps the order; single points of many structures lose it from the third
    change of structure on; the results do not depend on it."""
    jm = from_torch(tiny(), "float64")
    auto, on, off = (JaxCalculator(jm, kernels="xla", spatial_order=s) for s in ("auto", True, False))
    nacl = structures()["nacl"]
    for _ in range(3):                                   # other positions: the same structure
        nacl.positions += 0.02
        results(auto, nacl)
    assert auto._swaps == 0 and auto._nl["ordered"]
    ordered = []
    for name in ("triclinic", "slab", "molecule", "nacl", "triclinic"):
        at = structures()[name]
        ref = results(off, at)
        for calc in (auto, on):
            assert np.abs(results(calc, at) - ref).max() <= 1e-11 * max(1.0, np.abs(ref).max())
        assert on._nl["ordered"] and not off._nl["ordered"]
        ordered.append(auto._nl["ordered"])
    assert ordered == [True, True, False, False, False]
    with pytest.raises(ValueError, match="spatial_order"):
        JaxCalculator(jm, kernels="xla", spatial_order="yes")


@pytest.mark.skipif("fused-reference" not in KERNELS, reason="cuequivariance-jax not installed")
def test_spatial_order_in_chunks():
    m = from_torch(tiny(), "float64")
    at = bulk("NaCl", "rocksalt", a=5.64).repeat((4, 4, 3))
    at = at[np.random.default_rng(4).permutation(len(at))]
    at.positions += np.random.default_rng(5).normal(0, 0.05, at.positions.shape)
    one = results(JaxCalculator(m, kernels="fused-reference", chunk_atoms=None, spatial_order=False), at)
    calc = JaxCalculator(m, kernels="fused-reference", chunk_atoms=40)
    chunked = results(calc, at)
    assert calc.n_chunks > 1 and calc._nl["key_extra"][0] is True
    assert np.abs(one - chunked).max() <= 1e-11 * max(1.0, np.abs(one).max())


@pytest.mark.skipif("fused-reference" not in KERNELS, reason="cuequivariance-jax not installed")
def test_out_of_memory_is_not_a_kernel_failure(monkeypatch):
    m = tiny()
    at = structures()["nacl"]
    calc = JaxCalculator(from_torch(m, "float64"), kernels="fused-reference")

    def oom(avals):
        raise RuntimeError("RESOURCE_EXHAUSTED: Out of memory while trying to allocate 9.99GiB")
    monkeypatch.setattr(calc, "_zeros", oom)
    with pytest.raises(RuntimeError, match="RESOURCE_EXHAUSTED"):
        results(calc, at)
    assert calc.kernels == "fused-reference"

    def broken(avals):
        raise RuntimeError("CUDA_ERROR_INVALID_PTX")
    calc = JaxCalculator(from_torch(m, "float64"), kernels="fused-reference")
    monkeypatch.setattr(calc, "_zeros", broken)
    with pytest.warns(RuntimeWarning, match="using XLA"):
        a = results(calc, at)
    b = results(TorchCalculator(m, dtype="float64"), at)
    assert calc.kernels == "xla" and np.abs(a - b).max() <= 1e-11 * max(1.0, np.abs(b).max())


def test_chunked_kernel_failure_falls_back_to_xla(monkeypatch):
    def broken(*args):
        raise RuntimeError("CUDA_ERROR_INVALID_PTX")
    monkeypatch.setattr(jax_md, "_cuex_call", broken)
    m = tiny()
    at = bulk("NaCl", "rocksalt", a=5.64).repeat((4, 4, 3))
    at.positions += np.random.default_rng(5).normal(0, 0.05, at.positions.shape)
    calc = JaxCalculator(from_torch(m, "float64"), kernels="fused-reference", chunk_atoms=40)
    with pytest.warns(RuntimeWarning, match="using XLA"):
        a = results(calc, at)
    b = results(TorchCalculator(m, dtype="float64"), at)
    assert calc.n_chunks == 0 and calc.kernels == "xla"    # XLA runs as one call
    assert np.abs(a - b).max() <= 1e-11 * max(1.0, np.abs(b).max())


CUTOFFS, WIDE_ROWS = {}, {}
CUTOFFS["elements"] = dict(rcut_elem={"H": 3.0, "Na": 3.5, "Cl": 3.8})
WIDE_ROWS["sodium"] = dict(rcut=5.0, rcut_elem={"H": 3.0, "O": 3.0, "Na": 5.0})


def close(calc, ref, at, tol=1e-12):
    a, b = results(calc, at), results(ref, at)
    return np.abs(a - b).max() <= tol * max(1.0, np.abs(b).max())


def real_config(name):
    """A configuration file on conftest's species, with non-zero FiLM and readout."""
    h = json.loads((CONFIGS / f"{name}.json").read_text())["hparams"]
    known = set(inspect.signature(ILMlip.__init__).parameters)
    torch.manual_seed(0)
    m = ILMlip(SPECIES, **{k: v for k, v in h.items() if k in known})
    with torch.no_grad():
        for p in m.parameters():
            if p.abs().sum() == 0:
                p.normal_(0, 0.1)
        m.feat_std.uniform_(0.5, 1.5)
    return m.eval()


@pytest.mark.parametrize("case", CASES)
def test_cases_match_torch(case):
    """Each switch case of test_switches equals torch."""
    m = build(**CASES[case])
    assert close(JaxCalculator(from_torch(m, "float64"), kernels="xla"), TorchCalculator(m, dtype="float64"),
                 structures()["triclinic"])


@pytest.mark.parametrize("kernels", KERNELS)
def test_all_options_match_torch(kernels):
    """All switches together equal torch on every test structure."""
    m = build(**ALL)
    jc, tc = JaxCalculator(from_torch(m, "float64"), kernels=kernels), TorchCalculator(m, dtype="float64")
    for at in structures().values():
        assert close(jc, tc, at)
    assert jc.kernels == kernels                       # no fall back to the XLA kernels


@pytest.mark.parametrize("kernels", KERNELS)
def test_real_configuration(kernels):
    """The shipped S configuration equals torch."""
    m = real_config("S")
    assert close(JaxCalculator(from_torch(m, "float64"), kernels=kernels), TorchCalculator(m, dtype="float64"),
                 structures()["nacl"])


@pytest.mark.parametrize("kernels", KERNELS)
def test_many_species_per_bond(kernels, monkeypatch):
    """Beyond _PAIR_MATMUL pairs: layer 0 as bond sums per width of the taper groups, pair weights per bond."""
    monkeypatch.setattr(jax_md, "_PAIR_MATMUL", 1)
    m = build(**ALL)
    calc = JaxCalculator(from_torch(m, "float64"), kernels=kernels)
    assert close(calc, TorchCalculator(m, dtype="float64"), structures()["nacl"])
    assert "layer0" in calc._nl["args"][10]


@pytest.mark.parametrize("kernels", KERNELS)
@pytest.mark.parametrize("cutoffs", CUTOFFS)
def test_element_cutoffs(kernels, cutoffs):
    """Element cutoffs with all switches equal torch."""
    m = build(**ALL, **CUTOFFS[cutoffs])
    assert close(JaxCalculator(from_torch(m, "float64"), kernels=kernels), TorchCalculator(m, dtype="float64"),
                 structures()["nacl"])


@pytest.mark.parametrize("kernels", KERNELS)
def test_charge_spin_folded(kernels):
    """A model with charge and spin folded to one state equals torch at that state."""
    m = build(**ALL, charge_spin=True)
    q, s = 1, 2
    at = structures()["molecule"]
    at.info.update(charge=q, spin=s)
    assert close(JaxCalculator(from_torch(m, "float64", q, s), kernels=kernels),
                 TorchCalculator(m, dtype="float64", charge=q, spin=s), at)


@pytest.mark.skipif("fused-reference" not in KERNELS, reason="cuequivariance-jax not installed")
@pytest.mark.parametrize("neighbour_list", ["host", "device"])
@pytest.mark.parametrize("cutoffs", WIDE_ROWS)
def test_row_classes_along_md(neighbour_list, cutoffs):
    """Table rows of two widths, lists rebuilt along MD on the host or the device."""
    m = build(**ALL, **WIDE_ROWS[cutoffs])
    at = Atoms("H" * 40 + "O" * 22 + "Na" * 2, positions=np.indices((4, 4, 4)).reshape(3, -1).T * 2.25,
               cell=np.eye(3) * 9.0, pbc=True)
    at.positions += np.random.default_rng(2).normal(0, 0.1, at.positions.shape)
    MaxwellBoltzmannDistribution(at, temperature_K=1500, rng=np.random.default_rng(1))
    calc = JaxCalculator(from_torch(m, "float64"), kernels="fused-reference", skin=0.3, neighbour_list=neighbour_list)
    ref = TorchCalculator(m, dtype="float64")
    assert close(calc, ref, at, 1e-11)
    assert calc._nl["key_extra"][-1][0] == 2
    at.calc = calc
    dyn = VelocityVerlet(at, 1 * units.fs)
    for _ in range(3):
        dyn.run(4)
        b = at.copy()
        b.calc = ref
        np.testing.assert_allclose(at.get_forces(), b.get_forces(), atol=1e-10)


@pytest.mark.skipif("fused-reference" not in KERNELS, reason="cuequivariance-jax not installed")
def test_rows_cap_traces_the_step():
    calc = JaxCalculator(from_torch(build(**ALL), "float64"), kernels="fused-reference")
    assert 0 < calc._rows_cap(20, 100_000) < 1 << 40


def test_message_polynomial_takes_the_backflow_from_the_radial_operand():
    """fused.message_poly in cuequivariance's own evaluator (torch, CPU) equals the XLA bond sum, and its
    backward polynomial gives the gradients of an explicit einsum reference for all three inputs."""
    cuet = pytest.importorskip("cuequivariance_torch")
    jnp = jax.numpy
    from itertools import product
    from ilmlip.fused import message_poly
    u, n_rad, n_back, n_y, n_node, n_out, a, K, N = 3, 2, 2, 4, 4, 5, 6, 4, 7
    rng = np.random.default_rng(0)
    def unique(q0, n_q, n):
        keys = list(product(range(n_rad), range(n_y), range(q0, q0 + n_q), range(n_out)))
        return [(*keys[i], float(rng.normal())) for i in rng.choice(len(keys), n, replace=False)]
    paths, back = unique(0, n_node, 9), unique(n_rad, n_back, 6)      # duplicates break the backward polynomial
    sources = [(paths, n_node), (back, n_rad + n_back)]
    rb, Y = rng.normal(size=(a * K, (n_rad + n_back) * u)), rng.normal(size=(a * K, n_y))
    It, nbr, ind_i = rng.normal(size=(N, n_node * u)), rng.integers(N, size=a * K), np.repeat(np.arange(a), K)
    want = jax_md._xla_edge(sources, n_rad + n_back, n_out, u)(
        jnp.asarray(rb), jnp.asarray(Y), [(jnp.asarray(It), jnp.asarray(nbr))], jnp.asarray(ind_i), ((a, K),), a)
    poly = message_poly(sources, n_rad + n_back, n_y, n_out, u)
    t = lambda x: torch.tensor(x, dtype=torch.float64)
    nbr_t, ind_t = torch.tensor(nbr), torch.tensor(ind_i)
    x = [t(rb).requires_grad_(), t(Y).requires_grad_(), t(It).requires_grad_()]
    got = cuet.SegmentedPolynomial(poly, method="naive")(x, {2: nbr_t}, {0: torch.zeros(a, 1)}, {0: ind_t})[0]
    np.testing.assert_allclose(got.detach().numpy(), np.asarray(want).reshape(a, -1), atol=1e-12)

    R, node = x[0].reshape(-1, n_rad + n_back, u), x[2].reshape(N, n_node, u)[nbr_t]
    bond = [sum(c * R[:, r] * x[1][:, lm, None] * (node if k == 0 else R)[:, q] for p, k in ((paths, 0), (back, 1))
                for r, lm, q, o2, c in p if o2 == o) for o in range(n_out)]
    ref = torch.zeros(a, n_out, u, dtype=torch.float64).index_add(0, ind_t, torch.stack(bond, 1)).reshape(a, -1)
    np.testing.assert_allclose(got.detach().numpy(), ref.detach().numpy(), atol=1e-12)
    ct = t(rng.normal(size=ref.shape))
    want_grad = torch.autograd.grad(ref, x, ct)
    backward = cuet.SegmentedPolynomial(poly.backward([True] * 3, [True])[0], method="naive")
    grads = backward([v.detach() for v in x] + [ct], {2: nbr_t, 3: ind_t}, {2: torch.zeros(N, 1)}, {2: nbr_t})
    for g, w in zip(grads, want_grad):
        np.testing.assert_allclose(g.numpy(), w.numpy(), atol=1e-11)


def test_float32_model_with_rounded_coupling_factors():
    """lmax_feat 2: factors that float32 rounds, as in the XL and XXL configurations."""
    m = build(lmax_feat=2, dtype=torch.float32)
    at = structures()["triclinic"]
    a, b = results(JaxCalculator(from_torch(m, "float32"), kernels="xla"), at), results(TorchCalculator(m, dtype="float32"), at)
    assert np.abs(a - b).max() <= 1e-4 * max(1.0, np.abs(b).max())


SHIPPED = dict(ALL, bound=16.0, zbl_switch=10.0)    # gate rms, no feat_bound: the whitening stays folded
BOUNDED = dict(test_bounds.CASES, shipped=SHIPPED, shipped_3_layers=dict(SHIPPED, n_layers=3))
FULL = ("all", "all_3_layers", "shipped", "shipped_3_layers")
POINTS = {case: ("nacl", "triclinic", "slab", "molecule", "nacl_half", "squeezed", "nacl_06")
          if case in FULL else ("triclinic", "squeezed") for case in BOUNDED}


def bounded_points():
    """The test structures, the compressed ones of test_bounds and NaCl at 0.6 of its cell."""
    nacl = structures()["nacl"]
    nacl.set_cell(nacl.cell.array * 0.6, scale_atoms=True)
    return {**structures(), **test_bounds.dense(), "nacl_06": nacl}


@pytest.fixture
def xla_for_fused(monkeypatch):
    """Where cuEquivariance is missing, the chunked steps and the 'fused-reference' calculators run the XLA couplings."""
    if "fused-reference" not in KERNELS:
        energies = jax_md._energies
        monkeypatch.setattr(jax_md, "_energies", lambda T, dt, kernels: energies(T, dt, "xla"))


CHUNKED = {
    "S": lambda: tiny(**SHAPES["S"]), "XL": lambda: tiny(**SHAPES["XL"]),
    "all_3_layers": lambda: build(**CASES["all_3_layers"]),
    "bounded_all_3_layers": lambda: build(**BOUNDED["all_3_layers"]),
    "bounded_shipped_3_layers": lambda: build(**BOUNDED["shipped_3_layers"]),
    "element_cutoffs": tiny_vcut,
}


@pytest.mark.parametrize("case", CHUNKED)
def test_chunks_equal_one_call(case, xla_for_fused):
    """Rows of the halo layers, gated one by one where there is a gate, give the energies and forces of the
    dense table, and both those of torch."""
    m = CHUNKED[case]()
    at = bulk("NaCl", "rocksalt", a=5.64).repeat((4, 4, 3))
    at.positions += np.random.default_rng(0).normal(0, 0.05, at.positions.shape)
    jm = from_torch(m, "float64")
    one = results(JaxCalculator(jm, kernels="fused-reference", chunk_atoms=None), at)
    calc = JaxCalculator(jm, kernels="fused-reference", chunk_atoms=40)
    chunked, ref = results(calc, at), results(TorchCalculator(m, dtype="float64"), at)
    assert calc.n_chunks > 1
    assert np.abs(one - chunked).max() <= 1e-11 * np.abs(one).max()
    assert np.abs(one - ref).max() <= 1e-11 * np.abs(ref).max()


def test_tables_carry_the_bounds():
    """jax_md.tables copies the bound keywords, and None for an unbounded model."""
    T = jax_md.tables(build(**BOUNDED["all"]))
    assert (T["bound"], T["bond_scale"], T["feat_bound"], T["zbl_switch"]) == (2.0, 0.05, 8.0, 10.0)
    assert T["bond_rms"].shape == (2,) and (T["bond_rms"] != 1).all() and "arch" in T
    T = jax_md.tables(build(**ALL))
    assert (T["bound"], T["feat_bound"], T["zbl_switch"], T["bond_rms"]) == (None, None, None, None)
    T = jax_md.tables(build(**BOUNDED["shipped"]))
    assert (T["bound"], T["feat_bound"], T["zbl_switch"]) == (16.0, None, 10.0)


@pytest.mark.parametrize("kernels", KERNELS)
@pytest.mark.parametrize("case", BOUNDED)
def test_bounded_matches_torch(case, kernels):
    """Each keyword alone and all four, up to three layers, with Gram readout and per-species layer 0, on
    structures down to 0.5 of the lattice: energy, forces, per-atom energies and stress."""
    m = build(**BOUNDED[case])
    jc, tc = JaxCalculator(from_torch(m, "float64"), kernels=kernels), TorchCalculator(m, dtype="float64")
    points = bounded_points()
    for name in POINTS[case]:
        assert close(jc, tc, points[name]), name


@pytest.mark.parametrize("kernels", KERNELS)
@pytest.mark.parametrize("case", ["all", "shipped"])
def test_bounded_many_species_per_bond(case, kernels, monkeypatch):
    """Layer 0 as bond sums per width of the taper groups, the gate on their concatenation."""
    monkeypatch.setattr(jax_md, "_PAIR_MATMUL", 1)
    m = build(**BOUNDED[case])
    calc = JaxCalculator(from_torch(m, "float64"), kernels=kernels)
    assert close(calc, TorchCalculator(m, dtype="float64"), bounded_points()["nacl_half"])
    assert "layer0" in calc._nl["args"][10]


CONFIG_VARIANTS = dict(rms=dict(bound=2.0, feat_bound=8.0), shipped=dict(bound=16.0, feat_bound=None))
CONFIG_RUNS = [(name, variant) for variant, names in (
    ("rms", ["XS", "S", "M", "L", "XL", "XXL"]), ("shipped", ["S", "XL", "XXL"])) for name in names]


@pytest.mark.parametrize("name,variant", CONFIG_RUNS)
def test_bounded_real_configuration(name, variant):
    """The configurations (taper, FiLM, shared pairs, back-flow, Gram readout, element cutoffs) with
    random weights, at the lattice and at 0.6 of it, under the squash setting (feat_bound) and the shipped one."""
    h = dict(json.loads((test_bounds.CONFIGS / f"{name}.json").read_text())["hparams"], **CONFIG_VARIANTS[variant])
    known = set(inspect.signature(ILMlip.__init__).parameters)
    torch.manual_seed(0)
    m = ILMlip(SPECIES, **{k: (tuple(v) if k == "hidden" else v) for k, v in h.items() if k in known})
    with torch.no_grad():
        for p in m.parameters():
            if p.abs().sum() == 0:
                p.normal_(0, 0.1)
        m.head.normal_(0, 0.1)
        m.elem_bias.normal_(0, 0.1)
    calibrate(m.eval())
    assert m.bound and m.zbl_switch
    assert (m.feat_bound is None) == (variant == "shipped")
    jc, tc = JaxCalculator(from_torch(m, "float64"), kernels="xla"), TorchCalculator(m, dtype="float64")
    points = bounded_points()
    for at in (points["nacl"], points["nacl_06"]):
        assert close(jc, tc, at)


@pytest.mark.parametrize("case", ["all", "shipped"])
def test_bounded_row_classes(case, xla_for_fused):
    """Table rows of two widths (a wide cutoff of Na) with the bounds."""
    m = build(**BOUNDED[case], **WIDE_ROWS["sodium"])
    at = Atoms("H" * 40 + "O" * 22 + "Na" * 2, positions=np.indices((4, 4, 4)).reshape(3, -1).T * 2.25,
               cell=np.eye(3) * 9.0, pbc=True)
    at.positions += np.random.default_rng(2).normal(0, 0.1, at.positions.shape)
    calc = JaxCalculator(from_torch(m, "float64"), kernels="fused-reference", skin=0.3)
    assert close(calc, TorchCalculator(m, dtype="float64"), at, 1e-11)
    assert calc._nl["key_extra"][-1][0] == 2


@pytest.mark.parametrize("case", ["core", "all", "shipped"])
def test_core_switch_dimer(case):
    """A CO dimer from deep in the core (the learned energy is off) through the switch to outside it."""
    m = build(**BOUNDED[case])
    jc, tc = JaxCalculator(from_torch(m, "float64"), kernels="xla"), TorchCalculator(m, dtype="float64")
    zc, seen = m.zbl_switch, set()
    for r in np.linspace(0.45, 0.92, 8):
        at = Atoms("CO", positions=[[0, 0, 0], [r, 0, 0]])
        assert close(jc, tc, at), r
        z = float(m.zbl_atoms(test_bounds.collate([test_bounds.graph(at, m.rcut, m.species)])).min())
        seen.add("core" if z >= zc else "switch" if z > 1e-3 else "outside")
    assert seen >= {"core", "switch"}


def test_bounded_float32_model():
    """A bounded float32 model is within 1e-4 of torch float32."""
    m = build(**BOUNDED["all"], dtype=torch.float32)
    at = structures()["triclinic"]
    a, b = results(JaxCalculator(from_torch(m, "float32"), kernels="xla"), at), results(TorchCalculator(m, dtype="float32"), at)
    assert np.abs(a - b).max() <= 1e-4 * max(1.0, np.abs(b).max())


FULL_TF32, TF32_GEOMETRY = {"DEFAULT", "HIGHEST"}, {"HIGHEST"}


@pytest.mark.parametrize("make, dtype, precisions", [
    (tiny, "tf32", FULL_TF32), (tiny, "float32", TF32_GEOMETRY),
    (lambda: build(**ALL), "tf32", FULL_TF32), (lambda: build(**BOUNDED["all"]), "tf32", FULL_TF32)],
    ids=["tiny-tf32", "tiny-float32", "all-tf32", "bounded-tf32"])
def test_step_keeps_the_precision_policy(make, dtype, precisions):
    """tf32 uses the default matmul precision only for the network's products (the geometry and the pair
    weights stay in full precision), float32 the highest throughout; with x64 on (this module enables it)
    neither has float64 tensors."""
    import re
    calc = JaxCalculator(from_torch(make(), dtype), kernels="xla")
    results(calc, structures()["nacl"])
    avals, static = calc._signature(calc._key(calc._width))
    with jax.default_matmul_precision(calc.model.precision):
        text = calc.model.md_step(calc.kernels, calc._nl["ordered"]).lower(*avals, **static).as_text()
    assert set(re.findall(r"precision = \[(\w+), \w+\]", text)) == precisions
    assert not re.search(r"tensor<[^>]*f64>", text)


@pytest.mark.parametrize("engine", ["auto", "cpp"])
def test_bounded_model_loads_on_cpp(engine):
    """A bounded model loads on the C++ engine (engine auto and cpp) and equals torch."""
    from ilmlip import load_calculator
    m = build(**BOUNDED["all"])
    calc = load_calculator(m, device="cpu", engine=engine, dtype="float64")
    assert calc.engine == "cpp"
    assert close(calc, TorchCalculator(m, dtype="float64"), structures()["triclinic"], 1e-11)


def test_check_finite_is_exact():
    """check_finite accepts large finite positions whose sum overflows; it refuses nan, inf and inf - inf."""
    check_finite(np.zeros((5, 3)))
    check_finite(np.full((2, 3), 1e308))                  # finite, but their sum overflows
    for bad in (np.nan, np.inf, -np.inf):
        pos = np.ones((5, 3))
        pos[3, 1] = bad
        with pytest.raises(FloatingPointError):
            check_finite(pos)
    with pytest.raises(FloatingPointError):               # inf - inf is nan
        check_finite(np.array([[np.inf, -np.inf, 0.0]]))


@pytest.mark.parametrize("kernels", KERNELS)
@pytest.mark.parametrize("mode", ["out", "io"])
def test_host_memory_steps_equal_device_steps(mode, kernels, monkeypatch):
    """Host-memory steps (out, io), with and without spinning, return the bits of the device step."""
    m = from_torch(tiny(), "float64")
    at = bulk("NaCl", "rocksalt", a=5.64).repeat((2, 2, 2))
    at.positions += np.random.default_rng(0).normal(0, 0.05, at.positions.shape)
    ref = results(JaxCalculator(m, kernels=kernels, spatial_order=True), at)
    monkeypatch.setattr(jax_md, "_HOST_MEMORY", mode)
    assert same_on_device(results(JaxCalculator(m, kernels=kernels, spatial_order=True), at), ref, m.device)
    monkeypatch.setattr(jax_md, "_SPIN", True)
    assert same_on_device(results(JaxCalculator(m, kernels=kernels, spatial_order=True), at), ref, m.device)


def test_positions_only_change_skips_the_structure_compare():
    """A positions-only change skips the structure comparison and gives the results of the full one."""
    at = bulk("NaCl", "rocksalt", a=5.64).repeat((2, 2, 1))
    other = at.copy()
    other.numbers[0] = 17
    m = from_torch(tiny(), "float64")
    calc, full = (JaxCalculator(m, kernels="xla", skin=0.3, background_compile=False, async_rebuild=False)
                  for _ in range(2))
    at.calc = calc
    at.get_forces()
    full.calculate(at.copy(), ["energy", "forces"])
    st = calc._nl
    assert calc._same_structure(st, at) and not calc._same_structure(st, other)
    assert calc._same_structure(st, other, same_atoms=True)         # the caller vouches for the numbers
    for k in range(4):
        at.positions += np.random.default_rng(k).normal(0, 0.05, at.positions.shape)
        assert calc.check_state(at) == ["positions"]
        f, e = at.get_forces(), at.get_potential_energy()           # calculate(atoms, [...], ["positions"])
        full.calculate(at.copy(), ["energy", "forces"])             # every change assumed
        assert same_on_device(f, full.results["forces"], calc.device)
        assert same_on_device(e, full.results["energy"], calc.device)


def test_ase_semantics_of_the_snapshot():
    """Results are cached for unchanged atoms, the snapshot is a copy, a change of numbers or cell
    is noticed, a non-finite position is refused before anything is recalculated."""
    at = bulk("NaCl", "rocksalt", a=5.64).repeat((2, 1, 1))
    calc = JaxCalculator(from_torch(tiny(), "float32"), kernels="xla")
    at.calc = calc
    f0 = at.get_forces().copy()
    n = calc._calls
    assert np.array_equal(at.get_forces(), f0) and at.get_potential_energy() and calc._calls == n
    assert calc.atoms is not at and calc.atoms is calc._snap
    at.positions[0, 0] += 0.01                            # in place: the snapshot must not follow
    assert calc.check_state(at) == ["positions"] and not np.array_equal(at.get_forces(), f0)
    assert calc._calls == n + 1 and np.array_equal(calc.atoms.positions, at.positions)
    at.set_cell(at.cell.array * 1.01, scale_atoms=True)
    assert calc.check_state(at) == ["positions", "cell"]
    at.get_forces()
    at.numbers[0] = 17                                    # Na -> Cl, in place
    assert calc.check_state(at) == ["numbers"]
    at.get_forces()
    assert np.array_equal(calc.atoms.numbers, at.numbers) and calc.check_state(at) == []
    calls = calc._calls
    bad = at.copy()
    bad.positions[1, 2] = np.nan
    bad.calc = calc
    with pytest.raises(FloatingPointError):
        bad.get_forces()
    assert calc._calls == calls + 1 and np.array_equal(calc._snap.positions, at.positions)


@pytest.mark.skipif("fused-reference" not in KERNELS, reason="cuequivariance-jax not installed")
@pytest.mark.parametrize("shape", list(SHAPES))
@pytest.mark.parametrize("ranks", [(4, 2), (2, 4)], ids=["bispec_rank_below_rank", "bispec_rank_above_rank"])
@pytest.mark.parametrize("lead", [0, 3])
def test_gram_polynomial_is_the_triangles_and_the_bispectrum(shape, ranks, lead):
    from ilmlip.fused import gram2_poly, stack_plan
    T = from_torch(build(**SHAPES[shape], gram_rank=ranks[0], bispec_rank=ranks[1]), "float64").T
    G, lf = T["arch"]["gram"], T["lmax_feat"]
    r, rb, nb, n_m, R = G["rank"], G["bispec_rank"], G["n_bispec"], (lf + 1) ** 2, 4
    u, n_seg, quad, cubic, at = jax_md._gram2(G, lf, lead)
    assert lead or n_seg * u % 4 == 0
    rng = np.random.default_rng(0)
    P, Q = rng.normal(size=(R, n_m, r)), rng.normal(size=(R, n_m, rb))
    off, tri = 0, []
    for l in range(lf + 1):
        tri.append(np.asarray(jax_md._triangle(r)(P[:, off:off + 2 * l + 1])))
        off += 2 * l + 1
    terms = lambda k, sel: jax_md._coupling_terms(G["products"][k], sel)
    c2, f, names, *_ = stack_plan([(1, ["c2", "c3"], n_m, {3: list(range(nb))}, terms)], rb)
    A = Q.reshape(R, -1)
    bispec = f(*[dict(A=A, c2=c2(A, A)[0])[x] for x in names])[0]
    old = np.concatenate(tri + [bispec], axis=1)
    x = np.zeros((R, 2 * n_m, u))
    x[:, :n_m, :r], x[:, n_m:, :rb] = P, Q
    x = x.reshape(R, -1)
    new = gram2_poly(2 * n_m, u, n_seg, quad, cubic)(x, x)[0]
    assert new.shape == (R, n_seg * u) and np.isfinite(new).all()
    assert np.abs(new[:, at] - old).max() <= 1e-12 * np.abs(old).max()


def test_host_memory_is_on_by_default_for_a_gpu_only(monkeypatch):
    """_host_mode: empty gives out on a GPU and off on a CPU; off and 0 disable it, out and io force it."""
    class Device:
        def __init__(self, platform):
            self.platform = platform
    for setting, gpu, cpu in (("", "out", ""), ("off", "", ""), ("0", "", ""), ("out", "out", "out"), ("io", "io", "io")):
        monkeypatch.setattr(jax_md, "_HOST_MEMORY", setting)
        assert (jax_md._host_mode(Device("gpu")), jax_md._host_mode(Device("cpu"))) == (gpu, cpu), setting


@pytest.mark.parametrize("kernels", KERNELS)
@pytest.mark.parametrize("bounded", [False, True])
def test_gram_projection_with_columns_not_a_multiple_of_4(kernels, bounded):
    """h + n_m (r + rb) = 8 + 9 * 5 = 53 columns of the Gram projection (45 when bounded: the scalars are late)."""
    kw = dict(lmax_feat=2, gram_rank=3, bispec_rank=2)
    m = build(**{**test_bounds.CASES["all"], **kw} if bounded else kw)
    jc, tc = JaxCalculator(from_torch(m, "float64"), kernels=kernels), TorchCalculator(m, dtype="float64")
    for name in ("nacl", "triclinic"):
        assert close(jc, tc, structures()[name])
    assert jc.kernels == kernels                       # no fall back to the XLA kernels


def test_read_columns_is_a_slice_when_the_rows_read_come_first():
    """_read_columns selects a slice when the read rows come first, a gather otherwise; same product."""
    import jax.numpy as jnp
    rng = np.random.default_rng(0)
    x = jnp.asarray(rng.normal(size=(3, 12)))
    for read, gather in ((np.arange(5), False), (np.array([1, 4, 5, 9]), True)):
        W = np.zeros((12, 2))
        W[read] = rng.normal(size=(len(read), 2))
        select, Wr = jax_md._read_columns(W)
        assert Wr.shape[0] % 4 == 0 and np.abs(select(x) @ Wr - x @ W).max() <= 1e-14
        assert ("gather" in str(jax.make_jaxpr(select)(x))) == gather
