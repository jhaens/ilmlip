"""The fused bond sum with G bonds of a row per kernel element equals the sum with one bond per element."""
import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
pytest.importorskip("cuequivariance_jax")
import jax.numpy as jnp
from ase.build import bulk

from conftest import tiny
from test_switches import ALL, build
from ilmlip import jax_md

U, N_RAD, N_Y, N_NODE, N_OUT, N_SRC = 3, 3, 5, 2, 4, 7
BLOCKS = ((3, 4), (2, 6), (2, 5))            # row classes of different widths, one prime


def paths(rng, n, n_node):
    return [(int(rng.integers(N_RAD)), int(rng.integers(N_Y)), int(rng.integers(n_node)), int(rng.integers(N_OUT)),
             float(rng.normal())) for _ in range(n)]


def bonds(rng):
    e = sum(n * K for n, K in BLOCKS)
    rad = rng.normal(size=(e, N_RAD * U))
    rad[rng.random(e) < 0.2] = 0.0           # empty slots: no radial function
    nbr = rng.integers(0, N_SRC, e).astype(np.int32)
    ind_i = np.concatenate([np.repeat(np.arange(r0, r0 + n), K) for (n, K), r0 in zip(BLOCKS, (0, 3, 5))]).astype(np.int32)
    return (jnp.asarray(rad), jnp.asarray(rng.normal(size=(e, N_Y))), jnp.asarray(rng.normal(size=(N_SRC, N_NODE * U))),
            jnp.asarray(nbr), jnp.asarray(ind_i))


@pytest.mark.parametrize("kind", ["edge", "message"])
@pytest.mark.parametrize("group", [2, 3, 4, 5, 6, 12])
def test_group_equals_one_bond_per_element(kind, group):
    rng = np.random.default_rng(1)
    own = paths(rng, 10, N_NODE)
    sources = [(own, N_NODE)] + ([(paths(rng, 6, N_RAD), N_RAD)] if kind == "message" else [])
    call = jax_md._cuex_call(jnp.float64, "naive")
    one = jax_md._cuex_edge(call, "t", N_Y, sources, N_RAD, N_OUT, U)
    many = jax_md._cuex_edge(call, "t", N_Y, sources, N_RAD, N_OUT, U, group)
    ref = jax_md._xla_edge(sources, N_RAD, N_OUT, U)
    rad, Y, x, nbr, ind_i = bonds(rng)
    a = sum(n for n, _ in BLOCKS)
    run = lambda op: lambda rad, Y, x: op(rad, Y, [(x, nbr)], ind_i, BLOCKS, a)
    scalar = lambda f: lambda *z: jnp.sum(jnp.sin(f(*z)) * jnp.arange(1, a * N_OUT * U + 1).reshape(a, -1))
    args = (rad, Y, x)
    for other in (one, ref):
        assert jnp.abs(run(many)(*args) - run(other)(*args)).max() < 1e-12
        for g, h in zip(jax.grad(scalar(run(many)), (0, 1, 2))(*args), jax.grad(scalar(run(other)), (0, 1, 2))(*args)):
            assert jnp.abs(g - h).max() < 1e-12


@pytest.mark.parametrize("group", [1, 2, 3, 4, 6])
def test_backflow_factors_as_an_operand_equal_columns_of_y(group):
    """The back-flow's gated Y columns as their own operand give the sum of the same columns appended to Y."""
    rng = np.random.default_rng(3)
    n_flow = 3
    own = paths(rng, 10, N_NODE)
    flow = [(l, lm, r, o, c) for l, lm, r, o, c in paths(rng, 6, N_RAD)]
    flow = [(l, lm % n_flow, r, o, c) for l, lm, r, o, c in flow]
    call = jax_md._cuex_call(jnp.float64, "naive")
    appended = [(own, N_NODE), ([(l, N_Y + lm, r, o, c) for l, lm, r, o, c in flow], N_RAD)]
    separate = [(own, N_NODE), (flow, N_RAD)]
    old = jax_md._cuex_edge(call, "t", N_Y + n_flow, appended, N_RAD, N_OUT, U)
    new = jax_md._cuex_edge(call, "t", N_Y, separate, N_RAD, N_OUT, U, group, n_flow)
    rad, Y, x, nbr, ind_i = bonds(rng)
    Yf = jnp.asarray(rng.normal(size=(Y.shape[0], n_flow)))
    a = sum(n for n, _ in BLOCKS)
    f_old = lambda rad, Y, Yf, x: old(rad, jnp.concatenate([Y, Yf], axis=1), [(x, nbr)], ind_i, BLOCKS, a)
    f_new = lambda rad, Y, Yf, x: new(rad, Y, [(x, nbr)], ind_i, BLOCKS, a, flow=Yf)
    scalar = lambda f: lambda *z: jnp.sum(jnp.sin(f(*z)) * jnp.arange(1, a * N_OUT * U + 1).reshape(a, -1))
    args = (rad, Y, Yf, x)
    assert jnp.abs(f_new(*args) - f_old(*args)).max() < 1e-12
    for g, h in zip(jax.grad(scalar(f_new), (0, 1, 2, 3))(*args), jax.grad(scalar(f_old), (0, 1, 2, 3))(*args)):
        assert jnp.abs(g - h).max() < 1e-12


def test_group_is_a_divisor_of_the_width():
    assert [jax_md._edge_group(K, 4) for K in (1, 2, 5, 6, 8, 18, 20, 22)] == [1, 2, 1, 3, 4, 3, 4, 2]
    assert jax_md._edge_group(20, 1) == 1 and jax_md._edge_group(4, 12) == 4


def test_table_width_is_a_multiple_of_the_group(monkeypatch):
    m = jax_md.from_torch(tiny(), "float64")
    fused, plain = (jax_md.JaxCalculator(m, kernels=k) for k in ("fused-reference", "xla"))
    for group, want in ((1, (18, 20)), (2, (18, 20)), (3, (18, 21)), (4, (20, 20)), (5, (20, 20)), (8, (24, 24))):
        monkeypatch.setattr(jax_md, "_EDGE_GROUP", group)
        assert (fused._fit(18), fused._fit(20)) == want
        assert (plain._fit(17), plain._fit(18)) == (18, 18)


@pytest.mark.parametrize("bounded", [False, True])
def test_calculator_group_sizes_give_the_same_forces(monkeypatch, bounded):
    """With the bounds the back-flow's gated Y columns are an operand of the fused sum (group > 1) or columns of Y."""
    at = bulk("NaCl", "rocksalt", a=5.64).repeat((4, 4, 3))
    at.positions += np.random.default_rng(0).normal(0, 0.05, at.positions.shape)

    def results(group, chunk):
        monkeypatch.setattr(jax_md, "_EDGE_GROUP", group)
        m = jax_md.from_torch(build(**dict(ALL, bound=16.0, zbl_switch=10.0)) if bounded else tiny(), "float64")   # steps are cached per model
        calc = jax_md.JaxCalculator(m, kernels="fused-reference", chunk_atoms=chunk, compute_stress=True)
        b = at.copy()
        b.calc = calc
        out = np.concatenate([[b.get_potential_energy()], b.get_forces().ravel(), b.get_stress()])
        assert group <= 2 or calc._width % group == 0
        return out
    ref = results(1, None)
    for group, chunk in ((4, None), (3, None), (8, None), (4, 40)):
        assert np.abs(results(group, chunk) - ref).max() <= 1e-11 * max(1.0, np.abs(ref).max())
