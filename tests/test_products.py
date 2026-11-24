"""The products of layer 0 as one polynomial for all channel classes, and with c2 written into the stack."""
import inspect
import json
from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("cuequivariance")
import jax

jax.config.update("jax_enable_x64", True)

import ilmlip
from conftest import SPECIES
from ilmlip import jax_md
from ilmlip.model import ILMlip

SHIPPED = Path(ilmlip.__file__).parent / "configs"
LIMIT = 512


def evaluate(poly, inputs, shape, indices, name):
    """cuEquivariance's numpy reference of a polynomial, as the `call` of jax_md._cuex_stack."""
    out, = poly(*[np.asarray(x) for x in inputs])
    assert out.shape == shape
    calls.append((name, sum(len(d.paths) for _, d in poly.operations)))
    return out


calls = []


def stacks(name, monkeypatch):
    """The (groups, channels) of every product stack of a model of this config."""
    h = json.loads((SHIPPED / f"{name}.json").read_text())["hparams"]
    known = set(inspect.signature(ILMlip.__init__).parameters)
    torch.manual_seed(0)
    T = jax_md.tables(ILMlip(SPECIES, **{k: v for k, v in h.items() if k in known}).eval())
    seen, real = [], jax_md._cuex_stack
    with monkeypatch.context() as m:
        m.setattr(jax_md, "_cuex_call", lambda dt, method: evaluate)
        m.setattr(jax_md, "_cuex_stack", lambda call, groups, u, *a, **kw: seen.append((groups, u)) or real(call, groups, u, *a, **kw))
        jax_md._energy_fns(T, np.float64, "naive")
    return seen


def run(groups, u, A, limit):
    calls.clear()
    apply, rows = jax_md._cuex_stack(evaluate, groups, u, limit)
    return apply(A, len(A)), rows, list(calls)


def per_class(groups, A):
    """Every class on its own channels, no c2 in the stack: what one call per class computes."""
    ends = np.cumsum([0] + [n1 * c for _, n1, _, c in groups])
    out = [run([g], g[3], A[:, e0:e1], 0) for g, e0, e1 in zip(groups, ends[:-1], ends[1:])]
    return np.concatenate([F for F, _, _ in out], axis=1), [r for _, (r,), _ in out]


@pytest.mark.parametrize("name", ["XS", "S", "XL"])
def test_merged_equals_per_class(name, monkeypatch):
    rng = np.random.default_rng(0)
    for groups, u in stacks(name, monkeypatch):
        A = rng.normal(size=(7, sum(n1 * c for _, n1, _, c in groups)))
        ref, ref_rows = per_class(groups, A)
        for limit in (0, LIMIT):
            F, rows, used = run(groups, u, A, limit)
            assert rows == ref_rows
            assert F.shape == ref.shape
            np.testing.assert_allclose(F, ref, rtol=1e-12, atol=1e-12 * np.abs(ref).max())
            names = [n for n, _ in used]
            assert names.count("vv_stack") == 1
            if limit and "vv_c2" not in names:
                assert dict(used)["vv_stack"] <= LIMIT


def head_rows(groups, every):
    """Every `every`-th wanted row of each order of each class: the rows a readout would read."""
    return [{o: list(rows[::every]) for o, rows in want.items()} for _, _, want, _ in groups]


@pytest.mark.parametrize("name", ["XS", "S", "XL"])
def test_head_rows_come_first(name, monkeypatch):
    """F with the rows read first is F with its columns permuted, and a matrix on them (zero but for the head's
    rows) gives the same result when its rows are permuted the same way: the readout reads a slice."""
    rng = np.random.default_rng(1)
    for groups, u in stacks(name, monkeypatch):
        A = rng.normal(size=(5, sum(n1 * c for _, n1, _, c in groups)))
        for limit in (0, LIMIT):
            ref, rows, _ = run(groups, u, A, limit)
            for every in (1, 2, 3):
                head = head_rows(groups, every)
                calls.clear()
                order = [jax_md._head_first(want, h) for (_, _, want, _), h in zip(groups, head)]
                apply, new_rows = jax_md._cuex_stack(evaluate, groups, u, limit, order)
                F = apply(A, len(A))
                assert new_rows == rows
                perm = apply.perm
                if perm is None:
                    np.testing.assert_array_equal(F, ref)
                    continue
                assert sorted(perm) == list(range(ref.shape[1]))
                np.testing.assert_array_equal(F, ref[:, perm])
                # the columns of the rows of head, in F without heads
                read, at = [], 0
                for (_, n1, want, c), h, r in zip(groups, head, rows):
                    start = 0
                    for o in sorted(r):
                        for q, x in enumerate(r[o]):
                            if x in h.get(o, ()):
                                read += range(at + (start + q) * c, at + (start + q + 1) * c)
                        start += len(r[o])
                    at += sum(len(v) for v in r.values()) * c
                W = np.zeros((ref.shape[1], 3))
                W[read] = rng.normal(size=(len(read), 3))
                np.testing.assert_allclose(F @ W[perm], ref @ W, rtol=1e-12, atol=1e-12 * np.abs(ref @ W).max())
                assert set(np.flatnonzero(np.abs(W[perm]).sum(1))) == set(range(len(read)))
