"""lockstep.imap: tasks sharing one batched model equal tasks run one at a time; failures, early exit, out of memory."""
import os
import time

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)

from ase import units
from ase.md.verlet import VelocityVerlet

from conftest import structures, tiny
from ilmlip.batched import BatchedModel
from ilmlip.jax_md import JaxCalculator, from_torch
from ilmlip import lockstep
from ilmlip.lockstep import imap


def md(calc, item):
    k, atoms, steps = item
    atoms.calc = calc
    VelocityVerlet(atoms, 1.0 * units.fs).run(steps)
    return k, atoms.positions, atoms.get_potential_energy()


def md_or_fail(calc, item):
    if item[0] == 1:
        item[1].positions[0] = item[1].positions[1]     # two atoms on one spot: the model fails
    return md(calc, item)


def md_or_die(calc, item):
    if item[0] == 1:                         # dies waiting for its third result
        real, n = calc._conn.recv_bytes, [0]

        def recv():
            n[0] += 1
            if n[0] == 3:
                time.sleep(0.2)
                os._exit(3)
            return real()
        calc._conn.recv_bytes = recv
    return md(calc, item)


def sleep(calc, item):
    time.sleep(item)
    return item


def model():
    return BatchedModel(from_torch(tiny(), "float64"))


@pytest.mark.parametrize("split", [False, True])
def test_lockstep_md_equals_md_one_at_a_time(split, monkeypatch):
    ref = list(structures().values())
    items = [(k, a.copy(), 8) for k, a in enumerate(ref)]
    jm = from_torch(tiny(), "float64")
    for a in ref:
        a.calc = JaxCalculator(jm, spatial_order=False)
        VelocityVerlet(a, 1.0 * units.fs).run(8)
    if split:                                    # as if the device took most of each step
        monkeypatch.setattr(lockstep, "_DECIDE", 4)
        monkeypatch.setattr(lockstep, "_device_bound", lambda times: True)
    batched, slots = model(), []
    real = batched.submit
    batched.submit = lambda s, k=None: slots.append(tuple(k)) or real(s, k)
    out = sorted(imap(batched, md, items), key=lambda r: r[0])
    assert [k for k, _, _ in out] == [0, 1, 2, 3]
    assert set(slots) == ({(0, 1, 2, 3), (0, 2), (1, 3)} if split else {(0, 1, 2, 3)})
    for a, (_, pos, e) in zip(ref, out):
        assert np.abs(a.positions - pos).max() <= 1e-10
        assert abs(a.get_potential_energy() - e) <= 1e-10 * len(a)


def test_more_tasks_than_processes():
    items = [(k, a, 2) for k, a in enumerate(list(structures().values()) * 2)]
    out = list(imap(model(), md, items, workers=3))
    assert sorted(k for k, _, _ in out) == list(range(8))


def test_a_failing_task_ends_alone():
    items = [(k, a, 3) for k, a in enumerate(structures().values())]
    done = []
    with pytest.raises(Exception):
        for k, _, _ in imap(model(), md_or_fail, items):
            done.append(k)
    assert sorted(done) == [0, 2, 3]


def test_leaving_early_stops_every_process():
    t0 = time.time()
    gen = imap(model(), sleep, [0.0, 60.0, 60.0])
    assert next(gen) == 0.0
    gen.close()
    assert time.time() - t0 < 30


def test_a_task_must_be_importable():
    with pytest.raises(Exception, match="pickle|local"):
        list(imap(model(), lambda calc, item: item, [1, 2]))


def test_a_process_that_dies_ends_alone():
    items = [(k, a, 6) for k, a in enumerate(structures().values())]
    done = []
    with pytest.raises(RuntimeError, match="task 1 ended"):
        for k, _, _ in imap(model(), md_or_die, items):
            done.append(k)
    assert sorted(done) == [0, 2, 3]


def test_out_of_memory_halves_the_calls():
    batched = model()
    real, calls = batched._run, []

    def run(*a, **k):
        calls.append(k["n_struct"])
        if k["n_struct"] > 2:
            raise RuntimeError("RESOURCE_EXHAUSTED: Out of memory")
        return real(*a, **k)
    batched._run = run
    out = list(imap(batched, md, [(k, a, 5) for k, a in enumerate(structures().values())]))
    assert len(out) == 4 and batched.max_structures == 2
    assert calls.count(4) <= 2                   # the full batch is tried at most twice
