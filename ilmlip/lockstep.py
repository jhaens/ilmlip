"""Many simulations on one GPU: each task runs in its own process with its own ASE calculator,
and the model in the calling process computes the forces of all of them in one batched call
(batched.py).

    from ilmlip.lockstep import imap

    def md(calc, T):                       # module level: the processes import it
        atoms = read("water.xyz")
        atoms.calc = calc
        Bussi(atoms, 0.5 * units.fs, temperature_K=T, taut=100 * units.fs,
              rng=np.random.default_rng()).run(20000)
        return T

    if __name__ == "__main__":             # the processes import this file too
        for T in imap("model.pt", md, [300, 400, 500]):
            print(T, "done")

A call is made once every running task has asked for forces, so the tasks step together while
their own work (integrator, thermostat, output) runs in parallel. Tasks, items and results pass
between the processes by pickle; the processes do not use the GPU.
"""
import multiprocessing as mp
import os
import pickle
import signal
import threading
import time

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from .batched import BatchedModel
from .calculator import _damping, _functional, with_d3

_DONE = object()
_CPU_ONLY = {"JAX_PLATFORMS": "cpu", "CUDA_VISIBLE_DEVICES": ""}
_ENV_LOCK = threading.Lock()
_DECIDE = 60                     # calls after which a device-bound run splits into two groups


class LockstepCalculator(Calculator):
    """A task's calculator: sends the structure to the model's process and waits for the result.
    Atoms, cell and charge/spin go as a pickle when they change; each call sends the positions and
    cell as float64 (b"P" + bytes) and gets b"R" + energy, forces and, if computed, the stress."""

    implemented_properties = ["energy", "free_energy", "forces", "stress"]

    def __init__(self, conn):
        super().__init__()
        self._conn, self._sent = conn, None

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        a = self.atoms
        info = {k: a.info[k] for k in ("charge", "spin") if k in a.info}
        s = self._sent
        if (s is None or len(s[0]) != len(a) or not np.array_equal(s[0], a.numbers)
                or not np.array_equal(s[1], a.pbc) or s[2] != info):
            self._sent = (a.numbers.copy(), a.pbc.copy(), info)
            self._conn.send(("structure", self._sent))
        self._conn.send_bytes(b"P" + np.concatenate([a.positions.ravel(), a.cell.array.ravel()]).tobytes())
        m = self._conn.recv_bytes()
        if m[:1] != b"R":
            raise pickle.loads(m)[1]
        r, n = np.frombuffer(m, offset=1), len(a)
        self.results = dict(energy=float(r[0]), free_energy=float(r[0]), forces=r[1:1 + 3 * n].reshape(n, 3).copy())
        if len(r) > 1 + 3 * n:
            self.results["stress"] = r[1 + 3 * n:].copy()


def _send(conn, tag, value):
    try:
        conn.send((tag, value))
    except Exception as exc:                # not picklable
        conn.send((tag, RuntimeError(f"{type(value).__name__}: {value} (could not be pickled: {exc})")))


def _work(conn, task, d3):
    signal.signal(signal.SIGINT, signal.SIG_IGN)       # Ctrl-C: the calling process stops everything
    calc = LockstepCalculator(conn)
    if d3:
        calc = with_d3(calc, *d3)
    while True:
        tag, item = conn.recv()
        if tag == "stop":
            return
        try:
            result = task(calc, item)
        except BaseException as exc:
            _send(conn, "failed", exc)
        else:
            _send(conn, "done", result)


def _start(task, n, d3=None):
    """n processes that see no GPU, and the model's end of their pipes.  d3: (damping, functional)."""
    ctx = mp.get_context("spawn")           # forking a process that holds a GPU is not safe
    conns, procs = [], []
    with _ENV_LOCK:
        saved = {k: os.environ.get(k) for k in _CPU_ONLY}
        os.environ.update(_CPU_ONLY)
        try:
            for _ in range(n):
                here, there = ctx.Pipe()
                conns.append(here)
                p = ctx.Process(target=_work, args=(there, task, d3))
                p.start()                   # a task that cannot be pickled fails here
                there.close()
                procs.append(p)
        except BaseException:
            _stop(conns, procs, now=True)
            raise
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
    return conns, procs


def _place(atoms, m):
    """Positions and cell of a b"P" message into atoms, in place."""
    r, n = np.frombuffer(m, offset=1), len(atoms)
    atoms.positions[:] = r[:3 * n].reshape(n, 3)
    atoms.cell.array[:] = r[3 * n:].reshape(3, 3)
    return atoms


def _submit(model, waiting, slot):
    """Start one call for all waiting tasks: (their pipes, structures, handle or error)."""
    conns = sorted(waiting, key=slot.get)
    structures = [waiting[c] for c in conns]
    waiting.clear()
    try:
        return conns, structures, model.submit(structures, [slot[c] for c in conns])
    except Exception as exc:
        return conns, structures, exc


def _finish(model, submitted, slot):
    """Send the results of a call: (when it started waiting, the pipes of processes that ended).
    A failed call is made again with compute(), which halves the calls when the device ran out
    of memory; failing again, each structure alone, so that only the one at fault gets the error."""
    conns, structures, handle = submitted
    t0 = time.perf_counter()
    try:
        if isinstance(handle, Exception):
            raise handle
        outs = handle.result()
    except Exception:
        try:
            outs = model.compute(structures, [slot[c] for c in conns])
        except Exception:
            outs = []
            for c, a in zip(conns, structures):
                try:
                    outs.append(model.compute([a], [slot[c]])[0])
                except Exception as exc:
                    outs.append(exc)
    dead = []
    for c, r in zip(conns, outs):
        try:
            if isinstance(r, Exception):
                _send(c, "error", r)
            else:
                stress = r["stress"] if model.compute_stress else []
                c.send_bytes(b"R" + np.concatenate([[r["energy"]], r["forces"].ravel(), stress]).tobytes())
        except OSError:
            dead.append(c)
    return t0, dead


def _device_bound(times):
    """Whether the waits for the device (from each finish's start to its result) took most of
    the steps, over the later half of the calls so far (the first ones compile)."""
    start, done = np.array(times[len(times) // 2:]).T
    return np.median(done - start) > 0.5 * np.median(np.diff(start))


def imap(model, task, items, workers=None, dispersion=None, functional=None, **kwargs):
    """task(calc, item) for every item, in `workers` processes (default: one per item), yielding
    results in the order the tasks finish.  task must be module-level (the processes import it).
    model: BatchedModel, checkpoint or pretrained name (kwargs as for BatchedModel).
    dispersion, functional: as for load_calculator; D3 is added in every task's calculator, so it
    runs on the CPU in the task processes.
    A failing task ends alone; its error is raised after the others.  Ctrl-C or leaving the loop
    early stops every task.  When the device is the bottleneck (large structures or models), four
    or more tasks form two groups that take turns on it."""
    items = list(items)
    n = min(workers or len(items), len(items))
    d3 = None if dispersion is None else (_damping(dispersion), _functional(model, functional))
    if not n:
        return
    model = model if isinstance(model, BatchedModel) else BatchedModel(model, **kwargs)
    conns, procs = _start(task, n, d3)
    slot = {c: k for k, c in enumerate(conns)}
    todo, busy, waiting, atoms, errors = iter(items), set(), {}, {}, []

    def ended(c):
        busy.discard(c)
        errors.append(RuntimeError(f"the process of task {slot[c]} ended"))

    def feed(c):
        item = next(todo, _DONE)
        try:
            if item is _DONE:
                c.send(("stop", None))
            else:
                c.send(("item", item))
                busy.add(c)
        except OSError:
            if item is not _DONE:
                ended(c)

    groups, flying, times = [conns], [None], []
    try:
        for c in conns:
            feed(c)
        while busy or any(flying):
            for g, members in enumerate(groups):
                if flying[g]:
                    t0, dead = _finish(model, flying[g], slot)
                    for c in dead:
                        ended(c)
                    if len(times) < _DECIDE:
                        times.append((t0, time.perf_counter()))
                    flying[g] = None
                for c in members:            # every running task of the group asks, or ends
                    while c in busy:
                        try:
                            m = c.recv_bytes()
                        except (EOFError, OSError):
                            ended(c)
                            break
                        if m[:1] == b"P":
                            waiting[c] = _place(atoms[c], m)
                            break
                        tag, value = pickle.loads(m)
                        if tag == "structure":
                            numbers, pbc, info = value
                            atoms[c] = Atoms(numbers, pbc=pbc, info=info)
                            continue
                        busy.discard(c)
                        feed(c)              # a new task asks at once
                        if tag == "done":
                            yield value
                        else:
                            errors.append(value)
                if waiting:
                    flying[g] = _submit(model, waiting, slot)
            if len(times) == _DECIDE:
                times.append(None)           # decided once
                if n >= 4 and _device_bound(times[:-1]):
                    groups, flying = [conns[0::2], conns[1::2]], flying + [None]
    finally:
        _stop(conns, procs, now=bool(busy))
    if errors:
        raise errors[0]


def _stop(conns, procs, now):
    """End the processes: at once when tasks still run, else after their stop message."""
    for p in procs:
        if now and p.is_alive():
            p.terminate()
    for c in conns:
        c.close()
    for p in procs:
        p.join(timeout=None if now else 10)
        if p.is_alive():
            p.terminate()
            p.join()
