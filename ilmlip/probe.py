"""How a structure runs with a model on this machine, in a few minutes: memory, how busy the GPU
is, the time of one model call and of one MD step per integrator, the run time of a long MD, and
whether a supercell or several MDs at once would use the GPU better.

    python -m ilmlip.probe md.xyz --model OAM-S-omat
    python -m ilmlip.probe md.xyz --model OAM-S-omat --steps 1e6 --dt 0.5 --skip multi --json probe.json

The structure is read with ase.io.read (the last frame); it should be ready for MD (cell, pbc).
Numbers are measured on short runs: an MD whose structure changes much (melting, a growing
neighbour count) runs at other speeds.
"""
import argparse
import json
import os
import subprocess
import sys
import threading
import time

import numpy as np

# JAX takes 75 % of the GPU memory at start by default; the probe measures what the model needs,
# and its worker processes share the GPU
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from ase import units                                          # noqa: E402
from ase.io import read                                        # noqa: E402

INTEGRATORS = ("nve", "langevin", "bussi", "nose-hoover", "berendsen", "andersen", "npt-mtk")
SECTIONS = ("integrators", "supercell", "multi")
_SMI = ("utilization.gpu", "power.draw", "power.limit", "temperature.gpu", "clocks.sm", "clocks.max.sm",
        "memory.used", "memory.total")


def integrator(name, atoms, dt_fs, T, rng):
    """An ASE integrator by name, with the usual coupling times."""
    from ase.md.andersen import Andersen
    from ase.md.bussi import Bussi
    from ase.md.langevin import Langevin
    from ase.md.nose_hoover_chain import IsotropicMTKNPT, NoseHooverChainNVT
    from ase.md.nvtberendsen import NVTBerendsen
    from ase.md.verlet import VelocityVerlet
    dt = dt_fs * units.fs
    if name == "nve":
        return VelocityVerlet(atoms, dt)
    if name == "langevin":
        return Langevin(atoms, dt, temperature_K=T, friction=0.01 / units.fs, fixcm=False, rng=rng)
    if name == "bussi":
        return Bussi(atoms, dt, temperature_K=T, taut=100 * units.fs, rng=rng)
    if name == "nose-hoover":
        return NoseHooverChainNVT(atoms, dt, temperature_K=T, tdamp=100 * dt)
    if name == "berendsen":
        return NVTBerendsen(atoms, dt, temperature_K=T, taut=100 * units.fs)
    if name == "andersen":
        return Andersen(atoms, dt, temperature_K=T, andersen_prob=0.01, rng=rng)
    if name == "npt-mtk":
        if atoms.cell.rank < 3:
            raise ValueError("npt-mtk needs a cell with three lattice vectors")
        return IsotropicMTKNPT(atoms, dt, temperature_K=T, pressure_au=1.01325 * units.bar,
                               tdamp=100 * dt, pdamp=1000 * dt)
    raise ValueError(f"unknown integrator {name!r}, one of {', '.join(INTEGRATORS)}")


def gpu_id(device="auto"):
    """The nvidia-smi id of the GPU that `device` ("cuda", "cuda:1") is: CUDA_VISIBLE_DEVICES
    renumbers the GPUs for the program, not for nvidia-smi."""
    k = int(str(device).partition(":")[2] or 0)
    visible = [v.strip() for v in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if v.strip()]
    return visible[k] if k < len(visible) else str(k)


class GpuSampler:
    """nvidia-smi samples of the GPU every `period` s while in a with block, from one nvidia-smi
    process that streams them (starting one per sample would load the host the MD runs on)."""

    def __init__(self, index=0, period=0.1):
        self.index, self.period, self.rows = index, period, []

    @staticmethod
    def available():
        try:
            subprocess.run(["nvidia-smi", "-L"], capture_output=True, check=True, timeout=10)
            return True
        except (OSError, subprocess.SubprocessError):
            return False

    def _cmd(self):
        return ["nvidia-smi", f"--id={self.index}", f"--query-gpu={','.join(_SMI)}", "--format=csv,noheader,nounits"]

    @staticmethod
    def _parse(line):
        vals = []
        for v in line.strip().split(","):
            try:
                vals.append(float(v))
            except ValueError:
                vals.append(float("nan"))
        return dict(zip(_SMI, vals)) if len(vals) == len(_SMI) else None

    def sample(self):
        out = subprocess.run(self._cmd(), capture_output=True, text=True, timeout=10)
        return self._parse(out.stdout)

    def _read(self):
        for line in self._proc.stdout:
            row = self._parse(line)
            if row:
                self.rows.append(row)

    def __enter__(self):
        self.rows = []
        self._proc = subprocess.Popen(self._cmd() + [f"-lms={int(1e3 * self.period)}"], stdout=subprocess.PIPE,
                                      stderr=subprocess.DEVNULL, text=True)
        self._thread = threading.Thread(target=self._read, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._proc.terminate()
        self._proc.wait()
        self._thread.join(timeout=5)

    def summary(self):
        if not self.rows:
            return {}
        col = lambda k: np.array([r[k] for r in self.rows])
        return dict(util_pct=float(np.nanmean(col("utilization.gpu"))),
                    power_w=float(np.nanmean(col("power.draw"))), power_limit_w=float(np.nanmax(col("power.limit"))),
                    temp_c=float(np.nanmax(col("temperature.gpu"))),
                    sm_clock_mhz=float(np.nanmean(col("clocks.sm"))), sm_clock_max_mhz=float(np.nanmax(col("clocks.max.sm"))),
                    gpu_memory_used_mib=float(np.nanmax(col("memory.used"))),
                    gpu_memory_total_mib=float(np.nanmax(col("memory.total"))), samples=len(self.rows))


def device_memory(calc):
    """The JAX allocator's bytes in use, peak and limit on the calculator's device, or {}."""
    dev = getattr(calc, "device", None)
    try:
        s = dev.memory_stats() or {}
    except Exception:
        return {}
    return {k: int(s[k]) for k in ("bytes_in_use", "peak_bytes_in_use", "bytes_limit") if k in s}


def thermalize(atoms, T, rng):
    """Maxwell-Boltzmann momenta at T and no centre-of-mass motion."""
    from ase.md import velocitydistribution as vd
    if hasattr(vd, "thermalize_momenta"):
        vd.thermalize_momenta(atoms, T, rng=rng)
    else:
        vd.MaxwellBoltzmannDistribution(atoms, temperature_K=T, rng=rng)
    vd.Stationary(atoms)


def first_call(atoms, calc):
    """Seconds of the first force call (it compiles, on JAX) and the device memory after it."""
    a = atoms.copy()
    a.calc = calc
    t0 = time.perf_counter()
    a.get_forces()
    return dict(first_call_s=time.perf_counter() - t0, memory=device_memory(calc))


class _nullcontext:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _windows(dyn, seconds, min_steps, windows=5):
    """The median seconds per step over `windows` timed windows of about seconds/windows each:
    a list rebuild or compile in one window does not set the result."""
    per = []
    chunk = 1
    t_end = time.perf_counter() + seconds
    while len(per) < windows or time.perf_counter() < t_end:
        n, t0 = 0, time.perf_counter()
        while n < max(min_steps, 1) or time.perf_counter() - t0 < seconds / windows:
            dyn.run(chunk)
            n += chunk
        per.append((time.perf_counter() - t0) / n)
        if len(per) >= 3 * windows:
            break
    return float(np.median(per)), float(max(per) / min(per))


def settle(dyn, window, limit=30.0):
    """Run until two windows in a row agree on the time per step within 15 % (or `limit` s): the
    first steps compile, widen tables and rebuild lists, some of it in background threads."""
    t_end, last = time.perf_counter() + limit, None
    while time.perf_counter() < t_end:
        n, t0 = 0, time.perf_counter()
        while n < 2 or time.perf_counter() - t0 < window:
            dyn.run(1)
            n += 1
        rate = (time.perf_counter() - t0) / n
        if last is not None and abs(rate - last) < 0.15 * min(rate, last):
            return
        last = rate


def time_md(atoms, calc, name, dt_fs, T, seconds, rng, warmup=10, min_steps=5, sampler=None, spread=False):
    """Seconds per MD step of one integrator: the median of timed windows after `warmup` steps
    and at least a second (the first steps compile, rebuild lists and widen tables); with
    spread, also the slowest window over the fastest."""
    a = atoms.copy()
    a.calc = calc
    thermalize(a, T, rng)
    dyn = integrator(name, a, dt_fs, T, rng)
    dyn.run(warmup)
    settle(dyn, max(0.5, seconds / 5))
    with (sampler if sampler is not None else _nullcontext()):
        s, ratio = _windows(dyn, seconds, min_steps)
    return (s, ratio) if spread else s


def duration(s):
    if s < 120:
        return f"{s:.0f} s"
    if s < 7200:
        return f"{s / 60:.0f} min"
    if s < 3 * 86400:
        return f"{s / 3600:.1f} h"
    return f"{s / 86400:.1f} d"


def near_cubic_repeats(cell, factor, pbc=(True, True, True)):
    """Repeats (a, b, c) with a*b*c close to `factor` that make the cell as cubic as possible,
    along the periodic directions only."""
    lengths = np.where(pbc, np.linalg.norm(np.asarray(cell), axis=1), np.inf)
    best, top = None, int(round(factor)) + 1
    for a in range(1, top + 1 if pbc[0] else 2):
        for b in range(1, top + 1 if pbc[1] else 2):
            c = max(1, int(round(factor / (a * b)))) if pbc[2] else 1
            if not 0.7 * factor <= a * b * c <= 1.4 * factor:
                continue
            L = (lengths * (a, b, c))[np.isfinite(lengths)]
            key = (L.max() / L.min(), abs(a * b * c - factor))
            if best is None or key < best[0]:
                best = (key, (a, b, c))
    return best[1] if best else (1, 1, 1)


def md_worker(args):
    """One of the processes of the 'several processes' test: compiles, says ready, waits for a
    line on stdin, runs MD for `seconds` and prints its steps per second as JSON."""
    from ilmlip import load_calculator
    atoms = read(args.structure)
    calc = load_calculator(args.model, device=args.device, engine=args.engine, dtype=args.dtype)
    rng = np.random.default_rng()
    a = atoms.copy()
    a.calc = calc
    thermalize(a, args.temperature, rng)
    dyn = integrator(args.integrator, a, args.dt, args.temperature, rng)
    dyn.run(10)
    print("ready", flush=True)
    sys.stdin.readline()
    n, t0 = 0, time.perf_counter()
    while time.perf_counter() - t0 < args.seconds:
        dyn.run(5)
        n += 5
    print(json.dumps(dict(steps_per_s=n / (time.perf_counter() - t0))), flush=True)


def processes_test(args, k, seconds):
    """k MD processes on the device at once: their summed steps per second, or None."""
    cmd = [sys.executable, "-m", "ilmlip.probe", args.structure, "--model", str(args.model), "--device",
           args.device, "--engine", args.engine, "--dtype", args.dtype, "--dt", str(args.dt), "--temperature",
           str(args.temperature), "--integrator", args.integrator, "--worker", "--seconds", str(seconds)]
    import tempfile
    cores = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or k)
    env = dict(os.environ, XLA_PYTHON_CLIENT_PREALLOCATE="false", OMP_NUM_THREADS=str(max(1, cores // k)))
    errs = [tempfile.TemporaryFile("w+") for _ in range(k)]
    procs = [subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=e, text=True, env=env)
             for e in errs]

    def why():
        for e in errs:
            e.seek(0)
            lines = [x for x in e.read().splitlines() if x.strip()]
            if lines:
                return lines[-1][:160]
        return "no message"

    try:
        for p in procs:
            if p.stdout.readline().strip() != "ready":
                p.wait(timeout=30)
                print(f"  a worker process ended: {why()}")
                return None
        sampler = GpuSampler(gpu_id(args.device)) if GpuSampler.available() else None
        with (sampler or _nullcontext()):
            for p in procs:
                p.stdin.write("go\n")
                p.stdin.flush()
            rates = [json.loads(p.stdout.readline())["steps_per_s"] for p in procs]
        return dict(steps_per_s=float(sum(rates)), gpu=sampler.summary() if sampler else {})
    except (ValueError, KeyError, OSError, subprocess.TimeoutExpired):
        print(f"  a worker process failed: {why()}")
        return None
    finally:
        for p in procs:
            p.kill()
            p.wait()
        for e in errs:
            e.close()


def _lockstep_md(calc, item):
    """The task of the lockstep test: MD in one of lockstep's processes; steps per second."""
    path, name, dt_fs, T, seconds, seed = item
    atoms = read(path)
    atoms.calc = calc
    rng = np.random.default_rng(seed)
    thermalize(atoms, T, rng)
    dyn = integrator(name, atoms, dt_fs, T, rng)
    dyn.run(20)
    n, t0 = 0, time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        dyn.run(5)
        n += 5
    return n / (time.perf_counter() - t0)


def lockstep_test(args, model, k, seconds):
    """k MDs in lockstep (one batched model call per step): their summed steps per second.  The
    MDs run their timed part at nearly the same time, since every one waits for the others."""
    from ilmlip.lockstep import imap
    from ilmlip.probe import _lockstep_md       # by module name: under -m this file is __main__
    items = [(args.structure, args.integrator, args.dt, args.temperature, seconds, s) for s in range(k)]
    kw = {k: v for k, v in (("dtype", args.dtype), ("device", args.device)) if v != "auto"}
    rates = list(imap(model, _lockstep_md, items, **kw))
    return dict(steps_per_s=float(sum(rates)))


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m ilmlip.probe", description=__doc__.split("\n\n")[0])
    p.add_argument("structure", help="file ase.io.read reads, ready for MD (cell and pbc set)")
    p.add_argument("--model", default="OAM-S-omat", help="pretrained name or checkpoint")
    p.add_argument("--device", default="auto")
    p.add_argument("--engine", default="auto")
    p.add_argument("--dtype", default="auto")
    p.add_argument("--dt", type=float, default=0.5, help="time step (fs)")
    p.add_argument("--temperature", type=float, default=300.0, help="K")
    p.add_argument("--steps", type=float, default=1e6, help="MD length for the run-time estimates")
    p.add_argument("--integrators", default=",".join(INTEGRATORS),
                   help=f"comma-separated, of {', '.join(INTEGRATORS)}")
    p.add_argument("--integrator", default="langevin", help="the integrator of the supercell and multi tests")
    p.add_argument("--seconds", type=float, default=3.0, help="timed seconds per measurement")
    p.add_argument("--max-atoms", type=int, default=300_000, help="largest supercell tried")
    p.add_argument("--copies", default="2,4,8", help="numbers of MDs at once in the multi test")
    p.add_argument("--skip", default="", help=f"comma-separated sections to skip: {', '.join(SECTIONS)}")
    p.add_argument("--json", help="also write the results to this file")
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = p.parse_args(argv)
    if args.worker:
        return md_worker(args)
    skip = {s for s in args.skip.split(",") if s}
    if skip - set(SECTIONS):
        p.error(f"--skip takes {', '.join(SECTIONS)}")
    names = [s for s in args.integrators.split(",") if s]
    for n in names + [args.integrator]:
        if n not in INTEGRATORS:
            p.error(f"unknown integrator {n!r}, one of {', '.join(INTEGRATORS)}")

    from ilmlip import load_calculator
    atoms = read(args.structure)
    if not atoms.pbc.any() or atoms.cell.volume <= 0:
        print("note: the structure has no periodic cell; supercells are skipped")
        skip.add("supercell")
    rng = np.random.default_rng()
    out = dict(structure=args.structure, model=str(args.model), atoms=len(atoms),
               formula=atoms.get_chemical_formula())
    t0 = time.perf_counter()
    calc = load_calculator(args.model, device=args.device, engine=args.engine, dtype=args.dtype)
    out.update(engine=calc.engine, device=str(getattr(calc, "device", "cpu")),
               kernels=getattr(calc, "kernels", None), load_s=time.perf_counter() - t0)
    on_gpu = "cuda" in out["device"].lower() or "gpu" in out["device"].lower()
    sampler = GpuSampler(gpu_id(args.device)) if on_gpu and GpuSampler.available() else None
    if sampler:
        out["gpu_name"] = subprocess.run(["nvidia-smi", f"--id={sampler.index}", "--query-gpu=name", "--format=csv,noheader"],
                                         capture_output=True, text=True).stdout.strip()
        idle = sampler.sample() or {}
        out["gpu_before"] = idle
    print(f"{out['formula']}, {len(atoms)} atoms; {out['model']} on {out['engine']} {out['device']}"
          f"{' (' + out['kernels'] + ' kernels)' if out['kernels'] else ''}{', ' + out['gpu_name'] if sampler else ''}")
    if sampler and idle:
        print(f"GPU before the probe: {idle['utilization.gpu']:.0f} % busy, {idle['memory.used']:.0f} MiB used "
              f"(other processes count too)")

    # model call
    call = first_call(atoms, calc)
    nve = time_md(atoms, calc, "nve", args.dt, args.temperature, args.seconds, rng, sampler=sampler)
    call.update(nve_step_ms=1e3 * nve, us_per_atom=1e6 * nve / len(atoms), atom_steps_per_s=len(atoms) / nve)
    if sampler:
        call["gpu"] = sampler.summary()
        call["gpu"]["during"] = f"NVE MD, {1e3 * nve:.2f} ms/step"
    out["call"] = call
    print(f"\nFirst model call (compiles): {call['first_call_s']:.1f} s.  NVE step: {1e3 * nve:.2f} ms "
          f"= {call['us_per_atom']:.3f} us/atom = {call['atom_steps_per_s'] / 1e6:.2f} M atom-steps/s")
    nl = {k: getattr(calc, k) for k in ("max_neighbours", "n_edges", "n_rebuilds") if hasattr(calc, k)}
    if nl.get("n_edges"):
        print(f"  neighbour list: {nl['n_edges'] / len(atoms):.0f} bonds per atom"
              + (f", at most {nl['max_neighbours']}" if nl.get("max_neighbours") else ""))
    out["neighbours"] = nl
    mem = call["memory"]
    if mem:
        print(f"  device memory: {mem.get('peak_bytes_in_use', 0) / 2**20:.0f} MiB at peak since the probe started, "
              f"with the weights ({mem.get('peak_bytes_in_use', 0) / len(atoms) / 1024:.1f} KiB per atom) of "
              f"{mem.get('bytes_limit', 0) / 2**20:.0f} MiB")
        if os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE") == "false":
            print("  (a plain JAX run takes 75 % of the GPU memory at start unless XLA_PYTHON_CLIENT_PREALLOCATE=false:"
                  " nvidia-smi then shows that pool, not this)")
    g = call.get("gpu") or {}
    if g:
        print(f"  GPU during {g['during']}: busy {g['util_pct']:.0f} % of the time, {g['power_w']:.0f} of "
              f"{g['power_limit_w']:.0f} W, {g['temp_c']:.0f} C, SM clock {g['sm_clock_mhz']:.0f} of "
              f"{g['sm_clock_max_mhz']:.0f} MHz")
        frac = g["power_w"] / g["power_limit_w"] if g["power_limit_w"] else float("nan")
        if g["util_pct"] < 60 or frac < 0.4:
            print("  -> the GPU waits most of the time (launches, host work, small kernels): it stays cool even with "
                  "memory in use. A larger cell or several MDs at once cost little extra per step.")
        else:
            print("  -> the GPU is the bottleneck: per-atom time will not drop much with larger cells.")

    # integrators
    if "integrators" not in skip:
        print(f"\nMD step ({args.dt} fs, {args.temperature:.0f} K) and time for {args.steps:.3g} steps:")
        out["integrators"] = {}
        for n in names:
            try:
                s = time_md(atoms, calc, n, args.dt, args.temperature, args.seconds, rng)
            except Exception as exc:
                print(f"  {n:12s} failed: {type(exc).__name__}: {str(exc)[:120]}")
                continue
            out["integrators"][n] = dict(step_ms=1e3 * s, run_s=s * args.steps,
                                         ns_per_day=args.dt * 1e-6 * 86400 / s)
            print(f"  {n:12s} {1e3 * s:8.2f} ms/step  {duration(s * args.steps):>8s}  "
                  f"({args.dt * 1e-6 * 86400 / s:.2f} ns/day; {1e3 * s - call['nve_step_ms']:+.2f} ms vs NVE)")

    # several MDs at once
    if "multi" not in skip:
        if not on_gpu:
            print("\nSeveral MDs at once: GPU only (on a CPU, split the cores among processes yourself)")
        else:
            single = 1.0 / time_md(atoms, calc, args.integrator, args.dt, args.temperature, args.seconds, rng)
            print(f"\nSeveral {args.integrator} MDs of this structure at once (one alone: {single:.0f} steps/s):")
            out["multi"] = dict(single_steps_per_s=single, lockstep={}, processes={})
            for k in [int(x) for x in args.copies.split(",") if x]:
                lock = proc = None
                try:
                    lock = lockstep_test(args, args.model, k, args.seconds * 2)
                except Exception as exc:
                    print(f"  lockstep x{k}: {type(exc).__name__}: {(str(exc).splitlines() or [''])[0][:100]}")
                proc = processes_test(args, k, args.seconds * 2)
                out["multi"]["lockstep"][k], out["multi"]["processes"][k] = lock, proc
                part = []
                for label, r in (("lockstep", lock), ("processes", proc)):
                    if r:
                        part.append(f"{label} {r['steps_per_s']:7.0f} steps/s summed ({r['steps_per_s'] / single:.1f}x one MD)")
                print(f"  {k:3d} MDs: " + ";  ".join(part or ["failed"]))
            print("  (lockstep: ilmlip.lockstep.imap, one batched model call per step for all; processes: "
                  "independent python processes on the same GPU)")


    # supercells
    if "supercell" not in skip:
        print(f"\nSupercells ({args.integrator} MD step; same model, a fresh calculator each, compiles once):")
        rows, best = [], None
        base = time_md(atoms, calc, args.integrator, args.dt, args.temperature, args.seconds, rng)
        rows.append(dict(repeat=[1, 1, 1], atoms=len(atoms), step_ms=1e3 * base,
                         atom_steps_per_s=len(atoms) / base, memory=call["memory"], gpu=call.get("gpu", {})))
        for factor in (8, 27, 64, 125, 216, 343, 512, 1000):
            rep = near_cubic_repeats(atoms.cell, factor, atoms.pbc)
            n_at = len(atoms) * int(np.prod(rep))
            if n_at > args.max_atoms:
                break
            big = atoms.repeat(rep)
            try:
                c = load_calculator(args.model, device=args.device, engine=args.engine, dtype=args.dtype)
                t = first_call(big, c)
                s, ratio = time_md(big, c, args.integrator, args.dt, args.temperature, args.seconds, rng,
                                   warmup=10, min_steps=2, sampler=sampler, spread=True)
                if sampler:
                    t["gpu"] = sampler.summary()
            except Exception as exc:
                print(f"  {'x'.join(map(str, rep)):8s} {n_at:8d} atoms: {type(exc).__name__}: {(str(exc).splitlines() or [''])[0][:100]}")
                break
            rows.append(dict(repeat=list(rep), atoms=n_at, step_ms=1e3 * s, atom_steps_per_s=n_at / s,
                             memory=t["memory"], gpu=t.get("gpu", {}), window_spread=ratio))
            del c
            if 1e3 * s > 2000 or (t["memory"].get("bytes_limit") and t["memory"].get("peak_bytes_in_use", 0)
                                  > 0.6 * t["memory"]["bytes_limit"]):
                break
        top = max(r["atom_steps_per_s"] for r in rows)
        for r in rows:
            mem = r["memory"].get("peak_bytes_in_use")
            busy = r["gpu"].get("util_pct")
            print(f"  {'x'.join(map(str, r['repeat'])):8s} {r['atoms']:8d} atoms {r['step_ms']:9.2f} ms/step "
                  f"{r['atom_steps_per_s'] / 1e6:6.2f} M atom-steps/s ({r['atom_steps_per_s'] / rows[0]['atom_steps_per_s']:.1f}x)"
                  + (f"  {mem / 2**20:7.0f} MiB" if mem else "") + (f"  GPU busy {busy:.0f} %" if busy is not None else "")
                  + ("  (noisy: windows differ by more than 1.5x)" if r.get("window_spread", 1) > 1.5 else ""))
            if best is None and r["atom_steps_per_s"] >= 0.85 * top:
                best = r
        out["supercell"] = dict(rows=rows, recommended=best)
        if best is not None and best["repeat"] != [1, 1, 1]:
            print(f"  -> {'x'.join(map(str, best['repeat']))} ({best['atoms']} atoms) gets {best['atom_steps_per_s'] / rows[0]['atom_steps_per_s']:.1f}x "
                  f"the atom-steps per second of the cell as given, within 15 % of the best size tried.")
        lim = call["memory"].get("bytes_limit")
        peaks = [(r["atoms"], r["memory"].get("peak_bytes_in_use")) for r in rows if r["memory"].get("peak_bytes_in_use")]
        if lim and len(peaks) >= 2 and peaks[-1][1] > peaks[-2][1]:
            per = (peaks[-1][1] - peaks[-2][1]) / (peaks[-1][0] - peaks[-2][0])
            fit = peaks[-1][0] + (0.9 * lim - peaks[-1][1]) / per
            print(f"  memory: {per / 1024:.1f} KiB per extra atom between the two largest sizes; about {fit:.2g} "
                  f"atoms would fit in 90 % of the {lim / 2**30:.0f} GiB")

    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=1, default=str)
    return out


if __name__ == "__main__":
    main()
