"""finetune.py --wise: N naive fine-tunes that differ only in the order of the batches, one after
the other or in processes of their own, and their weight average."""
import argparse
import contextlib
import json
import multiprocessing.connection
import shutil
import signal
import time
import traceback
from pathlib import Path

import torch

from . import runinfo
from .finetune import Log, fine_tune, parser
from .replay import find_replay, species_of

OWN = ("out", "device", "wise", "wise_parallel", "compile", "muon_graph", "resident", "save_every", "track",
       "track_project", "kernels")


def member_state(d, args):
    """The member folder d's state: "finished", "diverged" or None (to train); other settings are an error."""
    try:
        sm = json.loads((d / "summary.json").read_text())
    except (OSError, ValueError):
        return None
    p = parser()
    diff = [k for k, v in vars(args).items() if k not in OWN and sm["args"].get(k, p.get_default(k)) != v]
    if diff:
        raise SystemExit(f"{d} was trained with other settings ({', '.join(diff)}): choose another --out")
    if not (d / "model.pt").is_file():
        return None
    # an interrupted member writes no summary; fewer epochs than asked mean a stop on a non-finite loss
    return "finished" if sm["epochs_run"] == args.epochs else "diverged"


def check_diverged(dirs, states):
    bad = [d.name for d, s in zip(dirs, states) if s == "diverged"]
    if bad:
        raise SystemExit(f"{', '.join(bad)} stopped without a finite loss (see their log.txt) and would "
                         f"again with these settings: a lower --lr, in another --out")


def _member(d, k, threads):
    """Member k of a --wise run in a process of its own; its log, and an error, go to its log.txt."""
    torch.set_num_threads(threads)
    try:
        fine_tune(argparse.Namespace(**d), member=k, echo=False)
    except (Exception, SystemExit):
        with contextlib.suppress(OSError), open(Path(d["out"]) / "log.txt", "a", encoding="utf-8") as fh:
            fh.write(traceback.format_exc())
        raise


def _trained(k, d):
    try:
        s = json.loads((d / "summary.json").read_text())
    except (OSError, ValueError):
        return f"member {k:02d}  ended without a summary"
    if s["epochs_run"] != s["args"]["epochs"]:
        return f"member {k:02d}  stopped without a finite loss after {s['epochs_run']} of {s['args']['epochs']} epochs"
    v = s["fine_tuned"].get("valid", {})
    return (f"member {k:02d}  best epoch {s['best_epoch']}/{s['epochs_run']}"
            + (f", valid F MAE {v['f_mae']:.1f} meV/A" if "f_mae" in v else "")
            + f", {s['train_seconds'] / 3600:.2f} h")


def _failed(k, d, code):
    where = "its log.txt ends with the error" if (d / "log.txt").is_file() else "the error is on stderr"
    return (f"[error] member {k:02d}  killed by signal {-code}" if code < 0
            else f"[error] member {k:02d}  stopped with exit code {code}; {where}")


def train_parallel(args, todo, dirs, par, log):
    """The members todo, par at a time, each in a process of its own (spawned: CUDA does not survive
    a fork) on the visible GPU with the fewest members, so that a crash costs only its member."""
    gpus = torch.cuda.device_count() if args.device == "cuda" else 0
    load = dict.fromkeys([f"cuda:{i}" for i in range(gpus)] if gpus > 1 else [args.device], 0)
    threads = max(1, torch.get_num_threads() // par)
    ctx, queue, running = multiprocessing.get_context("spawn"), list(todo), {}
    # a SIGTERM to this process alone stops the members too, as Ctrl-C does
    handler = signal.getsignal(signal.SIGTERM)
    with contextlib.suppress(ValueError):
        signal.signal(signal.SIGTERM, signal.default_int_handler)
    try:
        while queue or running:
            while queue and len(running) < par:
                k = queue.pop(0)
                dev = min(load, key=load.get)
                p = ctx.Process(target=_member, args=(dict(vars(args), out=str(dirs[k]), device=dev), k, threads))
                p.start()
                running[p.sentinel] = (k, p, dev)
                load[dev] += 1
                log(f"member {k:02d}  started on {dev}")
            for s in multiprocessing.connection.wait(list(running)):
                k, p, dev = running.pop(s)
                p.join()
                load[dev] -= 1
                log(_trained(k, dirs[k]) if p.exitcode == 0 else _failed(k, dirs[k], p.exitcode))
                if p.exitcode == 0 and member_state(dirs[k], args) == "diverged" and queue:
                    log("           no further members start: this --out cannot be averaged")
                    queue.clear()
    except KeyboardInterrupt:
        for _, p, _ in running.values():
            p.terminate()
        for _, p, _ in running.values():
            p.join()
        raise
    finally:
        with contextlib.suppress(ValueError, TypeError):
            signal.signal(signal.SIGTERM, handler if handler is not None else signal.SIG_DFL)


def wise(args):
    """--wise naive fine-tunes in member_00, member_01, ... and their weight average as model.pt."""
    if args.mode != "naive":
        raise SystemExit("--wise averages naive fine-tunes: use it with --mode naive")
    out = Path(args.out or f"{Path(args.train[0]).stem}_{Path(args.foundation).stem}_naive_wise{args.wise}")
    if (out / "model.pt").exists():
        raise SystemExit(f"{out / 'model.pt'} exists: choose another --out or remove it")
    out.mkdir(parents=True, exist_ok=True)
    log = Log(out / "log.txt")
    log(f"ilmlip fine-tuning, WiSE of {args.wise} naive fine-tunes   ({time.strftime('%Y-%m-%d %H:%M')})")
    runinfo.show(log)
    # prepared once, so that every member measures the drift on the same structures
    rep = find_replay(args.replay, False, args.replay_n, args.seed, out, log, species_of(args.foundation))
    args = argparse.Namespace(**dict(vars(args), out=str(out), replay=str(rep.resolve()) if rep else "none",
                                     replay_source=args.replay))
    dirs = [out / f"member_{k:02d}" for k in range(args.wise)]
    states = [member_state(d, args) for d in dirs]
    check_diverged(dirs, states)
    todo = [k for k, s in enumerate(states) if s is None]
    for k in todo:
        d, old = dirs[k], dirs[k].with_name(dirs[k].name + ".incomplete")
        if d.exists() or d.is_symlink():
            if old.is_dir() and not old.is_symlink():
                shutil.rmtree(old)
            elif old.exists() or old.is_symlink():
                old.unlink()
            d.rename(old)
    par = min(args.wise_parallel, len(todo))
    log(f"members    {args.wise - len(todo)} of {args.wise} finished in {out.resolve()}"
        + (f"; {len(todo)} to train, {max(par, 1)} at a time" if todo else ""))
    try:
        if par > 1:
            train_parallel(args, todo, dirs, par, log)
        else:
            for k in todo:
                fine_tune(argparse.Namespace(**dict(vars(args), out=str(dirs[k]))), member=k)
                log(_trained(k, dirs[k]))
                check_diverged([dirs[k]], [member_state(dirs[k], args)])
    except KeyboardInterrupt:
        log("[stop] interrupted")
        raise SystemExit("interrupted: the same command trains the members that did not finish") from None
    states = [member_state(d, args) for d in dirs]
    check_diverged(dirs, states)
    left = [d.name for d, s in zip(dirs, states) if s is None]
    if left:
        raise SystemExit(f"{', '.join(left)} did not finish (see their log.txt); the same command "
                         f"trains them again")
    log.fh.close()
    return fine_tune(args, average_of=dirs)
