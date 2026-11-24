"""Live training curves in Weights & Biases or MLflow (--track), if that package is installed.

--track none (the default) imports neither.  A tracker that cannot start, or fails later, prints a
warning and training goes on; the files in --out remain the record.  wandb reads its usual variables;
without internet, set WANDB_MODE=offline and upload later with  wandb sync <out>/wandb/offline-run-*.
mlflow writes where MLFLOW_TRACKING_URI points (default: mlflow.db in the current folder; a tracking
server is safer for parallel runs).  --track-project is the wandb project or the MLflow experiment.
A resumed train.py run continues its run, whose id is in <out>/tracking.json.
"""
import atexit
import contextlib
import importlib.util
import json
import numbers
import time
import uuid
from pathlib import Path

BACKENDS = ("none", "wandb", "mlflow")


def available(backend):
    return importlib.util.find_spec(backend) is not None


def installed():
    return [b for b in BACKENDS[1:] if available(b)]


def add_arguments(g):
    g.add_argument("--track", choices=BACKENDS, default="none",
                   help=f"live metrics in wandb or mlflow (installed here: {', '.join(installed()) or 'neither'})")
    g.add_argument("--track-project", default="ilmlip", help="wandb project or MLflow experiment")


def flat(d, prefix="", numbers_only=True):
    """A nested dict as {"a/b": value}; numbers_only keeps the finite numbers."""
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flat(v, key + "/", numbers_only))
        elif isinstance(v, numbers.Real) and not isinstance(v, bool):
            if v == v and abs(v) != float("inf"):
                out[key] = float(v)
        elif not numbers_only:
            out[key] = v
    return out


class Tracker:
    """The metrics of one run, logged against the counter step_name (step or epoch)."""

    def __init__(self, backend, out, config, project, name=None, group=None, resume=False,
                 step_name="step", log=print):
        self.backend, self.step_name, self.log, self.run, self.id = backend, step_name, log, None, None
        if backend == "none":
            return
        if not available(backend):
            log(f"[track] {backend} is not installed here: no live tracking (pip install {backend})")
            return
        state = Path(out) / "tracking.json"
        old = None
        if resume:
            with contextlib.suppress(OSError, ValueError, KeyError):
                s = json.loads(state.read_text())
                old = s["id"] if s["backend"] == backend else None
        try:
            start = self._start_wandb if backend == "wandb" else self._start_mlflow
            start(Path(out), config, project, name or Path(out).name, group, old)
        except Exception as exc:
            log(f"[track] {backend} did not start ({type(exc).__name__}: {exc}): no live tracking")
            self.run = None
            return
        with contextlib.suppress(OSError):
            state.write_text(json.dumps(dict(backend=backend, id=self.id)))
        atexit.register(self.close, False)
        log(f"[track] {backend} run {self.id}" + (" (continued)" if old == self.id else ""))

    def _start_wandb(self, out, config, project, name, group, old):
        import wandb
        self.run = wandb.init(project=project, name=name, group=group, dir=out,
                              id=old or uuid.uuid4().hex[:12], resume="allow")
        self.id = self.run.id
        self.run.config.update(config, allow_val_change=True)
        # a resumed run repeats steps from its checkpoint: x = step_name, not wandb's own counter
        self.run.define_metric(self.step_name)
        self.run.define_metric("*", step_metric=self.step_name)

    def _start_mlflow(self, out, config, project, name, group, old):
        from mlflow.entities import Param
        from mlflow.tracking import MlflowClient
        self.run = c = MlflowClient()
        if old:
            with contextlib.suppress(Exception):
                c.update_run(c.get_run(old).info.run_id, status="RUNNING")
                self.id = old
                return
        exp = c.get_experiment_by_name(project)
        if exp is None:
            try:
                eid = c.create_experiment(project)
            except Exception:                    # created by a parallel run meanwhile
                eid = c.get_experiment_by_name(project).experiment_id
        else:
            eid = exp.experiment_id
        self.id = c.create_run(eid, run_name=name, tags={"group": group} if group else None).info.run_id
        params = [Param(k, str(v)[:500]) for k, v in flat(config, numbers_only=False).items()]
        for i in range(0, len(params), 100):
            c.log_batch(self.id, params=params[i:i + 100])

    def __call__(self, metrics, step):
        if self.run is None:
            return
        m = flat(metrics)
        try:
            if self.backend == "wandb":
                self.run.log({**m, self.step_name: step})
            else:
                from mlflow.entities import Metric
                ts = int(time.time() * 1000)
                self.run.log_batch(self.id, metrics=[Metric(k, v, ts, int(step)) for k, v in m.items()])
        except Exception as exc:
            self.log(f"[track] {self.backend} failed ({type(exc).__name__}: {exc}): no live tracking from here on")
            self.close(False)

    def summary(self, metrics):
        """Final numbers of the run (the run table's columns)."""
        if self.run is None:
            return
        m = flat(metrics)
        try:
            if self.backend == "wandb":
                self.run.summary.update(m)
            else:
                from mlflow.entities import Metric
                ts = int(time.time() * 1000)
                self.run.log_batch(self.id, metrics=[Metric(k, v, ts, 0) for k, v in m.items()])
        except Exception as exc:
            self.log(f"[track] {self.backend} failed ({type(exc).__name__}: {exc})")

    def close(self, ok=True):
        run, self.run = self.run, None
        if run is None:
            return
        with contextlib.suppress(Exception):
            if self.backend == "wandb":
                run.finish(exit_code=0 if ok else 1)
            else:
                run.set_terminated(self.id, status="FINISHED" if ok else "FAILED")
