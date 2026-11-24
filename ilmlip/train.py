"""Train an ilmlip model on a packed dataset (ilmlip.data).

    python -m ilmlip.train --packed data/ --hparams S --zbl --epochs 100 --compile --out runs/S

The defaults are the production recipe: Muon with AdamW on the other parameters (peak learning rates
--hparams "lr" and --lr-adamw), MAE loss with weights 1 / 1 / 3 for energy / forces / stress, TF32, the
fused kernels where cuEquivariance runs.

Re-running the same command resumes from out/checkpoint.pt, mid-epoch exactly.
SIGTERM, SIGINT or SIGUSR2 (LSF's run-limit warning) finishes the step and writes a checkpoint.
"""
import argparse
import hashlib
import inspect
import itertools
import json
import math
import signal
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from . import fused_torch, runinfo, tracking
from . import vcut
from .data import PackedLoader, PackedSplit, check_charge_spin
from .model import DTYPES, ILMlip, set_dtype

DELTA = dict(energy=0.05, forces=0.1, stress=0.01)     # Huber thresholds, eV/atom, eV/A, eV/A^3
_STOP = False
OLD_ARCH = ("rcut", "lmax", "lmax_feat", "n_basis", "n_channels", "product_order", "lmax_ind",
            "n_layers", "n_message", "hidden", "embed_dim", "zbl", "zbl_factor",
            "charge_spin", "max_charge", "max_spin")
# the architecture hyper-parameters: every keyword of the model that is not data or a table
ARCH_KEYS = set(inspect.signature(ILMlip.__init__).parameters) - {
    "self", "species", "atomic_energies", "energy_scale", "dtype", "rcut_elem", "kernels"}


def _on_signal(signum, frame):
    global _STOP
    print(f"\n[signal] {signal.Signals(signum).name}: finishing the step, then a checkpoint", flush=True)
    _STOP = True


def lr_at(step, base, warmup, total, floor=0.02, schedule="cosine", decay_frac=0.2):
    """Linear warm-up, then cosine decay to floor * base; wsd: the peak until the last decay_frac
    of the steps, then a linear decay to floor * base."""
    if warmup and step < warmup:
        return base * (step + 1) / warmup
    if total <= warmup:
        return base
    if schedule == "wsd":
        start = max(warmup, round(total * (1 - decay_frac)))
        t = min(1.0, max(0.0, (step - start) / max(1, total - start)))
        return base * (1 - (1 - floor) * t)
    t = min(1.0, (step - warmup) / max(1, total - warmup))
    return base * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * t)))


def orthogonalise(G):
    """U V^T of every matrix in G (..., m, n), by Muon's quintic Newton-Schulz iteration in the
    run's precision (that of G; its products in TF32 under --dtype tf32): singular values near 1."""
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G if G.dtype in (torch.float32, torch.float64) else G.float()
    tall = X.size(-2) > X.size(-1)
    if tall:
        X = X.mT
    shape = X.shape
    X = X.reshape(-1, *shape[-2:])
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(5):
        A = torch.bmm(X, X.mT)
        X = torch.baddbmm(X, torch.baddbmm(A, A, A, beta=b, alpha=c), X, beta=a)
    X = X.reshape(shape)
    return (X.mT if tall else X).to(G.dtype)


class MuonGraph:
    """Muon's step of one parameter group as one CUDA graph (on a CPU the same code runs eagerly).
    Gradients, learning rate and found_inf are copied into fixed buffers before each replay.  Matrices
    are oriented wide (m <= n): the small ones share one zero-padded batch and the large ones are batched
    by shape, so the Newton-Schulz iteration is a few batched products (zero padding leaves U V^T
    unchanged).  The momentum buffers are views of one flat buffer, so the optimiser state is unchanged."""

    SMALL = 128 * 128

    @staticmethod
    def key(group):
        return tuple(group[k] for k in ("weight_decay", "momentum", "rms"))

    def __init__(self, opt, group):
        ps = self.ps = list(group["params"])
        self.hyper = self.key(group)
        dev, dt = ps[0].device, ps[0].dtype
        sizes = [p.numel() for p in ps]
        flat = lambda: torch.zeros(sum(sizes), dtype=dt, device=dev)
        self.views = lambda t: [v.view_as(p) for v, p in zip(t.split(sizes), ps)]
        self.G, self.B, self.O = flat(), flat(), flat()
        self.gv, self.ov = self.views(self.G), self.views(self.O)
        for p, b in zip(ps, self.views(self.B)):
            st = opt.state[p]
            if "momentum_buffer" in st:
                b.copy_(st["momentum_buffer"])
            st["momentum_buffer"] = b
        rms = group["rms"]
        self.scale = torch.cat([torch.full((p.numel(),), rms * max(p.shape[-2:]) ** 0.5, dtype=dt, device=dev)
                                for p in ps])
        self.lr = torch.zeros((), dtype=dt, device=dev)
        self.found = torch.ones((), dtype=dt, device=dev)     # warm-up and capture: a skipped step
        mats = [(math.prod(p.shape[:-2]), *sorted(p.shape[-2:]), p.size(-2) > p.size(-1)) for p in ps]
        small = [i for i, (_, m, n, _) in enumerate(mats) if m * n <= self.SMALL]
        by_shape = {}
        for i, (_, m, n, _) in enumerate(mats):
            if m * n > self.SMALL:
                by_shape.setdefault((m, n), []).append(i)
        self.buckets = []
        for idx in ([small] if small else []) + list(by_shape.values()):
            rows, items = 0, []
            for i in idx:
                items.append((i, slice(rows, rows + mats[i][0]), *mats[i][1:]))
                rows += mats[i][0]
            buf = torch.zeros(rows, max(mats[i][1] for i in idx), max(mats[i][2] for i in idx),
                              dtype=dt, device=dev)
            self.buckets.append((buf, items))
        self.graph = None
        if dev.type == "cuda":
            side = torch.cuda.Stream(dev)
            side.wait_stream(torch.cuda.current_stream(dev))
            with torch.cuda.stream(side):
                for _ in range(2):
                    self.body()
            torch.cuda.current_stream(dev).wait_stream(side)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, capture_error_mode="thread_local"):
                self.body()
            self.graph = graph

    def body(self):
        wd, momentum, _ = self.hyper
        skip = self.found > 0
        ok = (~skip).to(self.G.dtype)
        self.G.masked_fill_(skip, 0.0)
        self.B.lerp_(self.G, ok * (1 - momentum))
        uv = self.views(self.G.lerp(self.B, momentum))
        for buf, items in self.buckets:
            for i, rows, m, n, tall in items:
                u = uv[i].reshape(-1, *uv[i].shape[-2:])
                buf[rows, :m, :n].copy_(u.mT if tall else u)
            X = orthogonalise(buf)
            for i, rows, m, n, tall in items:
                o = X[rows, :m, :n]
                self.ov[i].view(-1, *self.ov[i].shape[-2:]).copy_(o.mT if tall else o)
        lr = self.lr * ok
        self.O.mul_(self.scale).mul_(lr)
        if wd:
            torch._foreach_mul_(self.ps, 1 - lr * wd)
        torch._foreach_sub_(self.ps, self.ov)

    def step(self, lr, found_inf):
        torch._foreach_copy_(self.gv, [p.grad for p in self.ps])
        self.lr.fill_(lr)
        if found_inf is None:
            self.found.zero_()
        else:
            self.found.copy_(found_inf)
        if self.graph is None:
            self.body()
        else:
            self.graph.replay()


class Muon(torch.optim.Optimizer):
    """Muon (Jordan et al. 2024): Nesterov momentum, orthogonalised per weight matrix (the last two
    dimensions of a tensor; leading ones batch matrices) and scaled by rms sqrt(max(m, n)), an update
    of RMS rms * lr.  rms 0.2 is the update size of plain AdamW (Liu et al. 2025), so both can share
    the learning rate and decoupled weight decay.  With found_inf set (a device tensor, as for fused
    AdamW) a step with non-finite gradients changes nothing.  graph: a group whose tensors all have
    gradients steps as one MuonGraph (else per tensor).  names: the tensors'
    names (of one group), saved with the state, which then loads by name."""

    def __init__(self, params, lr, weight_decay=0.0, momentum=0.95, rms=0.2, graph=False, names=None):
        super().__init__(params, dict(lr=lr, weight_decay=weight_decay, momentum=momentum, rms=rms))
        self.found_inf, self.graph, self._graphs = None, graph, {}
        self.names = list(names) if names else None

    def state_dict(self):
        sd = super().state_dict()
        if self.names:
            sd["names"] = list(self.names)
        return sd

    def load_state_dict(self, state_dict):
        super().load_state_dict(self._in_my_order(state_dict))
        self._graphs = {}                         # the momentum buffers are new tensors

    def _in_my_order(self, sd):
        """sd indexed in this optimiser's tensor order, by its names (the order differs from
        process to process)."""
        if not self.names or "names" not in sd or len(self.param_groups) != 1 or len(sd["param_groups"]) != 1:
            return sd
        saved, state = sd["param_groups"][0]["params"], sd["state"]
        where = dict(zip(sd["names"], saved))
        order = [where.get(n) for n in self.names]
        group = dict(sd["param_groups"][0], params=list(range(len(order))))
        return dict(sd, param_groups=[group], state={k: state[i] for k, i in enumerate(order) if i in state})

    @torch.no_grad()
    def step(self):
        ok = None if self.found_inf is None else self.found_inf == 0
        for k, g in enumerate(self.param_groups):
            if self.graph and all(p.grad is not None for p in g["params"]):
                mg = self._graphs.get(k)
                if mg is None or mg.hyper != MuonGraph.key(g):
                    try:
                        mg = self._graphs[k] = MuonGraph(self, g)
                    except RuntimeError as exc:   # fall back to eager if the capture fails
                        warnings.warn(f"Muon: no CUDA graph ({exc}); eager from now on")
                        self.graph = False
                if self.graph:
                    mg.step(g["lr"], self.found_inf)
                    continue
            for p in g["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                if "momentum_buffer" not in st:
                    st["momentum_buffer"] = torch.zeros_like(p)
                buf = st["momentum_buffer"]
                new_buf = buf.lerp(p.grad, 1 - g["momentum"])
                o = orthogonalise(p.grad.lerp(new_buf, g["momentum"]))
                new_p = p * (1 - g["lr"] * g["weight_decay"]) - (g["lr"] * g["rms"] * max(p.shape[-2:]) ** 0.5) * o
                if ok is None:
                    buf.copy_(new_buf)
                    p.copy_(new_p)
                else:
                    buf.copy_(torch.where(ok, new_buf, buf))
                    p.copy_(torch.where(ok, new_p, p))


class MuonAdamW:
    """Muon on the weight matrices and AdamW on everything else, stepped as one optimiser."""

    def __init__(self, muon, adamw):
        self.muon, self.adamw = muon, adamw

    @property
    def param_groups(self):
        return self.adamw.param_groups + self.muon.param_groups     # AdamW first: its "fused" is the trainer's

    @property
    def found_inf(self):
        return self.muon.found_inf

    @found_inf.setter
    def found_inf(self, value):
        self.adamw.found_inf = self.muon.found_inf = value

    def step(self):
        self.adamw.step()
        self.muon.step()

    def zero_grad(self, set_to_none=True):
        self.adamw.zero_grad(set_to_none=set_to_none)
        self.muon.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return {"adamw": self.adamw.state_dict(), "muon": self.muon.state_dict()}

    def load_state_dict(self, state):
        self.adamw.load_state_dict(state["adamw"])
        self.muon.load_state_dict(state["muon"])


NOT_MUON = ("embed", "elem_bias", "head", "atomic_energies", "charge_", "spin_", "shift")


def muon_matrices(model):
    """Names of the weight matrices Muon updates: every parameter of two or more dimensions whose last
    two are the (in, out) or (out, in) of a channel map (leading ones batch matrices), except the
    per-element and per-state tables (embeddings, element biases, heads, reference energies, charge
    and spin)."""
    return [n for n, p in model.named_parameters()
            if p.dim() >= 2 and not any(k in n for k in NOT_MUON)]


def make_optimizer(model, shift, h, kind, momentum, rms, device, log, e0_decay=True, muon_graph=None):
    names = muon_matrices(model) if kind == "muon" else []
    groups = [dict(params=[p for n, p in model.named_parameters()
                           if n not in names and (e0_decay or n != "atomic_energies")])]
    if not e0_decay:
        groups.append(dict(params=[model.atomic_energies], weight_decay=0.0))
    if shift is not None:
        groups.append(dict(params=list(shift.parameters()), weight_decay=0.0))
    # fused on the GPU: one kernel, and non-finite steps are skipped on the device
    adamw = torch.optim.AdamW(groups, lr=h["lr"], amsgrad=True, weight_decay=h["weight_decay"],
                              **({"fused": True} if device.type == "cuda" else {}))
    if not names:
        return adamw
    params = dict(model.named_parameters())
    graph = device.type == "cuda" if muon_graph is None else muon_graph
    log(f"[optim] Muon on {sum(params[n].numel() for n in names):,} parameters in {len(names)} tensors"
        + (" (one CUDA graph)" if graph else "") + ": "
        + ", ".join(f"{n} {tuple(params[n].shape)}" for n in sorted(names)))
    muon = Muon([params[n] for n in names], lr=h["lr"],
                weight_decay=h["weight_decay"], momentum=momentum, rms=rms, names=names, graph=graph)
    return MuonAdamW(muon, adamw)


def penalty(x, kind, delta):
    if kind == "huber":
        return torch.nn.functional.huber_loss(x, torch.zeros_like(x), delta=delta)
    if kind == "mae":
        return x.abs().mean()
    return (x ** 2).mean()


def force_penalty(df, kind, delta):
    """mae: the length of each atom's force error, averaged over the atoms."""
    if kind == "mae":
        return torch.sqrt((df * df).sum(-1) + 1e-12).mean()
    return penalty(df, kind, delta)


def penalty_masked(x, mask, kind, delta):
    """Penalty over the rows where mask holds, without the host-device sync of mask.any()."""
    x = torch.where(mask[:, None], x, torch.zeros_like(x))
    if kind == "huber":
        v = torch.nn.functional.huber_loss(x, torch.zeros_like(x), delta=delta, reduction="sum")
    elif kind == "mae":
        v = x.abs().sum()
    else:
        v = (x ** 2).sum()
    return v / (mask.sum() * x.shape[1]).clamp(min=1)


def stress_mask(b):
    return torch.isfinite(b["stress"]).all(dim=1) & (b["volume"] > 1e-8)


def fingerprint(batches):
    """Eight hex digits of the validation energies (at the model's precision) and positions;
    equal = same structures."""
    h = hashlib.sha1()
    for b in batches:
        for k in ("energy", "positions"):
            x = b[k].detach().to(b["positions"].dtype)
            h.update(np.ascontiguousarray(x.cpu().numpy(), np.float64).tobytes())
    return h.hexdigest()[:8]


def parse_weights(spec):
    """'mptrj=4,sAlex_train=2' -> {'mptrj': 4.0, 'sAlex_train': 2.0}."""
    return {k.strip(): float(v) for k, v in (x.split("=") for x in (spec or "").split(",") if x.strip())}


def history_until(path, step):
    """Rewrite history.jsonl without the lines after step (logged after the checkpoint a run
    resumes from) and unreadable ones; the rows kept."""
    if not path.exists():
        return []
    rows, keep = [], []
    for line in path.read_text().splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if row["step"] <= step:
            rows.append(row)
            keep.append(line + "\n")
    tmp = path.with_suffix(".jsonl.tmp")
    tmp.write_text("".join(keep))
    tmp.replace(path)
    return rows


class CompiledStep:
    """Energy and forces traced into one graph (the position derivative included, so the parameter
    gradient is a single backward) and compiled.  Batches are padded to fixed shapes: atoms to
    batch_atoms, edges to a multiple of edge_bucket, structures to batch_structs + 1; padding atoms form
    an extra structure that is dropped, padding edges lie beyond rcut."""

    KEYS = ("positions", "mu", "ind_i", "ind_j", "shift", "batch", "volume")

    def __init__(self, model, batch_atoms, batch_structs, edge_bucket=8192, with_stress=False,
                 mode=None, log=print):
        self.model, self.log = model, log
        self.mode = None if mode in (None, "default") else mode
        self.with_stress, self.batch_atoms = bool(with_stress), int(batch_atoms)
        self.n_struct_pad, self.edge_bucket = int(batch_structs) + 1, int(edge_bucket)
        self.cache = {}
        self.keys = self.KEYS + (("charge", "spin") if model.charge_spin else ())
        self.names = list(dict(model.named_parameters()))
        self._params = tuple(dict(model.named_parameters())[n] for n in self.names)
        import torch._functorch.config as fc
        fc.donated_buffer = False                 # else create_graph=True is refused
        if model.kernels != "torch":
            fused_torch.engine(model)             # built here, not inside the trace

    def _pad(self, b):
        dev, dt = b["positions"].device, b["positions"].dtype
        na, ne, ns = b["n_atoms"], len(b["ind_i"]), b["n_struct"]
        A = max(self.batch_atoms, na)
        E = (ne // self.edge_bucket + 1) * self.edge_bucket
        S = max(self.n_struct_pad, ns + 1)
        pa, pe, far = A - na, E - ne, 2.0 * self.model.rcut
        lg = dict(dtype=torch.long, device=dev)
        d = na if pa > 0 else 0                   # the first padding atom
        ts = (torch.cat([b["positions"], torch.zeros(pa, 3, dtype=dt, device=dev)]),
              torch.cat([b["mu"], torch.zeros(pa, **lg)]),
              torch.cat([b["ind_i"], torch.full((pe,), d, **lg)]),
              torch.cat([b["ind_j"], torch.full((pe,), d, **lg)]),
              torch.cat([b["shift"], torch.full((pe, 3), far / 3 ** 0.5, dtype=dt, device=dev)]),
              torch.cat([b["batch"], torch.full((pa,), S - 1, **lg)]),
              torch.cat([b["volume"], torch.ones(S - ns, dtype=dt, device=dev)]))
        if self.model.charge_spin:                # padding slots: neutral singlets
            ts += (torch.cat([b["charge"].long(), torch.zeros(S - ns, **lg)]),
                   torch.cat([b["spin"].long(), torch.ones(S - ns, **lg)]))
        return ts, (A, E, S), ns

    def _mark(self):
        if self.mode == "reduce-overhead":
            torch.compiler.cudagraph_mark_step_begin()

    def _build(self, shape, tensors, params, new_step):
        from torch.fx.experimental.proxy_tensor import make_fx
        from torch.func import functional_call
        from .model import strained
        A, E, S = shape
        model, names = self.model, self.names

        def energy_forces(pos, *rest):
            ts, ps = rest[:len(self.keys) - 1], rest[len(self.keys) - 1:]
            pos = pos.detach().requires_grad_(True)
            bb = dict(zip(self.keys[1:], ts), positions=pos, n_atoms=A, n_struct=S)
            if not self.with_stress:
                e = functional_call(model, dict(zip(names, ps)), (bb,), {"compute_forces": False})["energy"]
                return e, -torch.autograd.grad(e.sum(), pos, create_graph=True)[0]
            strain = torch.zeros(S, 3, 3, dtype=pos.dtype, device=pos.device, requires_grad=True)
            sy = 0.5 * (strain + strain.transpose(1, 2))
            at_struct = bb["batch"]
            bb["positions"] = strained(pos, sy.index_select(0, at_struct))
            bb["shift"] = strained(bb["shift"], sy.index_select(0, at_struct.index_select(0, bb["ind_i"])))
            e = functional_call(model, dict(zip(names, ps)), (bb,), {"compute_forces": False})["energy"]
            gp, gs = torch.autograd.grad(e.sum(), [pos, strain], create_graph=True)
            sig = gs / bb["volume"].clamp(min=1e-12)[:, None, None]
            sig = 0.5 * (sig + sig.transpose(1, 2))
            return e, -gp, torch.stack([sig[:, 0, 0], sig[:, 1, 1], sig[:, 2, 2],
                                        sig[:, 1, 2], sig[:, 0, 2], sig[:, 0, 1]], dim=1)

        t0 = time.time()
        gm = make_fx(energy_forces, tracing_mode="real")(*tensors, *params)
        cg = torch.compile(gm, dynamic=False, **({"mode": self.mode} if self.mode else {}))
        if new_step:
            self._mark()
        cg(*tensors, *params)
        if tensors[0].is_cuda:
            torch.cuda.synchronize(tensors[0].device)
        self.log(f"[compile] (atoms={A}, edges={E}, structs={S}) in {time.time() - t0:.1f} s, "
                 f"{len(self.cache) + 1} cached")
        return cg

    def __call__(self, b, weights=None, new_step=True):
        """(energies, forces, stresses or None) of batch b.  weights replace parameters by name,
        as in functional_call; new_step=False for another call before the same backward (the
        outputs of a CUDA-graph step stay valid until the next step begins)."""
        tensors, shape, ns = self._pad(b)
        ps = self._params if not weights else tuple(weights.get(n, p) for n, p in zip(self.names, self._params))
        if shape not in self.cache:
            self.cache[shape] = self._build(shape, tensors, ps, new_step)
        if new_step:
            self._mark()
        res = self.cache[shape](*tensors, *ps)
        e, f = res[0][:ns], res[1][:b["n_atoms"]]
        return e, f, (res[2][:ns] if self.with_stress else None)


class EMA:
    """Moving average of the weights in float64 (float32 steps round away against the reference
    energies), stored with every checkpoint; the decay after optimiser step n is
    min(decay, (1 + n) / (10 + n))."""

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.detach().to(torch.float64, copy=True) for k, v in model.state_dict().items()
                       if v.dtype.is_floating_point}
        self._pairs = None

    @torch.no_grad()
    def update(self, model, n):
        if self._pairs is None or self._pairs[0] is not self.shadow:
            sd = model.state_dict()           # views of the live weights
            self._pairs = (self.shadow, [(self.shadow[k], sd[k]) for k in self.shadow if k in sd])
        pairs = self._pairs[1]
        torch._foreach_lerp_([a for a, _ in pairs], [b.to(torch.float64) for _, b in pairs],
                             1 - min(self.decay, (1 + n) / (10 + n)))


def ema_weights(ck, model, log=print):
    """The EMA weights of a checkpoint with the reference energies of its weights (a float32
    average of them cannot move), in `model`'s layout."""
    ema, raw = ck["ema"], ck["model"]["atomic_energies"]
    gap = (ema["atomic_energies"].double() - raw.double()).abs()
    if gap.max() > 1e-6:
        k = int(gap.argmax())
        log(f"[ema] reference energies from the weights, not the average (they differ by up to "
            f"{float(gap[k]):.3g} eV, {ck['species'][k]})")
    ema["atomic_energies"] = raw
    # buffers do not train: the model's own, as the average of an infinite entry is NaN
    trained = {n for n, _ in model.named_parameters()}
    ema = {k: v for k, v in ema.items() if k in trained}
    return ema


class SourceShift(nn.Module):
    """Learned per-element energy offsets for structures of non-reference sources.

    Constants per atom (forces and stress untouched), kept outside the model so that it stays on the
    reference sources' energy scale."""

    def __init__(self, n_species, is_ref, init=None, dtype=torch.float32):
        super().__init__()
        self.shift = nn.Parameter(torch.zeros(n_species, dtype=dtype) if init is None
                                  else torch.as_tensor(np.asarray(init), dtype=dtype))
        other = torch.ones(256, dtype=torch.bool)
        other[:len(is_ref)] = torch.tensor([not r for r in is_ref], dtype=torch.bool)
        self.register_buffer("other", other)

    def forward(self, b):
        per_atom = self.shift[b["mu"]].double().masked_fill(~self.other[b["src"]][b["batch"]], 0.0)
        return per_atom.new_zeros(b["n_struct"]).index_add(0, b["batch"], per_atom)

    def is_ref(self, b):
        return ~self.other[b["src"]]


def validate(model, batches, with_stress=False, shift=None, src_names=None):
    """Energy (meV/atom), force (meV/A) and stress (meV/A^3) errors; per source if
    src_names; e_mae_ref / e_mae_other with a SourceShift."""
    model.eval()
    eae = ese = fae = fse = sse = sae = 0.0
    n_at = n_st = n_sg = 0
    grp, per = {True: [0.0, 0], False: [0.0, 0]}, {}
    for b in batches:
        with torch.enable_grad():
            out = model(b, compute_stress=with_stress)
        e_pred = out["energy"].detach()
        if shift is not None:
            with torch.no_grad():
                e_pred = e_pred + shift(b)
        de = (e_pred - b["energy"]) / b["n_atoms_per_struct"]
        df = out["forces"] - b["forces"]
        eae += de.abs().sum().item()
        ese += (de ** 2).sum().item()
        fae += df.abs().sum().item()
        fse += (df ** 2).sum().item()
        if shift is not None:
            ref = shift.is_ref(b)
            for flag in (True, False):
                m = ref == flag
                grp[flag][0] += de[m].abs().sum().item()
                grp[flag][1] += int(m.sum())
        if src_names:
            src, ns = b["src"], len(src_names)
            e_s = torch.zeros(ns, dtype=de.dtype, device=de.device).index_add_(0, src, de.abs())
            n_s = torch.bincount(src, minlength=ns)
            fa = df.abs().sum(1)
            f_s = torch.zeros(ns, dtype=fa.dtype, device=fa.device).index_add_(0, src[b["batch"]], fa)
            a_s = torch.bincount(src[b["batch"]], minlength=ns) * 3
            for k, (e1, n1, f1, a1) in enumerate(zip(e_s.tolist(), n_s.tolist(), f_s.tolist(), a_s.tolist())):
                if n1:
                    per[k] = [x + y for x, y in zip(per.get(k, [0.0, 0, 0.0, 0]), (e1, n1, f1, a1))]
        if with_stress:
            m = stress_mask(b)
            if m.any():
                ds = out["stress"][m] - b["stress"][m]
                sse += (ds ** 2).sum().item()
                sae += ds.abs().sum().item()
                n_sg += int(m.sum()) * 6
        n_st += b["n_struct"]
        n_at += b["forces"].numel()
    model.train()
    if n_st == 0:
        return {}
    r = dict(e_mae=eae / n_st * 1000, e_rmse=(ese / n_st) ** 0.5 * 1000,
             f_mae=fae / n_at * 1000, f_rmse=(fse / n_at) ** 0.5 * 1000)
    if n_sg:
        r["s_rmse"], r["s_mae"] = (sse / n_sg) ** 0.5 * 1000, sae / n_sg * 1000
    for flag, key in ((True, "e_mae_ref"), (False, "e_mae_other")):
        if grp[flag][1]:
            r[key] = grp[flag][0] / grp[flag][1] * 1000
    if len(per) > 1:
        for k, (e1, n1, f1, a1) in sorted(per.items()):
            name = src_names[k] if k < len(src_names) else str(k)
            r[f"e_mae/{name}"], r[f"f_mae/{name}"] = e1 / n1 * 1000, f1 / max(a1, 1) * 1000
    return r


@torch.no_grad()
def gate_stats(model, batches):
    """Mean and minimum per layer of the factors model.gate_factors(batch) gives per atom."""
    tot, n, low = {}, {}, {}
    for b in batches:
        for k, g in enumerate(model.gate_factors(b)):
            tot[k] = tot.get(k, 0.0) + g.double().sum().item()
            n[k] = n.get(k, 0) + g.numel()
            low[k] = min(low.get(k, math.inf), g.min().item())
    return {f"gate_{q}/{k}": v for k in tot for q, v in (("mean", tot[k] / n[k]), ("min", low[k]))}


@torch.no_grad()
def feat_stats(model, batches):
    """Largest |x~| and the fraction of |x~| above tau / 2 over the atoms (x~: whitened features, before the squash)."""
    top, hot, n = 0.0, 0, 0
    for b in batches:
        x = model.whitened(b).abs()
        top, hot, n = max(top, x.max().item()), hot + (x > model.feat_bound / 2).sum().item(), n + x.numel()
    return {"feat_max": top, "feat_sat": hot / max(n, 1)}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_argument_group("data")
    g.add_argument("--packed", required=True, help="dataset directory (train/, val/, meta.json)")
    g.add_argument("--resident", choices=["auto", "device", "host"], default="auto",
                   help="per-atom data on the GPU or memory-mapped on the host "
                        "(auto: GPU if < half its free memory)")
    g.add_argument("--charge-spin", action="store_true", help="condition on total charge and spin (OMol25)")
    g.add_argument("--data-seed", type=int, default=None,
                   help="seed of the validation sample (default --seed); committee members share it")
    g.add_argument("--sources", default=None, help="train and validate on these sources only")
    g.add_argument("--source-weights", default=None, help="per-epoch weights, e.g. 'mptrj=4,sAlex_train=2'")
    g.add_argument("--val-per-source", action="store_true",
                   help="draw the validation sample evenly from every group")
    g.add_argument("--ref-sources", default=None,
                   help="sources that set the energy scale; the others get learned per-element offsets")
    g.add_argument("--freeze-e0", action="store_true", help="keep the reference energies fixed")
    g.add_argument("--scale-from", choices=["energy", "forces"], default="energy",
                   help="energy scale at step 0: std of E/atom, or the force RMS (for total energies)")
    g.add_argument("--score-refs", default=None, help="'E,F': also report score = e_mae/E + f_mae/F")
    g = p.add_argument_group("model")
    g.add_argument("--hparams", required=True, help="a configuration of ilmlip/configs (S, M, XL, "
                   "XXL, ...) or a JSON file of architecture and peak learning rate")
    g.add_argument("--lr", type=float, default=None, help="peak learning rate (default: --hparams)")
    g.add_argument("--lr-adamw", type=float, default=None,
                   help="with --optimizer muon: peak lr of the AdamW part (default: 3e-3)")
    g.add_argument("--zbl", action="store_true", help="add ZBL screened nuclear repulsion")
    g = p.add_argument_group("optimisation")
    g.add_argument("--steps", type=int, default=1_000_000)
    g.add_argument("--epochs", type=int, default=0, help="train N epochs instead of --steps")
    g.add_argument("--batch-atoms", type=int, default=8192)
    g.add_argument("--batch-structs", type=int, default=512)
    g.add_argument("--loss", choices=["mse", "huber", "mae"], default="mae",
                   help="mae: energy MAE per atom, mean length of the force errors, stress MAE")
    for q in ("energy", "forces", "stress"):
        g.add_argument(f"--loss-{q}", choices=["mse", "huber", "mae"], default=None,
                       help=f"loss of the {q} term (default: --loss)")
    g.add_argument("--w-energy", type=float, default=1.0)
    g.add_argument("--w-forces", type=float, default=1.0)
    g.add_argument("--w-stress", type=float, default=3.0)
    g.add_argument("--select-on", default="f_rmse", help="validation metric that picks best.pt")
    g.add_argument("--init", default=None,
                   help="start from this checkpoint (its EMA weights), fresh optimiser")
    g.add_argument("--refit-e0", type=int, default=0,
                   help="with --init: refit the reference energies on this many batches first")
    g.add_argument("--fold-shift", action="store_true",
                   help="with --init and --sources: add the checkpoint's per-element offsets "
                        "(--ref-sources) to its reference energies, i.e. start on the energy scale "
                        "of its other sources")
    g.add_argument("--optimizer", choices=["adamw", "muon"], default="muon",
                   help="muon: Muon on the weight matrices, AdamW on the rest")
    g.add_argument("--no-e0-decay", action="store_true",
                   help="no weight decay on the reference energies: decoupled decay removes lr x decay x |E0| "
                        "per step, more than Adam can restore where E0 is large (OMol's absolute energies)")
    g.add_argument("--muon-momentum", type=float, default=0.95)
    g.add_argument("--muon-rms", type=float, default=0.2,
                   help="RMS of a Muon update per unit learning rate (0.2: that of plain AdamW)")
    g.add_argument("--muon-graph", action=argparse.BooleanOptionalAction,
                   help="the Muon step as one CUDA graph (default: on a GPU)")
    g.add_argument("--weight-decay", type=float, default=1e-3)
    g.add_argument("--grad-clip", type=float, default=5.0, help="gradient norm clip")
    g.add_argument("--schedule", choices=["cosine", "wsd"], default="cosine",
                   help="wsd: warm-up, the peak, then a linear decay over the last --decay-frac of the steps")
    g.add_argument("--decay-frac", type=float, default=0.2)
    g.add_argument("--warmup", type=int, default=2000)
    g.add_argument("--ema", type=float, default=0.999, help="0 disables the moving average")
    g.add_argument("--compile", action="store_true", help="compile energy + forces (CUDA)")
    g.add_argument("--compile-mode", choices=["default", "reduce-overhead"], default="default",
                   help="reduce-overhead: replay the step as CUDA graphs")
    g.add_argument("--edge-bucket", type=int, default=8192)
    g.add_argument("--kernels", choices=("auto", *fused_torch.KERNELS), default="auto",
                   help="auto: fused on a CUDA device with cuequivariance-torch, else torch. fused: the bond sums "
                        "and products as cuEquivariance kernels (CUDA, needs cuequivariance-ops-torch-cu12); same "
                        "weights and checkpoints; fused-reference: the same in plain torch")
    g = p.add_argument_group("runtime")
    g.add_argument("--out", required=True, help="run directory, also the resume point")
    g.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    g.add_argument("--dtype", choices=DTYPES, default="tf32",
                   help="tf32: float32 with TF32 matrix products (they only act on NVIDIA GPUs; positions, "
                        "cell and strain stay float32); float32: no TF32; float64: double precision")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--log-every", type=int, default=50)
    g.add_argument("--val-every", type=int, default=2000)
    g.add_argument("--val-batches", type=int, default=50)
    g.add_argument("--ckpt-every", type=int, default=2000)
    g.add_argument("--stop-after", type=int, default=0,
                   help="stop at this step as on SIGTERM, schedule unchanged (resume continues)")
    tracking.add_arguments(g)
    args = p.parse_args(argv)
    if args.data_seed is None:
        args.data_seed = args.seed
    if args.lr_adamw is None and args.optimizer == "muon":
        args.lr_adamw = 3e-3
    return args


def main(argv=None):
    runinfo.called("ilmlip.train", argv)
    global _STOP
    _STOP = False
    args = parse_args(argv)
    signals = [signal.SIGINT, signal.SIGTERM]
    usr2 = getattr(signal, "SIGUSR2", None)          # LSF's run-limit warning; not on Windows
    # inherited as ignored: a launcher (pretraining/train.sh) relays it as SIGTERM
    if usr2 is not None and signal.getsignal(usr2) is not signal.SIG_IGN:
        signals.append(usr2)
    previous = {s: signal.signal(s, _on_signal) for s in signals}
    try:
        _train(args)
    finally:
        for s, handler in previous.items():
            signal.signal(s, signal.SIG_DFL if handler is None else handler)


def _train(args):
    global _STOP
    score_refs = [float(x) for x in args.score_refs.split(",")] if args.score_refs else None
    log = lambda *a: print(*a, flush=True)
    runinfo.show(log)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dtype = set_dtype(args.dtype)
    device = torch.device(args.device)
    args.kernels = fused_torch.choose(args.kernels, device, log)
    try:
        fused_torch.require(args.kernels, device)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from None

    from ase.data import chemical_symbols
    species = [chemical_symbols[z] for z in range(1, 101)]      # mu = Z - 1
    hp = Path(args.hparams)
    if not hp.exists():                       # a name such as S: the configuration shipped in configs/
        hp = Path(__file__).parent / "configs" / f"{args.hparams}.json"
    h = json.loads(hp.read_text())
    h = dict(h.get("hparams", h))
    h["hidden"] = tuple(h["hidden"])
    if args.lr is not None:
        h["lr"] = args.lr
    h["weight_decay"], h["grad_clip"] = args.weight_decay, args.grad_clip
    h["zbl"] = h.get("zbl", False) or args.zbl
    h.setdefault("zbl_factor", 0.65)
    if args.charge_spin:
        h.update(charge_spin=True, max_charge=h.get("max_charge", 10), max_spin=h.get("max_spin", 11))
    ckpt_path = out / "checkpoint.pt"
    init_ck = (torch.load(args.init, map_location="cpu", weights_only=True)
               if args.init and not ckpt_path.exists() else None)
    # the architecture is the checkpoint's: this run's own when resuming, else --init's
    arch_ck = torch.load(ckpt_path, map_location="cpu", weights_only=True) if ckpt_path.exists() else init_ck
    if arch_ck is not None:
        arch = arch_ck.get("hparams") or {}
        for k in OLD_ARCH:
            if k in arch:
                h[k] = tuple(arch[k]) if k == "hidden" else arch[k]
        for k in ARCH_KEYS - set(OLD_ARCH):           # later options: absent from the checkpoint = default
            if k in arch:
                h[k] = arch[k]
            else:
                h.pop(k, None)
        h["zbl"] = args.zbl or bool(arch.get("zbl", False))
        vcut.arch(h, arch)
    log("[hparams] " + json.dumps({k: (list(v) if isinstance(v, tuple) else v) for k, v in h.items()}))

    root = Path(args.packed)
    tr_split, va_split = PackedSplit(root / "train"), PackedSplit(root / "val")
    meta, src_names = tr_split.meta, tr_split.sources
    st = (meta.get("stats") or {}).get("train") or {}
    e0 = np.zeros(len(species))
    for z, v in (meta.get("e0") or {}).items():
        e0[int(z) - 1] = v
    log(f"[data] {root}: {len(tr_split):,} training structures, {int(tr_split.n_atoms.sum()):,} atoms; "
        f"{len(va_split):,} validation structures" + (f"; sources {src_names}" if src_names else ""))
    use_stress = args.w_stress > 0 and tr_split.any_stress
    if args.w_stress > 0 and not use_stress:
        log(f"[train] --w-stress {args.w_stress:g} ignored: the training data has no stress labels")
    # the checkpoints give stresses if this run or the one it starts from trained on them
    h["stress"] = use_stress or h.get("stress", init_ck is not None)
    if not h["stress"]:
        log("[train] no stress term: the checkpoints refuse stress (hparams stress=False)")

    ref_sources = [x for x in (args.ref_sources or "").split(",") if x]
    if args.fold_shift and (ref_sources or not args.init or not args.sources):
        raise SystemExit("--fold-shift needs --init and --sources, and no --ref-sources")
    shift_init = None
    if ref_sources and init_ck is None and not ckpt_path.exists():
        grp = meta.get("e0_groups") or {}
        if sorted(grp.get("ref_sources", [])) != sorted(ref_sources):
            raise SystemExit(f"--ref-sources {ref_sources}: meta.json has E0 fits "
                             f"for {grp.get('ref_sources')}")
        e0_ref, e0_other = np.zeros(len(species)), np.zeros(len(species))
        for arr, key in ((e0_ref, "ref"), (e0_other, "other")):
            for z, v in grp.get(key, {}).items():
                arr[int(z) - 1] = v
        both = (e0_ref != 0) & (e0_other != 0)
        e0 = np.where(e0_ref != 0, e0_ref, e0)
        shift_init = np.where(both, e0_other - e0_ref, 0.0)

    model = ILMlip(species, **{k: h[k] for k in ARCH_KEYS if k in h}, rcut_elem=h.get("rcut_elem"),
                        atomic_energies=e0, kernels=args.kernels, dtype=dtype).to(device)
    shift = None
    if ref_sources:
        unknown = set(ref_sources) - set(src_names)
        if unknown:
            raise SystemExit(f"--ref-sources: unknown {sorted(unknown)}; have {src_names}")
        shift = SourceShift(len(species), [n in ref_sources for n in src_names], shift_init,
                            dtype=dtype).to(device)
    log(f"[model] {model.n_parameters():,} parameters, {model.n_features} features, "
        f"reach {h['n_layers'] * h['rcut']:.1f} A")
    if model.rcut_elem:
        log(vcut.describe(model))

    def pick(split):
        keep = np.ones(len(split), bool)
        if args.sources:
            want = args.sources.split(",")
            if set(want) - set(src_names):
                raise SystemExit(f"--sources: unknown {sorted(set(want) - set(src_names))}; have {src_names}")
            keep &= np.isin(split.src, [src_names.index(s) for s in want])
        return np.where(keep)[0]

    weights = None
    if args.source_weights:
        sw = parse_weights(args.source_weights)
        if set(sw) - set(src_names):
            raise SystemExit(f"--source-weights: unknown {sorted(set(sw) - set(src_names))}; "
                             f"have {src_names}")
        weights = np.array([sw.get(n, 1.0) for n in src_names])[tr_split.src]
    kw = dict(device=device, dtype=dtype, batch_atoms=args.batch_atoms, batch_structs=args.batch_structs,
              seed=args.seed, resident=args.resident, log=log)
    kw["pair_rcut"] = vcut.by_z(model)
    train_loader = PackedLoader(tr_split, h["rcut"], shuffle=True, indices=pick(tr_split),
                                weights=weights, **kw)
    n_val = args.val_batches * args.batch_structs
    val_loader = PackedLoader(va_split, h["rcut"], shuffle=False, indices=pick(va_split), **kw).fixed(
        n_val, seed=args.data_seed, stratify=args.val_per_source)
    val_ref_loader = None
    if ref_sources and not args.sources:
        ids = [src_names.index(s) for s in ref_sources]
        val_ref_loader = PackedLoader(va_split, h["rcut"], shuffle=False,
                                      indices=np.where(np.isin(va_split.src, ids))[0], **kw).fixed(
            n_val, seed=args.data_seed, stratify=args.val_per_source)
    log(f"[val] fixed set: {sum(b['n_struct'] for b in val_loader):,} structures"
        + (f"; reference sources alone: {sum(b['n_struct'] for b in val_ref_loader):,}"
           if val_ref_loader else "")
        + f" (fingerprint {fingerprint(val_loader)})")
    if model.charge_spin:
        for split, idx in ((tr_split, train_loader.indices), (va_split, pick(va_split))):
            check_charge_spin(split, idx, model.max_charge, model.max_spin, log)

    if args.freeze_e0:
        model.atomic_energies.requires_grad_(False)
    opt = make_optimizer(model, shift, h, args.optimizer, args.muon_momentum, args.muon_rms, device, log,
                         e0_decay=not args.no_e0_decay, muon_graph=args.muon_graph)
    if args.lr_adamw is not None and args.optimizer != "muon":
        raise SystemExit("--lr-adamw sets the AdamW part of --optimizer muon")
    adamw_scale = args.lr_adamw / h["lr"] if args.lr_adamw is not None else 1.0
    if adamw_scale != 1.0:
        log(f"[optim] AdamW part at peak lr {args.lr_adamw:.3g} (Muon {h['lr']:.3g})")
    ema = EMA(model, args.ema) if args.ema else None
    step, best, epoch, steps_per_epoch = 0, float("inf"), 0, None
    batch_in_epoch = resume_batches = 0
    hist_path, seconds = out / "history.jsonl", 0.0

    if ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(ck["model"])
        if isinstance(opt, MuonAdamW) != ("muon" in ck["optimizer"]):
            raise SystemExit(f"{ckpt_path} holds the state of "
                             f"{'Muon' if 'muon' in ck['optimizer'] else 'AdamW'}, not of --optimizer {args.optimizer}")
        adamw = getattr(opt, "adamw", opt)
        saved = (ck["optimizer"]["adamw"] if adamw is not opt else ck["optimizer"])["param_groups"]
        for g, mine in zip(saved, adamw.param_groups):    # the implementation this device takes
            g.update({k: mine[k] for k in ("fused", "foreach", "capturable")})
        opt.load_state_dict(ck["optimizer"])
        if saved[0]["weight_decay"] != h["weight_decay"]:
            log(f"[warn] weight decay {saved[0]['weight_decay']:g} of the checkpoint kept, not "
                f"{h['weight_decay']:g} of {hp}")
        step, best, epoch = ck["step"], ck.get("best", float("inf")), ck.get("epoch", 0)
        steps_per_epoch = ck.get("steps_per_epoch") or None
        train_loader.block = ck.get("shuffle_block", train_loader.block)
        if ema:
            ema = EMA(model, args.ema)          # the loaded weights if the checkpoint has no average
            if ck.get("ema"):
                for k, v in ema_weights(ck, model, log).items():
                    if k in ema.shadow:
                        ema.shadow[k].copy_(v)
        if shift is not None and ck.get("src_shift") is not None:
            with torch.no_grad():
                shift.shift.copy_(torch.as_tensor(ck["src_shift"]).to(shift.shift))
        if ck.get("batch_in_epoch"):
            # the epoch order is a function of (seed, epoch): continue after the last batch
            resume_batches = int(ck["batch_in_epoch"])
            epoch -= 1
        rows = history_until(hist_path, step)
        seconds = ck.get("seconds", rows[-1]["seconds"] if rows else 0.0)
        log(f"[resume] step {step}, epoch {epoch + (1 if resume_batches else 0)}"
            + (f" at batch {resume_batches:,}" if resume_batches else "")
            + f", best {args.select_on} {best:.2f}")
    else:
        history_until(hist_path, -1)          # lines of an attempt that died before its first checkpoint

    if init_ck is not None:
        sd = dict(init_ck["model"])
        if init_ck.get("ema"):
            sd.update({k: v.to(sd[k].dtype) for k, v in ema_weights(init_ck, model, log).items() if k in sd})
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing or unexpected:
            raise SystemExit(f"--init {args.init}: missing {missing[:5]}, unexpected {list(unexpected)[:5]}")
        if shift is not None and init_ck.get("src_shift") is not None:
            with torch.no_grad():
                shift.shift.copy_(torch.as_tensor(init_ck["src_shift"]).to(shift.shift))
        if args.fold_shift:
            refs = init_ck.get("ref_sources") or []
            if init_ck.get("src_shift") is None or not refs:
                raise SystemExit(f"--fold-shift: {args.init} has no per-element offsets (no --ref-sources)")
            if set(args.sources.split(",")) & set(refs):
                raise SystemExit(f"--fold-shift: --sources must leave out {args.init}'s reference "
                                 f"sources {refs}")
            with torch.no_grad():
                model.atomic_energies.add_(torch.as_tensor(init_ck["src_shift"]).to(model.atomic_energies))
        log(f"[init] weights from {args.init}" + (" (EMA)" if init_ck.get("ema") else "")
            + ("; its per-element offsets added to the reference energies" if args.fold_shift else ""))
        if args.refit_e0 > 0:
            refit_e0(model, shift, train_loader, args.refit_e0, len(species), species, device, log)
        if ema:
            ema = EMA(model, args.ema)

    if init_ck is None and not ckpt_path.exists():
        probe = list(itertools.islice(train_loader, 4))
        model.fit_feature_normalisation(probe)
        if args.scale_from == "forces":
            f = torch.cat([b["forces"].reshape(-1) for b in probe])
            scale = float(f[torch.isfinite(f)].pow(2).mean().sqrt())
        else:
            scale = float(st.get("e_per_atom_std", float("nan")))
        if not math.isfinite(scale):
            raise SystemExit("meta.json has no stats.train.e_per_atom_std: use --scale-from forces")
        model.energy_scale.fill_(max(scale, 1e-3))
        log(f"[init] whitening from {len(probe)} batches, energy_scale {float(model.energy_scale):.4f}")
        if ema:
            ema = EMA(model, args.ema)

    cstep = None
    if args.compile and device.type == "cuda":
        cstep = CompiledStep(model, args.batch_atoms, args.batch_structs, edge_bucket=args.edge_bucket,
                             with_stress=use_stress, mode=args.compile_mode, log=log)
    kE, kF = args.w_energy, args.w_forces
    lE, lF, lS = (k or args.loss for k in (args.loss_energy, args.loss_forces, args.loss_stress))
    log(f"[train] loss {args.loss}: {kE:g} x energy/atom + {kF:g} x forces"
        + (f" + {args.w_stress:g} x stress" if use_stress else "")
        + (f"; energy {lE}, forces {lF}, stress {lS}" if any((args.loss_energy, args.loss_forces, args.loss_stress)) else ""))
    (out / "hparams.json").write_text(json.dumps(h, indent=2))
    track = tracking.Tracker(args.track, out, dict(args=vars(args), hparams=h), args.track_project,
                             resume=ckpt_path.exists(), log=log)

    def save(tag="checkpoint", extra=None):
        train_loader.check_counts()           # nothing saved from batches that lost edges
        payload = dict(model=model.state_dict(), optimizer=opt.state_dict(), step=step, best=best,
                       epoch=epoch, steps_per_epoch=steps_per_epoch, ema=ema.shadow if ema else None,
                       seconds=seconds + time.time() - t0,
                       hparams={k: (list(v) if isinstance(v, tuple) else v) for k, v in h.items()},
                       species=species, e0=model.atomic_energies.detach().double().tolist(),
                       e0_fit=e0.tolist(), args=vars(args),
                       src_shift=shift.shift.detach().cpu() if shift is not None else None,
                       ref_sources=ref_sources, batch_in_epoch=batch_in_epoch, shuffle_block=train_loader.block)
        payload.update(extra or {})
        (out / tag).parent.mkdir(parents=True, exist_ok=True)
        tmp = out / f"{tag}.pt.tmp"
        torch.save(payload, tmp)
        tmp.replace(out / f"{tag}.pt")

    def run_validation():
        m = validate(model, val_loader, use_stress, shift, src_names if len(src_names) > 1 else None)
        if m:
            m.update(gate_stats(model, val_loader))
        if m and model.feat_bound:
            m.update(feat_stats(model, val_loader))
        if m and val_ref_loader is not None:
            mr = validate(model, val_ref_loader, use_stress, shift)
            m.update({f"ref_{k}": v for k, v in mr.items() if not k.startswith("e_mae_")})
        if m and score_refs:
            m["score"] = m["e_mae"] / score_refs[0] + m["f_mae"] / score_refs[1]
        return m

    def val_line(m):
        txt = (f"E_MAE {m['e_mae']:8.2f} meV/atom  E_RMSE {m['e_rmse']:8.2f}  "
               f"F_MAE {m['f_mae']:8.2f} meV/A  F_RMSE {m['f_rmse']:8.2f}")
        if "s_mae" in m:
            txt += f"  S_MAE {m['s_mae']:6.2f} S_RMSE {m['s_rmse']:7.2f} meV/A^3"
        if "e_mae_ref" in m:
            txt += f"  | E_MAE ref {m['e_mae_ref']:.1f} other {m.get('e_mae_other', float('nan')):.1f}"
        if "score" in m:
            txt += f"  | score {m['score']:.3f}"
        if "ref_e_mae" in m:
            txt += f"  | ref-only: E_MAE {m['ref_e_mae']:.1f} F_MAE {m['ref_f_mae']:.1f}"
        gates = [k.rsplit("/", 1)[1] for k in m if k.startswith("gate_mean/")]
        line = ["gate factors (mean/min) " + "  ".join(
            f"L{k}: {m[f'gate_mean/{k}']:.3f}/{m[f'gate_min/{k}']:.3f}" for k in gates)] if gates else []
        if "feat_max" in m:
            line.append(f"|x~| max {m['feat_max']:.2f}, {100 * m['feat_sat']:.2f} % above tau/2")
        if line:
            txt += "\n    " + "  ".join(line)
        return txt

    def record(m, **fields):
        with hist_path.open("a") as fh:
            fh.write(json.dumps(dict(fields, seconds=seconds + time.time() - t0, **m)) + "\n")
        track({**{f"val/{k}": v for k, v in m.items()}, "epoch": fields.get("epoch")}, fields["step"])

    t0 = time.time()
    run_loss = torch.zeros((), dtype=torch.float64, device=device)   # read once per log line
    n_skip = torch.zeros((), device=device)
    run_n = seen_atoms = atoms_last = 0
    t_last = t0
    fused_skip = bool(opt.param_groups[0].get("fused"))
    model.train()
    opt.zero_grad(set_to_none=True)
    if not len(train_loader.indices):
        raise SystemExit("no training structures (check --sources)")
    total_steps = args.steps
    if args.epochs:
        (out / "epochs").mkdir(exist_ok=True)
        if steps_per_epoch is None:
            steps_per_epoch = max(1, len(train_loader))
        total_steps = steps_per_epoch * args.epochs
        log(f"[train] {args.epochs} epochs of ~{steps_per_epoch:,} steps -> {total_steps:,} steps")

    while not _STOP:
        if (epoch >= args.epochs) if args.epochs else (step >= args.steps):
            break
        epoch += 1
        step_at_epoch_start = step - resume_batches
        train_loader.epoch, train_loader.skip = max(0, epoch - 1), resume_batches
        batch_in_epoch, resume_batches = resume_batches, 0
        for b in train_loader:
            if _STOP or (not args.epochs and step >= args.steps):
                break
            batch_in_epoch += 1
            lr = lr_at(step, h["lr"], args.warmup, total_steps, schedule=args.schedule,
                       decay_frac=args.decay_frac)
            for gp in opt.param_groups:
                gp["lr"] = lr
            if adamw_scale != 1.0:
                for gp in getattr(opt, "adamw", opt).param_groups:
                    gp["lr"] = lr * adamw_scale
            if cstep is not None:
                try:
                    e_pred, f_pred, s_pred = cstep(b)
                    out_ = {"energy": e_pred, "forces": f_pred, "stress": s_pred}
                except Exception as exc:          # fall back to eager if the compile fails
                    log(f"[compile] failed ({type(exc).__name__}: {exc}); eager from now on")
                    cstep = None
            if cstep is None:
                out_ = model(b, compute_stress=use_stress)
            e_pred_ = out_["energy"] + (shift(b) if shift is not None else 0)
            de = (e_pred_ - b["energy"]) / b["n_atoms_per_struct"]
            df = out_["forces"] - b["forces"]
            loss = kE * penalty(de, lE, DELTA["energy"]) + kF * force_penalty(df, lF, DELTA["forces"])
            if use_stress:
                loss = loss + args.w_stress * penalty_masked(out_["stress"] - b["stress"], stress_mask(b),
                                                             lS, DELTA["stress"])
            if not fused_skip and not torch.isfinite(loss):
                log(f"[warn] non-finite loss at step {step}, batch skipped")
                opt.zero_grad(set_to_none=True)
                continue
            loss.backward()
            run_loss += loss.detach().double()
            run_n += 1
            seen_atoms += b["n_atoms"]
            gnorm = torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]],
                                                   h["grad_clip"])
            if fused_skip:
                opt.found_inf = (~torch.isfinite(gnorm)).float()
                n_skip += opt.found_inf
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if ema:
                ema.update(model, step)
            if args.stop_after and step >= args.stop_after:
                _STOP = True

            if step % args.log_every == 0:
                now = time.time()
                mean_loss = run_loss.item() / max(run_n, 1)
                train_loader.check_counts()
                rate = (seen_atoms - atoms_last) / max(now - t_last, 1e-9)
                log(f"  step {step:>8,}  loss {mean_loss:10.5f}  lr {lr:.2e}  "
                    f"{seen_atoms / (now - t0):>8,.0f} atoms/s  {(now - t0) / 60:6.1f} min  | now {rate:,.0f}"
                    + (f"  peak {torch.cuda.max_memory_allocated(device) / 2**30:.1f} GiB" if device.type == "cuda" else "")
                    + (f"  [{int(n_skip.item())} non-finite steps skipped]"
                       if fused_skip and n_skip.item() else ""))
                track({"train/loss": mean_loss, "train/lr": lr, "train/atoms_per_s": rate}, step)
                run_loss.zero_()
                n_skip.zero_()
                run_n, t_last, atoms_last = 0, now, seen_atoms
            if step % args.val_every == 0 or (not args.epochs and step >= args.steps):
                m = run_validation()
                if m:
                    log(f"  VAL step {step:>8,}  " + val_line(m))
                    record(m, step=step, lr=lr)
                    if m.get(args.select_on, float("inf")) < best:
                        best = m[args.select_on]
                        save("best", dict(metrics=m))
                        log(f"  -> new best {args.select_on} {best:.2f}")
            if step % args.ckpt_every == 0:
                save("checkpoint")

        if args.epochs and not _STOP:
            batch_in_epoch = 0
            measured = max(1, step - step_at_epoch_start)
            if abs(measured - steps_per_epoch) > 0.1 * steps_per_epoch:
                log(f"[train] epoch {epoch} took {measured:,} steps, estimate {steps_per_epoch:,}")
            m = run_validation()
            log(f"  EPOCH {epoch:>4}/{args.epochs}  step {step:>8,}  " + (val_line(m) if m else "(no val)"))
            save(f"epochs/epoch_{epoch:03d}", dict(epoch=epoch, metrics=m))
            record(m, epoch=epoch, step=step)
            if m and m.get(args.select_on, float("inf")) < best:
                best = m[args.select_on]
                save("best", dict(metrics=m, epoch=epoch))
                log(f"  -> new best {args.select_on} {best:.2f}")
            save("checkpoint")

    save("checkpoint")
    log(f"\n[done] {step:,} steps in {(time.time() - t0) / 3600:.2f} h, best {args.select_on} {best:.2f}")
    track.summary({f"best/{args.select_on}": best})
    track.close()


@torch.no_grad()
def refit_e0(model, shift, loader, n_batches, n_species, species, device, log, min_structures=20):
    """Refit the reference energies with the network frozen: ridge-stabilised least squares
    of the energy residuals, for elements present in at least min_structures structures."""
    XtX = torch.zeros(n_species, n_species, dtype=torch.float64)
    Xty = torch.zeros(n_species, dtype=torch.float64)
    n_with = torch.zeros(n_species, dtype=torch.float64)
    for b in itertools.islice(loader, n_batches):
        with torch.enable_grad():
            e = model(b, compute_forces=False)["energy"]
        if shift is not None:
            e = e + shift(b)
        C = torch.zeros(b["n_struct"], n_species, dtype=torch.float64, device=device).index_put_(
            (b["batch"], b["mu"]), torch.ones_like(b["mu"], dtype=torch.float64), accumulate=True)
        r = (b["energy"] - e).double()
        XtX += (C.T @ C).cpu()
        Xty += (C.T @ r).cpu()
        n_with += (C > 0).sum(0).double().cpu()
    fit = n_with >= min_structures
    delta = torch.zeros(n_species, dtype=torch.float64)
    if fit.any():
        Aff = XtX[fit][:, fit]
        lam = 1e-3 * float(torch.diag(Aff).mean())
        delta[fit] = torch.linalg.solve(Aff + lam * torch.eye(int(fit.sum()), dtype=torch.float64), Xty[fit])
    model.atomic_energies.add_(delta.to(model.atomic_energies))
    top = torch.argsort(-delta.abs())[:8]
    log(f"[e0] {int(fit.sum())} elements refitted; largest changes "
        + ", ".join(f"{species[int(k)]} {float(delta[k]):+.3f}" for k in top) + " eV")


if __name__ == "__main__":
    main()
