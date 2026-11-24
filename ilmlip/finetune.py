"""Fine-tune a pretrained ilmlip model on your own data.

    python -m ilmlip.finetune --foundation OAM-S-omat --train data.xyz            # naive
    python -m ilmlip.finetune --foundation S.pt --train data.xyz --mode lora
    python -m ilmlip.finetune --foundation S.pt --train data.xyz --mode multihead
    python -m ilmlip.finetune --foundation S.pt --train a.xyz b.xyz --mode multihead --replay none
    python -m ilmlip.finetune --foundation S.pt --train data.xyz --wise 8 --wise-parallel 4
    python -m ilmlip.finetune --foundation S.pt --train data.xyz --init run/model.pt
    python -m ilmlip.finetune prepare-replay --foundation S.pt --out .   # replay structures, ahead of time

--foundation: a pretrained model by name (ilmlip.pretrained, downloaded on first use) or a checkpoint file.

naive: every weight trains.  lora: the foundation stays frozen and every weight matrix trains a
low-rank update, merged on saving.  multihead: a readout per --train file on the shared descriptor,
plus one trained on replay structures labelled by the foundation (--replay none leaves it out).
scratch: the foundation's architecture, randomly initialised.  --wise N: N naive fine-tunes that
differ only in their batch order, and their weight average.  --init: another cycle of --epochs from
an earlier fine-tune of the same foundation.

Adam by default; --optimizer muon trains the network's weight matrices with Muon and the rest with
Adam (lora: always Adam).  The replay structures also measure the drift: the force MAE against the
foundation on structures the fine-tuning never sees.  The reference energies (--e0s) are not trained.

Output (--out): model.pt (best epoch, EMA weights), loaded as is by every calculator; log.txt,
history.csv, summary.json; multihead also model_<stem>.pt for every further --train file and
model_pt_head.pt for the replay head.  All options: docs/reference.md.
"""
import argparse
import ast
import contextlib
import csv
import itertools
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.func import functional_call

from . import fused_torch, pretrained, runinfo, tracking
from . import vcut
from .model import DTYPES, collate, graph, load, set_dtype
from .replay import REPLAY_FILES, find_replay, prepare_cli, trained_elements
from .train import CompiledStep, Muon, MuonAdamW, lr_at, muon_matrices, penalty_masked, stress_mask

KEYS = dict(energy=("REF_energy", "energy", "free_energy", "DFT_energy", "dft_energy", "total_energy"),
            forces=("REF_forces", "forces", "DFT_forces", "dft_forces", "force"),
            stress=("REF_stress", "stress", "DFT_stress", "dft_stress", "REF_virial", "virial",
                    "REF_virials", "virials"))
NO_LABELS = dict(energy=None, forces=None, stress=None)
# default peak lr per optimizer and mode (Muon moves the weights farther per unit lr)
LR = dict(adam=dict(naive=3e-3, lora=1e-2, multihead=3e-3), muon=dict(naive=1.5e-3, multihead=1.5e-3))


class Log:
    """Every line to log.txt and, with echo, to the terminal."""

    def __init__(self, path, echo=True):
        self.fh, self.echo = open(path, "a", encoding="utf-8"), echo

    def __call__(self, msg=""):
        for line in str(msg).split("\n"):
            line = f"{time.strftime('%H:%M:%S')}  {line}" if line else ""
            if self.echo:
                print(line, flush=True)
            self.fh.write(line + "\n")
        self.fh.flush()

    def section(self, title):
        self()
        self(f"--- {title} " + "-" * max(3, 70 - len(title)))


def _value(atoms, key):
    stores = (atoms.info, atoms.arrays, getattr(atoms.calc, "results", None) or {})
    return next((s[key] for s in stores if key in s), None)


def pick_keys(frames, given):
    """The label key per quantity: the given one, else the first candidate any frame has."""
    has = lambda key: any(_value(a, key) is not None for a in frames)
    out = {}
    for q, cands in KEYS.items():
        if given.get(q) and not has(given[q]):
            raise SystemExit(f"--{q}-key {given[q]!r}: no structure has it")
        out[q] = given.get(q) or next((k for k in cands if has(k)), None)
    return out


def to_graph(atoms, keys, rcut, species):
    """Graph and labels (NaN where missing); a key with 'virial' is read as a virial."""
    g = graph(atoms, rcut, species)
    e, f, s = (_value(atoms, keys[q]) if keys[q] else None for q in ("energy", "forces", "stress"))
    g["energy"] = float(e) if e is not None else np.nan
    g["forces"] = np.asarray(f, float).reshape(-1, 3) if f is not None else np.full((len(atoms), 3), np.nan)
    if s is not None and g["volume"] > 1e-8:
        s = np.asarray(s, float).reshape(-1)
        if s.size == 9:
            s = s.reshape(3, 3)[[0, 1, 2, 1, 0, 0], [0, 1, 2, 2, 2, 1]]
        if s.size != 6:
            raise ValueError(f"{keys['stress']!r} has {s.size} values, want 6 or 9")
        g["stress"] = -s / g["volume"] if "virial" in keys["stress"].lower() else s
    g["charge"], g["spin"] = int(atoms.info.get("charge", 0)), int(atoms.info.get("spin", 1))
    return g


def read_data(paths, given, rcut, species, model=None):
    """Structures, graphs and label keys; a charge/spin model must have their charges and
    multiplicities in its tables (it would clamp them)."""
    from ase.io import read
    per_file = [read(p, index=":") for p in paths]
    frames = [a for fs in per_file for a in fs]
    if not frames:
        raise SystemExit(f"no structures in {paths}")
    keys = pick_keys(frames, given)
    graphs = [to_graph(a, keys, rcut, species) for a in frames]
    if getattr(model, "charge_spin", False):
        where = [(p, k) for p, fs in zip(paths, per_file) for k in range(len(fs))]
        for (p, k), g in zip(where, graphs):
            if abs(g["charge"]) > model.max_charge or not 1 <= g["spin"] <= model.max_spin:
                raise SystemExit(f"{p}, structure {k}: charge {g['charge']} / multiplicity {g['spin']} outside "
                                 f"the foundation's tables (|Q| <= {model.max_charge}, 1 <= 2S+1 <= {model.max_spin})")
    return frames, graphs, keys


def describe(name, paths, graphs, keys, species):
    el = sorted({species[k] for g in graphs for k in np.unique(g["mu"])})
    lab = []
    for q in KEYS:
        n = sum(bool(np.isfinite(g[q]).all()) for g in graphs)
        lab.append(f"{q} <- {keys[q]!r} ({n})" if keys[q] and n else f"{q}: none")
    return (f"{name:<6}{', '.join(Path(p).name for p in paths)}: {len(graphs):,} structures, "
            f"{sum(g['n_atoms'] for g in graphs):,} atoms, elements {' '.join(el)}\n      labels: "
            + ", ".join(lab))


def batches(graphs, size, rng=None):
    idx = rng.permutation(len(graphs)) if rng is not None else np.arange(len(graphs))
    for k in range(0, len(idx), size):
        yield [graphs[i] for i in idx[k:k + size]]


def by_atoms(items, batch_atoms):
    """Batches of neighbouring items in their order, each extended while it stays within
    batch_atoms atoms; an item bigger than that forms a batch by itself."""
    items, start = list(items), 0
    while start < len(items):
        stop, atoms = start + 1, items[start]["n_atoms"]
        while stop < len(items) and atoms + items[stop]["n_atoms"] <= batch_atoms:
            atoms += items[stop]["n_atoms"]
            stop += 1
        yield items[start:stop]
        start = stop


def on_device(graphs, device, dtype):
    """The structures as views into one copy of their arrays on the device (join makes batches)."""
    if not graphs:
        return []
    na, ne = [g["n_atoms"] for g in graphs], [len(g["ind_i"]) for g in graphs]
    put = lambda x, d=dtype: torch.tensor(np.asarray(x), dtype=d, device=device)
    cat = lambda key: np.concatenate([g[key] for g in graphs])
    parts = dict(positions=put(cat("positions")).split(na), mu=put(cat("mu"), torch.long).split(na),
                 forces=put(cat("forces")).split(na), ind_i=put(cat("ind_i"), torch.long).split(ne),
                 ind_j=put(cat("ind_j"), torch.long).split(ne), shift=put(cat("shift")).split(ne),
                 energy=put([g["energy"] for g in graphs], torch.float64).split(1),
                 n_atoms_per_struct=put(na).split(1),
                 volume=put([g["volume"] for g in graphs]).split(1),
                 stress=put([g["stress"] for g in graphs]).split(1),
                 charge=put([g.get("charge", 0) for g in graphs], torch.long).split(1),
                 spin=put([g.get("spin", 1) for g in graphs], torch.long).split(1))
    return [dict(n_atoms=n, **{k: v[i] for k, v in parts.items()}) for i, n in enumerate(na)]


def join(views):
    """The batch of structures on the device that collate makes of their graphs."""
    cat = lambda key: torch.cat([v[key] for v in views])
    first = np.cumsum([0] + [v["n_atoms"] for v in views])[:-1]
    moved = lambda key: torch.cat([v[key] + int(o) for v, o in zip(views, first)])
    k = torch.arange(len(views), device=views[0]["mu"].device)
    return dict(positions=cat("positions").requires_grad_(True), mu=cat("mu"),
                ind_i=moved("ind_i"), ind_j=moved("ind_j"), shift=cat("shift"),
                batch=torch.cat([k[j].expand(v["n_atoms"]) for j, v in enumerate(views)]),
                n_atoms=int(sum(v["n_atoms"] for v in views)), n_struct=len(views),
                energy=cat("energy"), forces=cat("forces"), n_atoms_per_struct=cat("n_atoms_per_struct"),
                volume=cat("volume"), stress=cat("stress"), charge=cat("charge"), spin=cat("spin"))


def make_batch(items, device, dtype):
    return join(items) if torch.is_tensor(items[0]["mu"]) else collate(items, device=device, dtype=dtype)


def parse_e0s(spec, species):
    from ase.data import chemical_symbols
    d = ast.literal_eval(spec) if spec.strip().startswith("{") else json.loads(Path(spec).read_text())
    out = {}
    for k, v in d.items():
        sym = chemical_symbols[int(k)] if str(k).isdigit() else str(k)
        if sym not in species:
            raise SystemExit(f"--e0s: {k!r} is not an element of the model")
        out[species.index(sym)] = float(v)
    return out


def choose_e0s(spec, model, graphs, predict, log):
    """New E0s for all species (only those in graphs change)."""
    sp = model.species
    old = model.atomic_energies.detach().double().cpu().numpy()
    new = old.copy()
    C = np.stack([np.bincount(g["mu"], minlength=len(sp)) for g in graphs]).astype(float)
    present = np.flatnonzero(C.sum(0))
    E = np.array([g["energy"] for g in graphs])
    has = np.isfinite(E)
    Cp = C[has][:, present]
    if spec in ("estimated", "composition") and not has.any():
        log(f"[warn] --e0s {spec}: no training energies; keeping the foundation's")
        spec = "foundation"
    if spec == "estimated":
        resid = E[has] - predict([g for g, h in zip(graphs, has) if h])
        new[present] += np.linalg.lstsq(Cp, resid, rcond=None)[0]
        log("E0s estimated: per-element corrections of what the foundation misses on the training energies")
    elif spec == "composition":                        # scratch: no foundation energy to correct
        new[present] = np.linalg.lstsq(Cp, E[has], rcond=None)[0]
        log("E0s from the composition: least squares of the training energies (scratch)")
    elif spec == "foundation":
        log("E0s foundation: kept as trained")
    else:
        given = parse_e0s(spec, sp)
        missing = [sp[k] for k in present if k not in given]
        if missing:
            raise SystemExit(f"--e0s: no value for {missing}, which the training data has")
        for k, v in given.items():
            new[k] = v
        log("E0s given on the command line")
    log("  element     atoms   foundation E0   fine-tune E0        change   (eV)")
    for k in present:
        log(f"  {sp[k]:<7} {int(C[:, k].sum()):>9,} {old[k]:15.4f} {new[k]:14.4f} {new[k] - old[k]:13.4f}")
    rank = int(np.linalg.matrix_rank(Cp)) if has.any() else 0
    if spec in ("estimated", "composition") and rank < len(present):
        log(f"[note] the training energies fix only {rank} combination(s) of {len(present)} E0s; "
            f"other compositions need given E0s (e.g. isolated atoms)")
    return new


def _key(name):
    return name.replace(".", "__")


def is_head(name):
    return name.startswith("readout.") or name in ("elem_bias", "head", "atomic_energies")


class Adapter(nn.Module):
    """The tensors that replace the foundation's in a forward pass (functional_call)."""

    def __init__(self, model, mode, rank=8, alpha=None, heads=()):
        super().__init__()
        self.mode = mode
        self.A, self.B, self.heads = nn.ParameterDict(), nn.ParameterDict(), nn.ModuleDict()
        model.atomic_energies.requires_grad_(False)
        named = dict(model.named_parameters())
        if mode == "lora":
            model.requires_grad_(False)
            self.scale = float(alpha or rank) / rank
            self.base = {n: p for n, p in named.items() if p.dim() >= 2}
            for n, p in self.base.items():
                r = min(rank, p.shape[0], p[0].numel())
                a = torch.empty(r, p[0].numel(), dtype=p.dtype, device=p.device)
                nn.init.kaiming_uniform_(a, a=math.sqrt(5))
                self.A[_key(n)] = nn.Parameter(a)
                self.B[_key(n)] = nn.Parameter(torch.zeros(p.shape[0], r, dtype=p.dtype, device=p.device))
        elif mode == "multihead":
            self.head_names = [n for n in named if is_head(n)]
            for h in heads:
                self.heads[h] = nn.ParameterDict({_key(n): nn.Parameter(named[n].detach().clone(),
                                                                        requires_grad=named[n].requires_grad)
                                                  for n in self.head_names})

    def weights(self, head=0):
        """Head 0 is the model's own readout (and E0s); multihead: "1", "2", ... (the further
        --train files) and "pt" are copies of it."""
        if self.mode == "lora":
            return {n: W + self.scale * (self.B[_key(n)] @ self.A[_key(n)]).view_as(W)
                    for n, W in self.base.items()}
        if self.mode == "multihead" and head != 0:
            return {n: self.heads[str(head)][_key(n)] for n in self.head_names}
        return {}


class EMA:
    """Moving average of the trained tensors, decay ramped as min(decay, (1+n)/(10+n))."""

    def __init__(self, params, decay):
        self.params, self.decay, self.n = params, float(decay), 0
        self.avg = [p.detach().clone() for p in params]

    @torch.no_grad()
    def update(self):
        self.n += 1
        d = min(self.decay, (1 + self.n) / (10 + self.n))
        torch._foreach_lerp_(self.avg, self.params, 1.0 - d)

    @contextlib.contextmanager
    def applied(self):
        keep = [p.detach().clone() for p in self.params]
        with torch.no_grad():
            for a, p in zip(self.avg, self.params):
                p.copy_(a)
        try:
            yield
        finally:
            with torch.no_grad():
                for k, p in zip(keep, self.params):
                    p.copy_(k)


def forward(model, weights, b, stress):
    return functional_call(model, weights, (b,), dict(compute_stress=stress))


def residuals(out, b):
    """Per-atom energy, force and stress errors; NaN labels left out."""
    dE = ((out["energy"] - b["energy"]) / b["n_atoms_per_struct"])[torch.isfinite(b["energy"])]
    dF = (out["forces"] - b["forces"])[torch.isfinite(b["forces"]).all(1)]
    dS = (out["stress"] - b["stress"])[stress_mask(b)] if "stress" in out else None
    return dE, dF, dS


def loss_of(out, b, w, huber=None):
    """Weighted mean squared errors, or Huber losses with the threshold huber (eV/atom, eV/A),
    over the labels there are: (loss, number of labels, the weighted energy, force and stress
    terms), without waiting for the device."""
    terms = [(((out["energy"] - b["energy"]) / b["n_atoms_per_struct"])[:, None], torch.isfinite(b["energy"])),
             (out["forces"] - b["forces"], torch.isfinite(b["forces"]).all(1))]
    if "stress" in out:
        terms.append((out["stress"] - b["stress"], stress_mask(b)))
    zero = out["energy"].new_zeros(())
    parts, n = [], 0
    for (d, ok), wk in zip(terms, w):
        parts.append(wk * penalty_masked(d, ok, "huber" if huber else "mse", huber) if wk else zero)
        if wk:
            n = n + ok.sum()
    parts += [zero] * (3 - len(parts))
    return sum(parts, zero), n, torch.stack([x.double() for x in parts])


class Errors:
    """Error sums -> MAE / RMSE (meV), the loss (squared errors) and its energy, force and
    stress terms; f_vec_mae = mean per-atom |dF|."""

    def __init__(self):
        self.s = dict.fromkeys(("e1", "e2", "en", "f1", "f2", "fn", "s1", "s2", "sn", "fv", "fvn"), 0.0)

    def __add__(self, other):
        out = Errors()
        out.s = {k: v + other.s[k] for k, v in self.s.items()}
        return out

    def add(self, res):
        for q, d in zip("efs", res):
            if d is not None and d.numel():
                d = d.detach().double()
                self.s[q + "1"] += float(d.abs().sum())
                self.s[q + "2"] += float((d ** 2).sum())
                self.s[q + "n"] += d.numel()
        if res[1].numel():
            self.s["fv"] += float(res[1].detach().double().norm(dim=1).sum())
            self.s["fvn"] += res[1].shape[0]

    def result(self, w):
        out, loss = {}, 0.0
        for q, wk in zip("efs", w):
            n = self.s[q + "n"]
            if n:
                out[f"{q}_mae"] = 1000 * self.s[q + "1"] / n
                out[f"{q}_rmse"] = 1000 * math.sqrt(self.s[q + "2"] / n)
                out[f"loss_{q}"] = wk * self.s[q + "2"] / n
                loss += out[f"loss_{q}"]
        if self.s["fvn"]:
            out["f_vec_mae"] = 1000 * self.s["fv"] / self.s["fvn"]
        out["loss"] = loss
        return out


def errors(model, weights, items, batch_atoms, device, dtype, stress):
    """The Errors of the structures (graphs or on_device views), in batches of up to batch_atoms atoms."""
    model.eval()
    acc = Errors()
    weights = {k: v.detach() for k, v in weights.items()}
    for group in by_atoms(items, batch_atoms):
        b = make_batch(group, device, dtype)
        acc.add(residuals(forward(model, weights, b, stress), b))
    model.train()
    return acc


def evaluate(model, weights, items, batch_atoms, device, dtype, w, stress):
    """The errors on the structures (graphs or on_device views), in batches of up to batch_atoms atoms."""
    return errors(model, weights, items, batch_atoms, device, dtype, stress).result(w)


def interleave(lists):
    """(k, item) for the items of the lists, each list spread evenly over the sequence."""
    order = sorted(((i + 0.5) / len(l), k, i) for k, l in enumerate(lists) for i in range(len(l)))
    return [(k, lists[k][i]) for _, k, i in order]


@torch.no_grad()
def energies(model, graphs, batch_atoms, device, dtype):
    model.eval()
    out = [model(collate(g, device=device, dtype=dtype), compute_forces=False)["energy"]
           for g in by_atoms(graphs, batch_atoms)]
    model.train()
    return torch.cat(out).double().cpu().numpy()


def pseudolabel(model, graphs, batch_atoms, device, dtype, stress=True):
    """Label structures with the foundation itself (energy, forces, stress), in batches of up to batch_atoms atoms."""
    model.eval()
    for group in by_atoms(graphs, batch_atoms):
        out = model(collate(group, device=device, dtype=dtype), compute_stress=stress)
        e, f = (out[k].detach().double().cpu().numpy() for k in ("energy", "forces"))
        s = out["stress"].detach().double().cpu().numpy() if stress else None
        at = 0
        for j, g in enumerate(group):
            g["energy"], g["forces"] = float(e[j]), f[at:at + g["n_atoms"]]
            g["stress"] = s[j] if stress and g["volume"] > 1e-8 else np.full(6, np.nan)
            at += g["n_atoms"]
    model.train()


def export(model, weights):
    """The state_dict the calculators load: the model with `weights` in."""
    return {k: v.detach().cpu().clone() for k, v in {**model.state_dict(), **weights}.items()}


def save(path, sd, blob, meta):
    tmp = Path(str(path) + ".tmp")
    # meta as JSON types (args may hold paths), so that the file loads with weights_only=True
    torch.save(dict(model=sd, hparams=blob.get("hparams"), species=blob.get("species"),
                    e0=sd["atomic_energies"].tolist(), finetune=json.loads(json.dumps(meta, default=str))), tmp)
    tmp.replace(path)


def table(rows):
    cols = [(k, h) for k, h in (("e_mae", "E MAE"), ("e_rmse", "E RMSE"), ("f_mae", "F MAE"),
                                ("f_rmse", "F RMSE"), ("s_mae", "S MAE"), ("s_rmse", "S RMSE"))
            if any(k in m for _, _, m in rows)]
    wid = max([22] + [len(name) + 2 for name, _, _ in rows])
    out = [f"{'':<{wid}}{'structs':>8}" + "".join(f"{h:>9}" for _, h in cols)]
    for name, n, m in rows:
        out.append(f"{name:<{wid}}{n:>8,}" + "".join(f"{m[k]:9.2f}" if k in m else f"{'-':>9}" for k, _ in cols))
    return "\n".join(out) + "\n(meV/atom, meV/A, meV/A^3)"


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_argument_group("what")
    g.add_argument("--foundation", required=True,
                   help="a pretrained model (python -m ilmlip.pretrained lists them) or a checkpoint file")
    g.add_argument("--mode", choices=("naive", "lora", "multihead", "scratch"), default="naive")
    g.add_argument("--init", help="start from this fine-tune of the foundation (its model.pt) instead")
    g.add_argument("--train", nargs="+", required=True, help="training structures (multihead: a head per file)")
    g.add_argument("--valid", nargs="+", help="validation structures (default: --valid-fraction of --train)")
    g.add_argument("--valid-fraction", type=float, default=0.1)
    g.add_argument("--test", nargs="+", help="test structures, evaluated at the end")
    g.add_argument("--e0s", default="estimated", help="estimated | foundation | dict | file.json")
    g.add_argument("--energy-key")
    g.add_argument("--forces-key")
    g.add_argument("--stress-key", help="a key containing 'virial' is read as a virial")
    g.add_argument("--out", help="output directory (default <train>_<foundation>_<mode>)")
    g = p.add_argument_group("training")
    g.add_argument("--epochs", type=int, default=200)
    g.add_argument("--batch-size", type=int, default=4, help="structures per batch")
    g.add_argument("--lr", type=float, help="peak learning rate (default: per optimizer and mode, see LR)")
    g.add_argument("--w-energy", type=float, default=1.0)
    g.add_argument("--w-forces", type=float, default=10.0)
    g.add_argument("--w-stress", type=float, default=10.0)
    g.add_argument("--huber", type=float, help="Huber loss with this threshold instead of squared errors")
    g.add_argument("--ema", type=float, default=0.99)
    g.add_argument("--save-every", type=int, default=0, metavar="N",
                   help="also keep the model of every Nth epoch and of the last as model_epNNN.pt (1: every epoch)")
    g.add_argument("--clip", type=float, default=10.0, help="gradient-norm clip")
    g.add_argument("--optimizer", choices=("muon", "adam"), default="adam",
                   help="adam: Adam alone; muon: Muon on the network's trainable weight matrices, Adam on the "
                        "rest (lora: Adam alone; Muon fine-tunes were unstable in MD)")
    g.add_argument("--muon-graph", action=argparse.BooleanOptionalAction,
                   help="the Muon step as one CUDA graph (default: on a GPU)")
    g.add_argument("--seed", type=int, default=0, help="split, shuffling, LoRA initialisation")
    g.add_argument("--lora-rank", type=int, default=8)
    g.add_argument("--lora-alpha", type=float, help="default: the rank")
    g.add_argument("--replay", help="folder with the replay files, mptrj, a URL, a file, or none")
    g.add_argument("--replay-n", type=int, default=1000)
    g.add_argument("--replay-weight", type=float, default=1.0,
                   help="weight of the replay head's loss (0: no replay head)")
    g = p.add_argument_group("WiSE (naive)")
    g.add_argument("--wise", type=int, default=1, metavar="N",
                   help="N fine-tunes with different batch orders, model.pt = their weight average")
    g.add_argument("--wise-parallel", type=int, default=1, metavar="P",
                   help="train P of them at the same time (default: one after the other)")
    g = p.add_argument_group("runtime")
    g.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    g.add_argument("--dtype", choices=DTYPES, default="tf32",
                   help="tf32: float32 with TF32 matrix products (they only act on NVIDIA GPUs; geometry "
                        "stays float32); float32: no TF32; float64: double precision")
    g.add_argument("--compile", action=argparse.BooleanOptionalAction,
                   help="compile the training step (default: on a GPU)")
    g.add_argument("--kernels", choices=("auto", *fused_torch.KERNELS), default="auto",
                   help="auto: fused on a CUDA device with cuequivariance-torch, else torch. fused: the bond sums "
                        "and products as cuEquivariance kernels (CUDA, needs cuequivariance-ops-torch-cu12); same "
                        "weights and checkpoints; fused-reference: the same in plain torch")
    g.add_argument("--resident", choices=("device", "host"), default="device",
                   help="structures on the device, or copied there for every batch")
    tracking.add_arguments(g)
    return p


def main(argv=None):
    runinfo.called("ilmlip.finetune", argv)
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv[:1] == ["prepare-replay"]:
        return prepare_cli(argv[1:])
    args = parser().parse_args(argv)
    if args.wise < 1 or args.wise_parallel < 1:
        raise SystemExit("--wise and --wise-parallel are counts of at least 1")
    if args.wise > 1:
        from .wise import wise
        return wise(args)
    return fine_tune(args)


def fine_tune(args, member=None, average_of=None, echo=True):
    """One fine-tune; member k of a --wise run; or, average_of = the finished members'
    directories, their weight average, evaluated like a fine-tune instead of trained."""
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    dtype, device = set_dtype(args.dtype), torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit(f"--device {args.device}: torch {torch.__version__} cannot use a GPU here")
    try:
        ck_path = pretrained.path(args.foundation)
    except (ValueError, FileNotFoundError) as exc:
        raise SystemExit(f"--foundation: {exc}")
    multi, scratch = (args.mode == m for m in ("multihead", "scratch"))
    # multihead: a head per --train file (the first one the model's own readout) and the replay head
    groups = [[f] for f in args.train] if multi else [args.train]
    heads = [Path(g[0]).stem for g in groups]
    files = ["model.pt"] + [f"model_{h}.pt" for h in heads[1:]]
    pt = multi and args.replay_weight > 0 and args.replay != "none"
    extra = [(str(k), f) for k, f in enumerate(files) if k] + ([("pt", "model_pt_head.pt")] if pt else [])
    if multi and len(groups) + pt < 2:
        raise SystemExit("multihead needs a second head: replay structures (--replay) or another --train file")
    if len(set(heads)) < len(heads) or "pt_head" in heads[1:]:
        raise SystemExit("multihead names its heads by the --train files' stems: they must differ, and not be pt_head")
    for opt in ("valid", "test"):
        if len(groups) > 1 and getattr(args, opt) and len(getattr(args, opt)) != len(groups):
            raise SystemExit(f"--{opt}: one file per --train file, in the same order (multihead)")
    if args.init and not Path(args.init).is_file():
        raise SystemExit(f"--init: no checkpoint at {args.init}")
    missing = [f for _, f in extra if args.init and not Path(args.init).with_name(f).is_file()]
    if missing:
        raise SystemExit(f"--init {args.init}: multihead continues its other heads, {', '.join(missing)} next to it")
    out = Path(args.out or f"{Path(args.train[0]).stem}_{ck_path.stem}_{args.mode}")
    if (out / "model.pt").exists():
        raise SystemExit(f"{out / 'model.pt'} exists: choose another --out or remove it")
    out.mkdir(parents=True, exist_ok=True)
    log = Log(out / "log.txt", echo)

    what = (f", WiSE member {member}" if member is not None
            else f", the weight average of {len(average_of)} fine-tunes" if average_of else "")
    log(f"ilmlip fine-tuning, mode {args.mode}{what}   ({time.strftime('%Y-%m-%d %H:%M')})")
    runinfo.show(log)
    args.kernels = fused_torch.choose(args.kernels, device, log)
    try:
        fused_torch.require(args.kernels, device)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from None
    model, blob = load(str(ck_path), "cpu", untrained=scratch, kernels=args.kernels)
    model = model.to(device, dtype)
    n_net = sum(p.numel() for p in model.parameters())
    log(f"foundation {ck_path}" + ("  (architecture only: from scratch)" if scratch else "")
        + f"\n           {n_net:,} parameters, rcut {model.rcut} A, ZBL {'on' if model.zbl else 'off'}")
    log(f"device     {device}, {args.dtype}; torch {torch.__version__}\noutput     {out.resolve()}")

    log.section("data")
    given = dict(energy=args.energy_key, forces=args.forces_key, stress=args.stress_key)
    sp = model.species
    trained = trained_elements(sp, model.atomic_energies.detach().cpu())
    trains, valids, tests, split = [], [], [], []
    for k, group in enumerate(groups):
        mine = lambda more: more if len(groups) == 1 else [more[k]]
        _, train, keys = read_data(group, given, model.rcut, sp, model)
        log(describe("train", group, train, keys, sp))
        if args.valid:
            _, valid, k2 = read_data(mine(args.valid), given, model.rcut, sp, model)
            log(describe("valid", mine(args.valid), valid, k2, sp))
        else:
            n_val = (max(1, round(args.valid_fraction * len(train)))
                     if args.valid_fraction > 0 and len(train) > 1 else 0)
            perm = rng.permutation(len(train))
            split.append(sorted(int(i) for i in perm[:n_val]))
            valid, train = [train[i] for i in split[-1]], [train[i] for i in sorted(perm[n_val:])]
            log(f"split  {len(train)} train / {len(valid)} valid (random, seed {args.seed})")
        test = []
        if args.test:
            _, test, k3 = read_data(mine(args.test), given, model.rcut, sp, model)
            log(describe("test", mine(args.test), test, k3, sp))
        trains.append(train)
        valids.append(valid)
        tests.append(test)
    split = None if args.valid else split[0] if len(groups) == 1 else dict(zip(heads, split))
    for t in (*trains, *valids, *tests):
        t[:] = [vcut.prune(g, model) for g in t]
    if member:
        rng = np.random.default_rng([args.seed, member])
        log(f"batches    shuffled with the seed ({args.seed}, {member})")
    every = [g for t in trains for g in t]
    has = lambda q, gs: any(np.isfinite(g[q]).all() for g in gs)
    if not (has("energy", every) or has("forces", every)):
        raise SystemExit("the training data has neither energies nor forces")
    stress = args.w_stress > 0 and has("stress", every)
    # stresses only from a model trained on them: the foundation's, or this fine-tune's
    found_stress = model.stress and not scratch
    model.stress = stress or found_stress
    blob["hparams"] = dict(blob.get("hparams") or {}, stress=model.stress)
    w = (args.w_energy, args.w_forces, args.w_stress if stress else 0.0)
    if any(valids) and not any(has(q, gs) for q, wk in zip(KEYS, w) if wk for gs in valids):
        log("[warn] the validation structures have no labels: the last epoch is kept")
        valids = [[] for _ in valids]
    elif not any(valids):
        log("[warn] no validation set: the last epoch is kept")
    if scratch:
        sample = [every[i] for i in sorted(rng.choice(len(every), 64, replace=False))] if len(every) > 64 else every
        model.fit_feature_normalisation([collate(g, device=device, dtype=dtype)
                                         for g in batches(sample, args.batch_size)])
        F = np.concatenate([g["forces"].reshape(-1) for g in every])
        f_rms = float(np.sqrt(np.nanmean(F ** 2))) if np.isfinite(F).any() else 1.0
        model.energy_scale.fill_(max(f_rms, 1e-3))
        log(f"scratch: whitening from {len(sample)} structures, energy scale {f_rms:.3f} eV")
    most = lambda gs: sum(sorted(g["n_atoms"] for g in gs)[-args.batch_size:])
    batch_atoms = max(most(t) for t in trains)
    # evaluation keeps no graph for a second derivative: four training batches, at least a pretraining batch
    budget = max(2048, 4 * batch_atoms)

    log.section("model")
    rep_train = rep_valid = []
    rep_dir = None if scratch else find_replay(args.replay, pt, args.replay_n, args.seed, out, log, trained)
    if rep_dir is not None:
        def labelled(name):
            from ase.io import read
            frames = read(rep_dir / name, ":")
            known = [a for a in frames if set(a.get_chemical_symbols()) <= set(trained)]
            if len(known) < len(frames):
                log(f"[warn] {name}: {len(frames) - len(known)} of {len(frames)} structures have "
                    f"elements the foundation was not trained on, left out")
            gs = [to_graph(a, NO_LABELS, model.rcut, sp) for a in known]
            gs = [vcut.prune(g, model) for g in gs]
            pseudolabel(model, gs, budget, device, dtype, found_stress)
            # far-from-equilibrium frames would dominate the loss
            return [g for g in gs if np.abs(g["forces"]).max() <= 5.0]
        rep_valid = labelled(REPLAY_FILES[1])
        rep_train = labelled(REPLAY_FILES[0]) if pt else []
        if pt and not rep_train:
            raise SystemExit(f"no replay structures in {rep_dir / REPLAY_FILES[0]}")
        log(f"           labelled by the foundation: {len(rep_train)} train the replay head, "
            f"{len(rep_valid)} measure the drift")
    if multi:
        log("heads      " + ", ".join(f"{h} -> {f}" for h, f in zip(heads, files))
            + (", replay -> model_pt_head.pt" if pt else ""))
    if args.init:
        init = torch.load(args.init, map_location="cpu", weights_only=True)
        try:
            model.load_state_dict(init["model"])
        except (KeyError, RuntimeError) as exc:
            raise SystemExit(f"--init {args.init}: not a model of the foundation's architecture ({exc})")
        log(f"init       {Path(args.init).resolve()}")
        prev = (init.get("finetune") or {}).get("args") or {}
        as_saved = lambda v: json.loads(json.dumps(v, default=str))
        other = [k for k in ("train", "valid", "valid_fraction", "seed") if k in prev and prev[k] != as_saved(getattr(args, k))]
        if other:
            log(f"[warn] --init was trained with another {', '.join(other)}: its training structures "
                f"may be in this validation split")
    adapter = Adapter(model, args.mode, args.lora_rank, args.lora_alpha, heads=[h for h, _ in extra])
    if args.init:
        with torch.no_grad():
            for h, f in extra:
                sd = torch.load(Path(args.init).with_name(f), map_location="cpu", weights_only=True)["model"]
                for n in adapter.head_names:
                    adapter.heads[h][_key(n)].copy_(sd[n])

    log.section("reference energies (E0)")
    own_e0 = lambda k: model.atomic_energies if k == 0 else adapter.heads[str(k)]["atomic_energies"]
    if args.init:
        e0s = [own_e0(k).detach().double().cpu().numpy() for k in range(len(trains))]
        log("E0s of --init, unchanged")
    else:
        if scratch and args.e0s == "estimated":
            args.e0s = "composition"
        predict = lambda gs: energies(model, gs, budget, device, dtype)
        e0s = []
        for h, train in zip(heads, trains):
            if len(trains) > 1:
                log(f"head {h}")
            e0s.append(choose_e0s(args.e0s, model, train, predict, log))
        with torch.no_grad():
            for k, e0 in enumerate(e0s):
                own_e0(k).copy_(torch.as_tensor(e0, dtype=dtype))

    params = [p for p in list(model.parameters()) + list(adapter.parameters()) if p.requires_grad]
    n_train = sum(p.numel() for p in params)
    base_lr = float((blob.get("hparams") or {}).get("lr", 1e-3))
    opt_name = "adam" if args.mode == "lora" else args.optimizer
    lr = args.lr if args.lr is not None else base_lr if scratch else LR[opt_name][args.mode]
    spe = sum(math.ceil(len(t) / args.batch_size) for t in trains)
    total = args.epochs * spe
    warm = max(1, total // 20)
    names = set(muon_matrices(model)) if args.optimizer == "muon" else set()
    mats = [p for n, p in model.named_parameters() if n in names and p.requires_grad]
    log(f"{args.mode}: {n_train:,} trainable parameters; lr {lr:.2e}, cosine to 2 % with {warm} warm-up "
        f"steps, " + (f"Muon on {sum(p.numel() for p in mats):,} of them in {len(mats)} matrices + " if mats else "")
        + f"Adam (amsgrad), clip {args.clip:g}, EMA {args.ema:g}; {args.epochs} epochs x {spe} steps"
        + (f"; Huber loss, threshold {args.huber:g}" if args.huber else ""))
    if args.optimizer == "muon" and not mats:
        log(f"{args.mode}: no trainable weight matrices for Muon, Adam alone")
    # fused on the GPU: one kernel, which also skips a step without a finite loss or gradient there
    fused = device.type == "cuda"
    in_muon = {id(p) for p in mats}
    opt = torch.optim.Adam([p for p in params if id(p) not in in_muon], lr=lr, amsgrad=True,
                           **({"fused": True} if fused else {}))
    if mats:
        opt = MuonAdamW(Muon(mats, lr=lr, graph=fused if args.muon_graph is None else args.muon_graph), opt)
    ema = EMA(params, args.ema)
    if args.resident == "device":
        trains, valids, tests = ([on_device(gs, device, dtype) for gs in s] for s in (trains, valids, tests))
        rep_train, rep_valid = (on_device(gs, device, dtype) for gs in (rep_train, rep_valid))
    compiled = (device.type == "cuda" if args.compile is None else args.compile) and not average_of
    csteps = rstep = None
    if compiled:
        mode = "reduce-overhead" if device.type == "cuda" else None
        csteps = [CompiledStep(model, most(t), args.batch_size, with_stress=stress, mode=mode, log=log)
                  for t in trains]
        if pt:
            rstep = CompiledStep(model, most(rep_train), args.batch_size, with_stress=found_stress, mode=mode, log=log)
    log("data       " + ("kept on the device" if args.resident == "device" else "copied for every batch")
        + ("" if average_of else f"; training step {'compiled' if compiled else 'eager'}"))
    ev = lambda gs, wt, ww=w, st=stress: evaluate(model, wt, gs, budget, device, dtype, ww, st)
    sets = [(s if len(trains) == 1 else f"{s} {heads[k]}", k, gs)
            for s, parts in (("train", trains), ("valid", valids), ("test", tests)) for k, gs in enumerate(parts)]
    log.section("start" if scratch or args.init else "zero-shot")
    zero = {n: ev(gs, adapter.weights(k)) for n, k, gs in sets if gs}
    log(table([(n, len(gs), zero[n]) for n, _, gs in sets if gs]))
    # WiSE members and their average are one group, named after the --wise --out
    as_run = ((f"{out.parent.name}/{out.name}", out.parent.name) if member is not None
              else (f"{out.name}/average", out.name) if average_of else (out.name, None))
    track = tracking.Tracker(args.track, out, vars(args), args.track_project, *as_run,
                             step_name="epoch", log=log)
    if "valid" in zero and not scratch and not average_of:
        track(dict(valid=zero["valid"]), 0)

    e0_of = lambda k: dict(zip(sp, map(float, e0s[k])))
    meta = dict(mode=args.mode, foundation=str(ck_path), e0s="init" if args.init else args.e0s,
                e0=e0_of(0), args=vars(args), replay=str(rep_dir.resolve()) if rep_dir else None)
    if member is not None:
        meta["wise_member"] = member
    step, epoch, best, best_epoch, best_sds = 0, 0, math.inf, 0, None
    seen_atoms, t_train = 0, 0.0
    w_rep = (args.w_energy, args.w_forces, args.w_stress)
    if average_of:
        members = [json.loads((d / "summary.json").read_text()) for d in average_of]
        best_sds, best_epoch, epoch = [average([d / "model.pt" for d in average_of])], "average", args.epochs
        step, t_train = sum(s["steps"] for s in members), sum(s["train_seconds"] for s in members)
        seen_atoms = sum(s["atoms_per_second"] * s["train_seconds"] for s in members)
        meta["wise"] = [str(d.resolve()) for d in average_of]
        log.section(f"results: the weight average of {len(average_of)} fine-tunes")
        log(f"the members trained {step:,} steps in {t_train:.0f} s together")
    else:
        log.section("training (validation errors of the EMA weights)")
        log("epoch     lr     train loss  valid loss |   E MAE  E RMSE   F MAE  F RMSE")
        log("loss: weighted errors" + (" (train: Huber, valid: squared)" if args.huber else " (squared)")
            + ("; train without the replay head" if pt else "") + "; its terms are in history.csv")
        hist_f = open(out / "history.csv", "w", newline="")
        hist = None
        rep_iter = (g for _ in itertools.count() for g in batches(rep_train, args.batch_size, rng))
        as_out = lambda r: dict(energy=r[0], forces=r[1], **({} if r[2] is None else {"stress": r[2]}))

        def gradients(k, b, rb):
            """The loss of a step of head k, its number of labels and the loss terms (energy, forces,
            stress, replay); the gradients in the parameters."""
            if csteps is not None:
                pred = as_out(csteps[k](b, adapter.weights(k)))
                rpred = as_out(rstep(rb, adapter.weights("pt"), new_step=False)) if pt else None
            else:
                pred = forward(model, adapter.weights(k), b, stress)
                rpred = forward(model, adapter.weights("pt"), rb, found_stress) if pt else None
            loss, n, parts = loss_of(pred, b, w, args.huber)
            rl = parts.new_zeros(())
            if pt:
                lp, nr, _ = loss_of(rpred, rb, w_rep, args.huber)
                rl = lp * args.replay_weight
                loss, n = loss + rl, n + nr
            opt.zero_grad(set_to_none=True)
            loss.backward()
            return loss, n, torch.cat([parts, rl.double()[None]])

        def keep(epoch, vals, tag=""):
            """The --train files' heads as they are now (state dicts); these and the replay head are
            saved, as model.pt etc. or with tag before the .pt (--save-every)."""
            sds = [export(model, adapter.weights(k)) for k in range(len(trains))]
            for k, sd in enumerate(sds):
                save(out / f"{files[k][:-3]}{tag}.pt", sd, blob, dict(meta, epoch=epoch, valid=vals[k], e0=e0_of(k)))
            if pt:
                sd = export(model, adapter.weights("pt"))
                e0 = dict(zip(sp, map(float, sd["atomic_energies"])))
                save(out / f"model_pt_head{tag}.pt", sd, blob, dict(meta, epoch=epoch, head="pt", e0=e0))
            return sds

        run = torch.zeros(4, dtype=torch.float64, device=device)      # energy, forces, stress, replay terms
        n_ok = torch.zeros((), dtype=torch.long, device=device)
        try:
            for epoch in range(1, args.epochs + 1):
                t0 = time.time()
                run.zero_()
                n_ok.zero_()
                for k, group in interleave([list(batches(t, args.batch_size, rng)) for t in trains]):
                    lr_now = lr_at(step, lr, warm, total)
                    for gp in opt.param_groups:
                        gp["lr"] = lr_now
                    b = make_batch(group, device, dtype)
                    rb = make_batch(next(rep_iter), device, dtype) if pt else None
                    try:
                        loss, n, parts = gradients(k, b, rb)
                    except Exception as exc:             # fall back to eager if the compile fails
                        if csteps is None:
                            raise
                        log(f"[compile] failed ({type(exc).__name__}: {exc}); eager from now on")
                        csteps = rstep = None
                        loss, n, parts = gradients(k, b, rb)
                    ok = (n > 0) & torch.isfinite(loss) & torch.isfinite(
                        torch.nn.utils.clip_grad_norm_(params, args.clip))
                    if fused:
                        opt.found_inf = (~ok).float()
                        opt.step()
                    elif ok:
                        opt.step()
                    ema.update()
                    step += 1
                    run += torch.where(ok, parts.detach(), 0.0)
                    n_ok += ok
                    seen_atoms += b["n_atoms"] + (rb["n_atoms"] if pt else 0)
                n_b = int(n_ok)
                terms = (run / max(n_b, 1)).tolist()
                t_train += time.time() - t0
                if n_b < spe:
                    log(f"[warn] epoch {epoch}: {spe - n_b} of {spe} steps skipped (no usable labels, "
                        f"or a non-finite loss or gradient)")
                with ema.applied():
                    accs = [errors(model, adapter.weights(k), gs, budget, device, dtype, stress)
                            for k, gs in enumerate(valids)]
                    vals = [a.result(w) if gs else {} for a, gs in zip(accs, valids)]
                    m = sum(accs[1:], accs[0]).result(w) if any(valids) else {}
                    drift = ev(rep_valid, adapter.weights(0), w_rep, found_stress) if rep_valid else {}
                    better = (m["loss"] < best) if m else epoch == args.epochs
                    if better:
                        best, best_epoch = m.get("loss", math.nan), epoch
                        best_sds = keep(epoch, vals)
                    if args.save_every > 0 and (epoch % args.save_every == 0 or epoch == args.epochs):
                        keep(epoch, vals, f"_ep{epoch:03d}")
                cell = lambda k, p=2: f"{m[k]:7.{p}f}" if k in m else f"{'-':>7}"
                log(f"{epoch:5d} {lr_now:9.2e} {sum(terms[:3]):11.4e} "
                    f"{m['loss'] if m else math.nan:11.4e} | {cell('e_mae')} {cell('e_rmse')} "
                    f"{cell('f_mae', 1)} {cell('f_rmse', 1)}"
                    + (f" | drift {drift['f_mae']:7.1f}" if drift else "")
                    + f" | {time.time() - t0:5.1f} s" + ("  *" if better and m else ""))
                row = dict(epoch=epoch, lr=lr_now, train_loss=sum(terms[:3]), seconds=time.time() - t0,
                           **dict(zip(("train_loss_e", "train_loss_f", "train_loss_s", "train_loss_replay"), terms)),
                           **{f"valid_{k}": v for k, v in m.items()},
                           **{f"drift_{k}": v for k, v in drift.items()})
                if hist is None:
                    hist = csv.DictWriter(hist_f, fieldnames=list(row))
                    hist.writeheader()
                hist.writerow(row)
                hist_f.flush()
                track(dict(train=dict(loss=row["train_loss"], lr=lr_now, seconds=row["seconds"]),
                           valid=m, drift=drift), epoch)
                if n_b == 0:
                    log(f"[stop] no finite loss in epoch {epoch}: the run has diverged")
                    break
        except KeyboardInterrupt:
            if member is not None:
                log("[stop] interrupted")
                raise
            log("[stop] interrupted: reporting the best epoch so far")
        hist_f.close()
        if compiled:                                     # frees the CUDA graphs before another fine-tune here
            csteps = rstep = None
            torch._dynamo.reset()
        if best_sds is None:
            with ema.applied():
                best_sds, best_epoch = keep(epoch, [{} for _ in trains]), epoch
        log.section(f"results: best epoch {best_epoch} (EMA weights)")
        log(f"trained {step:,} steps in {t_train:.0f} s ({seen_atoms / max(t_train, 1e-9):,.0f} atoms/s)")

    names = dict(model.named_parameters())
    finals = [{k: v.to(device) for k, v in sd.items() if k in names} for sd in best_sds]
    res = {n: ev(gs, finals[k]) for n, k, gs in sets if gs}
    before = "start" if scratch or args.init else "zero-shot"
    after = "trained" if scratch else "average" if average_of else "fine-tuned"
    log(table([r for n, _, gs in sets if gs for r in ((f"{n} {before}", len(gs), zero[n]),
                                                       (f"{n} {after}", len(gs), res[n]))]))
    drift = ev(rep_valid, finals[0], w_rep, found_stress) if rep_valid else {}
    if drift:
        log(f"drift      from the foundation on {len(rep_valid)} structures: force MAE {drift['f_mae']:.1f} meV/A, "
            f"energy MAE {drift['e_mae']:.1f} meV/atom"
            + (f", stress MAE {drift['s_mae']:.1f} meV/A^3" if "s_mae" in drift else ""))
    if average_of:
        log(wise_table([(d.name, s) for d, s in zip(average_of, members)], dict(fine_tuned=res, drift=drift)))
        save(out / "model.pt", best_sds[0], blob, meta)
    summary = dict(meta, n_trainable=n_train, n_network=n_net, lr=lr, steps=step, epochs_run=epoch,
                   best_epoch=best_epoch, train_seconds=t_train,
                   atoms_per_second=seen_atoms / max(t_train, 1e-9), split_valid=split,
                   n={n: len(gs) for n, _, gs in sets}, zero_shot=zero, fine_tuned=res, drift=drift)
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    track.summary(dict(zero_shot=zero, fine_tuned=res, drift=drift, best_epoch=best_epoch))
    track.close()
    log("saved      " + ", ".join(str(out / f) for f in files + [f for h, f in extra if h == "pt"]))
    log.fh.close()
    return summary


def wise_table(members, avg):
    """Errors of the members (name, summary), their mean and standard deviation, and the average's."""
    cols = [(n, q) for n in ("valid", "test") if n in avg["fine_tuned"] for q in ("e_mae", "f_mae")]
    cols += [("drift", "f_mae")] if avg["drift"] else []
    get = lambda s, n, q: ((s.get("drift") or {}) if n == "drift" else s["fine_tuned"].get(n, {})).get(q, math.nan)
    v = np.array([[get(s, n, q) for n, q in cols] for _, s in members]).reshape(len(members), len(cols))
    rows = [(name, f"{s['best_epoch']}/{s['epochs_run']}", x) for (name, s), x in zip(members, v)]
    rows += [("mean", "", v.mean(0)), ("std", "", v.std(0, ddof=1)),
             ("average", "", [get(avg, n, q) for n, q in cols])]
    head = f"{'':<12}{'best epoch':>11}" + "".join(f"{n} {q[0].upper()}".rjust(10) for n, q in cols)
    return "\n".join([head] + [f"{a:<12}{b:>11}" + "".join(f"{x:10.2f}" for x in c) for a, b, c in rows]
                     + ["(MAE: E meV/atom, F meV/A per component)"])


def average(paths):
    """The mean of the checkpoints' weights; a tensor that is the same in all of them is kept."""
    model, _ = load(paths[0], untrained=True)
    sds = [torch.load(p, map_location="cpu", weights_only=True)["model"] for p in paths]
    first = sds[0]
    for p, sd in zip(paths[1:], sds[1:]):
        if sd.keys() != first.keys() or any(sd[k].shape != v.shape for k, v in first.items()):
            raise SystemExit(f"{p} and {paths[0]} are not the same model")
    out = {}
    for k, v in first.items():
        vs = [sd[k] for sd in sds]
        if all(torch.equal(v, x) for x in vs[1:]):
            out[k] = v
        elif v.is_floating_point():
            out[k] = torch.stack(vs).double().mean(0).to(v.dtype)
        else:
            raise SystemExit(f"{k} differs between {paths[0]} and the others")
    return out


if __name__ == "__main__":
    main()
