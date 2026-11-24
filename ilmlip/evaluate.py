"""Errors of checkpoints on a packed split, overall and per source.

    python -m ilmlip.evaluate best.pt data/                    # val/
    python -m ilmlip.evaluate best.pt S_seed{1,2,3,4}/best.pt data/ --dump c.npz

Several checkpoints form a committee: each member's errors, those of the mean, and the spread
(energy std per structure, force deviation per atom).  A model trained with --ref-sources gets its
per-source energy offsets, unless --no-shift.  A model fine-tuned on some sources is on their
energy scale only, so --sources keeps just those.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from . import runinfo
from . import vcut
from .data import PackedLoader, PackedSplit, check_charge_spin
from .model import load
from .train import ema_weights

# the OMol25 leaderboard's S2EF subsets
OMOL_SUBSETS = {"biomolecules": ("biomolecules",), "electrolytes": ("elytes",),
                "metal_complexes": ("metal_complexes",),
                "neutral_organics": ("ani2x", "orbnet_denali", "geom_orca6", "trans1x", "rgd")}


def load_member(path, args, names, dt):
    model, ck = load(path, args.device)
    if args.ema:
        if not ck.get("ema"):
            raise SystemExit(f"{path}: no EMA weights stored")
        sd = model.state_dict()
        for k, v in ema_weights(ck, model).items():
            if k in sd:
                sd[k].copy_(v.to(sd[k].dtype))
    model = model.to(args.device, dt).eval()
    offsets = None
    if ck.get("src_shift") is not None and ck.get("ref_sources") and not args.no_shift:
        offsets = (torch.as_tensor(ck["src_shift"]).to(args.device, torch.float64),
                   torch.tensor([n not in ck["ref_sources"] for n in names], device=args.device))
    return model, offsets


def predict(member, b, stress):
    model, offsets = member
    out = model(b, compute_stress=stress)
    e = out["energy"].detach().double()
    if offsets is not None:
        shift, other = offsets
        e = e + torch.zeros_like(e).index_add(0, b["batch"],
                                              torch.where(other[b["src"]][b["batch"]], shift[b["mu"]], 0.0))
    return e, out["forces"].detach().double(), out["stress"].detach().double() if stress else None


class Errors:
    """Error sums per source."""

    def __init__(self, n_src):
        self.n_src, self.acc = n_src, {}

    def _add(self, key, index, val):
        t = torch.zeros(self.n_src, dtype=torch.float64, device=val.device).index_add_(0, index, val.double())
        self.acc[key] = self.acc.get(key, 0) + t.cpu().numpy()

    def add(self, b, e, f, s):
        de, df = e - b["energy"].double(), f - b["forces"].double()
        n_at, src = b["n_atoms_per_struct"].double(), b["src"]
        rows = dict(e_abs=de.abs() / n_at, e_sq=(de / n_at) ** 2, es_abs=de.abs(), n=torch.ones_like(de))
        if s is not None:
            ok = torch.isfinite(b["stress"]).all(1) & (b["volume"] > 0)
            ds = torch.where(ok[:, None], s - b["stress"].double(), 0)
            rows.update(s_abs=ds.abs().sum(1), s_sq=(ds ** 2).sum(1), s_n=6.0 * ok)
        for key, val in rows.items():
            self._add(key, src, val)
        for key, val in dict(f_abs=df.abs().sum(1), f_sq=(df ** 2).sum(1),
                             f_n=torch.full_like(df[:, 0], 3.0)).items():
            self._add(key, src[b["batch"]], val)

    def metrics(self, sel):
        s = {k: float(v[sel].sum()) for k, v in self.acc.items()}
        r = dict(n_structures=int(s["n"]), e_mae=1e3 * s["e_abs"] / s["n"],
                 e_rmse=1e3 * (s["e_sq"] / s["n"]) ** 0.5, e_mae_structure=1e3 * s["es_abs"] / s["n"],
                 f_mae=1e3 * s["f_abs"] / s["f_n"], f_rmse=1e3 * (s["f_sq"] / s["f_n"]) ** 0.5)
        if s.get("s_n"):
            r.update(s_mae=1e3 * s["s_abs"] / s["s_n"], s_rmse=1e3 * (s["s_sq"] / s["s_n"]) ** 0.5)
        return r

    def table(self, names):
        res = {"all": self.metrics(slice(None))}
        for k, name in enumerate(names):
            if len(names) > 1 and self.acc["n"][k]:
                res[name] = self.metrics([k])
        return res


class Spread:
    """A committee's spread against its error: means, Pearson r, per-structure values."""

    KEYS = ("src", "n_atoms", "e_err", "e_std", "f_err", "f_dev", "f_dev_max")

    def __init__(self, keep):
        self.sums, self.keep = {}, ({k: [] for k in self.KEYS} if keep else None)

    def _pair(self, name, x, y):
        for key, v in (("n", torch.ones_like(x)), ("x", x), ("y", y),
                       ("xx", x * x), ("yy", y * y), ("xy", x * y)):
            self.sums[f"{name}_{key}"] = self.sums.get(f"{name}_{key}", 0.0) + float(v.sum())

    def add(self, b, E, F):
        n_at = b["n_atoms_per_struct"].double()
        e_std = E.std(0, unbiased=False) / n_at
        e_err = (E.mean(0) - b["energy"].double()).abs() / n_at
        f_dev = ((F - F.mean(0)) ** 2).sum(-1).mean(0).sqrt()
        f_err = (F.mean(0) - b["forces"].double()).norm(dim=1)
        self._pair("e", e_std, e_err)
        self._pair("f", f_dev, f_err)
        if self.keep is not None:
            B, bt = len(n_at), b["batch"]
            per = lambda v: torch.zeros(B, dtype=v.dtype, device=v.device).index_add(0, bt, v) / n_at
            vmax = torch.zeros(B, dtype=f_dev.dtype, device=f_dev.device).scatter_reduce(
                0, bt, f_dev, "amax", include_self=False)
            for key, v in zip(self.KEYS, (b["src"], n_at, e_err, e_std, per(f_err), per(f_dev), vmax)):
                self.keep[key].append(v.cpu().numpy())

    def result(self):
        out = {}
        for name, unit in (("e", "meV/atom, per structure"), ("f", "meV/A, per atom")):
            s = {k[2:]: v for k, v in self.sums.items() if k.startswith(name + "_")}
            n = s["n"]
            cov = n * s["xy"] - s["x"] * s["y"]
            var = (n * s["xx"] - s["x"] ** 2) * (n * s["yy"] - s["y"] ** 2)
            out[name] = dict(of=unit, mean_spread=1e3 * s["x"] / n, mean_error=1e3 * s["y"] / n,
                             pearson_r=cov / var ** 0.5 if var > 0 else float("nan"))
        return out


def omol_leaderboard(err, names):
    """The leaderboard's columns for OMol25: energy MAE per structure, force MAE, per subset."""
    if not set(names) & set(OMOL_SUBSETS["neutral_organics"]):
        return None
    rows = {"all": err.metrics(slice(None))}
    for sub, domains in OMOL_SUBSETS.items():
        idx = [names.index(d) for d in domains if d in names]
        if idx and err.acc["n"][idx].sum():
            rows[sub] = err.metrics(idx)
    return {k: dict(energy_mae_meV=r["e_mae_structure"], forces_mae_meV_A=r["f_mae"],
                    n_structures=r["n_structures"]) for k, r in rows.items()}


def row(name, r):
    return (f"  {name:18s} n {r['n_structures']:>8,}  E {r['e_mae']:8.2f} (RMSE {r['e_rmse']:8.2f})"
            f"  E/struct {r['e_mae_structure']:9.1f}  F {r['f_mae']:7.1f} (RMSE {r['f_rmse']:7.1f})"
            + (f"  S {r['s_mae']:6.2f}" if "s_mae" in r else ""))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint", nargs="+", help="one checkpoint, or several: a committee")
    p.add_argument("data", help="packed dataset directory")
    p.add_argument("--split", default="val")
    p.add_argument("--sources", help="only the structures of these sources, e.g. 'mptrj,sAlex_train'")
    p.add_argument("--ema", action="store_true", help="the checkpoints' EMA weights")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    p.add_argument("--batch-atoms", type=int, default=4096)
    p.add_argument("--out", help="write the numbers as JSON")
    p.add_argument("--dump", help="committee: per-structure errors and spreads as .npz")
    p.add_argument("--no-shift", action="store_true", help="without the per-source energy offsets")
    args = p.parse_args(argv)
    runinfo.show()

    dt = getattr(torch, args.dtype)
    split = PackedSplit(Path(args.data) / args.split)
    names = split.sources or ["all"]
    indices = None
    if args.sources:
        want = args.sources.split(",")
        if set(want) - set(names):
            raise SystemExit(f"--sources: unknown {sorted(set(want) - set(names))}; have {names}")
        indices = np.where(np.isin(split.src, [names.index(s) for s in want]))[0]
        if not len(indices):
            raise SystemExit(f"--sources: no structures of {want} in {split.path}")
    members =[load_member(c, args, names, dt) for c in args.checkpoint]
    rcut = {float(m.rcut) for m, _ in members}
    if len(rcut) > 1:
        raise SystemExit(f"the checkpoints have different cutoffs {sorted(rcut)}")
    vcut.same([m for m, _ in members])
    loader = PackedLoader(split, rcut.pop(), device=args.device, dtype=dt,
                          pair_rcut=vcut.by_z(members[0][0]),
                          batch_atoms=max(args.batch_atoms, int(split.n_atoms.max())),
                          batch_structs=1 << 20, shuffle=False, indices=indices)
    for m, _ in members:
        if m.charge_spin:
            check_charge_spin(split, loader.indices, m.max_charge, m.max_spin)
    stress, K = split.any_stress and all(m.stress for m, _ in members), len(members)
    errs = [Errors(len(names)) for _ in range(K + (K > 1))]
    spread = Spread(keep=bool(args.dump)) if K > 1 else None
    for b in loader:
        preds = [predict(m, b, stress) for m in members]
        for err, (e, f, s) in zip(errs, preds):
            err.add(b, e, f, s)
        if K > 1:
            E, F = torch.stack([x[0] for x in preds]), torch.stack([x[1] for x in preds])
            S = torch.stack([x[2] for x in preds]).mean(0) if stress else None
            errs[-1].add(b, E.mean(0), F.mean(0), S)
            spread.add(b, E, F)
    loader.check_counts()

    print(f"{split.path}: E meV/atom, E/struct meV, F meV/A" + (", S meV/A^3" if stress else ""))
    if K == 1:
        res = errs[0].table(names)
        for name, r in res.items():
            print(row(name, r))
        out = dict(checkpoint=args.checkpoint[0], ema=args.ema, data=str(split.path), metrics=res)
    else:
        singles = [e.metrics(slice(None)) for e in errs[:K]]
        for c, r in zip(args.checkpoint, singles):
            print(row(Path(c).parent.name, r))
        res = errs[-1].table(names)
        for name, r in res.items():
            print(row(f"mean {name}", r))
        sp = spread.result()
        for name, r in sp.items():
            print(f"  spread {name}: mean {r['mean_spread']:.2f}, error {r['mean_error']:.2f} ({r['of']}), "
                  f"Pearson r {r['pearson_r']:.3f}")
        out = dict(checkpoints=args.checkpoint, ema=args.ema, data=str(split.path), members=singles,
                   metrics=res, spread=sp)
        if args.dump:
            np.savez_compressed(args.dump, **{k: np.concatenate(v) for k, v in spread.keep.items()})
    if args.sources:
        out["sources"] = want
    lb = omol_leaderboard(errs[-1], names)
    if lb:
        out["omol_leaderboard"] = lb
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    main()
