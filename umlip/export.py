"""Write a training or fine-tuning checkpoint as a model file for release.

    python export.py runs/S_omat/best.pt OAM-S-omat DIR --about "..."

DIR/OAM-S-omat.pt keeps what the calculators and fine-tuning read (weights, architecture,
species, reference energies), the precision it was trained in and the validation errors; no
optimiser state, no other training arguments.  DIR/models.json gets the model's entry for
ilmlip.pretrained, whose names are the only ones accepted (committee members such as
OAM-L-omat-seed2 included).
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from ilmlip import load, pretrained
from ilmlip.model import trained_dtype

KEEP = ("model", "hparams", "species", "src_shift", "ref_sources", "metrics")


def main():
    p = argparse.ArgumentParser(description="Write a checkpoint as a model file for release.")
    p.add_argument("checkpoint")
    p.add_argument("name", help="a model of ilmlip.pretrained or a committee member, e.g. OAM-S-omat")
    p.add_argument("dir", type=Path, help="the folder of the model files and models.json")
    p.add_argument("--about", default="", help="one line for the entry in models.json")
    args = p.parse_args()
    name = pretrained.canonical(args.name)
    if name not in pretrained.FILES:
        p.error(f"{args.name!r} is not a model file of ilmlip.pretrained: {', '.join(pretrained.MODELS)} "
                f"or a committee member (<model>-seed1 ... <model>-seed4)")

    model, ck = load(args.checkpoint)
    blob = {k: ck[k] for k in KEEP if ck.get(k) is not None}
    blob["e0"] = blob["model"]["atomic_energies"].tolist()
    blob["args"] = dict(dtype=trained_dtype(ck))
    args.dir.mkdir(parents=True, exist_ok=True)
    dest = args.dir / f"{name}.pt"
    tmp = dest.with_name(dest.name + ".tmp")
    torch.save(blob, tmp)
    torch.load(tmp, map_location="cpu", weights_only=True)
    load(tmp)
    tmp.replace(dest)

    data = dest.read_bytes()
    index = args.dir / "models.json"
    models = json.loads(index.read_text()) if index.exists() else {}
    models[name] = dict(file=dest.name, sha256=hashlib.sha256(data).hexdigest(),
                        bytes=len(data), about=args.about)
    index.write_text(json.dumps(models, indent=2) + "\n")
    print(f"{dest}: {len(data) / 1e6:.2f} MB; entry in {index}")


if __name__ == "__main__":
    main()
