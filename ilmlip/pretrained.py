"""The pretrained ilmlip models by name.

    from ilmlip.pretrained import names, path
    names()             # OMOL-S, ..., OAM-S-omat, ..., OAM-L-omat-committee, ...
    path("OAM-S-omat")  # the checkpoint file

    python -m ilmlip.pretrained                # list the names
    python -m ilmlip.pretrained OAM-XL-omat    # download these into the cache, or: all

A name stands for the file <name>.pt in the folders listed in ILMLIP_MODELS (os.pathsep-separated),
then in umlip/ of this repository (it holds OAM-S-omat), then in ~/.cache/ilmlip/models.  A model in
none of them is downloaded into the cache on first use, from ILMLIP_REPO (default REPO): a raw-file
URL or a local folder with the model files and models.json, as umlip/export.py writes them.

Names are not case sensitive, and "_" may stand for "-".  OMOL is OMol25 and takes the total charge
and spin multiplicity as inputs; MATPES is MatPES at the PBE or r2SCAN level; OAM is OMat24, sAlex
and MPtrj, then -mpa (MPtrj and sAlex) or -omat (OMat24).  A committee is a model and four more of
its kind, <name>-seed1 ... <name>-seed4, that differ only in their random seed
(calculator.CommitteeCalculator reports their spread).
"""
import argparse
import hashlib
import json
import os
import sys
import urllib.request
from pathlib import Path

REPO = "https://github.com/jhaens/ilmlip-models/releases/download/v1.0"
BUNDLED = Path(__file__).resolve().parent.parent / "umlip"
MODELS = ["OMOL-XS", "OMOL-S", "OMOL-M", "OMOL-L", "OMOL-XL", "SOLIDACID-S", "SOLIDACID-M", "SOLIDACID-L",
          *(f"MATPES-{level}-{size}" for level in ("PBE", "r2SCAN") for size in ("XS", "S", "M", "L", "XL", "XXL")),
          *(f"OAM-{size}-{stage}" for stage in ("mpa", "omat") for size in ("XS", "S", "M", "L", "XL", "XXL")),
          "OAM-L-mpa-WiSE10", "OAM-L-omat-WiSE10", "OAM-XXL-mpa-WiSE10", "OAM-S30-mpa"]

COMMITTEES = {f"{m}-committee": [m] + [f"{m}-seed{k}" for k in range(1, 5)]
              for m in ("OAM-L-mpa", "OAM-L-omat")}
FILES = MODELS + [f for ms in COMMITTEES.values() for f in ms[1:]]
_KNOWN = {n.lower(): n for n in FILES + list(COMMITTEES)}

oam_omat_model_aliases = {
    "XS": ["pippin", "r2d2", "dandelion"],
    "S": ["frodo", "hansolo", "triss"],
    "M": ["aragorn", "chewbacca", "vesemir"],
    "L": ["legolas", "luke", "yennefer"],
    "XL": ["gandalf", "obiwan", "geralt"],
    "XXL": ["bombadil", "yoda", "ciri"],
}
_ALIAS = {a: _KNOWN[f"oam-{size.lower()}-omat"] for size, aliases in oam_omat_model_aliases.items() for a in aliases}


def names():
    """The pretrained models and committees."""
    return MODELS + list(COMMITTEES)


def canonical(name):
    """The registered spelling of a model name, or None."""
    key = str(name).strip().lower().replace("_", "-")
    return _KNOWN.get(key) or _ALIAS.get(key)


def folders():
    """Where the checkpoint files are looked for, in this order; downloads go to the last."""
    env = [Path(p).expanduser() for p in os.environ.get("ILMLIP_MODELS", "").split(os.pathsep) if p]
    return env + [BUNDLED, Path.home() / ".cache" / "ilmlip" / "models"]


def members(name):
    """The member names of a committee, or None for a single model."""
    return list(COMMITTEES[canonical(name)]) if canonical(name) in COMMITTEES else None


def find(name):
    """The file of a model (registered spelling) in the folders, or None."""
    for d in folders():
        for f in (d / f"{name}.pt", d / f"{name.lower()}.pt"):
            if f.is_file():
                return f


def open_source(repo, file):
    if "://" in repo:
        return urllib.request.urlopen(f"{repo.rstrip('/')}/{file}", timeout=60)
    return open(Path(repo) / file, "rb")


def download(repo, entry, dest):
    with open_source(repo, entry["file"]) as src:
        data = src.read()
    if hashlib.sha256(data).hexdigest() != entry["sha256"]:
        raise ValueError(f"{entry['file']}: the sha256 does not match models.json, not saved")
    tmp = dest.with_name(f"{dest.name}.{os.getpid()}.part")     # processes starting together do not share it
    try:
        tmp.write_bytes(data)
        tmp.replace(dest)
    finally:
        tmp.unlink(missing_ok=True)


def fetch(name):
    """Download the file of a model (registered spelling) from the model repository into the cache."""
    repo = os.environ.get("ILMLIP_REPO") or REPO
    if not repo:
        raise FileNotFoundError(f"{name}.pt is in none of {', '.join(str(d) for d in folders())}; put it "
                                f"there, set ILMLIP_MODELS to the folder that holds it, or set "
                                f"ILMLIP_REPO to the model repository to download it from")
    with open_source(repo, "models.json") as f:
        entry = json.load(f).get(name)
    if entry is None:
        raise FileNotFoundError(f"{name} is not in the models.json of {repo}")
    dest = folders()[-1] / f"{name}.pt"
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {name} ({entry['bytes'] / 1e6:.1f} MB) from {repo}", file=sys.stderr, flush=True)
    download(repo, entry, dest)
    return dest


def path(model):
    """The checkpoint file of a model name (or of an existing path); a known model that is in
    none of the folders is downloaded into the cache."""
    if Path(model).expanduser().is_file():
        return Path(model).expanduser()
    name = canonical(model)
    if name is None:
        raise ValueError(f"{model!r} is neither a checkpoint file nor a pretrained model "
                         f"({', '.join(names())})")
    if name in COMMITTEES:
        raise ValueError(f"{name} is a committee of {', '.join(COMMITTEES[name])}: one file per member")
    return find(name) or fetch(name)


def main():
    p = argparse.ArgumentParser(description="List the pretrained models, or download them into the cache.")
    p.add_argument("names", nargs="*", help="models or committees to download, or all; none lists them")
    args = p.parse_args()
    if not args.names:
        for n in names():
            print(f"{n:<22} {'present' if all(find(m) for m in members(n) or [n]) else ''}")
        return
    for n in FILES if args.names == ["all"] else args.names:
        for m in members(n) or [n]:
            print(path(m))


if __name__ == "__main__":
    main()
