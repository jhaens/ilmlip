"""Replay structures for finetune.py: random frames of MPtrj (downloaded) or of a structure file,
among those of the foundation's elements, unlabelled; a fine-tune labels them with its foundation."""
import argparse
import pickle
from pathlib import Path

import numpy as np

from . import runinfo

REPLAY_FILES = ("replay_set.extxyz", "drift_test_set.extxyz")

# MACE's MPtrj frames (MIT; MPtrj CC BY 4.0, Deng et al. 2023); only the structures are used
REPLAY_URLS = {"mptrj": "https://github.com/ACEsuit/mace-foundations/releases/download/mace_mp_0b/"
                        "mp_traj_combined.xyz"}


def download(url, log=print):
    """url into $XDG_CACHE_HOME/ilmlip, once."""
    import os
    import urllib.request
    cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "ilmlip"
    path = cache / url.rstrip("/").rsplit("/", 1)[-1]
    if not path.is_file():
        cache.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.name}.{os.getpid()}.part")     # processes starting together do not share it
        log(f"replay     downloading {url} into {cache} ...")
        try:
            urllib.request.urlretrieve(url, tmp)
        except OSError as exc:
            tmp.unlink(missing_ok=True)
            raise SystemExit(f"download failed ({exc}); run prepare-replay where there is internet")
        tmp.replace(path)
    return path


def _xyz_frames(path, n, rng, keep):
    """n random frames of an (ext)xyz file that keep accepts: one pass for the offsets, then only
    frames drawn among those not drawn yet, until n are kept or none are left."""
    import io
    from ase.io import read
    offs = []
    with open(path, "rb") as fh:
        while (head := fh.readline()).strip():
            offs.append(fh.tell() - len(head))
            for _ in range(int(head) + 1):
                fh.readline()
        out, left = [], np.arange(len(offs))
        while len(out) < n and len(left):
            pick = rng.choice(len(left), min(n - len(out), len(left)), replace=False)
            for k in np.sort(left[pick]):
                fh.seek(offs[k])
                head = fh.readline()
                body = b"".join(fh.readline() for _ in range(int(head) + 1))
                a = read(io.StringIO((head + body).decode()), format="extxyz")
                if keep(a):
                    out.append(a)
            left = np.delete(left, pick)
    return out


def prepare_replay(source, n, seed, out, log=print, species=None):
    """replay_set.extxyz (90 %) and drift_test_set.extxyz (10 %) of n random structures with
    only the elements in species (None: any), unlabelled: a fine-tune labels them with its foundation."""
    from ase import Atoms
    from ase.io import read, write
    rng = np.random.default_rng(seed)
    keep = lambda a: species is None or set(a.get_chemical_symbols()) <= set(species)
    if source in REPLAY_URLS or str(source).startswith(("http://", "https://", "file:")):
        source = download(REPLAY_URLS.get(source, source), log)
    path = Path(source)
    if path.suffix in (".xyz", ".extxyz"):
        frames = _xyz_frames(path, n, rng, keep)
    else:
        all_ = [a for a in read(path, index=":") if keep(a)]
        frames = [all_[i] for i in np.sort(rng.choice(len(all_), min(n, len(all_)), replace=False))]
    frames = [frames[i] for i in rng.permutation(len(frames))]
    clean = [Atoms(numbers=a.numbers, positions=a.positions, cell=a.cell, pbc=a.pbc,
                   info=dict(replay_source=str(source), replay_seed=seed)) for a in frames]
    if len(clean) < 10:
        only = "" if species is None else " with only the foundation's elements"
        hint = "--replay-n" if len(clean) == n else "all there are"
        raise SystemExit(f"replay: {len(clean)} structures{only} in {source} ({hint}), at least 10 needed")
    k, out = len(clean) // 10, Path(out)
    out.mkdir(parents=True, exist_ok=True)
    write(out / REPLAY_FILES[1], clean[:k], format="extxyz")
    write(out / REPLAY_FILES[0], clean[k:], format="extxyz")
    log(f"replay     {len(clean) - k} + {k} structures from {source} (seed {seed}) -> {out}")
    return out


def find_replay(arg, pt, n, seed, out, log, species=None):
    """The folder with replay_set/drift_test_set.extxyz, or None; pt: a replay head trains on
    them, so MPtrj is the default.  Drawn structures have only the elements in species."""
    ready = lambda d: all((Path(d) / f).is_file() for f in REPLAY_FILES)
    if arg == "none":
        return None
    if arg is None:
        arg = Path.cwd() if ready(Path.cwd()) else ("mptrj" if pt else None)
        if arg is None:
            log("replay     none, so no drift (prepare-replay, or --replay mptrj)")
            return None
    if Path(arg).is_dir() and ready(arg):
        log(f"replay     {Path(arg).resolve()}")
        return Path(arg)
    return prepare_replay(arg, n, seed, out, log, species)


def trained_elements(species, e0):
    """The elements with a reference energy, those a model was trained on (a checkpoint lists every element)."""
    return [s for s, e in zip(species, np.asarray(e0, float).reshape(-1)) if e != 0]


def species_of(foundation):
    """The elements a foundation was trained on: a pretrained model's name or a checkpoint file."""
    import torch
    from . import pretrained
    try:
        ck = torch.load(pretrained.path(foundation), map_location="cpu", weights_only=True)
        return trained_elements(ck["species"], ck["model"]["atomic_energies"])
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, pickle.UnpicklingError) as exc:
        raise SystemExit(f"--foundation {foundation}: not a model checkpoint ({type(exc).__name__}: {exc})")


def prepare_cli(argv):
    p = argparse.ArgumentParser(prog="ilmlip.finetune prepare-replay",
                                description="Write replay_set.extxyz (90 %) and "
                                            "drift_test_set.extxyz (10 %).")
    p.add_argument("--source", default="mptrj", help="mptrj (download), a URL or a structure file")
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=".")
    p.add_argument("--foundation", help="only structures of this model's elements: a pretrained model or a file")
    a = p.parse_args(argv)
    runinfo.show()
    return prepare_replay(a.source, a.n, a.seed, a.out, species=species_of(a.foundation) if a.foundation else None)
