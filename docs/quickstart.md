# Quick start

What do you want to do? Each card jumps to a short recipe; most have a ready script in `examples/`.

<div class="cards">
<a class="card" href="#molecular-dynamics"><b>Run MD</b><span>a pretrained model in ASE</span></a>
<a class="card" href="#fine-tuning"><b>Fine-tune</b><span>adapt a model to your own data</span></a>
<a class="card" href="#average-fine-tunes-wise-ft"><b>Average fine-tunes</b><span>WiSE-FT: several runs, one model</span></a>
<a class="card" href="#hessians-and-vibrations"><b>Vibrations</b><span>the analytic Hessian</span></a>
<a class="card" href="#train-on-your-own-data"><b>Train from scratch</b><span>pack a dataset, train, evaluate</span></a>
<a class="card" href="#which-engine"><b>Choose GPU or CPU</b><span>which engine for which job</span></a>
<a class="card" href="#many-small-systems-in-one-call"><b>Many small systems</b><span>many MDs or beads in one GPU call</span></a>
<a class="card" href="#millions-of-atoms"><b>Millions of atoms</b><span>chunks on one GPU, LAMMPS on many</span></a>
<a class="card" href="#for-coding-agents"><b>For coding agents</b><span>instructions to paste into an LLM agent</span></a>
</div>

## Install

```sh
pip install ilmlip
pip install "jax[cuda12]>=0.11.2,<0.12"  # MD on the GPU
```

Optional packages: [reference](reference.html#install).

## Molecular dynamics

```python
from ase import units
from ase.io import read
from ase.md.nose_hoover_chain import NoseHooverChainNVT
from ilmlip import load_calculator

atoms = read("train.xyz")
atoms.calc = load_calculator("OAM-S-omat")     # or a fine-tuned "naive/model.pt"
NoseHooverChainNVT(atoms, 0.5 * units.fs, temperature_K=300, tdamp=50 * units.fs,
                   trajectory="md.traj").run(1000)
```

Any ASE code can use the calculator. Which model: [models](models.html#which-one); other
ensembles: [command builder](builder.html).

> **Example** `examples/lisi/md.py`: Li13Si4 at 300 K. Laptop CPU, a minute.

## Fine-tuning

```sh
cd examples/lisi
python -m ilmlip.finetune --foundation OAM-S-omat --train train.xyz --test test.xyz \
    --out naive
```

`naive/model.pt` loads in every calculator. `--mode lora`, `multihead` or `scratch` train in other
ways ([finetune](reference.html#finetune)).

> **Example** `examples/lisi/finetune.sh`: all four modes on 10 frames; the naive fine-tune halves
> the force error on `test.xyz`. Laptop CPU, two minutes each.

### Average fine-tunes (WiSE-FT)

```sh
python -m ilmlip.finetune --foundation OAM-S-omat --train train.xyz --test test.xyz \
    --wise 4 --wise-parallel 2
```

Four fine-tunes that differ only in the order of the batches, two at a time; `model.pt` is the
average of their weights. Compare the test errors in the logs with a single run.

## Hessians and vibrations

```python
calc = load_calculator("OMOL-S", engine="jax")     # or engine="torch"
H = calc.get_hessian(atoms)                         # (3N, 3N), eV/Å²
```

Analytic second derivatives ([get_hessian](reference.html#get_hessian)).

> **Example** `examples/hessian/hessian_nh3.py`: the modes of ammonia at its minimum and at the
> transition state of its inversion, and the barrier. Laptop CPU, seconds.

## Train on your own data

```sh
python -m ilmlip.data dataset/ train.xyz --val val.xyz                         # pack once
python -m ilmlip.train --packed dataset/ --hparams S --epochs 100 --out runs/S  # GPU
python -m ilmlip.evaluate runs/S/best.pt dataset/                               # errors
```

Options: [data](reference.html#data), [train](reference.html#train), [evaluate](reference.html#evaluate).

## Running fast

### Which engine

| you run | use |
|---|---|
| long MD runs | a GPU: `load_calculator(model, engine="jax", kernels="fused")` |
| short MD or single points of small systems (tens of atoms) | the CPU: one independent process per core, each with `load_calculator(model, device="cpu", threads=1)`, e.g. one per structure or temperature; at ~50 atoms one core reaches about a third of the speed of a B200 |
| many small MDs or relaxations | one model on a GPU computes the forces of all of them at every step: [lockstep](#many-mds-on-one-gpu); a few MDs run faster as processes with a model each |
| thousands of single points | one model on a GPU computes them in batches: [BatchedModel](#many-structures-in-one-call) |
| path integrals | one model on a GPU computes all beads of a step at once: [i-PI](#path-integrals-with-i-pi) |
| millions of atoms | one GPU in [chunks](#millions-of-atoms), several GPUs with [LAMMPS](reference.html#lammps) |

All calculator options: [load_calculator](reference.html#load_calculator).

### Many small systems in one call

A call on the GPU has a fixed cost: on an RTX A5000, S takes 0.36 ms + 0.25 µs per atom. Independent
systems evaluated together pay it once. 32 structures in one call, against one after the other:

| atoms per structure | 64 | 512 | 4096 |
|---|---|---|---|
| speed-up (S) | 6.1x | 2.7x | none |

### Many MDs on one GPU

```python
import numpy as np
from ase import units
from ase.io import read
from ase.md.bussi import Bussi
from ilmlip.lockstep import imap

def md(calc, T):                  # any ASE code with this calculator, at module level
    atoms = read("water.xyz")
    atoms.calc = calc
    Bussi(atoms, 0.5 * units.fs, temperature_K=T, taut=100 * units.fs,
          rng=np.random.default_rng()).run(20000)
    return T

if __name__ == "__main__":        # the processes import this file
    for T in imap("OAM-S-omat", md, [300, 350, 400]):
        print(T, "done")
```

Pass `rng=np.random.default_rng()`: ASE's default legacy NumPy generator costs about as much per step
as the model itself.

Each item runs in a process of its own with its own calculator; the calling process holds the
model and computes the forces of all MDs waiting for them in one call. This pays for many small
MDs: 16 water MDs of 96 atoms on an RTX A5000 run 2x faster than 16 processes with a model each
and 4x faster than one after the other; a few MDs run fastest as processes of their own.

> **Example** `examples/lockstep/md.py`: five water and KOH MDs on one GPU.

### Many structures in one call

```python
from ilmlip.batched import BatchedModel
results = BatchedModel("OAM-S-omat").compute(structures)   # a dict per structure
```

### Path integrals with i-PI

With `<batch_size>` (i-PI 3.3 and newer), the ILMlip driver computes all beads of a step in one
call: 32 beads about 7x faster on an A5000.

> **Example** `examples/ipi/`: 64 water molecules, 32 beads; `examples/README.md` explains `pimd.xml`. GPU.

### Millions of atoms

- **One GPU:** above 100,000 atoms, when one call would not fit, the system is split into chunks
  with a halo, with the same result (`chunk_atoms`, fused kernels only). S runs 10 million atoms on
  an A100 80GB.
- **Several GPUs:** LAMMPS with its domain decomposition ([lammps](reference.html#lammps)).

## For coding agents

An LLM agent that writes ILMlip scripts tends to get some facts wrong, because they differ from typical MLIP, MD and ASE habits: model names, charge and spin, which engine runs, the defaults of fine-tuning and training. Paste this block into the instructions file of your agent (`CLAUDE.md`, `AGENTS.md`, ...) and it uses these instead of general knowledge. The [reference](reference.html) has everything else.

```text
ILMlip: facts that differ from typical MLIP/MD conventions; prefer them.

MODELS
- load_calculator("OAM-S-omat"): pretrained names, case-insensitive;
  python -m ilmlip.pretrained lists them. Do not invent names. Only
  OAM-S-omat is in umlip/ of a clone, the others download on first use
  (offline: python -m ilmlip.pretrained NAME beforehand).
- Sizes XS..XXL, S is a good start. OAM-<size>-omat: inorganic MD.
  OAM-<size>-mpa: Materials Project energies (phase diagrams, Matbench).
  MATPES-{PBE,r2SCAN}-<size>. OMOL-<size> (XS-XL): molecules.
  SOLIDACID-<size> (S-L).
- Same cost, often better (average of ten runs): OAM-L-mpa-WiSE10,
  OAM-L-omat-WiSE10, OAM-XXL-mpa-WiSE10.
- Uncertainty: OAM-L-mpa-committee, OAM-L-omat-committee (5x cost);
  calc.results gets energy_std, forces_std, force_dev (per atom),
  max_force_dev, member_energies; calc.members are the members.
- OMOL-* only: load_calculator("OMOL-S", charge=1, spin=2), fixed for
  the calculator; spin is the MULTIPLICITY 2S+1 (default 1);
  atoms.info["charge"/"spin"] must match or be absent. Other models
  raise ValueError when charge or spin is given at all (also 0, 1).
- No stress: OMOL-*, SOLIDACID-* (get_stress raises, no NPT); a fine-
  tune has stress if its foundation or its data had it.
- Langevin, Bussi, Andersen: pass rng=np.random.default_rng() (ASE's
  legacy generator costs as much per step as the model).

ENGINES
- Default: the first that runs of JAX on a GPU, PyTorch on a GPU, C++
  on the CPU, JAX on the CPU, PyTorch on the CPU. device="cpu": C++,
  else JAX, else PyTorch. JAX needs "jax[cuda12]" for the GPU; PyTorch
  CUDA alone gives the slow torch engine.
- Fallbacks (GPU -> CPU, engine -> next, fused -> xla kernels) warn,
  except the default without a GPU; check calc.engine, .device,
  .kernels. Options the engine lacks are dropped with a warning;
  engine="cpp" with a GPU device raises.
- JAX compiles at the first call and per new padded size: time MD after
  a warm-up. dtype defaults to TF32: finite differences need "float64".
- Tens of atoms: one process per CPU core, device="cpu", threads=1.
  Many small systems on a GPU: ilmlip.lockstep.imap (task at module
  level; not for a few MDs) or ilmlip.batched.BatchedModel(...)
  .compute(atoms_list).

FINE-TUNING AND TRAINING
- python -m ilmlip.finetune --foundation NAME_OR_FILE --train a.xyz
  [--out DIR]. Defaults: --mode naive (lora, multihead, scratch),
  --optimizer adam, --epochs 200, --w-forces 10, 10 % of --train held
  out, --e0s estimated (per-element correction fitted to your energies).
- DIR/model.pt (best epoch, EMA weights) loads in every calculator; DIR
  defaults to <train>_<foundation>_<mode>. No resume: a DIR with
  model.pt is refused, --init continues one. --wise N (naive only)
  averages N members; rerun to finish.
- --mode multihead downloads ~600 MB of MPtrj unless --replay is a local
  path or replay_set.extxyz + drift_test_set.extxyz are in the cwd.
- ilmlip.train reads packed data only: python -m ilmlip.data DIR
  train.xyz --val val.xyz (energy and forces on every frame), then
  python -m ilmlip.train --packed DIR --hparams S --epochs N --out RUN
  (without --epochs: --steps 1000000). The same command RESUMES from
  RUN/checkpoint.pt; a fresh run needs a new --out. Best: RUN/best.pt.

I-PI AND LAMMPS
- i-PI >= 3.3: copy examples/ipi/driver.py (never name it ipi.py),
  pbc="False", <batch_size> = beads; a model without stress needs
  stress=False in the driver parameters.
- LAMMPS: python -m ilmlip.lammps CHECKPOINT.pt model.mliap takes a
  file, not a name (python -m ilmlip.pretrained NAME prints it); charge
  and spin are fixed there (--charge, --spin); several GPUs for one run
  only through LAMMPS.
```
