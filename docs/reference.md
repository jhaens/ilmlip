# API and CLI

## Install

Python 3.12 or newer.

```sh
pip install ilmlip                # model, training, fine-tuning (PyTorch)
pip install "jax[cuda12]>=0.11.2,<0.12"  # molecular dynamics on the GPU
pip install "ilmlip[fused]"       # optional: cuEquivariance kernels for MD and training
pip install "ilmlip[wandb]"       # optional: live tracking (or "ilmlip[mlflow]")
pip install "ilmlip[d3]"          # optional: the D3 dispersion correction (LGPL-3.0)
```

The fused kernels use NVIDIA's cuEquivariance: its Python packages are Apache-2.0, the CUDA
kernels (`cuequivariance-ops-*-cu12`) a separately licensed NVIDIA binary. Without them the same model
runs on plain XLA (JAX) or PyTorch operations.

## Python

### load_calculator

```python
from ilmlip import load_calculator

atoms.calc = load_calculator("OAM-S-omat")                # JAX on a GPU, C++ on a CPU
atoms.calc = load_calculator("OMOL-S", charge=1, spin=2)  # charge and spin multiplicity
atoms.calc = load_calculator("model.pt", device="cpu", engine="jax")
atoms.calc = load_calculator("OAM-L-omat-committee")      # mean and spread of five
atoms.calc = load_calculator("OAM-S-omat", dispersion="d3")   # plus Grimme's D3 (pip install dftd3)
```

| argument | description | default |
|---|---|---|
| `model` | a [pretrained model](models.html) by name, a checkpoint file, an `ILMlip` module, or a list of them (a committee) | `"OAM-S-omat"` |
| `device` | `"auto"`, `"cuda"` (also `"gpu"`, `"cuda:1"`) or `"cpu"` | `"auto"`: the GPU when the engine can use one |
| `engine` | `"jax"`, `"cpp"` (the C++ engine, CPU only), `"torch"` (the training model itself: slow, for checks) or `"auto"` | `"auto"`: JAX on a GPU, C++ on a CPU |
| `kernels` | JAX on a GPU: `"fused"` (cuEquivariance), `"xla"`, or `"auto"` (fused when installed) | `"auto"` |
| `dtype` | `"tf32"`, `"float32"`, `"float64"` or `"auto"` ([precision](#precision)) | `"auto"`: the precision the model was trained in |
| `charge`, `spin` | total charge and spin multiplicity 2S+1, for models that take them (OMOL); fixed for the calculator's lifetime. Other models raise `ValueError` when either is given, even `0` or `1` | OMOL: `0`, `1` |
| `threads` | threads of the C++ engine (alias `n_threads`) | OpenMP's default (`OMP_NUM_THREADS`) |
| `compute_stress` | compute the stress (for NPT and cell relaxations). Models trained without stresses (OMOL, SOLIDACID) have none: `get_stress` raises | `True` |
| `dispersion` | off, or add Grimme's D3 to energy, forces and stress ([below](#dispersion)): `None`, `"d3"` (= `"d3bj"`, Becke-Johnson damping), `"d3zero"`, `"d3bjm"`, `"d3zerom"` or `"d3op"`. The pretrained OMOL and SOLIDACID models raise `ValueError` | `None` (off) |
| `functional` | the functional whose D3 parameters are used | `"pbe"`; `"r2scan"` for MATPES-r2SCAN models |
| **more** | | |
| `skin` | neighbour-list skin in Å: larger means fewer rebuilds and more pairs; widened up to 2x when the host cannot keep up. JAX, C++ | `1.5` |
| `async_rebuild` | rebuild the neighbour list in a background thread. JAX, C++ | `True` |
| `background_compile` | compile the next neighbour-table width in a background thread. JAX | on a GPU |
| `neighbour_list` | where MD rebuilds the list: `"host"` (vesin, in a background thread) or `"device"` (a cell list on the GPU, no host round trip; the host still builds the first list of a structure and takes over when the device list outgrows its sizes, runs out of memory, an atom moves more than half the skin, or the system runs in chunks). JAX | `"host"` |
| `chunk_atoms` | evaluate large systems in chunks with a halo, with the same result: `"auto"` (above 100,000 atoms, when one call would not fit), atoms per chunk, or `None`. Needs the fused kernels. JAX | `"auto"` |
| `spatial_order` | sort the atoms along a space-filling curve inside a step (results stay in the order of the Atoms). `"auto"`: on until the calculator has seen three different structures, so that many single points do not each compile a step; or `True`, `False`. JAX | `"auto"` |

**What runs.** What cannot run on the machine falls back with a warning: the GPU to the CPU, the C++
engine to JAX to PyTorch, the fused kernels to XLA. `calc.engine`, `calc.device` and (JAX)
`calc.kernels` say what runs. Options of another engine are ignored. Invalid values raise `ValueError`,
as does `engine="cpp"` with a GPU device.

- **JAX** runs each MD step as one compiled call; it compiles at the first call and again for each
  new padded size, so time MD after a warm-up.
- **C++** is a hand-written CPU engine (forward and backward, OpenMP over the atoms, SIMD), compiled
  at first use for the CPU it runs on ([`ILMLIP_BUILD`](#environment-variables)).
- **PyTorch** evaluates the training model directly.

All three give the same energies, forces and stress to numerical precision.

<span id="precision"></span>**Precision.** `"tf32"` is float32 in which the network's matrix products
run in TensorFloat-32 on NVIDIA GPUs of the Ampere generation or newer; positions, cell and strain stay
float32, and on CPUs it is plain float32. `"auto"` is `"float64"` for a model trained in double precision
(none of the released ones), else `"tf32"`. In every dtype the reference energies (E0) are added to the
site energies, and these summed, in float64. `dtype="float64"` with JAX enables 64-bit arrays for the
whole process.

**Charge and spin.** `atoms.info["charge"]` and `["spin"]` must match the calculator's values or be
absent. The range is charge −10 … 10 and multiplicity 1 … 11.

**Committees.** A committee returns the mean of its members. `calc.results` also holds
`energy_std`, `forces_std`, `force_dev` (per atom), `max_force_dev` and `member_energies` (and
`stress_std` when stress is computed); `calc.members` are the member calculators.

<span id="dispersion"></span>**Dispersion.** Off by default; `dispersion` adds Grimme's D3 from the [dftd3](https://github.com/dftd3/simple-dftd3)
package (`pip install dftd3`, LGPL-3.0; `ImportError` without it) through ASE's `SumCalculator`. The
pretrained OMOL models are refused by name (their ωB97M-V reference has VV10 dispersion), SOLIDACID models
too (SOLID-ACID-0 is PBE-D3 already); for your own checkpoints, fine-tuned ones included, you decide, with
`functional` (default `"pbe"`). **D3 runs on the CPU, about single-threaded, and slows the simulation down a lot.**

### get_hessian

```python
calc = load_calculator("OMOL-S", engine="jax")      # or engine="torch"
H = calc.get_hessian(atoms)                         # (3N, 3N), eV/Å²
lam, vec = calc.get_hessian(atoms, method="lanczos", k=7)   # the 7 lowest modes
```

Analytic second derivatives of the energy (double backward) at fixed cell, with the neighbour list
of this geometry. JAX and PyTorch engines only; the C++ engine and committees raise
`NotImplementedError` (use `calc.members[0]`), and systems run in chunks are not supported. Fused
kernels whose second derivatives fail fall back to XLA with a warning. Translations, and rotations of
a molecule, appear as near-zero eigenvalues; each negative eigenvalue is an imaginary mode.
`examples/hessian/hessian_nh3.py` computes the frequencies of ammonia.

| argument | description | default |
|---|---|---|
| `atoms` | the structure | the last one calculated |
| `method` | `"dense"`: the symmetric (3N, 3N) float64 matrix; `"lanczos"`: the `k` lowest eigenvalues (ascending) and their eigenvectors (3N, k) from Hessian-vector products, for systems too large for the dense matrix | `"dense"` |
| `k` | Lanczos: how many eigenvalues (fewer than 3N) | `6` |
| `chunk` | dense: rows computed together (JAX halves it when the device runs out of memory) | `16` |
| `tol` | Lanczos: relative accuracy of the eigenvalues | exact in float64, `1e-5` otherwise |

### BatchedModel

```python
from ilmlip.batched import BatchedModel

model = BatchedModel("OAM-S-omat")
results = model.compute(structures)     # a dict per structure: energy, forces, stress
```

Many independent structures in one GPU call ([when it pays](quickstart.html#many-small-systems-in-one-call)).

| argument | description | default |
|---|---|---|
| `model` | a pretrained name, a checkpoint file or a loaded JAX model | required |
| `kernels`, `dtype`, `charge`, `spin`, `device` | as for `load_calculator` | `"auto"`, `"tf32"`, `None`, `None`, the GPU |
| `compute_stress` | `None`: when the model was trained on stresses | `None` |
| **more** | | |
| `skin` | neighbour-list skin in Å, wider than for one structure, since one rebuild holds up the call of all | `2.0` |
| `width_step` | neighbour-table widths are multiples of this (larger: fewer compiled shapes); with the fused kernels they are also multiples of 4 | `2` |
| `neighbour_list` | `"host"`, `"device"` or `"auto"` (device on a GPU); a failing device list falls back to the host for good | `"auto"` |

### lockstep.imap

```python
from ilmlip.lockstep import imap

for result in imap("OAM-S-omat", task, items):   # task(calc, item), defined at module level
    print(result)
```

Runs `task(calc, item)` for every item in a process of its own; the calling process holds the model
and computes the forces of all tasks waiting for them in one call. Results come in the order the
tasks finish. A failing task ends alone, and its error is raised after the others. Example:
[many MDs on one GPU](quickstart.html#many-mds-on-one-gpu).

| argument | description | default |
|---|---|---|
| `model` | a `BatchedModel`, a pretrained name or a checkpoint file | required |
| `task` | a module-level function `task(calc, item)`: the processes import it | required |
| `items` | one task per item | required |
| `workers` | processes at a time | one per item |
| `dispersion`, `functional` | as for `load_calculator`: each task's calculator includes D3, computed in the task's process on the CPU (in parallel); the batched model call is unchanged | `None`, the model's |
| `**kwargs` | passed to `BatchedModel` | |

### i-PI driver

`examples/ipi/driver.py` connects ILMlip to [i-PI](https://ipi-code.org/) ≥ 3.3: the beads that
i-PI hands over together (`<batch_size>`) are one model call. Copy the file; do not name it `ipi.py`
(it would shadow i-PI's package). Use `pbc="False"` so that i-PI hands over unwrapped positions. The
`<parameters>` are `template` (a structure file in i-PI's atom order), `model` (name or file), and
optionally `kernels`, `dtype`, `skin`, `charge`, `spin`, `device` and `stress` (`False`: a zero
virial; needed for models without stresses). `examples/README.md` explains `pimd.xml`.

### ilmlip.pretrained

```python
from ilmlip import pretrained

pretrained.names()                # the pretrained models and committees
pretrained.path("OAM-XL-omat")    # its file, downloaded once
```

Names are case-insensitive, and `_` may stand for `-`.

| function | returns |
|---|---|
| `names()` | the pretrained models and committees |
| `path(model)` | the checkpoint file of a model: found in the folders, else downloaded into the cache; a file path is returned as it is. Raises `ValueError` for an unknown name or a committee, `FileNotFoundError` when no model repository is set |
| `members(name)` | the five model names of a committee, `None` for a single model |
| `folders()` | where model files are looked for, in order: `ILMLIP_MODELS`, `umlip/` of a clone, `~/.cache/ilmlip/models` (downloads go there) |
| `canonical(name)` | the registered spelling of a name, or `None` |
| `find(name)` | the model's file in the folders, or `None` (no download) |

### load, fold_charge_spin

```python
from ilmlip import fold_charge_spin, load

model, checkpoint = load("model.pt")
neutral = fold_charge_spin(model, charge=0, spin=1)
```

| function | arguments | returns |
|---|---|---|
| `load(path, device="cpu", untrained=False, kernels="torch")` | a checkpoint file (not a name: use `pretrained.path`); `untrained=True`: its architecture with fresh weights; `kernels`: `"torch"`, `"fused"` or `"fused-reference"` | the `ILMlip` module (PyTorch, float64, eval mode) and the checkpoint dict |
| `fold_charge_spin(model, charge=None, spin=None)` | a model with charge and spin inputs (default 0, 1) | a plain model, exactly equal to it at this total charge and multiplicity. A model without these inputs is returned as it is (it raises `ValueError` if charge or spin is given); an already folded model accepts only its own values |

## Command line

### pretrained

```sh
python -m ilmlip.pretrained                     # list the models, those on this machine marked
python -m ilmlip.pretrained OAM-XL-omat OMOL-S  # download them and print their files
python -m ilmlip.pretrained all                 # every model and committee member
```

| argument | description | default |
|---|---|---|
| `names` | models or committees to download (their files are printed), or `all` alone | none: list the models |

### finetune

```sh
python -m ilmlip.finetune --foundation OAM-S-omat --train data.xyz       # all weights
python -m ilmlip.finetune --foundation OAM-S-omat --train data.xyz --mode lora
python -m ilmlip.finetune --foundation S.pt --train a.xyz b.xyz --mode multihead
python -m ilmlip.finetune --foundation S.pt --train data.xyz --wise 8 --wise-parallel 4
python -m ilmlip.finetune --foundation S.pt --train data.xyz --init run/model.pt
```

| option | description | default |
|---|---|---|
| **what** | | |
| `--foundation` | a pretrained model (by name) or a checkpoint file | required |
| `--mode` | `naive`: every weight trains; `lora`: the weights stay frozen and low-rank updates train, merged on saving; `multihead`: a head per `--train` file plus one on replay structures; `scratch`: the foundation's architecture with random weights, as a baseline | `naive` |
| `--train` | structure files with energies and forces ([labels](#data)); multihead: one head per file (distinct file names, none called `pt_head`) | required |
| `--valid` | validation files; multihead: one per `--train` file, in the same order | a fraction of `--train` |
| `--test` | test files, evaluated before and after training (they do not pick the epoch); multihead: one per `--train` file | |
| `--out` | output folder; must not hold a `model.pt` yet | `<train>_<foundation>_<mode>`; with `--wise N`: `…_naive_wise<N>` |
| **more** | | |
| `--init` | continue from an earlier fine-tune of the same foundation (its `model.pt`) for another `--epochs`, with a fresh schedule and its E0; multihead also continues its other heads | |
| `--valid-fraction` | the fraction of `--train` held out for validation | `0.1` |
| `--e0s` | reference energies (E0): `estimated` (the foundation's plus a per-element correction fitted to the training energies), `foundation`, `composition` (a least-squares fit to the training energies alone; `scratch` always uses it), a JSON dict such as `'{"H": -13.6}'` covering every element of the data, or a `.json` file. Ignored with `--init` | `estimated` |
| `--energy-key`, `--forces-key`, `--stress-key` | where the labels are, if not under the [usual names](#data); a stress key containing `virial` is read as a virial | found automatically |
| **training** | | |
| `--epochs` | epochs | `200` |
| `--batch-size` | structures per batch | `4` |
| `--lr` | peak learning rate (warm-up over 5 % of the steps, then cosine decay to 2 % of the peak) | Adam: `3e-3` (`lora` `1e-2`); Muon: `1.5e-3`; `scratch`: the foundation's |
| `--w-energy`, `--w-forces`, `--w-stress` | loss weights | `1`, `10`, `10` |
| `--optimizer` | `adam`, or `muon`: Muon on the network's weight matrices and Adam on the rest (`lora`: Adam only). Muon fine-tunes move the weights ~4x farther and were less stable in MD | `adam` |
| **more** | | |
| `--huber` | a Huber loss with this threshold (eV/atom, eV/Å) instead of squared errors, for training only | |
| `--ema` | decay of the moving average of the weights; the average is what is saved | `0.99` |
| `--clip` | gradient-norm clip | `10` |
| `--seed` | seed of the split, the shuffling, the LoRA initialisation and the replay draw | `0` |
| `--save-every` | also keep the model of every Nth epoch and of the last as `model_epNNN.pt` (heads: `model_<stem>_epNNN.pt`, `model_pt_head_epNNN.pt`) | `0`: off |
| `--lora-rank`, `--lora-alpha` | rank r and α of the LoRA updates W + (α/r)·B·A | `8`, the rank |
| `--muon-graph`, `--no-muon-graph` | the Muon step as one CUDA graph | on a GPU |
| **replay** | | |
| `--replay` | structures labelled by the foundation: for the replay head (multihead) and for measuring drift from the foundation. A folder from [prepare-replay](#finetune-prepare-replay), `mptrj` (download, ~600 MB, cached), a URL, a structure file, or `none` | `replay_set.extxyz` and `drift_test_set.extxyz` in the current folder; without them multihead downloads MPtrj frames |
| `--replay-n` | how many structures are drawn from `mptrj`, a URL or a file, among those with only the foundation's elements | `1000` |
| `--replay-weight` | weight of the replay head's loss; `0`: no replay head | `1` |
| **WiSE** | | |
| `--wise N` | N naive fine-tunes that differ only in the order of the batches; `model.pt` is their weight average. Running the same command again finishes the unfinished runs | `1` |
| `--wise-parallel P` | run P of them at a time, in separate processes on the visible GPUs (or the CPU) | `1` |
| **runtime** | | |
| `--device` | a PyTorch device: `cuda`, `cuda:1`, `cpu` | the GPU if PyTorch sees one |
| `--dtype` | `tf32` (float32, matrix products in TF32 on NVIDIA GPUs), `float32` (no TF32) or `float64` | `tf32` |
| `--compile`, `--no-compile` | compile the training step: CUDA graphs on a GPU, `torch.compile` on a CPU; falls back to eager if compiling fails | on a GPU |
| `--kernels` | `auto` (`fused` where it can run, else `torch`), `torch`, `fused` (bond sums and products as cuEquivariance CUDA kernels; GPU) or `fused-reference` (the same polynomials in plain PyTorch). Weights and checkpoints are the same | `auto` |
| `--resident` | structures kept on the `device`, or copied there for every batch (`host`, for very large sets) | `device` |
| `--track`, `--track-project` | [live curves](#live-tracking) in `wandb` or `mlflow`; the project or experiment | `none`, `ilmlip` |

**Output** in `--out`:

- `model.pt`: the best epoch by validation loss (the last epoch without validation data), EMA weights; loads in every calculator.
- `log.txt`: a header with date, command, host and versions, then the run.
- `history.csv`: per epoch the learning rate, time, losses and errors on train, validation and drift.
- `summary.json`: final errors.
- Multihead also writes `model_<stem>.pt` (the head of each further `--train` file) and `model_pt_head.pt` (the replay head).
- `--wise N` writes the runs to `member_00`, `member_01`, … and their average to `model.pt`, `log.txt` and `summary.json`.
- Replay structures drawn for the run are saved as `replay_set.extxyz` and `drift_test_set.extxyz`.

### finetune prepare-replay

```sh
python -m ilmlip.finetune prepare-replay --out replay/
python -m ilmlip.finetune prepare-replay --foundation SOLIDACID-S --out replay/
```

Draws replay structures once, to reuse with `--replay replay/`.

| option | description | default |
|---|---|---|
| `--source` | `mptrj` (download, cached in `~/.cache/ilmlip`), a URL or a structure file | `mptrj` |
| `--n` | structures to draw (at least 10): 90 % go to `replay_set.extxyz`, 10 % to `drift_test_set.extxyz` | `1000` |
| `--foundation` | draw only structures with this model's elements: a name or a checkpoint file | any elements |
| `--seed` | seed of the draw | `0` |
| `--out` | folder of the two files | `.` |

### data

```sh
python -m ilmlip.data dataset/ train.xyz [more.xyz ...] --val val.xyz
```

Packs structure files into the dataset folder that `train` reads.

| argument | description | default |
|---|---|---|
| `out` | the dataset folder to write: `train/`, `val/`, `meta.json` | required |
| `train` | training structure files | required |
| `--val` | validation structure files | required |

Labels are looked up in `atoms.info`, then `atoms.arrays`, then the results of an attached calculator:
energy from `REF_energy`, `energy`, `free_energy`, `DFT_energy` or `total_energy`; forces from
`REF_forces`, `forces`, `DFT_forces` or `force`; stress from `REF_stress`, `stress` or `DFT_stress`
(6 Voigt components or 3×3; optional per structure); total charge and multiplicity from
`info["charge"]` and `info["spin"]` (default 0 and 1). Units are eV, eV/Å and eV/Å³ with ASE's
sign of the stress. A structure without energy or forces is an error. Packed files form a single
source named `data`, so `--sources`, `--source-weights` and `--ref-sources` of `train` do not apply.

### train

```sh
python -m ilmlip.train --packed dataset/ --hparams S --epochs 100 --out runs/S
```

Defaults are the recipe of the released models: Muon, MAE losses with weights 1 / 1 / 3, TF32 and
fused kernels.

| option | description | default |
|---|---|---|
| **essentials** | | |
| `--packed` | the dataset folder from [`ilmlip.data`](#data) | required |
| `--hparams` | the size: `XS`, `S`, `M`, `L`, `XL` or `XXL` (the files of `ilmlip/configs/`), or a JSON [configuration](#configuration-files) | required |
| `--out` | run folder; running the same command again [resumes](#resuming) from it | required |
| `--epochs` | train this many epochs instead of `--steps` | `0`: use `--steps` |
| `--steps` | training steps | `1000000` |
| `--scale-from` | the initial energy scale: from the spread of the energy per atom (`energy`), or from the force RMS (`forces`, for total energies such as CP2K's) | `energy` |
| `--zbl` | add screened nuclear repulsion (ZBL); the shipped configurations already have it | off |
| `--charge-spin` | condition the model on total charge and spin, as for OMol25 | off |
| **loss** | | |
| `--loss` | `mae` (energy per atom, length of the force error, stress), `mse` or `huber` (thresholds 0.05 eV/atom, 0.1 eV/Å, 0.01 eV/Å³) | `mae` |
| `--w-energy`, `--w-forces`, `--w-stress` | loss weights. Stress counts where the data have it; without stress data, or with `--w-stress 0`, the model is marked as having no stress | `1`, `1`, `3` |
| `--select-on` | the validation metric that picks `best.pt`: `e_mae`, `e_rmse`, `f_mae`, `f_rmse`, `s_mae`, `s_rmse`, `score`, or per source `f_mae/<source>` | `f_rmse` |
| **more** | | |
| `--loss-energy`, `--loss-forces`, `--loss-stress` | the loss of one term, e.g. `huber` on energy only | `--loss` |
| `--score-refs` | `E,F`: also report score = e_mae/E + f_mae/F | |
| **optimiser** | | |
| `--optimizer` | `muon`: Muon (orthogonalised Nesterov momentum) on the weight matrices, AdamW on the rest; or `adamw` alone. Must match on resume | `muon` |
| `--lr` | peak learning rate (Muon's, or AdamW's with `--optimizer adamw`) | the configuration's (`2.5e-3`) |
| `--lr-adamw` | with Muon: peak learning rate of the AdamW part | `3e-3` |
| `--batch-atoms`, `--batch-structs` | limits of a batch, whichever is reached first | `8192`, `512` |
| `--schedule` | `cosine`, or `wsd`: warm-up, constant peak, then a linear decay over the last `--decay-frac` of the steps; both end at 2 % of the peak | `cosine` |
| `--warmup` | steps of linear warm-up | `2000` |
| `--ema` | decay of the moving average of the weights (kept in float64), at most (1 + n)/(10 + n) after step n; `0`: off | `0.999` |
| **more** | | |
| `--decay-frac` | with `--schedule wsd`: the fraction of the steps that decays | `0.2` |
| `--weight-decay` | decoupled weight decay (a change is ignored on resume) | `1e-3` |
| `--no-e0-decay` | no weight decay on the reference energies: where E0 is large (OMol's absolute energies), decay removes more than Adam restores | off |
| `--grad-clip` | gradient-norm clip | `5` |
| `--muon-momentum`, `--muon-rms` | Muon's momentum, and the RMS of its update per unit learning rate (0.2 matches AdamW, so both share a learning rate) | `0.95`, `0.2` |
| `--muon-graph`, `--no-muon-graph` | the Muon step as one CUDA graph | on a GPU |
| **starting point** | | |
| `--init` | start from this checkpoint (its EMA weights and architecture) with a fresh optimiser; ignored once `--out` holds a checkpoint | |
| `--refit-e0` | with `--init`: refit the reference energies on this many batches first | `0` |
| `--freeze-e0` | keep the reference energies fixed | off |
| `--fold-shift` | with `--init` and `--sources`: add the checkpoint's per-source offsets to its reference energies (not with `--ref-sources`) | off |
| **data sources** | | |
| `--sources` | train and validate on these sources only (comma-separated) | all |
| `--source-weights` | per-epoch weights of the sources, e.g. `mptrj=4,sAlex_train=2` | 1 each |
| `--ref-sources` | the sources that set the energy scale; the others get learned per-element offsets | |
| `--val-per-source` | draw the validation sample evenly from every group of the dataset | off |
| `--data-seed` | seed of the validation sample (committee members share it) | `--seed` |
| `--resident` | per-atom data on the GPU (`device`) or memory-mapped on the host (`host`); `auto`: on the GPU if it takes less than half its free memory | `auto` |
| **runtime** | | |
| `--device` | `cuda` or `cpu` | the GPU if PyTorch sees one |
| `--dtype` | `tf32` (float32, matrix products in TF32 on NVIDIA GPUs), `float32` or `float64` | `tf32` |
| `--kernels` | `auto` (`fused` where it can run, else `torch`), `torch`, `fused` or `fused-reference`, as for `finetune` | `auto` |
| `--compile` | compile energy and forces (GPU only; ignored on a CPU) | off |
| `--seed` | random seed | `0` |
| `--track`, `--track-project` | [live curves](#live-tracking) in `wandb` or `mlflow`; the project or experiment | `none`, `ilmlip` |
| **more** | | |
| `--compile-mode` | `default`, or `reduce-overhead`: replay the step as CUDA graphs | `default` |
| `--edge-bucket` | with `--compile`: pad the pairs of a batch to a multiple of this, so that shapes repeat | `8192` |
| `--log-every` | steps between log lines | `50` |
| `--val-every` | steps between validations (with `--epochs` also at every epoch end) | `2000` |
| `--val-batches` | size of the validation sample, in batches of `--batch-structs` | `50` |
| `--ckpt-every` | steps between checkpoints | `2000` |
| `--stop-after` | stop at this step as on SIGTERM; the schedule is unchanged, so the same command resumes | `0`: off |

<span id="resuming"></span>**Output and resuming.** `--out` gets `checkpoint.pt` (the resume point), `best.pt`
(best `--select-on`), `hparams.json`, `history.jsonl` and, with `--epochs`, `epochs/epoch_NNN.pt`.
Running the same command again resumes mid-epoch; the architecture then comes from the checkpoint.
SIGTERM, SIGINT or SIGUSR2 (LSF's warning before the run limit) finish the step and write a checkpoint.
Use a new `--out` for a fresh run.

#### Configuration files

`--hparams my.json` takes `{"hparams": {...}}`; start from a copy of `ilmlip/configs/S.json`. A file
or folder named like a size in the current directory takes precedence over the shipped one.

| key | meaning |
|---|---|
| `rcut` | the largest cutoff (Å) |
| `rcut_elem` | per-element cutoffs (Å, at most `rcut`); a pair interacts within the larger of its two. Elements not listed get `rcut`. XS to L have them (4 to 6 Å), XL and XXL a uniform 6 Å |
| `lmax`, `lmax_feat`, `lmax_ind` | highest degree of the spherical harmonics: of the first layer's pair products; of every other product and feature; of the messages (`lmax_ind` ≤ `lmax_feat` ≤ `lmax`) |
| `n_layers` | message-passing layers |
| `n_basis` | Chebyshev functions of the radial basis |
| `taper` | channels per degree l = 0 … `lmax` (non-increasing; sets `n_channels`) |
| `n_message` | channels of the messages |
| `product_order` | highest order of the products of the base (1 to 4) |
| `gram_rank`, `bispec_rank` | ranks of the last layer's Gram and bispectrum readout features |
| `embed_dim`, `hidden` | width of the chemical embedding; widths of the readout's hidden layers |
| `zbl`, `zbl_switch` | add ZBL repulsion; the ZBL energy (eV) at which the learned energy is switched off entirely (needs `zbl`) |
| `bound`, `bond_scale` | feature bounds: the RMS gate κ; the scale of the bond sums |
| `lr` | peak learning rate (`--lr` overrides it) |

### evaluate

```sh
python -m ilmlip.evaluate runs/S/best.pt dataset/                         # errors on dataset/val
python -m ilmlip.evaluate runs/S/best.pt runs/S_seed1/best.pt dataset/   # a committee
```

The errors of a checkpoint on a packed split, over all structures and per source: energy MAE and
RMSE (meV/atom), energy MAE per structure (meV), force MAE and RMSE (meV/Å, per Cartesian component)
and, where model and data have stress, stress MAE and RMSE (meV/Å³). Several checkpoints are evaluated
as a committee: each member, their mean, and how well the spread predicts the error (mean spread,
mean error, Pearson r; energies per structure, forces per atom). On OMol25 data, `--out` also gets
the leaderboard's columns per subset.

| argument | description | default |
|---|---|---|
| `checkpoint` | one checkpoint, or several (a committee) | required |
| `data` | the packed dataset folder | required |
| `--split` | the split to evaluate | `val` |
| `--sources` | only these sources, e.g. `mptrj,sAlex_train` | all |
| `--out` | also write the numbers to this JSON file | |
| **more** | | |
| `--ema` | use the checkpoints' EMA weights | off |
| `--device` | `cuda` or `cpu` | the GPU if PyTorch sees one |
| `--dtype` | `float32` or `float64` | `float32` |
| `--batch-atoms` | atoms per batch (raised to the largest structure) | `4096` |
| `--dump` | committee: per-structure errors and spreads to this `.npz` file | |
| `--no-shift` | without the per-source energy offsets of models trained with `--ref-sources` | off |

### probe

```sh
python -m ilmlip.probe md.xyz --model OAM-S-omat
python -m ilmlip.probe md.xyz --model OAM-S-omat --steps 1e6 --dt 0.5 --skip multi --json probe.json
```

How a structure ready for MD runs with a model on this machine, measured in a few minutes: the time per
MD step for each integrator and of a run of `--steps`, how busy the GPU is and how much memory it needs,
and whether a supercell or several MDs at once (lockstep or separate processes) would use it better.

| argument | description | default |
|---|---|---|
| `structure` | a file `ase.io.read` reads | required |
| `--model` | a pretrained name or a checkpoint file | `OAM-S-omat` |
| `--device`, `--engine`, `--dtype` | as for `load_calculator` | `auto` |
| `--dt` | time step (fs) | `0.5` |
| `--temperature` | K | `300` |
| `--steps` | MD length for the run-time estimates | `1e6` |
| `--integrators` | comma-separated, of `nve`, `langevin`, `bussi`, `nose-hoover`, `berendsen`, `andersen`, `npt-mtk` (needs a cell) | all |
| `--integrator` | the integrator of the supercell and multi tests | `langevin` |
| `--seconds` | timed seconds per measurement | `3` |
| `--skip` | comma-separated sections to skip: `integrators`, `supercell`, `multi` | none |
| **more** | | |
| `--max-atoms` | largest supercell tried | `300000` |
| `--copies` | numbers of MDs at once in the multi test | `2,4,8` |
| `--json` | also write the results to this file | |

### lammps

```sh
python -m ilmlip.lammps model.pt model.mliap
```

Writes an ML-IAP file that runs the model in JAX inside LAMMPS:

```
newton          on
pair_style      mliap unified model.mliap 0
pair_coeff      * * Na Cl
```

`lmp -k on g 1 -sf kk -pk kokkos newton on neigh half -in in.lmp`. LAMMPS must be built with KOKKOS
(CUDA), ML-IAP, ML-SNAP, PYTHON and MLIAP_ENABLE_PYTHON, with this package, JAX and CuPy installed in
its Python. Several GPUs use LAMMPS's domain decomposition. For a model without stresses, LAMMPS's
pressure, NPT and box relaxations are not valid (the export warns).

| argument | description | default |
|---|---|---|
| `checkpoint` | the model's checkpoint file | required |
| `out` | the ML-IAP file to write | required |
| `--dtype` | `auto` (the training precision), `tf32`, `float32` or `float64` | `auto` |
| `--kernels` | `auto`, `fused` (cuEquivariance), `xla` or `fused-reference` (tests); all on a dense neighbour table | `auto` |
| `--charge`, `--spin` | total charge and multiplicity of an OMOL model, fixed in the file | OMOL: `0`, `1` |
| `--elements` | only these elements (models with element cutoffs): LAMMPS's cutoff becomes the largest of their pair cutoffs; other elements are refused at run time | all |

## Live tracking

`--track wandb` or `--track mlflow` sends the curves of training and fine-tuning to
[Weights & Biases](https://wandb.ai/site) or [MLflow](https://mlflow.org/), if that package is
installed. The default, `none`, imports neither. If the tracker fails, training goes on; the files
in `--out` are written either way. A resumed training continues its tracked run; the runs of
`--wise` and their average form one group.

```sh
python -m ilmlip.finetune --foundation OAM-S-omat --train data.xyz --out ft --track wandb
wandb leet ft/wandb                                  # the curves in the terminal, no account needed
WANDB_MODE=offline python -m ilmlip.train ... --out runs/S --track wandb   # no internet
wandb sync runs/S/wandb/offline-run-*                # upload later
```

```sh
python -m ilmlip.train ... --track mlflow            # MLflow's default store, or MLFLOW_TRACKING_URI
mlflow ui                                            # browse at http://localhost:5000
MLFLOW_TRACKING_URI=http://server:5000 python -m ilmlip.finetune ... --track mlflow
```

More in the docs of [wandb leet](https://docs.wandb.ai/models/ref/cli/wandb-leet),
[wandb sync](https://docs.wandb.ai/ref/cli/wandb-sync) and
[MLflow tracking](https://mlflow.org/docs/latest/ml/tracking/).

## Environment variables

| variable | description |
|---|---|
| `ILMLIP_MODELS` | folders of model files (separated by `:`, `;` on Windows), searched first |
| `ILMLIP_REPO` | the model repository to download from: a raw-file URL or a local folder |
| `ILMLIP_BUILD` | where the C++ engine is compiled to. Default: `build/` of a writable clone, else `$XDG_CACHE_HOME/ilmlip/build` (`~/.cache/ilmlip/build`) |
| `ILMLIP_CPU_BLOCK` | atoms per readout block of the C++ engine, rounded to a multiple of 6 in 6 … 504 (tuning; default 48, or 96 for wide readouts) |
| `ILMLIP_JAX_EDGE_GROUP` | JAX engine, fused kernels (tuning): bonds of an atom that the bond-sum kernel handles together. Default: 4 for models with up to 16 message channels (`n_message`), else 2; a positive integer overrides it. With the fused kernels, neighbour-table widths are multiples of 4 when it is unset, and of the given value when that is above 2. The results do not change |
| `ILMLIP_JAX_HOST_MEMORY` | JAX engine: the compiled step writes its result to pinned host memory instead of copying it afterwards. Default: on for a GPU, off for a CPU; `off` or `0` switches it off, `out` switches it on, `io` also reads the positions from host memory |
| `CXX` | the compiler of the C++ engine (default `g++`, then `c++`) |
| `OMP_NUM_THREADS` | threads of the C++ engine when `threads` is not given |

The `ILMLIP_JAX_*` variables are read once, when the JAX engine is first imported: set them before the first calculator is created. A value that is not allowed raises `ValueError`.
