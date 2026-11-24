# Examples

| Folder | What it shows | Runs on |
|---|---|---|
| [lisi/](lisi) | fine-tuning in the four ways, then molecular dynamics | laptop CPU, minutes |
| [hessian/](hessian) | vibrations from the analytic Hessian | laptop CPU, seconds |
| [lockstep/](lockstep) | several MDs in one process, one model call per step for all | GPU |
| [ipi/](ipi) | path-integral MD with i-PI, all beads in one model call | GPU |

## Fine-tuning and MD: lisi/

10 AIMD frames of [Li13Si4](http://doi.org/10.1107/s1600536813029759) (`train.xyz`, 204 atoms) and 20 frames of another trajectory
(`test.xyz`).

```sh
cd lisi
sh finetune.sh      # naive, LoRA, scratch, multihead; each log compares test.xyz before and after
python md.py        # 1000 steps of NVT at 300 K with the foundation model (or a fine-tuned one)
```

One fine-tune takes about two minutes on a laptop CPU and halves the force error on `test.xyz`.

## Hessian: hessian/

```sh
cd hessian
python hessian_nh3.py
```

Ammonia at its minimum and at the planar transition state of its inversion: the vibrational modes
(one imaginary at the transition state) and the inversion barrier.

## Several MDs on one GPU: lockstep/

Water at 300, 400 and 500 K, and water with one and two KOH at 300 K (Bussi, 0.5 fs, 10 ps):
five MDs share one model: each step is one model call for all five, where five processes with
a model each would compete for the GPU.

```sh
cd lockstep
python md.py        # the RUNS list holds the structures and temperatures
```

Each MD is a plain ASE function `md(calc, run)`; `imap(model, md, RUNS)` runs it for every
entry in a process of its own with its own calculator and yields the results as the runs finish.
Anything that asks a calculator for forces works the same way (relaxations, NEB images, ...). The
steps of one MD depend on each other, so one MD cannot be batched; many can. This pays for small
systems: on an A5000, 32 structures in one call are 6x (64 atoms) and
2.7x (512 atoms) faster than one at a time with the S model; from a few thousand atoms on, one
structure already fills the GPU.

## Path integrals with i-PI: ipi/

```sh
pip install "ipi>=3.3"
cd ipi
i-pi pimd.xml       # 64 water molecules, 32 beads, 300 K
```

i-PI >= 3.3 hands the model several beads at once (`<batch_size>`); the driver
([driver.py](ipi/driver.py), which loads `ilmlip.ipi`) evaluates all 32 in one call on the GPU
instead of one after the other.

- `<batch_size>`: the number of beads.
- `pbc="False"`: i-PI passes unwrapped positions, so the neighbour lists last; the model applies
  the periodic boundaries itself.
- `<parameters>`: `template` (species and pbc; positions and cell come from i-PI) and `model` (a
  pretrained name or a `model.pt`). Optional: `dtype` (`tf32`, `float32`, `float64`), `kernels`,
  `skin`, `charge`, `spin`, `stress` (`False`: no virial, for NVT only).

To run the model in another process, or on another node, replace the `<ffdirect>` block of
`pimd.xml` by
`<ffsocket name="model" mode="unix" pbc="False"> <address> model </address> <batch_size> 32 </batch_size> </ffsocket>`,
start i-PI, and then the driver:

```sh
i-pi pimd.xml &
i-pi-py_driver -m custom -P driver.py -u -a model -o template=water.xyz,model=OAM-S-omat
```

The beads may then arrive in any order; the driver matches each to the neighbour list it had
before.
