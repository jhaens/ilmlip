<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://jhaens.github.io/ilmlip/assets/logo-white.svg">
    <img src="https://jhaens.github.io/ilmlip/assets/logo-black.svg" alt="ILMlip logo" width="180">
  </picture>
</p>

<p align="center">
  Fast universal interatomic potentials, pretrained on large datasets<br>
  and fine-tunable on material specific data.
</p>

<p align="center">
  <a href="https://jhaens.github.io/ilmlip/builder.html">Command builder</a> ·
  <a href="https://jhaens.github.io/ilmlip/quickstart.html">Quick start</a> ·
  <a href="https://jhaens.github.io/ilmlip/models.html">Models</a> ·
  <a href="https://jhaens.github.io/ilmlip/reference.html">API and CLI</a>
</p>

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://jhaens.github.io/ilmlip/assets/speed-gpu-dark.svg">
    <img src="https://jhaens.github.io/ilmlip/assets/speed-gpu-light.svg" alt="MD speed against system size on one NVIDIA B200" width="400">
  </picture>
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://jhaens.github.io/ilmlip/assets/speed-cpu-dark.svg">
    <img src="https://jhaens.github.io/ilmlip/assets/speed-cpu-light.svg" alt="MD speed against system size on a laptop, 2 cores" width="400">
  </picture>
</p>

<p align="center"><sub>Molecular dynamics of fcc Cu through ASE. GPU: the lines end where the
B200's 183 GB run out; DPA-4.1C and DPA-4.1 are the Air-OAM models. CPU: two performance cores of an
Intel Core Ultra 5 125U.</sub></p>

## Features

- **Pretrained models** in sizes XS to XXL: inorganic materials (OMat24, sAlex, MPtrj; MatPES at PBE and r2SCAN), molecules with charge and spin (OMol25), solid acids (SOLID-ACID-0).
- **Molecular dynamics** in ASE (JAX on a GPU, a C++ engine on a CPU) and in LAMMPS.
- **Fine-tuning**: all weights, LoRA, multi-head with replay, weight averages (WiSE).
- **Training** from scratch on one GPU.

## Install

```sh
pip install ilmlip                # model, training, fine-tuning (PyTorch)
pip install "jax[cuda12]>=0.11.2,<0.12"  # molecular dynamics on the GPU
```

## Molecular dynamics

```python
from ase.io import read
from ilmlip import load_calculator

atoms = read("structure.xyz")
atoms.calc = load_calculator("OAM-S-omat")
```

## Fine-tuning

```sh
python -m ilmlip.finetune --foundation OAM-S-omat --train data.xyz
```

## Training

```sh
python -m ilmlip.data dataset/ train.xyz --val val.xyz
python -m ilmlip.train --packed dataset/ --hparams S --zbl --loss huber \
    --scale-from forces --epochs 100 --compile --out runs/S
```

More in [examples](https://github.com/jhaens/ilmlip/tree/main/examples).

## Citation

If you use ILMlip, please cite the code base (paper in preparation)

```bibtex
@unpublished{ilmlip,
      title={{ILMlip}: fast and universal interatomic potentials},
      author={J. H{\"a}nseroth and C. Dre{\ss}ler},
      year={2026},
      url={https://github.com/jhaens/ilmlip},
      note={in preparation},
}
```

and the datasets of the model you use; all references are in
[references.bib](https://jhaens.github.io/ilmlip/assets/references.bib).

## License

Code and model files are free for noncommercial use under the
[PolyForm Noncommercial License 1.0.0](https://github.com/jhaens/ilmlip/blob/main/LICENSE.md). For commercial use, please contact Jonas
Hänseroth. The training data keep their own licences ([models](https://jhaens.github.io/ilmlip/models.html#licence)).
