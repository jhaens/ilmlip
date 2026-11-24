# Models

Every model is loaded by its name; `ilmlip.pretrained.names()` lists them. A model that is not on
this machine is downloaded once into `~/.cache/ilmlip/models` (from `ILMLIP_REPO`, see
[reference](reference.html#ilmlippretrained)); `OAM-S-omat` comes with a clone of the repository.

```python
from ilmlip import load_calculator
atoms.calc = load_calculator("OAM-L-omat")
```

## Which one

| for | take |
|---|---|
| inorganic materials, MD | `OAM-<size>-omat` |
| energies compatible with the Materials Project: phase diagrams, Matbench Discovery | `OAM-<size>-mpa` |
| PBE without +U, or r2SCAN | `MATPES-PBE-<size>`, `MATPES-r2SCAN-<size>` |
| molecules, with their total charge and spin | `OMOL-<size>` |
| solid acids | `SOLIDACID-<size>` |
| a weight average of ten runs (WiSE): often more accurate, same cost | `OAM-L-mpa-WiSE10`, `OAM-L-omat-WiSE10`, `OAM-XXL-mpa-WiSE10` |
| an uncertainty with every step | a committee, e.g. `OAM-L-omat-committee` |

Sizes go from XS (fastest) to XXL (most accurate); S is a good start. For your own data, fine-tune
one of these ([command builder](builder.html#finetune)).

## All models

| models | sizes | trained on | level of theory | references |
|---|---|---|---|---|
| `OAM-<size>-omat` | XS S M L XL XXL | OMat24, sAlex and MPtrj (OAM); last stage OMat24 | PBE and PBE+U, OMat24's settings | [Barroso-Luque et al. 2026](https://doi.org/10.1038/s43588-026-00996-w), [Schmidt et al. 2023](https://doi.org/10.1002/adma.202210788), [Deng et al. 2023](https://doi.org/10.1038/s42256-023-00716-3) |
| `OAM-<size>-mpa` | XS S M L XL XXL | OAM; last stage MPtrj and sAlex | PBE and PBE+U, Materials Project energies | same, and [Jain et al. 2013](https://doi.org/10.1063/1.4812323), [Ong et al. 2015](https://doi.org/10.1016/j.commatsci.2014.10.037) |
| `MATPES-PBE-<size>` | XS S M L XL XXL | MatPES (PBE) | PBE without +U, VASP | [Kaplan et al. 2025](https://arxiv.org/abs/2503.04070) |
| `MATPES-r2SCAN-<size>` | XS S M L XL XXL | MatPES (r2SCAN) | r2SCAN, VASP | [Kaplan et al. 2025](https://arxiv.org/abs/2503.04070) |
| `OMOL-<size>` | XS S M L XL | OMol25; total charge and spin multiplicity are inputs | ωB97M-V/def2-TZVPD | [Levine et al. 2026](https://arxiv.org/abs/2505.08762) |
| `SOLIDACID-<size>` | S M L | SOLID-ACID-0 | PBE-D3 (CP2K) | [Hänseroth et al. 2026](https://arxiv.org/abs/2609.38531), [Hänseroth et al. 2026](https://doi.org/10.1038/s43246-026-01338-z) |

Names are case-insensitive. OMOL and SOLIDACID were trained without stresses: no stress, NPT or cell
relaxation with them.

**WiSE models.** `OAM-L-mpa-WiSE10`, `OAM-L-omat-WiSE10` and `OAM-XXL-mpa-WiSE10` are the weight average of ten runs of the
last training stage that differ only in the order of their batches. They cost as much to run as one
model.

**OAM-S30-mpa.** The size S trained longer, 30 + 15 epochs instead of the 8 + 10 of `OAM-S-mpa`: a little more
accurate on Matbench Discovery (F1 0.801 against 0.788), at the same cost to run.

**Committees.** `OAM-L-omat-committee` and `OAM-L-mpa-committee` are five models of one kind that differ only in their random seed (the model
and `<model>-seed1` … `-seed4`). They return the mean, and the spread in `calc.results`
(`energy_std`, `force_dev`, `max_force_dev`).

## Sizes

| size | parameters | descriptor | readout layers | cutoff |
|---|---|---|---|---|
| XS | 140,456 | harmonics l ≤ 2 in the first layer's pair products, l ≤ 1 elsewhere | 1 × 256 | per element, 4–6 Å |
| S | 300,612 | as XS, more channels | 3 × 256 | per element, 4–6 Å |
| M | 849,732 | as S | 3 × 512 | per element, 4–6 Å |
| L | 963,587 | l ≤ 2 in the features | 3 × 512 | per element, 4–6 Å |
| XL | 6,953,980 | l ≤ 3 in the pair products, l ≤ 2 in features and messages | 3 × 1024 | 6 Å |
| XXL | 27,822,308 | l ≤ 3 throughout | 3 × 2048 | 6 Å |

Every model covers elements 1 to 100. It has two layers of equivariant message passing, with products
of the atomic base up to third order (XXL: second). One MLP shared by all elements reads out the
energy; forces and stress are its derivatives. Element cutoffs are larger for large cations and for
anions; a pair interacts within the larger of its two. Screened nuclear repulsion (ZBL) takes over at
very short distances. OMOL models have a few more parameters for charge and spin (S: 309,828).

## Citation

Please cite ILMlip (paper in preparation) and the dataset of the model you use.

## Datasets

**OMat24.** L. Barroso-Luque et al., *The Open Materials 2024 (OMat24) inorganic materials dataset
and models*, Nat. Comput. Sci. 6, 642 (2026),
[doi:10.1038/s43588-026-00996-w](https://doi.org/10.1038/s43588-026-00996-w).
[Data](https://huggingface.co/datasets/facebook/OMAT24), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

**sAlex**, the subsample of Alexandria released with OMat24 (same reference). Alexandria:
J. Schmidt et al., *Machine-Learning-Assisted Determination of the Global Zero-Temperature Phase
Diagram of Materials*, Adv. Mater. 35, 2210788 (2023),
[doi:10.1002/adma.202210788](https://doi.org/10.1002/adma.202210788); J. Phys. Mater. 9, 025014 (2026),
[doi:10.1088/2515-7639/ae6620](https://doi.org/10.1088/2515-7639/ae6620). [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

**MPtrj.** B. Deng et al., *CHGNet as a pretrained universal neural network potential for
charge-informed atomistic modelling*, Nat. Mach. Intell. 5, 1031 (2023),
[doi:10.1038/s42256-023-00716-3](https://doi.org/10.1038/s42256-023-00716-3).
[Data](https://doi.org/10.6084/m9.figshare.23713842), MIT; from the Materials Project: A. Jain et al.,
APL Mater. 1, 011002 (2013), [doi:10.1063/1.4812323](https://doi.org/10.1063/1.4812323);
S. P. Ong et al., Comput. Mater. Sci. 97, 209 (2015),
[doi:10.1016/j.commatsci.2014.10.037](https://doi.org/10.1016/j.commatsci.2014.10.037).
Materials Project [terms of use](https://next-gen.materialsproject.org/about/terms).

**MatPES.** A. D. Kaplan, R. Liu, J. Qi, T. W. Ko, B. Deng, J. Riebesell, G. Ceder, K. A. Persson,
S. P. Ong, *A Foundational Potential Energy Surface Dataset for Materials*,
[arXiv:2503.04070](https://arxiv.org/abs/2503.04070) (2025).
[Data](https://huggingface.co/datasets/materialyze/matpes) (version 2025.2),
[BSD-3-Clause](https://github.com/materialyzeai/matpes/blob/main/LICENSE), Copyright (c) 2022, Materialyze.AI lab.

**OMol25.** D. S. Levine et al., *The Open Molecules 2025 (OMol25) Dataset, Evaluations, and
Models*, [arXiv:2505.08762](https://arxiv.org/abs/2505.08762) (2026).
[Data](https://huggingface.co/facebook/OMol25), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

**SOLID-ACID-0.** J. Hänseroth, R. A. von Stackelberg, C. Dreßler, *Enabling Domain-Specific
Atomistic Models: A Machine Learning Potential for the Solid Acid Family*,
[arXiv:2609.38531](https://arxiv.org/abs/2609.38531) (2026); J. Hänseroth et al.,
*High-throughput screening and mechanistic insights into solid acid proton conductors*,
Commun. Mater. 7, 222 (2026), [doi:10.1038/s43246-026-01338-z](https://doi.org/10.1038/s43246-026-01338-z).
[Data](https://huggingface.co/datasets/jhaens/solidacid-0_model_dataset), MIT.

**Matbench Discovery**, the benchmark the `-mpa` models are made for: J. Riebesell et al.,
Nat. Mach. Intell. 7, 836 (2025), [doi:10.1038/s42256-025-01055-1](https://doi.org/10.1038/s42256-025-01055-1).

All of these as BibTeX: [references.bib](assets/references.bib).

## Licence

The model files are free for noncommercial use under the same
[PolyForm Noncommercial License 1.0.0](https://polyformproject.org/licenses/noncommercial/1.0.0)
as the code; commercial use needs a separate licence. This covers the model files only: the
datasets above keep their own licences. The models were fitted to subsets of these data (the
splits, filters and energy references are described in the paper) and contain fitted parameters,
not the data. They were trained from random initial weights, not from the published checkpoints of
OMat24, OMol25 or MatPES, and are not endorsed by the authors of these datasets, Meta or the
Materials Project. Provided as is, without warranty.

## How long can I run?

How many MD steps fit into a wall time, and how much MD time they are at 1 fs per step; the slider
sets the system size.

<div class="estimate"></div>

Rough estimates from MD of fcc Cu through ASE (NVE), measured at a few sizes and interpolated
in between. Other systems, densities, numbers of neighbours, software versions and the load of the
machine change them. Setup:

- GPU NVIDIA B200 180GB (host AMD EPYC 9555): ILMlip S in JAX with the fused kernels (tf32) and [MACE-MP-0 small](https://github.com/ACEsuit/mace) (float32, ran out of GPU memory above 50,000 atoms)
- Laptop CPU Intel Core Ultra 5 125U, 32 GB of memory: 2 cores are its two performance cores (one thread each); values above 100,000 atoms are extrapolated, and about 1.5 million atoms fit into its memory
