"""i-PI driver for path-integral MD: the beads that i-PI hands over together are one model call
(batched.py).  i-PI >= 3.3 hands over up to <batch_size> structures at once; with batch_size 1, or
with older versions, each call gets one bead.

In process (i-PI and the model in one Python), with a copy of examples/ipi/driver.py (a file named
ipi.py would shadow i-PI's own package):

    <ffdirect name="model" pbc="False" threaded="True">
      <pes> custom </pes>
      <pes_path> driver.py </pes_path>
      <parameters> {template: init.xyz, model: model.pt} </parameters>
      <batch_size> 32 </batch_size>
    </ffdirect>

Over a socket (<ffsocket mode="unix" pbc="False"> with <batch_size> 32 </batch_size>):

    i-pi-py_driver -m custom -P driver.py -u -a <address> -o template=init.xyz,model=model.pt

batch_size: the number of beads (or a divisor).  pbc="False": positions as i-PI integrates them,
not wrapped into the cell, so the neighbour lists last (the model applies the periodic boundaries
itself).  template: a structure file with the atoms in i-PI's order (its species and pbc are used;
positions and cell come from i-PI).  model: checkpoint, pretrained name or BatchedModel.  Optional:
kernels, dtype, skin, charge, spin, device, stress (False: i-PI gets a zero virial; a model not
trained on stresses needs it).

Each bead uses the neighbour list of the closest bead of the last call (over a socket, i-PI may hand
them over in another order); a bead handed over twice in one call is computed once.  An error outside
the main thread, as in i-PI's force thread with <ffdirect threaded="True">, ends the process: i-PI
would wait for the forces forever.
"""
import os
import sys
import threading
import traceback

import numpy as np
from ase.io import read
from ase.units import Bohr, Hartree
from scipy.optimize import linear_sum_assignment

from .batched import BatchedModel
from .verlet import check_finite

__DRIVER_CLASS__ = "Driver"

_SAMPLE = 64                 # atoms compared to match structures to lists


def _bool(v):
    return v if isinstance(v, bool) else str(v).strip().lower() in ("1", "true", "yes", "on")


class Driver:
    def __init__(self, template, model, kernels="auto", dtype="tf32", skin=2.0, charge=None, spin=None,
                 device=None, stress=True, **_):
        self.template = read(str(template))
        if str(dtype) == "float64":
            from .jax_md import enable_x64
            enable_x64()
        self.model = model if isinstance(model, BatchedModel) else BatchedModel(
            str(model), kernels=str(kernels), skin=float(skin), dtype=str(dtype),
            charge=None if charge is None else int(charge), spin=None if spin is None else int(spin),
            device=device, compute_stress=_bool(stress))
        n = len(self.template)
        self._pick = np.linspace(0, n - 1, min(n, _SAMPLE)).astype(int)
        self._refs = np.zeros((0, 3 * len(self._pick)))    # per list: the sampled positions it had last

    def _slots(self, pos):
        """A list per bead: the assignment to the lists of the last calls with the smallest
        sampled displacements; new lists only for beads left over."""
        sample = np.stack([p[self._pick].ravel() for p in pos])
        old, n = self._refs, len(pos)
        # |a - b|^2 as one matrix product
        d = np.maximum((sample ** 2).sum(1)[:, None] + (old ** 2).sum(1) - 2 * sample @ old.T, 0)
        ks = d.argmin(1) if len(old) >= n else None
        if ks is None or len(np.unique(ks)) < n:
            # a new list costs more than any assignment to old ones
            ks = linear_sum_assignment(np.hstack([d, np.full((n, n), 1 + d.sum())]))[1]
            new = ks >= len(old)
            ks[new] = len(old) + np.arange(new.sum())
        self._refs = np.zeros((max(len(old), ks.max() + 1), sample.shape[1]))
        self._refs[:len(old)] = old
        self._refs[ks] = sample
        return [int(k) for k in ks]

    def compute(self, cell, pos):
        batched = isinstance(cell, list)
        cells, pos = (cell, pos) if batched else ([cell], [pos])
        pos = [Bohr * np.asarray(p, float).reshape(-1, 3) for p in pos]
        cells = [Bohr * np.asarray(h, float).T for h in cells]        # i-PI: vectors as columns
        for p in pos:
            if len(p) != len(self.template):
                raise ValueError(f"{len(p)} atoms, but the template has {len(self.template)}")
            check_finite(p)
        first = {}
        for i, (h, p) in enumerate(zip(cells, pos)):
            first.setdefault((p.tobytes(), h.tobytes()), i)
        unique = sorted(set(first.values()))
        structures = []
        for i in unique:
            a = self.template.copy()
            a.positions, a.cell = pos[i], cells[i]
            structures.append(a)
        res = dict(zip(unique, self.model.compute(structures, self._slots([pos[i] for i in unique]))))
        out = []
        for h, p in zip(cells, pos):
            r = res[first[(p.tobytes(), h.tobytes())]]
            out.append((r["energy"] / Hartree, r["forces"] * (Bohr / Hartree), r["virial"].T / Hartree, ""))
        return out if batched else out[0]

    def __call__(self, cell, pos):
        try:
            return self.compute(cell, pos)
        except Exception:
            if threading.current_thread() is threading.main_thread():
                raise
            traceback.print_exc()
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(1)
