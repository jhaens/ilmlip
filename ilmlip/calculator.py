"""ASE calculators: the fastest one for a model (load_calculator), the PyTorch model, and a
committee of models.

    from ilmlip import load_calculator
    atoms.calc = load_calculator("OAM-S-omat")                 # GPU (JAX) if there is one, else CPU (C++)
    atoms.calc = load_calculator("omol.pt", charge=1, spin=2)  # models with charge and spin inputs
    atoms.calc = load_calculator("OAM-S-omat", device="cpu", engine="jax")

load_calculator returns one of jax_md.JaxCalculator, cpu.CpuCalculator, TorchCalculator (the training
model evaluated directly, for checks) or CommitteeCalculator.
"""
import contextlib
import copy
import inspect
import warnings
from pathlib import Path

import numpy as np
import torch
from ase.calculators.calculator import Calculator, PropertyNotImplementedError, all_changes

from .model import DTYPES, NO_STRESS, collate, fold_charge_spin, graph, load, refuse_charge_spin, trained_dtype


@contextlib.contextmanager
def _matmul_tf32(on):
    """PyTorch's TF32 switch for float32 matrix products, for one evaluation."""
    old = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = on
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old


class TorchCalculator(Calculator):
    """The training model evaluated directly.  dtype: "tf32" (float32, matrix products in
    TensorFloat-32 on GPUs that have it), "float32" or "float64"."""

    implemented_properties = ["energy", "free_energy", "forces", "energies", "stress"]
    engine = "torch"

    def __init__(self, model, device="cpu", dtype="tf32", charge=None, spin=None, compute_stress=True,
                 **kw):
        super().__init__(**kw)
        if str(dtype) not in DTYPES:
            raise ValueError(f"dtype must be one of {'|'.join(DTYPES)}, got {dtype!r}")
        self.compute_stress = bool(compute_stress)
        if isinstance(model, (str, Path)):
            model, _ = load(model)
        model = fold_charge_spin(model, charge, spin)
        self.charge_spin_state = getattr(model, "folded_charge_spin", None)
        self.tf32 = str(dtype) == "tf32"
        self.device = torch.device(device)
        self.dtype = torch.float32 if self.tf32 else getattr(torch, str(dtype))
        self.model = copy.deepcopy(model).to(self.device, self.dtype).eval()
        # the E0s as the model has them, unrounded: forward adds them in float64
        self.model.atomic_energies.data = model.atomic_energies.detach().to(self.device, torch.float64, copy=True)

    def check_state(self, atoms, tol=1e-15):
        changes = Calculator.check_state(self, atoms, tol)      # not super(): CommitteeCalculator reuses it
        if self.atoms is not None and any(atoms.info.get(k) != self.atoms.info.get(k)
                                          for k in ("charge", "spin")):
            changes.append("charge_spin")
        return changes

    def _graph(self, atoms):
        cs = self.charge_spin_state
        if cs is not None:
            got = (int(atoms.info.get("charge", cs[0])), int(atoms.info.get("spin", cs[1])))
            if got != tuple(cs):
                raise ValueError(f"atoms.info has charge/spin {got}, this calculator is for {cs}")
        m = self.model
        g = graph(atoms, m.rcut, m.species)
        from . import vcut
        g = vcut.prune(g, m)
        return g

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        m = self.model
        g = self._graph(self.atoms)
        if "stress" in properties and not self.model.stress:
            raise PropertyNotImplementedError(NO_STRESS)
        stress = self.compute_stress and self.model.stress and g["volume"] > 0
        with _matmul_tf32(self.tf32):
            out = m(collate([g], self.device, self.dtype), compute_stress=stress)
        e = float(out["energy"][0].detach())
        if not np.isfinite(e):
            raise FloatingPointError(f"non-finite energy {e}")
        self.results = dict(energy=e, free_energy=e,
                            forces=out["forces"].detach().cpu().double().numpy(),
                            energies=out["atomic_energy"].detach().cpu().double().numpy())
        if stress:
            self.results["stress"] = out["stress"][0].detach().cpu().double().numpy()

    def get_hessian(self, atoms=None, method="dense", k=6, chunk=16, tol=None):
        """d2E/dx2 in eV/A^2 at fixed cell and with the neighbour list of this geometry; atoms: the
        last ones calculated by default.  "dense": the symmetric (3N, 3N) float64 array, `chunk`
        rows at a time.  "lanczos": the k lowest eigenvalues (ascending) and eigenvectors (3N, k)
        from Hessian-vector products; tol: their relative accuracy (default exact in float64,
        1e-5 in float32 and TF32)."""
        if method not in ("dense", "lanczos"):
            raise ValueError(f"get_hessian: method must be 'dense' or 'lanczos', not {method!r}")
        atoms = self.atoms if atoms is None else atoms
        if atoms is None:
            raise ValueError("get_hessian: no atoms given and none calculated yet")
        batch = collate([self._graph(atoms)], self.device, self.dtype)
        x = batch["positions"]
        with _matmul_tf32(self.tf32):
            (dE,) = torch.autograd.grad(self.model(batch, compute_forces=False)["energy"].sum(), x,
                                        create_graph=True)
            if method == "lanczos":
                from scipy.sparse.linalg import LinearOperator, eigsh

                def hvp(v):
                    v = torch.as_tensor(v, dtype=x.dtype, device=x.device).reshape(x.shape)
                    return torch.autograd.grad(dE, x, v, retain_graph=True)[0].reshape(-1).double().cpu().numpy()

                if tol is None:
                    tol = 0 if x.dtype == torch.float64 else 1e-5
                return eigsh(LinearOperator((x.numel(),) * 2, matvec=hvp, dtype=np.float64), k=k, which="SA", tol=tol)
            eye = torch.eye(x.numel(), dtype=x.dtype, device=x.device)
            rows = []
            for i in range(0, len(eye), chunk):
                v = eye[i:i + chunk].reshape(-1, *x.shape)
                try:
                    h = torch.autograd.grad(dE, x, v, is_grads_batched=True, retain_graph=True)[0]
                except RuntimeError:    # an operation without a batching rule (or no memory for the batch)
                    h = torch.stack([torch.autograd.grad(dE, x, r, retain_graph=True)[0] for r in v])
                rows.append(h.reshape(len(v), -1))
        H = torch.cat(rows).double().cpu().numpy()
        return 0.5 * (H + H.T)


class CommitteeCalculator(Calculator):
    """Mean and spread of several calculators.  results: energy_std (eV), force_dev per
    atom = sqrt(mean_k |F_k - <F>|^2) (eV/A) and its maximum, max_force_dev."""

    implemented_properties = ["energy", "free_energy", "forces", "energies", "stress"]

    def __init__(self, members, **kw):
        super().__init__(**kw)
        self.members = list(members)
        if not self.members:
            raise ValueError("a committee needs at least one member")

    check_state = TorchCalculator.check_state

    def get_hessian(self, *args, **kwargs):
        raise NotImplementedError("get_hessian: not for a committee; take a member's, e.g. calc.members[0].get_hessian")

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        want = ["energy", "forces"] + [p for p in ("stress", "energies") if p in properties]
        outs = [{p: m.get_property(p, self.atoms) for p in want} for m in self.members]
        E = np.array([o["energy"] for o in outs])
        F = np.stack([o["forces"] for o in outs])
        dev = np.sqrt(((F - F.mean(0)) ** 2).sum(-1).mean(0))
        self.results = dict(energy=float(E.mean()), free_energy=float(E.mean()), forces=F.mean(0),
                            member_energies=E, energy_std=float(E.std()), forces_std=F.std(0),
                            force_dev=dev, max_force_dev=float(dev.max()) if len(dev) else 0.0)
        for key in ("stress", "energies"):
            if key in want:
                X = np.stack([o[key] for o in outs])
                self.results.update({key: X.mean(0), f"{key}_std": X.std(0)})


# fastest first
_ORDER = [("jax", "cuda"), ("torch", "cuda"), ("cpp", "cpu"), ("jax", "cpu"), ("torch", "cpu")]


def _jax_devices():
    try:
        import jax
        return jax.devices()
    except Exception:                       # not installed, or no backend starts
        return []


def _usable(engine, device):
    """None if (engine, device) runs here, else why not."""
    if engine == "jax":
        devs = _jax_devices()
        if not devs:
            return "JAX is not installed (pip install jax)"
        if device == "cuda" and not any(d.platform == "gpu" for d in devs):
            return "JAX sees no GPU (pip install 'jax[cuda12]')"
        return None
    if engine == "torch":
        return None if device == "cpu" or torch.cuda.is_available() else "PyTorch cannot find GPU"
    try:
        from .cpu import library
        library()
    except Exception as exc:                 # no compiler, or the build failed
        return f"the C++ engine cannot be built here ({str(exc).splitlines()[0][:200]})"
    return None


def _candidates(device, engine):
    """(engine, device) pairs to try: the requested device first, then the engine, then speed."""
    return sorted(_ORDER, key=lambda p: (device not in ("auto", p[1]), engine not in ("auto", p[0])))


def _options(cls, options, engine):
    """The options cls takes (own and inherited keywords); the others are dropped with a warning."""
    known = set()
    for c in inspect.getmro(cls):
        if "__init__" in vars(c):
            known |= set(inspect.signature(c.__init__).parameters)
    ok = {k: v for k, v in options.items() if k in known}
    dropped = sorted(set(options) - set(ok))
    if dropped:
        warnings.warn(f"options {', '.join(dropped)} do not apply to the {engine} engine: ignored",
                      RuntimeWarning, stacklevel=3)
    return ok


def load_calculator(model="OAM-S-omat", device="auto", engine="auto", kernels="auto", dtype="auto",
                    charge=None, spin=None, threads=None, dispersion=None, functional=None, **options):
    """The fastest ASE calculator for a model on this machine.

    model: a pretrained model (ilmlip.pretrained.names(), e.g. "OAM-S-omat"; downloaded on first use unless 
    it is on this machine), a checkpoint file, an ILMlip module, or a list of these (a committee: 
    CommitteeCalculator).
    device: "auto" (the GPU when JAX can use one, else the CPU), "cuda" (or "gpu", "cuda:1"), "cpu".
    engine: "auto" (GPU: "jax", CPU: "cpp"), "jax", "cpp" (the compiled C++ engine, CPU only) or
    "torch" (the training model, evaluated directly: slow, for checks).
    kernels: JAX on a GPU: "auto", "fused" (cuEquivariance) or "xla".
    dtype: "auto" ("float64" for models trained in float64, else "tf32"), "tf32" (float32; on NVIDIA
    GPUs since Ampere the network's matrix products in TensorFloat-32, positions, cell and strain in
    full float32; on CPUs the same as float32), "float32" or "float64" (for JAX this enables 64-bit
    arrays process-wide).
    charge, spin: total charge and multiplicity (default 0 and 1), only for models that take them
    (OMOL); other models refuse them.
    threads: threads of the cpp engine (default: OpenMP's default count, OMP_NUM_THREADS if set).
    options: for the calculator chosen, e.g. compute_stress, skin, chunk_atoms.
    dispersion: None, or Grimme's D3 added to energy, forces and stress, computed on the CPU by
    the dftd3 package (pip install dftd3): "d3" (= "d3bj", Becke-Johnson damping), "d3zero",
    "d3bjm", "d3zerom" or "d3op".  functional: the D3 parameters, default that of the training
    data (PBE; r2SCAN for MATPES-r2SCAN).  The pretrained OMOL models (VV10) and SOLIDACID
    models (PBE-D3) refuse D3: they have dispersion already; for any other model you decide.

    What cannot run here falls back, with a warning: the GPU to the CPU, cpp to jax to
    torch, fused kernels to xla.  The calculator records its choice in .engine, .device
    and (JAX) .kernels.
    """
    from . import pretrained
    device = str(device).lower()
    device = "cuda" + device[3:] if device.startswith("gpu") else device
    kind = device.split(":")[0]
    if kind not in ("auto", "cuda", "cpu") or engine not in ("auto", "jax", "cpp", "torch"):
        raise ValueError(f"device auto|cuda|cpu and engine auto|jax|cpp|torch, not {device!r}, {engine!r}")
    if kernels not in ("auto", "fused", "xla"):
        raise ValueError(f"kernels must be auto|fused|xla, got {kernels!r}")
    if dtype not in ("auto",) + DTYPES:
        raise ValueError(f"dtype must be auto|{'|'.join(DTYPES)}, got {dtype!r}")
    if engine == "cpp" and kind == "cuda":
        raise ValueError("the cpp engine runs on the CPU: device='cpu', or engine='jax' for the GPU")
    threads = options.pop("n_threads", threads)
    if dispersion is not None:
        damping, functional = _damping(dispersion), _functional(model, functional)
        calc = load_calculator(model, device, engine, kernels, dtype, charge, spin, threads, **options)
        return with_d3(calc, damping, functional)
    names = pretrained.members(model) if isinstance(model, (str, Path)) else None
    if names is not None or isinstance(model, (list, tuple)):
        calcs = [load_calculator(m, device, engine, kernels, dtype, charge, spin, threads, **options)
                 for m in (names or model)]
        return CommitteeCalculator(calcs)
    if isinstance(model, (str, Path)):
        model, ck = load(pretrained.path(model))
        dtype = trained_dtype(ck) if dtype == "auto" else dtype
    elif dtype == "auto":
        dtype = "tf32"

    why = {}
    for eng, dev in _candidates(kind, engine):
        why[eng, dev] = _usable(eng, dev)
        if why[eng, dev] is None:
            break
    # device="auto" without a GPU means the CPU: no GPU candidates are reported
    gpu_here = kind != "auto" or torch.cuda.is_available()
    skipped = [f"{e} on {d}: {r}" for (e, d), r in why.items() if r and (d == "cpu" or gpu_here)]
    if skipped:
        warnings.warn(f"ilmlip runs with {eng} on the {dev}: " + "; ".join(skipped), RuntimeWarning,
                      stacklevel=2)
    full = dev if dev == "cpu" or kind != "cuda" else device
    if eng == "jax":
        return _jax_calculator(model, full, kernels, dtype, charge, spin, options)
    if eng == "cpp":
        from .cpu import CpuCalculator
        opts = _options(CpuCalculator, options, eng)
        return CpuCalculator(model, dtype=dtype, charge=charge, spin=spin, n_threads=threads, **opts)
    opts = _options(TorchCalculator, options, eng)
    return TorchCalculator(model, device=full, dtype=dtype, charge=charge, spin=spin, **opts)


def _jax_calculator(model, device, kernels, dtype, charge, spin, options):
    from . import jax_md
    if dtype == "float64":
        jax_md.enable_x64()
    if isinstance(model, jax_md.JaxModel):
        refuse_charge_spin(model.charge_spin_state, charge, spin)
    else:
        model = jax_md.from_torch(model, dtype, charge, spin, device)
    on_gpu = model.device.platform == "gpu"
    if on_gpu and kernels != "xla" and not jax_md.fused_available():
        warnings.warn("the fused GPU kernels (cuEquivariance) are not installed: XLA kernels instead "
                      "(pip install cuequivariance-jax cuequivariance-ops-jax-cu12)", RuntimeWarning,
                      stacklevel=3)
        kernels = "xla"
    if not on_gpu:
        if kernels == "fused":
            warnings.warn("fused kernels run on the GPU only: XLA kernels on the CPU", RuntimeWarning,
                          stacklevel=3)
        kernels = "xla"
    opts = _options(jax_md.JaxCalculator, options, "jax")
    return jax_md.JaxCalculator(model, kernels=kernels, **opts)


_DAMPING = {"d3": "d3bj", "d3bj": "d3bj", "d3zero": "d3zero", "d3bjm": "d3bjm", "d3zerom": "d3zerom", "d3op": "d3op"}


def _damping(dispersion):
    """The D3 variant of a name; ImportError when the dftd3 package is missing."""
    damping = _DAMPING.get(str(dispersion).lower().replace("(", "").replace(")", "").replace("-", ""))
    if damping is None:
        raise ValueError(f"dispersion must be one of {'|'.join(_DAMPING)}, got {dispersion!r}")
    try:
        import dftd3.ase  # noqa: F401
    except ImportError as exc:
        raise ImportError("dispersion needs the dftd3 package: pip install dftd3") from exc
    return damping


def _functional(model, functional=None):
    """functional, else that of a pretrained model's training data, for its D3 parameters; ValueError
    for the pretrained OMOL and SOLIDACID models (by name)."""
    from . import pretrained
    names = model if isinstance(model, (list, tuple)) else [model]
    name = " ".join(pretrained.canonical(m) or "" for m in names if isinstance(m, (str, Path))).upper()
    if "OMOL" in name:
        raise ValueError("OMOL models are trained on wB97M-V, which has VV10 dispersion: no D3 on top")
    if "SOLIDACID" in name:
        raise ValueError("SOLID-ACID-0 is computed with PBE-D3: its models have dispersion already")
    return functional or ("r2scan" if "R2SCAN" in name else "pbe")


def with_d3(calc, dispersion="d3bj", functional="pbe", **d3_options):
    """calc plus Grimme's D3 (dftd3 package, on the CPU) as an ASE SumCalculator; d3_options go
    to dftd3.ase.DFTD3."""
    damping = _damping(dispersion)
    from dftd3.ase import DFTD3
    from ase.calculators.mixing import SumCalculator
    return SumCalculator([calc, DFTD3(method=functional, damping=damping, **d3_options)])
