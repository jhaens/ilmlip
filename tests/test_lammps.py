"""The ML-IAP model (ilmlip.lammps) against the PyTorch model, through a stand-in for LAMMPS:
ghost atoms, the pair list, the forward and reverse exchange, and the assembly of forces and virial."""
import inspect
import json
import pickle
import threading

import numpy as np
import pytest
import torch
from ase import Atoms

import test_bounds
from conftest import KERNELS, SPECIES, structures, tiny
from test_switches import CASES, build, calibrate
from ilmlip.model import ILMlip, collate, graph, neighbours

jax = pytest.importorskip("jax")
from ilmlip.lammps import ILMlipMLIAP  # noqa: E402
from ilmlip.jax_md import tables  # noqa: E402

jax.config.update("jax_enable_x64", True)


class Data:
    """What LAMMPS passes to compute_forces, built from an ASE structure."""

    energy = property(None, lambda self, e: setattr(self, "total", e))     # write-only, as in LAMMPS

    def __init__(self, atoms, rcut):
        pos, cell, n = atoms.get_positions(), np.asarray(atoms.cell), len(atoms)
        pbc = atoms.pbc if atoms.cell.rank == 3 else np.zeros(3, bool)
        i, j, S = neighbours(pos, cell, pbc, rcut)
        ghost, owner, xyz = {}, list(range(n)), list(pos)
        pj = []
        for b, s in zip(j, S):
            if not np.any(s):
                pj.append(b)
                continue
            key = (int(b), tuple(np.rint(s).astype(int)))
            if key not in ghost:
                ghost[key] = len(owner)
                owner.append(int(b))
                xyz.append(pos[b] + s @ cell)
            pj.append(ghost[key])
        pi, pj, xyz = np.asarray(i), np.asarray(pj), np.asarray(xyz)
        back = pj >= n                           # LAMMPS also lists pairs centred on ghosts
        self.pair_i = np.concatenate([pi, pj[back]]).astype(np.int32)
        self.pair_j = np.concatenate([pj, pi[back]]).astype(np.int32)
        self.rij = xyz[self.pair_j] - xyz[self.pair_i]
        self.nlocal, self.ntotal, self.npairs = n, len(xyz), len(self.pair_i)
        mu = np.array([SPECIES.index(s) for s in atoms.get_chemical_symbols()])
        self.owner = np.asarray(owner)
        self.elems = mu[self.owner].astype(np.int32)
        self.eatoms, self.energy, self.fij = np.zeros(n), None, None

    def forward_exchange(self, src, dst, vec_len):
        dst[:] = src
        dst[self.nlocal:] = src[self.owner[self.nlocal:]]

    def reverse_exchange(self, src, dst, vec_len):
        dst[:] = src
        np.add.at(dst, self.owner[self.nlocal:], src[self.nlocal:])
        dst[self.nlocal:] = 0

    def update_pair_forces(self, fij):
        assert fij.dtype == np.float64 and fij.flags.c_contiguous     # LAMMPS reads it as double*
        self.fij = np.array(fij)

    def forces(self):
        f = np.zeros((self.ntotal, 3))
        np.add.at(f, self.pair_i, self.fij)
        np.add.at(f, self.pair_j, -self.fij)
        out = f[:self.nlocal].copy()
        np.add.at(out, self.owner[self.nlocal:], f[self.nlocal:])
        return out


class Recording(Data):
    """Data that records its exchanges (direction, width, dtype): LAMMPS matches them with the other ranks'."""

    def __init__(self, atoms, rcut):
        super().__init__(atoms, rcut)
        self.calls = []

    def record(self, kind, src, vec_len):
        assert src.shape == (self.ntotal, vec_len)
        self.calls.append((kind, vec_len, src.dtype))

    def forward_exchange(self, src, dst, vec_len):
        self.record("forward", src, vec_len)
        super().forward_exchange(src, dst, vec_len)

    def reverse_exchange(self, src, dst, vec_len):
        self.record("reverse", src, vec_len)
        super().reverse_exchange(src, dst, vec_len)


TINY = {"tiny": {}, "tiny_feat_2": dict(lmax_feat=2, product_order=4), "tiny_3_layers": dict(n_layers=3)}
VARIANTS = {k: CASES[k] for k in ("all", "all_3_layers", "default", "taper", "ranks")}
VARIANTS["cutoffs"] = dict(CASES["all"], rcut_elem={"H": 3.0, "Na": 3.5, "Cl": 3.8})


def make(case):
    """conftest's tiny model or a model of test_switches."""
    return tiny(**TINY[case]) if case in TINY else build(**VARIANTS[case])


@pytest.mark.parametrize("kernels", KERNELS)
@pytest.mark.parametrize("case", [*TINY, *VARIANTS])
@pytest.mark.parametrize("name", list(structures()))
def test_matches_the_torch_model(name, case, kernels, tmp_path):
    m = make(case)
    at = structures()[name]
    periodic = bool(at.pbc.any())
    ref = m(collate([graph(at, m.rcut, m.species)]), compute_stress=periodic)
    model = ILMlipMLIAP(tables(m), "float64", kernels)
    with open(tmp_path / "m.mliap", "wb") as fh:
        pickle.dump(model, fh)
    with open(tmp_path / "m.mliap", "rb") as fh:
        model = pickle.load(fh)
    d = Data(at, m.rcut)
    model.compute_forces(d)
    e_ref = float(ref["energy"][0])
    assert abs(d.total - e_ref) < 1e-10 * max(1.0, abs(e_ref))
    np.testing.assert_allclose(d.eatoms, ref["atomic_energy"].detach().numpy(), atol=1e-10)
    np.testing.assert_allclose(d.forces(), ref["forces"].detach().numpy(), atol=1e-9)
    if periodic:
        w = np.einsum("pa,pb->ab", d.rij, d.fij) / at.get_volume()
        voigt = [w[0, 0], w[1, 1], w[2, 2], w[1, 2], w[0, 2], w[0, 1]]
        np.testing.assert_allclose(voigt, ref["stress"][0].detach().numpy(), atol=1e-9)


@pytest.mark.parametrize("kernels", KERNELS)
@pytest.mark.parametrize("case, e_tol, f_tol", [("tiny", 1e-5, 1e-4), ("all", 1e-4, 1e-3)])
def test_float32_without_x64(case, e_tol, f_tol, kernels):
    """The float32 export runs with JAX's x64 off (as in LAMMPS): still float64 forces for LAMMPS."""
    m = make(case)
    at = structures()["nacl"]
    model = ILMlipMLIAP(tables(m), "float64", kernels)
    ref = Data(at, m.rcut)
    model.compute_forces(ref)
    with jax.enable_x64(False):
        model = ILMlipMLIAP(tables(m), "float32", kernels)
        d = Data(at, m.rcut)
        model.compute_forces(d)
    assert abs(d.total - ref.total) < e_tol * len(at)
    np.testing.assert_allclose(d.forces(), ref.forces(), atol=f_tol)


@pytest.mark.parametrize("kernels", KERNELS)
def test_float32_with_absolute_e0s(kernels):
    """E0s of 2e3 and 7e4 eV (OMol25): LAMMPS gets site energies and the energy with them added and
    summed in float64."""
    m = tiny()
    with torch.no_grad():
        m.atomic_energies.fill_(-2047.4)
        m.atomic_energies[SPECIES.index("Cl")] = -70040.9
    at = structures()["nacl"].repeat((2, 2, 2))
    ref = m(collate([graph(at, m.rcut, m.species)]), compute_forces=False)
    with jax.enable_x64(False):
        d = Data(at, m.rcut)
        ILMlipMLIAP(tables(m), "float32", kernels).compute_forces(d)
    assert np.abs(d.eatoms - ref["atomic_energy"].detach().numpy()).max() < 1e-6
    assert abs(d.total - float(ref["energy"][0])) < 1e-5 and abs(d.eatoms.sum() - d.total) < 1e-6


def test_non_finite_energy_raises():
    m = tiny()
    with torch.no_grad():
        m.energy_scale.fill_(float("nan"))
    with pytest.raises(FloatingPointError):
        ILMlipMLIAP(tables(m), "float64", "xla").compute_forces(Data(structures()["nacl"], m.rcut))


def test_table_overflow_is_redone():
    """A dense table too narrow for the neighbour count: the step is redone wider."""
    m = tiny()
    at = structures()["slab"]
    model = ILMlipMLIAP(tables(m), "float64", "xla")
    ref = Data(at, m.rcut)
    model.compute_forces(ref)
    model._run["width"] = 4
    d = Data(at, m.rcut)
    model.compute_forces(d)
    assert model._run["width"] > 4
    assert d.total == pytest.approx(ref.total, abs=1e-12)
    np.testing.assert_allclose(d.forces(), ref.forces(), atol=1e-12)


@pytest.mark.parametrize("kernels", KERNELS)
@pytest.mark.parametrize("case, dtype, atol", [("tiny", "float64", 1e-12), ("tiny_3_layers", "float64", 1e-12),
                                               ("tiny", "float32", 1e-4), ("all", "float64", 1e-12),
                                               ("all_3_layers", "float64", 1e-12)])
def test_every_rank_exchanges_alike(case, dtype, atol, kernels):
    """A step that redoes a too narrow table and a rank without local atoms exchange like a normal step."""
    m = make(case)
    n_layers = m.n_layers
    at = structures()["slab"]
    with jax.enable_x64(dtype == "float64"):
        model = ILMlipMLIAP(tables(m), dtype, kernels)
        normal, narrow, empty = (Recording(at, m.rcut) for _ in range(3))
        model.compute_forces(normal)
        model._run["width"] = 4
        model.compute_forces(narrow)
        assert model._run["width"] > 4
        empty.nlocal = empty.npairs = 0                    # ghosts only
        model.compute_forces(empty)
    assert [call[0] for call in normal.calls] == ["forward"] * (n_layers - 1) + ["reverse"] * (n_layers - 1)
    assert narrow.calls == empty.calls == normal.calls
    assert empty.total == 0.0
    assert narrow.total == pytest.approx(normal.total, abs=atol)
    np.testing.assert_allclose(narrow.forces(), normal.forces(), atol=atol)


@pytest.mark.parametrize("kernels", KERNELS)
@pytest.mark.parametrize("case", ["tiny", "all"])
def test_atoms_of_a_null_type_are_left_out(case, kernels):
    """pair_style hybrid with a type mapped to NULL: LAMMPS gives its atoms the element -1 and no pairs,
    and no reference energy."""
    m = make(case)
    with torch.no_grad():
        m.atomic_energies.fill_(-2047.4)
    mol = structures()["molecule"]
    model = ILMlipMLIAP(tables(m), "float64", kernels)
    ref = Data(mol, m.rcut)
    model.compute_forces(ref)
    d = Data(mol + Atoms("Cu", positions=[[60.0, 0.0, 0.0]]), m.rcut)
    d.elems[-1] = -1
    model.compute_forces(d)
    assert d.total == pytest.approx(ref.total, abs=1e-10)
    np.testing.assert_allclose(d.eatoms, [*ref.eatoms, 0.0], atol=1e-10)
    np.testing.assert_allclose(d.forces(), [*ref.forces(), np.zeros(3)], atol=1e-10)


def test_sizes_only_grow():
    """Fewer pairs or atoms than before: the compiled shapes stay (no compile in a long run)."""
    m = tiny()
    big = structures()["nacl"].repeat((2, 2, 2))
    small = big[:-10]                                  # vacancies: fewer atoms, pairs and ghosts
    model = ILMlipMLIAP(tables(m), "float64", "xla")
    model.compute_forces(Data(big, m.rcut))
    before = dict(model._run["sizes"])
    d = Data(small, m.rcut)
    model.compute_forces(d)
    assert model._run["sizes"] == before
    ref = m(collate([graph(small, m.rcut, m.species)]), compute_stress=False)
    np.testing.assert_allclose(d.forces(), ref["forces"].detach().numpy(), atol=1e-9)



def test_dtype_names():
    m = tiny()
    assert ILMlipMLIAP(tables(m), "tf32", "xla").precision == "default"
    model = ILMlipMLIAP(tables(m), "float32", "xla")
    assert model.dtype == "float32" and model.precision == "highest"
    with pytest.raises(ValueError):
        ILMlipMLIAP(tables(m), "half", "xla")


def test_more_pairs_of_species_than_one_matrix_product():
    """Beyond _PAIR_MATMUL pairs of species, layer 0 runs per bond sum (no layer0_species)."""
    m = build(**CASES["all"])
    at = structures()["triclinic"] + structures()["nacl"][:6]        # Si O C H Na Cl: 36 pairs
    at.set_cell(at.cell * 2.0, scale_atoms=True)
    ref = m(collate([graph(at, m.rcut, m.species)]))
    d = Data(at, m.rcut)
    ILMlipMLIAP(tables(m), "float64", "xla").compute_forces(d)
    assert abs(d.total - float(ref["energy"][0])) < 1e-10
    np.testing.assert_allclose(d.forces(), ref["forces"].detach().numpy(), atol=1e-9)


def test_species_that_arrive_later():
    """Atoms of a new element migrate to this rank: the pair tables grow, the old species keep their energies."""
    m = build(**CASES["all"])
    model = ILMlipMLIAP(tables(m), "float64", "xla")
    d = Data(structures()["slab"], m.rcut)
    model.compute_forces(d)
    for name in ("nacl", "triclinic"):
        at = structures()[name]
        ref = m(collate([graph(at, m.rcut, m.species)]))
        d = Data(at, m.rcut)
        model.compute_forces(d)
        assert abs(d.total - float(ref["energy"][0])) < 1e-10
        np.testing.assert_allclose(d.forces(), ref["forces"].detach().numpy(), atol=1e-9)


class World:
    """The ranks of a run: the rows each has posted to the others, and a barrier at every exchange."""

    def __init__(self, n):
        self.ranks, self.slots = [], [None] * n
        self.barrier = threading.Barrier(n, timeout=600)


class Rank(Data):
    """One MPI rank of a domain decomposition (domain[i]: the rank of atom i): its own atoms, its ghosts
    (other ranks' atoms and periodic images) and exchanges with the other ranks, which run in threads."""

    def __init__(self, atoms, rcut, domain, rank, world):
        pos, cell, n = atoms.get_positions(), np.asarray(atoms.cell), len(atoms)
        pbc = atoms.pbc if atoms.cell.rank == 3 else np.zeros(3, bool)
        i, j, S = neighbours(pos, cell, pbc, rcut)
        mine = np.flatnonzero(domain == rank)
        row = np.zeros(n, int)                       # an atom's row on its owner
        for r in set(domain):
            row[domain == r] = np.arange((domain == r).sum())
        ghost, gid, xyz, pj = {}, list(mine), list(pos[mine]), []
        owner = [(rank, k) for k in range(len(mine))]
        keep = np.isin(i, mine)
        for b, s in zip(j[keep], S[keep]):
            if domain[b] == rank and not np.any(s):
                pj.append(row[b])
                continue
            key = (int(b), tuple(np.rint(s).astype(int)))
            if key not in ghost:
                ghost[key] = len(gid)
                gid.append(int(b))
                owner.append((int(domain[b]), int(row[b])))
                xyz.append(pos[b] + s @ cell)
            pj.append(ghost[key])
        pi, pj, xyz = np.searchsorted(mine, i[keep]), np.asarray(pj, int), np.asarray(xyz).reshape(-1, 3)
        back = pj >= len(mine)                       # LAMMPS also lists pairs centred on ghosts
        self.pair_i = np.concatenate([pi, pj[back]]).astype(np.int32)
        self.pair_j = np.concatenate([pj, pi[back]]).astype(np.int32)
        self.rij = xyz[self.pair_j] - xyz[self.pair_i]
        self.nlocal, self.ntotal, self.npairs = len(mine), len(xyz), len(self.pair_i)
        mu = np.array([SPECIES.index(s) for s in atoms.get_chemical_symbols()])
        self.gid, self.owner = np.asarray(gid, int), np.asarray(owner, int).reshape(-1, 2)
        self.elems = mu[self.gid].astype(np.int32)
        self.eatoms, self.energy, self.fij = np.zeros(len(mine)), None, None
        self.rank, self.world, self.calls = rank, world, []

    def exchange(self, kind, src):
        w = self.world
        w.slots[self.rank] = src.copy()
        w.barrier.wait()
        assert len({s.shape[1] for s in w.slots}) == 1, "ranks exchange different widths"
        out = np.zeros_like(src)
        if kind == "forward":
            for k, (q, r) in enumerate(self.owner):
                out[k] = w.slots[q][r]
        else:                                        # every copy of my atoms, ghosts included, adds up
            for other in w.ranks:
                rows = np.flatnonzero(other.owner[:, 0] == self.rank)
                np.add.at(out, other.owner[rows, 1], w.slots[other.rank][rows])
        w.barrier.wait()
        self.calls.append((kind, src.shape[1]))
        return out

    def forward_exchange(self, src, dst, vec_len):
        dst[:] = self.exchange("forward", src)

    def reverse_exchange(self, src, dst, vec_len):
        dst[:] = self.exchange("reverse", src)


def slabs(atoms, n):
    """Domains of atoms cut into n slabs along x (scaled, if there is a cell)."""
    x = (atoms.get_scaled_positions() if atoms.cell.rank == 3 else atoms.positions)[:, 0]
    return np.floor(n * (x - x.min()) / (np.ptp(x) + 1e-9)).astype(int)


def run_ranks(model, atoms, domain):
    """compute_forces of one copy of the model per rank, as MPI processes would, and what LAMMPS sums
    from them: (energy, forces, site energies, stress or None, the ranks)."""
    world = World(domain.max() + 1)
    world.ranks = ranks = [Rank(atoms, model.tables["rcut"], domain, r, world) for r in range(len(world.slots))]
    errors = []

    def work(d):
        try:
            pickle.loads(pickle.dumps(model)).compute_forces(d)
        except BaseException as exc:                 # the other ranks would wait for ever
            errors.append(exc)
            world.barrier.abort()

    threads = [threading.Thread(target=work, args=(d,)) for d in ranks]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if errors:
        raise errors[0]
    forces, site, virial = np.zeros((len(atoms), 3)), np.zeros(len(atoms)), np.zeros((3, 3))
    for d in (d for d in ranks if d.nlocal):         # a rank without atoms returns no forces
        f = np.zeros((d.ntotal, 3))
        np.add.at(f, d.pair_i, d.fij)
        np.add.at(f, d.pair_j, -d.fij)
        np.add.at(forces, d.gid, f)                  # the reverse exchange of the forces
        site[d.gid[:d.nlocal]] = d.eatoms
        virial += np.einsum("pa,pb->ab", d.rij, d.fij)
    w = virial / atoms.get_volume() if atoms.pbc.any() else None
    stress = None if w is None else [w[0, 0], w[1, 1], w[2, 2], w[1, 2], w[0, 2], w[0, 1]]
    return sum(d.total for d in ranks), forces, site, stress, ranks


def agree(model, torch_model, atoms, domain, tol=1e-10):
    """The ranks' energy, forces, site energies and stress against the PyTorch model on the whole structure;
    returns the ranks."""
    periodic = bool(atoms.pbc.any())
    b = collate([graph(atoms, torch_model.rcut, torch_model.species)])
    ref = torch_model(b, compute_stress=periodic)
    energy, forces, site, stress, ranks = run_ranks(model, atoms, domain)
    want = [(ref["energy"][0], energy), (ref["forces"], forces), (ref["atomic_energy"], site)]
    if periodic:
        want.append((ref["stress"][0], stress))
    for a, c in want:
        a = a.detach().numpy()
        assert np.abs(a - c).max() <= tol * max(1.0, np.abs(a).max())
    return ranks


def bounded_config(name, **bounds):
    """A configuration file on conftest's species with random FiLM, readout, head and E0s and fitted bounds."""
    h = dict(json.loads((test_bounds.CONFIGS / f"{name}.json").read_text())["hparams"], **bounds)
    known = set(inspect.signature(ILMlip.__init__).parameters)
    torch.manual_seed(0)
    m = ILMlip(SPECIES, **{k: (tuple(v) if k == "hidden" else v) for k, v in h.items() if k in known})
    with torch.no_grad():
        for p in m.parameters():
            if p.abs().sum() == 0:
                p.normal_(0, 0.1)
        m.head.normal_(0, 0.1)
        m.elem_bias.normal_(0, 0.1)
        m.atomic_energies.normal_(0, 2.0)
    calibrate(m.eval())
    return m


def switching(m, atoms):
    """Whether some atom's ZBL energy is within the core switch (0 < z / z_c < 1)."""
    z = m.zbl_atoms(collate([graph(atoms, m.rcut, m.species)])).detach().numpy() / m.zbl_switch
    return bool(((z > 0) & (z < 1)).any())


def points():
    return {**structures(), **test_bounds.dense()}


def test_unbounded_over_ranks():
    """The rank emulation itself: an unbounded model on structures cut into slabs, ghosts of ghosts included."""
    m = build(**CASES["all_3_layers"])
    model = ILMlipMLIAP(tables(m), "float64", "xla")
    for name, n in (("nacl", 2), ("triclinic", 2), ("molecule", 3)):
        at = points()[name]
        assert len(set(slabs(at, n))) == n
        ranks = agree(model, m, at, slabs(at, n))
        assert ranks[0].calls == [("forward", 12)] * 2 + [("reverse", 12)] * 2


@pytest.mark.parametrize("case", test_bounds.CASES)
def test_bounded_keywords_over_ranks(case):
    """Each bound alone and all together, in two and three layers: the gates of the ghost atoms, the squash,
    the core switch."""
    m = build(**test_bounds.CASES[case])
    model = ILMlipMLIAP(tables(m), "float64", "xla")
    for name, n in (("nacl_half", 3), ("squeezed", 2)):
        at = points()[name]
        assert not m.zbl_switch or switching(m, at)
        agree(model, m, at, slabs(at, n))


CONFIGS = [("S", dict(bound=2.0, feat_bound=8.0), ("nacl", "triclinic", "nacl_half", "squeezed", "slab")),
           ("S", dict(feat_bound=8.0), ("nacl_half", "squeezed")),          # the shipped gate, bound 16
           ("L", dict(bound=2.0, feat_bound=8.0), ("nacl_half", "squeezed"))]


@pytest.mark.parametrize("name, bounds, names", CONFIGS, ids=["S-rms", "S-shipped", "L-rms"])
def test_bounded_configurations_over_ranks(name, bounds, names):
    """The shipped configurations with random weights, bounds on (the S gate at kappa 16, the rest at 2) and
    feat_bound 8, on structures cut into two or three slabs; the core switch is active in some."""
    m = bounded_config(name, **bounds)
    assert m.bound and m.zbl_switch and m.feat_bound
    model = ILMlipMLIAP(tables(m), "float64", "xla")
    for pt in names:
        at = points()[pt]
        n = 3 if pt == "nacl_half" else 2
        assert len(set(slabs(at, n))) == n
        assert pt not in ("nacl_half", "squeezed") or switching(m, at)
        agree(model, m, at, slabs(at, n))


@pytest.mark.parametrize("case", ["all", "all_3_layers"])
def test_bounded_every_rank_exchanges_alike(case):
    """The first message carries the gate as one more column; a rank without atoms exchanges the same widths."""
    m = build(**test_bounds.CASES[case])
    at = points()["nacl_half"]
    domain = np.where(slabs(at, 3) == 1, 0, slabs(at, 3))       # the middle slab joins rank 0: rank 1 has no atoms
    ranks = agree(ILMlipMLIAP(tables(m), "float64", "xla"), m, at, domain)
    assert ranks[1].nlocal == 0
    n, k = m.n_layers - 1, 4 * 3
    assert all(d.calls == [("forward", k + 1)] + [("forward", k)] * (n - 1)
               + [("reverse", k)] * (n - 1) + [("reverse", k + 1)] for d in ranks)
