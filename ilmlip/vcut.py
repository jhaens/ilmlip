"""Element-specific cutoffs: a pair interacts within the larger of its elements' cutoffs.

Anions get 4 A * r_ion / r_ion(O2-) (Pauling's radii); other elements get twice their covalent bond
to O.  Both are clipped to [4, 6] A.

    python -m ilmlip.vcut table [--anions ...] [--set P=5]
"""
import argparse
import hashlib
import json

import numpy as np

CPU_TAG = 3000
ANION_RADII = {"N": 1.71, "O": 1.40, "F": 1.36, "P": 2.12, "S": 1.84, "Cl": 1.81,   # Pauling's crystal radii (A)
               "As": 2.22, "Se": 1.98, "Br": 1.95, "Sb": 2.45, "Te": 2.21, "I": 2.16}
ANIONS = ("P", "As", "Sb", "S", "Se", "Te", "Cl", "Br", "I", "N", "O", "F")
BY_HAND = {}


def rule(lo=4.0, hi=6.0, factor=2.0, partner="O", anions=ANIONS):
    from ase.data import atomic_numbers, chemical_symbols, covalent_radii
    rp = float(covalent_radii[atomic_numbers[partner]])
    rc = {chemical_symbols[z]: factor * (float(covalent_radii[z]) + rp) for z in range(1, 101)}
    rc.update({el: lo * ANION_RADII[el] / ANION_RADII["O"] for el in anions})
    return {el: round(min(hi, max(lo, x)), 2) for el, x in rc.items()}


def pair_cutoffs(species, rcut, rcut_elem):
    rc = np.array([float(rcut_elem.get(s, rcut)) for s in species])
    if not ((rc > 0) & (rc <= rcut)).all():
        raise ValueError(f"rcut_elem: every cutoff must lie in (0, rcut] = (0, {rcut:g}]")
    return np.maximum(rc[:, None], rc[None, :])


def table(m):
    elem = getattr(m, "rcut_elem", None)
    return pair_cutoffs(m.species, m.rcut, elem) if elem else None


def attach(m, rcut_elem):
    import torch
    from ase.data import chemical_symbols
    m.rcut_elem = {str(k): float(v) for k, v in rcut_elem.items()} if rcut_elem else None
    rc = table(m)
    if rc is None:
        return
    unknown = sorted(set(m.rcut_elem) - set(chemical_symbols[1:]))
    if unknown:
        raise ValueError(f"rcut_elem: not element symbols: {unknown}")
    if m.zbl:
        from .model import zbl_tables
        rcov = zbl_tables(m.species)[1]
        reach = np.minimum(m.zbl_factor * (rcov[:, None] + rcov[None, :]), m.rcut)
        if (rc < reach - 1e-9).any():
            a, b = np.unravel_index(np.argmax(reach - rc), rc.shape)
            raise ValueError(f"rcut_elem: {m.species[a]}-{m.species[b]} has the cutoff {rc[a, b]:g} A, "
                             f"inside its ZBL repulsion ({reach[a, b]:.2f} A)")
    m.register_buffer("vc_scale", torch.tensor(m.rcut / rc), persistent=False)
    own = [m.rcut_elem.get(s, m.rcut) for s in m.species]
    m.register_buffer("vc_cutoffs", torch.tensor(np.rint(np.asarray(own) * 1000), dtype=torch.int32))


def describe(m):
    rc = np.array([m.rcut_elem.get(s, m.rcut) for s in m.species])
    top = np.argsort(-rc, kind="stable")[:6]
    return (f"[model] element cutoffs {rc.min():g}-{rc.max():g} A, a pair the larger of its two "
            f"({int((rc > rc.min()).sum())} elements above {rc.min():g}: "
            + ", ".join(f"{m.species[k]} {rc[k]:g}" for k in top) + ", ...)")


def arch(h, arch):
    h.pop("rcut_elem", None)
    if arch.get("rcut_elem"):
        h["rcut_elem"] = dict(arch["rcut_elem"])


def prune(g, m):
    rc = table(m)
    if rc is None or not len(g["ind_i"]):
        return g
    i, j, mu = g["ind_i"], g["ind_j"], g["mu"]
    vec = g["positions"][j] - g["positions"][i] + g["shift"]
    keep = np.einsum("ij,ij->i", vec, vec) < rc[mu[i], mu[j]] ** 2
    return dict(g, ind_i=i[keep], ind_j=j[keep], shift=g["shift"][keep])


def same(models):
    t = [table(m) for m in models]
    if any((a is None) != (t[0] is None) or (a is not None and not np.array_equal(a, t[0])) for a in t):
        raise SystemExit("the checkpoints have different element cutoffs")


def by_z(m, zmax=100):
    rc = table(m)
    if rc is None:
        return None
    from ase.data import atomic_numbers
    z = np.array([atomic_numbers[s] for s in m.species])
    out = np.full((zmax + 1, zmax + 1), float(m.rcut))
    out[np.ix_(z, z)] = rc
    return out


def cache_name(rc):
    return f"n_edges_vc{hashlib.sha1(np.ascontiguousarray(rc, np.float64).tobytes()).hexdigest()[:12]}.npy"


def count_species(loader, atom):
    import torch
    if loader.resident == "device":
        return loader.a_Z[atom].long()
    return torch.as_tensor(np.asarray(loader.split.Z[atom], np.int64), device=loader.device)


def within(loader, Z, i, j, pos, S, st, structs, periodic):
    import torch
    from .data import _rowvec_mat
    rc2 = getattr(loader, "_vc_rc2", None)
    if rc2 is None:
        rc2 = loader._vc_rc2 = torch.tensor(loader.pair_rcut ** 2, dtype=torch.float64, device=loader.device)
    dp = pos[j] - pos[i]
    vec = dp + _rowvec_mat(S, structs["cell"][st]) if periodic else dp
    return (vec * vec).sum(1) < rc2[Z[i], Z[j]]


def list_cutoff(rc, mu, rcut):
    """The largest pair cutoff among the species mu (rcut without element cutoffs)."""
    if rc is None:
        return rcut
    present = np.unique(mu)
    return float(rc[np.ix_(present, present)].max())


def verlet_keep(rc, mu, skin, i, j, S, d):
    if rc is None:
        return i, j, S, d
    keep = d < rc[mu[i], mu[j]] + skin
    return i[keep], j[keep], S[keep], d[keep]


def verlet_inside(rc, mu, i, j, d, inside):
    if rc is None:
        return inside
    return d < rc[mu[i], mu[j]]


def jax_tables(m):
    rc = table(m)
    return None if rc is None else dict(scale=m.rcut / rc, rc2=rc * rc, rc=rc)


def host_table(tables):
    v = tables.get("vcut")
    return None if v is None else np.asarray(v["rc"], np.float64)


def scaled(T, r, mi, mj):
    v = T.get("vcut")
    return r if v is None else r * v["scale"][mi, mj]


def inside(T, ok, vec, mi, mj):
    v = T.get("vcut")
    if v is None:
        return ok
    import jax.numpy as jnp
    return ok & (jnp.sum(vec * vec, axis=-1) < v["rc2"][mi, mj])


def listed(T, ok, vec, mi, mj, skin):
    """verlet_keep on the device: pairs within their own cutoff + skin."""
    v = T.get("vcut")
    if v is None:
        return ok
    import jax.numpy as jnp
    return ok & (jnp.sum(vec * vec, axis=-1) < (v["rc"][mi, mj] + skin) ** 2)


def far(T, out, rij, pi, pj, mu_all):
    v = T.get("vcut")
    if v is None:
        return out
    import jax.numpy as jnp
    return out | (jnp.sum(rij * rij, axis=-1) >= v["rc2"][mu_all[pi], mu_all[pj]])


def lammps_centres(T, rij, pi, pj, mu_all, n_rows):
    if T.get("vcut") is None:
        return pi
    import jax.numpy as jnp
    return jnp.where(far(T, jnp.zeros(pi.shape, bool), rij, pi, pj, mu_all), n_rows, pi)


def lammps_elements(model, elements, out):
    """Restrict an exported ML-IAP model to these elements: LAMMPS's pair cutoff (2 rcutfac)
    becomes their largest pair cutoff; written to out again."""
    import pickle
    v = model.tables.get("vcut")
    unknown = sorted(set(elements) - set(model.element_types))
    if unknown:
        raise SystemExit(f"--elements: not in the model: {unknown}")
    if v is not None:
        k = [model.element_types.index(e) for e in elements]
        model.rcutfac = 0.5 * float(np.asarray(v["rc"])[np.ix_(k, k)].max())
    model.vc_elements = [model.element_types.index(e) for e in elements]
    with open(out, "wb") as fh:
        pickle.dump(model, fh)


def lammps_check(model, data, xp):
    """Refuse atoms of other elements than an --elements export was made for (once)."""
    allowed = getattr(model, "vc_elements", None)
    if allowed is None or getattr(model, "_vc_checked", False):
        return
    e = np.asarray(xp.asnumpy(data.elems) if xp is not np else data.elems)
    bad = sorted(set(int(x) for x in np.unique(e)) - set(allowed) - {-1})
    if bad:
        raise RuntimeError(f"this model file was exported for {[model.element_types[k] for k in allowed]} only "
                           f"(its pair cutoff), not {[model.element_types[k] for k in bad]}")
    model._vc_checked = True


def pack(m, ints, floats):
    rc = table(m)
    if rc is not None:
        ints.append(CPU_TAG)
        floats += list((m.rcut / rc).reshape(-1)) + list((rc * rc).reshape(-1))


def main(argv=None):
    p = argparse.ArgumentParser(description="Element-specific cutoffs: the rule.")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("table")
    s.add_argument("--lo", type=float, default=4.0, help="smallest cutoff (A)")
    s.add_argument("--hi", type=float, default=6.0, help="largest cutoff (A)")
    s.add_argument("--factor", type=float, default=2.0, help="cutoff / bond length to --partner")
    s.add_argument("--partner", default="O")
    s.add_argument("--anions", nargs="*", default=list(ANIONS), metavar="EL",
                   help="elements cut by their ionic radius (default: ANIONS; --anions alone: none)")
    s.add_argument("--set", nargs="*", default=[f"{k}={v:g}" for k, v in BY_HAND.items()], metavar="EL=A",
                   help="cutoffs set by hand (default: BY_HAND; --set alone: none)")
    a = p.parse_args(argv)
    from ase.data import atomic_numbers
    if a.partner not in atomic_numbers:
        p.error(f"--partner {a.partner}: not an element")
    bad = [el for el in a.anions if el not in ANION_RADII]
    if bad:
        p.error(f"--anions {' '.join(bad)}: no ionic radius in ANION_RADII ({' '.join(ANION_RADII)})")
    rc = rule(a.lo, a.hi, a.factor, a.partner, a.anions)
    for item in a.set:
        el, _, val = item.partition("=")
        try:
            rc[el] = float(val) if el in rc else None
        except ValueError:
            rc[el] = None
        if rc[el] is None:
            p.error(f"--set {item}: want ELEMENT=CUTOFF, e.g. P=5")
    print(json.dumps(rc, indent=1))


if __name__ == "__main__":
    main()
