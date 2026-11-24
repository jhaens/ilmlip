"""The bounds B1 (gate), B2 (feature squash), B3 (core switch) and the ZBL taper of ilmlip.

Each statement is proved symbolically with sympy and then compared with the package's own code
(ILMlip._gate, ILMlip._whiten, core_switch, repulsion), so the statement and the code cannot drift apart.

    python bounds_sympy.py        # about 10 s; the first failed statement stops it
"""
from types import SimpleNamespace

import numpy as np
import sympy as sp
import torch

from ilmlip.model import SCREENING, ILMlip, core_switch, repulsion

F64 = torch.float64
POS = dict(positive=True)
u = sp.Symbol("u")


def ok(name, cond):
    print(f"  [{'ok' if cond else 'FAIL'}] {name}")
    assert cond, name


def tex(name, e):
    print(f"  LaTeX {name}: {sp.latex(e)}")


def gate():
    print("B1  gate  f = a0/sigma (1 + m/(sigma kappa)^2)^(-1/2),  m = mean(A^2) >= 0")
    a0, sig, kap, m = sp.symbols("a_0 sigma kappa m", **POS)
    s = sp.Symbol("s", nonnegative=True)                                    # s = sqrt(m)
    f = a0 / sig / sp.sqrt(1 + s**2 / (sig * kap) ** 2)
    rms = s * f                                                             # RMS component of f A
    tex("f", f.subs(s, sp.sqrt(m)))
    tex("RMS(fA)", rms.subs(s, sp.sqrt(m)))
    x = s / (sig * kap)
    ok("RMS(fA) = a0 kappa x/sqrt(1+x^2), x = sqrt(m)/(sigma kappa)",
       sp.simplify(rms - a0 * kap * x / sp.sqrt(1 + x**2)) == 0)
    ok("RMS^2 = (a0 kappa)^2 - (a0 kappa)^2/(1+q), q = m/(sigma kappa)^2 >= 0  =>  RMS < a0 kappa",
       sp.simplify(rms**2 - ((a0 * kap) ** 2 - (a0 * kap) ** 2 / (1 + s**2 / (sig * kap) ** 2))) == 0)
    d = sp.simplify(sp.diff(rms, s))
    tex("d RMS / d sqrt(m)", d.subs(s, sp.sqrt(m)))
    ok("d RMS/d sqrt(m) = a0/sigma (1+x^2)^(-3/2) > 0  (strictly monotone increasing)",
       sp.simplify(d - a0 / sig * (1 + x**2) ** sp.Rational(-3, 2)) == 0 and d.is_positive)
    ok("lim_{m->oo} RMS = a0 kappa", sp.limit(rms, s, sp.oo) == a0 * kap)
    ok("lim_{m->0} f = a0/sigma  (no change at low density)", sp.limit(f, s, 0) == a0 / sig)
    g = sp.sqrt(1 / (1 + s**2 / (sig * kap) ** 2))                          # gate g = f sigma/a0, in (0, 1]
    ok("at the training RMS (m = sigma^2): g = (1 + 1/kappa^2)^(-1/2); kappa = 16: g = %.5f"
       % float(g.subs({s: sig, kap: 16})), sp.simplify(g.subs(s, sig) - 1 / sp.sqrt(1 + 1 / kap**2)) == 0)

    print("  code: ILMlip._gate on random A, single tensor and list of taper groups")
    stub = SimpleNamespace(bound=16.0, bond_scale=0.05, bond_rms=torch.tensor([0.37, 2.5], dtype=F64), _tap=None)
    gen = torch.Generator().manual_seed(0)
    for t in (0, 1):
        for scale in (1e-6, 1.0, 1e3, 1e9):
            A = scale * torch.randn(7, 5, 9, generator=gen, dtype=F64)
            fA, fac = ILMlip._gate(stub, t, A)
            mm = (A**2).mean((1, 2))
            ref = 0.05 / stub.bond_rms[t] / torch.sqrt(1 + mm / (stub.bond_rms[t] * 16.0) ** 2)
            rms_num = (fA**2).mean((1, 2)).sqrt()
            ok(f"layer {t}, |A| ~ {scale:g}: f == formula, max RMS(fA) = {rms_num.max():.6g} <= a0 kappa = 0.8",
               torch.allclose(fac, ref, rtol=1e-14) and bool((rms_num <= 0.8 * (1 + 1e-12)).all()))   # saturates at a0 kappa up to rounding
    parts = [torch.randn(4, 3, 5, generator=gen, dtype=F64), 50 * torch.randn(4, 6, 2, generator=gen, dtype=F64)]
    out, fac = ILMlip._gate(stub, 0, parts)
    mm = sum((p**2).sum((1, 2)) for p in parts) / (3 * 5 + 6 * 2)
    ok("list input: one f per atom from the pooled mean of squares",
       torch.allclose(fac, 0.05 / 0.37 / torch.sqrt(1 + mm / (0.37 * 16.0) ** 2)))


def squash():
    print("B2  squash  h(x) = x (1 + (x/tau)^2)^(-1/2)")
    x, tau = sp.Symbol("x", real=True), sp.Symbol("tau", **POS)
    h = x / sp.sqrt(1 + (x / tau) ** 2)
    tex("h", h)
    ok("odd: h(-x) = -h(x)", sp.simplify(h.subs(x, -x) + h) == 0)
    ok("h^2 = tau^2 y/(1+y) < tau^2, y = (x/tau)^2  =>  |h| < tau",
       sp.simplify(h**2 - tau**2 * (x / tau) ** 2 / (1 + (x / tau) ** 2)) == 0)
    dh = sp.simplify(sp.diff(h, x))
    ok("h'(x) = (1 + (x/tau)^2)^(-3/2) in (0, 1]: monotone increasing, 1-Lipschitz",
       sp.simplify(dh - (1 + (x / tau) ** 2) ** sp.Rational(-3, 2)) == 0 and dh.subs(x, 0) == 1)
    ok("lim_{x->+-oo} h = +-tau", sp.limit(h, x, sp.oo) == tau and sp.limit(h, x, -sp.oo) == -tau)
    mod = SimpleNamespace(feat_mean=torch.zeros(4, dtype=F64), feat_std=torch.ones(4, dtype=F64), feat_bound=3.0)
    v = torch.tensor([[-1e9, -3.0, 0.0, 1e-9], [1e9, 3.0, 1.0, 2.0]], dtype=F64)
    w = ILMlip._whiten(mod, v)
    ok("code: ILMlip._whiten == h and bounded by tau = 3",
       torch.allclose(w, v / torch.sqrt(1 + (v / 3.0) ** 2)) and bool((w.abs() <= 3.0 * (1 + 1e-12)).all()))


def switch():
    print("B3  core switch  s(u) = 1 - 10u^3 + 15u^4 - 6u^5,  u = z/zc clipped to [0, 1]")
    S = 1 - 10 * u**3 + 15 * u**4 - 6 * u**5
    D = [sp.diff(S, u, k) for k in range(4)]
    tex("s", S)
    tex("s'", sp.factor(D[1]))
    ok("s(0) = 1, s(1) = 0", S.subs(u, 0) == 1 and S.subs(u, 1) == 0)
    ok("s'(0) = s'(1) = s''(0) = s''(1) = 0", all(D[k].subs(u, e) == 0 for k in (1, 2) for e in (0, 1)))
    ok("s' = -30 u^2 (1-u)^2 <= 0 on [0,1]: monotone decreasing from 1 to 0",
       sp.expand(D[1] + 30 * u**2 * (1 - u) ** 2) == 0)
    ok("s'''(0) = s'''(1) = -60 != 0: exactly C2, not C3 (the clip joins the constants 1 and 0)",
       D[3].subs(u, 0) == D[3].subs(u, 1) == -60)
    print("  => e * s(z(x)) is C2 in x: z(x) is C2 (taper below), s joins the constants 1 and 0 with equal 1st and 2nd derivatives")
    zz = torch.linspace(-3, 15, 2001, dtype=F64)
    ok("code: core_switch(z, zc = 10) == polynomial inside, 1 below 0, 0 above zc",
       np.allclose(core_switch(zz, 10.0).numpy(),
                   np.where(zz < 0, 1.0, np.where(zz > 10, 0.0, sp.lambdify(u, S, "numpy")(zz.numpy() / 10)))))


def taper():
    print("ZBL taper  T(u) = (1-u)^3 (1 + 3u + 6u^2 + 10u^3 + 15u^4 + 21u^5),  u = r/rmax")
    P = sum(sp.Rational((k + 1) * (k + 2), 2) * u**k for k in range(6))
    T = (1 - u) ** 3 * P
    tex("T expanded", sp.expand(T))
    ok("T = 1 - 28u^6 + 48u^7 - 21u^8  (P = (1-u)^-3 series cut at u^5)",
       sp.expand(T - (1 - 28 * u**6 + 48 * u**7 - 21 * u**8)) == 0)
    ok("at 0: T = 1 and T', ..., T^(5) = 0 (the ZBL energy is unchanged to O(u^6), T(u) = 1 - 28 u^6)",
       T.subs(u, 0) == 1 and all(sp.diff(T, u, k).subs(u, 0) == 0 for k in range(1, 6))
       and sp.diff(T, u, 6).subs(u, 0) == -20160)
    ok("at 1: T = T' = T'' = 0, T'''(1) = -336 != 0: C2 at rmax",
       [sp.diff(T, u, k).subs(u, 1) for k in range(4)] == [0, 0, 0, -336])
    ok("T' = -168 u^5 (1-u)^2 <= 0: monotone decreasing", sp.expand(sp.diff(T, u) + 168 * u**5 * (1 - u) ** 2) == 0)
    print("  T(u) at u = 0.25, 0.5, 0.75: " + ", ".join(f"{float(T.subs(u, v)):.4f}" for v in (0.25, 0.5, 0.75)))
    r = sp.Symbol("r", **POS)
    Z1, Z2, rmax = 14, 8, sp.Rational(143, 100)
    red = r * (Z1**0.23 + Z2**0.23) / 0.46850
    V = 14.399645 * Z1 * Z2 * sum(w * sp.exp(-c * red) for w, c in SCREENING) / r * T.subs(u, r / rmax)
    V0 = sp.lambdify(r, V, "numpy")
    rs = np.linspace(0.05, 1.40, 500)                       # up to just below rmax (relative comparison)
    code_V = repulsion(torch.tensor(rs, dtype=F64), Z1, Z2, float(rmax)).numpy()
    ok("code: repulsion() == C z1 z2 phi(r)/r * T(r/rmax) (max rel dev %.1e)" % np.max(np.abs(code_V / V0(rs) - 1)),
       np.allclose(code_V, V0(rs), rtol=1e-12, atol=0))
    ok("pair energy: V = V' = V'' = 0 at rmax (the 1/r and screening factors are smooth there), 0 beyond",
       all(abs(sp.diff(V, r, k).subs(r, rmax)) < 1e-12 for k in range(3))
       and float(repulsion(torch.tensor(2.0, dtype=F64), Z1, Z2, float(rmax))) == 0)


def main():
    for part in (gate, squash, switch, taper):
        part()
    print("all statements verified")


if __name__ == "__main__":
    main()
