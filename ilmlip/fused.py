"""The couplings as segmented polynomials for the fused kernels of cuEquivariance.

Every coupling is channelwise (output channel k only multiplies inputs of channel k), so each is a
polynomial with one uniform mode u = channels. Operands are (batch, segments * u).
"""
import numpy as np


def edge_poly(paths, n_rad, n_y, n_node, n_out, u, group=1):
    """out[o] += c * radial[l] * Y[lm] * node[r], paths (l, lm, r, o, c).  group G: G bonds of a row per
    batch element: radial and Y hold G bonds side by side (G * n_rad vectors, G * n_y scalars) and the
    nodes are G operands, one per bond; the output is one."""
    import cuequivariance as cue
    vec = lambda n: cue.SegmentedOperand([(u,)] * n)
    operations = []
    for g in range(group):
        d = cue.SegmentedTensorProduct.from_subscripts("u,,u,u")
        for op, n in ((0, group * n_rad), (2, n_node), (3, n_out)):
            for _ in range(n):
                d.add_segment(op, (u,))
        for _ in range(group * n_y):
            d.add_segment(1, ())
        for l, lm, r, o, c in paths:
            d.add_path(g * n_rad + l, g * n_y + lm, r, o, c=c)
        operations.append(([0, 1, 2 + g, 2 + group], d))
    inputs = [vec(group * n_rad), cue.SegmentedOperand([()] * (group * n_y))] + [vec(n_node)] * group
    return cue.SegmentedPolynomial(inputs, [vec(n_out)], operations)


def message_poly(sources, n_rad, n_y, n_out, u, group=1, n_flow=0):
    """edge_poly of two sources (paths, n_node) writing one output.  The back-flow (second source)
    takes the radial operand as its node: n_rad segments, the message's radial functions then the
    back-flow's, which is also n_node of the second.  group: as in edge_poly, the back-flow's paths are
    one operation over all G bonds.  n_flow: the Y factors of the back-flow are an operand of their own
    (n_flow scalars per bond, after Y), and its paths index them."""
    import cuequivariance as cue
    if len(sources) != 2:
        raise ValueError(f"message_poly takes two sources (message, back-flow), got {len(sources)}")
    (paths, n_node), (flow, _) = sources
    vec = lambda n: cue.SegmentedOperand([(u,)] * n)
    sca = lambda n: cue.SegmentedOperand([()] * n)
    first = 3 if n_flow else 2
    out, m = first + group, n_flow or n_y
    operations = []
    for g in range(group):
        d = cue.SegmentedTensorProduct.from_subscripts("u,,u,u")
        for op, n in ((0, group * n_rad), (2, n_node), (3, n_out)):
            for _ in range(n):
                d.add_segment(op, (u,))
        for _ in range(group * n_y):
            d.add_segment(1, ())
        for l, lm, r, o, c in paths:
            d.add_path(g * n_rad + l, g * n_y + lm, r, o, c=c)
        operations.append(([0, 1, first + g, out], d))
    d = cue.SegmentedTensorProduct.from_subscripts("u,,u,u")
    for op, n in ((0, group * n_rad), (2, group * n_rad), (3, n_out)):
        for _ in range(n):
            d.add_segment(op, (u,))
    for _ in range(group * m):
        d.add_segment(1, ())
    for g in range(group):
        for l, lm, r, o, c in flow:
            d.add_path(g * n_rad + l, g * m + lm, g * n_rad + r, o, c=c)
    operations.append(([0, 2 if n_flow else 1, 0, out], d))
    inputs = [vec(group * n_rad), sca(group * n_y)] + ([sca(group * n_flow)] if n_flow else []) + [vec(n_node)] * group
    return cue.SegmentedPolynomial(inputs, [vec(n_out)], operations)


def radial_poly(paths, n_rad, n_y, n_out, u):
    """out[o] += c * radial[s] * Y[lm], paths (s, lm, o, c)."""
    import cuequivariance as cue
    d = cue.SegmentedTensorProduct.from_subscripts("u,,u")
    for op, n in ((0, n_rad), (2, n_out)):
        for _ in range(n):
            d.add_segment(op, (u,))
    for _ in range(n_y):
        d.add_segment(1, ())
    for p in paths:
        d.add_path(*p[:3], c=p[3])
    return cue.SegmentedPolynomial.eval_last_operand(d)


def couple_poly(terms, n_left, n_right, n_out, u):
    """out[o] += cg * left[l] * right[r], terms (l, r, cg, o)."""
    import cuequivariance as cue
    d = cue.SegmentedTensorProduct.from_subscripts("u,u,u")
    for op, n in ((0, n_left), (1, n_right), (2, n_out)):
        for _ in range(n):
            d.add_segment(op, (u,))
    for l, r, cg, o in terms:
        d.add_path(l, r, o, c=cg)
    return cue.SegmentedPolynomial.eval_last_operand(d)


def product_poly(inputs, n_out, u, ops):
    """Several products written into one output.  ops: (buffers, terms), a term (*segments, c, o)
    with a segment of each input buffer."""
    import cuequivariance as cue
    operations = []
    for bufs, terms in ops:
        d = cue.SegmentedTensorProduct.from_subscripts(",".join(["u"] * (len(bufs) + 1)))
        for k, b in enumerate(bufs):
            for _ in range(inputs[b]):
                d.add_segment(k, (u,))
        for _ in range(n_out):
            d.add_segment(len(bufs), (u,))
        for *seg, c, o in terms:
            d.add_path(*seg, o, c=c)
        operations.append((list(bufs) + [len(inputs)], d))
    operand = lambda n: cue.SegmentedOperand([(u,)] * n)
    return cue.SegmentedPolynomial([operand(n) for n in inputs], [operand(n_out)], operations)


def _place(terms, m, inputs, out):
    """Terms (*rows, c, o) of a class with m copies of every segment, at these segment offsets of
    the inputs; out(o): the segment of the output where row o starts."""
    return [(*(b + i * m + q for b, i in zip(inputs, t[:-2])), t[-2], out(t[-1]) + q)
            for t in terms for q in range(m)]


def gram2_poly(n_x, u, n_out, quad, cubic):
    """The Gram features from x, n_x segments of u channels, read twice: as scalars (operand 0, n_x * u
    segments) and as vectors (operand 1, the same array).  out[o] += c * x[a] * x[b] for the paths
    (a, b, o, c) of `quad` (a a scalar, b a vector) and out[o] += c * x[a] * x[b] * x[d] for the paths
    (a, b, d, o, c) of `cubic` (all vectors); n_out segments of u channels."""
    import cuequivariance as cue
    d2 = cue.SegmentedTensorProduct.from_subscripts(",u,u")
    d3 = cue.SegmentedTensorProduct.from_subscripts("u,u,u,u")
    for _ in range(n_x * u):
        d2.add_segment(0, ())
    for d, vectors in ((d2, (1,)), (d3, (0, 1, 2))):
        for op in vectors:
            for _ in range(n_x):
                d.add_segment(op, (u,))
        for _ in range(n_out):
            d.add_segment(d.num_operands - 1, (u,))
    for p in quad:
        d2.add_path(*p[:3], c=p[3])
    for p in cubic:
        d3.add_path(*p[:4], c=p[4])
    inputs = [cue.SegmentedOperand([()] * (n_x * u)), cue.SegmentedOperand([(u,)] * n_x)]
    operations = [([0, 1, 2], d2), ([1, 1, 1, 2], d3)]
    return cue.SegmentedPolynomial(inputs, [cue.SegmentedOperand([(u,)] * n_out)],
                                   [op for op in operations if op[1].num_paths])


def _expanded(c3, c2):
    """The terms (i, j, k, c, o) of order 3 as products of three rows of A: c3 terms (l, r, cg, o) on
    (c2, A), c2 terms (l, r, cg, o) on (A, A).  The product is symmetric in i, j, k: equal ones add."""
    first = {}
    for l, r, cg, o in c2:
        first.setdefault(o, []).append((l, r, cg))
    out = {}
    for l2, k, cg, o in c3:
        for i, j, c in first.get(l2, ()):
            key = (*sorted((i, j, k)), o)
            out[key] = out.get(key, 0.0) + cg * c
    return [(*key[:-1], c, key[-1]) for key, c in out.items()]


def stack_plan(classes, u, limit=0):
    """The products of channel classes as at most two polynomials writing one array F.

    A class (m, has, n1, want, terms[, order]) has m * u channels, laid out as m segments of u channels
    per segment of its channels (segment s, copy q: s * m + q), so each of its terms is repeated m
    times; classes follow one another in A.  has: the couplings ("c2", ...); n1: rows of A; want:
    {order: rows}; terms(key, rows) -> ([(l, r, cg, o)], n_out).  The rows of F are those of the classes
    one after the other, each [order 1 | 2 | 3 | 4], unless the classes have an order ({(order, row): a
    key} of every row): the rows of all classes then follow by key.  c2 is a polynomial of its own,
    unless the order-3 products written as products of A (no order 4) have at most `limit` paths in all.
    Returns (c2_poly or None, f_poly, input names, rows per class, perm): perm[j] is the column of F
    in the order without keys that column j holds, None if there is no difference; W[perm] for a matrix
    W on F's columns in that order.
    """
    info, keyed = [], []
    for c, (m, has, n1, want, terms, *order) in enumerate(classes):
        orders = [o for o in (1, 2, 3, 4) if want.get(o)]
        for o in orders[1:]:
            assert f"c{o}" in has, f"order {o} read but no c{o}"
        r = {o: [int(x) for x in want[o]] for o in orders}
        start = {o: sum(len(r[p]) for p in orders if p < o) for o in orders}
        keyed += [((order[0][o, x] if order else 0), c, g) for g, (o, x) in
                  enumerate((o, x) for o in orders for x in r[o])]
        info.append((m, has, n1, terms, orders, r, start, sum(len(x) for x in r.values())))
    at, here = [[0] * n for *_, n in info], 0       # the segment of F of each row of each class
    for _, c, g in sorted(keyed):
        at[c][g], here = here, here + info[c][0]
    old_at, seg = 0, np.empty(here, int)            # seg[segment without keys] = its segment in F
    for (m, *_, n), a in zip(info, at):
        for g, x in enumerate(a):
            seg[old_at + g * m:old_at + (g + 1) * m] = x + np.arange(m)
        old_at += n * m
    perm = None
    if not np.array_equal(seg, np.arange(old_at)):
        perm = (np.argsort(seg)[:, None] * u + np.arange(u)).reshape(-1)

    def build(expand):
        c2, ops, rows, n = [], {}, [], dict(A=0, c2=0)
        for (m, has, n1, terms, orders, r, start, _), out in zip(info, at):
            shift = lambda ts, o: [(*t[:-1], start[o] + t[-1]) for t in ts]
            copy = lambda o: [(x, 1.0, start[o] + q) for q, x in enumerate(r.get(o, ()))]
            A, S = n["A"], n["c2"]
            put = lambda key, ts, *src: ops.setdefault(key, []).extend(_place(ts, m, src, out.__getitem__))
            put(("A",), copy(1), A)
            higher = 3 in r or 4 in r
            if higher:
                t2, n2 = terms("c2", None)
                if expand and 4 not in r:
                    put(("A", "A"), shift(terms("c2", r[2])[0], 2) if 2 in r else [], A, A)
                    put(("A", "A", "A"), shift(_expanded(terms("c3", r[3])[0], t2), 3), A, A, A)
                else:
                    c2 += _place(t2, m, (A, A), lambda o: S + o * m)
                    n["c2"] += n2 * m
                    put(("c2",), copy(2), S)
                    put(("c2", "A"), shift(terms("c3", r[3])[0], 3) if 3 in r else [], S, A)
                    put(("c2", "c2"), shift(terms("c4", r[4])[0], 4) if 4 in r else [], S, S)
            elif 2 in r:
                put(("A", "A"), shift(terms("c2", r[2])[0], 2), A, A)
            n["A"] += n1 * m
            rows.append(r)
        return c2, {k: v for k, v in ops.items() if v}, rows, n

    plan = build(True) if limit else None
    if plan is None or sum(len(t) for t in plan[1].values()) > limit:
        plan = build(False)
    c2, ops, rows, n = plan
    names = [x for x in ("c2", "A") if any(x in b for b in ops)]
    f_poly = product_poly([n[x] for x in names], old_at, u, [([names.index(x) for x in b], t) for b, t in ops.items()])
    return (couple_poly(c2, n["A"], n["A"], n["c2"], u) if c2 else None), f_poly, names, rows, perm
