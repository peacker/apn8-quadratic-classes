#!/usr/bin/env python3
"""typeaudit.py: which n=8 symmetry types (types8.tsv) does each class have, and did our run of that type find it?

Input: LUT file (TSV, column 5 = inv, column 6 = LUT as in bgp26 new8 files) and the autq AUTQ_PRINT output for
the same lines (A columns of every element of G = {(A,B): beta(Ax,Ay) = B beta(x,y)}).
For each element of prime order p, B is computed from beta; the type signature is the multiplicity of every
irreducible factor f != x+1 of x^p - 1 in A and in B (dim ker f(A) / deg f), minimised over powers (A^k, B^k).
The same signature is computed for the A, B columns of types8.tsv. Then for each type, the classes that have an
element of that type must be among the classes our run of that type found (found sets given as TYPE=file lists,
files with the inv in column 4).
usage: typeaudit.py LUTS.tsv AUTQ.txt types8.tsv TYPE=found.tsv ...
"""
import sys, collections

n = 8
def apply(c, x):
    r = 0
    for i in range(n):
        if x >> i & 1: r ^= c[i]
    return r
def compose(a, b): return [apply(a, b[i]) for i in range(n)]       # a o b
def mpow(a, k):
    r = [1 << i for i in range(n)]
    for _ in range(k): r = compose(a, r)
    return r
def rank(vs):
    b = []
    for v in vs:
        for x in b: v = min(v, v ^ x)
        if v: b.append(v)
    return len(b)
def polyeval(f, a):   # f as int bitmask of coefficients, returns matrix f(a) columns
    res = [0] * n; pw = [1 << i for i in range(n)]
    d = 0
    while f >> d:
        if f >> d & 1: res = [res[i] ^ pw[i] for i in range(n)]
        pw = compose(a, pw); d += 1
    return res
def pmod(a, b):
    db = b.bit_length() - 1
    while a and a.bit_length() - 1 >= db: a ^= b << (a.bit_length() - 1 - db)
    return a
def irreducible_factors(p):
    # irreducible factors of (x^p - 1)/(x - 1) over GF(2), by trial division with all polys of degree <= p-1
    g = (1 << p) - 1   # 1 + x + ... + x^{p-1}
    out = []
    d = 1
    while g.bit_length() > 1:
        for f in range(1 << d, 1 << (d + 1)):
            if f & 1 and pmod(g, f) == 0 and all(pmod(f, h) for h in out if h.bit_length() < f.bit_length()):
                # f irreducible? check no smaller factor
                irr = all(pmod(f, h) for h in range(2, 1 << ((f.bit_length() - 1) // 2 + 1)) if h != f)
                if irr:
                    while pmod(g, f) == 0:
                        q, r = 0, g
                        while r and r.bit_length() >= f.bit_length():
                            s = r.bit_length() - f.bit_length(); q ^= 1 << s; r ^= f << s
                        g = q
                    out.append(f)
        d += 1
        if d > p: break
    return out
FACT = {}
def sig(A, B, p):
    if p not in FACT: FACT[p] = irreducible_factors(p)
    fs = FACT[p]
    best = None
    for k in range(1, p):
        Ak, Bk = mpow(A, k), mpow(B, k)
        s = tuple(n - rank(polyeval(f, Ak)) for f in fs) + tuple(n - rank(polyeval(f, Bk)) for f in fs)
        if best is None or s < best: best = s
    return (p, best)
def order(A):
    I = [1 << i for i in range(n)]; X = A
    for t in range(1, 300):
        if X == I: return t
        X = compose(A, X)
def is_prime(m): return m > 1 and all(m % q for q in range(2, int(m ** .5) + 1))

luts = [l.rstrip("\n").split("\t") for l in open(sys.argv[1])]
groups = []; cur = []
for l in open(sys.argv[2]):
    if l.startswith("A"): cur.append(list(map(int, l.split()[1:])))
    elif l.startswith("|G|"): groups.append(cur); cur = []
assert len(groups) == len(luts), (len(groups), len(luts))
types = {}
for l in open(sys.argv[3]):
    f = l.rstrip("\n").split("\t")
    A = list(map(int, f[5].split(","))); B = list(map(int, f[6].split(",")))
    p = int(f[1]); oa, ob = order(A), order(B)
    types.setdefault(sig(A, B, p), []).append(int(f[0]))
found = {}
for arg in sys.argv[4:]:
    t, fn = arg.split("=")
    s = found.setdefault(int(t), set())
    for fl in fn.split(","):
        for l in open(fl): s.add(l.split("\t")[3])
classtypes = collections.defaultdict(set)
for row, G in zip(luts, groups):
    F = list(map(int, row[5].split(","))); inv = row[4]
    beta = lambda x, y: F[x ^ y] ^ F[x] ^ F[y] ^ F[0]
    for A in G:
        o = order(A)
        if not is_prime(o): continue
        basis = []
        for i in range(n):
            for j in range(i + 1, n):
                u, v = beta(1 << i, 1 << j), beta(A[i], A[j])
                for bu, bv in basis:
                    if u ^ bu < u: u ^= bu; v ^= bv
                if u: basis.append((u, v)); basis.sort(reverse=True)
        B = []
        for k in range(n):
            u, v = 1 << k, 0
            for bu, bv in basis:
                if u ^ bu < u: u ^= bu; v ^= bv
            B.append(v)
        s = sig(A, B, o)
        tl = types.get(s)
        assert tl, ("no type for", s)
        assert len(tl) == 1, ("ambiguous", s, tl)
        classtypes[inv].add(tl[0])
per = collections.defaultdict(set)
for inv, ts in classtypes.items():
    for t in ts: per[t].add(inv)
bad = 0
for t in sorted(per):
    have = found.get(t)
    if have is None:
        print(f"type {t}: {len(per[t])} published classes have it; no found set given"); continue
    miss = per[t] - have
    print(f"type {t}: {len(per[t])} published classes have it; our run found {len(have)}; missing {len(miss)}")
    bad += len(miss)
print("classes with a prime-order self-equivalence:", len(classtypes), " total missing:", bad)
