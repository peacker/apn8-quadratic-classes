#!/usr/bin/env python3
"""Do the 15 radical lines of the non-bent 4-space U come from an F_4-structure? Look for a linear omega
(omega^2 + omega + 1 = 0) that maps every radical line to itself. Also: is pi(R) restricted structure linear?
usage: m4f4.py LUTS.tsv N"""
import sys, collections, itertools
N = int(sys.argv[2]); par = lambda x: bin(x).count("1") & 1
def rank(vs):
    b = []
    for v in vs:
        for x in b: v = min(v, v ^ x)
        if v: b.append(v)
    return len(b)
def solve_lin(pairs):
    """linear map from (x -> y) pairs on a spanning set; returns dict basis or None if inconsistent"""
    basis = []  # (x, y) reduced
    for x, y in pairs:
        for bx, by in basis:
            if x ^ bx < x: x ^= bx; y ^= by
        if x: basis.append((x, y)); basis.sort(reverse=True)
        elif y: return None
    return basis
res = collections.Counter()
for li, l in enumerate(open(sys.argv[1])):
    if li >= N: break
    F = list(map(int, l.rstrip("\n").split("\t")[-1].split(",")))
    beta = lambda a, b: F[a ^ b] ^ F[a] ^ F[b] ^ F[0]
    pi = {}
    for a in range(1, 256):
        cols = [beta(a, 1 << i) for i in range(8)]
        for u in range(1, 256):
            if all(par(u & c) == 0 for c in cols): pi[a] = u; break
    pre = collections.defaultdict(list)
    for a, u in pi.items(): pre[u].append(a)
    nb = set(pre); iml = sorted(nb); U = None
    for a, b in itertools.combinations(iml, 2):
        if a ^ b not in nb: continue
        S2 = {0, a, b, a ^ b}
        for c in iml:
            if c in S2: continue
            S3 = S2 | {x ^ c for x in S2}
            if not S3 - {0} <= nb: continue
            for d in iml:
                if d in S3: continue
                S4 = S3 | {x ^ d for x in S3}
                if S4 - {0} <= nb: U = S4; break
            if U: break
        if U: break
    lines = [sorted(pre[c]) for c in U if c]
    # pick lines greedily to span F_2^8
    chosen = []; cur = []
    for L in lines:
        if rank(cur + L[:2]) == len(cur) + 2: chosen.append(L); cur += L[:2]
        if len(cur) == 8: break
    found = 0
    for orient in range(16):
        pairs = []
        for k, L in enumerate(chosen):
            a, b = L[0], L[1]
            if orient >> k & 1: pairs += [(a, b), (b, a ^ b)]
            else: pairs += [(a, a ^ b), (a ^ b, b)]
        B = solve_lin(pairs)
        def app(x):
            y = 0
            for bx, by in B:
                if x ^ bx < x: x ^= bx; y ^= by
            return y
        if all(set(app(x) for x in L) == set(L) for L in lines): found += 1
    res[found] += 1
print("classes", min(N, li + 1), " number of F_4-structures preserving all 15 radical lines:", dict(res))
