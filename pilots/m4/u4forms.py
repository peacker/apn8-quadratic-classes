#!/usr/bin/env python3
"""Extract the non-bent 4-space U of M = 4 classes as 4 quadratic forms (28-bit masks, gen6 monomial order
(0,1),(0,2),...,(6,7)): one line "q1 q2 q3 q4" per class (first U found).
usage: u4forms.py LUTS.tsv SHARD NSHARDS > out"""
import sys, itertools
sh, nsh = int(sys.argv[2]), int(sys.argv[3]); par = lambda x: bin(x).count("1") & 1
pidx = {}; k = 0
for i in range(8):
    for j in range(i + 1, 8): pidx[(i, j)] = k; k += 1
for li, l in enumerate(open(sys.argv[1])):
    if li % nsh != sh: continue
    F = list(map(int, l.rstrip("\n").split("\t")[-1].split(",")))
    beta = lambda a, b: F[a ^ b] ^ F[a] ^ F[b] ^ F[0]
    nb = set()
    for a in range(1, 256):
        cols = [beta(a, 1 << i) for i in range(8)]
        for u in range(1, 256):
            if all(par(u & c) == 0 for c in cols): nb.add(u); break
    iml = sorted(nb); U = None
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
                if S4 - {0} <= nb: U = (a, b, c, d); break
            if U: break
        if U: break
    if not U: continue
    qs = []
    for u in U:
        q = 0
        for (i, j), p in pidx.items():
            if par(u & beta(1 << i, 1 << j)): q |= 1 << p
        qs.append(q)
    print(*qs, flush=True)
