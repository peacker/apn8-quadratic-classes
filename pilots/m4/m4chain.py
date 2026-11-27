#!/usr/bin/env python3
"""For M = 4 classes: chains U (4-dim non-bent) < C5 < C6 of output component spaces.
B5 = bent components in C5, B6 = bent in C6, Z = number of 2-subspaces {a,b} with beta(a,b) in the annihilator of C6
(the condition set for the last two forms). Prints the (B5, B6, Z) distribution over all chains and the per-class max.
usage: m4chain.py LUTS.tsv N"""
import sys, collections
N = int(sys.argv[2])
par = lambda x: bin(x).count("1") & 1
allc = collections.Counter(); best = collections.Counter()
for li, l in enumerate(open(sys.argv[1])):
    if li >= N: break
    F = list(map(int, l.rstrip("\n").split("\t")[-1].split(",")))
    beta = lambda a, b: F[a ^ b] ^ F[a] ^ F[b] ^ F[0]
    # pi and bentness
    nb = set()
    for a in range(1, 256):
        cols = [beta(a, 1 << i) for i in range(8)]
        for u in range(1, 256):
            if all(par(u & c) == 0 for c in cols): nb.add(u); break
    # count beta values over 2-subspaces
    Nv = collections.Counter()
    for a in range(1, 256):
        for b in range(a + 1, 256):
            if b < (a ^ b): Nv[beta(a, b)] += 1
    # U
    iml = sorted(nb); U = None
    for a in iml:
        for b in iml:
            if b <= a or a ^ b not in nb: continue
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
        if U: break
    seen5 = set(); bestv = None
    for c5 in range(256):
        if c5 in U: continue
        C5 = frozenset(U | {x ^ c5 for x in U})
        if C5 in seen5: continue
        seen5.add(C5)
        B5 = sum(1 for x in C5 if x and x not in nb)
        seen6 = set()
        for c6 in range(256):
            if c6 in C5: continue
            C6 = frozenset(C5 | {x ^ c6 for x in C5})
            if C6 in seen6: continue
            seen6.add(C6)
            B6 = sum(1 for x in C6 if x and x not in nb)
            ann = [v for v in range(1, 256) if all(par(v & c) == 0 for c in C6)]
            Z = sum(Nv[v] for v in ann)
            allc[(B5, B6, Z)] += 1
            if bestv is None or (B5, B6) > bestv[:2]: bestv = (B5, B6, Z)
    best[bestv] += 1
print("classes", min(N, li + 1))
print("per-class best (B5,B6,Z):", best.most_common(10))
print("all chains (B5,B6,Z):", allc.most_common(15))
