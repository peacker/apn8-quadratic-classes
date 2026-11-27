#!/usr/bin/env python3
"""Structure of M = 4 quadratic APN (8,8)-functions: the 4-dim subspaces U inside Im(pi) u {0}
(pi = ortho-derivative; c.F is non-bent iff c in Im pi; radical of c.F = pi^-1(c) u {0}).
For each class: number of such U, and for each U: radical dims of its 15 components, dim of the span of all
radicals, and whether the radicals of U form part of a spread (pairwise trivial intersection is automatic).
usage: m4struct.py LUTS.tsv N"""
import sys, collections, itertools
N = int(sys.argv[2])
def rank(vs):
    b = []
    for v in vs:
        for x in b: v = min(v, v ^ x)
        if v: b.append(v)
    return len(b)
def span(vs):
    s = {0}
    for v in vs:
        s |= {x ^ v for x in s}
    return s
stats = collections.Counter(); nU = collections.Counter(); spandim = collections.Counter(); raddims = collections.Counter()
for li, l in enumerate(open(sys.argv[1])):
    if li >= N: break
    F = list(map(int, l.rstrip("\n").split("\t")[-1].split(",")))
    pi = [0] * 256; pre = collections.defaultdict(list)
    for a in range(1, 256):
        row = [F[a ^ x] ^ F[a] ^ F[x] ^ F[0] for x in range(256)]
        for u in range(1, 256):
            if all(bin(u & r).count("1") % 2 == 0 for r in row[:1] + [row[1 << i] for i in range(8)]):
                pi[a] = u; break
        pre[pi[a]].append(a)
    im = set(pre)
    # 4-dim subspaces inside im u {0}
    Us = set()
    iml = sorted(im)
    for a in iml:
        for b in iml:
            if b <= a: continue
            if a ^ b not in im: continue
            S2 = {0, a, b, a ^ b}
            for c in iml:
                if c in S2 or c < b: continue
                S3 = S2 | {x ^ c for x in S2}
                if not S3 - {0} <= im: continue
                for d in iml:
                    if d in S3 or d < c: continue
                    S4 = S3 | {x ^ d for x in S3}
                    if S4 - {0} <= im: Us.add(frozenset(S4))
    nU[len(Us)] += 1
    for U in Us:
        rd = tuple(sorted(len(pre[c]) + 1 for c in U if c))
        raddims[rd] += 1
        allrad = [a for c in U if c for a in pre[c]]
        spandim[rank(allrad)] += 1
print("classes:", min(N, li + 1))
print("number of 4-dim U per class:", dict(nU))
print("radical sizes (|pi^-1(c)|+1) over U:", dict(raddims.most_common(8)))
print("dim span of radicals of U:", dict(spandim))
