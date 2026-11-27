#!/usr/bin/env python3
"""Random 6-dimensional restrictions of (8,8) functions, as start lines for varext (64 hex values).
x -> F(Mx) for a random injective linear M: F_2^6 -> F_2^8, minus F(0) (EA-equivalent starts).
usage: restrict6.py LUTFILE(tsv, LUT in last column, comma separated) PER_FUNCTION SEED > starts"""
import sys, random
random.seed(int(sys.argv[3])); per = int(sys.argv[2])
def rank(vs):
    b = []
    for v in vs:
        for x in b: v = min(v, v ^ x)
        if v: b.append(v)
    return len(b)
for l in open(sys.argv[1]):
    F = list(map(int, l.rstrip("\n").split("\t")[-1].split(",")))
    if len(F) != 256: continue
    for _ in range(per):
        while True:
            cols = [random.randrange(1, 256) for _ in range(6)]
            if rank(cols) == 6: break
        out = []
        for x in range(64):
            y = 0
            for i in range(6):
                if x >> i & 1: y ^= cols[i]
            out.append(F[y] ^ F[0])
        print(",".join(f"{v:02x}" for v in out))
