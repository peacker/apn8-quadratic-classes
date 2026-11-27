#!/usr/bin/env python3
"""Acceptance rate of step 1 (the beta_6 chain) with xorshift64 (cx*) and splitmix64 (cs2_*).
Each *.last file is the last counter line of one run of
    mass8 -mass 20000000 -seed SEED -tries 0          (built with and without -DSPLITMIX; independent seeds)
fields: accepted chains, chain attempts, ... ; the full logs (one line per 1000 chains) are not included.
usage: python3 -I summary.py DIR"""
import glob, math, os, sys
D = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
res = {}
for name, pat in (("xorshift64", "cx*.last"), ("splitmix64", "cs2_*.last")):
    S = A = 0
    for fn in sorted(glob.glob(os.path.join(D, pat))):
        t = open(fn).read().split(); S += int(t[0]); A += int(t[1])
    p = S / A; se = math.sqrt(p * (1 - p) / A); res[name] = (p, se)
    print(f"{name}: {S:,} accepted of {A:,} attempts, p = {p:.6f} +- {se:.6f}")
(px, sx), (ps, ss) = res["xorshift64"], res["splitmix64"]
print(f"difference {100 * (px / ps - 1):.2f} %, {(px - ps) / math.hypot(sx, ss):.1f} standard errors")
