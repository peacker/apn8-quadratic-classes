#!/usr/bin/env python3
"""Recompute the statistics of Sections 5.2-5.3 from the compact files of this artifact and compare them with
numbers.tex. No raw output rows are needed.

Inputs (all in this folder or in ../symmetric):
  calls.tsv.gz              one line per included call of the uniform sampler (written by make_numbers.py)
  freq_of_freq.tsv          how many classes outside the published lists were drawn once, twice, ...
  ../symmetric/symmetric_group_orders.txt   |G(C)| of the 12,775 symmetric classes (symmetric/engine/autq)
  numbers.tex               for |P| and |K| (counts of the database) and for the comparison

usage: python3 estimate/reproduce.py
"""
import math, os, re, statistics, sys
from fractions import Fraction

here = os.path.dirname(os.path.abspath(__file__))
num = {}
for l in open(os.path.join(here, "numbers.tex")):
    m = re.match(r"\\newcommand\{\\(\w+)\}\{(?:\\prov\{)?(.*?)\}?\}$", l.strip())
    if m: num[m.group(1)] = m.group(2)
def val(name): return float(num[name].replace("{,}", "").replace(",", ""))

import gzip
calls = [l.rstrip("\n").split("\t") for l in gzip.open(os.path.join(here, "calls.tsv.gz"), "rt")][1:]
starts = sum(int(c[3]) for c in calls); att = sum(int(c[4]) for c in calls); tries = sum(int(c[5]) for c in calls)
rows = sum(int(c[7]) for c in calls); n = rows // 2
outP = sum(int(c[10]) for c in calls); outK = sum(int(c[11]) for c in calls)
G = [int(l.split()[2]) for l in open(os.path.join(here, "..", "symmetric", "symmetric_group_orders.txt"))]
symW = sum(Fraction(1, g) for g in G); nSym = len(G)
P, K = int(val("nPub")), int(val("nKnownNoFair"))
D = sum(int(l.split()[1]) for l in open(os.path.join(here, "freq_of_freq.tsv")) if l[0].isdigit())

def count(W0, no):
    f = no / n; return W0 * f / (1 - f), W0 * math.sqrt(f * (1 - f) / n) / (1 - f) ** 2, f
WP = P - nSym + float(symW); WK = K - nSym + float(symW)
mP, sP, fP = count(WP, outP); mK, sK, fK = count(WK, outK)
g = 1
for i in range(8): g *= 256 - 2 ** i
Nb = 2 ** 168 * (starts / att) * rows * 64 / tries; W = Nb / g / g; rel = 1 / math.sqrt(n)
WU = W + 3 * W * rel; room = WU - (WP + D); lam = n / WU; exp_missing = room * math.exp(-lam)
res = {
    "nFairDraws": n, "nOutPDraws": outP, "nOutKDraws": outK, "pubPctOut": 100 * fP,
    "pubOutsideExact": mP, "pubOutsideSE": sP, "nFairOutsideExact": mK, "nFairOutsideSEexact": sK,
    "massW": W, "massWse": W * rel, "massWU": WU, "massLam": lam, "nMassClasses": W - float(symW) + nSym,
}
for nm, gpu in (("Cpu", "0"), ("Gpu", "1")):
    cl = [c for c in calls if c[9] == gpu]
    mode = statistics.mode(int(c[3]) for c in cl)
    d = [int(c[7]) // 2 for c in cl if int(c[3]) == mode]
    res[f"disp{nm}"] = statistics.pvariance(d) / statistics.mean(d)
print(f"symmetric weight: {symW} = {float(symW):.4f} ({nSym} classes)")
print(f"expected number of unknown values (model): {exp_missing:.2e}; with the 0.00135 tail: {exp_missing + 0.00135:.5f}")
bad = 0
for k, v in res.items():
    ref = val(k); tol = max(abs(ref) * 2e-3, 0.5 if abs(ref) >= 10 else 2e-3)   # numbers.tex rounds to integers or decimals
    if k in ("nMassClasses",): tol = 100      # rounded to hundreds in numbers.tex
    ok = abs(v - ref) <= tol; bad += not ok
    print(f"{'ok ' if ok else 'BAD'} {k:22s} recomputed {v:,.4f}   numbers.tex {num[k]}")
sys.exit(1 if bad else 0)
