#!/usr/bin/env python3
"""Fast class lookup for many quadratic APN lookup tables (n = 8 symmetric search).

Same output format as testall.py (source line apn inv M status seconds) and the same invariant hash,
but: the differential spectrum of the ortho-derivative is computed in one vectorised step, and the
CCZ-to-permutation test is run only for invariants not seen before (M and status are class invariants,
taken from --known TSV files: testall outputs or quad/db/n8_index.tsv).

usage: fastclass.py --n 8 --source NAME --input LUTS --out OUT.tsv [--shard i --nshards k] [--known F ...]
"""
import argparse, hashlib, os, re, sys, time
import numpy as np
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, "..", "lib"))
from apnsearch import ccz, fast  # noqa: E402
from apnsearch.invariants import ortho_derivative, _fwht  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, required=True); ap.add_argument("--source", required=True)
ap.add_argument("--input", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshards", type=int, default=1)
ap.add_argument("--known", nargs="*", default=[])
a = ap.parse_args()
n = a.n; N = 1 << n
X = np.arange(N)
parity = np.array([bin(x).count("1") & 1 for x in range(N)], dtype=np.int64)
A = np.arange(1, N)[:, None] ^ X[None, :]            # rows a = 1..N-1: x ^ a

def inv_of(F):
    pi = ortho_derivative(F, n)
    D = pi[A] ^ pi[None, :]                           # (N-1) x N
    cnt = np.zeros((N - 1, N), dtype=np.int64)
    np.add.at(cnt, (np.repeat(np.arange(N - 1), N), D.ravel()), 1)
    vals, c = np.unique(cnt, return_counts=True)
    ddt = tuple((int(v), int(k)) for v, k in zip(vals, c))
    W = _fwht(1 - 2 * parity[pi[:, None] & X[None, :]])
    walsh = np.unique(np.abs(W[:, 1:]), return_counts=True)
    return hashlib.sha1(repr((ddt, tuple((int(x), int(y)) for x, y in zip(*walsh)))).encode()).hexdigest()[:16]

known = {}                                            # inv -> (M, status)
for fn in a.known:
    for l in open(fn):
        f = l.rstrip("\n").split("\t")
        if fn.endswith("n8_index.tsv"):
            if f[0] != "inv": known.setdefault(f[0], (f[1], f[2]))
        elif len(f) >= 6 and f[2] == "1": known.setdefault(f[3], (f[4], f[5]))
num = re.compile(r"\d+")
out = open(a.out, "a")
with open(a.input) as fh:
    for i, line in enumerate(fh):
        if i % a.nshards != a.shard: continue
        vals = num.findall(line)
        if len(vals) != N: continue
        F = np.array(list(map(int, vals)), dtype=np.int64); t0 = time.time()
        apn = fast.is_apn(F); inv = M = st = "-"
        if apn:
            inv = inv_of(F)
            if inv in known: M, st = known[inv]
            else:
                st, info = ccz.ccz_to_permutation_dual(F, n, quadratic=True); M = info.get("max_dim_NB")
                known[inv] = (str(M), st)
                if st == "permutation": open(a.out + ".PERMUTATION", "a").write(f"{a.source}\t{i}\t{line}")
        out.write(f"{a.source}\t{i}\t{int(apn)}\t{inv}\t{M}\t{st}\t{time.time() - t0:.3f}\n")
out.close()
