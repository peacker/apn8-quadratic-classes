#!/usr/bin/env python3
"""Capture-recapture estimates of the number of quadratic APN classes at n = 8, overall and per M.
Lists: P = published (quad/db/n8_index.tsv, without bgp26-only rows as a separate list), S = classes seen by our
sampler varext (all server result files), W = bgp26 switching classes. Also the sampler's frequency counts
(f1, f2, Chao1). usage: recapture.py"""
import glob, collections, os
Q = "../../learning/quad"
pub = {}; bgp = set()
with open(f"{Q}/db/n8_index.tsv") as f:
    f.readline()
    for l in f:
        t = l.split("\t")
        srcs = set(t[3].split(","))
        if srcs == {"bgp26"}: bgp.add(t[0]); continue
        pub[t[0]] = t[1]
        if "bgp26" in srcs: bgp.add(t[0])
cnt = collections.Counter(); Mof = {}
for fn in glob.glob(f"{Q}/server_results/epyc*/results/varext/*.tsv"):
    with open(fn) as f:
        for l in f:
            t = l.split("\t", 6)
            if len(t) < 7 or t[2] != "1": continue
            cnt[t[3]] += 1; Mof[t[3]] = t[4]
print("sampler: functions", sum(cnt.values()), "classes", len(cnt))
for M in ("3", "4", "all"):
    P = {i for i, m in pub.items() if M == "all" or m == M}
    S = {i for i in cnt if M == "all" or Mof[i] == M}
    both = len(P & S)
    lp = len(P) * len(S) / both if both else float("nan")
    fc = collections.Counter(cnt[i] for i in S)
    # counts are functions; each class is found as a pair, so use half-counts
    f1 = sum(v for k, v in fc.items() if k <= 2); f2 = sum(v for k, v in fc.items() if 3 <= k <= 4)
    chao = len(S) + f1 * f1 / (2 * f2) if f2 else float("nan")
    print(f"M={M}: published {len(P)}, sampler {len(S)}, both {both}, union {len(P | S)}, "
          f"Lincoln-Petersen {lp:,.0f}, sampler f1 {f1} f2 {f2} Chao1 {chao:,.0f}")
W = bgp - set(pub)
print("bgp26 new classes:", len(W), "of them seen by the sampler:", len(W & set(cnt)),
      "M of them:", collections.Counter(Mof.get(i, '?') for i in W & set(cnt)))
