#!/usr/bin/env python3
"""Per-start yield table from runs/perstart: start line, CPU s, G7, functions, classes, new classes, M4."""
import numpy as np, glob, os, collections
known = np.load("data/known.npy")
rows = []
for f in glob.glob("runs/perstart/s*.cpu"):
    b = f[:-4]; ln = int(os.path.basename(b)[1:])
    c = open(f).read().split(); cpu = float(c[0]) if c and c[0] != "G7" else 0; g7 = int(c[-1])
    cls = {}
    if os.path.exists(b + ".tsv"):
        for l in open(b + ".tsv"):
            t = l.split("\t")
            if t[2] == "1": cls[t[3]] = t[4]
    inv = np.array([int(i, 16) for i in cls], dtype=np.uint64)
    new = int((~np.isin(inv, known)).sum()) if len(inv) else 0
    rows.append((ln, cpu, g7, len(cls), new, sum(1 for m in cls.values() if m == "4")))
rows.sort(key=lambda r: -r[4])
tot = np.array([r[1:] for r in rows]).sum(0)
print("starts", len(rows), "cpu %.0f s, G7 %d, classes %d, new %d, M4 %d" % tuple(tot))
print("distribution of new per start:", collections.Counter(r[4] for r in rows).most_common())
print("distribution of classes per start:", sorted(collections.Counter(r[3] for r in rows).items()))
for r in rows[:10]: print(r)
