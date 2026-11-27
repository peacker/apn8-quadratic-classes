#!/usr/bin/env python3
"""Evaluate testall outputs: functions, distinct classes, classes new to us (not in data/known.npy), and M values.
usage: evalrun.py LABEL CPU_SECONDS file.tsv ...   (prints one summary line; writes LABEL_new.tsv with new rows)"""
import numpy as np, os, sys, collections
here = os.path.dirname(os.path.abspath(__file__))
known = np.load(os.path.join(here, "data", "known.npy"))
label, cpu = sys.argv[1], float(sys.argv[2])
nf = 0; cls = {}; nonapn = 0
for fn in sys.argv[3:]:
    for l in open(fn):
        t = l.rstrip("\n").split("\t")
        if len(t) < 7: continue
        if t[2] != "1": nonapn += 1; continue
        nf += 1
        if t[3] not in cls: cls[t[3]] = t
invs = np.array([int(i, 16) for i in cls], dtype=np.uint64)
isk = np.isin(invs, known)
new = [t for (i, t), k in zip(cls.items(), isk) if not k]
Mall = collections.Counter(t[4] for t in cls.values()); Mnew = collections.Counter(t[4] for t in new)
with open(os.path.join(here, label + "_new.tsv"), "w") as f:
    for t in new: f.write("\t".join(t) + "\n")
h = cpu / 3600
print(f"{label}: cpu {h:.2f} h, functions {nf}, classes {len(cls)}, new {len(new)} ({len(new)/max(1,len(cls)):.1%}), "
      f"new/cpu-h {len(new)/h:.1f}, M all {dict(Mall)}, M new {dict(Mnew)}, M4 new/cpu-h {Mnew.get('4',0)/h:.2f}, nonAPN {nonapn}")
