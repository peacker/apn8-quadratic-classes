#!/usr/bin/env python3
"""Build data/known.npy: sorted uint64 of every class invariant we know (published index + our new classes).
usage: knownset.py  (reads learning/quad/db)"""
import numpy as np, os
db = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "learning", "quad", "db")
vals = []
for fn in ("n8_index.tsv", "n8_new_classes.tsv"):
    with open(os.path.join(db, fn)) as f:
        f.readline()
        vals.append(np.array([int(l[:16], 16) for l in f if l[16:17] == "\t"], dtype=np.uint64))
k = np.unique(np.concatenate(vals))
np.save(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "known.npy"), k)
print(len(k), "known invariants")
