#!/usr/bin/env python3
"""Write testall-format rows (source line apn inv M status secs) for every function of Langevin's file into
learning/quad/local_results/langevin25/shard_<i>.tsv. Class key from orthokey.c (checked against apnsearch);
M and status are class invariants, taken from the index, the new classes, or testall (unknown20_test.tsv).
'line' is the 0-based line number in the decompressed file (line 0 is empty)."""
import glob, os
R = "../sampler"   # learning/quad in the working tree (see README, Paths)
ms = {}
for fn, a, b in ((f"{R}/db/n8_index.tsv", 1, 2), (f"{R}/db/n8_new_classes.tsv", 1, 2)):
    with open(fn) as f:
        f.readline()
        for l in f:
            t = l.split("\t", 3); ms.setdefault(t[0], (t[a], t[b]))
for l in open("unknown20_test.tsv"):
    t = l.split("\t"); ms[t[3]] = (t[4], t[5])
od = f"{R}/local_results/langevin25"; os.makedirs(od, exist_ok=True)
for fn in sorted(glob.glob("keys/k*.tsv")):
    i = fn.split("/k")[1].split(".")[0]
    with open(fn) as f, open(f"{od}/shard_{int(i):02d}.tsv", "w") as o:
        for l in f:
            n, st, k = l.rstrip("\n").split("\t")
            if st != "Q": continue
            M, s = ms[k]
            o.write(f"langevin25\t{n}\t1\t{k}\t{M}\t{s}\t0\n")
