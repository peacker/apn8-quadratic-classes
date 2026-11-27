import sys, glob, collections, numpy as np
R = "../sampler/db"   # learning/quad/db in the working tree (see README, Paths)
src = {}
with open(f"{R}/n8_index.tsv") as f:
    f.readline()
    for l in f:
        t = l.rstrip("\n").split("\t")
        if t[0] != "nonquadratic": src[t[0]] = t[3]
new = set()
with open(f"{R}/n8_new_classes.tsv") as f:
    f.readline()
    for l in f: new.add(l[:16])
keys = collections.Counter()
for fn in glob.glob("keys/k*.tsv"):
    for l in open(fn):
        t = l.split("\t")
        if t[1] == "Q": keys[t[2].strip()] += 1
L = set(keys)
print("functions", sum(keys.values()), "distinct classes", len(L))
print("multiplicity of classes:", sorted(collections.Counter(keys.values()).items())[:10])
inpub = L & set(src); innew = L & new; unk = L - set(src) - new
print("in published index", len(inpub), " in our new classes", len(innew), " in neither", len(unk))
c = collections.Counter()
for k in inpub:
    for s in src[k].split(","): c[s] += 1
print("by source:", c.most_common())
m = set(k for k, v in src.items() if "millions25" in v.split(","))
print("millions25 classes", len(m), " covered by Langevin file", len(m & L))
open("unknown_keys.txt", "w").write("".join(k + "\n" for k in sorted(unk)))
