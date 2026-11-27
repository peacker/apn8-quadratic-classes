#!/usr/bin/env python3
"""Build the n = 8 quadratic APN class index and the report of new classes.

Input: result TSVs of testall.py (source line apn inv M status seconds [lut]) under the given directories:
  published lists (point 1): results/<source>/shard_*.tsv
  sampler (point 3):         results/varext/w*.tsv
Output (in OUTDIR):
  n8_index.tsv        inv  M  status  sources(comma)  count     one line per class (ortho-derivative invariant)
  n8_new_classes.tsv  inv  M  status  first_source  lut       sampler classes not in any published list
  report.md           counts per source, overlaps, new classes, permutations (if any)
usage: build_db.py OUTDIR RESULTDIR [RESULTDIR ...]
"""
import collections, glob, os, sys

out, dirs = sys.argv[1], sys.argv[2:]
os.makedirs(out, exist_ok=True)
cls = {}                                   # inv -> [M, status, set(sources), count]
src_count = collections.Counter()
src_inv = collections.defaultdict(set)
nonapn = collections.Counter()
perms = []
sampler = {}                               # inv -> (M, status, source, lut)
for d in dirs:
    for f in glob.glob(os.path.join(d, "*", "*.tsv")):
        sampler_name = os.path.basename(os.path.dirname(f))
        is_sampler = sampler_name in ("varext", "switch2", "gen6", "halfcompl")
        for line in open(f):
            t = line.rstrip("\n").split("\t")
            if len(t) < 7:
                continue
            src = sampler_name if is_sampler else t[0]
            if t[2] != "1":
                nonapn[src] += 1
                continue
            inv, M, st = t[3], t[4], t[5]
            if st == "permutation":
                perms.append((src, t[1], inv))
            if is_sampler:
                if inv not in sampler or (len(t) > 7 and not sampler[inv][3]):
                    sampler[inv] = (M, st, t[0], t[7] if len(t) > 7 else "")
                src_count[src] += 1
                src_inv[src + " (search)"].add(inv)
                continue
            src_count[src] += 1
            src_inv[src].add(inv)
            c = cls.setdefault(inv, [M, st, set(), 0])
            c[2].add(src)
            c[3] += 1

with open(os.path.join(out, "n8_index.tsv"), "w") as f:
    f.write("inv\tM\tstatus\tsources\tcount\n")
    for inv, (M, st, ss, c) in sorted(cls.items()):
        f.write(f"{inv}\t{M}\t{st}\t{','.join(sorted(ss))}\t{c}\n")
new = {inv: v for inv, v in sampler.items() if inv not in cls}
with open(os.path.join(out, "n8_new_classes.tsv"), "w") as f:
    f.write("inv\tM\tstatus\tfirst_source\tlut\n")
    for inv, (M, st, s, lut) in sorted(new.items()):
        f.write(f"{inv}\t{M}\t{st}\t{s}\t{lut}\n")

L = ["# n = 8 quadratic APN classes: index and report", ""]
L += ["| source | APN functions | distinct classes (invariant) | classes only in this source |", "|---|---|---|---|"]
for s in sorted(x for x in src_inv if not x.endswith("(search)")):
    only = sum(1 for inv in src_inv[s] if cls[inv][2] == {s})
    L.append(f"| {s} | {src_count[s]} | {len(src_inv[s])} | {only} |")
L += ["", f"**Published lists together: {len(cls)} classes.**"]
st = collections.Counter(v[1] for v in cls.values())
L.append(f"Permutation test: {dict(st)}; M distribution: {dict(collections.Counter(v[0] for v in cls.values()))}")
for name in ("varext", "switch2", "gen6", "halfcompl"):
    sinv = src_inv.get(name + " (search)", set())
    snew = sum(1 for i in sinv if i not in cls)
    L.append(f"**Search {name}:** {src_count[name]} APN functions, {len(sinv)} classes, {snew} not in any published list")
L += ["", f"**All searches together: {len(new)} new classes**; test: {dict(collections.Counter(v[1] for v in new.values()))}"]
if nonapn:
    L.append(f"Non-APN lines (should be 0): {dict(nonapn)}")
L += ["", "**Permutations found: " + (str(perms) if perms else "none") + "**"]
open(os.path.join(out, "report.md"), "w").write("\n".join(L) + "\n")
print("\n".join(L))
