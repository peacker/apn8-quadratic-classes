import sys, re, hashlib
import os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
import numpy as np
from apnsearch.invariants import ortho_invariant
c = {l.split("\t")[0]: l.split("\t")[2].strip() for l in open(sys.argv[2])}
bad = 0
for i, line in enumerate(open(sys.argv[1])):
    if i >= 60: break
    v = list(map(int, re.findall(r"\d+", line)))
    if len(v) != 256: continue
    k = hashlib.sha1(repr(ortho_invariant(np.array(v), 8)).encode()).hexdigest()[:16]
    bad += k != c[str(i)]
print("mismatch", bad)
