#!/usr/bin/env python3
"""Rarefaction data and figure for the sampler (n = 8).
Each positive G7 gives one class (2 EA-equivalent functions), so one draw = one class occurrence.
Draws are put in a random order (seed 1): the curve is the expected discovery curve of the sampler.
usage: rarefaction.py         -> rarefaction.npz (snapshot), rarefaction.pdf
       rarefaction.py --plot  -> rarefaction.pdf from the existing rarefaction.npz (needs only numpy, matplotlib)"""
import glob, numpy as np, collections, sys, os
Q = "../../../learning/quad"
import re
if "--plot" in sys.argv:
    z = np.load("rarefaction.npz")
    wmid, r_all, r_new, D, csD, cnD = (z[k] for k in ("wmid", "r_all", "r_new", "D", "cs", "cn"))
    lp = float(z["lp"]) if "lp" in z else float(re.search(r"\\nLP\}\{([\d{,}]+)\}", open("../numbers.tex").read()).group(1).replace("{,}", ""))
else:
    pub, early = set(), set()
    with open(f"{Q}/db/n8_index.tsv") as f:
        f.readline()
        for l in f:
            t = l.rstrip("\n").split("\t")
            if t[0] == "nonquadratic": continue
            pub.add(t[0])
            if not set(t[3].split(",")) <= {"bgp26", "langevin25"}: early.add(t[0])   # as in tools/make_numbers.py
    # freeze the snapshot: read each result file only up to its size now; tools/make_numbers.py reads the same bytes
    # counters of the uniform sampler first: a sync copies the rows before the counters, so every counter
    # line recorded here has its rows in the files recorded next
    sums = sorted(glob.glob(f"{Q}/server_results/epyc*/gen/m*.sum")); sumsizes = [os.path.getsize(fn) for fn in sums]
    files = sorted(glob.glob(f"{Q}/server_results/epyc*/results/varext/*.tsv"))
    sizes = [os.path.getsize(fn) for fn in files]
    def lines(fn, limit):
        with open(fn, "rb") as f:
            pos = 0
            for l in f:
                pos += len(l)
                if pos > limit: break
                yield l.decode()
    cnt = collections.Counter()
    for fn, sz in zip(files, sizes):
        for l in lines(fn, sz):
            if l.startswith("mass_"): continue          # uniform sampler: not part of this figure
            t = l.split("\t", 4)
            if len(t) > 4 and t[2] == "1": cnt[t[3]] += 1
    keys = list(cnt); lp = len(early) * len(keys) / len(early & set(keys))   # Lincoln-Petersen estimate
    ispub = np.array([k in pub for k in keys])
    reps = np.array([cnt[k] // 2 for k in keys], dtype=np.int64)
    draws = np.repeat(np.arange(len(keys), dtype=np.int32), reps)
    rng = np.random.default_rng(1); rng.shuffle(draws)
    _, first = np.unique(draws, return_index=True)
    isfirst = np.zeros(len(draws), bool); isfirst[first] = True
    isnew = isfirst & ~ispub[draws]
    W = 200000
    nwin = len(draws) // W
    r_all = isfirst[:nwin * W].reshape(nwin, W).mean(1); r_new = isnew[:nwin * W].reshape(nwin, W).mean(1)
    wmid = (np.arange(nwin) + 0.5) * W
    cs = np.cumsum(isfirst); cn = np.cumsum(isnew)
    step = 100000; D = np.arange(step, len(draws) + 1, step); csD, cnD = cs[D - 1], cn[D - 1]
    np.savez("rarefaction.npz", wmid=wmid, r_all=r_all, r_new=r_new, D=D, cs=csD, cn=cnD, ntot=len(draws), lp=lp,
             files=np.array([os.path.relpath(fn, Q) for fn in files]), sizes=np.array(sizes, np.int64),
             sumfiles=np.array([os.path.relpath(fn, Q) for fn in sums]), sumsizes=np.array(sumsizes, np.int64))
    print("draws", len(draws), "classes", len(keys), "new", int(isnew.sum()), "last window share unseen %.4f new %.4f" % (r_all[-1], r_new[-1]))
try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
except ImportError:
    sys.exit("rarefaction.npz written; matplotlib missing, rarefaction.pdf NOT updated")
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.6), constrained_layout=True)
M = 1e6
a.plot(wmid / M, 100 * r_all, ".", ms=2, color="#2a78d6", label="class not seen before")
a.plot(wmid / M, 100 * r_new, ".", ms=2, color="#eb6834", label="... and not in published lists")
a.set_xlabel("draws (millions)"); a.set_ylabel("% of draws"); a.set_ylim(0, 100); a.legend(frameon=False, fontsize=8)
a.set_title("(a) Share of draws that give a new class", fontsize=9, loc="left")
b.plot(D / M, csD / M, "-", color="#2a78d6", label="classes seen")
b.plot(D / M, cnD / M, "-", color="#eb6834", label="not in published lists")
b.axhline(lp / M, color="#2a78d6", ls="--", lw=0.8); b.text(0.3, lp / M + 0.09, f"estimated total {lp / M:.2f} M", fontsize=8, color="#2a78d6")
b.set_xlabel("draws (millions)"); b.set_ylabel("classes (millions)"); b.set_ylim(0, 6.5); b.legend(frameon=False, fontsize=8, loc="center right")
b.set_title("(b) Classes found", fontsize=9, loc="left")
fig.savefig("rarefaction.pdf")
