import shutil
#!/usr/bin/env python3
"""runq.py: finish the n = 8 symmetric types 11, 14, 43 on this Mac (2026-11-19).
Jobs are nested slices (-sh I K D ...). A job running longer than TMAX seconds is killed and split into 32 children
one level deeper (-sh J 32 D+2), so a few huge subtrees never block the cores. Resumable: state in run/.
Slices finished earlier by queue.sh (q/t{T}_{I}_{K}.done, same partition) count as done.
usage: runq.py [JOBS] [TMAX] [TYPES as type:slices:depth,...]"""
import os, subprocess, sys, time, json
os.chdir(os.path.dirname(os.path.abspath(__file__)))
JOBS = int(sys.argv[1]) if len(sys.argv) > 1 else 15
TMAX = int(sys.argv[2]) if len(sys.argv) > 2 else 300
BIN = ["./sym8g", "-dfs", "-dedupe", "-symdepth", "28", "-cp", "3", "8", "-mtf", "-late", "7", "8", "-nsin", "1024", "-cpcache"]
os.makedirs("run/out", exist_ok=True)
def key(t, spec): return f"t{t}_" + "_".join(f"{i}-{k}-{d}" for i, k, d in spec)
done, split = {}, set()
if os.path.exists("run/done.tsv"):
    for l in open("run/done.tsv"): f = l.rstrip("\n").split("\t"); done[f[0]] = f
splitrule = {}                                               # key -> (K, D) of its children (old lines: 32, d + 2)
if os.path.exists("run/split.tsv"):
    for l in open("run/split.tsv"):
        f = l.split()
        if f: split.add(f[0]); splitrule[f[0]] = (int(f[1]), int(f[2])) if len(f) >= 3 else None
TYPES = sys.argv[3] if len(sys.argv) > 3 else "11:1000:8,14:1000:8,43:500:16"   # type:slices:depth, in run order
init = []
for x in TYPES.split(","):
    t, K, D = map(int, x.split(":")); init += [(t, [(i, K, D)]) for i in range(K)]
order = [int(x.split(":")[0]) for x in TYPES.split(",")]
def children(t, spec):
    i, k, d = spec[-1]
    r = splitrule.get(key(t, spec))
    K, D = r if r else (32, d + 2)
    return [(t, spec + [(j, K, D)]) for j in range(K)]
def choose_split(t, spec):
    """first depth below the slice with >= 64 nodes inside the slice (quick -countdepth runs); else d + 2"""
    d0 = spec[-1][2]
    args = BIN + [a for i, k, d in spec for a in ("-sh", str(i), str(k), str(d))]
    for D in range(d0 + 1, 28):
        try:
            r = subprocess.run(args + ["-countdepth", str(D)], stdin=open(f"W/w_{t}.txt"), capture_output=True, text=True, timeout=30)
        except subprocess.TimeoutExpired:
            break
        c = [l for l in r.stderr.splitlines() if l.startswith("count:")]
        if c and int(c[0].split(":")[2].split()[0]) >= 64: return (32, D)
    return (32, min(d0 + 2, 27))
todo = []
def expand(t, spec):
    kk = key(t, spec)
    if kk in done: return
    if len(spec) == 1:
        i, k, d = spec[0]; old = f"q/t{t}_{i}_{k}.done"
        if os.path.exists(old):
            txt = open(old).read().strip(); leaves = int(txt.split("leaves ")[1].split(",")[0])
            done[kk] = [kk, str(t), "q", txt.split("secs ")[-1], str(leaves), f"q/t{t}_{i}_{k}.out"]
            with open("run/done.tsv", "a") as h: h.write("\t".join(done[kk]) + "\n")
            return
    if kk in split:
        for c in children(t, spec): expand(*c)
    else: todo.append((t, spec))
for t, spec in init: expand(t, spec)
todo.sort(key=lambda x: order.index(x[0]))                  # stable: types in the given order
log = open("run/progress.log", "a")
def say(m): log.write(time.strftime("%F %T ") + m + "\n"); log.flush()
say(f"start: {len(done)} done, {len(todo)} to run, {len(split)} split, JOBS={JOBS} TMAX={TMAX}")
running = {}                                                 # proc -> (t, spec, start, outname)
last = 0
while todo or running:
    if shutil.disk_usage('.').free < 15 * 2**30:   # disk guard: a full disk stops macOS swap and causes 'out of memory'
        if time.time() - globals().get('_lastwarn', 0) > 600: say('LOW DISK (<15 GB free): not starting new jobs'); globals()['_lastwarn'] = time.time()
    else:
     while todo and len(running) < JOBS:
        t, spec = todo.pop(0); kk = key(t, spec)
        args = BIN + [a for i, k, d in spec for a in ("-sh", str(i), str(k), str(d))]
        deep = spec[-1][2] >= 27                             # cannot split further: no time limit
        out = open(f"run/out/{kk}.out", "w"); err = open(f"run/out/{kk}.err", "w")
        p = subprocess.Popen(["nice", "-n", "5"] + args, stdin=open(f"W/w_{t}.txt"), stdout=out, stderr=err)
        running[p] = (t, spec, time.time(), kk, deep)
    time.sleep(0.5)
    for p in list(running):
        t, spec, t0, kk, deep = running[p]
        if p.poll() is not None:
            del running[p]
            e = open(f"run/out/{kk}.err").read()
            dl = [l for l in e.splitlines() if l.startswith("done")]
            if p.returncode == 0 and dl:
                leaves = int(dl[0].split("leaves ")[1].split(",")[0])
                done[kk] = [kk, str(t), "runq", f"{time.time() - t0:.0f}", str(leaves), f"run/out/{kk}.out"]
                with open("run/done.tsv", "a") as h: h.write("\t".join(done[kk]) + "\n")
                if leaves == 0: os.remove(f"run/out/{kk}.out")
                else: subprocess.Popen(["nice", "gzip", "-f", f"run/out/{kk}.out"])   # postproc reads .out.gz
                os.remove(f"run/out/{kk}.err")
            else:
                say(f"FAILED {kk} rc={p.returncode}: retry"); todo.insert(0, (t, spec))
        elif not deep and time.time() - t0 > TMAX:
            p.kill(); p.wait(); del running[p]
            split.add(kk); splitrule[kk] = choose_split(t, spec)
            with open("run/split.tsv", "a") as h: h.write(f"{kk}\t{splitrule[kk][0]}\t{splitrule[kk][1]}\n")
            say(f"split {kk} into {splitrule[kk][0]} at depth {splitrule[kk][1]}")
            todo[0:0] = children(t, spec)
            os.remove(f"run/out/{kk}.out"); os.remove(f"run/out/{kk}.err")
    if time.time() - last > 60:
        last = time.time()
        per = {}
        for f in done.values(): per[f[1]] = per.get(f[1], 0) + 1
        say(f"done {len(done)} (by type {per}), running {len(running)}, queued {len(todo)}, split {len(split)}")
say("ALL DONE")
subprocess.run(["./postproc.sh"])
