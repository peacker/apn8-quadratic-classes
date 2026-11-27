#!/usr/bin/env python3
"""Compute every count used in the paper from one snapshot and write numbers.tex (LaTeX macros).
Inputs: learning/quad/db/n8_index.tsv (published lists, incl. bgp26), the sampler result files
learning/quad/server_results/epyc*/results/varext/*.tsv, the symmetric classes (SYM file: inv in column 5),
the profile outputs (n8search/pilots/data/prof_new.tsv, prof_pub.tsv),
the per-call counters of the uniform sampler (server_results/epyc*/gen/m*.sum).
usage: tools/make_numbers.py SYMFILE > numbers.tex"""
import glob, collections, sys, datetime, math, statistics
import numpy as np
R = "../../learning/quad"
def fmt(x): return f"{x:,}".replace(",", "{,}")
def pct(a, b, d=1): return f"{100 * a / b:.{d}f}"
idx = {}
with open(f"{R}/db/n8_index.tsv") as f:
    f.readline()
    for l in f:
        t = l.rstrip("\n").split("\t")
        if t[0] == "nonquadratic": continue
        idx[t[0]] = (t[1], t[2], set(t[3].split(",")))
LATE = {"bgp26", "langevin25"}               # lists we compare with after the fact (not used for Lincoln-Petersen)
early = {i for i, v in idx.items() if not v[2] <= LATE}
bgp = {i for i, v in idx.items() if "bgp26" in v[2]}
bgpnew = {i for i, v in idx.items() if v[2] <= LATE and "bgp26" in v[2]}
lang = {i for i, v in idx.items() if "langevin25" in v[2]}
langnew = {i for i, v in idx.items() if v[2] == {"langevin25"}}
z = np.load("figures/rarefaction.npz")          # snapshot: the files and byte sizes read by figures/rarefaction.py
snap = {f"{R}/{fn}": int(sz) for fn, sz in zip(z["files"], z["sizes"])}
sumsnap = {f"{R}/{fn}": int(sz) for fn, sz in zip(z["sumfiles"], z["sumsizes"])} if "sumfiles" in z.files else None
def lines(fn, sn=snap):
    with open(fn, "rb") as f:
        pos = 0
        for l in f:
            pos += len(l)
            if pos > sn[fn]: break
            yield l.decode()
cnt = collections.Counter(); Mof = {}; st = collections.Counter()
# uniform sampler rows (tens of millions): per source (one call) the number of rows, and for each draw (rows 2i, 2i+1
# of a call are one positive G7) the key of row 2i as an integer, with the source id; compact arrays, not tuples
import array
sid = {}; nrows = array.array("q"); mkeys = array.array("Q"); msid = array.array("I")
mstat = collections.Counter()
for fn in snap:
    if True:
        for l in lines(fn):
            t = l.split("\t", 7)
            if len(t) < 7: continue                 # partial last line of a file being synced
            if l.startswith("mass_"):               # uniform sampler: counted separately below
                assert t[2] == "1", l[:80]          # a failed key would drop the draw from f
                j = sid.get(t[0])
                if j is None: j = sid[t[0]] = len(nrows); nrows.append(0)
                nrows[j] += 1
                if int(t[1]) % 2 == 0: mkeys.append(int(t[3], 16)); msid.append(j)
                if t[3] not in Mof: Mof[t[3]] = t[4]
                mstat[t[5]] += 1
                continue
            if t[2] != "1": continue
            st[t[5]] += 1
            cnt[t[3]] += 1; Mof[t[3]] = t[4]
mkeys = np.frombuffer(mkeys, np.uint64); msid = np.frombuffer(msid, np.uint32)
def nsrc(src): j = sid.get(src); return 0 if j is None else nrows[j]
S = set(cnt); new = S - set(idx); both = S & early
def chao(cl):
    fc = collections.Counter(cnt[i] // 2 for i in cl)
    return len(cl) + fc[1] ** 2 / (2 * fc[2])
out = {}
out["nPubEarly"] = len(early); out["nBgpNew"] = len(bgpnew); out["nPub"] = len(idx)
out["nFunctions"] = sum(cnt.values()); out["nDraws"] = sum(cnt.values()) // 2
out["nSamplerClasses"] = len(S); out["nSamplerPub"] = len(S & set(idx)); out["nNew"] = len(new)
lp = len(early) * len(S) / len(both); ch = chao(S)
out["nBothEarly"] = len(both); out["nLP"] = round(lp); out["nChao"] = round(ch)
out["pctLPChao"] = pct(abs(lp - ch), lp, 2)
pubM4 = {i for i in idx if idx[i][0] == "4"}; S4 = {i for i in S if Mof[i] == "4"}
early4 = pubM4 & early
out["nMfourPub"] = len(pubM4); out["nMfourNew"] = len(S4 - set(idx))
out["nMfourLP"] = round(len(early4) * len(S4) / len(early4 & S4)); out["nMfourChao"] = round(chao(S4))
out["nBgpSeen"] = len(bgpnew & S); out["pctBgpSeen"] = pct(len(bgpnew & S), len(bgpnew))
symrows = [l.split("\t") for l in open(sys.argv[1])]
sym = {t[4] for t in symrows}
SYMW = sum(256 / int(t[3]) for t in symrows)      # sum of 1/|G(C)| over the symmetric classes
out["nSym"] = len(sym); out["nSymSeen"] = len(sym & S); out["pctSymSeen"] = pct(len(sym & S), len(sym))
out["drawsPerSym"] = f"{sum(cnt[i] // 2 for i in sym & S) / len(sym):.2f}"
out["drawsPerClass"] = f"{out['nDraws'] / lp:.2f}"
sl = out["nDraws"] / lp / 3                       # Poisson model for |G(C)| = 3: mean draws lambda/3
out["symLamThird"] = f"{sl:.2f}"; out["pctSymPred"] = pct(1 - math.exp(-sl), 1)
out["pctSeen"] = pct(len(S), lp)
G7_PER_POS, G7_PER_CORE_HOUR = 2029, 365000      # measured on the servers (sections/sampler.tex)
e7 = out["nDraws"] * G7_PER_POS
out["nGsevenTests"] = f"{e7 / 10 ** 10:.1f}\\cdot10^{{10}}"; out["coreHours"] = f"{e7 / G7_PER_CORE_HOUR / 10 ** 5:.1f}\\cdot10^5"
fc = collections.Counter(cnt[i] // 2 for i in S)
out["nFone"] = fc[1]; out["nFtwo"] = fc[2]; out["pctGT"] = pct(fc[1], out["nDraws"])
if int(z["ntot"]) != out["nDraws"]: sys.exit(f"figures/rarefaction.npz has {int(z['ntot'])} draws, the data {out['nDraws']}: rerun figures/rarefaction.py")
out["pctLastUnseen"] = f"{100 * z['r_all'][-1]:.1f}"; out["pctLastNew"] = f"{100 * z['r_new'][-1]:.1f}"
out["nLangFunctions"] = sum(1 for fn in glob.glob(f"{R}/local_results/langevin25/*.tsv") for _ in open(fn))
out["nLangClasses"] = len(lang); out["nLangNew"] = len(langnew); out["nLangSeen"] = len(langnew & S)
out["nLangUnseen"] = len(langnew - S); out["pctLangSeen"] = pct(len(langnew & S), len(langnew))
# zero-truncated Poisson fit to the draw counts of the classes without a self-equivalence that are in no early list
hc = collections.Counter(cnt[i] // 2 for i in S if i not in early and i not in sym)
Sg = sum(hc.values()); mean = sum(k * c for k, c in hc.items()) / Sg
lam = mean
for _ in range(200): lam = mean * (1 - math.exp(-lam))
unseen = Sg * math.exp(-lam) / (1 - math.exp(-lam))
out["nFitSeen"] = Sg; out["fitLambda"] = f"{lam:.2f}"; out["nFitUnseen"] = round(unseen)
out["pctFitPred"] = pct(1 - math.exp(-lam), 1)
out["nFitFoneFmt"] = hc[1]; out["nFitFonePredFmt"] = round(Sg * lam * math.exp(-lam) / (1 - math.exp(-lam)))
out["nFitFtenFmt"] = hc[10]; out["nFitFtenPredFmt"] = round(Sg * math.exp(-lam) * lam ** 10 / math.factorial(10) / (1 - math.exp(-lam)))
# zero-truncated negative binomial (Poisson-gamma) on the same counts: coefficient of variation 1/sqrt(a) of the rates
from scipy.optimize import minimize
from scipy.special import gammaln
ks = np.array(sorted(hc), float); cs = np.array([hc[int(k)] for k in ks], float)
def nbll(x):
    l, a = math.exp(x[0]), math.exp(x[1])
    lp_ = gammaln(ks + a) - gammaln(a) - gammaln(ks + 1) + ks * np.log(l / (l + a)) + a * np.log(a / (l + a))
    return -(cs * (lp_ - math.log1p(-(a / (l + a)) ** a))).sum()
rn = minimize(nbll, [math.log(lam), math.log(1000)], method="Nelder-Mead", options={"maxiter": 4000, "xatol": 1e-9, "fatol": 1e-6})
out["nbCV"] = f"{100 / math.sqrt(math.exp(rn.x[1])):.0f}"
hs = collections.Counter(cnt[i] // 2 for i in sym & S); ls = sum(k * c for k, c in hs.items()) / sum(hs.values()); l3 = ls
for _ in range(200): l3 = ls * (1 - math.exp(-l3))
out["nSymUnseen"] = len(sym - S); out["nSymUnseenPred"] = round(sum(hs.values()) * math.exp(-l3) / (1 - math.exp(-l3)))
assert set(st) <= {"none"} and set(mstat) <= {"none"}, (st, mstat)   # no draw is CCZ-equivalent to a permutation
for nm in ("new", "pub"):
    rows = [l.split() for l in open(f"../../n8search/pilots/data/prof_{nm}.tsv")]
    n = len(rows); reach = sum(r[6] == "1" for r in rows)
    out[f"profN{nm}"] = n; out[f"profReach{nm}"] = pct(reach, n)
    out[f"profBfive{nm}"] = pct(sum(r[3] == "28" for r in rows), n)
    out[f"profBsix{nm}"] = pct(sum(r[3] == "30" and r[5] in ("52", "50") for r in rows), n)
    out[f"profBentFour{nm}"] = pct(sum(int(r[2]) > 0 for r in rows), n, 2)
# uniform sampler (n8search/mass, rows with source mass_w<ID>_k<k>). Counter lines: 9 fields ending in
# "splitmix2" = splitmix64 with a hashed seed (used); 8 fields = xorshift64 (reported separately, biased);
# "splitmix" = splitmix64 with the faulty seeding of 2026-11-22 08:50-09:19 (left out). A call is used only if
# all its rows have synced (number of rows = outputs) and its search never hit the node limit, so that f and
# N_beta come from exactly the same draws.
calls = {"s": [], "x": []}
if sumsnap is None:   # older snapshot without the counter files: keep only calls up to the last call of the same
    # server whose outputs are in the snapshot (later calls with no output would otherwise slip in)
    cut = collections.Counter()
    for fn in glob.glob(f"{R}/server_results/epyc*/gen/m*.sum"):
        srv = fn.split("/server_results/")[1].split("/")[0]; wid = fn.rsplit("/m", 1)[1][:-4]
        for l in open(fn):
            t = l.split()
            if len(t) >= 8 and int(t[6]) > 0 and nsrc(f"mass_w{wid}_{t[1].replace('=', '')}") == int(t[6]):
                cut[srv] = max(cut[srv], int(t[0]))
for fn in (sumsnap or glob.glob(f"{R}/server_results/epyc*/gen/m*.sum")):
    srv = fn.split("/server_results/")[1].split("/")[0]; wid = fn.rsplit("/m", 1)[1][:-4]
    L = [l.split() for l in (lines(fn, sumsnap) if sumsnap else open(fn))]
    L = [t for t in L if len(t) >= 8 and (sumsnap or int(t[0]) <= cut[srv])]
    # duration of a call = time since the previous counter line of the same worker (one process per worker);
    # a gap longer than 3 medians (worker restarted) and the first call count as one median call
    dt = [int(b[0]) - int(a[0]) for a, b in zip(L, L[1:])]; md = statistics.median(dt) if dt else 0
    dur = [md] + [d if d <= 3 * md else md for d in dt]
    for t, du in zip(L, dur):
        if len(t) == 9 and t[8] in ("splitmix2", "splitmix2g"): g = "s"   # splitmix2g: GPU sampler (n8search/cuda), same draws
        elif len(t) == 8: g = "x"
        else: continue
        src = f"mass_w{wid}_{t[1].replace('=', '')}"; o = int(t[6])
        if nsrc(src) != o: continue       # rows not synced yet
        assert int(t[7]) < 10 ** 6, l                  # node limit hit: outputs would be undercounted
        calls[g].append((srv, int(t[0]), int(t[2]), int(t[3]), int(t[4]), o, src, du, t[-1] == "splitmix2g", int(t[7])))
K = set(idx) | S
WK = len(K) - len(sym) + SYMW
Karr = np.array(sorted(int(k, 16) for k in K), np.uint64)
allfair = np.setdiff1d(np.unique(mkeys), Karr)          # classes outside K found by the uniform sampler (any generator)
g = 1
for i in range(8): g *= 256 - 2 ** i
def draws_of(cl):                                       # keys of the draws of these calls, in the order of the calls
    ids = np.array([sid[c[6]] for c in cl if c[6] in sid], np.int64)   # a call without outputs has no rows
    rank = np.full(len(nrows), -1, np.int64); rank[ids] = np.arange(len(ids))
    r = rank[msid]; sel = r >= 0
    order = np.argsort(r[sel], kind="stable")
    return mkeys[sel][order]
def uniform(cl):
    keys = draws_of(cl)
    nf = len(keys); out_ = ~np.isin(keys, Karr); no = int(out_.sum()); fo = no / nf
    m = WK * fo / (1 - fo); mse = WK * math.sqrt(fo * (1 - fo) / nf) / (1 - fo) ** 2
    A = sum(c[2] for c in cl); B = sum(c[3] for c in cl); T = sum(c[4] for c in cl); O = sum(c[5] for c in cl)
    Nb = 2 ** 168 * (A / B) * O * 64 / T; rel = 1 / math.sqrt(O // 2)
    cls = Nb / g / g - SYMW + len(sym)
    return dict(nf=nf, no=no, fo=fo, m=m, mse=mse, Nb=Nb, rel=rel, pos=O // 2, cls=cls, cse=Nb / g / g * rel, keys=keys, out=out_)
calls["s"].sort(key=lambda c: c[1])                 # time order of the calls (for the discovery curve)
u, x = uniform(calls["s"]), uniform(calls["x"])
assert u["nf"] == u["pos"], (u["nf"], u["pos"])
m = u["m"]
out["symW"] = f"{SYMW:,.2f}".replace(",", "{,}"); out["nKnownNoFair"] = len(K)
out["nFairDraws"] = u["nf"]; out["nFairOut"] = u["no"]; out["pctFairOut"] = pct(u["no"], u["nf"], 2)
out["nFairOutside"] = int(round(m, -2)); out["nFairOutsideSE"] = int(round(u["mse"], -1)); out["nFairFound"] = len(allfair)
out["nFairMissing"] = int(round(m - len(allfair), -2)); out["nFairTotal"] = int(round(len(K) + m, -3))
KA = K | {f"{int(k):016x}" for k in allfair}
out["nKnown"] = len(KA); out["nMissing"] = round(lp) - out["nKnown"]; out["pctKnown"] = pct(out["nKnown"], lp)
Mk = collections.Counter(idx[i][0] if i in idx else Mof[i] for i in KA)
out["nMtwoKnown"] = Mk["2"]; out["nMthreeKnown"] = Mk["3"]; out["nMfourKnown"] = Mk["4"]
assert sum(Mk.values()) == len(KA), Mk
e = math.floor(math.log10(u["Nb"]))
out["massNbeta"] = f"{u['Nb'] / 10 ** e:.4f}\\cdot10^{{{e}}}"; out["massRelSE"] = f"{100 * u['rel']:.2f}"
out["nMassClasses"] = int(round(u["cls"], -2)); out["nMassClassesSE"] = int(round(u["cse"], -2))
# xorshift64: the same two counts, for the comparison in the paper
out["nXorPos"] = x["pos"]; out["pctXorOut"] = pct(x["no"], x["nf"], 3)
out["nXorClasses"] = int(round(x["cls"], -3)); out["nXorClassesSE"] = int(round(x["cse"], -2))
out["pctXorExcess"] = pct(x["cls"] - u["cls"], u["cls"], 1)
out["xorSigma"] = f"{(x['cls'] - u['cls']) / math.hypot(x['cse'], u['cse']):.1f}"
# rate of the uniform sampler: positives per day over the window that every server covers
by = collections.defaultdict(list)
for c in calls["s"]: by[c[0]].append(c[1])
t0 = max(min(v) for v in by.values()); t1 = min(max(v) for v in by.values())
rate = sum(c[5] // 2 for c in calls["s"] if t0 < c[1] <= t1) / ((t1 - t0) / 86400)
out["fairDrawsPerDay"] = f"{rate / 1e6:.1f}"
# cost: CPU calls in core-hours (one process per call), GPU calls in GPU-hours (one GPU per call)
cc = [c for c in calls["s"] if not c[8]]; gc = [c for c in calls["s"] if c[8]]
ch = sum(c[7] for c in cc) / 3600; cpos = sum(c[5] // 2 for c in cc)
out["fairCoreHours"] = int(round(ch, -2))
def rate_of(cl): return sum(c[5] // 2 for c in cl) / (sum(c[7] for c in cl) / 3600)
# speed by hardware: servers epyc1-4 are 2 x AMD EPYC 7763; epyc5-8 have Turing GPUs (RTX 2080 Ti, Quadro RTX 8000,
# TITAN RTX) and Xeon Gold 5222 CPUs; epyc9 has A100 GPUs and an EPYC 7742 (CPU calls there and on epyc5-8 count in
# the core-hours, but not in the rate per EPYC 7763 thread)
E7763 = {"epyc1", "epyc2", "epyc3", "epyc4"}
r7763 = rate_of([c for c in cc if c[0] in E7763]); out["fairDrawsPerCoreHour"] = round(r7763)
out["fairDrawsPerSocket"] = int(round(128 * r7763, -2))   # one EPYC 7763 processor: 64 cores, 128 threads
out["pctCoreHoursEpyc"] = round(100 * sum(c[7] for c in cc if c[0] in E7763) / 3600 / ch)
out["fairHalvingCoreHours"] = int(round(math.log(2) * (WK + m) / (cpos / ch), -3))
if gc:
    gh = sum(c[7] for c in gc) / 3600; gpos = sum(c[5] // 2 for c in gc)
    out["fairGpuHours"] = int(round(gh)); out["fairDrawsPerGpuHour"] = int(round(gpos / gh, -2))
    out["fairDrawsPerGpuHourTuring"] = int(round(rate_of([c for c in gc if c[0] != "epyc9"]), -2))
    out["fairDrawsPerGpuHourAmpere"] = int(round(rate_of([c for c in gc if c[0] == "epyc9"]), -2))
    out["gpuPerSocket"] = f"{out['fairDrawsPerGpuHourTuring'] / (128 * out['fairDrawsPerCoreHour']):.1f}"
    out["aHundredPerSocket"] = f"{out['fairDrawsPerGpuHourAmpere'] / (128 * out['fairDrawsPerCoreHour']):.1f}"
hc = math.log(2) * (WK + m) / (cpos / ch); mleft = max(m - len(allfair), 2)
out["fairHalvings"] = round(math.log2(mleft)); e = math.floor(math.log10(hc * math.log2(mleft)))
out["fairLastCoreHours"] = f"{hc * math.log2(mleft) / 10 ** e:.0f}\\cdot10^{{{e}}}"
out["fairHalvingDays"] = f"{math.log(2) * (WK + m) / rate:.1f}"
out["fairWeight"] = f"{(WK + m) / 1e6:.2f}"
tmax = max(c[1] for c in calls["s"])
dd = datetime.datetime.fromtimestamp(tmax, datetime.timezone.utc).date(); out["snapDate"] = f"{dd.day}~{dd:%B}~{dd.year}"
# the abstract and the text say that the two counts of the uniform sampler agree
gap = (u["cls"] - len(K) - m) / math.hypot(u["cse"], u["mse"])
out["countsSigma"] = f"{abs(gap):.1f}"
assert abs(gap) < 2, (len(K) + m, u["cls"], u["cse"], u["mse"])
# is the search complete? Every class outside K has a trivial group, so each draw hits it with probability 1/W,
# W = W_K + (classes outside K). After n draws a given class is still unseen with probability (1 - 1/W)^n.
keys, om = u["keys"], u["out"]
po = np.nonzero(om)[0]; uq, fi, cn = np.unique(keys[po], return_index=True, return_counts=True)
D = len(uq); lam_o = len(po) / D; fk = collections.Counter(cn.tolist())
first = np.sort(po[fi])                                # draw index (time order) at which each class was first drawn
nf = u["nf"]; Wf = WK + len(allfair); lamW = nf / Wf
out["nUniOutDistinct"] = D; out["uniDrawsPerOut"] = f"{lam_o:.1f}"; out["uniFone"] = fk[1]; out["uniFtwo"] = fk[2]
out["uniChaoUnseen"] = f"{fk[1] ** 2 / (2 * fk[2]):.1f}" if fk[1] and fk[2] else "0"
out["uniPoissonUnseen"] = f"{D * math.exp(-lam_o) / (1 - math.exp(-lam_o)):.2f}"
out["uniLamW"] = f"{lamW:.1f}"; e = math.floor(math.log10(math.exp(-lamW)))
out["uniMissProb"] = f"{math.exp(-lamW) / 10 ** e:.0f}\\cdot10^{{{e}}}"
Xmax = u["m"] + 3 * u["mse"]; out["uniXmax"] = int(round(Xmax, -2)); v = Xmax * math.exp(-lamW); e = math.floor(math.log10(v))   # an upper bound: round up
out["uniExpMissing"] = f"{v:.2f}" if v >= 0.01 else f"{math.ceil(v / 10 ** e)}\\cdot10^{{{e}}}"
fpred = len(allfair) / Wf; out["uniFpred"] = f"{100 * fpred:.3f}"; out["pctFairOutFine"] = f"{100 * u['fo']:.3f}"
out["uniFsigma"] = f"{(u['fo'] - fpred) / math.sqrt(fpred * (1 - fpred) / nf):.1f}"
out["nFairOutsideExact"] = int(round(u["m"])); out["nFairOutsideSEexact"] = int(round(u["mse"]))
out["nFairExcess"] = int(round(u["m"] - len(allfair))); out["nFairTotalFine"] = int(round(len(K) + u["m"], -2))
out["uniLastNewM"] = f"{(nf - 1 - first[-1]) / 1e6:.1f}"          # draws (millions) since the last new class
L10 = 10 ** 7; out["uniNewLastTenM"] = int((first >= nf - L10).sum())
half = nf // 2; out["uniNewSecondHalf"] = int((first >= half).sum()); out["uniNewFirstHalf"] = int((first < half).sum())
out["nNonstd"] = sum(c[9] for c in calls["s"]); out["nFairPositives"] = u["pos"]
for nm, cl in (("Cpu", [c for c in calls["s"] if not c[8]]), ("Gpu", [c for c in calls["s"] if c[8]])):
    if cl:
        v = uniform(cl); out[f"uni{nm}Draws"] = v["nf"]; out[f"pctUni{nm}Out"] = f"{100 * v['fo']:.3f}"
        out[f"uni{nm}SE"] = f"{100 * math.sqrt(v['fo'] * (1 - v['fo']) / v['nf']):.3f}"
# the same, with the published lists P alone as the known set (for the abstract: no first sampler)
P = set(idx); assert set(sym) <= P
Parr = np.array(sorted(int(k, 16) for k in P), np.uint64); WP = len(P) - len(sym) + SYMW
outP = ~np.isin(keys, Parr); noP = int(outP.sum()); fP = noP / nf
mP = WP * fP / (1 - fP); mPse = WP * math.sqrt(fP * (1 - fP) / nf) / (1 - fP) ** 2
poP = np.nonzero(outP)[0]; uqP, fiP, cnP = np.unique(keys[poP], return_index=True, return_counts=True)
DP = len(uqP); lamP = nf / (WP + DP); fkP = collections.Counter(cnP.tolist())
out["pubPctOut"] = f"{100 * fP:.2f}"; out["pubOutside"] = int(round(mP, -2)); out["pubOutsideSE"] = int(round(mPse))
out["pubTotal"] = int(round(len(P) + mP, -2)); out["pubFound"] = DP; out["pubDrawsPerOut"] = f"{len(poP) / DP:.1f}"
out["pubFone"] = fkP[1]; out["pubLamW"] = f"{lamP:.1f}"
XP = mP + 3 * mPse; v = XP * math.exp(-lamP); e = math.floor(math.log10(v))
out["pubExpMissing"] = f"{v:.2f}" if v >= 0.01 else f"{math.ceil(v / 10 ** e)}\\cdot10^{{{e}}}"
uall = set(f"{int(k):016x}" for k in np.unique(keys))
out["nUniClasses"] = len(uall); out["nUniNew"] = DP; out["nNewNotUni"] = len(new - uall); out["nNewAll"] = len((S | KA) - P)
out["nKnownNotUni"] = len(KA - uall)
# what the uniform sampler found on its own (the paper is built around it)
out["nUniPub"] = len(uall & P); unew = uall - P
out["pctXorOutPub"] = f"{100 * float((~np.isin(x['keys'], Parr)).mean()):.2f}"
out["nUniBgpSeen"] = len(bgpnew & uall); out["pctUniBgpSeen"] = pct(len(bgpnew & uall), len(bgpnew))
out["nUniLangSeen"] = len(langnew & uall); out["nUniLangUnseen"] = len(langnew - uall)
out["pctUniLangSeen"] = pct(len(langnew & uall), len(langnew))
out["nUniMfourNew"] = sum(1 for i in unew if Mof[i] == "4"); out["nUniMtwoNew"] = sum(1 for i in unew if Mof[i] == "2")
out["pubFtwo"] = fkP[2]; e = math.floor(math.log10(math.exp(-lamP)))
out["pubMissProb"] = f"{math.exp(-lamP) / 10 ** e:.0f}\\cdot10^{{{e}}}"; out["pubXmax"] = int(round(XP, -2))
firstP = np.sort(poP[fiP]); out["pubLastNewM"] = f"{(nf - 1 - firstP[-1]) / 1e6:.1f}"
out["pubNewSecondHalf"] = int((firstP >= half).sum()); out["pubNewFirstHalf"] = int((firstP < half).sum())
out["pubOutsideExact"] = int(round(mP)); out["pubTotalExact"] = int(round(len(P) + mP))
out["pubSigma"] = f"{(mP - DP) / mPse:.1f}"
fpP = DP / (WP + DP); out["pubFpred"] = f"{100 * fpP:.3f}"; out["pctPubOutFine"] = f"{100 * fP:.3f}"
for nm, cl in (("Cpu", [c for c in calls["s"] if not c[8]]), ("Gpu", [c for c in calls["s"] if c[8]])):
    kk = draws_of(cl); ff = float((~np.isin(kk, Parr)).mean())
    out[f"pctPub{nm}Out"] = f"{100 * ff:.2f}"; out[f"pub{nm}SE"] = f"{100 * math.sqrt(ff * (1 - ff) / len(kk)):.2f}"
gridP = np.linspace(0, nf, 401).astype(np.int64)
np.savez("figures/uniform.npz", grid=gridP, found=np.searchsorted(firstP, gridP), m=mP, mse=mPse, D=DP, WK=WP)
# two halves of the uniform draws (time order): the first half and P give a known set K1 that does not depend on
# the second half; the share of second-half draws outside K1 counts the classes outside K1 without a model
h1 = keys[:half]; h2 = keys[half:]; n2 = len(h2)
K1arr = np.union1d(Parr, np.unique(h1)); W1 = WP + (len(K1arr) - len(Parr))
o2 = ~np.isin(h2, K1arr); no2 = int(o2.sum()); f2 = no2 / n2
m1 = W1 * f2 / (1 - f2); m1se = W1 * math.sqrt(max(f2 * (1 - f2), 1 / n2) / n2) / (1 - f2) ** 2
D2 = len(np.unique(h2[o2])); lam2 = n2 / W1
out["halfKnown"] = len(K1arr); out["halfDrawsOut"] = no2; out["halfNewClasses"] = D2
out["halfOutside"] = f"{m1:.1f}"; out["halfOutsideSE"] = f"{m1se:.1f}"; out["halfLam"] = f"{lam2:.1f}"
v = (m1 + 3 * m1se) * math.exp(-lam2); e = math.floor(math.log10(v))
out["halfExpMissing"] = f"{v:.2f}" if v >= 0.01 else f"{math.ceil(v / 10 ** e)}\\cdot10^{{{e}}}"
out["halfTotal"] = int(round(len(K1arr) + m1)); out["halfTotalSE"] = f"{m1se:.0f}"
grid = np.linspace(0, nf, 401).astype(np.int64)
np.savez("figures/uniform_k.npz", grid=grid, found=np.searchsorted(first, grid), m=u["m"], mse=u["mse"], D=len(allfair),
         nf=nf, WK=WK, lamW=lamW)
# numbers that depend only on published data or on the exhaustive classification stay fixed; all others depend on
# how long the samplers run and are wrapped in \prov (defined in main.tex: red while the runs go on)
FIXED = {"nPubEarly", "nBgpNew", "nPub", "nMfourPub", "nSym", "symW", "nLangFunctions", "nLangClasses", "nLangNew"}
print(f"% generated by tools/make_numbers.py on {datetime.date.today()}; do not edit")
for k, v in out.items():
    v = fmt(v) if isinstance(v, int) else v
    if k not in FIXED and not k.startswith("prof"): v = f"\\prov{{{v}}}"
    print(f"\\newcommand{{\\{k}}}{{{v}}}")
