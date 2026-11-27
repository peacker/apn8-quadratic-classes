#!/bin/bash
# varext_worker.sh ID  -- loop: generate quadratic APN (8,8)-functions with varext, then test and record them
# (results/varext/w<ID>.tsv: one line per function, lookup table kept for the first function of each invariant).
# Learning data: gen/d<ID>.txt (label, start line, 21-byte bilinear form of each positive G7 and 1% of negatives).
# Since 2026-11-13 each varext call does STARTS starts (40000 at first, i.e. 100 of the former 400-start batches;
# 200000 since 2026-11-14, when the search became about 11 times faster),
# so the 157 MB start file is read once per call instead of once per 400 starts; starts with a vanishing
# component are skipped (-skipr0). Calls are numbered k = 0, 1, ... in gen/w<ID>.k (sources varext_w<ID>_k<k>),
# with seeds 4000000000 + ID * 1000000 + k: disjoint from the former seeds ID * 1000000 + b (all < 1.1e9).
# The counter is persisted before each call so that restarts never repeat work.
# Log: since 2026-11-15 the per-G7 lines of each call ("start ...", about 3 GB per hour per server) are no longer
# kept. After each call one line goes to gen/w<ID>.sum: time, k, G7 tested, last log line. A call that fails keeps
# its full log as gen/w<ID>_k<k>.err.
# Stop all workers: touch ~/quad8/STOP_VAREXT
# Polite: before each call the worker waits while the 1-minute load is above LOAD_LIMIT (default 200), so other
# users of the server get priority; start the workers with nice -n 19.
ID=$1
STARTS=${STARTS:-200000}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1  # one thread per process
cd "$(dirname "$0")/.."
mkdir -p gen results/varext
k=$(cat gen/w${ID}.k 2>/dev/null || echo 0)
while [ ! -f STOP_VAREXT ]; do
  while [ "$(cut -d' ' -f1 /proc/loadavg | cut -d. -f1)" -gt "${LOAD_LIMIT:-200}" ] && [ ! -f STOP_VAREXT ]; do sleep 60; done
  [ -f STOP_VAREXT ] && break
  echo $((k + 1)) > gen/w${ID}.k
  f=gen/w${ID}_k${k}.txt; L=gen/w${ID}_k${k}.log
  quad/varext -in data/QAPN-6-8 -seed $((4000000000 + ID * 1000000 + k)) -starts $STARTS -tries 256 -cap 64 -skipr0 \
    -dump gen/d${ID}.txt > $f 2> $L || mv $L gen/w${ID}_k${k}.err
  [ -f $L ] && echo "$(date +%s) k=$k G7=$(grep -c '^start' $L) $(tail -1 $L)" >> gen/w${ID}.sum && rm -f $L
  python3 quad/testall.py --n 8 --source varext_w${ID}_k${k} --input $f --out results/varext/w${ID}.tsv --keep-lut
  rm -f $f
  k=$((k + 1))
done
