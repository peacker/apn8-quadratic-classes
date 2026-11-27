#!/bin/bash
# mass_gpu_worker.sh ID DEV -- the uniform sampler on GPU DEV (quad/mass8g, n8search/cuda/mass8g.cu). Same output as
# mass_worker.sh: rows in results/varext/w<ID>.tsv with source mass_w<ID>_k<k>, and per call one line in
# gen/m<ID>.sum (time, k, starts, chain attempts, L6 tries, valid L6, outputs, nonstd, "splitmix2g").
# testall.py of call k runs in the background while the GPU makes call k + 1; its rows are appended to w<ID>.tsv
# first and the counter line after them, so a synced counter line always has its rows. Lookup tables are kept
# only for classes not in data/known_keys.txt. Seeds 9000000000 + ID * 1000000 + k. Stop: touch ~/quad8/STOP_MASS.
# Up to POSTS (default 2) testall.py run at once: on a fast GPU (A100) one call takes less time than its testall.py
# on one core. The appends hold a lock, so the rows of two calls never interleave; counter lines may be out of order in k.
ID=$1; DEV=$2
STARTS=${STARTS:-6000000}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$(dirname "$0")/.."
mkdir -p gen results/varext
k=$(cat gen/m${ID}.k 2>/dev/null || echo 0)
post() {   # call k: classify, append rows, then the counter line
  local k=$1 sum=$2 f=gen/m${ID}_k$1.txt t=gen/t${ID}_k$1.tsv
  rm -f $t
  nice -n 10 python3 quad/testall.py --n 8 --source mass_w${ID}_k${k} --input $f --out $t --keep-lut --known data/known_keys.txt \
    && { flock 9; cat $t >> results/varext/w${ID}.tsv && { [ -z "$sum" ] || echo "$sum" >> gen/m${ID}.sum; }; } 9> gen/m${ID}.lock \
    && rm -f $f $t
}
POSTS=${POSTS:-2}; pids=()
while [ ! -f STOP_MASS ]; do
  echo $((k + 1)) > gen/m${ID}.k
  f=gen/m${ID}_k${k}.txt; L=gen/m${ID}_k${k}.log
  if quad/mass8g -mass $STARTS -seed $((9000000000 + ID * 1000000 + k)) -tries 4096 -cap 64 -dev $DEV > $f 2> $L; then
    sum="$(date +%s) k=$k $(tail -1 $L)"; rm -f $L
  else
    sum=""; mv $L gen/m${ID}_k${k}.err
  fi
  while [ ${#pids[@]} -ge $POSTS ]; do wait ${pids[0]}; pids=("${pids[@]:1}"); done
  post $k "$sum" & pids+=($!)
  k=$((k + 1))
done
wait
