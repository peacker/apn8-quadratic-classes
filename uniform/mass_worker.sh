#!/bin/bash
# mass_worker.sh ID  -- fair sampler (n8search/mass): every APN bilinear map beta8 is drawn with the same
# probability, so every class with weight 1/|G|. Same output as varext_worker.sh (results/varext/w<ID>.tsv,
# testall rows, source mass_w<ID>_k<k>), so sync, build_db and push are unchanged.
# Per call one line in gen/m<ID>.sum: time, k, and the counters of the call
# (starts, chain attempts, L6 tries, valid L6, outputs, nonstandard), needed for the mass estimate.
# Seeds 6000000000 + ID * 1000000 + k (disjoint from varext seeds 4e9 + ...). Counter gen/m<ID>.k.
# Stop all fair workers: touch ~/quad8/STOP_MASS. Start with nice -n 19. Waits while the load is above LOAD_LIMIT.
ID=$1
STARTS=${STARTS:-30000}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$(dirname "$0")/.."
mkdir -p gen results/varext
k=$(cat gen/m${ID}.k 2>/dev/null || echo 0)
while [ ! -f STOP_MASS ]; do
  while [ "$(cut -d' ' -f1 /proc/loadavg | cut -d. -f1)" -gt "${LOAD_LIMIT:-200}" ] && [ ! -f STOP_MASS ]; do sleep 60; done
  [ -f STOP_MASS ] && break
  echo $((k + 1)) > gen/m${ID}.k
  f=gen/m${ID}_k${k}.txt; L=gen/m${ID}_k${k}.log
  if quad/mass8 -mass $STARTS -seed $((6000000000 + ID * 1000000 + k)) -tries 4096 -cap 64 > $f 2> $L; then
    echo "$(date +%s) k=$k $(tail -1 $L)" >> gen/m${ID}.sum; rm -f $L
  else
    mv $L gen/m${ID}_k${k}.err
  fi
  python3 quad/testall.py --n 8 --source mass_w${ID}_k${k} --input $f --out results/varext/w${ID}.tsv --keep-lut
  rm -f $f
  k=$((k + 1))
done
