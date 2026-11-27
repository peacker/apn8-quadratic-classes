#!/bin/bash
# prun.sh LABEL STARTFILE NWORK SECONDS [STARTS]  -- run NWORK varext workers for SECONDS wall, then testall.
# Output in runs/LABEL/: w<i>_k<k>.tsv (testall), cpu.txt (sum of worker CPU seconds from varext + testall).
L=$1; S=$2; NW=$3; T=$4; ST=${5:-200000}
D=runs/$L; mkdir -p $D; end=$(( $(date +%s) + T ))
Q=../../learning/quad
for i in $(seq 1 $NW); do
  ( k=0
    while [ $(date +%s) -lt $end ]; do
      s=$(( 7000000 + i * 10000 + k ))
      /usr/bin/time -p ./varext -in $S -seed $s -starts $ST -tries 256 -cap 64 -skipr0 > $D/o${i}_$k.txt 2> $D/e${i}_$k.log
      python3 $Q/testall.py --n 8 --source ${L}_${i}_$k --input $D/o${i}_$k.txt --out $D/w${i}.tsv --keep-lut > /dev/null 2>&1
      grep '^user\|^sys' $D/e${i}_$k.log | awk '{s+=$2} END{print s}' >> $D/cpu_varext.txt
      rm -f $D/o${i}_$k.txt; tail -1 $D/e${i}_$k.log > /dev/null; rm -f $D/e${i}_$k.log
      k=$((k+1))
    done ) &
done
wait
awk '{s+=$1} END{print s}' $D/cpu_varext.txt > $D/cpu.txt
echo "$L done: varext cpu $(cat $D/cpu.txt) s"
