#!/bin/bash
# grun.sh LABEL NWORK SECONDS "GEN COMMAND (without -seed/-samples)" [SAMPLES]  -- coordinate-approach pilot runner
L=$1; NW=$2; T=$3; CMD=$4; SM=${5:-100}
D=runs/$L; mkdir -p $D; end=$(( $(date +%s) + T ))
Q=../../learning/quad
for i in $(seq 1 $NW); do
  ( k=0
    while [ $(date +%s) -lt $end ]; do
      s=$(( 9000000 + i * 10000 + k ))
      /usr/bin/time -p $CMD -seed $s -samples $SM > $D/o${i}_$k.txt 2> $D/e${i}_$k.log
      grep -v '^user\|^sys\|^real' $D/e${i}_$k.log | grep '^C6' >> $D/c6.log
      [ -s $D/o${i}_$k.txt ] && python3 $Q/testall.py --n 8 --source ${L}_${i}_$k --input $D/o${i}_$k.txt --out $D/w${i}.tsv --keep-lut > /dev/null 2>&1
      grep '^user\|^sys' $D/e${i}_$k.log | awk '{s+=$2} END{print s}' >> $D/cpu_gen.txt
      rm -f $D/o${i}_$k.txt $D/e${i}_$k.log; k=$((k+1))
    done ) &
done
wait
awk '{s+=$1} END{print s}' $D/cpu_gen.txt > $D/cpu.txt; echo "$L done: gen cpu $(cat $D/cpu.txt) s"
