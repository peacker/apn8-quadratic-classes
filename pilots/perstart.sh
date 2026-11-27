#!/bin/bash
# perstart.sh NSTARTS STARTS_PER_RUN NPAR SEED -- per-start yield: for NSTARTS random lines of QAPN-6-8, run varext
# on that single start (STARTS_PER_RUN draws), test, and record classes and new classes per start.
# Output runs/perstart/s<line>.tsv (testall) and summary via perstart_eval.py.
NS=$1; SP=$2; NP=$3; SEED=$4
D=runs/perstart; mkdir -p $D
Q=../../learning/quad
python3 -c "
import random; random.seed($SEED)
print('\n'.join(str(x) for x in random.sample(range(866470), $NS)))" > $D/lines.txt
run() { ln=$1; f=$D/s$ln; sed -n "$((ln+1))p" data/QAPN-6-8 > $f.start
  /usr/bin/time -p ./varext -in $f.start -seed $((ln+5)) -starts $SP -tries 256 -cap 64 -skipr0 > $f.out 2> $f.err
  python3 $Q/testall.py --n 8 --source s$ln --input $f.out --out $f.tsv --keep-lut > /dev/null 2>&1
  grep '^user' $f.err | awk '{print $2}' > $f.cpu; echo "G7 $(grep -c '^start' $f.err)" >> $f.cpu; rm -f $f.out $f.err; }
export -f run; export D Q SP
cat $D/lines.txt | xargs -P $NP -I{} bash -c 'run {}'
