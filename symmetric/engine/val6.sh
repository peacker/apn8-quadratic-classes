#!/bin/bash
# val6.sh ID: n = 6 type ID, reference (no -tr/-scale/-ahead) vs sym8k -tr -scale -ahead; prints keys equality
cd "$(dirname "$0")/.."
line=$(awk -F'\t' -v i=$1 '$1==i' types6.tsv); A=$(echo "$line" | cut -f6); B=$(echo "$line" | cut -f7); p=$(echo "$line" | cut -f2)
W=s8/W6/w_$1.txt; [ -f $W ] || python3 sym_setup.py 6 --amat "$A" --bmat "$B" --mod 0x43 -o $W > /dev/null
C="-dfs -dedupe -symdepth 15 -cp 3 6 -mtf -nsin 256 -cpcache"
X=""; [ "$A" != "1,2,4,8,16,32" ] && X="-scale $A $B"
timeout 600 s8/sym8j $C < $W > s8/W6/r_$1.out 2>/dev/null; r1=$?
timeout 600 s8/sym8k -tr $X -ahead $C < $W > s8/W6/k_$1.out 2> s8/W6/k_$1.err; r2=$?
a=$(grep '^K' s8/W6/r_$1.out | cut -d' ' -f1 | sort -u | tr '\n' ' '); b=$(grep '^K' s8/W6/k_$1.out | cut -d' ' -f1 | sort -u | tr '\n' ' ')
sc=$(grep -o "scaling: pruned [0-9]*" s8/W6/k_$1.err | awk '{print $3}')
echo -e "$1\tp=$p\trc=$r1/$r2\tkeys=$(echo $a | wc -w | tr -d ' ')/$(echo $b | wc -w | tr -d ' ')\tscale_pruned=${sc:-0}\t$([ "$a" = "$b" ] && echo SAME || echo DIFF)"
