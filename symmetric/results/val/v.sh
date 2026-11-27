#!/bin/bash
# v.sh TYPE BIN [extra]: full run, keys and time
cd "$(dirname "$0")/.."
T=$1; BIN=$2; shift 2
line=$(awk -F'\t' -v i=$T '$1==i' ../types8.tsv); A=$(echo "$line" | cut -f6); B=$(echo "$line" | cut -f7)
X=""; [ "$A" != "1,2,4,8,16,32,64,128" ] && X="-scale $A $B"
s=$(date +%s)
./$BIN -tr $X "$@" -dfs -dedupe -symdepth 28 -cp 3 8 -mtf -late 7 8 -nsin 1024 -cpcache < W/w_$T.txt > val/${T}_$BIN.out 2> val/${T}_$BIN.err
echo -e "$T\t$BIN\t$(( $(date +%s) - s ))\t$(grep -c '^K' val/${T}_$BIN.out)\t$(grep '^done' val/${T}_$BIN.err)" >> val/res.tsv
