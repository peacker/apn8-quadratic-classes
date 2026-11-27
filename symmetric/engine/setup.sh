#!/bin/bash
# setup.sh ID: W input (with Z(B) and Z(A)) for n = 8 type ID of types8.tsv, in s8/W/w_ID.txt
cd "$(dirname "$0")/.."
line=$(awk -F'\t' -v i=$1 '$1==i' types8.tsv); A=$(echo "$line" | cut -f6); B=$(echo "$line" | cut -f7)
python3 sym_setup.py 8 --amat "$A" --bmat "$B" --mod 0x11b -o s8/W/w_$1.txt > /dev/null
