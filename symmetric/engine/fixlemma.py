"""fixlemma.py TYPES.tsv: dim Fix(A) = f, dim Fix(B) = m for each type.
Lemma: a quadratic APN F with beta(Ax,Ay) = B beta(x,y) needs m >= f when f >= 3
(restriction to Fix(A) gives beta: Fix(A) x Fix(A) -> Fix(B), each beta(a,.) of rank f-1; if m = f-1 every nonzero
component of this restriction is bent, a vectorial bent map F_2^f -> F_2^(f-1), impossible by Nyberg since f-1 > f/2)."""
import sys
def rank(rows):
    r = 0; rows = list(rows)
    for bit in reversed(range(16)):
        piv = next((i for i in range(r, len(rows)) if rows[i] >> bit & 1), None)
        if piv is None: continue
        rows[r], rows[piv] = rows[piv], rows[r]
        for i in range(len(rows)):
            if i != r and rows[i] >> bit & 1: rows[i] ^= rows[r]
        r += 1
    return r
def fixdim(cols):
    n = len(cols); return n - rank([c ^ (1 << j) for j, c in enumerate(cols)])
for l in open(sys.argv[1]):
    f = l.rstrip("\n").split("\t")
    if not f[0].isdigit(): continue
    A = list(map(int, f[5].split(","))); B = list(map(int, f[6].split(",")))
    fa, fb = fixdim(A), fixdim(B)
    tag = "old-excluded" if fb < fa - 1 else ("NEW-excluded" if fb == fa - 1 and fa >= 3 else "")
    print(f[0], f[1], fa, fb, tag, sep="\t")
