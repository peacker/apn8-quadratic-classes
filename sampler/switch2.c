/*
 * switch2: two-coordinate switching of quadratic APN (8,8)-functions.
 *
 * A quadratic F is APN iff its bilinear form B_F(a, b) = F(a+b) + F(a) + F(b) + F(0) is nonzero for every
 * 2-dimensional subspace {0, a, b, a+b}. Keep a 6-dimensional subspace C6 of the components of F (the first 6
 * rows of a random invertible output matrix A) and replace the other two coordinates by quadratic forms
 * g7, g8 (28 monomials x_i x_j). The new function is APN iff for every 2-subspace s on which all kept components
 * vanish (s in Z), (beta_g7(s), beta_g8(s)) != (0, 0), where beta_g is the bilinear form of g.
 * For each g7 (2^28 values, reduced modulo the 6 kept forms: 2^22 classes), the condition on g8 is linear:
 * beta_g8(s) = 1 for every s in Z with beta_g7(s) = 0. Solutions g8 form an affine space; modulo the forms of
 * C6 + g7 (which vanish on these s) we output one g8 per coset, up to -cap outputs per (F, C6).
 * F itself (its own two coordinates) is always among the solutions: a built-in check.
 *
 * usage: switch2 -in LUTFILE -seed S -funcs K [-subspaces C] [-cap X]
 *   picks K random lines of LUTFILE (one 256-value table per line) and C random C6 per function.
 * Output: one (8,8) lookup table per line.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint64_t rs = 0x9E3779B97F4A7C15ULL;
static inline uint64_t rnd(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return rs; }

static int pidx[8][8];   /* monomial x_i x_j (i < j) -> bit 0..27 */

/* bilinear form of the quadratic form q (28-bit) on (a, b): sum q_ij (a_i b_j + a_j b_i) */
static inline uint32_t mono_mask(int a, int b) {  /* bit ij set iff a_i b_j + a_j b_i = 1 */
    uint32_t m = 0;
    for (int i = 0; i < 8; i++) for (int j = i + 1; j < 8; j++)
        if ((((a >> i) & (b >> j)) ^ ((a >> j) & (b >> i))) & 1) m |= 1u << pidx[i][j];
    return m;
}

/* quadratic part (28-bit) of a Boolean function given by 256 values: q_ij = f(e_i+e_j)+f(e_i)+f(e_j)+f(0) */
static uint32_t qpart(const uint8_t *f) {
    uint32_t q = 0;
    for (int i = 0; i < 8; i++) for (int j = i + 1; j < 8; j++)
        if (f[(1 << i) | (1 << j)] ^ f[1 << i] ^ f[1 << j] ^ f[0]) q |= 1u << pidx[i][j];
    return q;
}

static inline int par32(uint32_t x) { return __builtin_parity(x); }

/* reduce v modulo an echelon basis (pivot = highest bit) */
static uint32_t reduce(uint32_t v, const uint32_t *bas, int nb) {
    for (int i = 0; i < nb; i++) if ((v ^ bas[i]) < v) v ^= bas[i];
    return v;
}
static int insert(uint32_t v, uint32_t *bas, int nb) {
    v = reduce(v, bas, nb);
    if (!v) return nb;
    bas[nb++] = v;
    for (int i = nb - 1; i > 0 && bas[i] > bas[i - 1]; i--) { uint32_t t = bas[i]; bas[i] = bas[i - 1]; bas[i - 1] = t; }
    return nb;
}

int main(int argc, char **argv) {
    const char *in = NULL;
    long funcs = 1, subs = 4, cap = 1000;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-in")) in = argv[++i];
        else if (!strcmp(argv[i], "-seed")) { rs = strtoull(argv[++i], 0, 10) * 0x9E3779B97F4A7C15ULL + 11; for (int k = 0; k < 9; k++) rnd(); }
        else if (!strcmp(argv[i], "-funcs")) funcs = atol(argv[++i]);
        else if (!strcmp(argv[i], "-subspaces")) subs = atol(argv[++i]);
        else if (!strcmp(argv[i], "-cap")) cap = atol(argv[++i]);
    }
    if (!in) { fprintf(stderr, "usage: switch2 -in LUTFILE -seed S -funcs K [-subspaces C] [-cap X]\n"); return 1; }
    int k = 0;
    for (int i = 0; i < 8; i++) for (int j = i + 1; j < 8; j++) pidx[i][j] = k++;

    /* index the lines of the input (offsets), then read random ones */
    FILE *f = fopen(in, "r");
    if (!f) { perror(in); return 1; }
    long cap_off = 1 << 16, noff = 0;
    long *off = malloc(sizeof(long) * cap_off);
    off[noff++] = 0;
    int c;
    long pos = 0;
    while ((c = fgetc(f)) != EOF) { pos++; if (c == '\n') { if (noff == cap_off) { cap_off *= 2; off = realloc(off, sizeof(long) * cap_off); } off[noff++] = pos; } }
    noff--;
    fprintf(stderr, "switch2: %ld input functions\n", noff);

    /* 2-subspaces {0,a,b,a+b}, each once: a < b < a^b */
    static int SA[11000], SB[11000];
    static uint32_t SM[11000];
    int ns = 0;
    for (int a = 1; a < 256; a++) for (int b = a + 1; b < 256; b++) if (b < (a ^ b)) { SA[ns] = a; SB[ns] = b; SM[ns] = mono_mask(a, b); ns++; }

    char line[4096];
    int F[256];
    for (long t = 0; t < funcs; t++) {
        long li = rnd() % noff;
        fseek(f, off[li], SEEK_SET);
        if (!fgets(line, sizeof line, f)) continue;
        char *p = line; int nv = 0;
        while (*p && nv < 256) { while (*p && (*p < '0' || *p > '9')) p++; if (!*p) break; F[nv++] = (int)strtol(p, &p, 10); }
        if (nv != 256) continue;
        for (long sc = 0; sc < subs; sc++) {
            /* random invertible 8x8 output matrix: rows r[0..7] */
            int r[8], bas[8], nb = 0;
            while (nb < 8) {
                int v = rnd() & 255, w = v;
                for (int i = 0; i < nb; i++) if ((w ^ bas[i]) < w) w ^= bas[i];
                if (!w) continue;
                r[nb] = v;
                bas[nb++] = w;
                for (int i = nb - 1; i > 0 && bas[i] > bas[i - 1]; i--) { int tt = bas[i]; bas[i] = bas[i - 1]; bas[i - 1] = tt; }
            }
            /* kept components c_0..c_5 = r[0..5] . F ; their quadratic parts span the forms we work modulo */
            uint8_t comp[8][256];
            for (int i = 0; i < 8; i++) for (int x = 0; x < 256; x++) comp[i][x] = __builtin_parity(r[i] & F[x]);
            uint32_t cb[8]; int ncb = 0;
            for (int i = 0; i < 6; i++) ncb = insert(qpart(comp[i]), cb, ncb);
            /* Z: 2-subspaces where all kept components vanish, i.e. B_F(a,b) . r[i] = 0 for i < 6 */
            static uint32_t ZM[11000];
            int nz = 0;
            for (int s = 0; s < ns; s++) {
                int a = SA[s], b = SB[s];
                int B = F[a ^ b] ^ F[a] ^ F[b] ^ F[0], ok = 1;
                for (int i = 0; i < 6 && ok; i++) if (__builtin_parity(r[i] & B)) ok = 0;
                if (ok) ZM[nz++] = SM[s];
            }
            /* g7 runs over representatives modulo span(cb): the 22 non-pivot bits are free */
            uint32_t pivmask = 0;
            for (int i = 0; i < ncb; i++) pivmask |= 1u << (31 - __builtin_clz(cb[i]));
            int freebits[28], nfree = 0;
            for (int bit = 0; bit < 28; bit++) if (!((pivmask >> bit) & 1)) freebits[nfree++] = bit;
            long outs = 0, g7tried = 0;
            uint32_t own7 = reduce(qpart(comp[6]), cb, ncb);
            for (uint32_t gi = 1; gi < (1u << nfree) && outs < cap; gi++) {  /* gi = 0: g7 in span(C6), never APN */
                uint32_t g7 = 0;
                for (int q = 0; q < nfree; q++) if ((gi >> q) & 1) g7 |= 1u << freebits[q];
                g7tried++;
                /* linear system for g8: rows = ZM[s] (28 bits), rhs 1, for s with beta_g7(s) = 0 */
                uint32_t rows[64]; int rhs[64], nr = 0, bad = 0;   /* echelon form, at most 28 independent rows */
                for (int s = 0; s < nz && !bad; s++) {
                    if (par32(ZM[s] & g7)) continue;
                    uint32_t v = ZM[s]; int rh = 1;
                    for (int i = 0; i < nr; i++) if ((v ^ rows[i]) < v) { v ^= rows[i]; rh ^= rhs[i]; }
                    if (!v) { if (rh) bad = 1; continue; }
                    rows[nr] = v; rhs[nr] = rh; nr++;
                    for (int i = nr - 1; i > 0 && rows[i] > rows[i - 1]; i--) {
                        uint32_t tv = rows[i]; rows[i] = rows[i - 1]; rows[i - 1] = tv;
                        int th = rhs[i]; rhs[i] = rhs[i - 1]; rhs[i - 1] = th;
                    }
                }
                if (bad) continue;
                /* particular solution: back substitution with free variables 0 */
                uint32_t g8 = 0;
                for (int i = nr - 1; i >= 0; i--) {
                    int pb = 31 - __builtin_clz(rows[i]);
                    int val = rhs[i] ^ par32(rows[i] & g8 & ~(1u << pb));
                    if (val) g8 |= 1u << pb;
                }
                /* other solutions: g8 + kernel; kernel modulo span(cb, g7) -- enumerate the quotient */
                uint32_t mb[9]; int nm = 0;
                for (int i = 0; i < ncb; i++) mb[nm++] = cb[i];
                nm = insert(g7, mb, nm);
                /* kernel basis of the system: for each non-pivot column */
                uint32_t rp = 0;
                for (int i = 0; i < nr; i++) rp |= 1u << (31 - __builtin_clz(rows[i]));
                uint32_t kb[28]; int nk = 0, nkq = 0;
                uint32_t kq[28];
                for (int col = 0; col < 28; col++) {
                    if ((rp >> col) & 1) continue;
                    uint32_t v = 1u << col;
                    for (int i = nr - 1; i >= 0; i--) {
                        int pb = 31 - __builtin_clz(rows[i]);
                        if (par32(rows[i] & v & ~(1u << pb))) v |= 1u << pb;
                    }
                    kb[nk++] = v;
                    uint32_t red = reduce(v, mb, nm);
                    if (red) { int before = nm; nm = insert(v, mb, nm); if (nm > before) kq[nkq++] = v; }
                }
                (void)kb;
                for (uint32_t e = 0; e < (1u << nkq) && outs < cap; e++) {
                    uint32_t h = g8;
                    for (int q = 0; q < nkq; q++) if ((e >> q) & 1) h ^= kq[q];
                    for (int x = 0; x < 256; x++) {
                        int v = 0;
                        for (int i = 0; i < 6; i++) v |= comp[i][x] << i;
                        /* evaluate quadratic forms g7, h at x */
                        int b7 = 0, b8 = 0;
                        for (int i = 0; i < 8; i++) if ((x >> i) & 1) for (int j = i + 1; j < 8; j++) if ((x >> j) & 1) {
                            b7 ^= (g7 >> pidx[i][j]) & 1; b8 ^= (h >> pidx[i][j]) & 1; }
                        v |= b7 << 6 | b8 << 7;
                        printf("%d%c", v, x == 255 ? '\n' : ',');
                    }
                    outs++;
                }
            }
            fprintf(stderr, "func line %ld subspace %ld: |Z| = %d, g7 tried %ld, outputs %ld (own g7 rep %u)\n",
                    li, sc, nz, g7tried, outs, own7);
        }
    }
    fclose(f);
    return 0;
}
