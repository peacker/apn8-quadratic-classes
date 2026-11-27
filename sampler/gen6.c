/*
 * gen6: coordinate approach for quadratic APN (8,8)-functions with relaxed filters.
 *
 * Quadratic forms on F_2^8 are 28-bit masks over the monomials x_i x_j (i < j). A component is bent iff its
 * alternating matrix has rank 8. Following Algorithm 3.1 of arXiv:2508.04644, but sampling instead of enumerating,
 * and with configurable bent counts:
 *   1. a random (8,4)-bent function C4 (4 forms; file from bent_to_forms.py);
 *   2. random fifth forms v5 until the coset v5 + span(C4) has B5 - 15 bent forms (default B5 = 30);
 *   3. random sixth forms v6 until v6 + span(C5) has B6 - B5 bent forms (default B6 = 52; arXiv:2508.04644
 *      used B6 >= 54);
 *   4. all APN completions by two more forms g7, g8, decided exactly as in switch2.c: for every 2-subspace s on which
 *      the six forms vanish, (beta_g7(s), beta_g8(s)) != (0, 0); g7 runs over the 2^22 classes modulo span(C6), and
 *      g8 is the solution of a linear system.
 * Output: one (8,8) lookup table per line (stdout); one log line per C6 (stderr): B5 B6 |Z| outputs.
 *
 * usage: gen6 -bent FORMS -seed S -samples K [-b5 30] [-b6 52] [-cap 64] [-zmax 0]
 *        gen6 -test LUTFILE   (take the six first coordinates of a known APN function; it must be found again)
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static uint64_t rs = 0x9E3779B97F4A7C15ULL;
static inline uint64_t rnd(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return rs; }
static int pidx[8][8];
static int SA[11000], SB[11000], ns;
static uint32_t SM[11000];

static inline uint32_t mono_mask(int a, int b) {
    uint32_t m = 0;
    for (int i = 0; i < 8; i++) for (int j = i + 1; j < 8; j++)
        if ((((a >> i) & (b >> j)) ^ ((a >> j) & (b >> i))) & 1) m |= 1u << pidx[i][j];
    return m;
}
static int rank_form(uint32_t q) {
    uint8_t rows[8] = {0};
    for (int i = 0; i < 8; i++) for (int j = i + 1; j < 8; j++) if ((q >> pidx[i][j]) & 1) { rows[i] |= 1 << j; rows[j] |= 1 << i; }
    int r = 0;
    for (int bit = 7; bit >= 0; bit--) {
        int p = -1;
        for (int i = r; i < 8; i++) if ((rows[i] >> bit) & 1) { p = i; break; }
        if (p < 0) continue;
        uint8_t t = rows[p]; rows[p] = rows[r]; rows[r] = t;
        for (int i = 0; i < 8; i++) if (i != r && ((rows[i] >> bit) & 1)) rows[i] ^= rows[r];
        r++;
    }
    return r;
}
static inline int bent(uint32_t q) { return rank_form(q) == 8; }
static inline int par(uint32_t x) { return __builtin_parity(x); }
static uint32_t reduce(uint32_t v, const uint32_t *b, int nb) { for (int i = 0; i < nb; i++) if ((v ^ b[i]) < v) v ^= b[i]; return v; }
static int insert(uint32_t v, uint32_t *b, int nb) {
    v = reduce(v, b, nb);
    if (!v) return nb;
    b[nb++] = v;
    for (int i = nb - 1; i > 0 && b[i] > b[i - 1]; i--) { uint32_t t = b[i]; b[i] = b[i - 1]; b[i - 1] = t; }
    return nb;
}
static int eval_form(uint32_t q, int x) {
    int v = 0;
    for (int i = 0; i < 8; i++) if ((x >> i) & 1) for (int j = i + 1; j < 8; j++) if ((x >> j) & 1) v ^= (q >> pidx[i][j]) & 1;
    return v;
}

/* all APN completions of the six forms f[0..5]; prints LUTs; returns number printed; *zsize = |Z| */
static long extend6(const uint32_t *f, long cap, int zmax, int *zsize) {
    uint32_t cb[8]; int ncb = 0;
    for (int i = 0; i < 6; i++) ncb = insert(f[i], cb, ncb);
    static uint32_t ZM[11000];
    int nz = 0;
    for (int s = 0; s < ns; s++) {
        int ok = 1;
        for (int i = 0; i < 6 && ok; i++) if (par(SM[s] & f[i])) ok = 0;
        if (ok) ZM[nz++] = SM[s];
    }
    *zsize = nz;
    if (ncb < 6 || (zmax && nz > zmax)) return 0;
    uint32_t pivmask = 0;
    for (int i = 0; i < ncb; i++) pivmask |= 1u << (31 - __builtin_clz(cb[i]));
    int freebits[28], nfree = 0;
    for (int bit = 0; bit < 28; bit++) if (!((pivmask >> bit) & 1)) freebits[nfree++] = bit;
    static uint8_t lowbits[256];
    for (int x = 0; x < 256; x++) { int v = 0; for (int i = 0; i < 6; i++) v |= eval_form(f[i], x) << i; lowbits[x] = v; }
    long outs = 0;
    for (uint32_t gi = 1; gi < (1u << nfree) && outs < cap; gi++) {
        uint32_t g7 = 0;
        for (int q = 0; q < nfree; q++) if ((gi >> q) & 1) g7 |= 1u << freebits[q];
        uint32_t rows[64]; int rhs[64], nr = 0, bad = 0;
        for (int s = 0; s < nz && !bad; s++) {
            if (par(ZM[s] & g7)) continue;
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
        uint32_t g8 = 0;
        for (int i = nr - 1; i >= 0; i--) {
            int pb = 31 - __builtin_clz(rows[i]);
            if (rhs[i] ^ par(rows[i] & g8 & ~(1u << pb))) g8 |= 1u << pb;
        }
        uint32_t mb[9]; int nm = 0;
        for (int i = 0; i < ncb; i++) mb[nm++] = cb[i];
        nm = insert(g7, mb, nm);
        uint32_t rp = 0;
        for (int i = 0; i < nr; i++) rp |= 1u << (31 - __builtin_clz(rows[i]));
        uint32_t kq[28]; int nkq = 0;
        for (int col = 0; col < 28; col++) {
            if ((rp >> col) & 1) continue;
            uint32_t v = 1u << col;
            for (int i = nr - 1; i >= 0; i--) {
                int pb = 31 - __builtin_clz(rows[i]);
                if (par(rows[i] & v & ~(1u << pb))) v |= 1u << pb;
            }
            int before = nm;
            nm = insert(v, mb, nm);
            if (nm > before) kq[nkq++] = v;
        }
        for (uint32_t e = 0; e < (1u << nkq) && outs < cap; e++) {
            uint32_t h = g8;
            for (int q = 0; q < nkq; q++) if ((e >> q) & 1) h ^= kq[q];
            for (int x = 0; x < 256; x++) {
                int v = lowbits[x] | eval_form(g7, x) << 6 | eval_form(h, x) << 7;
                printf("%d%c", v, x == 255 ? '\n' : ',');
            }
            outs++;
        }
    }
    return outs;
}

/* number of bent forms in the coset v + span(basis[0..nb)) */
static int coset_bent(uint32_t v, const uint32_t *basis, int nb) {
    int c = 0;
    for (int t = 0; t < (1 << nb); t++) {
        uint32_t q = v;
        for (int i = 0; i < nb; i++) if ((t >> i) & 1) q ^= basis[i];
        c += bent(q);
    }
    return c;
}

int main(int argc, char **argv) {
    const char *bentf = NULL, *testf = NULL;
    long samples = 1, cap = 64;
    int b5 = 30, b6 = 52, zmax = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-bent")) bentf = argv[++i];
        else if (!strcmp(argv[i], "-test")) testf = argv[++i];
        else if (!strcmp(argv[i], "-seed")) { rs = strtoull(argv[++i], 0, 10) * 0x9E3779B97F4A7C15ULL + 13; for (int k = 0; k < 9; k++) rnd(); }
        else if (!strcmp(argv[i], "-samples")) samples = atol(argv[++i]);
        else if (!strcmp(argv[i], "-b5")) b5 = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-b6")) b6 = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-cap")) cap = atol(argv[++i]);
        else if (!strcmp(argv[i], "-zmax")) zmax = atoi(argv[++i]);
    }
    int k = 0;
    for (int i = 0; i < 8; i++) for (int j = i + 1; j < 8; j++) pidx[i][j] = k++;
    for (int a = 1; a < 256; a++) for (int b = a + 1; b < 256; b++) if (b < (a ^ b)) { SA[ns] = a; SB[ns] = b; SM[ns] = mono_mask(a, b); ns++; }

    if (testf) {  /* the six first coordinates of a known APN function must lead back to it */
        FILE *t = fopen(testf, "r");
        int F[256];
        for (int x = 0; x < 256; x++) if (fscanf(t, " %d ,", &F[x]) != 1) { fprintf(stderr, "bad test LUT\n"); return 1; }
        uint32_t f[6] = {0};
        for (int c = 0; c < 6; c++) for (int i = 0; i < 8; i++) for (int j = i + 1; j < 8; j++) {
            int a = 1 << i, b = 1 << j;
            if (((F[a ^ b] ^ F[a] ^ F[b] ^ F[0]) >> c) & 1) f[c] |= 1u << pidx[i][j];
        }
        int z; long o = extend6(f, 100000, 0, &z);
        fprintf(stderr, "test: |Z| = %d, outputs %ld\n", z, o);
        return 0;
    }
    if (!bentf) { fprintf(stderr, "usage: gen6 -bent FORMS -seed S -samples K [-b5 30] [-b6 52] [-cap 64] [-zmax 0]\n"); return 1; }
    FILE *bf = fopen(bentf, "r");
    long nbent = 0, capb = 1 << 17;
    uint32_t (*B4)[4] = malloc(sizeof(uint32_t) * 4 * capb);
    while (nbent < capb && fscanf(bf, "%u %u %u %u", &B4[nbent][0], &B4[nbent][1], &B4[nbent][2], &B4[nbent][3]) == 4) nbent++;
    fclose(bf);
    fprintf(stderr, "gen6: %ld (8,4)-bent functions, targets B5 = %d, B6 = %d\n", nbent, b5, b6);
    for (long smp = 0; smp < samples; smp++) {
        clock_t t0 = clock();
        long li = rnd() % nbent;
        uint32_t f[6];
        for (int i = 0; i < 4; i++) f[i] = B4[li][i];
        if (coset_bent(0, f, 4) != 15) { fprintf(stderr, "line %ld: not (8,4)-bent\n", li); continue; }
        long tr5 = 0, tr6 = 0;
        int ok5 = 0, ok6 = 0;
        for (; tr5 < 2000000 && !ok5; tr5++) { f[4] = rnd() & 0xFFFFFFF; if (coset_bent(f[4], f, 4) == b5 - 15) ok5 = 1; }
        if (!ok5) { fprintf(stderr, "line %ld: no B5 = %d after %ld trials\n", li, b5, tr5); continue; }
        for (; tr6 < 2000000 && !ok6; tr6++) { f[5] = rnd() & 0xFFFFFFF; if (coset_bent(f[5], f, 5) == b6 - b5) ok6 = 1; }
        if (!ok6) { fprintf(stderr, "line %ld: no B6 = %d after %ld trials\n", li, b6, tr6); continue; }
        int z;
        long o = extend6(f, cap, zmax, &z);
        fprintf(stderr, "C6 from bent line %ld: B5 %d (%ld trials) B6 %d (%ld trials) |Z| %d outputs %ld time %.1fs\n",
                li, b5, tr5, b6, tr6, z, o, (double)(clock() - t0) / CLOCKS_PER_SEC);
        fflush(stdout);
    }
    return 0;
}
