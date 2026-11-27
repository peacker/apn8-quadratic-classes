/*
 * profile: bent-component profile of quadratic (8,8)-functions, to compare with the filters of the coordinate
 * approach of Beierle-Langevin-Leander-Polujan-Rasoolzadeh (arXiv:2508.04644, Algorithm 3.1).
 *
 * For each function F (one lookup table per line): the component c.F is bent iff its quadratic form has rank 8.
 * Over the subspaces of the component space (indexed by c in F_2^8):
 *   nbent        number of bent components (of 255)
 *   nbent4       number of 4-dim subspaces whose 15 nonzero components are all bent ((8,4)-bent subfunctions);
 *                Hypothesis 3.2 says this is >= 1 for every quadratic APN (8,8)-function
 *   maxB5        max number of bent components (of 31) of a 5-dim subspace containing a bent 4-dim subspace
 *   maxB6        max number of bent components (of 63) of a 6-dim subspace
 *   reach        1 iff a chain C4 (all bent) < C5 (B5 = 30) < C6 (B6 >= 54) exists, i.e. F can be produced by the
 *                filtered Algorithm 3.1 (before J2-deduplication and its other heuristics)
 *   best6        max B6 over 6-dim subspaces C6 that contain some C5 with B5 = 30 over a bent C4 (-1 if none)
 * Output: TSV line per function: index nbent nbent4 maxB5 maxB6 best6 reach
 * usage: profile < luts.txt
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int rank8(uint8_t *rows) {  /* rank of an 8x8 binary matrix (rows destroyed) */
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

/* all subspaces of F_2^8 of dimension d as 256-bit membership masks (via reduced echelon bases) */
typedef struct { uint64_t w[4]; } set256;
static set256 *subs[9];
static int nsub[9];
static int piv[8], row[8], cur_d;
static set256 *dst; static int cnt;
static void emit_sub(void) {
    set256 s = {{0, 0, 0, 0}};
    for (int t = 0; t < (1 << cur_d); t++) {
        int v = 0;
        for (int i = 0; i < cur_d; i++) if ((t >> i) & 1) v ^= row[i];
        s.w[v >> 6] |= 1ULL << (v & 63);
    }
    dst[cnt++] = s;
}
static void rows_rec(int r) {
    if (r == cur_d) { emit_sub(); return; }
    int freemask = 0;
    for (int bit = 0; bit < piv[r]; bit++) {
        int isp = 0;
        for (int s = 0; s < cur_d; s++) if (piv[s] == bit) isp = 1;
        if (!isp) freemask |= 1 << bit;
    }
    int nf = __builtin_popcount(freemask);
    for (int x = 0; x < (1 << nf); x++) {
        int v = 1 << piv[r], k = 0;
        for (int bit = 0; bit < piv[r]; bit++) if ((freemask >> bit) & 1) { if ((x >> k) & 1) v |= 1 << bit; k++; }
        row[r] = v;
        rows_rec(r + 1);
    }
}
static void piv_rec(int r, int maxbit) {
    if (r == cur_d) { rows_rec(0); return; }
    for (int p = maxbit; p >= cur_d - r - 1; p--) { piv[r] = p; piv_rec(r + 1, p - 1); }
}
static void build(int d, int expected) {
    subs[d] = malloc(sizeof(set256) * expected);
    dst = subs[d]; cnt = 0; cur_d = d;
    piv_rec(0, 7);
    nsub[d] = cnt;
}
static inline int sub_of(const set256 *a, const set256 *b) {  /* a subset of b */
    for (int i = 0; i < 4; i++) if (a->w[i] & ~b->w[i]) return 0;
    return 1;
}

int main(void) {
    build(4, 200787); build(5, 97155); build(6, 10795);
    fprintf(stderr, "subspaces: dim4 %d, dim5 %d, dim6 %d\n", nsub[4], nsub[5], nsub[6]);
    char line[4096];
    int F[256];
    long idx = 0;
    while (fgets(line, sizeof line, stdin)) {
        char *p = line; int nv = 0;
        while (*p && nv < 256) { while (*p && (*p < '0' || *p > '9')) p++; if (!*p) break; F[nv++] = (int)strtol(p, &p, 10); }
        if (nv != 256) { idx++; continue; }
        /* quadratic part of each coordinate: matrix rows of the alternating form */
        uint8_t Q[8][8];  /* Q[k][i] = row i of the form of coordinate k */
        for (int k = 0; k < 8; k++) for (int i = 0; i < 8; i++) {
            uint8_t r = 0;
            for (int j = 0; j < 8; j++) if (i != j) {
                int a = 1 << i, b = 1 << j;
                int v = (F[a ^ b] ^ F[a] ^ F[b] ^ F[0]) >> k & 1;
                r |= v << j;
            }
            Q[k][i] = r;
        }
        set256 bent = {{0, 0, 0, 0}};
        int nbent = 0;
        for (int c = 1; c < 256; c++) {
            uint8_t rows[8] = {0};
            for (int k = 0; k < 8; k++) if ((c >> k) & 1) for (int i = 0; i < 8; i++) rows[i] ^= Q[k][i];
            if (rank8(rows) == 8) { bent.w[c >> 6] |= 1ULL << (c & 63); nbent++; }
        }
        #define NB(s) (__builtin_popcountll((s)->w[0] & bent.w[0]) + __builtin_popcountll((s)->w[1] & bent.w[1]) + \
                       __builtin_popcountll((s)->w[2] & bent.w[2]) + __builtin_popcountll((s)->w[3] & bent.w[3]))
        /* bent 4-subspaces */
        static int b4[200787]; int nb4 = 0;
        for (int i = 0; i < nsub[4]; i++) if (NB(&subs[4][i]) == 15) b4[nb4++] = i;
        /* 5-subspaces containing a bent 4-subspace; B5 = 30 ones */
        int maxB5 = -1;
        static int good5[97155]; int ng5 = 0;
        for (int i = 0; i < nsub[5]; i++) {
            int B5 = NB(&subs[5][i]);
            if (B5 <= maxB5 && B5 != 30) continue;
            int has = 0;
            for (int j = 0; j < nb4 && !has; j++) if (sub_of(&subs[4][b4[j]], &subs[5][i])) has = 1;
            if (!has) continue;
            if (B5 > maxB5) maxB5 = B5;
            if (B5 == 30) good5[ng5++] = i;
        }
        int maxB6 = 0, best6 = -1, reach = 0;
        for (int i = 0; i < nsub[6]; i++) {
            int B6 = NB(&subs[6][i]);
            if (B6 > maxB6) maxB6 = B6;
            if (B6 <= best6) continue;
            for (int j = 0; j < ng5; j++) if (sub_of(&subs[5][good5[j]], &subs[6][i])) { best6 = B6; break; }
        }
        if (best6 >= 54) reach = 1;
        printf("%ld\t%d\t%d\t%d\t%d\t%d\t%d\n", idx, nbent, nb4, maxB5, maxB6, best6, reach);
        fflush(stdout);
        idx++;
    }
    return 0;
}
