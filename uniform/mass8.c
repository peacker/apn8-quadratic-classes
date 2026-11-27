/*
 * varext: the "variable approach" of Beierle-Langevin-Leander-Polujan-Rasoolzadeh (arXiv:2508.04644, Sec. 4)
 * for quadratic APN (8,8)-functions, as a random sampler.
 *
 * Theorem 4.1 there: for quadratic APN F : F_2^k -> F_2^8 and linear L : F_2^k -> F_2^8,
 *   G(x, x_{k+1}) = F(x) + x_{k+1} L(x)  is APN  iff  L(a) is not in Im(D_a F) for every a != 0,
 * where D_a F(x) = F(x+a) + F(x) + F(a) + F(0) is linear in x for quadratic F.
 *
 * Per start function F (a random line of QAPN-6-8: quadratic APN (6,8)-functions, 64 hex values per line):
 *   1. up to -tries random L6 : F_2^6 -> F_2^8 satisfying the condition give (7,8)-functions G7;
 *   2. for each G7, a backtracking finds all linear L7 : F_2^7 -> F_2^8 satisfying the condition, up to the
 *      symmetry described in bt(); each solution gives an APN
 *      (8,8)-function G8. At most -cap solutions per G7 are output.
 * Output: one (8,8) lookup table per line, decimal, comma separated.
 *
 * -skipr0: drop the start functions that have a vanishing component (some c != 0 with c.F affine, i.e. the 15
 *   values B(e_i, e_j) of the bilinear form span less than F_2^8). They are 4.5 % of QAPN-6-8, and their G7 extend
 *   4-7 times less often than the others (n8search/bias/APN_VS_RANDOM.md). Line numbers in the log and the dump
 *   stay those of the input file.
 * usage: varext -in QAPN-6-8 -seed S -starts K [-tries 256] [-cap 64] [-skipr0]
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint64_t rs = 0x9E3779B97F4A7C15ULL;
#ifdef SPLITMIX
#define RNGTAG " splitmix2"
/* splitmix64: the output is a nonlinear function of the state (multiplications), unlike xorshift64, whose
 * outputs are F_2-linear in the seed state. The APN conditions are F_2-linear algebra, so a linear generator
 * can bias them. */
static inline uint64_t rnd(void) { uint64_t z = (rs += 0x9E3779B97F4A7C15ULL); z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL; return z ^ (z >> 31); }
#else
#define RNGTAG ""
static inline uint64_t rnd(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return rs; }
#endif

typedef struct { uint64_t w[4]; } set256;
static inline void s_add(set256 *s, int v) { s->w[v >> 6] |= 1ULL << (v & 63); }
static inline int s_has(const set256 *s, int v) { return (s->w[v >> 6] >> (v & 63)) & 1; }

/* Im(D_a F) for every a, as 256-bit sets; F has 2^k entries */
static void images(const int *F, int k, set256 *Im) {
    /* F is quadratic (also in -g7 mode), so Im(D_a F) is the span of the k values B(a, e_i) = D_a F(e_i) */
    int K = 1 << k;
    for (int a = 1; a < K; a++) {
        uint8_t sp[256];
        int n = 1;
        sp[0] = 0;
        memset(&Im[a], 0, sizeof(set256));
        s_add(&Im[a], 0);
        for (int i = 0; i < k; i++) {
            int b = F[(1 << i) ^ a] ^ F[1 << i] ^ F[a] ^ F[0];
            if (s_has(&Im[a], b)) continue;
            for (int j = 0; j < n; j++) { sp[n + j] = sp[j] ^ b; s_add(&Im[a], sp[j] ^ b); }
            n *= 2;
        }
    }
}

/* Im[a] is a linear subspace of F_2^8 (dim 6 for a quadratic APN (7,8)-function). Precompute, for 0 < a < K,
 * the bitmask of each coset Im[a] + t, indexed by a coset id of t. The search only needs a = 1 (K = 2). */
static uint8_t coset_id[128][256];
static set256 coset_mask[128][256];
static int ncos[128];
static void cosets(const set256 *Im, int K) {
    for (int a = 1; a < K; a++) {
        memset(coset_id[a], 0xff, 256);
        ncos[a] = 0;
        for (int t = 0; t < 256; t++) {
            if (coset_id[a][t] != 0xff) continue;
            int id = ncos[a]++;
            memset(&coset_mask[a][id], 0, sizeof(set256));
            for (int v = 0; v < 256; v++) if (s_has(&Im[a], v)) { coset_id[a][v ^ t] = id; s_add(&coset_mask[a][id], v ^ t); }
        }
    }
}
static long nodes, node_limit = 2000000000L;
static int nocut = 0, nosearch = 0;
static long depthcount[8];  /* nodes by number of assigned basis vectors (statistics) */  /* test: disable the Lemma 4.5 symmetry cut */

static int G7[128], L7[128];
static set256 Im7[128];
static long out_count, cap_left;

typedef uint64_t v4u __attribute__((vector_size(32)));
#if defined(__clang__) || __GNUC__ >= 12
#define SHUF(x, a, b, c, d) __builtin_shufflevector(x, x, a, b, c, d)
#else
#define SHUF(x, a, b, c, d) __builtin_shuffle(x, (v4u){a, b, c, d})
#endif
#if defined(__aarch64__)
#include <arm_neon.h>
#elif defined(__AVX2__)
#include <immintrin.h>
/* the bytes of x permuted by i -> i ^ h (h < 32) */
static inline v4u bytexor(v4u x, int h) {
    __m256i a = (__m256i)x;
    __m256i idx = _mm256_xor_si256(_mm256_setr_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                                                    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15),
                                   _mm256_set1_epi8((char)(h & 15)));
    a = _mm256_shuffle_epi8(a, idx);
    __m256i b = _mm256_permute2x128_si256(a, a, 1);
    return (v4u)_mm256_blendv_epi8(a, b, _mm256_set1_epi8((char)-(h >> 4 & 1)));
}
#else
static inline v4u bytexor(v4u x, int h) {
    v4u t, m;
#define STAGE(k, M) t = ((x & M) << (8 << k)) | ((x >> (8 << k)) & M); m = (v4u){0, 0, 0, 0} - (uint64_t)(h >> k & 1); x ^= (x ^ t) & m;
    const v4u M0 = (v4u){0, 0, 0, 0} + 0x00ff00ff00ff00ffULL, M1 = (v4u){0, 0, 0, 0} + 0x0000ffff0000ffffULL;
    STAGE(0, M0) STAGE(1, M1)
#undef STAGE
    t = (x << 32) | (x >> 32); m = (v4u){0, 0, 0, 0} - (uint64_t)(h >> 2 & 1); x ^= (x ^ t) & m;
    t = SHUF(x, 1, 0, 3, 2); m = (v4u){0, 0, 0, 0} - (uint64_t)(h >> 3 & 1); x ^= (x ^ t) & m;
    t = SHUF(x, 2, 3, 0, 1); m = (v4u){0, 0, 0, 0} - (uint64_t)(h >> 4 & 1); x ^= (x ^ t) & m;
    return x;
}
#endif
/* the set {y ^ v : y in x} (bit y of the set is bit y & 7 of byte y >> 3) */
/* Inside a byte, the translate moves bit k to bit k ^ m (m = v & 7). NL[m][n] (NH[m][n]) is the byte that the low
 * (high) nibble n becomes, so a byte b becomes NL[m][b & 15] | NH[m][b >> 4]: two table lookups per byte. */
static uint8_t NL[8][32] __attribute__((aligned(32))), NH[8][32] __attribute__((aligned(32)));
__attribute__((constructor)) static void nibble_tables(void) {
    for (int m = 0; m < 8; m++) for (int n = 0; n < 32; n++)
        for (int k = 0; k < 4; k++) if (n >> k & 1) { NL[m][n] |= 1 << (k ^ m); NH[m][n] |= 1 << ((k + 4) ^ m); }
}
#if defined(__aarch64__)
static inline v4u vxlate(v4u x, int v) {
    static const uint8_t io[32] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                                   16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31};
    uint8x16x2_t t;
    memcpy(&t, &x, 32);
    uint8x16_t hh = vdupq_n_u8((uint8_t)(v >> 3)), f = vdupq_n_u8(15);
    uint8x16_t L = vld1q_u8(NL[v & 7]), H = vld1q_u8(NH[v & 7]);
    uint8x16_t r0 = vqtbl2q_u8(t, veorq_u8(vld1q_u8(io), hh)), r1 = vqtbl2q_u8(t, veorq_u8(vld1q_u8(io + 16), hh));
    r0 = vorrq_u8(vqtbl1q_u8(L, vandq_u8(r0, f)), vqtbl1q_u8(H, vshrq_n_u8(r0, 4)));
    r1 = vorrq_u8(vqtbl1q_u8(L, vandq_u8(r1, f)), vqtbl1q_u8(H, vshrq_n_u8(r1, 4)));
    v4u y;
    memcpy(&y, &r0, 16); memcpy((char *)&y + 16, &r1, 16);
    return y;
}
#elif defined(__AVX2__)
static inline v4u vxlate(v4u x, int v) {
    __m256i a = (__m256i)bytexor(x, v >> 3), f = _mm256_set1_epi8(15);
    __m256i L = _mm256_load_si256((const __m256i *)NL[v & 7]), H = _mm256_load_si256((const __m256i *)NH[v & 7]);
    return (v4u)_mm256_or_si256(_mm256_shuffle_epi8(L, _mm256_and_si256(a, f)),
                                _mm256_shuffle_epi8(H, _mm256_and_si256(_mm256_srli_epi16(a, 4), f)));
}
#else
static inline v4u vxlate(v4u x, int v) {
    x = bytexor(x, v >> 3);
    v4u t, m;
#define STAGE(k, M) t = ((x & M) << (1 << k)) | ((x >> (1 << k)) & M); m = (v4u){0, 0, 0, 0} - (uint64_t)(v >> k & 1); x ^= (x ^ t) & m;
    const v4u M0 = (v4u){0, 0, 0, 0} + 0x5555555555555555ULL, M1 = (v4u){0, 0, 0, 0} + 0x3333333333333333ULL,
              M2 = (v4u){0, 0, 0, 0} + 0x0f0f0f0f0f0f0f0fULL;
    STAGE(0, M0) STAGE(1, M1) STAGE(2, M2)
#undef STAGE
    return x;
}
#endif

/* The two parts of the translate by v = 8h + m: bytexor(x, h) moves byte i to byte i ^ h (AVX2 and plain C versions
 * above), bitxlate(x, m) moves bit k of every byte to bit k ^ m. A set translated by many values can keep its 8
 * bitxlate's; each translate is then one bytexor. */
#if defined(__aarch64__)
static inline v4u bytexor(v4u x, int h) {
    static const uint8_t io[32] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                                   16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31};
    uint8x16x2_t t;
    memcpy(&t, &x, 32);
    uint8x16_t hh = vdupq_n_u8((uint8_t)h);
    uint8x16_t r0 = vqtbl2q_u8(t, veorq_u8(vld1q_u8(io), hh)), r1 = vqtbl2q_u8(t, veorq_u8(vld1q_u8(io + 16), hh));
    v4u y;
    memcpy(&y, &r0, 16); memcpy((char *)&y + 16, &r1, 16);
    return y;
}
static inline v4u bitxlate(v4u x, int m) {
    uint8x16_t r0, r1, f = vdupq_n_u8(15);
    memcpy(&r0, &x, 16); memcpy(&r1, (char *)&x + 16, 16);
    uint8x16_t L = vld1q_u8(NL[m]), H = vld1q_u8(NH[m]);
    r0 = vorrq_u8(vqtbl1q_u8(L, vandq_u8(r0, f)), vqtbl1q_u8(H, vshrq_n_u8(r0, 4)));
    r1 = vorrq_u8(vqtbl1q_u8(L, vandq_u8(r1, f)), vqtbl1q_u8(H, vshrq_n_u8(r1, 4)));
    v4u y;
    memcpy(&y, &r0, 16); memcpy((char *)&y + 16, &r1, 16);
    return y;
}
#elif defined(__AVX2__)
static inline v4u bitxlate(v4u x, int m) {
    __m256i a = (__m256i)x, f = _mm256_set1_epi8(15);
    __m256i L = _mm256_load_si256((const __m256i *)NL[m]), H = _mm256_load_si256((const __m256i *)NH[m]);
    return (v4u)_mm256_or_si256(_mm256_shuffle_epi8(L, _mm256_and_si256(a, f)),
                                _mm256_shuffle_epi8(H, _mm256_and_si256(_mm256_srli_epi16(a, 4), f)));
}
#else
static inline v4u bitxlate(v4u x, int v) {
    v4u t, m;
#define STAGE(k, M) t = ((x & M) << (1 << k)) | ((x >> (1 << k)) & M); m = (v4u){0, 0, 0, 0} - (uint64_t)(v >> k & 1); x ^= (x ^ t) & m;
    const v4u M0 = (v4u){0, 0, 0, 0} + 0x5555555555555555ULL, M1 = (v4u){0, 0, 0, 0} + 0x3333333333333333ULL,
              M2 = (v4u){0, 0, 0, 0} + 0x0f0f0f0f0f0f0f0fULL;
    STAGE(0, M0) STAGE(1, M1) STAGE(2, M2)
#undef STAGE
    return x;
}
#endif
#if defined(__aarch64__)
static inline int vpop(v4u x) {
    uint8x16_t a, b;
    memcpy(&a, &x, 16); memcpy(&b, (char *)&x + 16, 16);
    return vaddvq_u8(vcntq_u8(a)) + vaddvq_u8(vcntq_u8(b));
}
#else
static inline int vpop(v4u x) {
    return __builtin_popcountll(x[0]) + __builtin_popcountll(x[1]) + __builtin_popcountll(x[2]) + __builtin_popcountll(x[3]);
}
#endif
#if defined(__AVX2__)
static inline int vany(v4u x) { return !_mm256_testz_si256((__m256i)x, (__m256i)x); }
#else
static inline int vany(v4u x) { return ((x[0] | x[1]) | (x[2] | x[3])) != 0; }
#endif

/* Backtracking over L7 : F_2^7 -> F_2^8, which needs L7(x) outside Im7[x] for every x != 0.
 * State at depth d: L7 is fixed on a subspace V of dimension d, and R is a set of 7 - d indices such that the
 * vectors e_U (U a nonempty subset of R) represent the nonzero cosets of V. Q[d][U] is the set of values that
 * L7(e_U) can still take:  { y : y ^ L7(beta) not in Im7[e_U ^ beta] for every beta in V }.
 * The next vector to fix is the e_U with fewest candidates among all 2^(7-d) - 1 sets (not only the basis vectors
 * e_j: that gives a smaller tree). Fixing L7(e_U) = y removes one index j of U from R, and for the remaining U'
 *   Q[d+1][U'] = Q[d][U']  and  (Q[d][U' ^ U] translated by y),
 * one translate and one AND per set. A node dies as soon as one of its sets is empty.
 * The whole tree is searched (up to -cap solutions), so the order of the candidates does not matter.
 * Depth 3 has its own routine, bt3().
 * History: until 2026-11-14 the search branched only on the basis vectors, in a random order, rebuilding the sets
 * from scratch at every node; the solutions are the same; checked against the earlier version on the same G7. */
static v4u Q[8][128];
static int R[8];
static int xs[8], ys[8];   /* the vector fixed at depth d and its value */
static int sym = 0;        /* 1: the search keeps one of each pair L, L + B(e_0, .) and emit() prints both (-cap counts pairs) */
static void emit1(void) {
    for (int x = 0; x < 256; x++) {
        int v = (x < 128) ? G7[x] : (G7[x - 128] ^ L7[x - 128]);
        printf("%d%c", v, x == 255 ? '\n' : ',');
    }
    out_count++;
}
static void emit(void) {
    int vec[128], val[128], n = 1;
    vec[0] = 0; val[0] = 0;
    for (int d = 0; d < 7; d++) {
        for (int i = 0; i < n; i++) { vec[n + i] = vec[i] ^ xs[d]; val[n + i] = val[i] ^ ys[d]; }
        n *= 2;
    }
    for (int i = 0; i < 128; i++) L7[vec[i]] = val[i];
    emit1();
    if (sym) {
        for (int x = 0; x < 128; x++) L7[x] ^= G7[1 ^ x] ^ G7[1] ^ G7[x] ^ G7[0];   /* + B(e_0, x) */
        emit1();
    }
}
static void bt(int d);
/* Depth 3 (4 indices left). Branch on bu (L7(e_bu) = y) and at once on a second vector W (L7(e_W) = z), then check
 * the three lines through W of the remaining quotient. Almost every (y, z) dies at the first line, so the first
 * stages run over all candidates without branches and only keep the survivors. */
#define MAXP 65537
static v4u SW[256], A0[256], B0[256], T0[MAXP];
static int YS[256], TI[MAXP], TZ[MAXP];
static void bt3(void) {
    const int d = 3;
    int Rd = R[d], pc[128];
    int bu = 0, bc = 1 << 30;
    for (int U = Rd; U; U = (U - 1) & Rd) { int c = pc[U] = vpop(Q[d][U]); if (c <= bc) { bc = c; bu = U; } }
    int R4 = Rd & ~(1 << __builtin_ctz(bu));
    int W = 0; long wc = 1L << 60;
    for (int U = R4; U; U = (U - 1) & R4) { long c = (long)pc[U] * pc[U ^ bu]; if (c <= wc) { wc = c; W = U; } }
    int R5 = R4 & ~(1 << __builtin_ctz(W));
    int P[3], np = 0;
    for (int U = R5; U; U = (U - 1) & R5) P[np++] = U;
    xs[3] = bu; xs[4] = W; R[4] = R4; R[5] = R5;
    /* stages 1 and 2: for every y, the set of z and the two sets of the first line through W */
    v4u sb = Q[d][bu], qW = Q[d][W], qWu = Q[d][W ^ bu];
    v4u a1 = Q[d][P[0]], a2 = Q[d][P[0] ^ bu], b1 = Q[d][P[0] ^ W], b2 = Q[d][P[0] ^ W ^ bu];
    v4u X8[3][8];   /* the three sets translated by every y, with their within-byte translates precomputed */
    for (int m8 = 0; m8 < 8; m8++) { X8[0][m8] = bitxlate(qWu, m8); X8[1][m8] = bitxlate(a2, m8); X8[2][m8] = bitxlate(b2, m8); }
    int n2 = 0;
    for (int w = 0; w < 4; w++) for (uint64_t m = sb[w]; m; m &= m - 1) {
        int y = w * 64 + __builtin_ctzll(m);
        int h = y >> 3, m8 = y & 7;
        v4u sw = qW & bytexor(X8[0][m8], h), a = a1 & bytexor(X8[1][m8], h), b = b1 & bytexor(X8[2][m8], h);
        SW[n2] = sw; YS[n2] = y; A0[n2] = a; B0[n2] = b;
        n2 += vany(sw) & vany(a) & vany(b);
    }
    nodes += bc; depthcount[4] += bc;
    /* stage 3: the first line for every (y, z). First the list of pairs (i, z): the lowest bit of each word is
     * written without a branch, the other bits (rare) in a loop. Unpredictable branches cost more than the translates. */
    int np2 = 0;
    for (int i = 0; i < n2; i++) {
        v4u sw = SW[i];
        for (int w = 0; w < 4; w++) {
            uint64_t m = sw[w];
            TI[np2] = i; TZ[np2] = w * 64 + __builtin_ctzll(m | 1ULL << 63);
            np2 += m != 0;
            for (m &= m - 1; m; m &= m - 1) { TI[np2] = i; TZ[np2++] = w * 64 + __builtin_ctzll(m); }
        }
    }
    nodes += np2; depthcount[5] += np2;
    int nt = 0;
    for (int j = 0; j < np2; j++) {
        int i = TI[j], z = TZ[j];
        v4u t = A0[i] & vxlate(B0[i], z);
        TI[nt] = i; TZ[nt] = z; T0[nt] = t;
        nt += vany(t);
    }
    if (nodes > node_limit) return;
    /* the few (y, z) left: the other two lines, then the generic search */
    for (int j = 0; j < nt; j++) {
        if (cap_left <= 0 || nodes > node_limit) return;
        int y = YS[TI[j]], z = TZ[j];
        ys[3] = y; ys[4] = z;
        Q[5][P[0]] = T0[j];
        int k;
        for (k = 1; k < 3; k++) {
            v4u a = Q[d][P[k]] & vxlate(Q[d][P[k] ^ bu], y), b = Q[d][P[k] ^ W] & vxlate(Q[d][P[k] ^ W ^ bu], y);
            v4u t = a & vxlate(b, z);
            Q[5][P[k]] = t;
            if (!vany(t)) break;
        }
        if (k < 3) continue;
        bt(5);
    }
}
static void bt(int d) {
    if (cap_left <= 0 || ++nodes > node_limit) return;
    depthcount[d]++;
    if (d == 3) { bt3(); return; }
    int Rd = R[d];
    if (!Rd) { emit(); cap_left--; return; }
    int bu = 0, bc = 1 << 30;
    if (d == 0 && !nocut) bu = 1;
    else for (int U = Rd; U; U = (U - 1) & Rd) { int c = vpop(Q[d][U]); if (c <= bc) { bc = c; bu = U; } }
    int R2 = Rd & ~(1 << __builtin_ctz(bu));
    R[d + 1] = R2;
    xs[d] = bu;
    v4u sb = Q[d][bu];
    /* The remaining symmetry, alpha = e_0 (see below): it keeps v_0 and maps L7(x) to L7(x) ^ B(e_0, x), with
     * B(e_0, x) != 0 for x not in {0, e_0}. At depth 1 keep the smaller of y and y ^ B(e_0, x); emit() prints both. */
    int bb = (sym && d == 1) ? (G7[1 ^ bu] ^ G7[1] ^ G7[bu] ^ G7[0]) : 0;
    int seen = 0;
    for (int w = 0; w < 4; w++) for (uint64_t m = sb[w]; m; m &= m - 1) {
        if (cap_left <= 0 || nodes > node_limit) return;
        int y = w * 64 + __builtin_ctzll(m);
        if (d == 0 && !nocut) {
            /* Symmetry (Lemma 4.5 of arXiv:2508.04644): L and L + B(alpha, .) give EA-equivalent extensions, where
             * B(alpha, x) = D_x G7(alpha) is bilinear. Adding B(alpha, .) shifts v_0 = L(e_0) by B(alpha, e_0),
             * which runs over Im7[e_0] as alpha varies. So v_0 is fixed to one representative per coset of Im7[e_0]
             * (Q[0][1] already excludes the coset Im7[e_0] itself). */
            int bit = 1 << coset_id[1][y];
            if (seen & bit) continue;
            seen |= bit;
            sym = 1;
        }
        if (bb && y > (y ^ bb)) continue;
        ys[d] = y;
        int dead = 0;
        if (d == 2) {   /* the children of depth 2 almost never die here: no early exit */
            for (int U = R2; U; U = (U - 1) & R2) {
                v4u q = Q[d][U] & vxlate(Q[d][U ^ bu], y);
                Q[d + 1][U] = q;
                dead |= !vany(q);
            }
        } else
        for (int U = R2; U; U = (U - 1) & R2) {
            v4u q = Q[d][U] & vxlate(Q[d][U ^ bu], y);
            Q[d + 1][U] = q;
            if (!vany(q)) { dead = 1; break; }
        }
        if (dead) { if (++nodes <= node_limit) depthcount[d + 1]++; continue; }   /* = a node that dies at once */
        bt(d + 1);
    }
}
/* start the search: Q[0][U] is the complement of Im7[e_U] */
static void bt0(void) {
    sym = 0;
    R[0] = 127;
    for (int U = 1; U < 128; U++) {
        const set256 *m = &Im7[U];
        Q[0][U] = ~(v4u){m->w[0], m->w[1], m->w[2], m->w[3]};
    }
    bt(0);
}

int main(int argc, char **argv) {
    const char *in = NULL;
    long starts = 1, tries = 256, cap = 64;
    const char *g7file = NULL, *nbrfile = NULL; int nbrpos = -1; long massS = 0;
    const char *dumpfile = NULL;  /* learning data: label + bilinear form of each tested G7 */
    int dump_neg_permille = 10;
    int skipr0 = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-in")) in = argv[++i];
        else if (!strcmp(argv[i], "-seed")) {
#ifdef SPLITMIX
            /* splitmix64 steps its state by a constant, so the state must not be linear in the seed (seed s+1 would
             * give the stream of seed s shifted by one output). Hash the seed into the state. */
            uint64_t z = strtoull(argv[++i], 0, 10) ^ 0x5DEECE66DULL;
            for (int k = 0; k < 3; k++) { z ^= z >> 33; z *= 0xFF51AFD7ED558CCDULL; z ^= z >> 33; z *= 0xC4CEB9FE1A85EC53ULL; z ^= z >> 33; }
            rs = z;
#else
            rs = strtoull(argv[++i], 0, 10) * 0x9E3779B97F4A7C15ULL + 7; for (int k = 0; k < 9; k++) rnd();
#endif
        }
        else if (!strcmp(argv[i], "-starts")) starts = atol(argv[++i]);
        else if (!strcmp(argv[i], "-tries")) tries = atol(argv[++i]);
        else if (!strcmp(argv[i], "-cap")) cap = atol(argv[++i]);
        else if (!strcmp(argv[i], "-nodes")) node_limit = atol(argv[++i]);
        else if (!strcmp(argv[i], "-g7")) g7file = argv[++i];
        else if (!strcmp(argv[i], "-mass")) massS = atol(argv[++i]);
        else if (!strcmp(argv[i], "-nbr")) nbrfile = argv[++i];
        else if (!strcmp(argv[i], "-nbrpos")) nbrpos = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-nocut")) nocut = 1;
        else if (!strcmp(argv[i], "-nosearch")) nosearch = 1;   /* test: count valid L6 only */
        else if (!strcmp(argv[i], "-dump")) dumpfile = argv[++i];
        else if (!strcmp(argv[i], "-dumpneg")) dump_neg_permille = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-skipr0")) skipr0 = 1;
    }
    if (massS) {
        /* -mass S: S starts; each start is a uniform APN beta6 (F_2^6 -> F_2^8), built variable by variable from
         * scratch with random L at every level and a restart from scratch on any rejection; then -tries random L6,
         * the L7 search on each valid one. Unbiased estimate of the number of APN alternating bilinear
         * beta8 : N_beta = 2^120 * P(chain accepted) * 2^48 * (positive outputs * 64) / tries.
         * stderr every 1000 starts and at the end: starts chain_attempts tries valid outputs nonstd */
        FILE *g7dump = dumpfile ? fopen(dumpfile, "wb") : NULL;   /* -dump in -mass mode: 128-byte G7 + int32 outputs per valid L6 (GPU tests) */
        long att = 0, ntries = 0, nvalid = 0, nout = 0, nonstd = 0;
        int F[128]; set256 Im[128];
        for (long s = 0; s < massS; s++) {
            for (;;) {   /* uniform APN beta6 */
                att++;
                F[0] = 0; F[1] = 0; int ok = 1;
                for (int k = 1; k <= 5 && ok; k++) {
                    int K = 1 << k, L[64], v[8];
                    images(F, k, Im);
                    for (int i = 0; i < k; i++) v[i] = rnd() & 255;
                    L[0] = 0;
                    for (int a = 1; a < K && ok; a++) { L[a] = L[a & (a - 1)] ^ v[__builtin_ctz(a)]; if (s_has(&Im[a], L[a])) ok = 0; }
                    if (ok) for (int x = 0; x < K; x++) F[K + x] = F[x] ^ L[x];
                }
                if (ok) break;
            }
            images(F, 6, Im);
            for (long t = 0; t < tries; t++) {
                int L6[64], v6[6], a;
                ntries++;
                for (int d = 0; d < 6; d++) v6[d] = rnd() & 255;
                L6[0] = 0;
                for (a = 1; a < 64; a++) { L6[a] = L6[a & (a - 1)] ^ v6[__builtin_ctz(a)]; if (s_has(&Im[a], L6[a])) break; }
                if (a < 64) continue;
                nvalid++;
                if (nosearch) continue;
                for (int x = 0; x < 64; x++) { G7[x] = F[x]; G7[64 + x] = F[x] ^ L6[x]; }
                images(G7, 7, Im7); cosets(Im7, 2);
                nodes = 0; L7[0] = 0; cap_left = cap;
                long before = out_count; bt0();
                long o = out_count - before; nout += o; if (o != 0 && o != 2) nonstd++;
                if (g7dump) { uint8_t b[132]; for (int x = 0; x < 128; x++) b[x] = G7[x]; int32_t oo = o; memcpy(b + 128, &oo, 4); fwrite(b, 1, 132, g7dump); }
                if (nodes > node_limit) nonstd += 1000000;
            }
            if (g7dump && s + 1 == massS) fclose(g7dump);
            if ((s + 1) % 1000 == 0 || s + 1 == massS) { fprintf(stderr, "%ld %ld %ld %ld %ld %ld%s\n", s + 1, att, ntries, nvalid, nout, nonstd, RNGTAG); fflush(stderr); }
        }
        return 0;
    }
    if (nbrfile) {
        /* -nbr FILE: lines "id hex42" = bilinear form of a positive G7 (pairs i<j<7). For every one-entry change
         * B(e_i,e_j) ^= delta, test APN and run the L7 search. stderr: id pos delta apn outputs nodes. */
        FILE *g = fopen(nbrfile, "r"); char id[64], hx[64];
        while (fscanf(g, "%63s %63s", id, hx) == 2) {
            int B[21]; for (int p = 0; p < 21; p++) { unsigned v; sscanf(hx + 2 * p, "%2x", &v); B[p] = v; }
            for (int pos = -1; pos < 21; pos++) for (int del = (pos < 0 ? 0 : 1); del < (pos < 0 ? 1 : 256); del++) {
                if (nbrpos >= 0 && pos >= 0 && pos != nbrpos) continue;
                int Bp[21]; memcpy(Bp, B, sizeof B); if (pos >= 0) Bp[pos] ^= del;
                for (int x = 0; x < 128; x++) { int v = 0, p = 0;
                    for (int i = 0; i < 7; i++) for (int j = i + 1; j < 7; j++, p++) if ((x >> i & 1) && (x >> j & 1)) v ^= Bp[p];
                    G7[x] = v; }
                images(G7, 7, Im7);
                int ok = 1;
                for (int a = 1; a < 128 && ok; a++) { int c = 0; for (int w = 0; w < 4; w++) c += __builtin_popcountll(Im7[a].w[w]); if (c != 64) ok = 0; }
                long before = out_count; nodes = 0;
                if (ok) { cosets(Im7, 2); L7[0] = 0; cap_left = cap; bt0(); }
                fprintf(stderr, "%s %d %d %d %ld %ld\n", id, pos, del, ok, out_count - before, nodes);
            }
        }
        fprintf(stderr, "depth"); for (int d = 0; d < 8; d++) fprintf(stderr, " %ld", depthcount[d]); fprintf(stderr, "\n");
        fflush(stdout); return 0;
    }
    if (g7file) {
        FILE *g = fopen(g7file, "r");
        for (int x = 0; x < 128; x++) if (fscanf(g, " %d ,", &G7[x]) != 1) { fprintf(stderr, "bad -g7\n"); return 1; }
        fclose(g);
        images(G7, 7, Im7); cosets(Im7, 2);
        int dims_ok = 1;
        for (int a = 1; a < 128; a++) { int c = 0; for (int w = 0; w < 4; w++) c += __builtin_popcountll(Im7[a].w[w]); if (c != 64) dims_ok = 0; }
        fprintf(stderr, "test mode: all Im(D_a G7) of size 64: %s\n", dims_ok ? "yes" : "no");
        L7[0] = 0; cap_left = cap; nodes = 0; bt0();
        fprintf(stderr, "test mode: %ld nodes, %ld outputs; nodes by depth:", nodes, out_count);
        for (int d = 0; d < 8; d++) fprintf(stderr, " %ld", depthcount[d]);
        fprintf(stderr, "\n");
        return 0;
    }
    if (!in) { fprintf(stderr, "usage: varext -in QAPN-6-8 -seed S -starts K [-tries T] [-cap C] [-skipr0]\n"); return 1; }
    /* load all start functions (64 hex values per line) */
    FILE *f = fopen(in, "r");
    if (!f) { perror(in); return 1; }
    int cap_lines = 1 << 20, nlines = 0;
    uint8_t (*S6)[64] = malloc((size_t)cap_lines * 64);
    long *orig = malloc((size_t)cap_lines * sizeof(long));  /* line number in the input file */
    long lineno = 0;
    char line[2048];
    while (fgets(line, sizeof line, f)) {
        char *p = line; int k = 0;
        while (*p && k < 64) {
            while (*p == ',' || *p == ' ') p++;
            if (*p == '\n' || !*p) break;
            S6[nlines][k++] = (uint8_t)strtol(p, &p, 16);
        }
        if (k != 64) continue;
        long this_line = lineno++;
        if (skipr0) {
            /* rank of the 15 values B(e_i, e_j), i < j < 6, by xor-basis insertion */
            uint8_t basis[8] = {0};
            int r = 0;
            for (int i = 0; i < 6; i++) for (int j = i + 1; j < 6; j++) {
                const uint8_t *F = S6[nlines];
                uint8_t v = F[(1 << i) ^ (1 << j)] ^ F[1 << i] ^ F[1 << j] ^ F[0];
                for (int b = 7; b >= 0 && v; b--) {
                    if (!(v >> b & 1)) continue;
                    if (!basis[b]) { basis[b] = v; r++; v = 0; } else v ^= basis[b];
                }
            }
            if (r < 8) continue;
        }
        orig[nlines] = this_line;
        if (++nlines == cap_lines) break;
    }
    fclose(f);
    fprintf(stderr, "varext: %d start functions%s\n", nlines, skipr0 ? " (without vanishing components)" : "");
    FILE *dump = dumpfile ? fopen(dumpfile, "a") : NULL;
    int F6[64];
    set256 Im6[64];
    for (long s = 0; s < starts; s++) {
        long idx = rnd() % nlines;
        for (int x = 0; x < 64; x++) F6[x] = S6[idx][x];
        images(F6, 6, Im6);
        for (long t = 0; t < tries; t++) {
            /* random linear L6 */
            int L6[64], v6[6];
            for (int d = 0; d < 6; d++) v6[d] = rnd() & 255;
            /* L6(a), built in the order a = 1, 2, ..., stops at the first a with L6(a) in Im6[a] */
            int a;
            L6[0] = 0;
            for (a = 1; a < 64; a++) {
                L6[a] = L6[a & (a - 1)] ^ v6[__builtin_ctz(a)];
                if (s_has(&Im6[a], L6[a])) break;
            }
            if (a < 64) continue;
            for (int x = 0; x < 64; x++) { G7[x] = F6[x]; G7[64 + x] = F6[x] ^ L6[x]; }
            images(G7, 7, Im7);
            cosets(Im7, 2);
            nodes = 0;
            L7[0] = 0;
            cap_left = cap;
            long before = out_count;
            bt0();  /* A = 0: nothing assigned */
            if (dump) {
                int label = out_count > before;
                if (label || (long)(rnd() % 1000) < dump_neg_permille) {
                    /* bilinear form of G7 on pairs (i, j), i < j < 7: B(e_i, e_j) = G7(e_i+e_j)+G7(e_i)+G7(e_j)+G7(0) */
                    fprintf(dump, "%d %ld ", label, orig[idx]);
                    for (int i = 0; i < 7; i++) for (int j = i + 1; j < 7; j++) {
                        int a = 1 << i, b2 = 1 << j;
                        fprintf(dump, "%02x", G7[a ^ b2] ^ G7[a] ^ G7[b2] ^ G7[0]);
                    }
                    fprintf(dump, "\n");
                    fflush(dump);
                }
            }
            fprintf(stderr, "start %ld (line %ld) try %ld: %ld nodes, %ld outputs so far%s\n", s, orig[idx], t, nodes, out_count,
                    nodes > node_limit ? " (node limit)" : "");
        }
    }
    fflush(stdout);
    return 0;
}
