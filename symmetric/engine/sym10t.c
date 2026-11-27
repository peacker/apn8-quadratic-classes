/*
 * sym10: exhaustive search for quadratic APN functions of F_2^n (n <= 10) in a linear space W of quadratic forms,
 * typically the forms with a prescribed self-equivalence (input from sym_setup.py).
 *
 * F is given by beta(e_i, e_j) = sum_k t_k w_k(i, j), t in F_2^d (w_k: reduced row echelon basis of W in search
 * order). Search order: input variables e_1, e_2, .. (level s adds e_s), inside a level the pairs (0, s), (1, s), ..
 * At pair (i, s) the bits t_k whose pivot lies in that pair are chosen (all 2^(#) combinations); the value
 * y = beta(e_i, e_s) is then final. APN on V' = <e_0..e_s> (variable approach): L(a) = beta(a, e_s) must avoid
 * Im_V beta(a, .) for every a != 0 in V = <e_0..e_{s-1}>; L(a) is checked as soon as y for the highest bit i of a is
 * known. A leaf (all levels done) is a quadratic APN function; it is printed as a lookup table
 * F(x) = sum_{i<j} x_i x_j beta(e_i, e_j) (F agrees with the symmetric function up to linear terms).
 * Output normalisation: beta(e_0, e_1) is restricted to the orbit representatives listed after the basis (see
 * sym_setup.py); -nonorm disables it.
 * usage: sym10 -knuth PROBES [-seed S] < W.txt
 *        sym10 -dfs [-print MAX | -dedupe] [-limit NODES] [-shard I K [-sdepth D]] [-pshard R K0 -pdepth D0] [-symdepth P] < W.txt
 * -symdepth P: output-side symmetry reduction (stabiliser chain in C(B)) on the first P pairs; replaces the
 * beta(e_0, e_1) normalisation. Needs the Z(B) basis at the end of W.txt (sym_setup.py).
 * -cp K1 K2: input-side checkpoint pruning when beta is complete on V_k, K1 <= k <= K2 (needs the Z(A) basis);
 * -nsin M: number of sampled input maps per k (default 24).
 * -progress FILE: every 60 s write elapsed time, nodes, estimated explored fraction and time left to FILE.
 * -offset FILE: lines "pair value": affine search space, beta = offset + span(W) (from intertwine.py).
 * -force FILE: lines "pair value": the value of beta on that pair is fixed (e.g. beta on Fix(A) x Fix(A) fixed to a
 * class representative). Combine with -cp only together with -cpfix U, when every forced pair lies in
 * <e_0..e_{U-1}> x <e_0..e_{U-1}>: the input maps are then the identity there, forced pairs are compared for
 * equality (not minimised) in the output canonical form, so only maps that keep the forced prefix are used.
 * -dedupe: print every leaf whose ortho-derivative spectrum fingerprint was not seen before in this process
 * ("K<fingerprint> <lookup table>"); the fingerprint is a cheap invariant, so classify the printed tables further.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAXN 10
#define MAXP 45
#define MAXD 460
static int n, N, d, P, pairs_i[MAXP], pairs_j[MAXP], pidx[MAXN][MAXN];
static uint16_t wv[MAXD][MAXP];                    /* basis vectors: value at each pair */
static int pivpair[MAXD];
static uint8_t allowed01[1 << MAXN];               /* allowed values of beta(e_0, e_1) (output normalisation) */
static int forced[MAXP];                            /* -force: forced[p] = value + 1 (0 = free) */
static uint16_t offset[MAXP];                       /* -offset: beta = offset + sum t_k w_k (affine search space) */
static int kfirst[MAXP], kcount[MAXP];             /* basis vectors with pivot in pair p: kfirst[p] .. +kcount[p] */
static uint16_t val[MAXP];                         /* current pair values (sum of chosen t_k w_k) */
static uint16_t hb[1 << MAXN][MAXN];               /* hb[a][j] = beta(a, e_j) for a in the current span */
static uint16_t fb[MAXN][1 << MAXN][MAXN];         /* per level s and a in V: reduced echelon basis of */
static uint8_t fbn[MAXN][1 << MAXN];               /* Im_V beta(a, .) (pivot = highest bit, decreasing) */
static uint16_t Lv[MAXN][1 << MAXN];
static long long nodes, leaves, node_limit = -1;
static int print_max = 0, printed = 0, nonorm = 0, dedupe = 0;
static int shard = 0, nshards = 1, shard_depth = 6;
/* -sh I K D (repeatable, increasing D): nested slices: keep only every K-th node (offset I) at depth D, counted among
 * the nodes kept by the shallower filters. A slice that is too slow is split by appending one more -sh. */
static int nsh = 0, sh_i[8], sh_k[8], sh_d[8]; static long long sh_cnt[8];
static long long shard_count = 0;
/* -pshard R K0 -pdepth D0: a first shard level (keep the depth-D0 nodes whose counter is R mod K0), so that one heavy
 * shard of an earlier run (same R, K0, D0) can be split again at a deeper -sdepth */
static int pshard = 0, npshards = 1, pshard_depth = 0;
static long long pshard_count = 0;
static uint64_t rs = 88172645463325252ULL;
static inline uint64_t rnd(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return rs; }
static uint64_t rs2 = 0x2545F4914F6CDD1DULL;          /* separate generator for the Knuth probes (the pruning one is */
static inline uint64_t rnd2(void) { rs2 ^= rs2 << 13; rs2 ^= rs2 >> 7; rs2 ^= rs2 << 17; return rs2; }  /* reseeded per node) */

/* forbidden values of L(a) at level s: the subspace Im_V beta(a, .) = span{beta(a, e_j) : j < s}; stored as a
 * reduced echelon basis (O(2^s s^2) per level instead of a 2^s x 2^s table) */
/* -tr: input translations. U_s = {u in V_s : some X in Z(A) fixes V_s pointwise, X e_j - e_j in V_s for j >= s and
 * X e_s = e_s + u}. X is invertible (X = I + N, N^2 = 0), keeps every pair (i, j) with j < s, and adds the vector
 * c(u) = (beta(e_i, u))_{i<s} to the level-s values T_s = (beta(e_i, e_s))_{i<s}. So only the lexicographically least
 * member of T_s + span{c(u)} can be the least member of its orbit: with a reduced echelon basis of that span (pivot =
 * most significant bit, component 0 first), T_s is least iff it has 0 at every pivot. trmask[s][i] = the pivot bits
 * in component i. Same soundness argument as -symdepth / -cp: the least element of each orbit is never pruned. */
static int use_tr = 0, nU[MAXN + 1];
static uint16_t Ub[MAXN + 1][MAXN];
static uint16_t trmask[MAXN + 1][MAXN];
static long long tr_skipped = 0;
static uint16_t trB[MAXN + 1][MAXN][MAXN]; static int trpc[MAXN + 1][MAXN], trnb[MAXN + 1];
/* -scale ACOLS BCOLS: also the maps X in Z(A) with X = id on V_s and X e_s = A^k e_s (when V_s is A-invariant and
 * such an invertible X exists). Then beta(x, X e_s) = B^k beta(A^-k x, e_s), so T_s -> T'_s = B^k T_s(A^-k .). With
 * the translations the orbit of T_s contains red(T'_s) (reduced modulo span c(u)); prune if it is lexicographically
 * smaller than T_s on the prefix of components already known (pivots in component c only change components >= c). */
static int use_scale = 0, Acol[MAXN], Bcol[MAXN], ordA = 1;
static int nk[MAXN + 1], sk_k[MAXN + 1][16];
static uint16_t AinvK[16][MAXN], BK[16][MAXN];          /* columns of A^-k and B^k, k < ordA */
static int sc_m[MAXN + 1][16][MAXN];                     /* prefix length decidable at pair (i, s) */
static long long sc_pruned = 0;
static void tr_level(int s) {
    memset(trmask[s], 0, sizeof trmask[s]);
    if (!use_tr || !nU[s]) return;
    uint16_t B[MAXN][MAXN]; int pc[MAXN], pb[MAXN], nb = 0;     /* basis vectors, pivot component and bit */
    for (int q = 0; q < nU[s]; q++) {
        uint16_t v[MAXN];
        for (int i = 0; i < s; i++) v[i] = hb[Ub[s][q]][i];       /* beta(u, e_i) */
        for (int r = 0; r < nb; r++) if ((v[pc[r]] >> pb[r]) & 1) for (int i = 0; i < s; i++) v[i] ^= B[r][i];
        int c = 0; while (c < s && !v[c]) c++;
        if (c == s) continue;
        int b = 31 - __builtin_clz(v[c]);
        for (int r = 0; r < nb; r++) if ((B[r][c] >> b) & 1) for (int i = 0; i < s; i++) B[r][i] ^= v[i];
        memcpy(B[nb], v, sizeof v); pc[nb] = c; pb[nb] = b; nb++;
    }
    for (int r = 0; r < nb; r++) trmask[s][pc[r]] |= (uint16_t)(1 << pb[r]);
    memcpy(trB[s], B, sizeof B); memcpy(trpc[s], pc, sizeof pc); trnb[s] = nb;
}

static void level_start(int s) {
    int V = 1 << s;
    tr_level(s);
    for (int a = 1; a < V; a++) {
        uint16_t *B = fb[s][a]; int nb = 0;
        for (int j = 0; j < s; j++) {
            uint16_t v = hb[a][j];
            for (int q = 0; q < nb; q++) if ((v >> (31 - __builtin_clz(B[q]))) & 1) v ^= B[q];
            if (!v) continue;
            int pv = 31 - __builtin_clz(v);
            for (int q = 0; q < nb; q++) if ((B[q] >> pv) & 1) B[q] ^= v;
            int q = nb++;                                  /* insert keeping pivots decreasing */
            while (q > 0 && B[q - 1] < v) { B[q] = B[q - 1]; q--; }
            B[q] = v;
        }
        fbn[s][a] = nb;
    }
    Lv[s][0] = 0;
}

static inline int forbidden(int s, int a, uint16_t L) {
    const uint16_t *B = fb[s][a]; int nb = fbn[s][a];
    for (int q = 0; q < nb; q++) if ((L >> (31 - __builtin_clz(B[q]))) & 1) L ^= B[q];
    return L == 0;
}

static void level_end(int s) {
    int V = 1 << s;
    for (int a = 0; a < V; a++) {
        hb[a][s] = Lv[s][a];
        for (int j = 0; j < s; j++) hb[V | a][j] = hb[a][j] ^ Lv[s][1 << j];
        hb[V | a][s] = Lv[s][a];
    }
}

/* check y = beta(e_i, e_s) at level s: L(a) for a in [2^i, 2^(i+1)) */
static int check(int s, int i, uint16_t y) {
    int lo = 1 << i, hi = 2 << i;
    for (int a = lo; a < hi; a++) {
        uint16_t L = Lv[s][a ^ lo] ^ y;
        if (forbidden(s, a, L)) return 0;
    }
    return 1;
}

/* -ahead: exact look-ahead. At pair (i, s) the pairs (k, s+1), k < em[p], may already be fixed by W (no free bit
 * left for them). Their level-(s+1) APN check for a in V_m needs only beta(a, e_j), j <= s, which is known for
 * a < 2^(i+1). So that check is done now instead of after level s: L(a) = beta(a, e_(s+1)) must not lie in
 * span(fb[s][a], beta(a, e_s)). Same test as check(s+1, ...) later, so the search is unchanged. */
static int use_ahead = 0, em[MAXP], emprev[MAXP];
static long long ahead_pruned = 0;
static int early_ok(int s, int i, uint16_t y, const uint16_t *nx) {
    int p = pidx[i][s], m = em[p], m0 = emprev[p], lo = 1 << i;
    for (int a = 1 << m0; a < (1 << m); a++) {
        uint16_t Ls = (a & lo) ? (uint16_t)(Lv[s][a ^ lo] ^ y) : Lv[s][a];
        uint16_t L = 0; for (int k = 0; k < m; k++) if ((a >> k) & 1) L ^= nx[k];
        const uint16_t *B = fb[s][a]; int nb = fbn[s][a];
        for (int q = 0; q < nb; q++) { int pv = 31 - __builtin_clz(B[q]); if ((L >> pv) & 1) L ^= B[q]; if ((Ls >> pv) & 1) Ls ^= B[q]; }
        if (L == 0 || L == Ls) { ahead_pruned++; return 0; }
    }
    return 1;
}

static void set_y(int s, int i, uint16_t y) {
    int lo = 1 << i, hi = 2 << i;
    for (int a = lo; a < hi; a++) Lv[s][a] = Lv[s][a ^ lo] ^ y;
}

static void apply_t(int k) {                        /* toggle t_k: val += w_k */
    for (int p = pivpair[k]; p < P; p++) val[p] ^= wv[k][p];
}

static void print_leaf(void) {
    for (int x = 0; x < N; x++) {
        int y = 0;
        for (int p = 0; p < P; p++) if (((x >> pairs_i[p]) & (x >> pairs_j[p])) & 1) y ^= val[p];
        printf("%d%c", y, x == N - 1 ? '\n' : ',');
    }
}


/* cheap class fingerprint of a leaf: differential spectrum of the ortho-derivative pi (pi(a) = the nonzero u
 * orthogonal to the image of x -> beta(a, x)); only the first leaf with a given fingerprint is printed */
#define HSIZE (1 << 20)
static uint64_t seen_keys[HSIZE];
static long long distinct = 0;
static uint64_t leaf_key(void) {
    static int pi[1 << MAXN], cnt[(1 << MAXN) + 1], dd[1 << MAXN];
    pi[0] = 0;
    for (int a = 1; a < N; a++) {
        int R[MAXN], piv[MAXN], nr = 0;
        for (int m = 0; m < n; m++) {
            int vv = hb[a][m];
            for (int q = 0; q < nr; q++) if ((vv >> piv[q]) & 1) vv ^= R[q];
            if (!vv) continue;
            int pb = 31 - __builtin_clz(vv);
            for (int q = 0; q < nr; q++) if ((R[q] >> pb) & 1) R[q] ^= vv;
            R[nr] = vv; piv[nr++] = pb;
        }
        int pivmask = 0;
        for (int q = 0; q < nr; q++) pivmask |= 1 << piv[q];
        int fcol = 0;
        while ((pivmask >> fcol) & 1) fcol++;
        int u = 1 << fcol;
        for (int q = 0; q < nr; q++) if ((R[q] >> fcol) & 1) u |= 1 << piv[q];
        pi[a] = u;
    }
    memset(cnt, 0, sizeof(int) * (N + 1));
    for (int a = 1; a < N; a++) {
        memset(dd, 0, sizeof(int) * N);
        for (int x = 0; x < N; x++) dd[pi[x ^ a] ^ pi[x]]++;
        for (int b = 0; b < N; b++) cnt[dd[b]]++;
    }
    uint64_t h = 1469598103934665603ULL;
    for (int v = 0; v <= N; v++) if (cnt[v]) h = (h ^ (uint64_t)(v * 1000003 + cnt[v])) * 1099511628211ULL;
    /* stronger key (2026-11-20): also the distribution of |Walsh| values of the ortho-derivative. Two classes with the
     * same differential fingerprint were hidden from each other by -dedupe (only the first leaf per key is printed). */
    static int wc[(1 << MAXN) + 1], wv2[1 << MAXN];
    memset(wc, 0, sizeof(int) * (N + 1));
    for (int b = 1; b < N; b++) {
        for (int x = 0; x < N; x++) wv2[x] = __builtin_parity(b & pi[x]) ? -1 : 1;
        for (int len = 1; len < N; len <<= 1)
            for (int i = 0; i < N; i += len << 1)
                for (int j = i; j < i + len; j++) { int u = wv2[j], v = wv2[j + len]; wv2[j] = u + v; wv2[j + len] = u - v; }
        for (int a = 0; a < N; a++) wc[wv2[a] < 0 ? -wv2[a] : wv2[a]]++;
    }
    for (int v = 0; v <= N; v++) if (wc[v]) h = (h ^ (uint64_t)(v * 7919 + wc[v] + 0x5bd1e995ULL)) * 1099511628211ULL;
    return h ? h : 1;
}

/* ---- output-side symmetry: stabiliser chain in the centraliser C(B) ----
 * Z(B) = {X : XB = BX} has basis Mz[0..z-1]. The stabiliser of the prefix values y_0..y_{p-1} in C(B) is the set of
 * invertible X = sum c_j Mz[j] with X y_i = y_i: an affine system in c, kept in reduced echelon form per pair index.
 * At pair p only one value per orbit (under sampled stabiliser elements; union-find, so orbits can only be too fine:
 * always sound) is tried. Used for pairs p < symdepth while the stabiliser is not trivial. */
#define MAXZ 128
typedef unsigned __int128 u128;
static int z = 0, symdepth = 0, nsamp = 12;
static uint16_t Mz[MAXZ][MAXN];
typedef struct { int rank; u128 row[MAXZ]; uint8_t rhs[MAXZ]; int piv[MAXZ]; } Sys;
static Sys sys[MAXP + 1];
static int uf[1 << MAXN];
static long long sym_skipped = 0;

static int ctz128(u128 v) { uint64_t lo = (uint64_t)v; return lo ? __builtin_ctzll(lo) : 64 + __builtin_ctzll((uint64_t)(v >> 64)); }
static int par128(u128 v) { return __builtin_parityll((uint64_t)v) ^ __builtin_parityll((uint64_t)(v >> 64)); }

static void sys_add(Sys *S, u128 coef, int rhs) {
    for (int q = 0; q < S->rank; q++) if ((coef >> S->piv[q]) & 1) { coef ^= S->row[q]; rhs ^= S->rhs[q]; }
    if (!coef) return;                                  /* dependent (consistent: X = I is always a solution) */
    int pv = ctz128(coef);
    for (int q = 0; q < S->rank; q++) if ((S->row[q] >> pv) & 1) { S->row[q] ^= coef; S->rhs[q] ^= rhs; }
    S->row[S->rank] = coef; S->rhs[S->rank] = rhs; S->piv[S->rank] = pv; S->rank++;
}

static void sys_fix_vector(Sys *S, uint16_t y) {      /* add X y = y */
    for (int r = 0; r < n; r++) {
        u128 coef = 0;
        for (int j = 0; j < z; j++) {
            uint16_t im = 0;
            for (int i = 0; i < n; i++) if ((y >> i) & 1) im ^= Mz[j][i];
            if ((im >> r) & 1) coef |= (u128)1 << j;
        }
        sys_add(S, coef, (y >> r) & 1);
    }
}

static int rank10(const uint16_t *cols) {
    uint16_t r[MAXN]; memcpy(r, cols, sizeof(uint16_t) * n);
    int k = 0;
    for (int b = 0; b < n; b++) {
        int pv = -1;
        for (int i = k; i < n; i++) if ((r[i] >> b) & 1) { pv = i; break; }
        if (pv < 0) continue;
        uint16_t t = r[pv]; r[pv] = r[k]; r[k] = t;
        for (int i = 0; i < n; i++) if (i != k && ((r[i] >> b) & 1)) r[i] ^= r[k];
        k++;
    }
    return k;
}

/* deterministic randomness: the samples used for pruning at a node depend only on the prefix (pairs 0..p-1) and the
 * base seed, so that all shards make identical pruning decisions above the shard depth (otherwise the shards would
 * not partition the same tree). */
static uint64_t base_seed = 0x9E3779B97F4A7C15ULL;
static void reseed(int p) {
    uint64_t h = base_seed ^ (uint64_t)(p + 1) * 0xBF58476D1CE4E5B9ULL;
    for (int q = 0; q < p; q++) { h ^= val[q]; h *= 0x100000001B3ULL; h ^= h >> 29; }
    rs = h ? h : 1;
    for (int q = 0; q < 4; q++) rnd();
}

static int uf_find(int x) { while (uf[x] != x) { uf[x] = uf[uf[x]]; x = uf[x]; } return x; }

/* orbits of F_2^n under sampled invertible elements of the stabiliser described by S; returns 0 if trivial */
static int orbits(const Sys *S) {
    if (S->rank >= z) return 0;
    for (int x = 0; x < N; x++) uf[x] = x;
    int got = 0;
    for (int tries = 0; tries < 8 * nsamp && got < nsamp; tries++) {
        u128 c = 0;
        uint8_t isp[MAXZ] = {0};
        for (int q = 0; q < S->rank; q++) isp[S->piv[q]] = 1;
        for (int j = 0; j < z; j++) if (!isp[j] && (rnd() & 1)) c |= (u128)1 << j;
        for (int q = 0; q < S->rank; q++) {
            int v = S->rhs[q] ^ par128(S->row[q] & c & ~((u128)1 << S->piv[q]));
            if (v) c |= (u128)1 << S->piv[q];
        }
        uint16_t X[MAXN];
        for (int i = 0; i < n; i++) { X[i] = 0; for (int j = 0; j < z; j++) if ((c >> j) & 1) X[i] ^= Mz[j][i]; }
        if (rank10(X) < n) continue;
        got++;
        for (int x = 0; x < N; x++) {
            int y = 0;
            for (int i = 0; i < n; i++) if ((x >> i) & 1) y ^= X[i];
            int a = uf_find(x), b = uf_find(y);
            if (a != b) { if (a < b) uf[b] = a; else uf[a] = b; }
        }
    }
    return got > 0;
}

/* ---- input side: checkpoint pruning at the end of a level ----
 * When beta is known on V_k = <e_0..e_{k-1}>, take sampled invertible S in Z(A) = {X : XA = AX} with S V_k = V_k
 * (precomputed per k). Q = beta(S., S.) on V_k is again a partial solution of the same search. Bring Q to the greedy
 * lexicographic form under C(B) (orbits from sampled stabiliser elements, the output map built explicitly as a
 * product of generators). If the result is lexicographically smaller than the current prefix, the branch is pruned.
 * Sound: every applied map is a genuine symmetry, and the lexicographically least member of each orbit under
 * C(A) x C(B) is never pruned (by this rule or by the output chain), so each orbit keeps a representative. */
#define MAXS 4096
static int za = 0, cp_min = 0, cp_max = -1, ns_in = 24, cp_fix = 0;     /* cp_fix: bit mask of fixed e_i */
static Sys sys_forced;                               /* output maps fixing every forced value */
static uint16_t Ma[MAXZ][MAXN];
static uint16_t Sk[MAXN + 1][MAXS][MAXN];
static int nSk[MAXN + 1];
static long long cp_calls = 0, cp_pruned = 0;

static void precompute_inputs(void) {
    for (int k = 2; k <= n; k++) {
        /* homogeneous conditions on c in F_2^za: for i < k, bits r >= k of (sum c_j Ma[j][i]) vanish */
        Sys S; memset(&S, 0, sizeof S);
        for (int i = 0; i < k; i++) for (int r = k; r < n; r++) {
            u128 coef = 0;
            for (int j = 0; j < za; j++) if ((Ma[j][i] >> r) & 1) coef |= (u128)1 << j;
            if (coef) sys_add(&S, coef, 0);
        }
        /* -cpfix U / -cpmask M: the input maps are the identity on these e_i (they then keep forced U x U pairs) */
        for (int i = 0; i < n; i++) if ((cp_fix >> i) & 1) for (int r = 0; r < n; r++) {
            u128 coef = 0;
            for (int j = 0; j < za; j++) if ((Ma[j][i] >> r) & 1) coef |= (u128)1 << j;
            sys_add(&S, coef, r == i);
        }
        nSk[k] = 0;
        uint8_t isp[MAXZ] = {0};
        for (int q = 0; q < S.rank; q++) isp[S.piv[q]] = 1;
        for (int tries = 0; tries < 200 * ns_in && nSk[k] < ns_in; tries++) {
            u128 c = 0;
            for (int j = 0; j < za; j++) if (!isp[j] && (rnd() & 1)) c |= (u128)1 << j;
            for (int q = 0; q < S.rank; q++)
                if (S.rhs[q] ^ par128(S.row[q] & c & ~((u128)1 << S.piv[q]))) c |= (u128)1 << S.piv[q];
            uint16_t X[MAXN];
            for (int i = 0; i < n; i++) { X[i] = 0; for (int j = 0; j < za; j++) if ((c >> j) & 1) X[i] ^= Ma[j][i]; }
            int okfix = 1;
            for (int i = 0; i < n; i++) if (((cp_fix >> i) & 1) && X[i] != (1 << i)) okfix = 0;
            if (!okfix) continue;
            if (rank10(X) < n) continue;
            memcpy(Sk[k][nSk[k]++], X, sizeof(uint16_t) * n);
        }
        if (getenv("CPDEBUG")) {
            int dist = 0;
            for (int a = 0; a < nSk[k]; a++) { int dup = 0; for (int b = 0; b < a && !dup; b++) dup = !memcmp(Sk[k][a], Sk[k][b], sizeof(uint16_t) * n); dist += !dup; }
            fprintf(stderr, "cp k=%d: za=%d rank=%d free=%d sampled=%d distinct=%d\n", k, za, S.rank, za - S.rank, nSk[k], dist);
        }
    }
}

static inline uint16_t matvec(const uint16_t *X, uint16_t v) {
    uint16_t r = 0;
    for (int i = 0; i < n; i++) if ((v >> i) & 1) r ^= X[i];
    return r;
}

/* -orb0: orbit minimum of every value under the output maps of sys_forced (64 random generators, closure by BFS).
 * Pair 0 of canon_smaller then needs no BFS when the minimum differs from P[0]: smaller -> prune, larger -> keep.
 * Any generator is a group element, so a smaller value found this way is a genuine smaller image (sound). */
static int use_orb0 = 0, orb0_ready = 0;
static uint16_t orb0min[1 << MAXN];
static long long orb0_hits = 0;
static void orb0_build(void) {
    static uint16_t G[64][MAXN]; int ng = 0;
    Sys S = sys_forced;
    uint8_t isp[MAXZ] = {0};
    for (int q = 0; q < S.rank; q++) isp[S.piv[q]] = 1;
    for (int tries = 0; tries < 8 * 64 && ng < 64; tries++) {
        u128 c = 0;
        for (int j = 0; j < z; j++) if (!isp[j] && (rnd2() & 1)) c |= (u128)1 << j;
        for (int q = 0; q < S.rank; q++) {
            int v = S.rhs[q] ^ par128(S.row[q] & c & ~((u128)1 << S.piv[q]));
            if (v) c |= (u128)1 << S.piv[q];
        }
        uint16_t X[MAXN];
        for (int i = 0; i < n; i++) { X[i] = 0; for (int j = 0; j < z; j++) if ((c >> j) & 1) X[i] ^= Mz[j][i]; }
        if (rank10(X) < n) continue;
        memcpy(G[ng++], X, sizeof(uint16_t) * n);
    }
    static int qu[1 << MAXN]; static uint8_t vis[1 << MAXN];
    memset(vis, 0, sizeof vis);
    for (int v0 = 0; v0 < N; v0++) {
        if (vis[v0]) continue;
        int h = 0, t = 0, mn = v0; vis[v0] = 1; qu[t++] = v0;
        while (h < t) { int v = qu[h++]; for (int g = 0; g < ng; g++) { int w = matvec(G[g], (uint16_t)v); if (!vis[w]) { vis[w] = 1; qu[t++] = w; if (w < mn) mn = w; } } }
        for (int a = 0; a < t; a++) orb0min[qu[a]] = (uint16_t)mn;
    }
    orb0_ready = 1;
    if (getenv("CPDEBUG")) { int no = 0; for (int v = 0; v < N; v++) no += orb0min[v] == v; fprintf(stderr, "orb0: %d generators, %d orbits\n", ng, no); }
}

/* -cpcache: in canon_smaller, once L[0..p-1] = P[0..p-1] the output maps allowed at pair p are the stabiliser of the
 * current path prefix, i.e. sys[p] of the DFS. Its orbits on F_2^n (orbit minimum and a path of generators to it) are
 * computed once per path prefix (stamped) instead of once per input map and pair. Generators are elements of the
 * stabiliser, so every image used is genuine (sound); with NG random generators the orbits are complete w.h.p. */
#define NG 24
static int use_cache = 0;
static long long stamp_ctr = 1, pstamp[MAXP + 1], cstamp[MAXP + 1];
static uint16_t crt[MAXP + 1][1 << MAXN], cprev[MAXP + 1][1 << MAXN];
static uint8_t cgen[MAXP + 1][1 << MAXN];
static uint16_t CG[MAXP + 1][NG][MAXN];
static long long cache_builds = 0;
static void cache_build(int p) {
    Sys *S = &sys[p];
    int ng = 0;
    uint8_t isp[MAXZ] = {0};
    for (int q = 0; q < S->rank; q++) isp[S->piv[q]] = 1;
    static uint16_t Hinv[NG][MAXN], img[1 << MAXN], invt[1 << MAXN];
    for (int tries = 0; tries < 8 * NG && ng < NG && S->rank < z; tries++) {
        u128 c = 0;
        for (int j = 0; j < z; j++) if (!isp[j] && (rnd2() & 1)) c |= (u128)1 << j;
        for (int q = 0; q < S->rank; q++)
            if (S->rhs[q] ^ par128(S->row[q] & c & ~((u128)1 << S->piv[q]))) c |= (u128)1 << S->piv[q];
        uint16_t X[MAXN];
        for (int i = 0; i < n; i++) { X[i] = 0; for (int j = 0; j < z; j++) if ((c >> j) & 1) X[i] ^= Mz[j][i]; }
        if (rank10(X) < n) continue;
        memcpy(CG[p][ng], X, sizeof(uint16_t) * n);
        for (int v = 0; v < N; v++) img[v] = matvec(X, (uint16_t)v);
        for (int v = 0; v < N; v++) invt[img[v]] = (uint16_t)v;
        for (int i = 0; i < n; i++) Hinv[ng][i] = invt[1 << i];
        ng++;
    }
    static uint8_t vis[1 << MAXN]; static int qu[1 << MAXN];
    memset(vis, 0, N);
    for (int v0 = 0; v0 < N; v0++) {                  /* increasing: the first unvisited value is its orbit minimum */
        if (vis[v0]) continue;
        int h = 0, t = 0; vis[v0] = 1; qu[t++] = v0; crt[p][v0] = (uint16_t)v0;
        while (h < t) {
            int v = qu[h++];
            for (int g = 0; g < ng; g++) {
                int w = matvec(Hinv[g], (uint16_t)v);   /* CG[p][g] maps w back to v */
                if (!vis[w]) { vis[w] = 1; crt[p][w] = (uint16_t)v0; cprev[p][w] = (uint16_t)v; cgen[p][w] = (uint8_t)g; qu[t++] = w; }
            }
        }
    }
    cstamp[p] = pstamp[p];
    cache_builds++;
}
static int canon_smaller_cached(uint16_t *L, int len, const uint16_t *P) {
    for (int p = 0; p < len; p++) {
        if (cstamp[p] != pstamp[p]) cache_build(p);
        uint16_t m = crt[p][L[p]];
        if (m < P[p]) return 1;
        if (m > P[p]) return 0;
        for (int w = L[p]; w != m; ) {
            const uint16_t *X = CG[p][cgen[p][w]];
            for (int q = p; q < len; q++) L[q] = matvec(X, L[q]);
            w = cprev[p][w];
        }
    }
    return 0;
}

/* greedy form of L (len values) under C(B), compared with P: 1 if a genuinely smaller image is found */
static int canon_smaller(uint16_t *L, int len, const uint16_t *P) {
    if (use_cache) return canon_smaller_cached(L, len, P);
    if (use_orb0 && len > 0 && !forced[0] && z > 0) {
        if (!orb0_ready) orb0_build();
        uint16_t m = orb0min[L[0]];
        if (m < P[0]) { orb0_hits++; return 1; }
        if (m > P[0]) { orb0_hits++; return 0; }
    }
    static Sys S; S = sys_forced;
    static uint16_t G[64][MAXN];
    static int prevv[1 << MAXN], prevg[1 << MAXN], qu[1 << MAXN];
    static uint8_t vis[1 << MAXN];
    for (int p = 0; p < len; p++) {
        if (forced[p]) {                                /* forced pair: no minimisation; images must agree */
            if (L[p] != P[p]) return 0;
            sys_fix_vector(&S, L[p]);
            continue;
        }
        if (S.rank < z) {
            int ng = 0;
            uint8_t isp[MAXZ] = {0};
            for (int q = 0; q < S.rank; q++) isp[S.piv[q]] = 1;
            for (int tries = 0; tries < 8 * nsamp && ng < nsamp; tries++) {
                u128 c = 0;
                for (int j = 0; j < z; j++) if (!isp[j] && (rnd() & 1)) c |= (u128)1 << j;
                for (int q = 0; q < S.rank; q++) {
                    int v = S.rhs[q] ^ par128(S.row[q] & c & ~((u128)1 << S.piv[q]));
                    if (v) c |= (u128)1 << S.piv[q];
                }
                uint16_t X[MAXN];
                for (int i = 0; i < n; i++) { X[i] = 0; for (int j = 0; j < z; j++) if ((c >> j) & 1) X[i] ^= Mz[j][i]; }
                if (rank10(X) < n) continue;
                memcpy(G[ng++], X, sizeof(uint16_t) * n);
            }
            if (ng) {
                /* BFS orbit of L[p] under the generators, keep the path to the minimum */
                memset(vis, 0, N);
                int h = 0, t = 0, start = L[p], mn = start;
                vis[start] = 1; prevv[start] = -1; qu[t++] = start;
                while (h < t) {
                    int v = qu[h++];
                    for (int g = 0; g < ng; g++) {
                        int w = matvec(G[g], (uint16_t)v);
                        if (!vis[w]) { vis[w] = 1; prevv[w] = v; prevg[w] = g; qu[t++] = w; if (w < mn) mn = w; }
                    }
                }
                if (mn < start) {
                    int path[1 << MAXN], np = 0;
                    for (int v = mn; v != start; v = prevv[v]) path[np++] = prevg[v];
                    for (int q = p; q < len; q++) {
                        uint16_t v = L[q];
                        for (int e = np - 1; e >= 0; e--) v = matvec(G[path[e]], v);
                        L[q] = v;
                    }
                }
            }
        }
        if (L[p] < P[p]) return 1;
        if (L[p] > P[p]) return 0;
        sys_fix_vector(&S, L[p]);
    }
    return 0;
}

/* per-k statistics (CPDEBUG): calls, prunes, time, maps tried */
#include <mach/mach_time.h>
static long long cpk_calls[MAXN + 1], cpk_pr[MAXN + 1], cpk_maps[MAXN + 1]; static uint64_t cpk_t[MAXN + 1];
static int checkpoint_inner(int k);
static int use_mtf = 0;
static int late_k = 99, late_ns = 1 << 30;           /* -late K M: at checkpoints k >= K use only the first M maps */
static int checkpoint(int k) {
    if (k < cp_min || k > cp_max || !za || !nSk[k]) return 0;
    uint64_t t0 = mach_absolute_time();
    int r = checkpoint_inner(k);
    cpk_t[k] += mach_absolute_time() - t0; cpk_calls[k]++; cpk_pr[k] += r;
    return r;
}
/* checkpoint after level k - 1 is complete (beta known on V_k): 1 = prune */
static int checkpoint_inner(int k) {
    cp_calls++;
    reseed(k * (k - 1) / 2);
    int Pk = k * (k - 1) / 2;
    static uint16_t L[MAXP];
    int nq = nSk[k]; if (k >= late_k && late_ns < nq) nq = late_ns;
    for (int q = 0; q < nq; q++) {
        const uint16_t *X = Sk[k][q];
        for (int p = 0; p < Pk; p++) {
            uint16_t x = X[pairs_i[p]], y = X[pairs_j[p]], b = 0;
            for (int j = 0; j < k; j++) if ((y >> j) & 1) b ^= hb[x][j];
            L[p] = b;
        }
        cpk_maps[k]++;
        if (canon_smaller(L, Pk, val)) {
            cp_pruned++;
            if (use_mtf && q > 0) {                     /* -mtf: move the successful map to the front (same set of maps) */
                uint16_t T[MAXN]; memcpy(T, Sk[k][q], sizeof T);
                memmove(Sk[k][1], Sk[k][0], sizeof(Sk[k][0]) * q);
                memcpy(Sk[k][0], T, sizeof T);
            }
            return 1;
        }
    }
    return 0;
}

static inline uint16_t mv(const uint16_t *X, uint16_t v) { uint16_t r = 0; for (int i = 0; i < n; i++) if ((v >> i) & 1) r ^= X[i]; return r; }
static int scale_ok(int s, int i) {                      /* 0 = prune (call with val[pair (i, s)] set) */
    for (int q = 0; q < nk[s]; q++) {
        int k = sk_k[s][q], m = sc_m[s][q][i];
        if (!m) continue;
        uint16_t T[MAXN], U[MAXN];
        for (int c = 0; c <= i; c++) T[c] = val[pidx[c][s]];
        for (int c = 0; c < m; c++) {
            uint16_t x = AinvK[k][c], t = 0;
            for (int c2 = 0; c2 <= i; c2++) if ((x >> c2) & 1) t ^= T[c2];
            U[c] = mv(BK[k], t);
        }
        for (int r = 0; r < trnb[s]; r++) {
            int c = trpc[s][r]; if (c >= m) continue;
            uint16_t bit = trmask[s][c] & trB[s][r][c] & (uint16_t)(1u << (31 - __builtin_clz(trB[s][r][c])));
            if (U[c] & bit) for (int c2 = c; c2 < m; c2++) U[c2] ^= trB[s][r][c2];
        }
        for (int c = 0; c < m; c++) { if (U[c] < T[c]) { sc_pruned++; return 0; } if (U[c] > T[c]) break; }
    }
    return 1;
}

static void dfs(int p, int depth);

/* -gbasis F C0 C1 ..: canonical basis of the intertwiner space under GL(<e_F .. e_{n-1}>). The intertwiner
 * T_s = beta(e_s, .) (s >= F) is identified with the tuple (beta(e_C0, e_s), beta(e_C1, e_s), ..) (it is determined
 * by it, e.g. C = 0 2 4 for F_4-linear T on F_4^3), compared lexicographically. Kept only if T_F < T_F+1 < .. and
 * each T_s is the least element of T_s + span(T_F .. T_{s-1}): the greedy (lexicographically least) basis of the
 * span, one representative per GL orbit of injective T. In the search order this is exactly "lexicographically
 * least in its GL(<e_F..>) orbit" (T_F decides first, then T_F+1, ..), so like -symdepth and -cp it keeps the least
 * member of every orbit: sound together with them and with -force on U x U pairs (U = indices < F). */
static int gb_F = -1, gb_nc = 0, gb_c[MAXN];
static long long gb_pruned = 0;
static int gb_ok(int p, uint16_t y) {
    int s = pairs_j[p], i = pairs_i[p];
    if (gb_F < 0 || s < gb_F) return 1;
    int cc = -1;                                   /* column index of this pair, if any */
    for (int c = 0; c < gb_nc; c++) if (gb_c[c] == i) cc = c;
    if (cc < 0) return 1;
    int last = (cc == gb_nc - 1);                  /* all components known: ties also decide */
    uint16_t t[MAXN][MAXN];                        /* t[k][c] = component c of T_{F+k}, c <= cc */
    int m = s - gb_F + 1;
    for (int k = 0; k < m; k++) for (int c = 0; c <= cc; c++)
        t[k][c] = (k == m - 1 && c == cc) ? y : val[pidx[gb_c[c]][gb_F + k]];
    if (m >= 2) {                                  /* T_{s-1} < T_s on the known prefix */
        int c = 0;
        while (c <= cc && t[m - 2][c] == t[m - 1][c]) c++;
        if ((c <= cc && t[m - 2][c] > t[m - 1][c]) || (c > cc && last)) { gb_pruned++; return 0; }
    }
    for (int v = 1; v < (1 << (m - 1)); v++) {    /* T_s <= T_s + v on the known prefix */
        int c = 0;
        for (; c <= cc; c++) {
            uint16_t w = t[m - 1][c];
            for (int k = 0; k < m - 1; k++) if ((v >> k) & 1) w ^= t[k][c];
            if (t[m - 1][c] < w) break;
            if (t[m - 1][c] > w) { gb_pruned++; return 0; }
        }
    }
    return 1;
}

static void after_pair(int p, int depth) {
    int s = pairs_j[p], i = pairs_i[p];
    if (i + 1 < s) { dfs(p + 1, depth); return; }
    level_end(s);                                   /* level s complete */
    if (checkpoint(s + 1)) return;
    if (s + 1 == n) {
        leaves++;
        if (dedupe) {
            uint64_t h = leaf_key(), slot = h & (HSIZE - 1);
            while (seen_keys[slot] && seen_keys[slot] != h) slot = (slot + 1) & (HSIZE - 1);
            if (!seen_keys[slot]) { seen_keys[slot] = h; distinct++; printf("K%016llx ", (unsigned long long)h); print_leaf(); fflush(stdout); }
        } else if (printed < print_max) { print_leaf(); printed++; fflush(stdout); }
        return;
    }
    level_start(s + 1);
    dfs(p + 1, depth);
}

/* progress: for each level of the current path, the index of the child being explored and the number of children;
 * the explored fraction is the mixed-radix number sum_d idx_d / cnt_d * prod_{e<d} 1 / cnt_e (assumes subtrees of
 * similar size: an estimate). Written every 60 s to the -progress file: elapsed s, nodes, fraction, estimated s left. */
static int pidx_path[MAXP + 1], pcnt_path[MAXP + 1];
static const char *progress_file = 0;
static time_t t_start, t_last;
static void progress_write(int final) {
    double f = 0, mult = 1;
    for (int d = 0; d <= MAXP && pcnt_path[d] > 0; d++) { f += mult * pidx_path[d] / pcnt_path[d]; mult /= pcnt_path[d]; }
    if (final) f = 1;
    time_t now = time(0);
    double el = difftime(now, t_start), left = f > 0 ? el * (1 - f) / f : -1;
    FILE *fp = fopen(progress_file, "a");
    if (fp) {
        fprintf(fp, "elapsed_s %.0f nodes %lld leaves %lld fraction %.6f est_left_s %.0f%s path", el, nodes, leaves, f, left,
                final ? " DONE" : "");
        for (int d = 0; d < 6 && pcnt_path[d] > 0; d++) fprintf(fp, " %d/%d", pidx_path[d], pcnt_path[d]);
        fprintf(fp, "\n");
        fclose(fp);
    }
    t_last = now;
}

static int count_depth = -1; static long long count_nodes = 0;
static void dfs(int p, int depth) {
    if (count_depth >= 0 && depth == count_depth) { count_nodes++; return; }
    if (node_limit >= 0 && nodes >= node_limit) return;
    nodes++;
    if (progress_file && (nodes & 0x3FFF) == 0 && difftime(time(0), t_last) >= 60) progress_write(0);
    int s = pairs_j[p], i = pairs_i[p], c = kcount[p], k0 = kfirst[p];
    if (p < symdepth) reseed(p);
    int use_sym = (p < symdepth) && orbits(&sys[p]);
    /* pass 1: the valid choices of the new bits, in Gray-code order (all filtering happens here, before recursing,
     * so the orbit table is not overwritten by deeper levels while in use) */
    static long long codes_buf[MAXP][1 << MAXN];
    long long *codes = codes_buf[p];
    int nv = 0;
    long long cur = 0;
    for (long long q = 0; q < (1LL << c); q++) {
        if (q) { int b = __builtin_ctzll(q); apply_t(k0 + b); cur ^= 1LL << b; }
        uint16_t y = val[p];
        if (forced[p]) { if (y != forced[p] - 1) continue; }
        else {
            if (p == 0 && !allowed01[y]) continue;     /* both filters keep the orbit minimum: sound together */
            if (use_sym && uf_find(y) != y) { sym_skipped++; continue; }
        }
        if (y & trmask[s][i]) { tr_skipped++; continue; }
        if (!gb_ok(p, y) || !check(s, i, y)) continue;
        if (nk[s] && !scale_ok(s, i)) continue;          /* after check: most values fail check */
        if (use_ahead && em[p] > emprev[p]) {
            uint16_t nx[MAXN]; for (int k = 0; k < em[p]; k++) nx[k] = val[pidx[k][s + 1]];
            if (!early_ok(s, i, y, nx)) continue;
        }
        if (depth + 1 == pshard_depth && npshards > 1 && pshard_count++ % npshards != pshard) continue;
        if (depth + 1 == shard_depth && nshards > 1 && shard_count++ % nshards != shard) continue;
        { int skip = 0; for (int l = 0; l < nsh; l++) if (depth + 1 == sh_d[l] && sh_cnt[l]++ % sh_k[l] != sh_i[l]) { skip = 1; break; } if (skip) continue; }
        codes[nv++] = cur;
    }
    /* pass 2: explore them in the same order */
    pcnt_path[p] = nv;
    for (int k = 0; k < nv; k++) {
        if (node_limit >= 0 && nodes >= node_limit) break;
        long long diff = cur ^ codes[k];
        for (int b = 0; b < c; b++) if ((diff >> b) & 1) apply_t(k0 + b);
        cur = codes[k];
        pidx_path[p] = k;
        uint16_t y = val[p];
        if (p + 1 <= symdepth) { sys[p + 1] = sys[p]; sys_fix_vector(&sys[p + 1], y); pstamp[p + 1] = ++stamp_ctr; }
        set_y(s, i, y);
        after_pair(p, depth + 1);
    }
    for (int b = 0; b < c; b++) if ((cur >> b) & 1) apply_t(k0 + b);    /* back to code 0 */
    pcnt_path[p] = 0;
}

int main(int argc, char **argv) {
    long long probes = 0;
    int mode_dfs = 0;
    for (int a = 1; a < argc; a++) {
        if (!strcmp(argv[a], "-knuth")) probes = atoll(argv[++a]);
        else if (!strcmp(argv[a], "-dfs")) mode_dfs = 1;
        else if (!strcmp(argv[a], "-print")) print_max = atoi(argv[++a]);
        else if (!strcmp(argv[a], "-limit")) node_limit = atoll(argv[++a]);
        else if (!strcmp(argv[a], "-shard")) { shard = atoi(argv[++a]); nshards = atoi(argv[++a]); }
        else if (!strcmp(argv[a], "-orb0")) use_orb0 = 1;
        else if (!strcmp(argv[a], "-cpcache")) use_cache = 1;
        else if (!strcmp(argv[a], "-mtf")) use_mtf = 1;
        else if (!strcmp(argv[a], "-tr")) use_tr = 1;
        else if (!strcmp(argv[a], "-ahead")) use_ahead = 1;
        else if (!strcmp(argv[a], "-scale")) {
            use_scale = 1; char *s1 = argv[++a], *s2 = argv[++a];
            for (int i = 0; i < MAXN; i++) { Acol[i] = (int)strtol(s1, &s1, 10); if (*s1) s1++; Bcol[i] = (int)strtol(s2, &s2, 10); if (*s2) s2++; }
        }
        else if (!strcmp(argv[a], "-sh") && nsh < 8) { sh_i[nsh] = atoi(argv[++a]); sh_k[nsh] = atoi(argv[++a]); sh_d[nsh] = atoi(argv[++a]); nsh++; }
        else if (!strcmp(argv[a], "-late")) { late_k = atoi(argv[++a]); late_ns = atoi(argv[++a]); }
        else if (!strcmp(argv[a], "-countdepth")) count_depth = atoi(argv[++a]);
        else if (!strcmp(argv[a], "-sdepth")) shard_depth = atoi(argv[++a]);
        else if (!strcmp(argv[a], "-pshard")) { pshard = atoi(argv[++a]); npshards = atoi(argv[++a]); }
        else if (!strcmp(argv[a], "-pdepth")) pshard_depth = atoi(argv[++a]);
        else if (!strcmp(argv[a], "-nonorm")) nonorm = 1;
        else if (!strcmp(argv[a], "-dedupe")) dedupe = 1;
        else if (!strcmp(argv[a], "-symdepth")) symdepth = atoi(argv[++a]);
        else if (!strcmp(argv[a], "-nsamp")) nsamp = atoi(argv[++a]);
        else if (!strcmp(argv[a], "-cp")) { cp_min = atoi(argv[++a]); cp_max = atoi(argv[++a]); }
        else if (!strcmp(argv[a], "-nsin")) { ns_in = atoi(argv[++a]); if (ns_in > MAXS) ns_in = MAXS; }
        else if (!strcmp(argv[a], "-cpfix")) cp_fix = (1 << atoi(argv[++a])) - 1;
        else if (!strcmp(argv[a], "-cpmask")) cp_fix = atoi(argv[++a]);
        else if (!strcmp(argv[a], "-gbasis")) {
            gb_F = atoi(argv[++a]);
            while (a + 1 < argc && argv[a + 1][0] != '-') gb_c[gb_nc++] = atoi(argv[++a]);
        }
        else if (!strcmp(argv[a], "-progress")) progress_file = argv[++a];
        else if (!strcmp(argv[a], "-offset")) {
            FILE *ff = fopen(argv[++a], "r"); int pp, vv;
            while (ff && fscanf(ff, "%d %d", &pp, &vv) == 2) if (pp >= 0 && pp < MAXP) offset[pp] = (uint16_t)vv;
            if (ff) fclose(ff);
        }
        else if (!strcmp(argv[a], "-force")) {
            FILE *ff = fopen(argv[++a], "r"); int pp, vv;
            while (ff && fscanf(ff, "%d %d", &pp, &vv) == 2) if (pp >= 0 && pp < MAXP) forced[pp] = vv + 1;
            if (ff) fclose(ff);
        }
        else if (!strcmp(argv[a], "-seed")) rs ^= (uint64_t)atoll(argv[++a]) * 0x9E3779B97F4A7C15ULL;
    }
    for (int q = 0; q < 20; q++) rnd();
    rs2 ^= rs; for (int q = 0; q < 20; q++) rnd2();
    if (scanf("%d %d", &n, &d) != 2) return 1;
    N = 1 << n; P = 0;
    for (int j = 0; j < n; j++) for (int i = 0; i < j; i++) { pairs_i[P] = i; pairs_j[P] = j; pidx[i][j] = P++; }
    for (int p = 0; p < P; p++) { kfirst[p] = -1; kcount[p] = 0; }
    for (int k = 0; k < d; k++) {
        int pp, pb;
        if (scanf("%d %d", &pp, &pb) != 2) return 1;
        pivpair[k] = pp;
        for (int p = 0; p < P; p++) { int x; if (scanf("%d", &x) != 1) return 1; wv[k][p] = (uint16_t)x; }
        if (kfirst[pp] < 0) kfirst[pp] = k;
        kcount[pp]++;
    }
    for (int p = 0; p < P; p++) if (kfirst[p] < 0) kfirst[p] = 0;
    int nrep = 0;
    if (scanf("%d", &nrep) == 1 && !nonorm) {
        for (int q = 0; q < nrep; q++) { int x; if (scanf("%d", &x) == 1) allowed01[x] = 1; }
    } else for (int x = 0; x < N; x++) allowed01[x] = 1;
    if (scanf("%d", &z) == 1 && z > 0 && z <= MAXZ) {
        for (int j = 0; j < z; j++) for (int i = 0; i < n; i++) { int x; if (scanf("%d", &x) != 1) return 1; Mz[j][i] = (uint16_t)x; }
    } else z = 0;
    if (!z) symdepth = 0;
    if (scanf("%d", &za) == 1 && za > 0 && za <= MAXZ) {
        for (int j = 0; j < za; j++) for (int i = 0; i < n; i++) { int x; if (scanf("%d", &x) != 1) return 1; Ma[j][i] = (uint16_t)x; }
    } else za = 0;
    if (za && cp_max >= cp_min) precompute_inputs();
    if (use_tr && za) for (int s = 1; s < n; s++) {      /* U_s: linear conditions on c in F_2^za */
        Sys S; memset(&S, 0, sizeof S);
        for (int j = 0; j < n; j++) for (int r = 0; r < n; r++) {
            if (j >= s && r < s) continue;               /* X e_j - e_j in V_s for j >= s; X e_j = e_j for j < s */
            u128 coef = 0;
            for (int m = 0; m < za; m++) if ((Ma[m][j] >> r) & 1) coef |= (u128)1 << m;
            sys_add(&S, coef, r == j);
        }
        /* solutions c = c0 + kernel; u(c) = X e_s + e_s. Collect the u of a particular solution and kernel vectors */
        uint8_t isp[MAXZ] = {0}; for (int q = 0; q < S.rank; q++) isp[S.piv[q]] = 1;
        uint16_t uu[MAXZ + 1]; int nuu = 0;
        for (int f = -1; f < za; f++) {
            if (f >= 0 && isp[f]) continue;
            u128 c = f >= 0 ? (u128)1 << f : 0; int hom = f >= 0;
            for (int q = 0; q < S.rank; q++)
                if ((hom ? 0 : S.rhs[q]) ^ par128(S.row[q] & c & ~((u128)1 << S.piv[q]))) c |= (u128)1 << S.piv[q];
            uint16_t col = 0; for (int m = 0; m < za; m++) if ((c >> m) & 1) col ^= Ma[m][s];
            if (!hom) continue;                             /* X = I is a solution; only the kernel matters */
            uu[nuu++] = col;                                /* kernel element: N e_s, lies in V_s */
        }
        nU[s] = 0;                                          /* basis of span(uu) */
        uint16_t R[MAXN]; int nr = 0;
        for (int q = 0; q < nuu; q++) {
            uint16_t v = uu[q];
            for (int r = 0; r < nr; r++) if ((v >> (31 - __builtin_clz(R[r]))) & 1) v ^= R[r];
            if (!v) continue;
            R[nr++] = v; Ub[s][nU[s]++] = uu[q];
        }
        if (getenv("CPDEBUG")) fprintf(stderr, "tr s=%d: dim U = %d\n", s, nU[s]);
    }
    if (use_scale && za) {
        uint16_t M[MAXN], I[MAXN]; for (int i = 0; i < n; i++) { M[i] = (uint16_t)Acol[i]; I[i] = (uint16_t)(1 << i); }
        uint16_t Pk[MAXN]; memcpy(Pk, M, sizeof Pk); ordA = 1;
        while (memcmp(Pk, I, sizeof(uint16_t) * n) && ordA < 16) { uint16_t T[MAXN]; for (int i = 0; i < n; i++) T[i] = mv(M, Pk[i]); memcpy(Pk, T, sizeof T); ordA++; }
        /* A^-k = A^(ord-k); B^k */
        for (int k = 0; k < ordA; k++) {
            uint16_t Ap[MAXN], Bp[MAXN]; memcpy(Ap, I, sizeof Ap); memcpy(Bp, I, sizeof Bp);
            for (int e = 0; e < (ordA - k) % ordA; e++) for (int i = 0; i < n; i++) Ap[i] = mv(M, Ap[i]);
            uint16_t BB[MAXN]; for (int i = 0; i < n; i++) BB[i] = (uint16_t)Bcol[i];
            for (int e = 0; e < k; e++) for (int i = 0; i < n; i++) Bp[i] = mv(BB, Bp[i]);
            memcpy(AinvK[k], Ap, sizeof Ap); memcpy(BK[k], Bp, sizeof Bp);
        }
        for (int s = 1; s < n; s++) {
            nk[s] = 0;
            int inv = 1; for (int j = 0; j < s; j++) if (Acol[j] >> s) inv = 0;  /* A V_s = V_s */
            if (!inv) continue;
            for (int k = 1; k < ordA; k++) {
                uint16_t Ak = I[s]; for (int e = 0; e < k; e++) Ak = mv(M, Ak);
                Sys S; memset(&S, 0, sizeof S); int bad = 0;
                for (int j = 0; j <= s && !bad; j++) for (int r = 0; r < n; r++) {
                    u128 coef = 0; for (int m = 0; m < za; m++) if ((Ma[m][j] >> r) & 1) coef |= (u128)1 << m;
                    int rhs = j < s ? (r == j) : ((Ak >> r) & 1);
                    /* inconsistency check: reduce */
                    u128 cc = coef; int rr = rhs;
                    for (int q = 0; q < S.rank; q++) if ((cc >> S.piv[q]) & 1) { cc ^= S.row[q]; rr ^= S.rhs[q]; }
                    if (!cc && rr) { bad = 1; break; }
                    sys_add(&S, coef, rhs);
                }
                if (bad) continue;
                uint8_t isp[MAXZ] = {0}; for (int q = 0; q < S.rank; q++) isp[S.piv[q]] = 1;
                int found = 0;
                for (int tries = 0; tries < 2000 && !found; tries++) {
                    u128 c = 0;
                    for (int j = 0; j < za; j++) if (!isp[j] && (rnd() & 1)) c |= (u128)1 << j;
                    for (int q = 0; q < S.rank; q++)
                        if (S.rhs[q] ^ par128(S.row[q] & c & ~((u128)1 << S.piv[q]))) c |= (u128)1 << S.piv[q];
                    uint16_t X[MAXN]; for (int i = 0; i < n; i++) { X[i] = 0; for (int j = 0; j < za; j++) if ((c >> j) & 1) X[i] ^= Ma[j][i]; }
                    int ok = rank10(X) == n && X[s] == Ak;
                    for (int j = 0; j < s; j++) if (X[j] != (1 << j)) ok = 0;
                    if (ok) found = 1;
                }
                if (!found) continue;
                int q = nk[s]++; sk_k[s][q] = k;
                for (int i = 0; i < s; i++) {               /* prefix length known after pair (i, s) */
                    int m = 0; while (m < s && !(AinvK[k][m] >> (i + 1))) m++;
                    sc_m[s][q][i] = m;
                }
            }
            if (getenv("CPDEBUG")) fprintf(stderr, "scale s=%d: %d powers\n", s, nk[s]);
        }
    }
    memset(&sys_forced, 0, sizeof(Sys));
    if (z) for (int p = 0; p < P; p++) if (forced[p]) sys_fix_vector(&sys_forced, (uint16_t)(forced[p] - 1));
    sys[0] = sys_forced;
    for (int p = 0; p <= MAXP; p++) { cstamp[p] = -1; pstamp[p] = 0; }
    if (use_cache) { int need = cp_max * (cp_max - 1) / 2 - 1, anyf = 0; for (int p = 0; p < P; p++) anyf |= forced[p] != 0;
        if (anyf || !z || symdepth < need) { fprintf(stderr, "-cpcache needs -symdepth >= %d, z > 0 and no -force: disabled\n", need); use_cache = 0; } }
    fprintf(stderr, "sym10: n=%d, d=%d free bits; per level:", n, d);
    for (int s = 1; s < n; s++) { int c = 0; for (int p = 0; p < P; p++) if (pairs_j[p] == s) c += kcount[p]; fprintf(stderr, " %d", c); }
    fprintf(stderr, "\n");
    for (int p = 0; p < P; p++) {                         /* -ahead tables */
        int s = pairs_j[p], i = pairs_i[p]; em[p] = 0;
        if (s + 1 >= n) continue;
        for (int m = 0; m <= i; m++) {
            int q = pidx[m][s + 1], det = !forced[q];
            for (int k = 0; k < d && det; k++) if (pivpair[k] > p && pivpair[k] <= q && wv[k][q]) det = 0;
            if (!det) break;
            em[p] = m + 1;
        }
    }
    for (int p = 0; p < P; p++) { int s = pairs_j[p], i = pairs_i[p]; emprev[p] = (i > 0) ? em[pidx[i - 1][s]] : 0; }
    if (use_ahead) { fprintf(stderr, "ahead:"); for (int p = 0; p < P; p++) if (em[p] > emprev[p]) fprintf(stderr, " (%d,%d)->%d", pairs_i[p], pairs_j[p], em[p]); fprintf(stderr, "\n"); }
    memset(hb, 0, sizeof hb);
    memcpy(val, offset, sizeof val);

    if (probes > 0) {
        double est[MAXP + 1] = {0}, leafest = 0;
        for (long long pr = 0; pr < probes; pr++) {
            memcpy(val, offset, sizeof val); memset(hb, 0, sizeof hb);
            level_start(1);
            double w = 1;
            int p = 0, alive = 1;
            while (alive) {
                int s = pairs_j[p], i = pairs_i[p], c = kcount[p], k0 = kfirst[p];
                /* collect the valid choices */
                static uint32_t ok[1 << 12]; int nok = 0;
                uint16_t base = val[p];
                int use_sym = (p < symdepth) && orbits(&sys[p]);
                for (long long q = 0; q < (1LL << c); q++) {
                    uint16_t y = base;
                    for (int b = 0; b < c; b++) if ((q >> b) & 1) y ^= wv[k0 + b][p];
                    if (forced[p]) { if (y != forced[p] - 1) continue; }
                    else {
                        if (p == 0 && !allowed01[y]) continue;     /* both filters keep the orbit minimum: sound together */
                        if (use_sym && uf_find(y) != y) continue;
                    }
                    if (!(y & trmask[s][i]) && gb_ok(p, y) && check(s, i, y)) {
                        if (use_ahead && em[p] > emprev[p]) {
                            uint16_t nx[MAXN];
                            for (int k = 0; k < em[p]; k++) { int qq = pidx[k][s + 1]; nx[k] = val[qq];
                                for (int b = 0; b < c; b++) if ((q >> b) & 1) nx[k] ^= wv[k0 + b][qq]; }
                            if (!early_ok(s, i, y, nx)) continue;
                        }
                        ok[nok++] = (uint32_t)q;
                    }
                }
                if (!nok) break;
                w *= nok;
                uint32_t q = ok[rnd2() % nok];
                for (int b = 0; b < c; b++) if ((q >> b) & 1) apply_t(k0 + b);
                if (p + 1 <= symdepth) { sys[p + 1] = sys[p]; sys_fix_vector(&sys[p + 1], val[p]); pstamp[p + 1] = ++stamp_ctr; }
                set_y(s, i, val[p]);
                est[p + 1] += w;
                if (i + 1 == s) {
                    level_end(s);
                    if (checkpoint(s + 1)) break;
                    if (s + 1 == n) { leafest += w; break; }
                    level_start(s + 1);
                }
                p++;
            }
        }
        double total = 1;
        fprintf(stderr, "Knuth estimate (%lld probes):\n", probes);
        for (int p = 1; p <= P; p++) {
            total += est[p] / probes;
            fprintf(stderr, "  pair %2d (%d,%d): %.3g\n", p, pairs_i[p - 1], pairs_j[p - 1], est[p] / probes);
        }
        fprintf(stderr, "total nodes %.3g, leaves (APN functions in W) %.3g; checkpoints %lld calls, %lld pruned\n", total, leafest / probes, cp_calls, cp_pruned);
        return 0;
    }
    if (mode_dfs) {
        level_start(1);
        t_start = t_last = time(0);
        dfs(0, 0);
        if (progress_file) progress_write(1);
        if (gb_F >= 0) fprintf(stderr, "greedy basis: %lld pruned\n", gb_pruned);
        if (use_tr) fprintf(stderr, "translations: skipped %lld values\n", tr_skipped);
        if (use_ahead) fprintf(stderr, "ahead: pruned %lld values\n", ahead_pruned);
        if (use_scale) fprintf(stderr, "scaling: pruned %lld values\n", sc_pruned);
        fprintf(stderr, "symmetry: skipped %lld values (symdepth %d, z = %d); checkpoints %lld calls, %lld pruned (za = %d)\n",
                sym_skipped, symdepth, z, cp_calls, cp_pruned, za);
        fprintf(stderr, "shard counters: depth %d: %lld, depth %d: %lld\n", pshard_depth, pshard_count, shard_depth, shard_count);
        if (count_depth >= 0) fprintf(stderr, "count: depth %d: %lld nodes (time %.0f s)\n", count_depth, count_nodes, difftime(time(0), t_start));
        { mach_timebase_info_data_t tb; mach_timebase_info(&tb); double tot = 0;
          for (int k = 0; k <= n; k++) if (cpk_calls[k]) { double sec = (double)cpk_t[k] * tb.numer / tb.denom / 1e9; tot += sec;
            fprintf(stderr, "cp k=%d: calls %lld pruned %lld maps %lld time %.1f s\n", k, cpk_calls[k], cpk_pr[k], cpk_maps[k], sec); }
          fprintf(stderr, "cp total time %.1f s, wall %.0f s\n", tot, difftime(time(0), t_start)); }
        if (use_cache) fprintf(stderr, "cpcache: %lld orbit tables built\n", cache_builds);
        if (use_orb0) fprintf(stderr, "orb0: %lld decided by the table\n", orb0_hits);
        fprintf(stderr, "done: nodes %lld, leaves %lld, distinct fingerprints %lld%s\n", nodes, leaves, distinct,
                (node_limit >= 0 && nodes >= node_limit) ? " (node limit reached)" : "");
    }
    return 0;
}
