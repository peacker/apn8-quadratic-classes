// mass8g.cu: the uniform sampler of mass8.c (-mass mode) on an NVIDIA GPU.
//
// Same draws as mass8 -mass S -tries T: S starts, each a uniformly random APN bilinear beta6 (built variable by
// variable, restart from scratch on any rejection), then T uniformly random L6; every valid L6 gives a G7, and
// the exhaustive L7 search decides whether G7 has solutions. Everything up to "G7 has k leaves" runs on the GPU:
//   k_gen      one thread per start: chain, then the T tries; valid G7 are appended to a buffer;
//   k_init     Q0[U] = complement of Im7[U] (one block of 128 threads per G7);
//   k_expand   depth 0 -> 1 and 1 -> 2, one warp per node (same rules as bt() in mass8.c: Lemma 4.5 coset
//              representatives at depth 0, the order-2 pairing y <= y ^ B(e_0, x) at depth 1);
//   k_rec, k_item, k_tail   depth 3 onwards as in bt3(), in three flat stages (see below).
// A G7 with leaves is searched again by the CPU code of mass8.c (cpu.c), which prints its outputs exactly as
// mass8 does; 2 * leaves must equal the CPU outputs (checked, reported as "mismatch").
// The branching order is that of mass8.c (fewest candidates, ties to the smallest U), so node counts agree too.
//
// Random numbers: splitmix64; the state of start i of a call is a hash of (seed, i), so starts are independent
// of the GPU schedule and a call is reproducible. Each 64-bit output gives up to 8 uniform bytes.
//
// stderr, at the end of a call (same fields as mass8): starts chain_attempts tries valid outputs nonstd splitmix2g
//
// usage: mass8g -mass S -seed SEED [-tries 4096] [-cap 64] [-dev D]
//        mass8g -check FILE [-dev D]     FILE: 132-byte records (G7, int32 CPU outputs) from mass8 -dump
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>

extern "C" long cpu_g7(const unsigned char *g, long cap);   // cpu.c: mass8.c's search, prints the outputs

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "CUDA %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(2); } } while (0)

struct S8 { uint32_t w[8]; };   // a subset of F_2^8: bit y is bit y & 31 of word y >> 5 (= mass8.c's set256)

__device__ __forceinline__ bool any8(const S8 &a) {
    return ((a.w[0] | a.w[1]) | (a.w[2] | a.w[3]) | (a.w[4] | a.w[5]) | (a.w[6] | a.w[7])) != 0;
}
__device__ __forceinline__ int pop8(const S8 &a) {
    int c = 0;
#pragma unroll
    for (int i = 0; i < 8; i++) c += __popc(a.w[i]);
    return c;
}
__device__ __forceinline__ S8 and8(const S8 &a, const S8 &b) {
    S8 r;
#pragma unroll
    for (int i = 0; i < 8; i++) r.w[i] = a.w[i] & b.w[i];
    return r;
}
// bit i of x -> bit i ^ m inside each byte (m < 8)
__device__ __forceinline__ uint32_t bitx(uint32_t x, int m) {
    uint32_t t;
    t = ((x & 0x55555555u) << 1) | ((x >> 1) & 0x55555555u); x = (m & 1) ? t : x;
    t = ((x & 0x33333333u) << 2) | ((x >> 2) & 0x33333333u); x = (m & 2) ? t : x;
    t = ((x & 0x0F0F0F0Fu) << 4) | ((x >> 4) & 0x0F0F0F0Fu); x = (m & 4) ? t : x;
    return x;
}
// {y ^ v : y in x}
__device__ __forceinline__ S8 xl(const S8 &x, int v) {
    int h = v >> 5;
    uint32_t a[8], b[8];
#pragma unroll
    for (int i = 0; i < 8; i++) a[i] = x.w[i];
#pragma unroll
    for (int i = 0; i < 8; i++) b[i] = (h & 1) ? a[i ^ 1] : a[i];
#pragma unroll
    for (int i = 0; i < 8; i++) a[i] = (h & 2) ? b[i ^ 2] : b[i];
#pragma unroll
    for (int i = 0; i < 8; i++) b[i] = (h & 4) ? a[i ^ 4] : a[i];
    uint32_t sel = 0x3210u ^ (uint32_t)(((v >> 3) & 3) * 0x1111);
    int m = v & 7;
    S8 r;
#pragma unroll
    for (int i = 0; i < 8; i++) r.w[i] = bitx(__byte_perm(b[i], 0, sel), m);
    return r;
}
// bit v of s, without dynamic register indexing
__device__ __forceinline__ uint32_t has8(const S8 &s, int v) {
    int w = v >> 5;
    uint32_t x01 = (w & 1) ? s.w[1] : s.w[0], x23 = (w & 1) ? s.w[3] : s.w[2];
    uint32_t x45 = (w & 1) ? s.w[5] : s.w[4], x67 = (w & 1) ? s.w[7] : s.w[6];
    uint32_t lo = (w & 2) ? x23 : x01, hi = (w & 2) ? x67 : x45;
    return (((w & 4) ? hi : lo) >> (v & 31)) & 1;
}
__device__ __forceinline__ S8 ld8(const uint32_t *p) {
    S8 r;
    const uint4 *q = (const uint4 *)p;
    uint4 u = __ldg(q), v = __ldg(q + 1);
    r.w[0] = u.x; r.w[1] = u.y; r.w[2] = u.z; r.w[3] = u.w; r.w[4] = v.x; r.w[5] = v.y; r.w[6] = v.z; r.w[7] = v.w;
    return r;
}
__device__ __forceinline__ void st8(uint32_t *p, const S8 &a) {
    uint4 *q = (uint4 *)p;
    q[0] = make_uint4(a.w[0], a.w[1], a.w[2], a.w[3]); q[1] = make_uint4(a.w[4], a.w[5], a.w[6], a.w[7]);
}
// compact index of U (subset of R) and back: the bits of R in increasing order are numbered 0, 1, ...
__device__ __forceinline__ int pext7(int U, int R) {
    int r = 0, k = 0;
    for (int b = R; b; b &= b - 1, k++) if (U & b & -b) r |= 1 << k;
    return r;
}
__device__ __forceinline__ int pdep7(int i, int R) {
    int r = 0, k = 0;
    for (int b = R; b; b &= b - 1, k++) if (i >> k & 1) r |= b & -b;
    return r;
}
// k-th set bit (k from 0) of a set, in increasing order; 256 if none
__device__ __forceinline__ int kth(const S8 &s, int k) {
#pragma unroll
    for (int i = 0; i < 8; i++) {
        int c = __popc(s.w[i]);
        if (k < c) { uint32_t x = s.w[i]; for (int j = 0; j < k; j++) x &= x - 1; return i * 32 + __ffs(x) - 1; }
        k -= c;
    }
    return 256;
}

// ---------------------------------------------------------------- generation of G7
__device__ __forceinline__ uint64_t mix64(uint64_t z) {
    z ^= z >> 33; z *= 0xFF51AFD7ED558CCDULL; z ^= z >> 33; z *= 0xC4CEB9FE1A85EC53ULL; z ^= z >> 33; return z;
}
struct Rng {
    uint64_t s;
    __device__ __forceinline__ uint64_t next() {
        uint64_t z = (s += 0x9E3779B97F4A7C15ULL);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL; z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL; return z ^ (z >> 31);
    }
};
// insert v into an xor basis indexed by leading bit
__device__ __forceinline__ void binsert(uint8_t *bs, int v) {
    for (int b = 7; b >= 0 && v; b--) { if (!(v >> b & 1)) continue; if (!bs[b]) { bs[b] = v; return; } v ^= bs[b]; }
}
__device__ __forceinline__ bool bmember(const uint8_t *bs, int v) {
    for (int b = 7; b >= 0 && v; b--) if (v >> b & 1) { if (!bs[b]) return false; v ^= bs[b]; }
    return v == 0;
}
#define GEN_T 128
struct GenCount { unsigned long long att, valid, bad; };
__global__ void __launch_bounds__(GEN_T) k_gen(uint64_t seed, long long first, int nstarts, int tries,
                                               uint8_t *g7out, int *ng7, int cap, GenCount *cnt) {
    __shared__ uint32_t C[63 * GEN_T];   // C[(g - 1) * GEN_T + tid]: the dual functionals of Im6[gray(g)]
    int tid = threadIdx.x;
    long long i = (long long)blockIdx.x * GEN_T + tid;
    unsigned long long att = 0, valid = 0, bad = 0;
    if (i < nstarts) {
        Rng r; r.s = mix64(seed ^ mix64((uint64_t)(first + i) + 0x632BE59BD9B4E019ULL));
        uint8_t F[64];
        for (;;) {   // uniform APN beta6, as in mass8.c
            att++;
            F[0] = 0; F[1] = 0;
            bool ok = true;
            for (int k = 1; k <= 5 && ok; k++) {
                int K = 1 << k;
                uint64_t rv = r.next();
                uint8_t v[8];
                for (int j = 0; j < k; j++) v[j] = (uint8_t)(rv >> (8 * j));
                uint8_t L[32];
                L[0] = 0;
                for (int a = 1; a < K && ok; a++) {
                    L[a] = L[a & (a - 1)] ^ v[__ffs(a) - 1];
                    uint8_t bs[8] = {0, 0, 0, 0, 0, 0, 0, 0};
                    for (int j = 0; j < k; j++) binsert(bs, F[(1 << j) ^ a] ^ F[1 << j] ^ F[a] ^ F[0]);
                    if (bmember(bs, L[a])) ok = false;
                }
                if (ok) for (int x = 0; x < K; x++) F[K + x] = F[x] ^ L[x];
            }
            if (ok) break;
        }
        // Im6[a] (dimension 5 for APN beta6) as the common kernel of 8 - rank functionals, packed in bytes
        for (int g = 1; g < 64; g++) {
            int a = g ^ (g >> 1);
            uint8_t bs[8] = {0, 0, 0, 0, 0, 0, 0, 0};
            for (int j = 0; j < 6; j++) binsert(bs, F[(1 << j) ^ a] ^ F[1 << j] ^ F[a] ^ F[0]);
            for (int b = 7; b >= 0; b--) if (bs[b]) for (int c = 7; c >= 0; c--) if (c != b && bs[c] && (bs[c] >> b & 1)) bs[c] ^= bs[b];
            uint32_t pk = 0; int nf = 0;
            for (int f = 0; f < 8; f++) if (!bs[f]) {
                int c = 1 << f;
                for (int p = 0; p < 8; p++) if (bs[p] && (bs[p] >> f & 1)) c |= 1 << p;
                if (nf < 4) pk |= (uint32_t)c << (8 * nf);
                nf++;
            }
            if (nf != 3) bad++;
            C[(g - 1) * GEN_T + tid] = pk;
        }
        for (int t = 0; t < tries; t++) {
            uint64_t rv = r.next();
            uint32_t L = 0; int g;
            for (g = 1; g < 64; g++) {
                L ^= (uint32_t)(rv >> (8 * (__ffs(g) - 1))) & 255;
                uint32_t x = C[(g - 1) * GEN_T + tid] & (L * 0x01010101u);
                x ^= x >> 4; x ^= x >> 2; x ^= x >> 1;
                if ((x & 0x01010101u) == 0) break;   // L(a) in Im6[a]
            }
            if (g < 64) continue;
            valid++;
            int o = atomicAdd(ng7, 1);
            if (o < cap) {
                uint8_t *G = g7out + (size_t)o * 128;
                for (int x = 0; x < 64; x++) {
                    int l = 0;
                    for (int d = 0; d < 6; d++) if (x >> d & 1) l ^= (int)(rv >> (8 * d)) & 255;
                    G[x] = F[x]; G[64 + x] = F[x] ^ l;
                }
            }
        }
    }
    // block sums
    __shared__ unsigned long long s0[GEN_T / 32], s1[GEN_T / 32], s2[GEN_T / 32];
    for (int o = 16; o; o >>= 1) { att += __shfl_down_sync(~0u, att, o); valid += __shfl_down_sync(~0u, valid, o); bad += __shfl_down_sync(~0u, bad, o); }
    if ((tid & 31) == 0) { s0[tid >> 5] = att; s1[tid >> 5] = valid; s2[tid >> 5] = bad; }
    __syncthreads();
    if (tid == 0) {
        unsigned long long A = 0, V = 0, B = 0;
        for (int w = 0; w < GEN_T / 32; w++) { A += s0[w]; V += s1[w]; B += s2[w]; }
        atomicAdd(&cnt->att, A); atomicAdd(&cnt->valid, V); atomicAdd(&cnt->bad, B);
    }
}

// ---------------------------------------------------------------- the L7 search
// Node storage: the sets of a node at depth d, indexed by the compact index of U in R (2^(7-d) slots of 8 words,
// slot 0 unused). Header: (G7 index, R).
__global__ void __launch_bounds__(128) k_init(const uint8_t *g7, int n, uint32_t *Q0, int2 *hdr, int *bad) {
    int g = blockIdx.x, a = threadIdx.x;
    if (g >= n) return;
    const uint8_t *G = g7 + (size_t)g * 128;
    if (a == 0) { hdr[g] = make_int2(g, 127); return; }
    S8 s;
#pragma unroll
    for (int i = 0; i < 8; i++) s.w[i] = 0;
    s.w[0] = 1;
    for (int i = 0; i < 7; i++) {
        int e = 1 << i, b = G[a ^ e] ^ G[a] ^ G[e] ^ G[0];
        if (!(s.w[b >> 5] >> (b & 31) & 1)) { S8 t = xl(s, b); for (int j = 0; j < 8; j++) s.w[j] |= t.w[j]; }
    }
    if (pop8(s) != 64) atomicAdd(bad, 1);
#pragma unroll
    for (int i = 0; i < 8; i++) s.w[i] = ~s.w[i];
    st8(Q0 + ((size_t)g * 128 + a) * 8, s);
}

// one warp per parent node at depth D (D = 0 or 1); children written to C/chdr; tot counts the nodes made
template <int D>
__global__ void __launch_bounds__(128) k_expand(const uint32_t *P, const int2 *phdr, int np, const uint8_t *g7,
                                                uint32_t *C, int2 *chdr, int *nc, int capc, int *overflow) {
    constexpr int M = 7 - D, NS = 1 << M, NC = NS / 2;   // parent has NS slots, child NC slots
    int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    if (warp >= np) return;
    int2 h = phdr[warp];
    int g = h.x, R = h.y;
    const uint32_t *Pn = P + (size_t)warp * NS * 8;
    int bu;
    if (D == 0) bu = 1;
    else {   // fewest candidates, ties to the smallest U (compact order = order of U)
        int bc = 1 << 20, bi = NS;
        for (int i = lane + 1; i < NS; i += 32) { int c = pop8(ld8(Pn + i * 8)); if (c < bc || (c == bc && i < bi)) { bc = c; bi = i; } }
        for (int o = 16; o; o >>= 1) {
            int c2 = __shfl_xor_sync(~0u, bc, o), i2 = __shfl_xor_sync(~0u, bi, o);
            if (c2 < bc || (c2 == bc && i2 < bi)) { bc = c2; bi = i2; }
        }
        bu = pdep7(bi, R);
    }
    S8 sb = ld8(Pn + pext7(bu, R) * 8);
    int R2 = R & ~(bu & -bu);
    const uint8_t *G = g7 + (size_t)g * 128;
    int bb = D == 1 ? (G[1 ^ bu] ^ G[1] ^ G[bu] ^ G[0]) : 0;
    S8 im1;   // D == 0: Im7[e_0] = complement of Q0[1]
    if (D == 0) for (int i = 0; i < 8; i++) im1.w[i] = ~sb.w[i];
    int reps[3], nr = 0;
    for (int w = 0; w < 8; w++) for (uint32_t m = sb.w[w]; m; m &= m - 1) {
        int y = w * 32 + __ffs(m) - 1;
        if (D == 0) {   // one representative per coset of Im7[e_0]
            bool dup = false;
            for (int r = 0; r < nr; r++) { int z = y ^ reps[r]; if (im1.w[z >> 5] >> (z & 31) & 1) dup = true; }
            if (dup) continue;
            reps[nr++] = y;
        } else if (y > (y ^ bb)) continue;
        S8 c0, c1;
        bool ok = true;
        int j0 = lane + 1, j1 = lane + 33;
        if (j0 < NC) { int U = pdep7(j0, R2); c0 = and8(ld8(Pn + pext7(U, R) * 8), xl(ld8(Pn + pext7(U ^ bu, R) * 8), y)); ok &= any8(c0); }
        if (NC > 33 && j1 < NC) { int U = pdep7(j1, R2); c1 = and8(ld8(Pn + pext7(U, R) * 8), xl(ld8(Pn + pext7(U ^ bu, R) * 8), y)); ok &= any8(c1); }
        if (!__all_sync(~0u, ok)) continue;
        int o = 0;
        if (lane == 0) o = atomicAdd(nc, 1);
        o = __shfl_sync(~0u, o, 0);
        if (o >= capc) { if (lane == 0) atomicAdd(overflow, 1); continue; }
        uint32_t *Cn = C + (size_t)o * NC * 8;
        if (j0 < NC) st8(Cn + j0 * 8, c0);
        if (NC > 33 && j1 < NC) st8(Cn + j1 * 8, c1);
        if (lane == 0) chdr[o] = make_int2(g, R2);
        if (D == 0 && nr == 3) {}   // at most three representatives
    }
}

// depths 5-7 from the three sets of depth 5 (indexed by U in {lo, hi, lo | hi}); returns the number of leaves
__device__ __forceinline__ int bt5(const S8 &qlo, const S8 &qhi, const S8 &qall) {
    int clo = pop8(qlo), chi = pop8(qhi), call = pop8(qall);
    // fewest candidates, ties to the smallest U: lo < hi < lo | hi
    int which = 0, bc = clo;
    if (chi < bc) { bc = chi; which = 1; }
    if (call < bc) { bc = call; which = 2; }
    // remaining U after removing the lowest bit of bu: bu = lo -> hi; bu = hi -> lo; bu = lo|hi -> hi.
    // Q6 = Q5[rest] & (Q5[rest ^ bu] translated by y5)
    const S8 &sb = which == 0 ? qlo : which == 1 ? qhi : qall;
    const S8 &rest = which == 1 ? qlo : qhi;
    const S8 &other = which == 0 ? qall : which == 1 ? qall : qlo;
    int leaves = 0;
    for (int w = 0; w < 8; w++) for (uint32_t m = sb.w[w]; m; m &= m - 1) {
        int y = w * 32 + __ffs(m) - 1;
        leaves += pop8(and8(rest, xl(other, y)));
    }
    return leaves;
}

// The depth-3 sets are indexed by their compact index j in R3 (4 bits); pext preserves xor and the order of the
// bits, so the bt3() logic of mass8.c runs unchanged on j. q3(j) = Q3[j] = Q2[t1[j]] & (Q2[t2[j]] translated by y2).
#ifndef EXP
#define EXP 0
#endif
__device__ __forceinline__ S8 q3(const uint32_t *P, const uint8_t *t, int j, int y2) {
    return and8(ld8(P + t[j] * 8), xl(ld8(P + t[16 + j] * 8), y2));
}
// Depth 3 onwards, in three stages:
//   k_rec    one warp per depth-2 node, one lane per child (= depth-3 node): the 15 popcounts, the choices of bt3()
//            (bu3, W, the lines P0, P1, P2), and a record with the sets that the next stage reads;
//   k_proc   one warp per group of 4 records: the 8 within-byte translates of qWu, a2, b1, b2 go to shared memory,
//            then the lanes share the (record, y) pairs of the group, y a candidate of bu3: the sets of W and of the
//            first line P0 after L7(e_bu3) = y, and for each z of W the first line after L7(e_W) = z; if it
//            survives, a tail entry (record, y, z);
//   k_tail   one thread per tail entry: the other two lines and depths 5-7 (bt5), leaves added to the G7.
// Record (64 words): sb3, qW, qWu, a1, a2, b1, b2 (8 words each), then node, y2, bu2, bu3, W (compact indices in R3).
#define RECW 64
__device__ __forceinline__ uint32_t psel(int v) { return 0x3210u ^ (uint32_t)(((v >> 3) & 3) * 0x1111); }
__global__ void __launch_bounds__(128) k_rec(const uint32_t *N2, const int2 *H2, int base, int np, uint32_t *rec,
                                             int *nrec, int caprec, unsigned long long *stat) {
    __shared__ uint8_t tab[4][32];
    __shared__ uint8_t spc[16 * 128];
    int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31, wb = threadIdx.x >> 5, tid = threadIdx.x;
    if (warp >= np) return;
    int node = base + warp;
    int R2 = H2[node].y;
    const uint32_t *P = N2 + (size_t)node * 32 * 8;
    int bc = 1 << 20, bi = 32;
    if (lane >= 1) { bc = pop8(ld8(P + lane * 8)); bi = lane; }
    for (int o = 16; o; o >>= 1) {
        int c2 = __shfl_xor_sync(~0u, bc, o), i2 = __shfl_xor_sync(~0u, bi, o);
        if (c2 < bc || (c2 == bc && i2 < bi)) { bc = c2; bi = i2; }
    }
    int bu2 = pdep7(bi, R2);
    int R3 = R2 & ~(bu2 & -bu2);
    if (lane < 16) { int U = pdep7(lane, R3); tab[wb][lane] = pext7(U, R2); tab[wb][16 + lane] = pext7(U ^ bu2, R2); }
    __syncwarp();
    const uint8_t *t = tab[wb];
    S8 sb2 = ld8(P + bi * 8);
    unsigned long long n3 = 0, n4 = 0;
    for (int k = lane; k < bc; k += 32) {
        int y2 = kth(sb2, k);
        bool dead = false;
        for (int j = 1; j < 16; j++) { int cc = pop8(q3(P, t, j, y2)); spc[j * 128 + tid] = cc; dead |= cc == 0; }
        if (dead) continue;
        n3++;
        int bu3 = 1, bcc = spc[128 + tid];
        for (int j = 2; j < 16; j++) { int cc = spc[j * 128 + tid]; if (cc < bcc) { bcc = cc; bu3 = j; } }   // ties: smallest
        n4 += bcc;
        int R4 = 15 & ~(bu3 & -bu3);
        int W = 0, wc = 1 << 30;
        for (int U = R4; U; U = (U - 1) & R4) {   // descending with <=: ties to the smallest, as mass8.c
            int cc = spc[U * 128 + tid] * spc[(U ^ bu3) * 128 + tid];
            if (cc <= wc) { wc = cc; W = U; }
        }
        int R5 = R4 & ~(W & -W), P0 = R5;
        int r = atomicAdd(nrec, 1);
        if (r >= caprec) continue;   // the host sees the counter and redoes the chunk smaller
        uint32_t *R = rec + (size_t)r * RECW;
        st8(R, q3(P, t, bu3, y2)); st8(R + 8, q3(P, t, W, y2)); st8(R + 16, q3(P, t, W ^ bu3, y2));
        st8(R + 24, q3(P, t, P0, y2)); st8(R + 32, q3(P, t, P0 ^ bu3, y2));
        st8(R + 40, q3(P, t, P0 ^ W, y2)); st8(R + 48, q3(P, t, P0 ^ W ^ bu3, y2));
        *(uint4 *)(R + 56) = make_uint4(node, y2, bu2, bu3); R[60] = W;
    }
    if (stat) {
        for (int o = 16; o; o >>= 1) { n3 += __shfl_xor_sync(~0u, n3, o); n4 += __shfl_xor_sync(~0u, n4, o); }
        if (lane == 0) { atomicAdd(stat, n3); atomicAdd(stat + 1, n4); }
    }
}
// shared memory per warp: V[record][set k][variant m][word] for k = qWu, a2, b1, b2; then qW, a1, sb3 per record
#ifndef PG
#define PG 1
#endif
struct ProcSh { uint32_t V[PG][4][8][8]; uint32_t qW[PG][8], a1[PG][8], sb[PG][8]; };
__device__ __forceinline__ S8 trs(const uint32_t *b, int v) {   // b = V[r][k]: the set translated by v
    const uint32_t *c = b + (v & 7) * 8;
    int h = v >> 5; uint32_t sel = psel(v);
    S8 r;
#pragma unroll
    for (int i = 0; i < 8; i++) r.w[i] = __byte_perm(c[i ^ h], 0, sel);
    return r;
}
__device__ __forceinline__ S8 trs2(const uint32_t *b, int v, const uint32_t *b2, int u) {
    const uint32_t *c = b + (v & 7) * 8, *d = b2 + (u & 7) * 8;
    int h = v >> 5, g = u >> 5; uint32_t sel = psel(v), sel2 = psel(u);
    S8 r;
#pragma unroll
    for (int i = 0; i < 8; i++) r.w[i] = __byte_perm(c[i ^ h], 0, sel) & __byte_perm(d[i ^ g], 0, sel2);
    return r;
}
__device__ __forceinline__ S8 lds8(const uint32_t *p) {
    S8 r;
#pragma unroll
    for (int i = 0; i < 8; i++) r.w[i] = p[i];
    return r;
}
__global__ void __launch_bounds__(128) k_proc(const uint32_t *rec, int nrec, int4 *tail, int *ntail, int captail,
                                              unsigned long long *stat) {
    __shared__ ProcSh sh[4];
    int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    ProcSh &S = sh[threadIdx.x >> 5];
    int r0 = warp * PG;
    if (r0 >= nrec) return;
    int nr = nrec - r0 < PG ? nrec - r0 : PG;
    // the translates: lane handles (record i, set k = lane >> 3, variant m = lane & 7)
    {
        int k = lane >> 3, m = lane & 7;
        int off = k == 0 ? 16 : k == 1 ? 32 : k == 2 ? 40 : 48;
        for (int i = 0; i < nr; i++) {
            S8 x = ld8(rec + (size_t)(r0 + i) * RECW + off);
#pragma unroll
            for (int j = 0; j < 8; j++) S.V[i][k][m][j] = bitx(x.w[j], m);
        }
        for (int l = lane; l < 8 * nr; l += 32) {
            int i = l >> 3, w = l & 7;
            const uint32_t *R = rec + (size_t)(r0 + i) * RECW;
            S.sb[i][w] = R[w]; S.qW[i][w] = R[8 + w]; S.a1[i][w] = R[24 + w];
        }
    }
    __syncwarp();
    int cnt[PG], tot = 0;
    for (int i = 0; i < PG; i++) { cnt[i] = 0; if (i < nr) { S8 sb = lds8(S.sb[i]); cnt[i] = pop8(sb); } tot += cnt[i]; }
    unsigned long long n5 = 0;
    for (int qb = 0; qb < tot; qb += 32) {   // a round: one (record, y) per lane
        int q = qb + lane, i = 0, y = 0;
        S8 sw, a;
#pragma unroll
        for (int j = 0; j < 8; j++) { sw.w[j] = 0; a.w[j] = 0; }
        if (q < tot) {
            int k = q;
            while (k >= cnt[i]) { k -= cnt[i]; i++; }
            y = kth(lds8(S.sb[i]), k);
            sw = and8(lds8(S.qW[i]), trs(&S.V[i][0][0][0], y));
            a = and8(lds8(S.a1[i]), trs(&S.V[i][1][0][0], y));
            S8 b = trs2(&S.V[i][2][0][0], 0, &S.V[i][3][0][0], y);
            if (!any8(a) || !any8(b)) {
#pragma unroll
                for (int j = 0; j < 8; j++) sw.w[j] = 0;
            }
        }
        // the pairs (y, z), z in sw, spread over the lanes
        int c = pop8(sw), inc = c;
        for (int o = 1; o < 32; o <<= 1) { int t = __shfl_up_sync(~0u, inc, o); if (lane >= o) inc += t; }
        int T = __shfl_sync(~0u, inc, 31);
        n5 += c;
        for (int pb = 0; pb < T; pb += 32) {
            int p = pb + lane, src = 0;
            for (int st = 16; st; st >>= 1) { int v = __shfl_sync(~0u, inc, src + st - 1); if (v <= p) src += st; }
            if (src > 31) src = 31;
            S8 sws, as;
#pragma unroll
            for (int j = 0; j < 8; j++) { sws.w[j] = __shfl_sync(~0u, sw.w[j], src); as.w[j] = __shfl_sync(~0u, a.w[j], src); }
            int ys = __shfl_sync(~0u, y, src), is = __shfl_sync(~0u, i, src);
            int ex = __shfl_sync(~0u, inc - c, src);
            if (p < T) {
                int z = kth(sws, p - ex);
                // first line after L7(e_W) = z: a & (b1 + z) & (b2 + y + z)
                if (any8(and8(as, trs2(&S.V[is][2][0][0], z, &S.V[is][3][0][0], ys ^ z)))) {
                    int o = atomicAdd(ntail, 1);
                    if (o < captail) tail[o] = make_int4(r0 + is, ys, z, 0);
                }
            }
        }
    }
    if (stat) {
        for (int o = 16; o; o >>= 1) n5 += __shfl_xor_sync(~0u, n5, o);
        if (lane == 0 && n5) atomicAdd(stat + 2, n5);
    }
}
__global__ void k_tail(const uint32_t *rec, const int4 *tail, int nt, const uint32_t *N2, const int2 *H2, int *leaves) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nt) return;
    int4 e = tail[i];
    const uint32_t *R = rec + (size_t)e.x * RECW;
    int node = R[56], y2 = R[57], bu2 = R[58], bu3 = R[59], W = R[60], y = e.y, z = e.z;
    int2 h = H2[node];
    const uint32_t *P = N2 + (size_t)node * 32 * 8;
    int R2 = h.y, R3 = R2 & ~(bu2 & -bu2);
    uint8_t t[32];
    for (int j = 0; j < 16; j++) { int U = pdep7(j, R3); t[j] = pext7(U, R2); t[16 + j] = pext7(U ^ bu2, R2); }
    int R4 = 15 & ~(bu3 & -bu3), R5 = R4 & ~(W & -W);
    int P1 = R5 & ~(R5 & -R5), P2 = R5 & -R5;
    S8 t0 = and8(and8(ld8(R + 24), xl(ld8(R + 32), y)), and8(xl(ld8(R + 40), z), xl(ld8(R + 48), y ^ z)));
    S8 t1 = and8(and8(q3(P, t, P1, y2), xl(q3(P, t, P1 ^ bu3, y2), y)), xl(and8(q3(P, t, P1 ^ W, y2), xl(q3(P, t, P1 ^ W ^ bu3, y2), y)), z));
    if (!any8(t1)) return;
    S8 t2 = and8(and8(q3(P, t, P2, y2), xl(q3(P, t, P2 ^ bu3, y2), y)), xl(and8(q3(P, t, P2 ^ W, y2), xl(q3(P, t, P2 ^ W ^ bu3, y2), y)), z));
    if (!any8(t2)) return;
    int nl = bt5(t2, t1, t0);   // lo = P2, hi = P1, lo | hi = P0
    if (nl) atomicAdd(&leaves[h.x], nl);
}

// ---------------------------------------------------------------- host
static double now();
static int prof = 0; static double tk[4];
static void sync_t(int i, double &a) { if (!prof) return; CK(cudaDeviceSynchronize()); double b = now(); tk[i] += b - a; a = b; }
struct Search {
    int maxg, cap1, cap2, chunk1, caprec, capitem, captail;
    uint8_t *g7; uint32_t *Q0, *N1, *N2, *rec, *item; int2 *H0, *H1, *H2; int4 *tl; int *leaves, *cnt, *bad, *ovf;
    unsigned long long *stat;
    long long ntails = 0;
    void init(int mg) {
        CK(cudaFuncSetAttribute(k_proc, cudaFuncAttributePreferredSharedMemoryCarveout, 100));
        maxg = mg; cap1 = 3 * mg; cap2 = 1 << 21; chunk1 = cap2 / 256;
        caprec = 6 << 20; capitem = 0; captail = 4 << 20;
        CK(cudaMalloc(&g7, (size_t)maxg * 128)); CK(cudaMalloc(&Q0, (size_t)maxg * 128 * 32));
        CK(cudaMalloc(&N1, (size_t)cap1 * 64 * 32)); CK(cudaMalloc(&N2, (size_t)cap2 * 32 * 32));
        CK(cudaMalloc(&H0, (size_t)maxg * 8)); CK(cudaMalloc(&H1, (size_t)cap1 * 8)); CK(cudaMalloc(&H2, (size_t)cap2 * 8));
        CK(cudaMalloc(&rec, (size_t)caprec * RECW * 4)); CK(cudaMalloc(&tl, (size_t)captail * 16));
        CK(cudaMalloc(&leaves, (size_t)maxg * 4)); CK(cudaMalloc(&cnt, 32)); CK(cudaMalloc(&bad, 4)); CK(cudaMalloc(&ovf, 4));
        CK(cudaMalloc(&stat, 48)); CK(cudaMemset(stat, 0, 48)); CK(cudaMemset(bad, 0, 4)); CK(cudaMemset(ovf, 0, 4));
    }
    // leaves of the n G7 already in g7 (device). cnt: [0] depth-1 nodes, [1] depth-2 nodes, [2] G7 (k_gen),
    // [3] records, [4] items, [5] tail entries
    void run(int n, std::vector<int> &lv) {
        CK(cudaMemset(leaves, 0, (size_t)n * 4)); CK(cudaMemset(cnt, 0, 8));
        double ta = now();
        k_init<<<n, 128>>>(g7, n, Q0, H0, bad);
        k_expand<0><<<(n + 3) / 4, 128>>>(Q0, H0, n, g7, N1, H1, cnt, cap1, ovf);
        sync_t(0, ta);
        int n1; CK(cudaMemcpy(&n1, cnt, 4, cudaMemcpyDeviceToHost));
        for (int s = 0; s < n1; s += chunk1) {
            int m = n1 - s < chunk1 ? n1 - s : chunk1;
            CK(cudaMemset(cnt + 1, 0, 4));
            k_expand<1><<<(m + 3) / 4, 128>>>(N1 + (size_t)s * 64 * 8, H1 + s, m, g7, N2, H2, cnt + 1, cap2, ovf);
            int n2; CK(cudaMemcpy(&n2, cnt + 1, 4, cudaMemcpyDeviceToHost));
            if (n2 > cap2) n2 = cap2;
            sync_t(0, ta);
            int c2 = 32768;
            for (int s2 = 0; s2 < n2;) {
                int m2 = n2 - s2 < c2 ? n2 - s2 : c2;
                CK(cudaMemset(cnt + 3, 0, 12));
                k_rec<<<(m2 + 3) / 4, 128>>>(N2, H2, s2, m2, rec, cnt + 3, caprec, stat);
                int c[3]; CK(cudaMemcpy(c, cnt + 3, 4, cudaMemcpyDeviceToHost));
                if (c[0] > caprec) { c2 = m2 / 2; if (c2 < 1) { fprintf(stderr, "mass8g: record overflow\n"); exit(3); } continue; }
                sync_t(1, ta);
                int nw = (c[0] + PG - 1) / PG;
                k_proc<<<(nw + 3) / 4, 128>>>(rec, c[0], tl, cnt + 5, captail, stat);
                CK(cudaMemcpy(c + 2, cnt + 5, 4, cudaMemcpyDeviceToHost));
                if (c[2] > captail) { fprintf(stderr, "mass8g: tail overflow\n"); exit(3); }
                sync_t(2, ta);
                ntails += c[2];
                if (c[2]) k_tail<<<(c[2] + 127) / 128, 128>>>(rec, tl, c[2], N2, H2, leaves);
                sync_t(3, ta);
                s2 += m2;
            }
        }
        lv.resize(n);
        CK(cudaMemcpy(lv.data(), leaves, (size_t)n * 4, cudaMemcpyDeviceToHost));
        int b, o; CK(cudaMemcpy(&b, bad, 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(&o, ovf, 4, cudaMemcpyDeviceToHost));
        if (b || o) { fprintf(stderr, "mass8g: internal error (bad %d, overflow %d)\n", b, o); exit(3); }
    }
};

static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

int main(int argc, char **argv) {
    long long massS = 0; uint64_t seed = 1; int tries = 4096; long cap = 64; int dev = 0; const char *check = NULL;
    int batch = 32768; int verbose = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-mass")) massS = atoll(argv[++i]);
        else if (!strcmp(argv[i], "-seed")) seed = strtoull(argv[++i], 0, 10);
        else if (!strcmp(argv[i], "-tries")) tries = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-cap")) cap = atol(argv[++i]);
        else if (!strcmp(argv[i], "-dev")) dev = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-check")) check = argv[++i];
        else if (!strcmp(argv[i], "-batch")) batch = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-v")) verbose = 1;
        else if (!strcmp(argv[i], "-prof")) prof = 1;
    }
    CK(cudaSetDevice(dev));
    CK(cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync));   // wait without spinning: the CPU stays free
    Search S; S.init(batch);
    if (check) {
        FILE *f = fopen(check, "rb");
        if (!f) { perror(check); return 1; }
        std::vector<uint8_t> rec;
        uint8_t b[132];
        while (fread(b, 1, 132, f) == 132) rec.insert(rec.end(), b, b + 132);
        fclose(f);
        long n = rec.size() / 132, mism = 0, pos = 0, cpupos = 0;
        double t0 = now(), tg = 0;
        std::vector<uint8_t> buf((size_t)batch * 128);
        std::vector<int> lv;
        for (long s = 0; s < n; s += batch) {
            int m = n - s < batch ? n - s : batch;
            for (int j = 0; j < m; j++) memcpy(&buf[(size_t)j * 128], &rec[(size_t)(s + j) * 132], 128);
            CK(cudaMemcpy(S.g7, buf.data(), (size_t)m * 128, cudaMemcpyHostToDevice));
            double a = now(); S.run(m, lv); tg += now() - a;
            for (int j = 0; j < m; j++) {
                int32_t o; memcpy(&o, &rec[(size_t)(s + j) * 132 + 128], 4);
                if (o) cpupos++;
                if (lv[j]) pos++;
                if (2 * lv[j] != o) { mism++; if (mism <= 10) fprintf(stderr, "mismatch at %ld: GPU leaves %d, CPU outputs %d\n", s + j, lv[j], o); }
            }
        }
        unsigned long long st[6]; CK(cudaMemcpy(st, S.stat, 48, cudaMemcpyDeviceToHost));
        if (st[3]) fprintf(stderr, "survivors %llu: mean |sw| %.2f |a| %.2f |b| %.2f min %.2f\n", st[3], (double)st[0] / st[3], (double)st[1] / st[3], (double)st[2] / st[3], (double)st[4] / st[3]);
        fprintf(stderr, "check: %ld G7, %ld positive (CPU %ld), %ld mismatches; GPU %.2f s = %.0f G7/s (total %.2f s); nodes d3 %llu d4 %llu d5 %llu\n",
                n, pos, cpupos, mism, tg, n / tg, now() - t0, st[0], st[1], st[2]);
        if (prof) fprintf(stderr, "kernels: expand %.2f rec %.2f proc %.2f tail %.2f s; tail entries %lld\n", tk[0], tk[1], tk[2], tk[3], S.ntails);
        return mism != 0;
    }
    if (!massS) { fprintf(stderr, "usage: mass8g -mass S -seed SEED [-tries T] [-cap C] [-dev D] | -check FILE\n"); return 1; }
    // generation in chunks of starts sized so that the G7 buffer cannot plausibly overflow (an overflow aborts)
    int chunk = (int)((double)batch / 0.45 / (tries / 4096.0) / 2);   // about batch / 2 G7 expected per chunk
    if (chunk < GEN_T) chunk = GEN_T;
    GenCount *dc; CK(cudaMalloc(&dc, sizeof(GenCount))); CK(cudaMemset(dc, 0, sizeof(GenCount)));
    long long att = 0, nvalid = 0, nout = 0, nonstd = 0, mism = 0;
    std::vector<int> lv; std::vector<uint8_t> g((size_t)batch * 128);
    double t0 = now(), tgen = 0, tsearch = 0, tcpu = 0;
    for (long long s = 0; s < massS; s += chunk) {
        int m = massS - s < chunk ? (int)(massS - s) : chunk;
        CK(cudaMemset(S.cnt + 2, 0, 4));
        double a = now();
        k_gen<<<(m + GEN_T - 1) / GEN_T, GEN_T>>>(seed, s, m, tries, S.g7, S.cnt + 2, batch, dc);
        int ng; CK(cudaMemcpy(&ng, S.cnt + 2, 4, cudaMemcpyDeviceToHost));
        tgen += now() - a;
        if (ng > batch) { fprintf(stderr, "mass8g: G7 buffer overflow (%d > %d)\n", ng, batch); return 4; }
        a = now(); S.run(ng, lv); tsearch += now() - a;
        a = now();
        bool anypos = false;
        for (int j = 0; j < ng; j++) anypos |= lv[j] != 0;
        if (anypos) {
            CK(cudaMemcpy(g.data(), S.g7, (size_t)ng * 128, cudaMemcpyDeviceToHost));
            for (int j = 0; j < ng; j++) if (lv[j]) {
                long o = cpu_g7(&g[(size_t)j * 128], cap);
                nout += o; if (o != 0 && o != 2) nonstd++;
                if (o != 2 * lv[j]) { mism++; fprintf(stderr, "mass8g: mismatch, GPU leaves %d, CPU outputs %ld\n", lv[j], o); }
            }
        }
        tcpu += now() - a;
        if (verbose) fprintf(stderr, "chunk %lld: %d G7\n", s, ng);
    }
    GenCount hc; CK(cudaMemcpy(&hc, dc, sizeof hc, cudaMemcpyDeviceToHost));
    att = hc.att; nvalid = hc.valid;
    if (hc.bad) { fprintf(stderr, "mass8g: %llu starts with a degenerate Im6\n", hc.bad); return 5; }
    fflush(stdout);
    fprintf(stderr, "time %.1f s: gen %.1f search %.1f cpu %.1f; %.0f G7/s; mismatch %lld\n", now() - t0, tgen, tsearch, tcpu,
            nvalid / (now() - t0), mism);
    nonstd += mism * 1000000;   // a mismatch marks the call as nonstandard, like the node limit in mass8
    fprintf(stderr, "%lld %lld %lld %lld %lld %lld splitmix2g\n", massS, att, massS * tries, nvalid, nout, nonstd);
    return 0;
}
