/* autq.c: order of the linear self-equivalence group of a quadratic APN function F on F_2^n (n <= 8).
 * G = { (A,B) in GL(n)^2 : beta(Ax,Ay) = B beta(x,y) },  beta(x,y) = F(x+y)+F(x)+F(y)+F(0).
 * For APN F the values of beta span F_2^n, so B is fixed by A and G embeds in GL(n).
 * G is trivial  <=>  F has no linear self-equivalence (A,B) != (I,I)  <=>  (Cauchy) none of prime order.
 * Pruning: the ortho-derivative pi satisfies pi(Ax) = C pi(x) with C = B^{-T} linear; point invariants
 * (|pi^-1(pi(x))|, #{y : pi(x)+pi(y)+pi(x+y) = 0}) must agree; B and C are kept as partial linear maps.
 * Input: one LUT per line (any separators), values decimal or 0x-hex. Output per line: group order and
 * the multiset of element orders.
 * Build: cc -O3 -o autq autq.c
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

static int n, N;
static int F[256], beta[256][256], pi[256], inv1[256], inv2[256];
static long long nodes;
typedef struct { int nb; int u[16], v[16], piv[16]; } Map;   /* partial linear map, rows reduced by pivot */

static int map_add(Map *m, int u, int v) {   /* 1 = consistent */
    for (int i = 0; i < m->nb; i++) if ((u >> m->piv[i]) & 1) { u ^= m->u[i]; v ^= m->v[i]; }
    if (u == 0) return v == 0;
    int p = 31 - __builtin_clz(u);
    m->u[m->nb] = u; m->v[m->nb] = v; m->piv[m->nb] = p; m->nb++;
    return 1;
}

static int img[8];          /* A(e_i) */
static int Aspan[256];      /* A on span(e_0..e_{k-1}) */
static long long order_count;
static int els[2000000][8]; /* found automorphisms (column images) */

static void rec(int k, Map B, Map C) {
    nodes++;
    if (k == n) { if (order_count < 2000000) memcpy(els[order_count], img, sizeof img); order_count++; return; }
    int half = 1 << k;
    for (int y = 1; y < N; y++) {
        /* independence: y not in span of earlier images */
        int dep = 0; for (int s = 0; s < half; s++) if (Aspan[s] == y) { dep = 1; break; }
        if (dep) continue;
        int x0 = 1 << k;
        if (inv1[x0] != inv1[y] || inv2[x0] != inv2[y]) continue;
        Map B2 = B, C2 = C; int ok = 1;
        for (int s = 0; s < half && ok; s++) {
            int x = x0 ^ s, ax = y ^ Aspan[s];
            if (inv1[x] != inv1[ax] || inv2[x] != inv2[ax]) { ok = 0; break; }
            if (!map_add(&C2, pi[x], pi[ax])) ok = 0;
        }
        for (int j = 0; j < k && ok; j++) if (!map_add(&B2, beta[x0][1 << j], beta[y][img[j]])) ok = 0;
        if (!ok) continue;
        img[k] = y; for (int s = 0; s < half; s++) Aspan[half + s] = y ^ Aspan[s];
        rec(k + 1, B2, C2);
    }
}

static int apply(const int *c, int x) { int r = 0; for (int i = 0; i < n; i++) if ((x >> i) & 1) r ^= c[i]; return r; }

int main(int argc, char **argv) {
    n = argc > 1 ? atoi(argv[1]) : 8; N = 1 << n;
    char *line = malloc(1 << 16);
    while (fgets(line, 1 << 16, stdin)) {
        int cnt = 0; char *p = line;
        while (*p && cnt < N) {
            while (*p && !isxdigit((unsigned char)*p)) p++;
            if (!*p) break;
            char *e; long v = (p[0] == '0' && (p[1] == 'x' || p[1] == 'X')) ? strtol(p, &e, 16) : strtol(p, &e, 10);
            F[cnt++] = (int)v; p = e;
        }
        if (cnt < N) continue;
        for (int x = 0; x < N; x++) for (int y = 0; y < N; y++) beta[x][y] = F[x ^ y] ^ F[x] ^ F[y] ^ F[0];
        int apn = 1;
        pi[0] = 0;
        for (int x = 1; x < N; x++) {
            /* u != 0 with u . beta(x,y) = 0 for all y: kernel of the span of beta(x, .) */
            int found = -1, c = 0;
            for (int u = 1; u < N; u++) { int z = 1; for (int y = 0; y < N && z; y++) if (__builtin_parity(u & beta[x][y])) z = 0; if (z) { found = u; c++; } }
            if (c != 1) apn = 0; pi[x] = found;
        }
        if (!apn) { printf("not APN (or not quadratic)\n"); continue; }
        for (int x = 0; x < N; x++) { int a = 0, b = 0; for (int y = 0; y < N; y++) { if (pi[y] == pi[x]) a++; if ((pi[x] ^ pi[y] ^ pi[x ^ y]) == 0) b++; } inv1[x] = a; inv2[x] = b; }
        Map B = {0}, C = {0}; nodes = 0; order_count = 0; Aspan[0] = 0;
        rec(0, B, C);
        /* element orders */
        long long hist[300] = {0};
        long long lim = order_count < 2000000 ? order_count : 2000000;
        for (long long g = 0; g < lim; g++) {
            int x = 0; for (int t = 1; t <= 260; t++) {
                int ok = 1; for (int i = 0; i < n && ok; i++) { int v = 1 << i; for (int r = 0; r < t; r++) v = apply(els[g], v); if (v != (1 << i)) ok = 0; }
                if (ok) { x = t; break; }
            }
            hist[x]++;
        }
        if (getenv("AUTQ_PRINT")) for (long long g = 0; g < lim; g++) { printf("A"); for (int i = 0; i < n; i++) printf(" %d", els[g][i]); printf("\n"); }
        printf("|G| = %lld  nodes = %lld  orders:", order_count, nodes);
        for (int t = 1; t < 300; t++) if (hist[t]) printf(" %d^%lld", t, hist[t]);
        printf("\n"); fflush(stdout);
    }
    return 0;
}
