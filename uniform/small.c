/* small.c: count APN alternating bilinear maps beta: F_2^n x F_2^n -> F_2^m (quadratic APN (n,m)-functions mod affine)
 * (1) exactly, by building the function one variable at a time (Theorem 4.1 of BLLPR) with full enumeration;
 * (2) by the estimator: levels 1..n-2 random L with accept/reject, last level counted exactly.
 * usage: small n m samples seed */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
static int n, m;
static uint64_t rs;
static uint64_t rnd(void){ rs^=rs<<13; rs^=rs>>7; rs^=rs<<17; return rs; }
/* F as LUT on F_2^k (quadratic, F(0)=0, no linear part); Im(D_aF) = span{beta(a,e_i)} */
static int inim(const int *F, int k, int a, int v){ /* is v in span{F(a^e_i)^F(a)^F(e_i)} ? */
    int b[8], r=0;
    for(int i=0;i<k;i++){ int x=F[a^(1<<i)]^F[a]^F[1<<i]; for(int j=0;j<r;j++) if((x^b[j])<x) x^=b[j];
        if(x){ b[r++]=x; for(int j=r-1;j>0&&b[j]>b[j-1];j--){int t=b[j];b[j]=b[j-1];b[j-1]=t;} } }
    for(int j=0;j<r;j++) if((v^b[j])<v) v^=b[j];
    return v==0;
}
/* count all L : F_2^k -> F_2^m valid for F (DFS on basis values, check a with top bit j) */
static long double cntL(const int *F, int k, int j, int *L){
    if(j==k) return 1;
    long double c=0; int K=1<<j;
    for(int v=0; v<(1<<m); v++){
        int ok=1;
        for(int a=0;a<K&&ok;a++){ L[K+a]=L[a]^v; if(inim(F,k,K+a,L[K+a])) ok=0; }
        if(ok) c+=cntL(F,k,j+1,L);
    }
    return c;
}
static void extend(const int *F, int k, const int *L, int *G){ int K=1<<k; for(int x=0;x<K;x++){G[x]=F[x]; G[K+x]=F[x]^L[x];} }
static long double exact(const int *F, int k){ /* number of APN extensions of F to dimension n */
    if(k==n) return 1;
    if(k==n-1){ int L[256]; L[0]=0; return cntL(F,k,0,L); }
    long double c=0; int K=1<<k; int L[256], v[8];
    long tot=1L<<(m*k);
    for(long t=0;t<tot;t++){
        for(int i=0;i<k;i++) v[i]=(t>>(m*i))&((1<<m)-1);
        L[0]=0; int ok=1;
        for(int a=1;a<K&&ok;a++){ L[a]=L[a&(a-1)]^v[__builtin_ctz(a)]; if(inim(F,k,a,L[a])) ok=0; }
        if(!ok) continue;
        int G[256]; extend(F,k,L,G); c+=exact(G,k+1);
    }
    return c;
}
int main(int argc,char**argv){
    n=atoi(argv[1]); m=atoi(argv[2]); long S=atol(argv[3]); rs=strtoull(argv[4],0,10)*0x9E3779B97F4A7C15ULL+1;
    int F1[2]={0,0};
    if(S==0){ printf("exact %.0Lf\n", exact(F1,1)); return 0; }
    long double sum=0, sum2=0; double logw=0; for(int k=1;k<=n-2;k++) logw+=m*k;
    for(long s=0;s<S;s++){
        int F[256], G[256], k=1, ok=1; memcpy(F,F1,sizeof F1);
        for(k=1;k<=n-2&&ok;k++){
            int K=1<<k, L[256], v[8]; L[0]=0;
            for(int i=0;i<k;i++) v[i]=rnd()&((1<<m)-1);
            for(int a=1;a<K&&ok;a++){ L[a]=L[a&(a-1)]^v[__builtin_ctz(a)]; if(inim(F,k,a,L[a])) ok=0; }
            if(ok){ extend(F,k,L,G); memcpy(F,G,sizeof(int)*(2*K)); }
        }
        long double c=0;
        if(ok){ int L[256]; L[0]=0; c=cntL(F,n-1,0,L); }
        sum+=c; sum2+=c*c;
    }
    long double mean=sum/S, var=sum2/S-mean*mean, w=powl(2,logw);
    printf("estimate %.6Le  +- %.2Le (rel %.4Lf)\n", w*mean, w*sqrtl(var/S), sqrtl(var/S)/mean);
    return 0;
}
