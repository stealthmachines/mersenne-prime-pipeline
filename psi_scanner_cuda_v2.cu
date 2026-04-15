/*
 * psi_scanner_cuda_v2.cu -- Track A (v2, zero-free)
 * Zero-free: 80 exact zeros hardcoded, higher via Gram-point Lambert W.
 * Canonical Dn(r): dim=floor(n)%8+1, r=frac(n), k=(dim+1)/8, omega=0.5+0.5*sin(n).
 * Modes: <x1> <x2> [--mersenne]  or  --lattice <n1> <n2> <step>
 * Build: nvcc -O3 -arch=sm_75 -allow-unsupported-compiler -o psi_scanner_cuda_v2.exe psi_scanner_cuda_v2.cu
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#ifdef _WIN32
#include <windows.h>
#else
#include <time.h>
#endif

#define PASS1_B      500
#define PASS2_B      5000
#define PASS3_B      10000
#define SPIKE_THRESH 0.15
#define CONV_EPS     1e-4
#define BATCH_SIZE   (1 << 20)
#define PHI     1.6180339887498948482
#define LN_PHI  0.4812118250596034748
#define TWO_PI  6.2831853071795864769
#define INV_E   0.36787944117144232159
#define LOG2PI  1.8378770664093454836

__constant__ double d_zeros_exact[80] = {
    14.134725141734693,  21.022039638771555,  25.010857580145688,
    30.424876125859513,  32.935061587739189,  37.586178158825671,
    40.918719012147495,  43.327073280914999,  48.005150881167159,
    49.773832477672302,  52.970321477714460,  56.446247697063246,
    59.347044002602352,  60.831778524609809,  65.112544048081560,
    67.079810529494173,  69.546401711173979,  72.067157674481907,
    75.704690699083933,  77.144840068874805,  79.337375020249367,
    82.910380854160462,  84.735492981074628,  87.425274613125229,
    88.809111207634465,  92.491899270593585,  94.651344040519681,
    95.870634228245332,  98.831194218193159, 101.317851006956152,
   103.725538040478419, 105.446623052947866, 107.168611184276793,
   111.029535543169970, 111.874659177229233, 114.320220915452460,
   116.226680321519019, 118.790782866217474, 121.370125002980428,
   122.946829294236573, 124.256818554513985, 127.516683879564406,
   129.578704200821853, 131.087688531430975, 133.497737202990660,
   134.756510050820649, 138.116042054533808, 139.736208952121808,
   141.123707404415728, 143.111845808910186, 146.000982487395827,
   147.422765343849989, 150.053520421293562, 150.925257612895526,
   153.024693791188948, 156.112909294982618, 157.597591818986345,
   158.849988365204885, 161.188964138954152, 163.030709687408168,
   165.537069188392498, 167.184439971994828, 169.094515416791259,
   169.911976498590630, 173.411536520135680, 174.754191523438771,
   176.441434188575954, 178.377407776468757, 179.916484018400656,
   182.207078484665730, 184.874467848130730, 185.598783678433914,
   187.228922291882030, 189.415759393773366, 192.026656325978780,
   193.079726604550355, 195.265396680495222, 196.876481841084053,
   198.015309585175508, 201.264751178782752
};

/* Gram-point approx: t_k ~ 2*pi*k / W0(k/e), Newton 6-iter */
__device__ __forceinline__
double gram_zero_k(int k) {
    double z = (double)k * INV_E;
    double w = log(z + 1.0);
    double ew;
    ew=exp(w); w-=(w*ew-z)/(ew*(w+1.0));
    ew=exp(w); w-=(w*ew-z)/(ew*(w+1.0));
    ew=exp(w); w-=(w*ew-z)/(ew*(w+1.0));
    ew=exp(w); w-=(w*ew-z)/(ew*(w+1.0));
    ew=exp(w); w-=(w*ew-z)/(ew*(w+1.0));
    ew=exp(w); w-=(w*ew-z)/(ew*(w+1.0));
    return TWO_PI * (double)k / w;
}

__device__ __forceinline__
double zeta_zero(int k) {
    return (k < 80) ? d_zeros_exact[k] : gram_zero_k(k);
}

/* Pass 1: B=500 */
__global__
void pass1_kernel(const double * __restrict__ cands,
                  uint8_t      * __restrict__ flags, int n)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx >= n) return;
    double x=cands[idx], lx=log(x), lxm=log(x-1.0);
    double px=x, pm=x-1.0, mx=exp(0.5*lx), mm=exp(0.5*lxm);
    for (int k=0; k<PASS1_B; k++) {
        double t=zeta_zero(k), d=0.25+t*t;
        px -= 2.0*mx*(0.5*cos(t*lx)+t*sin(t*lx))/d;
        pm -= 2.0*mm*(0.5*cos(t*lxm)+t*sin(t*lxm))/d;
    }
    flags[idx] = ((px-pm) > SPIKE_THRESH*lx) ? 1 : 0;
}

/* Pass 2: B=5000 survivors */
__global__
void pass2_kernel(const double * __restrict__ cands,
                  uint8_t      * __restrict__ flags,
                  const int    * __restrict__ live, int nlive)
{
    int tid=blockIdx.x*blockDim.x+threadIdx.x;
    if (tid>=nlive) return;
    int idx=live[tid];
    double x=cands[idx], lx=log(x), lxm=log(x-1.0);
    double px=x, pm=x-1.0, mx=exp(0.5*lx), mm=exp(0.5*lxm);
    for (int k=0; k<PASS2_B; k++) {
        double t=zeta_zero(k), d=0.25+t*t;
        px -= 2.0*mx*(0.5*cos(t*lx)+t*sin(t*lx))/d;
        pm -= 2.0*mm*(0.5*cos(t*lxm)+t*sin(t*lxm))/d;
    }
    flags[idx] = ((px-pm) > SPIKE_THRESH*lx) ? 1 : 0;
}

/* Pass 3: adaptive convergence */
__global__
void pass3_kernel(const double * __restrict__ cands,
                  uint8_t      * __restrict__ flags,
                  double       * __restrict__ scores,
                  const int    * __restrict__ live, int nlive)
{
    int tid=blockIdx.x*blockDim.x+threadIdx.x;
    if (tid>=nlive) return;
    int idx=live[tid];
    double x=cands[idx];
    if (x<2.0) { flags[idx]=0; return; }
    double lx=log(x), lxm=log(x-1.0);
    double px=x, pm=x-1.0, mx=exp(0.5*lx), mm=exp(0.5*lxm);
    double prev=1e30, thresh=SPIKE_THRESH*lx;
    for (int k=0; k<PASS3_B; k++) {
        double t=zeta_zero(k), d=0.25+t*t;
        px -= 2.0*mx*(0.5*cos(t*lx)+t*sin(t*lx))/d;
        pm -= 2.0*mm*(0.5*cos(t*lxm)+t*sin(t*lxm))/d;
        if (k>=PASS2_B && (k%500)==0) {
            double dp=px-pm;
            if (fabs(dp-prev)<CONV_EPS*lx) {
                flags[idx]=(dp>thresh)?1:0; scores[idx]=dp/lx; return;
            }
            prev=dp;
        }
    }
    double dp=px-pm;
    flags[idx]=(dp>thresh)?1:0; scores[idx]=dp/lx;
}

/* Miller-Rabin */
static uint64_t mulmod64(uint64_t a, uint64_t b, uint64_t mod) {
#ifdef _WIN32
    unsigned __int64 hi, lo;
    lo=_umul128(a,b,&hi);
    double q_d=(double)hi*18446744073709551616.0+(double)lo;
    uint64_t q=(uint64_t)(q_d/(double)mod);
    uint64_t r=lo-q*mod;
    if ((int64_t)r<0) r+=mod; if (r>=mod) r-=mod; return r;
#else
    return (unsigned __int128)a*b%mod;
#endif
}
static uint64_t powmod64(uint64_t b, uint64_t e, uint64_t m) {
    uint64_t r=1; b%=m;
    while(e>0){if(e&1)r=mulmod64(r,b,m);b=mulmod64(b,b,m);e>>=1;}
    return r;
}
static int miller_rabin(uint64_t n) {
    if(n<2) return 0;
    if(n==2||n==3||n==5||n==7) return 1;
    if(n%2==0||n%3==0) return 0;
    uint64_t d=n-1; int r=0;
    while(d%2==0){d/=2;r++;}
    static const uint64_t w[]={2,3,5,7,11,13,17,19,23,29,31,37};
    for(int i=0;i<12;i++){
        uint64_t a=w[i]; if(a>=n) continue;
        uint64_t x=powmod64(a,d,n);
        if(x==1||x==n-1) continue;
        int c=1;
        for(int j=0;j<r-1;j++){x=mulmod64(x,x,n);if(x==n-1){c=0;break;}}
        if(c) return 0;
    }
    return 1;
}
static int compact(const uint8_t *f, int n, int *live) {
    int c=0; for(int i=0;i<n;i++) if(f[i]) live[c++]=i; return c;
}

/* Dn(r) canonical: dim=floor(nc)%8+1, r=frac(nc), k=(dim+1)/8, omega=0.5+0.5*sin(nc) */
static const double FIB8[8]  ={1,1,2,3,5,8,13,21};
static const double PRM8[8]  ={2,3,5,7,11,13,17,19};
static double dn_amp(double nc) {
    if(nc<0) return 0;
    double fl=floor(nc); int dim=((int)fl%8)+1;
    double r=nc-fl, k=(dim+1)/8.0;
    double om=0.5+0.5*sin(nc);
    return sqrt(PHI*FIB8[dim-1]*pow(2.0,dim)*PRM8[dim-1]*om)*pow(r+1e-9,k);
}
static double n_of_x(double x) {
    if(x<=1) return -1; double lx=log(x); if(lx<=0) return -1;
    return log(lx/LN_PHI)/LN_PHI - 0.5/PHI;
}
static double x_of_n(double n) {
    return exp(LN_PHI*exp((n+0.5/PHI)*LN_PHI));
}

int main(int argc, char **argv) {
    if(argc<3) {
        fprintf(stderr,
            "Usage:\n"
            "  psi_scanner_cuda_v2 <x1> <x2> [--mersenne]\n"
            "  psi_scanner_cuda_v2 --lattice <n1> <n2> <dn>\n");
        return 1;
    }
    int lat=0, mer=0;
    double x1=0,x2=0,ln1=0,ln2=0,ldn=0;
    if(strcmp(argv[1],"--lattice")==0) {
        if(argc<5){fprintf(stderr,"--lattice needs n1 n2 dn\n");return 1;}
        lat=1; ln1=atof(argv[2]); ln2=atof(argv[3]); ldn=atof(argv[4]);
        if(ldn<=0||ln2<=ln1){fprintf(stderr,"need dn>0 and n2>n1\n");return 1;}
        x1=x_of_n(ln1); x2=x_of_n(ln2);
    } else {
        x1=atof(argv[1]); x2=atof(argv[2]);
        mer=(argc>=4&&strcmp(argv[3],"--mersenne")==0);
        if(x1<2) x1=2; if(x2<x1){fprintf(stderr,"x2>=x1\n");return 1;}
    }
    printf("=== PSI SCANNER v2 (zero-free) ===\n");
    if(lat) printf("  n-space: %.4f..%.4f step=%.8f  x: %.3e..%.3e\n",ln1,ln2,ldn,x1,x2);
    else    printf("  x-range: %.0f..%.0f  mode=%s\n",x1,x2,mer?"mersenne":"prime");
    printf("  zeros: 80 exact + Gram Lambert-W  Dn(r): canonical\n");
    printf("  passes: B=%d->%d->%d\n==================================\n\n",PASS1_B,PASS2_B,PASS3_B);

    long long tc; double *hlat=NULL;
    if(lat) {
        long long steps=(long long)ceil((ln2-ln1)/ldn);
        if(steps<=0||steps>400000000LL){fprintf(stderr,"bad step count\n");return 1;}
        tc=steps;
        hlat=(double*)malloc(steps*sizeof(double));
        if(!hlat){fprintf(stderr,"OOM\n");return 1;}
        for(long long i=0;i<steps;i++) hlat[i]=x_of_n(ln1+(double)i*ldn);
    } else {
        tc=(long long)(x2-x1)+1;
    }

    int batch=(int)((tc<BATCH_SIZE)?tc:BATCH_SIZE);
    double  *dc; cudaMalloc(&dc,  batch*sizeof(double));
    uint8_t *df; cudaMalloc(&df,  batch*sizeof(uint8_t));
    double  *ds; cudaMalloc(&ds,  batch*sizeof(double));
    int     *dl; cudaMalloc(&dl,  batch*sizeof(int));
    double  *hc=(double*)malloc(batch*sizeof(double));
    uint8_t *hf=(uint8_t*)malloc(batch*sizeof(uint8_t));
    double  *hs=(double*)malloc(batch*sizeof(double));
    int     *hl=(int*)malloc(batch*sizeof(int));
    const int T=256;
    long long found=0,p1s=0,p2s=0;

#ifdef _WIN32
    LARGE_INTEGER freq,t0,t1; QueryPerformanceFrequency(&freq); QueryPerformanceCounter(&t0);
#else
    struct timespec t0,t1; clock_gettime(CLOCK_MONOTONIC,&t0);
#endif

    for(long long base=0;base<tc;base+=batch) {
        long long tb=tc-base; if(tb>batch) tb=batch; int nb=(int)tb;
        if(lat) memcpy(hc,hlat+base,nb*sizeof(double));
        else for(int i=0;i<nb;i++) hc[i]=x1+(double)(base+i);
        cudaMemcpy(dc,hc,nb*sizeof(double),cudaMemcpyHostToDevice);
        cudaMemset(df,0,nb*sizeof(uint8_t));
        /* P1 */
        pass1_kernel<<<(nb+T-1)/T,T>>>(dc,df,nb);
        cudaDeviceSynchronize();
        cudaMemcpy(hf,df,nb*sizeof(uint8_t),cudaMemcpyDeviceToHost);
        int n1=compact(hf,nb,hl); p1s+=n1; if(!n1) continue;
        /* P2 */
        cudaMemcpy(dl,hl,n1*sizeof(int),cudaMemcpyHostToDevice);
        cudaMemset(df,0,nb*sizeof(uint8_t));
        pass2_kernel<<<(n1+T-1)/T,T>>>(dc,df,dl,n1);
        cudaDeviceSynchronize();
        cudaMemcpy(hf,df,nb*sizeof(uint8_t),cudaMemcpyDeviceToHost);
        int n2=compact(hf,nb,hl); p2s+=n2; if(!n2) continue;
        /* P3 */
        cudaMemcpy(dl,hl,n2*sizeof(int),cudaMemcpyHostToDevice);
        cudaMemset(df,0,nb*sizeof(uint8_t)); cudaMemset(ds,0,nb*sizeof(double));
        pass3_kernel<<<(n2+T-1)/T,T>>>(dc,df,ds,dl,n2);
        cudaDeviceSynchronize();
        cudaMemcpy(hf,df,nb*sizeof(uint8_t),cudaMemcpyDeviceToHost);
        cudaMemcpy(hs,ds,nb*sizeof(double),cudaMemcpyDeviceToHost);
        /* confirm */
        for(int i=0;i<nb;i++) {
            if(!hf[i]) continue;
            uint64_t xi=(uint64_t)(hc[i]+0.5);
            if(!miller_rabin(xi)) continue;
            found++;
            double nc=n_of_x((double)xi), dn=dn_amp(nc), fr=nc-floor(nc);
            if(lat||mer)
                printf("  CANDIDATE p=%-12llu  n=%.6f  frac=%.4f  score=%.4f  Dn=%.4f%s\n",
                    (unsigned long long)xi,nc,fr,hs[i],dn,(fr<0.5)?"  [lower-half]":"");
            else if(found<=20||(uint64_t)(hc[i]+0.5)>(uint64_t)(x2-100))
                printf("  PRIME: %-14llu  n=%.6f  score=%.4f  Dn=%.4f\n",
                    (unsigned long long)xi,nc,hs[i],dn);
        }
    }

#ifdef _WIN32
    QueryPerformanceCounter(&t1);
    double el=(double)(t1.QuadPart-t0.QuadPart)/(double)freq.QuadPart;
#else
    clock_gettime(CLOCK_MONOTONIC,&t1);
    double el=(t1.tv_sec-t0.tv_sec)+(t1.tv_nsec-t0.tv_nsec)*1e-9;
#endif
    printf("\n=== RESULTS ===\n");
    printf("  candidates : %lld\n  pass1 surv : %lld\n  pass2 surv : %lld\n",tc,p1s,p2s);
    printf("  confirmed  : %lld\n  elapsed    : %.2f s\n",found,el);
    if(el>0) printf("  throughput : %.0f cand/s\n",(double)tc/el);
    printf("===============\n");
    cudaFree(dc);cudaFree(df);cudaFree(ds);cudaFree(dl);
    free(hc);free(hf);free(hs);free(hl); if(hlat) free(hlat);
    return 0;
}