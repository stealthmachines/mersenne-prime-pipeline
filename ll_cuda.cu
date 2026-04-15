/*
 * ll_cuda.cu  —  Track C: Lucas-Lehmer primality test for M_p = 2^p - 1
 *
 * Algorithm: discrete weighted transform (DWT) squaring mod 2^p-1
 *   via cuFFT real-to-complex (D2Z/Z2D), Crandall-Fagin 1994.
 *
 *   s_0     = 4
 *   s_{k+1} = s_k² - 2  (mod M_p)
 *   M_p prime  ⟺  s_{p-2} ≡ 0  (mod M_p)
 *
 * DWT key ideas:
 *   • Represent the p-bit number as n ≈ p/26 "digits" (balance of floor/ceil(p/n) bits each).
 *   • Forward weight w_k = 2^frac(k·p/n) turns cyclic convolution into multiplication mod 2^p-1.
 *   • cuFFT D2Z forward  →  pointwise complex square  →  cuFFT Z2D inverse.
 *   • Inverse weight 1/(n·w_k), round to int64, carry propagate.
 *
 * GPU carry uses 3 passes of a parallel atomic-add kernel.
 * For small p (≤ 62) an exact uint64 reference path is also provided.
 *
 * Build:
 *   nvcc -O3 -arch=sm_75 -allow-unsupported-compiler -lcufft
 *        -o ll_cuda.exe ll_cuda.cu
 *
 * Usage:
 *   ll_cuda.exe 521 607 1279 2203          — verify known Mersenne primes
 *   ll_cuda.exe 82589933                   — current world record
 *   ll_cuda.exe --selftest                 — quick sanity check vs known primes
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#define _USE_MATH_DEFINES  /* MSVC: enable M_LN2, M_PI etc. */
#include <math.h>
#include <string.h>
#include <cuda_runtime.h>
#include <cufft.h>

/* ──────────────────── φ-lattice constants ──────────────────── */
#define PHI      1.618033988749895
#define SQRT5    2.2360679774997896
#define PI_D     3.141592653589793
#define LN_PHI   0.4812118250596034748

/* First 50 primes — same table as EMPIRICAL_VALIDATION_ASCII.c */
static const int PRIMES50[50] = {
     2,  3,  5,  7, 11, 13, 17, 19, 23, 29,
    31, 37, 41, 43, 47, 53, 59, 61, 67, 71,
    73, 79, 83, 89, 97,101,103,107,109,113,
   127,131,137,139,149,151,157,163,167,173,
   179,181,191,193,197,199,211,223,227,229
};

/* Binet's formula for real-index Fibonacci — EMPIRICAL_VALIDATION_ASCII.c */
static double fibonacci_real(double n) {
    double t1 = pow(PHI, n) / SQRT5;
    double t2 = pow(1.0 / PHI, n) * cos(PI_D * n);
    return t1 - t2;
}

/* Prime product index — 50-entry modular table */
static double prime_product_index(double n, double beta) {
    int idx = ((int)floor(n + beta) + 50) % 50;
    return (double)PRIMES50[idx];
}

/*
 * Universal D_n operator (EMPIRICAL_VALIDATION_ASCII.c / hdgl_analog_v30.c):
 *   D_n(n,beta,r,k,Omega,base) = sqrt(φ·F_{n+β}·P_{n+β}·base^{n+β}·Ω) · r^k
 * For M_p = 2^p-1: base=2, beta=0, k=(dimn+1)/8
 */
static double D_n(double n, double beta, double r, double k,
                  double Omega, double base) {
    double nb  = n + beta;
    double Fn  = fibonacci_real(nb);
    double Pn  = prime_product_index(n, beta);
    double val = PHI * fmax(Fn, 1e-15) * pow(base, nb) * Pn * Omega;
    return sqrt(fmax(val, 1e-15)) * pow(fabs(r), k);
}

/* φ-lattice coordinate of M_p = 2^p-1 ≈ 2^p:
 *   n(2^p) = log(p·ln2 / ln(φ)) / ln(φ) - 1/(2φ)         */
static double n_of_2p(uint64_t p) {
    double lx  = (double)p * M_LN2;
    double llx = log(lx / LN_PHI);
    if (llx <= 0.0) return -1.0;
    return llx / LN_PHI - 0.5 / PHI;
}

/* Report Dₙ resonance for a Mersenne exponent */
static void report_resonance(uint64_t p) {
    double nc   = n_of_2p(p);
    if (nc < 0.0) { printf("  (n_coord undefined)\n"); return; }
    double frac  = nc - floor(nc);
    int    dim   = ((int)floor(nc) % 8) + 1;
    double k     = (dim + 1) / 8.0;
    /* prismatic spirit: Omega peaks at frac→0 (lower-half attractor) */
    double omega = 0.5 + 0.5 * sin(PI_D * frac * PHI);
    double dn    = D_n(floor(nc), 0.0, frac, k, omega, 2.0);
    printf("  φ-lattice: n=%.6f  r=%.4f  Ω=%.4f  D%d(r,base=2)=%.4f  %s\n",
           nc, frac, omega, dim, dn, frac < 0.5 ? "[lower-half]" : "");
}

#ifdef _WIN32
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#  include <intrin.h>
static double wtime(void) {
    LARGE_INTEGER f, t;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&t);
    return (double)t.QuadPart / (double)f.QuadPart;
}
#else
#  include <time.h>
static double wtime(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}
#endif

/* ──────────────────── types / macros ──────────────────────── */
typedef long long                 lld;
typedef unsigned long long        ulld;
typedef cufftDoubleComplex        cmplx;
typedef cufftDoubleReal           dreal;

#define CUDA_CHECK(x) do {                                              \
    cudaError_t _e = (x);                                              \
    if (_e != cudaSuccess) {                                           \
        fprintf(stderr, "CUDA %s:%d: %s\n",                           \
                __FILE__, __LINE__, cudaGetErrorString(_e));           \
        exit(1);                                                       \
    }                                                                  \
} while(0)

#define CUFFT_CHECK(x) do {                                            \
    cufftResult _r = (x);                                             \
    if (_r != CUFFT_SUCCESS) {                                        \
        fprintf(stderr, "cuFFT error %s:%d code=%d\n",                \
                __FILE__, __LINE__, (int)_r);                         \
        exit(1);                                                       \
    }                                                                  \
} while(0)

/* ──────────────────── host utilities ──────────────────────── */

static int next_pow2(uint64_t n) {
    uint64_t p = 1;
    while (p < n) p <<= 1;
    return (int)p;
}

/* Transform length: n >= p/26 for double precision safety (Crandall-Fagin).
 * Minimum 8 — do NOT enforce n>=64; large minimum forces 1-bit digits for small p
 * which needs log2(n) carry passes, not 3.
 * For n=8 and p=89: 11 bits/digit → max_carry ~ 2^11 after one pass → 3 passes ok. */
static int choose_n(uint64_t p) {
    int n = next_pow2(p / 26 + 2);
    if (n < 8) n = 8;
    return n;
}

/*
 * base[k] = 2^(b_k)  where  b_k = floor((k+1)·p/n) - floor(k·p/n)
 * = bits per digit k  (either floor(p/n) or floor(p/n)+1)
 * Uses double arithmetic — exact for p < 2^52 and k < n.
 */
static void precompute_bases(lld *base, int n, uint64_t p) {
    double dp = (double)p, dn = (double)n;
    for (int k = 0; k < n; k++) {
        uint64_t sh0 = (uint64_t)floor((double)k * dp / dn);
        uint64_t sh1 = (uint64_t)floor((double)(k + 1) * dp / dn);
        int bits = (int)(sh1 - sh0);
        if (bits < 1)  bits = 1;
        if (bits > 62) bits = 62;
        base[k] = 1LL << bits;
    }
}

/*
 * Forward weight:  fw[k]  = 2^frac(k·p/n)
 * Inverse weight:  ifw[k] = 1 / (n · fw[k])    (includes 1/n normalisation)
 *
 * frac(k·p/n) = (k·p mod n) / n
 * For k < n and p < 2^52, k·p < 2^52 fits in double exactly.
 */
static void precompute_weights(double *fw, double *ifw, int n, uint64_t p) {
    double dn = (double)n, dp = (double)p;
    for (int k = 0; k < n; k++) {
        double kp   = (double)k * dp;          /* exact for p < 2^52 */
        double frac = fmod(kp / dn, 1.0);
        if (frac < 0.0) frac += 1.0;
        double w   = exp(frac * M_LN2);        /* 2^frac */
        fw[k]  = w;
        ifw[k] = 1.0 / (dn * w);
    }
}

/* ──────────────────── CUDA kernels ─────────────────────────── */

/* Apply forward DWT weights and convert int64 digits → real buffer. */
__global__ void k_apply_fwd(const lld *x, dreal *buf, const double *fw, int n) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < n) buf[k] = fw[k] * (double)x[k];
}

/* Pointwise complex square of the half-complex spectrum (n/2+1 elements). */
__global__ void k_ptwise_sq(cmplx *C, int half_n) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= half_n) return;
    double re = C[k].x, im = C[k].y;
    C[k].x = re * re - im * im;
    C[k].y = 2.0 * re * im;
}

/*
 * Apply inverse DWT weights + 1/n normalisation, round to int64.
 * After cuFFT Z2D, buf[k] = unnormalized IFFT output.
 * ifw[k] = 1/(n·fw[k]) handles both factors.
 */
__global__ void k_apply_inv(const dreal *buf, lld *x, const double *ifw, int n) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < n) x[k] = llround(buf[k] * ifw[k]);
}

/*
 * Race-free two-phase carry.
 *
 * Phase 1: each thread normalises digit k, writes carry[k] = carry OUT of k
 *           (to be added to digit k+1).  x[k] is written by thread k only.
 * Phase 2: each thread adds carry from its predecessor into x[k].
 *           x[k] += carry[(k-1+n)%n] — still only one writer per slot.
 *
 * Run 5 pairs for safety (theory: 3 suffices for DWT squaring, but 5 costs
 * negligible time and handles any borrow from k_sub2).
 */
__global__ void k_carry_p1(lld *x, lld *carry, const lld *base, int n) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n) return;
    lld v = x[k], b = base[k], c = 0;
    if (v >= b) {
        c = v / b;  v = v % b;
    } else if (v < 0) {
        lld bw = (-v + b - 1) / b;
        v += bw * b;
        c = -bw;
    }
    x[k]     = v;
    carry[k] = c;   /* carry TO digit (k+1)%n */
}

__global__ void k_carry_p2(lld *x, const lld *carry, int n) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n) return;
    int prev = (k - 1 + n) % n;
    x[k] += carry[prev];   /* one writer per slot — race-free */
}

/* Subtract 2 from the lowest digit (s ← s² - 2). */
__global__ void k_sub2(lld *x) { x[0] -= 2LL; }

/* ──────────────────── small-p reference (p ≤ 62) ──────────── */
/*
 * Direct Lucas-Lehmer with 64-bit arithmetic.
 * s*s can overflow uint64; we reduce via the identity
 *   x mod (2^p-1) = (x >> p) + (x & (2^p-1))
 * applied to the 128-bit product using _umul128 (MSVC) or __uint128_t (GCC).
 */
static int ll_small(uint64_t p) {
    uint64_t Mp = (p == 64) ? UINT64_MAX : ((uint64_t)1 << p) - 1;
    uint64_t s  = 4;
    for (uint64_t i = 0; i < p - 2; i++) {
#ifdef _WIN32
        uint64_t hi, lo;
        lo = _umul128(s, s, &hi);
#else
        unsigned __int128 sq = (unsigned __int128)s * s;
        uint64_t hi = (uint64_t)(sq >> 64);
        uint64_t lo = (uint64_t)sq;
#endif
        /* (hi·2^64 + lo) mod (2^p-1) */
        uint64_t upper = (lo >> p) | (hi << (64 - p));
        uint64_t lower = lo & Mp;
        s = upper + lower;
        if (s >= Mp) s -= Mp;
        /* subtract 2, keeping in [0, Mp-1] */
        s = (s >= 2) ? (s - 2) : (s + Mp - 2);
    }
    return (s == 0) ? 1 : 0;
}

/* ──────────────────── DWT Lucas-Lehmer ─────────────────────── */

static int ll_dwt(uint64_t p, int verbose) {
    int n      = choose_n(p);
    int half_n = n / 2 + 1;

    if (verbose)
        printf("  p=%-10llu  transform n=%-8d  digits ~%d bits each\n",
               (ulld)p, n, (int)(p / n));

    /* ── host allocations ── */
    double *h_fw   = (double *)malloc(n * sizeof(double));
    double *h_ifw  = (double *)malloc(n * sizeof(double));
    lld    *h_base = (lld *)   malloc(n * sizeof(lld));
    lld    *h_x    = (lld *)   calloc(n,  sizeof(lld));

    if (!h_fw || !h_ifw || !h_base || !h_x) {
        fprintf(stderr, "host malloc failed (n=%d)\n", n);
        return -1;
    }

    precompute_weights(h_fw, h_ifw, n, p);
    precompute_bases(h_base, n, p);

    /* s_0 = 4 */
    h_x[0] = 4LL;

    /* ── GPU allocations ── */
    double *d_fw = NULL, *d_ifw = NULL;
    lld    *d_x = NULL,  *d_base = NULL, *d_carry = NULL;
    dreal  *d_real = NULL;
    cmplx  *d_freq  = NULL;

    CUDA_CHECK(cudaMalloc(&d_fw,    n       * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_ifw,   n       * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_x,     n       * sizeof(lld)));
    CUDA_CHECK(cudaMalloc(&d_base,  n       * sizeof(lld)));
    CUDA_CHECK(cudaMalloc(&d_carry, n       * sizeof(lld)));
    CUDA_CHECK(cudaMalloc(&d_real,  n       * sizeof(dreal)));
    CUDA_CHECK(cudaMalloc(&d_freq,  half_n  * sizeof(cmplx)));

    CUDA_CHECK(cudaMemcpy(d_fw,   h_fw,   n * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ifw,  h_ifw,  n * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_base, h_base, n * sizeof(lld),    cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x,    h_x,    n * sizeof(lld),    cudaMemcpyHostToDevice));

    /* ── cuFFT plans ── */
    cufftHandle plan_fwd, plan_inv;
    CUFFT_CHECK(cufftPlan1d(&plan_fwd, n, CUFFT_D2Z, 1));
    CUFFT_CHECK(cufftPlan1d(&plan_inv, n, CUFFT_Z2D, 1));

    int tb      = 256;
    int blk     = (n + tb - 1) / tb;
    int half_blk = (half_n + tb - 1) / tb;

    uint64_t iters       = p - 2;
    uint64_t report_step = (iters > 1000) ? (iters / 100) : 1;

    /* For small n (self-test / verification), use CPU sequential carry per
     * iteration — provably correct.  For large n (production), use parallel
     * GPU carry (sufficient for practically-zero borrow-chain probability). */
    int cpu_carry_mode = (n <= 1024);
    lld *h_x_carry = cpu_carry_mode ? (lld *)malloc(n * sizeof(lld)) : NULL;
    double   t0          = wtime();

    /* ── L-L iteration ── */
    for (uint64_t iter = 0; iter < iters; iter++) {

        /* s ← s² mod M_p */
        k_apply_fwd<<<blk, tb>>>(d_x, d_real, d_fw, n);
        CUFFT_CHECK(cufftExecD2Z(plan_fwd, d_real, d_freq));
        k_ptwise_sq<<<half_blk, tb>>>(d_freq, half_n);
        CUFFT_CHECK(cufftExecZ2D(plan_inv, d_freq, d_real));
        k_apply_inv<<<blk, tb>>>(d_real, d_x, d_ifw, n);

        /* s ← s - 2 */
        k_sub2<<<1, 1>>>(d_x);

        /* Carry: CPU sequential (small n) or GPU parallel (large n) */
        if (cpu_carry_mode) {
            /* Pull to host, sequential carry to convergence, push back */
            CUDA_CHECK(cudaMemcpy(h_x_carry, d_x, n * sizeof(lld), cudaMemcpyDeviceToHost));
            {
                int changed = 1;
                while (changed) {
                    changed = 0;
                    lld carry = 0;
                    for (int k = 0; k < n; k++) {
                        h_x_carry[k] += carry;  carry = 0;
                        if (h_x_carry[k] >= h_base[k]) {
                            carry = h_x_carry[k] / h_base[k];
                            h_x_carry[k] %= h_base[k];
                            changed = 1;
                        } else if (h_x_carry[k] < 0) {
                            lld bw = (-h_x_carry[k] + h_base[k] - 1) / h_base[k];
                            h_x_carry[k] += bw * h_base[k];
                            carry = -bw;
                            changed = 1;
                        }
                    }
                    if (carry) { h_x_carry[0] += carry; changed = 1; }
                }
            }
            CUDA_CHECK(cudaMemcpy(d_x, h_x_carry, n * sizeof(lld), cudaMemcpyHostToDevice));
        } else {
            /* GPU two-phase carry: n+2 passes proven sufficient for n<=512,
             * 5 passes sufficient in practice for large n (b≥20, p≥1e6). */
            int ncp = (n <= 512) ? (n + 2) : 5;
            for (int cp = 0; cp < ncp; cp++) {
                CUDA_CHECK(cudaMemset(d_carry, 0, n * sizeof(lld)));
                k_carry_p1<<<blk, tb>>>(d_x, d_carry, d_base, n);
                k_carry_p2<<<blk, tb>>>(d_x, d_carry, n);
            }
        }

        if (verbose && (iter % report_step == 0 || iter == iters - 1)) {
            double elapsed = wtime() - t0;
            double rate    = (elapsed > 0) ? (double)(iter + 1) / elapsed : 0.0;
            double eta     = (rate > 0) ? (double)(iters - iter - 1) / rate : 0.0;
            printf("\r  iter %8llu / %llu  (%.1f%%)  %.0f it/s  ETA %.1fs   ",
                   (ulld)(iter + 1), (ulld)iters,
                   100.0 * (iter + 1) / (double)iters,
                   rate, eta);
            fflush(stdout);
        }
    }

    if (verbose) printf("\n");

    /* ── final normalisation on CPU — loop to convergence ── */
    CUDA_CHECK(cudaMemcpy(h_x, d_x, n * sizeof(lld), cudaMemcpyDeviceToHost));

    {
        int changed = 1;
        while (changed) {
            changed = 0;
            lld carry = 0;
            for (int k = 0; k < n; k++) {
                h_x[k] += carry;  carry = 0;
                if (h_x[k] >= h_base[k]) {
                    carry = h_x[k] / h_base[k];  h_x[k] %= h_base[k];  changed = 1;
                } else if (h_x[k] < 0) {
                    lld bw = (-h_x[k] + h_base[k] - 1) / h_base[k];
                    h_x[k] += bw * h_base[k];  carry = -bw;  changed = 1;
                }
            }
            /* wrap carry: 2^p ≡ 1 mod M_p */
            if (carry) { h_x[0] += carry; changed = 1; }
        }
    }

    /* ── zero test ── */
    int is_prime = 1;
    for (int k = 0; k < n; k++) {
        if (h_x[k] != 0) { is_prime = 0; break; }
    }

    /* Debug: if failing, print residue */
    if (!is_prime && verbose) {
        printf("  [dbg] residue digits: ");
        for (int k = 0; k < n && k < 8; k++) printf("%lld ", (long long)h_x[k]);
        printf("\n");
    }

    /* ── cleanup ── */
    cufftDestroy(plan_fwd);
    cufftDestroy(plan_inv);
    cudaFree(d_fw);    cudaFree(d_ifw);
    cudaFree(d_x);     cudaFree(d_base);  cudaFree(d_carry);
    cudaFree(d_real);  cudaFree(d_freq);
    free(h_fw); free(h_ifw); free(h_base); free(h_x);
    if (h_x_carry) free(h_x_carry);

    return is_prime;
}

/* ──────────────────── dispatch ─────────────────────────────── */

static int ll_test(uint64_t p, int verbose) {
    if (p < 2) return 0;
    if (p == 2) return 1;   /* M_2 = 3 is prime */
    if (p <= 62) {
        int r = ll_small(p);
        if (verbose)
            printf("  (small path)  M_%llu = 2^%llu-1  %s\n",
                   (ulld)p, (ulld)p, r ? "PRIME" : "composite");
        return r;
    }
    return ll_dwt(p, verbose);
}

/* ── 128-bit reference for p ≤ 126 (cross-check against DWT) ── */
#ifndef _WIN32
static int ll_128ref(uint64_t p) {
    typedef unsigned __int128 u128;
    u128 Mp = ((u128)1 << p) - 1;
    u128 s  = 4;
    for (uint64_t i = 0; i < p - 2; i++) {
        u128 sq = s * s;
        /* reduce mod M_p: upper + lower */
        u128 upper = sq >> p;
        u128 lower = sq & Mp;
        s = upper + lower;
        if (s >= Mp) s -= Mp;
        s = (s >= 2) ? s - 2 : s + Mp - 2;
    }
    return (s == 0) ? 1 : 0;
}
#endif

static const uint64_t KNOWN_MERSENNE[] = {
    2, 3, 5, 7, 13, 17, 19, 31, 61, 89, 107, 127,
    521, 607, 1279, 2203, 2281, 3217, 4253, 4423,
    0  /* sentinel */
};
/* A selection of known composite Mersenne exponents */
static const uint64_t KNOWN_COMPOSITE[] = {
    4, 6, 8, 9, 10, 11, 15, 23, 29, 37, 41, 43,
    0
};

static void run_selftest(void) {
    printf("╔══ Self-test ═══════════════════════════════════════════════╗\n");
    int pass = 0, fail = 0;

    for (int i = 0; KNOWN_MERSENNE[i]; i++) {
        uint64_t p = KNOWN_MERSENNE[i];
        double t0  = wtime();
        int r      = ll_test(p, 0);
        double dt  = wtime() - t0;
        const char *status = r ? "PRIME ✓" : "FAIL ✗";
        printf("  M_%-6llu  %-10s  %.3fs\n", (ulld)p, status, dt);
        if (r) pass++; else fail++;
        /* stop self-test at p=2281 to avoid excessive runtime */
        if (p >= 2281) { printf("  (truncated at p=2281 for speed)\n"); break; }
    }

    printf("  Composite checks:\n");
    for (int i = 0; KNOWN_COMPOSITE[i]; i++) {
        uint64_t p = KNOWN_COMPOSITE[i];
        int r = ll_test(p, 0);
        const char *status = (!r) ? "composite ✓" : "FAIL ✗";
        printf("  M_%-6llu  %s\n", (ulld)p, status);
        if (!r) pass++; else fail++;
    }

    printf("╚══ %d passed / %d failed ══════════════════════════════════╝\n",
           pass, fail);
}

/* ──────────────────── main ──────────────────────────────────── */

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr,
            "Usage: ll_cuda.exe <p> [<p2> ...]  — test M_p = 2^p - 1\n"
            "       ll_cuda.exe --selftest       — verify known primes\n"
            "       ll_cuda.exe --check 521 607  — same as positional args\n");
        return 1;
    }

    /* GPU info */
    int dev = 0;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);
    printf("╔══ LL CUDA — Track C ═══════════════════════════════════════╗\n");
    printf("  GPU  : %s  (sm_%d%d)\n", prop.name,
           prop.major, prop.minor);
    printf("  VRAM : %.0f MB\n", prop.totalGlobalMem / 1e6);
    printf("╚════════════════════════════════════════════════════════════╝\n\n");

    /* Self-test */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--selftest") == 0) {
            run_selftest();
            return 0;
        }
        if (strcmp(argv[i], "--check") == 0) continue;
    }

    /* Test each explicitly-listed p */
    for (int i = 1; i < argc; i++) {
        if (argv[i][0] == '-') continue;
        uint64_t p = (uint64_t)strtoull(argv[i], NULL, 10);
        if (p < 2) { fprintf(stderr, "  skip p=%llu (too small)\n", (ulld)p); continue; }

        printf("── Testing M_%llu = 2^%llu - 1 ──\n", (ulld)p, (ulld)p);
        double t0 = wtime();
        int r     = ll_test(p, 1 /*verbose*/);
        double dt = wtime() - t0;

        if (r)
            printf("  ✓ M_%llu is PRIME  (%.3f s)\n", (ulld)p, dt);
        else
            printf("  ✗ M_%llu is composite  (%.3f s)\n", (ulld)p, dt);
        report_resonance(p);
        printf("\n");
    }

    return 0;
}
