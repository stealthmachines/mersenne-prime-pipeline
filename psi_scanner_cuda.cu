/*
 * psi_scanner_cuda.cu — Track A
 *
 * GPU-accelerated phi-lattice prime scanner using the Riemann explicit formula.
 *
 * Algorithm:
 *   delta_psi(x) = psi(x) - psi(x-1)
 *                = x - sum_k 2*Re(x^rho_k / rho_k) - log(2*pi)
 *   Spikes at primes (~ln p), near-zero at composites.
 *
 * Three-pass adaptive pipeline:
 *   Pass 1: B=500  zeros — cheap filter, flags spike candidates
 *   Pass 2: B=5000 zeros — on survivors only
 *   Pass 3: B=10000 zeros with convergence check — final survivors -> Miller-Rabin
 *
 * Miller-Rabin runs on CPU for confirmed candidates (numbers fit in uint64).
 *   For the Mersenne-exponent scan (p < 2^30) this is always sufficient.
 *
 * Zeta zeros loaded from zeta_zeros_10k.json at startup then copied to GPU.
 * Up to 10000 zeros stored in GPU constant memory (~80 KB, fits in L2 on Turing).
 *
 * Build:
 *   nvcc -O3 -arch=sm_75 -o psi_scanner_cuda psi_scanner_cuda.cu
 *   (sm_75 = Turing, RTX 2060)
 *
 * Usage:
 *   psi_scanner_cuda <x_start> <x_end> [--mersenne]
 *   --mersenne: scan candidate Mersenne exponents p in [x_start, x_end],
 *               report those that also score high in phi-lattice Dn field.
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

/* ── Constants ──────────────────────────────────────────────── */
#define MAX_ZEROS     10000
#define PASS1_B       500
#define PASS2_B       5000
#define PASS3_B       10000
#define SPIKE_THRESH  0.15        /* delta_psi/ln(x) > this = candidate  */
#define CONV_EPS      1e-4        /* convergence criterion pass 3  */
#define BATCH_SIZE    (1 << 20)   /* 1M candidates per GPU batch   */

#define PHI      1.6180339887498948482
#define LN_PHI   0.4812118250596034748
#define LOG10PHI 0.20901500768249601

/* ── Zeta zeros: first 5000 in constant memory (40KB), rest in global ─ */
#define CONST_ZEROS 5000
__constant__ double d_zeros_const[CONST_ZEROS];  /* 40KB fits */
__constant__ int    d_nzeros;                     /* total count */

/* ── Extended zeros in global memory (B > CONST_ZEROS) ───────── */
static double *d_zeros_global = NULL;

/* ── phi-lattice coordinate (device) ─────────────────────────── */
__device__ __forceinline__
double n_of_x_dev(double x) {
    if (x <= 1.0) return -1.0;
    double lx = log(x);
    if (lx <= 0.0) return -1.0;
    return log(lx / LN_PHI) / LN_PHI - 0.5 / PHI;
}

/* ── Single delta_psi(x) with B zeros ─────────────────────────
 *  Uses the explicit formula oscillatory sum:
 *  psi(x) ~ x - sum_k 2*|x^(0.5+it_k)| * Re(1/(0.5+it_k) * x^it_k) - log(2*pi)
 *  delta_psi(x) = psi(x) - psi(x-1), computed by differencing.
 *  For integer x, only prime x gives a genuine ln(x) spike.
 */
/* delta_psi(x,B,zeros_g):
 *  zeros 0..min(B,CONST_ZEROS)-1 from constant cache,
 *  zeros CONST_ZEROS..B-1 from global pointer zeros_g.
 *  Pass NULL for zeros_g when B <= CONST_ZEROS.
 */
__device__
double delta_psi_dev(double x, int B, const double * __restrict__ zeros_g) {
    if (x < 2.0) return 0.0;

    double lx  = log(x);
    double lxm = log(x - 1.0);
    double psi_x  = x;
    double psi_xm = x - 1.0;
    double mag_x  = exp(0.5 * lx);
    double mag_xm = exp(0.5 * lxm);

    int buse = (B < d_nzeros) ? B : d_nzeros;

    for (int k = 0; k < buse; k++) {
        double t = (k < CONST_ZEROS) ? d_zeros_const[k] : zeros_g[k];
        double denom = 0.25 + t * t;
        double re_x  = (0.5 * cos(t*lx)  + t * sin(t*lx))  / denom;
        double re_xm = (0.5 * cos(t*lxm) + t * sin(t*lxm)) / denom;
        psi_x  -= 2.0 * mag_x  * re_x;
        psi_xm -= 2.0 * mag_xm * re_xm;
    }

    psi_x  -= 1.8378770664093454836;
    psi_xm -= 1.8378770664093454836;
    return psi_x - psi_xm;
}

/* ── Pass 1 kernel: B=500, all from constant cache ───────────── */
__global__
void pass1_kernel(const double * __restrict__ candidates,
                  uint8_t      * __restrict__ flags,
                  int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    double dp = delta_psi_dev(candidates[idx], PASS1_B, NULL);
    flags[idx] = (dp > SPIKE_THRESH * log(candidates[idx])) ? 1 : 0;
}

/* ── Pass 2 kernel: B=5000, survivors only ───────────────────── */
__global__
void pass2_kernel(const double * __restrict__ candidates,
                  uint8_t      * __restrict__ flags,
                  const int    * __restrict__ live_idx,
                  int n_live,
                  const double * __restrict__ zeros_g)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_live) return;
    int idx   = live_idx[tid];
    double x  = candidates[idx];
    double dp = delta_psi_dev(x, PASS2_B, zeros_g);
    flags[idx] = (dp > SPIKE_THRESH * log(x)) ? 1 : 0;
}

/* ── Pass 3 kernel: adaptive convergence ────────────────────── */
__global__
void pass3_kernel(const double * __restrict__ candidates,
                  uint8_t      * __restrict__ flags,
                  double       * __restrict__ scores,
                  const int    * __restrict__ live_idx,
                  int n_live,
                  const double * __restrict__ zeros_g)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_live) return;
    int idx   = live_idx[tid];
    double x  = candidates[idx];

    if (x < 2.0) { flags[idx] = 0; return; }

    double lx  = log(x);
    double lxm = log(x - 1.0);
    double mag_x  = exp(0.5 * lx);
    double mag_xm = exp(0.5 * lxm);
    double psi_x  = x;
    double psi_xm = x - 1.0;
    double prev_dp = 1e30;

    double thresh = SPIKE_THRESH * lx;
    for (int k = 0; k < d_nzeros; k++) {
        double t = (k < CONST_ZEROS) ? d_zeros_const[k] : zeros_g[k];
        double denom = 0.25 + t * t;
        double re_x  = (0.5 * cos(t*lx)  + t * sin(t*lx))  / denom;
        double re_xm = (0.5 * cos(t*lxm) + t * sin(t*lxm)) / denom;
        psi_x  -= 2.0 * mag_x  * re_x;
        psi_xm -= 2.0 * mag_xm * re_xm;

        if (k >= PASS2_B && (k % 500) == 0) {
            double dp = psi_x - psi_xm - 1.8378770664093454836;
            if (fabs(dp - prev_dp) < CONV_EPS * lx) {
                flags[idx]  = (dp > thresh) ? 1 : 0;
                scores[idx] = dp / lx;
                return;
            }
            prev_dp = dp;
        }
    }

    double dp   = psi_x - psi_xm - 1.8378770664093454836;
    flags[idx]  = (dp > thresh) ? 1 : 0;
    scores[idx] = dp / lx;
}

/* ── CPU: Miller-Rabin for uint64 ────────────────────────────────
 * MSVC has no __int128. Use carry-split 64-bit mulmod:
 *   a*b mod m using (hi,lo) via _umul128 / manual split.
 * For n < 2^63, we can use double-precision for the quotient.
 */
static uint64_t mulmod64(uint64_t a, uint64_t b, uint64_t mod) {
    /* Use 128-bit intermediate via compiler intrinsic on MSVC x64 */
    unsigned __int64 hi, lo;
#ifdef _WIN32
    lo = _umul128(a, b, &hi);
    /* hi:lo / mod -> quotient q, remainder = hi:lo - q*mod */
    /* Use __udiv128 if available (VS 2019+), else use double approx */
    /* Double approximation is exact for mod < 2^52 (all our primes) */
    double q_d = (double)hi * 18446744073709551616.0 + (double)lo;
    uint64_t q = (uint64_t)(q_d / (double)mod);
    uint64_t r = lo - q * mod;
    /* Correct for rounding errors (+/- 1) */
    if ((int64_t)r < 0)          r += mod;
    if (r >= mod)                r -= mod;
    return r;
#else
    return (unsigned __int128)a * b % mod;
#endif
}
static uint64_t powmod64(uint64_t base, uint64_t exp, uint64_t mod) {
    uint64_t r = 1; base %= mod;
    while (exp > 0) {
        if (exp & 1) r = mulmod64(r, base, mod);
        base = mulmod64(base, base, mod);
        exp >>= 1;
    }
    return r;
}
static int miller_rabin(uint64_t n) {
    if (n < 2) return 0;
    if (n == 2 || n == 3 || n == 5 || n == 7) return 1;
    if (n % 2 == 0 || n % 3 == 0) return 0;
    uint64_t d = n - 1; int r = 0;
    while (d % 2 == 0) { d /= 2; r++; }
    static const uint64_t wit[] = {2,3,5,7,11,13,17,19,23,29,31,37};
    for (int i = 0; i < 12; i++) {
        uint64_t a = wit[i];
        if (a >= n) continue;
        uint64_t x = powmod64(a, d, n);
        if (x == 1 || x == n - 1) continue;
        int composite = 1;
        for (int j = 0; j < r - 1; j++) {
            x = mulmod64(x, x, n);
            if (x == n - 1) { composite = 0; break; }
        }
        if (composite) return 0;
    }
    return 1;
}

/* ── Compact JSON array parser for zeta zeros ────────────────── */
static int load_zeros(const char *path, double *out, int maxn) {
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); return 0; }
    int n = 0;
    int c;
    /* skip opening '[' */
    while ((c = fgetc(f)) != EOF && c != '[');
    while (n < maxn) {
        double v;
        if (fscanf(f, " %lf", &v) != 1) break;
        out[n++] = v;
        /* skip comma or closing ']' */
        while ((c = fgetc(f)) != EOF && c != ',' && c != ']');
        if (c == ']') break;
    }
    fclose(f);
    return n;
}

/* ── Prefix-sum compaction: build live_idx from flags ─────────── */
static int compact(const uint8_t *flags, int n, int *live_idx) {
    int cnt = 0;
    for (int i = 0; i < n; i++)
        if (flags[i]) live_idx[cnt++] = i;
    return cnt;
}

/* ── Dn(r) — exact formula from hdgl_analog_v30.c ───────────────
 *  D_n(r) = sqrt(phi * F_n * 2^n * P_n * omega) * |r|^k
 *  where k = (n+1)/8, n in [1..8], r = frac(n_coord).
 *  omega = 1.0 (lattice bootloader default).
 */
static const uint64_t FIB_TAB[8]   = {1,1,2,3,5,8,13,21};
static const uint64_t PRIME_TAB[8] = {2,3,5,7,11,13,17,19};

static double compute_Dn_r(int n, double r, double omega) {
    if (n < 1 || n > 8) return 0.0;
    int idx = n - 1;
    double k    = (double)(n + 1) / 8.0;
    double base = sqrt(PHI * FIB_TAB[idx] * pow(2.0, n)
                       * PRIME_TAB[idx] * omega);
    return base * pow(fabs(r), k);
}

/* Inverse: x such that n_of_x(x) == n_coord.
 *   n(x) = log_phi(log_phi(x)) - 1/(2*phi)
 *   => log_phi(x) = phi^(n + 1/(2*phi))
 *   => x = phi^(phi^(n + 1/(2*phi)))
 * Only valid for n_coord > ~2 (x grows doubly-exponentially).
 */
static double x_of_n(double n_coord) {
    double inner = pow(PHI, n_coord + 0.5 / PHI);
    if (inner > 709.0) return -1.0;   /* would overflow exp */
    return pow(PHI, inner);
}

/* ── Dn-guided window list ────────────────────────────────────────
 *  Walk n-space from n_lo to n_hi in steps of n_step.
 *  At each lattice point compute Dn(r) where r=frac(n).
 *  Keep windows where Dn exceeds dn_thresh.
 *  Return sorted (by Dn desc) array of [window_lo, window_hi] pairs.
 *  window half-width = max(50, ln(x)).
 *
 *  The strategy is purely acoustic: high Dn means the lattice is
 *  resonating strongly at this n — probable prime cluster nearby.
 */
#define MAX_WINDOWS 4096
typedef struct { double lo; double hi; double dn; double n_center; } Window;

static int cmp_window_dn(const void *a, const void *b) {
    double da = ((const Window *)a)->dn;
    double db = ((const Window *)b)->dn;
    return (da < db) - (da > db);   /* descending */
}

static int build_guided_windows(double n_lo, double n_hi, double n_step,
                                 double dn_thresh, Window *out, int max_w)
{
    int nw = 0;
    for (double n = n_lo; n <= n_hi && nw < max_w; n += n_step) {
        double frac = n - floor(n);
        int    dim  = ((int)floor(n) % 8) + 1;
        if (dim < 1) dim = 1;
        double dn = compute_Dn_r(dim, frac, 1.0);
        if (dn < dn_thresh) continue;

        double xc = x_of_n(n);
        if (xc < 2.0) continue;

        double half = log(xc);
        if (half < 50.0) half = 50.0;
        double lo = floor(xc) - half;
        double hi = floor(xc) + half;
        if (lo < 2.0) lo = 2.0;

        out[nw].lo       = lo;
        out[nw].hi       = hi;
        out[nw].dn       = dn;
        out[nw].n_center = n;
        nw++;
    }
    qsort(out, nw, sizeof(Window), cmp_window_dn);
    return nw;
}

/* ── Main ─────────────────────────────────────────────────────── */
int main(int argc, char **argv) {

    if (argc < 3) {
        fprintf(stderr,
            "Usage: psi_scanner_cuda <x_start> <x_end> [--mersenne] [--guided]\n"
            "  --mersenne  treat range as candidate Mersenne exponents p\n"
            "  --guided    interpret range as n-space [n_lo, n_hi]; step=0.01\n"
            "              opens GPU windows only where Dn(r) resonates\n");
        return 1;
    }

    double x_start  = atof(argv[1]);
    double x_end    = atof(argv[2]);
    int mersenne_mode = 0, guided_mode = 0;
    for (int i = 3; i < argc; i++) {
        if (strcmp(argv[i], "--mersenne") == 0) mersenne_mode = 1;
        if (strcmp(argv[i], "--guided")   == 0) guided_mode   = 1;
    }

    if (x_start < 2.0) x_start = 2.0;
    if (x_end < x_start) { fprintf(stderr, "x_end must be >= x_start\n"); return 1; }

    long long total_range = (long long)(x_end - x_start) + 1;

    printf("╔══ PSI SCANNER CUDA — Track A ══════════════════════════════╗\n");
    printf("  Range  : %.0f to %.0f  (%lld candidates)\n",
           x_start, x_end, total_range);
    const char *mode_str = guided_mode   ? "Dn-guided lattice scan" :
                           mersenne_mode ? "Mersenne exponent scan" :
                                           "General prime scan";
    printf("  Mode   : %s\n", mode_str);
    printf("  Passes : B=500 -> B=5000 -> B=10000 (adaptive convergence)\n");
    printf("╚════════════════════════════════════════════════════════════╝\n\n");

    /* ── Load zeta zeros ───────────────────────────────────────── */
    static double h_zeros[MAX_ZEROS];
    int nz = load_zeros("zeta_zeros_10k.json", h_zeros, MAX_ZEROS);
    if (nz < PASS1_B) {
        fprintf(stderr, "Need at least %d zeros, got %d\n", PASS1_B, nz);
        return 1;
    }
    printf("  Loaded %d zeta zeros\n", nz);

    /* Copy first CONST_ZEROS to constant memory, all to global memory */
    int nconst = (nz < CONST_ZEROS) ? nz : CONST_ZEROS;
    cudaMemcpyToSymbol(d_zeros_const, h_zeros, nconst * sizeof(double));
    cudaMemcpyToSymbol(d_nzeros, &nz, sizeof(int));
    /* Allocate and fill global zeros array (full 10k) */
    cudaMalloc(&d_zeros_global, nz * sizeof(double));
    cudaMemcpy(d_zeros_global, h_zeros, nz * sizeof(double), cudaMemcpyHostToDevice);

    /* ── GPU buffers ───────────────────────────────────────────── */
    int   batch = (total_range < BATCH_SIZE) ? (int)total_range : BATCH_SIZE;
    double  *d_cand;   cudaMalloc(&d_cand,   batch * sizeof(double));
    uint8_t *d_flags;  cudaMalloc(&d_flags,  batch * sizeof(uint8_t));
    double  *d_scores; cudaMalloc(&d_scores, batch * sizeof(double));
    int     *d_live;   cudaMalloc(&d_live,   batch * sizeof(int));

    double  *h_cand    = (double *)malloc(batch * sizeof(double));
    uint8_t *h_flags   = (uint8_t*)malloc(batch * sizeof(uint8_t));
    double  *h_scores  = (double *)malloc(batch * sizeof(double));
    int     *h_live    = (int    *)malloc(batch * sizeof(int));
    int     *d_live_h  = (int    *)malloc(batch * sizeof(int)); /* host compaction */

    int threads = 256;
    long long total_found = 0, pass1_survivors = 0, pass2_survivors = 0;

#ifdef _WIN32
    LARGE_INTEGER freq, t0, t1;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&t0);
#else
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
#endif

    /* ── Batch loop ────────────────────────────────────────────── */

    /* Declare all vars that a goto-cleanup might bypass */
    long long loop_end  = total_range;
    long long loop_step = (long long)batch;
    double    elapsed   = 0.0;

    /* In guided mode: x_start/x_end are n-space coordinates.
     * Build lattice windows sorted by Dn(r) descending, then
     * run the GPU pipeline on each window instead of a uniform sweep. */
    static Window guided_windows[MAX_WINDOWS];
    int n_guided_windows = 0;
    if (guided_mode) {
        double dn_thresh = 1.0;   /* only open windows where Dn > 1 */
        n_guided_windows = build_guided_windows(
            x_start, x_end, 0.01, dn_thresh,
            guided_windows, MAX_WINDOWS);
        printf("  Lattice windows (Dn > %.1f): %d\n\n", dn_thresh, n_guided_windows);
        if (n_guided_windows == 0) {
            fprintf(stderr, "No resonant windows found in n=[%.2f,%.2f]\n",
                    x_start, x_end);
            goto cleanup;
        }
    }

    /* Update loop control after guided windows are built */
    loop_end  = guided_mode ? (long long)n_guided_windows : total_range;
    loop_step = guided_mode ? 1LL : (long long)batch;

    for (long long base = 0; base < loop_end; base += loop_step) {
        long long this_batch = guided_mode ? 1LL : (total_range - base);
        if (!guided_mode && this_batch > batch) this_batch = batch;
        int nb = guided_mode ? 0 : (int)this_batch;  /* guided sets nb below */

        /* Fill candidate array */
        if (guided_mode) {
            /* base is the window index here, not an offset */
            if (base >= n_guided_windows) break;
            Window *w = &guided_windows[base];
            long long w_len = (long long)(w->hi - w->lo) + 1;
            nb = (int)(w_len < batch ? w_len : batch);
            for (int i = 0; i < nb; i++)
                h_cand[i] = w->lo + i;
            printf("  [window %lld] n=%.4f  D%d(r=%.4f)=%.4f  x∈[%.0f,%.0f]\n",
                   base,
                   w->n_center,
                   ((int)floor(w->n_center) % 8) + 1,
                   w->n_center - floor(w->n_center),
                   w->dn, w->lo, w->hi);
        } else {
            for (int i = 0; i < nb; i++)
                h_cand[i] = x_start + base + i;
        }

        cudaMemcpy(d_cand, h_cand, nb * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemset(d_flags, 0, nb * sizeof(uint8_t));

        /* Pass 1: B=500, full array */
        int blocks1 = (nb + threads - 1) / threads;
        pass1_kernel<<<blocks1, threads>>>(d_cand, d_flags, nb);
        cudaDeviceSynchronize();

        cudaMemcpy(h_flags, d_flags, nb * sizeof(uint8_t), cudaMemcpyDeviceToHost);
        int n1 = compact(h_flags, nb, d_live_h);
        pass1_survivors += n1;

        if (n1 == 0) continue;

        /* Pass 2: B=5000, survivors only */
        cudaMemcpy(d_live, d_live_h, n1 * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemset(d_flags, 0, nb * sizeof(uint8_t));
        int blocks2 = (n1 + threads - 1) / threads;
        pass2_kernel<<<blocks2, threads>>>(d_cand, d_flags, d_live, n1, d_zeros_global);
        cudaDeviceSynchronize();

        cudaMemcpy(h_flags, d_flags, nb * sizeof(uint8_t), cudaMemcpyDeviceToHost);
        int n2 = compact(h_flags, nb, d_live_h);
        pass2_survivors += n2;

        if (n2 == 0) continue;

        /* Pass 3: adaptive B=10000, survivors only */
        cudaMemcpy(d_live, d_live_h, n2 * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemset(d_flags,  0, nb * sizeof(uint8_t));
        cudaMemset(d_scores, 0, nb * sizeof(double));
        int blocks3 = (n2 + threads - 1) / threads;
        pass3_kernel<<<blocks3, threads>>>(d_cand, d_flags, d_scores, d_live, n2, d_zeros_global);
        cudaDeviceSynchronize();

        cudaMemcpy(h_flags,  d_flags,  nb * sizeof(uint8_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_scores, d_scores, nb * sizeof(double),  cudaMemcpyDeviceToHost);

        /* CPU Miller-Rabin confirmation */
        for (int i = 0; i < nb; i++) {
            if (!h_flags[i]) continue;
            uint64_t xi = (uint64_t)h_cand[i];
            if (!miller_rabin(xi)) continue;

            total_found++;
            double x    = h_cand[i];
            /* CPU n_of_x */
            double lx   = log(x);
            double n_xc = (lx > 0.0 && log(lx/LN_PHI) > 0.0)
                          ? log(log(x)/LN_PHI) / LN_PHI - 0.5/PHI
                          : -1.0;
            double frac = (n_xc > 0.0) ? (n_xc - floor(n_xc)) : 0.0;
            int    dim  = (n_xc > 0.0) ? (((int)floor(n_xc) % 8) + 1) : 1;
            if (dim < 1) dim = 1;
            double dn   = compute_Dn_r(dim, frac, 1.0);

            if (mersenne_mode) {
                /* For Mersenne scan: report p, n(p), score, D{dim}(r) */
                printf("  CANDIDATE p=%-12llu  n=%.6f  score=%.4f  D%d(r=%.4f)=%.4f",
                       (unsigned long long)xi, n_xc, h_scores[i], dim, frac, dn);
                /* extra: lower-half bias filter from Track B */
                if (frac < 0.5)
                    printf("  [lower-half ✓]");
                printf("\n");
            } else {
                if (total_found <= 20 || xi > (uint64_t)(x_end - 100))
                    printf("  PRIME: %-14llu  n=%.6f  score=%.4f\n",
                           (unsigned long long)xi, n_xc, h_scores[i]);
            }
        }
    }

#ifdef _WIN32
    QueryPerformanceCounter(&t1);
    elapsed = (double)(t1.QuadPart - t0.QuadPart) / (double)freq.QuadPart;
#else
    clock_gettime(CLOCK_MONOTONIC, &t1);
    elapsed = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
#endif

    printf("\n╔══ RESULTS ══════════════════════════════════════════════════╗\n");
    printf("  Total candidates   : %lld\n", total_range);
    printf("  After pass 1 (B=500)  : %lld\n", pass1_survivors);
    printf("  After pass 2 (B=5000) : %lld\n", pass2_survivors);
    printf("  Confirmed primes   : %lld\n", total_found);
    printf("  Elapsed            : %.2f s\n", elapsed);
    printf("  Throughput         : %.0f candidates/s\n",
           (double)total_range / elapsed);
    printf("╚════════════════════════════════════════════════════════════╝\n");

    /* Cleanup */
cleanup:
    cudaFree(d_cand); cudaFree(d_flags); cudaFree(d_scores); cudaFree(d_live);
    cudaFree(d_zeros_global);
    free(h_cand); free(h_flags); free(h_scores); free(h_live); free(d_live_h);

    return 0;
}
