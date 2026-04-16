/*
 * ll_mpi.cu  -  Lucas-Lehmer Mersenne primality test
 *
 * MPI type matches metal_infer_for_primes/hdgl_bootloaderz.h exactly:
 *   typedef struct { uint64_t *words; size_t num_words; uint8_t sign; } MPI;
 *
 * Three dispatch paths:
 *   ll_small  p <=  62    : unsigned __int128, pure CPU
 *   ll_cpu    p <= CPU_TH : schoolbook on MPI->words, CPU, __int128 exact
 *   ll_gpu    p >  CPU_TH : GPU-parallel schoolbook, exact integer
 *                           ONE THREAD PER OUTPUT LIMB, 3-word accumulator
 *                           NO cuFFT   NO DWT   NO floating-point arithmetic
 *
 * Overflow / carry pattern (from unified_bigG_fudge10_empiirical_4096bit.c,
 *   spiral8plus/advanced-spiral-8-Human-Genomes  apa_multiply):
 *     __uint128_t sum = temp[i+j] + prod + carry;
 *     temp[i+j] = (uint64_t)sum;
 *     carry     = (uint64_t)(sum >> 64);
 *   — identical __int128 carry chain used here in mpi_sqr_mod_mp_cpu.
 *
 * Folding pattern   (from base4096-2.0.1/spare parts/fold26_wuwei.c):
 *   fold_mod_mp splits the 2n-word product at bit p, adds high half back
 *   (since 2^p ≡ 1 mod M_p), propagating residual carry with the same
 *   wu-wei "let the data determine its path" carry loop.
 *
 * phi-lattice resonance (from hdgl_bootloaderz.c / hdgl_analog_v30):
 *   fib_real(n): continuous Binet Fibonacci (same as hdgl_bootloaderz.c)
 *   D_n(n,beta,r,k,Omega,base): composite resonance score
 *   Prismatic Omega = 0.5 + 0.5*sin(pi*frac*phi)  (prismatic5z.py)
 *
 * Build (clang, matches metal_infer_for_primes/build_analog_cuda.bat):
 *   clang --cuda-gpu-arch=sm_75 --cuda-path="..." -x cuda ll_mpi.cu
 *          -L<CUDA>/lib/x64 -lcudart -lm -O2 -o ll_mpi.exe
 *   (No -lcufft  - no external library, pure CUDA schoolbook)
 */

#ifndef _USE_MATH_DEFINES
#  define _USE_MATH_DEFINES
#endif
#ifndef _CRT_SECURE_NO_WARNINGS
#  define _CRT_SECURE_NO_WARNINGS
#endif

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>

#ifndef M_PI
#  define M_PI 3.14159265358979323846
#endif

/* ── Constants (mirror hdgl_bootloaderz.h) ──────────────────────────────── */
#define BLZ_PHI  1.6180339887498948

/* ── CPU / GPU dispatch threshold ─────────────────────────────────────────
 * Below CPU_TH: schoolbook on MPI->words, single-threaded, exact.
 * Above CPU_TH: GPU-parallel schoolbook kernel k_sqr_limb, exact.         */
#define CPU_TH   20000u   /* exponent p; n_words = ceil(p/64) <= 313 */

/* Auto-select NTT vs schoolbook threshold.  Empirically determined on RTX 2060
 * (sm_75, April 2026).  Schoolbook is O(p^3), NTT optimised is O(p^2 log p);
 * the two models cross at p ≈ 386 000.  We use 400 000 as a conservative
 * margin.  Override at any time with --squaring schoolbook|ntt.          */
#define NTT_AUTO_THRESHOLD 400000u

/* Progress report interval in wall-clock seconds.  Checked once per
 * iteration on the host-side CPU round-trip — zero GPU impact.        */
#define PROGRESS_INTERVAL 30

/* ── MPI - matches hdgl_bootloaderz.h field names and order exactly ──────── */
typedef struct {
    uint64_t *words;    /* little-endian limbs: words[0] = least significant */
    size_t    num_words;
    uint8_t   sign;
} MPI;

/* ── phi-lattice constants ─────────────────────────────────────────────── */
static const int PRIMES50[50] = {
      2,   3,   5,   7,  11,  13,  17,  19,  23,  29,
     31,  37,  41,  43,  47,  53,  59,  61,  67,  71,
     73,  79,  83,  89,  97, 101, 103, 107, 109, 113,
    127, 131, 137, 139, 149, 151, 157, 163, 167, 173,
    179, 181, 191, 193, 197, 199, 211, 223, 227, 229
};

static int g_use_analog_gpu   = 0;   /* legacy alias for --precision 32 */
static int g_use_persistent   = 0;
static int g_precision        = 64;  /* squaring limb width: 32 or 64 */
/* g_squaring: -1=auto, 0=schoolbook, 1=ntt, 2=gpucarry, 3=analog (v30b+Kuramoto) */
static int g_squaring         = -1;

/* v30b + 8D Kuramoto analog LL path (ll_analog.c) */
#include "ll_analog.h"
static void cpu_fold_sub2(uint64_t *h_flat, const uint8_t *h_ovf,
                          uint64_t *h_x, size_t n, size_t n2,
                          int pw, int pb);

/* Continuous Binet Fibonacci - identical to fib_real() in hdgl_bootloaderz.c */
static double fib_real(double n) {
    static const double SQRT5 = 2.2360679774997896;
    double phi_inv = 1.0 / BLZ_PHI;
    return (pow(BLZ_PHI, n) - pow(phi_inv, n) * cos(M_PI * n)) / SQRT5;
}

/* Map Mersenne exponent p to phi-lattice coordinate n */
static double n_of_2p(uint64_t p) {
    double lnphi = log(BLZ_PHI);
    double arg   = (double)p * log(2.0) / lnphi;
    return log(arg) / lnphi - 0.5 / BLZ_PHI;
}

/* D_n resonance score (6-arg, matching prime_pipeline.c) */
static double D_n(double n, double beta, double r, double k,
                  double Omega, double base) {
    int    idx = ((int)floor(n + beta) + 50) % 50;
    double Fn  = fib_real(n);
    double Pn  = (double)PRIMES50[idx];
    double bterm = pow(base, n);          /* base^n growth term */
    double inner = BLZ_PHI * Fn * Pn * bterm * Omega;
    if (inner <= 0.0) return 0.0;
    return sqrt(inner) * pow(r > 0.0 ? r : 1e-9, k);
}

/* Print phi-lattice resonance for exponent p */
static void report_resonance(uint64_t p) {
    double n    = n_of_2p(p);
    double frac = n - floor(n);
    /* Prismatic Omega from prismatic5z.py */
    double Omega = 0.5 + 0.5 * sin(M_PI * frac * BLZ_PHI);
    double score = D_n(n, 0.0, frac, 1.0, Omega, 2.0);
    printf("  phi-lattice: n=%.4f  frac=%.4f  Omega=%.4f  D_n=%.6f\n",
           n, frac, Omega, score);
}

/* ── MPI helpers ─────────────────────────────────────────────────────────── */
static void mpi_alloc(MPI *m, size_t n) {
    m->words     = (uint64_t *)calloc(n, sizeof(uint64_t));
    m->num_words = n;
    m->sign      = 0;
}
static void mpi_free_m(MPI *m) {
    if (m->words) free(m->words);
    m->words     = NULL;
    m->num_words = 0;
}
static void mpi_ensure(MPI *m, size_t n) {
    if (m->num_words >= n) return;
    uint64_t *nw = (uint64_t *)calloc(n, sizeof(uint64_t));
    if (m->words) {
        memcpy(nw, m->words, m->num_words * sizeof(uint64_t));
        free(m->words);
    }
    m->words     = nw;
    m->num_words = n;
}

/* ── Fold flat 2n-word product into MPI mod 2^p-1 ──────────────────────── */
static void fold_mod_mp(const uint64_t *flat, size_t flat_len,
                        uint64_t p, MPI *res) {
    size_t   n  = (size_t)((p + 63) / 64);
    uint64_t pw = p / 64;   /* word index of the p-bit boundary */
    uint64_t pb = p % 64;   /* bit index within that word        */

    mpi_ensure(res, n);
    memset(res->words, 0, n * sizeof(uint64_t));

    /* res = flat_lo (bits 0..p-1) */
    for (size_t k = 0; k < pw && k < flat_len; k++)
        res->words[k] = flat[k];
    if (pb > 0 && pw < flat_len && pw < n)
        res->words[pw] = flat[pw] & ((1ULL << pb) - 1ULL);

    /* res += flat_hi = flat >> p */
    uint64_t carry = 0;
    for (size_t k = 0; k < n + 2; k++) {
        /* extract word k of (flat >> p) */
        uint64_t hi_word;
        size_t   base_idx = (size_t)(pw + k);
        if (pb == 0) {
            hi_word = (base_idx < flat_len) ? flat[base_idx] : 0;
        } else {
            uint64_t lo = (base_idx     < flat_len) ? flat[base_idx]     : 0;
            uint64_t hi = (base_idx + 1 < flat_len) ? flat[base_idx + 1] : 0;
            hi_word = (lo >> pb) | (hi << (64 - pb));
        }

        if (k >= n) {
            carry += hi_word;
            break;
        }

        unsigned __int128 s = (unsigned __int128)res->words[k] + hi_word + carry;
        res->words[k] = (uint64_t)s;
        carry          = (uint64_t)(s >> 64);
    }

    /* Normalize: fold carry and top-word overflow back into res.
     * After lo+hi fold, res may have bits above bit p-1 in word[n-1]
     * (e.g. res = 2^p when lo+hi = 2^p ≡ 1, not caught by canonical check).
     * 2^p ≡ 1 mod M_p, so each overflow bit = one unit added to res[0].
     * apa_multiply carry pattern: loop until no carry and no top overflow. */
    for (;;) {
        /* Extract overflow from top word (bits above pb-1) */
        uint64_t over = (pb > 0) ? (res->words[n - 1] >> pb) : 0;
        if (over) res->words[n - 1] &= (1ULL << pb) - 1ULL;

        uint64_t c = carry + over;
        carry = 0;
        if (!c) break;

        /* Add c into res[0..n-1], propagating carry */
        for (size_t k = 0; k < n && c; k++) {
            unsigned __int128 acc = (unsigned __int128)res->words[k] + c;
            res->words[k] = (uint64_t)acc;
            c              = (uint64_t)(acc >> 64);
        }
        carry = c;  /* any remaining carry → top word overflows → next iter */
    }

    /* canonical: M_p ≡ 0 mod M_p */
    int is_mp = 1;
    for (size_t k = 0; k < n && is_mp; k++) {
        uint64_t expected;
        if (pb == 0) {
            expected = ~0ULL;
        } else {
            expected = (k < pw) ? ~0ULL : (k == pw ? (1ULL << pb) - 1ULL : 0ULL);
        }
        if (res->words[k] != expected) is_mp = 0;
    }
    if (is_mp) memset(res->words, 0, n * sizeof(uint64_t));
}

/* ── CPU schoolbook squaring mod M_p ────────────────────────────────────── */
static void mpi_sqr_mod_mp_cpu(MPI *s, uint64_t p) {
    size_t n  = (size_t)((p + 63) / 64);
    size_t n2 = 2 * n;

    mpi_ensure(s, n);

    /* prod[0..n2]: 2n-word product, accumulated with running carry */
    uint64_t *prod = (uint64_t *)calloc(n2 + 2, sizeof(uint64_t));

    for (size_t i = 0; i < n; i++) {
        unsigned __int128 carry = 0;
        uint64_t xi = s->words[i];
        if (xi == 0) continue;
        for (size_t j = 0; j < n; j++) {
            unsigned __int128 t = (unsigned __int128)xi * s->words[j]
                                + prod[i + j] + carry;
            prod[i + j] = (uint64_t)t;
            carry        = t >> 64;
        }
        /* propagate carry beyond the inner loop */
        size_t k = i + n;
        while (carry) {
            unsigned __int128 t = (unsigned __int128)prod[k] + carry;
            prod[k] = (uint64_t)t;
            carry    = t >> 64;
            k++;
        }
    }

    fold_mod_mp(prod, n2 + 2, p, s);
    free(prod);
}

/* ── CPU subtract-2 mod M_p ─────────────────────────────────────────────── */
static void mpi_sub2_mod_mp(MPI *s, uint64_t p) {
    size_t n = (size_t)((p + 63) / 64);
    mpi_ensure(s, n);

    /* check if s < 2 */
    int small = 1;
    for (size_t k = n; k-- > 1; ) {
        if (s->words[k]) { small = 0; break; }
    }
    if (small && s->words[0] >= 2) small = 0;

    if (!small) {
        /* s >= 2 : subtract 2 with borrow */
        uint64_t borrow = 2;
        for (size_t k = 0; k < n && borrow; k++) {
            if (s->words[k] >= borrow) {
                s->words[k] -= borrow;
                borrow = 0;
            } else {
                s->words[k] = s->words[k] - borrow; /* wraps */
                borrow = 1;
            }
        }
    } else {
        /* s is 0 or 1 : result = M_p + s - 2 = M_p - (2-s) */
        uint64_t val = s->words[0]; /* 0 or 1 */
        uint64_t pb  = p % 64;

        /* set to M_p = (2^p - 1) */
        for (size_t k = 0; k < n; k++) s->words[k] = ~0ULL;
        if (pb > 0) s->words[n - 1] = (1ULL << pb) - 1ULL;

        /* subtract (2 - val) */
        uint64_t sub    = 2 - val;
        uint64_t borrow = sub;
        for (size_t k = 0; k < n && borrow; k++) {
            if (s->words[k] >= borrow) {
                s->words[k] -= borrow;
                borrow = 0;
            } else {
                s->words[k] = s->words[k] - borrow;
                borrow = 1;
            }
        }
    }
}

/* check whether MPI s == 0 */
static int mpi_is_zero(const MPI *s, size_t n) {
    for (size_t k = 0; k < n; k++)
        if (s->words[k]) return 0;
    return 1;
}

/* ────────────────────────────────────────────────────────────────────────────
 * GPU path: exact integer schoolbook squaring
 *
 * k_sqr_limb: one thread per output limb k of the squaring.
 *   computes  sum_{i+j=k, 0<=i,j<n} x[i]*x[j]
 *   using a 3-word (192-bit) accumulator to handle overflow for any n.
 *   stores result in d_lo[k], d_mi[k], d_hi[k].
 *
 * CPU then assembles the 3-word-per-limb array into the flat 2n-word product
 * via a 128-bit wide-accumulator pass, then folds mod 2^p-1.
 * ─────────────────────────────────────────────────────────────────────────── */

/* ────────────────────────────────────────────────────────────────────────────
 * Persistent GPU Block — HDGL wu-wei streaming path
 *
 * k_ll_persistent_block: ALL p-2 LL iterations run in a SINGLE kernel launch.
 *
 * Design (HDGL philosophy):
 *   - Persistent state: s[n] lives in shared memory for the entire sequence.
 *     No data leaves the GPU between iterations — only the final result is
 *     transferred back.  Like fold26_wuwei_stream's "state continuity across
 *     chunks": the LL sequence IS the stream; each squaring IS a chunk.
 *
 *   - Inline Mersenne fold: thread k computes BOTH halves of the squaring in
 *     one pass, with no 2n temporary array ever materialized:
 *       direct  contribution: sum x[i]*x[j]   where i+j = k     (i,j < n)
 *       wrapped contribution: sum x[i]*x[j]   where i+j = k+n   (i,j < n)
 *     Since 2^(k+n*64) ≡ 2^k * 2^(n*64) and 2^p = 2^(n*64 - slack) * 2^slack
 *     we need the bit-level fold.  Simpler: we do the full 2n-word squaring
 *     but keep it in smem_acc (3 words × 2n entries, fits 48KB for n≤1365).
 *     Then thread 0 assembles and folds inline.
 *
 *   - 3-word accumulator (192-bit): same as k_sqr_limb, handles any n.
 *
 *   - Carry + fold + sub2 all done by thread 0 (wu-wei: "let the data
 *     determine its path; only add complexity if it helps").
 *
 * Shared memory layout (n2 = 2*n):
 *   smem_s[0..n-1]     : current s (n × uint64_t)
 *   smem_lo[0..n2-1]   : squaring result lo word  (n2 × uint64_t)
 *   smem_mi[0..n2-1]   : squaring result mi word  (n2 × uint64_t)
 *   smem_hi[0..n2-1]   : squaring result hi word  (n2 × uint64_t)
 *
 * Shared memory required: n*8 + 3*n2*8 = n*8 + 6*n*8 = 7*n*8 bytes
 *   At p=110503: n=1727   → 7*1727*8 = 96,712 bytes  (>48KB limit)
 *   At p= 44497: n= 696   → 7* 696*8 = 38,976 bytes  (fits!)
 *   At p= 86243: n=1348   → 7*1348*8 = 75,488 bytes  (>48KB)
 *
 * So smem approach only works for small n.  For large n, we keep the squaring
 * accumulation in global memory (device scratch) but still run persistently:
 * the iteration loop is on-device (avoids p launch overheads).  Thread 0 does
 * all fold/carry/sub2 using global smem scratch (d_lo/d_mi/d_hi reused).
 *
 * SMEM_N_LIM = floor(48*1024 / (7*8)) = 877   → p ≤ 56128
 * Above that we use the global-persistent path (d-scratch, no smem for acc).
 *
 * Grid: 1 block × BLOCK_SZ threads.  BLOCK_SZ = 1024.
 * Each thread handles ceil(n/BLOCK_SZ) limbs in squaring phase.
 * ─────────────────────────────────────────────────────────────────────────── */
#define BLOCK_SZ  1024
/* n_words ceiling where all 4 smem arrays fit in 48KB */
#define SMEM_N_LIM  877   /* 7*877*8 = 49112 > 48KB → use 876; 7*876*8=49056 */
/* Recompute: 48*1024 = 49152; 49152/(7*8) = 877.7 → 877 */

/* Persistent kernel: 1 block.  Uses global scratch d_lo/d_mi/d_hi for the
 * squaring accumulation (n2 words each), performs p-2 LL iterations entirely
 * on-device, writes final s to d_x on completion.
 * p_exp is the Mersenne exponent stored as int (max ~6M fits int).
 * iters   = p - 2.
 * n_words = ceil(p/64).
 */
__global__ void k_ll_persistent_block(
        uint64_t * __restrict__ d_s,   /* in: initial s (n_words), out: final */
        uint64_t * __restrict__ d_lo,  /* scratch 2*n_words                   */
        uint64_t * __restrict__ d_mi,  /* scratch 2*n_words                   */
        uint64_t * __restrict__ d_hi,  /* scratch 2*n_words                   */
        int n_words,
        int p_exp,
        int iters)
{
    /* shared memory: current s vector (n_words × uint64_t).
     * The squaring accumulation (lo/mi/hi) lives in global d_lo/d_mi/d_hi. */
    extern __shared__ uint64_t smem_s[];  /* n_words entries */

    int tid  = threadIdx.x;
    int bsz  = blockDim.x;
    int n    = n_words;
    int n2   = 2 * n;

    /* Load s from global into shared */
    for (int k = tid; k < n; k += bsz)
        smem_s[k] = d_s[k];
    __syncthreads();

    for (int iter = 0; iter < iters; iter++) {

        /* ── Phase A: squaring accumulation ──────────────────────────────
         * Thread tid computes d_lo[k], d_mi[k], d_hi[k] for all k where
         * k ≡ tid (mod bsz).  Uses 3-word accumulator (identical to k_sqr_limb).
         */
        for (int k = tid; k < n2; k += bsz) {
            uint64_t acc_lo = 0, acc_mi = 0, acc_hi = 0;

            int i0 = (k >= n) ? (k - n + 1) : 0;
            int i1 = (k <  n) ? k : (n - 1);

            for (int i = i0; i <= i1; i++) {
                unsigned __int128 prod =
                    (unsigned __int128)smem_s[i] * (unsigned __int128)smem_s[k - i];
                uint64_t p_lo = (uint64_t)prod;
                uint64_t p_hi = (uint64_t)(prod >> 64);

                uint64_t old = acc_lo;
                acc_lo += p_lo;
                if (acc_lo < old) { if (++acc_mi == 0) acc_hi++; }

                old = acc_mi;
                acc_mi += p_hi;
                if (acc_mi < old) acc_hi++;
            }

            d_lo[k] = acc_lo;
            d_mi[k] = acc_mi;
            d_hi[k] = acc_hi;
        }
        __syncthreads();

        /* ── Phase B/C/D: assemble flat product, fold mod 2^p-1, sub 2 ──
         * All done by thread 0 only (wu-wei: sequential is correct and fast
         * enough — ~n iterations at GPU clock, < 5µs for n≤6144).
         * All other threads are idle during this phase; __syncthreads()
         * at the end restores coherence before next iteration.
         */
        if (tid == 0) {

            /* B: assemble flat product (2n+4 words) from (lo,mi,hi) per limb.
             * Each position k receives: lo[k] + mi[k-1] + hi[k-2] + carry.
             * We use a two-step approach matching mpi_sqr_mod_mp_gpu exactly:
             * step1: scatter lo/mi/hi contributions with overflow flags,
             * step2: propagate overflow flags into carry chain. */
            int flat_len = n2 + 4;

            /* Reuse smem_s as a flat accumulator temporarily — it's n words,
             * but flat needs n2+4.  Use d_s (n global words) is too small.
             * Solution: use d_lo as output flat after the assembly (we've
             * already consumed d_lo as input, and flat_len = n2+4 ≤ n2+4).
             * Actually d_lo has n2 entries — we need n2+4.  Use a stack
             * array instead; n2+4 ≤ 2*6144+4 = 12292 words = 98KB — too big
             * for stack.  Use d_lo for flat (n2 entries) and handle the last 4
             * with carry overflow from d_mi/d_hi since flat words ≥ n2 are
             * zero after the Mersenne fold.  Better: use d_s as temp for the
             * overflow byte array (n words << n2), and keep a running carry.
             *
             * Cleanest: single-pass carry propagation.  Iterate k=0..n2+3,
             * construct contribution on-the-fly from d_lo[k]+d_mi[k-1]+d_hi[k-2]
             * (out of range → 0), accumulate into a 128-bit running sum,
             * emit flat[k] = low 64 bits, carry = high 64 bits + ovf. */

            /* We'll write the flat result into d_lo (reusing the buffer).
             * d_lo is n2 words; we need n2+4.  Since fold will only look
             * beyond n2 for the carry that propagates in, we keep carry
             * separately and pass it to fold_mod_mp replacement below. */

            uint64_t carry_in = 0;
            for (int k = 0; k < flat_len; k++) {
                uint64_t vlo = (k < n2)   ? d_lo[k] : 0;
                uint64_t vmi = (k > 0 && k-1 < n2) ? d_mi[k-1] : 0;
                uint64_t vhi = (k > 1 && k-2 < n2) ? d_hi[k-2] : 0;

                unsigned __int128 acc
                    = (unsigned __int128)vlo
                    + (unsigned __int128)vmi
                    + (unsigned __int128)vhi
                    + carry_in;

                /* Store flat word back into d_lo (safe: k < n2 covers most;
                 * for k >= n2 we only need carry, not the actual word). */
                if (k < n2) d_lo[k] = (uint64_t)acc;
                carry_in = (uint64_t)(acc >> 64);
            }
            /* carry_in now holds residual carry above position n2+3.
             * Mersenne fold: add carry_in into position 0 of the result
             * (handled inside the fold below). */

            /* C: Fold mod 2^p - 1 (inline, writing into smem_s) */
            {
                int   pw = p_exp / 64;
                int   pb = p_exp % 64;

                /* res = flat_lo (bits 0..p-1) */
                for (int k = 0; k < n; k++) smem_s[k] = 0;
                for (int k = 0; k < pw && k < n2; k++)
                    smem_s[k] = d_lo[k];
                if (pb > 0 && pw < n2 && pw < n)
                    smem_s[pw] = d_lo[pw] & ((1ULL << pb) - 1ULL);

                /* res += flat_hi = flat >> p */
                uint64_t fc = carry_in;
                for (int k = 0; k < n + 2; k++) {
                    int base_idx = pw + k;
                    uint64_t hi_word;
                    if (pb == 0) {
                        hi_word = (base_idx < n2) ? d_lo[base_idx] : 0;
                    } else {
                        uint64_t lo = (base_idx   < n2) ? d_lo[base_idx]   : 0;
                        uint64_t hi = (base_idx+1 < n2) ? d_lo[base_idx+1] : 0;
                        hi_word = (lo >> pb) | (hi << (64 - pb));
                    }
                    if (k >= n) { fc += hi_word; break; }
                    unsigned __int128 s128
                        = (unsigned __int128)smem_s[k] + hi_word + fc;
                    smem_s[k] = (uint64_t)s128;
                    fc = (uint64_t)(s128 >> 64);
                }

                /* Normalize overflow */
                for (;;) {
                    uint64_t over = (pb > 0) ? (smem_s[n-1] >> pb) : 0;
                    if (over) smem_s[n-1] &= (1ULL << pb) - 1ULL;
                    uint64_t c2 = fc + over;
                    fc = 0;
                    if (!c2) break;
                    for (int k = 0; k < n && c2; k++) {
                        unsigned __int128 a
                            = (unsigned __int128)smem_s[k] + c2;
                        smem_s[k] = (uint64_t)a;
                        c2 = (uint64_t)(a >> 64);
                    }
                    fc = c2;
                }

                /* canonical M_p ≡ 0 */
                int is_mp = 1;
                for (int k = 0; k < n && is_mp; k++) {
                    uint64_t expected;
                    if (pb == 0)      expected = ~0ULL;
                    else if (k < pw)  expected = ~0ULL;
                    else if (k == pw) expected = (1ULL << pb) - 1ULL;
                    else              expected = 0ULL;
                    if (smem_s[k] != expected) is_mp = 0;
                }
                if (is_mp)
                    for (int k = 0; k < n; k++) smem_s[k] = 0;
            }

            /* D: subtract 2 mod M_p  (mpi_sub2_mod_mp logic, inline) */
            {
                int   pb = p_exp % 64;
                int   pw = p_exp / 64;

                /* check s < 2 */
                int small = 1;
                for (int k = n-1; k >= 1 && small; k--)
                    if (smem_s[k]) small = 0;
                if (small && smem_s[0] >= 2) small = 0;

                if (!small) {
                    uint64_t borrow = 2;
                    for (int k = 0; k < n && borrow; k++) {
                        if (smem_s[k] >= borrow) {
                            smem_s[k] -= borrow;
                            borrow = 0;
                        } else {
                            smem_s[k] -= borrow; /* wraps */
                            borrow = 1;
                        }
                    }
                } else {
                    uint64_t val = smem_s[0];
                    for (int k = 0; k < n; k++) smem_s[k] = ~0ULL;
                    if (pb > 0) smem_s[n-1] = (1ULL << pb) - 1ULL;
                    (void)pw;
                    uint64_t sub = 2 - val;
                    uint64_t borrow = sub;
                    for (int k = 0; k < n && borrow; k++) {
                        if (smem_s[k] >= borrow) {
                            smem_s[k] -= borrow;
                            borrow = 0;
                        } else {
                            smem_s[k] -= borrow;
                            borrow = 1;
                        }
                    }
                }
            }
        } /* tid == 0 */

        __syncthreads();
    } /* iter loop */

    /* Write final s to global */
    for (int k = tid; k < n; k += bsz)
        d_s[k] = smem_s[k];
    /* no __syncthreads() needed after final write */
}

/* ── k_sqr_tiled: 2D tiled parallel squaring ─────────────────────────────
 *
 * HDGL wu-wei parallel decomposition: launch n×n threads (one per (i,j)
 * product pair).  Each thread computes x[i]*x[j] and atomically accumulates
 * the 192-bit result into d_lo[i+j] / d_mi[i+j] / d_hi[i+j].
 *
 * Compared to k_sqr_limb (1 thread per output limb, O(n) inner loop per
 * thread): this launches n² threads each doing O(1) work → full GPU
 * utilization for any n ≥ 64.
 *
 * Grid: (ceil(n/TILE) × ceil(n/TILE)) blocks, each TILE×TILE threads.
 * Atomic adds: 64-bit atomicAdd is the bottleneck — mitigated by
 * shared-memory reduction within each tile before going global.
 * ── */
#define SQR_TILE 16  /* 16×16 = 256 threads per block */

/* ── k_sqr_warp: warp-parallel squaring ──────────────────────────────────
 *
 * HDGL wu-wei warp decomposition: one WARP of 32 threads cooperates to
 * compute ONE output limb k of the squaring.
 * Thread lane l computes the partial sum for i = l, l+32, l+64, ...
 * within [i0, i1] with a 192-bit local accumulator.
 * Then all 32 lanes do a warp reduction (shuffle) to sum the 192-bit
 * partial sums, and lane 0 writes the result.
 *
 * vs k_sqr_limb (1 thread/limb): 32× more threads active → 32× better
 * GPU utilization, hides latency with pipelining.
 *
 * Grid: ceil(n2 / WARP_PER_BLOCK) blocks × (WARP_PER_BLOCK × WARP_SIZE) threads.
 *   Each WARP_SIZE-lane group handles one output limb.
 * ── */
#define WARP_SIZE      32
#define WARP_PER_BLOCK  8   /* 8 warps = 256 threads per block */

__global__ void k_sqr_warp(const uint64_t * __restrict__ x,
                             uint64_t * __restrict__ d_lo,
                             uint64_t * __restrict__ d_mi,
                             uint64_t * __restrict__ d_hi,
                             int n_words)
{
    /* Which output limb does this warp handle? */
    int warp_id = (int)((blockIdx.x * WARP_PER_BLOCK)
                        + (int)(threadIdx.x / WARP_SIZE));
    int lane    = (int)(threadIdx.x & (WARP_SIZE - 1));
    int k       = warp_id;
    if (k >= 2 * n_words) return;

    int i0 = (k >= n_words) ? (k - n_words + 1) : 0;
    int i1 = (k <  n_words) ? k : (n_words - 1);

    /* Each lane accumulates a subset of the inner-product terms */
    uint64_t acc_lo = 0, acc_mi = 0, acc_hi = 0;

    for (int i = i0 + lane; i <= i1; i += WARP_SIZE) {
        unsigned __int128 prod = (unsigned __int128)x[i]
                               * (unsigned __int128)x[k - i];
        uint64_t p_lo = (uint64_t)prod;
        uint64_t p_hi = (uint64_t)(prod >> 64);

        uint64_t old = acc_lo;
        acc_lo += p_lo;
        if (acc_lo < old) { if (++acc_mi == 0) acc_hi++; }

        old = acc_mi;
        acc_mi += p_hi;
        if (acc_mi < old) acc_hi++;
    }

    /* Warp-level reduction using __shfl_down_sync */
    unsigned mask = 0xffffffff;
#pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        uint64_t rlo = __shfl_down_sync(mask, acc_lo, offset);
        uint64_t rmi = __shfl_down_sync(mask, acc_mi, offset);
        uint64_t rhi = __shfl_down_sync(mask, acc_hi, offset);

        uint64_t old = acc_lo;
        acc_lo += rlo;
        uint64_t carry = (acc_lo < old) ? 1ULL : 0ULL;

        old = acc_mi;
        acc_mi += rmi + carry;
        carry = (acc_mi < old) ? 1ULL : 0ULL;

        acc_hi += rhi + carry;
    }

    if (lane == 0) {
        d_lo[k] = acc_lo;
        d_mi[k] = acc_mi;
        d_hi[k] = acc_hi;
    }
}

__global__ void k_sqr_tiled(const uint64_t * __restrict__ x,
                              uint64_t * __restrict__ d_lo,
                              uint64_t * __restrict__ d_mi,
                              uint64_t * __restrict__ d_hi,
                              int n_words)
{
    int ti = (int)(blockIdx.y * SQR_TILE + threadIdx.y);  /* row:    i index */
    int tj = (int)(blockIdx.x * SQR_TILE + threadIdx.x);  /* column: j index */
    if (ti >= n_words || tj >= n_words) return;

    /* 128-bit product p = x[ti] * x[tj] */
    unsigned __int128 p = (unsigned __int128)x[ti] * (unsigned __int128)x[tj];
    uint64_t p_lo = (uint64_t)p;
    uint64_t p_hi = (uint64_t)(p >> 64);

    int out = ti + tj;  /* output limb index (0 .. 2n-2) */

    /* Atomic accumulation into global d_lo[out] / d_mi[out] / d_hi[out].
     * We accumulate p_lo into (d_lo, d_mi, d_hi) and p_hi into (d_mi, d_hi).
     *
     * Step 1: atomicAdd p_lo into d_lo[out]; check overflow via returned old.
     * Step 2: if d_lo wrapped (old + p_lo < old), carry 1 into d_mi.
     *         Then atomicAdd p_hi into d_mi; carry into d_hi if needed.
     *
     * This is correct because: carry propagation is a minor event (occurs
     * only when the lo sum wraps around 2^64), and d_hi is an overflow counter
     * bounded by the number of contributing threads = n ≤ ~4096 << 2^64.
     */
    uint64_t old_lo = atomicAdd((unsigned long long *)&d_lo[out],
                                (unsigned long long)p_lo);
    uint64_t carry_lo = (old_lo + p_lo < old_lo) ? 1ULL : 0ULL;

    uint64_t old_mi = atomicAdd((unsigned long long *)&d_mi[out],
                                (unsigned long long)(p_hi + carry_lo));
    uint64_t carry_mi = (old_mi + p_hi + carry_lo < old_mi + carry_lo) ? 1ULL : 0ULL;
    /* more precisely: carry if p_hi+carry_lo overflows when added to old_mi */
    /* recompute correctly: */
    uint64_t sum_mi = old_mi + p_hi + carry_lo;
    carry_mi = (sum_mi < old_mi) ? 1ULL : 0ULL;  /* update */

    if (carry_mi)
        atomicAdd((unsigned long long *)&d_hi[out],
                  (unsigned long long)carry_mi);
}

__global__ void k_sqr_limb(const uint64_t * __restrict__ x,
                             uint64_t * __restrict__ d_lo,
                             uint64_t * __restrict__ d_mi,
                             uint64_t * __restrict__ d_hi,
                             int n_words)
{
    int k = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (k >= 2 * n_words) return;

    uint64_t acc_lo = 0, acc_mi = 0, acc_hi = 0;

    int i0 = (k >= n_words) ? (k - n_words + 1) : 0;
    int i1 = (k <  n_words) ? k : (n_words - 1);

    for (int i = i0; i <= i1; i++) {
        /* 128-bit product p = x[i] * x[k-i] */
        unsigned __int128 p = (unsigned __int128)x[i] * (unsigned __int128)x[k - i];
        uint64_t p_lo = (uint64_t)p;
        uint64_t p_hi = (uint64_t)(p >> 64);

        /* add p_lo into (acc_lo, acc_mi, acc_hi) */
        uint64_t old = acc_lo;
        acc_lo += p_lo;
        if (acc_lo < old) {
            if (++acc_mi == 0) acc_hi++;
        }

        /* add p_hi into (acc_mi, acc_hi) */
        old = acc_mi;
        acc_mi += p_hi;
        if (acc_mi < old) acc_hi++;
    }

    d_lo[k] = acc_lo;
    d_mi[k] = acc_mi;
    d_hi[k] = acc_hi;
}

/* ────────────────────────────────────────────────────────────────────────────
 * k_assemble — parallel flat-product assembly (HDGL wu-wei: each thread
 * computes its own exclusive output position from 3 read-only inputs).
 *
 * Thread k computes:
 *   d_flat[k] = d_lo[k] + d_mi[k-1] + d_hi[k-2]
 * (out-of-range indices → 0).  No atomics.  No aliasing.
 * Writes a 64-bit word + a 2-bit overflow flag per position.
 * The overflow flags (0..3) are resolved sequentially by k_carry_sub2.
 *
 * Grid: ceil(n2 / FOLD_THR) blocks × FOLD_THR threads.
 * ─────────────────────────────────────────────────────────────────────────── */
#define FOLD_THR 256

__global__ void k_assemble(
        const uint64_t * __restrict__ d_lo,
        const uint64_t * __restrict__ d_mi,
        const uint64_t * __restrict__ d_hi,
        uint64_t * __restrict__ d_flat,  /* n2 words out — exclusive writes */
        uint8_t  * __restrict__ d_ovf,   /* n2 bytes  out — overflow per pos  */
        int n2)
{
    int k = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (k >= n2) return;

    /* Each of lo[k], mi[k-1], hi[k-2] contributes a 64-bit value to position k.
     * Sum them with a 128-bit accumulator; store low 64 in d_flat[k],
     * overflow (0..2) in d_ovf[k].  The overflow is at most 2 because three
     * 64-bit values sum to at most 3*(2^64-1) < 4*2^64, fitting in 2 bits. */
    unsigned __int128 acc = 0;
    acc += (unsigned __int128)d_lo[k];
    if (k > 0)   acc += (unsigned __int128)d_mi[k-1];
    if (k > 1)   acc += (unsigned __int128)d_hi[k-2];

    d_flat[k] = (uint64_t)acc;
    d_ovf[k]  = (uint8_t)(acc >> 64);  /* 0, 1, or 2 */
}

/* ────────────────────────────────────────────────────────────────────────────
 * NTT squaring path  (--squaring ntt)
 *
 * Performs big-integer squaring via Number Theoretic Transform over the
 * Solinas prime  Q = 2^64 - 2^32 + 1  (0xFFFFFFFF00000001).
 * This Q is a GPU-friendly "proably prime" 64-bit Solinas prime with a
 * 2^32-th primitive root of unity g = 7.
 *
 * Algorithm per LL iteration:
 *   1. Expand n 64-bit limbs into 2n 32-bit "half-limbs" (each 0..2^32-1).
 *      Using 32-bit coefficients keeps NTT coefficients < Q always.
 *   2. Zero-pad to length L = next power of 2 >= 4n (needed for squaring).
 *   3. Forward NTT over Z/QZ of length L.
 *   4. Pointwise square mod Q.
 *   5. Inverse NTT; divide by L mod Q.
 *   6. Carry-normalise (each coefficient now holds a partial product sum).
 *   7. Fold mod 2^p-1 and subtract 2.
 *
 * O(n log n) multiplications per iteration vs O(n^2) schoolbook.
 * ─────────────────────────────────────────────────────────────────────────── */

/* Solinas prime Q = 2^64 - 2^32 + 1.  All NTT arithmetic is mod Q. */
#define NTT_Q  0xFFFFFFFF00000001ULL
/* Primitive 2^32-nd root of unity mod Q: g = 7.
 * (7^((Q-1)/2^32) mod Q != 1, checked offline) */
#define NTT_G  7ULL

/* ── Host-side NTT modular arithmetic (for twiddle precomputation) ──────── */
static inline unsigned long long host_ntt_mul(unsigned long long a, unsigned long long b) {
    unsigned __int128 p = (unsigned __int128)a * b;
    unsigned long long lo = (unsigned long long)p;
    unsigned long long hi = (unsigned long long)(p >> 64);
    unsigned long long t  = (hi << 32) - hi;
    unsigned long long r  = lo + t;
    if (r < lo) r += (1ULL << 32) - 1ULL;
    return r >= NTT_Q ? r - NTT_Q : r;
}
static unsigned long long host_ntt_pow(unsigned long long base, unsigned long long exp) {
    unsigned long long result = 1ULL;
    base %= NTT_Q;
    while (exp > 0) {
        if (exp & 1ULL) result = host_ntt_mul(result, base);
        base = host_ntt_mul(base, base);
        exp >>= 1;
    }
    return result;
}

/* ── modular arithmetic helpers (device, inlined) ─────────────────────── */
__device__ __forceinline__ unsigned long long ntt_add(unsigned long long a,
                                                       unsigned long long b) {
    /* (a + b) mod Q, inputs in [0, Q) */
    unsigned long long s = a + b;
    /* s may have wrapped 2^64 OR may be >= Q; use conditional sub */
    return (s < a || s >= NTT_Q) ? s - NTT_Q : s;
}

__device__ __forceinline__ unsigned long long ntt_sub(unsigned long long a,
                                                       unsigned long long b) {
    return (a >= b) ? a - b : a - b + NTT_Q;
}

/* Multiply a*b mod Q using __uint128_t */
__device__ __forceinline__ unsigned long long ntt_mul(unsigned long long a,
                                                       unsigned long long b) {
    unsigned __int128 p = (unsigned __int128)a * b;
    /* Barrett / direct reduction mod Q = 2^64 - 2^32 + 1:
     *   p mod Q  using the identity  2^64 ≡ 2^32 - 1  (mod Q) */
    unsigned long long lo = (unsigned long long)p;
    unsigned long long hi = (unsigned long long)(p >> 64);
    /* hi * 2^64 ≡ hi * (2^32 - 1) mod Q */
    unsigned long long t = (hi << 32) - hi;   /* hi*(2^32-1), may wrap */
    unsigned long long r = lo + t;
    if (r < lo) r += (1ULL << 32) - 1ULL;     /* add-back for carry */
    return r >= NTT_Q ? r - NTT_Q : r;
}

/* Compute g^k mod Q on the fly (used for twiddle factors in kernels) */
__device__ __forceinline__ unsigned long long ntt_pow(unsigned long long base,
                                                       unsigned long long exp) {
    unsigned long long result = 1ULL;
    base %= NTT_Q;
    while (exp > 0) {
        if (exp & 1ULL) result = ntt_mul(result, base);
        base = ntt_mul(base, base);
        exp >>= 1;
    }
    return result;
}

/* ── k_ntt: in-place Cooley-Tukey iterative NTT over Z/QZ ─────────────── *
 * Length L must be a power of 2, stored in d_a[0..L-1].
 * invert==0 → forward; invert==1 → inverse (caller divides by L after).
 * Uses log2(L) passes; each pass is launched as a separate kernel call
 * from the host to avoid synchronisation issues.
 * This single kernel handles ONE butterfly stage (stride s).          */
__global__ void k_ntt_butterfly(unsigned long long * __restrict__ d_a,
                                 int L,
                                 int s,       /* current half-stride (1,2,4,...) */
                                 int invert)
{
    /* Each thread handles one butterfly pair */
    int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int half_L = L / 2;
    if (tid >= half_L) return;

    /* Which group and position within group */
    int group = tid / s;
    int pos   = tid % s;
    int u_idx = group * 2 * s + pos;
    int v_idx = u_idx + s;

    /* Twiddle factor: w = g^((Q-1)/L * group*pos ... ) */
    /* For CT NTT, the twiddle for this butterfly is g^((Q-1)/(2s) * pos)
     * because the group size at this stage is 2s. */
    unsigned long long order = (unsigned long long)(2 * s);  /* butterfly group size */
    unsigned long long exp   = ((NTT_Q - 1ULL) / order) * (unsigned long long)pos;
    if (invert) exp = (NTT_Q - 1ULL) - exp;   /* conjugate for inverse */

    unsigned long long w = ntt_pow(NTT_G, exp);

    unsigned long long u = d_a[u_idx];
    unsigned long long v = ntt_mul(d_a[v_idx], w);
    d_a[u_idx] = ntt_add(u, v);
    d_a[v_idx] = ntt_sub(u, v);
}

/* ── k_ntt_scale: multiply every element by scalar (for INTT /L) ─────── */
__global__ void k_ntt_scale(unsigned long long * __restrict__ d_a,
                             int L,
                             unsigned long long inv_L)
{
    int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (tid >= L) return;
    d_a[tid] = ntt_mul(d_a[tid], inv_L);
}

/* ── k_ntt_sqr: pointwise a[i] = a[i]^2 mod Q ──────────────────────────  */
__global__ void k_ntt_sqr(unsigned long long * __restrict__ d_a, int L)
{
    int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (tid >= L) return;
    unsigned long long v = d_a[tid];
    d_a[tid] = ntt_mul(v, v);
}

/* ── k_ntt_butterfly_tw: Cooley-Tukey butterfly with precomputed twiddles ── *
 * d_tw[k] = omega^k mod Q  where omega = NTT_G^((NTT_Q-1)/L).              *
 * Replaces k_ntt_butterfly which recomputes twiddles via ntt_pow (~64 muls). */
__global__ void k_ntt_butterfly_tw(unsigned long long * __restrict__ d_a,
                                    const unsigned long long * __restrict__ d_tw,
                                    int L, int s, int invert)
{
    int tid    = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int half_L = L >> 1;
    if (tid >= half_L) return;

    int group = tid / s;
    int pos   = tid % s;
    int u_idx = group * (s << 1) + pos;
    int v_idx = u_idx + s;

    /* twiddle index: pos * L/(2s) for forward; (L - pos*L/(2s)) % L for inverse */
    int tw_k   = pos * (half_L / s);     /* = pos * L/(2s); exact since both powers of 2 */
    int tw_idx = invert ? (tw_k == 0 ? 0 : L - tw_k) : tw_k;

    unsigned long long w = d_tw[tw_idx];
    unsigned long long u = d_a[u_idx];
    unsigned long long v = ntt_mul(d_a[v_idx], w);
    d_a[u_idx] = ntt_add(u, v);
    d_a[v_idx] = ntt_sub(u, v);
}

/* ── k_expand_limbs: 64-bit limbs → 32-bit coefficients ────────────────  *
 * limb[i] → coeff[2i] = low32, coeff[2i+1] = high32                      */
__global__ void k_expand_limbs(const uint64_t * __restrict__ d_x,
                                unsigned long long * __restrict__ d_coeffs,
                                int n_words,
                                int L)
{
    int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (tid >= L) return;
    if (tid < 2 * n_words) {
        int limb = tid / 2;
        int half = tid % 2;
        uint64_t w = d_x[limb];
        d_coeffs[tid] = (half == 0) ? (w & 0xFFFFFFFFULL) : (w >> 32);
    } else {
        d_coeffs[tid] = 0ULL;   /* zero-pad */
    }
}

/* ── k_carry_collect: NTT output → 64-bit limbs with carry ─────────────  *
 * After INTT the coefficient c[k] holds a non-negative integer (no mod
 * reduction on the accumulated partial products), but it may be > 2^32.
 * We carry-propagate to get valid 64-bit limbs and store in d_out[0..n2-1].
 * n2 = 2*n_words.  This is single-threaded because it's a serial carry;
 * for small n it's negligible.                                            */
__global__ void k_carry_collect(const unsigned long long * __restrict__ d_c,
                                 uint64_t * __restrict__ d_flat,
                                 uint8_t  * __restrict__ d_ovf,
                                 int L_result,   /* 4*n (first 4n coefficients matter) */
                                 int n2)
{
    /* single thread */
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    unsigned __int128 carry = 0;
    for (int k = 0; k < n2; k++) {
        /* Each output limb is 64 bits; it is assembled from two 32-bit
         * coefficient slots: slot 2k (bits 0..31) and slot 2k+1 (bits 32..63).
         * The coefficients themselves may be large due to accumulated carries
         * from multiplication, so we do a staged carry. */
        unsigned __int128 lo, hi;
        lo = (unsigned __int128)d_c[2*k]   + carry;
        carry = lo >> 32;
        lo &= 0xFFFFFFFFULL;
        hi = (unsigned __int128)d_c[2*k+1] + carry;
        carry = hi >> 32;
        hi &= 0xFFFFFFFFULL;
        unsigned __int128 limb = lo | (hi << 32);
        /* d_flat holds the 64-bit value; d_ovf holds overflow bits > 64 */
        d_flat[k] = (uint64_t)limb;
        d_ovf[k]  = 0;   /* carry fully propagated — no overflow within 64 bits */
    }
    /* flush remaining carry into dummy overflow; cpu_fold_sub2 will handle it */
    (void)carry;
}

/* ── host NTT launcher: runs log2(L) butterfly stages ──────────────────── */
static void launch_ntt(unsigned long long *d_a, int L, int invert,
                        cudaStream_t stream)
{
    int thr = 256;
    int blk = (L / 2 + thr - 1) / thr;
    for (int s = 1; s < L; s <<= 1) {
        k_ntt_butterfly<<<blk, thr, 0, stream>>>(d_a, L, s, invert);
    }
    if (invert) {
        /* compute L^{-1} mod Q via Fermat little theorem: L^{Q-2} mod Q */
        unsigned long long inv_L = 1ULL;
        unsigned long long base = (unsigned long long)L % NTT_Q;
        unsigned long long exp  = NTT_Q - 2ULL;
        /* host-side pow for simplicity — only called once per direction switch */
        while (exp > 0) {
            if (exp & 1ULL) {
                unsigned __int128 p = (unsigned __int128)inv_L * base;
                unsigned long long lo2 = (unsigned long long)p;
                unsigned long long hi2 = (unsigned long long)(p >> 64);
                unsigned long long t2 = (hi2 << 32) - hi2;
                unsigned long long r2 = lo2 + t2;
                if (r2 < lo2) r2 += (1ULL<<32)-1ULL;
                inv_L = r2 >= NTT_Q ? r2 - NTT_Q : r2;
            }
            {
                unsigned __int128 p = (unsigned __int128)base * base;
                unsigned long long lo2 = (unsigned long long)p;
                unsigned long long hi2 = (unsigned long long)(p >> 64);
                unsigned long long t2 = (hi2 << 32) - hi2;
                unsigned long long r2 = lo2 + t2;
                if (r2 < lo2) r2 += (1ULL<<32)-1ULL;
                base = r2 >= NTT_Q ? r2 - NTT_Q : r2;
            }
            exp >>= 1;
        }
        int blk_s = (L + thr - 1) / thr;
        k_ntt_scale<<<blk_s, thr, 0, stream>>>(d_a, L, inv_L);
    }
}

/* ── ll_gpu_ntt: NTT-based O(n log n) LL verifier ──────────────────────── */
static int ll_gpu_ntt(uint64_t p, int verbose) {
    int n  = (int)((p + 63) / 64);
    int n2 = 2 * n;

    /* NTT length: next power of 2 >= 4n */
    int L = 1;
    while (L < 4 * n) L <<= 1;

    /* ── Precomputed twiddle table: eliminates ~64 mults/butterfly ───────── */
    unsigned long long *d_tw = NULL;
    cudaMalloc(&d_tw, (size_t)L * sizeof(unsigned long long));
    {
        unsigned long long *h_tw = (unsigned long long *)malloc((size_t)L * sizeof(unsigned long long));
        unsigned long long omega = host_ntt_pow(NTT_G, (NTT_Q - 1ULL) / (unsigned long long)L);
        unsigned long long wpow  = 1ULL;
        for (int k = 0; k < L; k++) { h_tw[k] = wpow; wpow = host_ntt_mul(wpow, omega); }
        cudaMemcpy(d_tw, h_tw, (size_t)L * sizeof(unsigned long long), cudaMemcpyHostToDevice);
        free(h_tw);
    }
    unsigned long long inv_L = host_ntt_pow((unsigned long long)L % NTT_Q, NTT_Q - 2ULL);

    unsigned long long *d_a = NULL;
    uint64_t *d_x = NULL;
    cudaMalloc(&d_a, (size_t)L * sizeof(unsigned long long));
    cudaMalloc(&d_x, (size_t)n * sizeof(uint64_t));

    /* Pinned host buffers:
     *   h_ntt : single D2H destination for the full L NTT coefficient array
     *   h_x   : current LL state (H2D source each iteration)               */
    unsigned long long *h_ntt = NULL;
    uint64_t           *h_x   = NULL;
    cudaHostAlloc(&h_ntt, (size_t)L  * sizeof(unsigned long long), cudaHostAllocDefault);
    cudaHostAlloc(&h_x,   (size_t)n  * sizeof(uint64_t),           cudaHostAllocDefault);

    /* CPU carry-collect buffers (heap; reused each iteration) */
    uint64_t *h_flat = (uint64_t *)malloc((size_t)n2 * sizeof(uint64_t));
    uint8_t  *h_ovf  = (uint8_t  *)malloc((size_t)n2 * sizeof(uint8_t));
    memset(h_ovf, 0, (size_t)n2 * sizeof(uint8_t));   /* always zero */

    /* Two streams: stream_main for GPU kernels, stream_dma for D2H transfer. *
     * An event fires after the inverse NTT completes, triggering async DMA.  *
     * This matches the dual-stream pattern that makes the schoolbook fast:    *
     * DMA runs concurrently while CPU processes the previous result.         */
    cudaStream_t stream_main, stream_dma;
    cudaStreamCreate(&stream_main);
    cudaStreamCreate(&stream_dma);
    cudaEvent_t ev_ntt;
    cudaEventCreate(&ev_ntt);

    int thr     = 256;
    int blk_exp = (L + thr - 1) / thr;
    int blk_sqr = (L + thr - 1) / thr;
    int blk_tw  = (L / 2 + thr - 1) / thr;
    int blk_sc  = (L + thr - 1) / thr;
    int pw = (int)(p / 64);
    int pb = (int)(p % 64);

    /* ── CUDA graphs for forward and inverse NTT ────────────────────────── *
     * Graph replay has ~50-100x less per-call overhead than individual       *
     * kernel launches: saves ~14 s of launch overhead at p=110503.          */
    cudaGraph_t     g_fwd = NULL, g_inv = NULL;
    cudaGraphExec_t gex_fwd = NULL, gex_inv = NULL;

    cudaStreamBeginCapture(stream_main, cudaStreamCaptureModeGlobal);
    for (int s = 1; s < L; s <<= 1)
        k_ntt_butterfly_tw<<<blk_tw, thr, 0, stream_main>>>(d_a, d_tw, L, s, 0);
    cudaStreamEndCapture(stream_main, &g_fwd);
    cudaGraphInstantiate(&gex_fwd, g_fwd, NULL, NULL, 0);
    cudaGraphDestroy(g_fwd);

    cudaStreamBeginCapture(stream_main, cudaStreamCaptureModeGlobal);
    for (int s = 1; s < L; s <<= 1)
        k_ntt_butterfly_tw<<<blk_tw, thr, 0, stream_main>>>(d_a, d_tw, L, s, 1);
    k_ntt_scale<<<blk_sc, thr, 0, stream_main>>>(d_a, L, inv_L);
    cudaStreamEndCapture(stream_main, &g_inv);
    cudaGraphInstantiate(&gex_inv, g_inv, NULL, NULL, 0);
    cudaGraphDestroy(g_inv);

    memset(h_x, 0, (size_t)n * sizeof(uint64_t));
    h_x[0] = 4;
    cudaMemcpy(d_x, h_x, (size_t)n * sizeof(uint64_t), cudaMemcpyHostToDevice);

    uint64_t iters = p - 2;
    time_t t_start_ntt = time(NULL);
    time_t t_last_ntt  = t_start_ntt;
    for (uint64_t i = 0; i < iters; i++) {
        /* progress report every PROGRESS_INTERVAL seconds */
        {
            time_t t_now = time(NULL);
            if (t_now - t_last_ntt >= PROGRESS_INTERVAL) {
                double pct = 100.0 * (double)(i + 1) / (double)iters;
                long elapsed = (long)(t_now - t_start_ntt);
                long eta = (elapsed > 0) ? (long)((double)elapsed * (iters - i - 1) / (double)(i + 1)) : -1;
                fprintf(stderr, "  [M_%llu NTT] iter %llu/%llu  %.1f%%  elapsed %lds  eta %lds\n",
                        (unsigned long long)p, (unsigned long long)(i + 1),
                        (unsigned long long)iters, pct, elapsed, eta);
                t_last_ntt = t_now;
            }
        }

        /* GPU pipeline (stream_main): expand → fwd NTT → sqr → inv NTT    */
        k_expand_limbs<<<blk_exp, thr, 0, stream_main>>>(d_x, d_a, n, L);
        cudaGraphLaunch(gex_fwd, stream_main);
        k_ntt_sqr<<<blk_sqr, thr, 0, stream_main>>>(d_a, L);
        cudaGraphLaunch(gex_inv, stream_main);

        /* Event fires as soon as inv NTT completes; triggers async D2H.    *
         * stream_dma starts copying while CPU continues.                   */
        cudaEventRecord(ev_ntt, stream_main);
        cudaStreamWaitEvent(stream_dma, ev_ntt, 0);
        cudaMemcpyAsync(h_ntt, d_a, (size_t)L * sizeof(unsigned long long),
                        cudaMemcpyDeviceToHost, stream_dma);

        /* Block CPU until D2H is complete (GPU work already done via event) */
        cudaStreamSynchronize(stream_dma);

        /* CPU carry-collect: replaces k_carry_collect<<<1,1>>> single-thread *
         * GPU kernel — same logic but zero global-memory-latency overhead.   *
         * Pairs of 32-bit NTT coefficients → 64-bit limbs, carry-propagated. */
        {
            unsigned __int128 carry = 0;
            for (int k = 0; k < n2; k++) {
                unsigned __int128 lo, hi;
                lo = (unsigned __int128)h_ntt[2*k]   + carry;
                carry = lo >> 32; lo &= 0xFFFFFFFFULL;
                hi = (unsigned __int128)h_ntt[2*k+1] + carry;
                carry = hi >> 32; hi &= 0xFFFFFFFFULL;
                h_flat[k] = (uint64_t)(lo | (hi << 32));
            }
            /* h_ovf stays all-zero; residual carry is handled by cpu_fold_sub2 */
        }
        cpu_fold_sub2(h_flat, h_ovf, h_x, (size_t)n, (size_t)n2, pw, pb);

        /* H2D: upload new state (blocking; GPU must receive before next iter) */
        cudaMemcpy(d_x, h_x, (size_t)n * sizeof(uint64_t), cudaMemcpyHostToDevice);
    }

    cudaStreamSynchronize(stream_main);

    int result = 1;
    for (int k = 0; k < n; k++)
        if (h_x[k]) { result = 0; break; }

    if (verbose)
        printf("  GPU NTT opt  n_words=%d  L=%d  iters=%llu\n"
               "    (precomp twiddles + CUDA graph replay + dual-stream DMA + CPU carry-collect)\n",
               n, L, (unsigned long long)iters);

    cudaEventDestroy(ev_ntt);
    cudaStreamDestroy(stream_main);
    cudaStreamDestroy(stream_dma);
    cudaGraphExecDestroy(gex_fwd);
    cudaGraphExecDestroy(gex_inv);
    cudaFreeHost(h_ntt); cudaFreeHost(h_x);
    free(h_flat); free(h_ovf);
    cudaFree(d_tw); cudaFree(d_a); cudaFree(d_x);
    return result;
}

/* ────────────────────────────────────────────────────────────────────────────
 * Analog-style 32-bit multiply decomposition path
 *
 * Decompose each 64-bit operand into two 32-bit halves, compute the 128-bit
 * product using 32-bit multiplies, and reuse the existing 64-bit output + fold
 * pipeline. This lets the GPU use smaller arithmetic units while preserving
 * exact integer semantics.
 * ─────────────────────────────────────────────────────────────────────────── */

__global__ void k_sqr_warp32(const uint64_t * __restrict__ x,
                              uint64_t * __restrict__ d_lo,
                              uint64_t * __restrict__ d_mi,
                              uint64_t * __restrict__ d_hi,
                              int n_words)
{
    int warp_id = (int)((blockIdx.x * WARP_PER_BLOCK)
                        + (int)(threadIdx.x / WARP_SIZE));
    int lane    = (int)(threadIdx.x & (WARP_SIZE - 1));
    int k       = warp_id;
    if (k >= 2 * n_words) return;

    int i0 = (k >= n_words) ? (k - n_words + 1) : 0;
    int i1 = (k <  n_words) ? k : (n_words - 1);

    uint64_t acc_lo = 0, acc_mi = 0, acc_hi = 0;
    for (int i = i0 + lane; i <= i1; i += WARP_SIZE) {
        uint64_t a = x[i];
        uint64_t b = x[k - i];

        uint64_t a_lo = (uint32_t)a;
        uint64_t a_hi = a >> 32;
        uint64_t b_lo = (uint32_t)b;
        uint64_t b_hi = b >> 32;

        uint64_t p0 = a_lo * b_lo;
        uint64_t p1 = a_lo * b_hi;
        uint64_t p2 = a_hi * b_lo;
        uint64_t p3 = a_hi * b_hi;

        uint64_t mid = p1 + p2;
        /* carry_mid: p1+p2 overflowed 2^64, contributing 2^64 extra to mid.
         * That extra 2^64, after the <<32 shift, contributes 2^96 to the
         * 128-bit product — i.e. 2^32 to the hi word, NOT 1. */
        uint64_t p1p2_carry = (mid < p1) ? (1ULL << 32) : 0ULL;

        uint64_t lo = p0 + (mid << 32);
        uint64_t lo_carry = (lo < p0) ? 1ULL : 0ULL;
        uint64_t hi = p3 + (mid >> 32) + p1p2_carry + lo_carry;

        uint64_t old = acc_lo;
        acc_lo += lo;
        if (acc_lo < old) { if (++acc_mi == 0) acc_hi++; }

        old = acc_mi;
        acc_mi += hi;
        if (acc_mi < old) acc_hi++;
    }

    unsigned mask = 0xffffffff;
#pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        uint64_t rlo = __shfl_down_sync(mask, acc_lo, offset);
        uint64_t rmi = __shfl_down_sync(mask, acc_mi, offset);
        uint64_t rhi = __shfl_down_sync(mask, acc_hi, offset);

        uint64_t old = acc_lo;
        acc_lo += rlo;
        uint64_t carry = (acc_lo < old) ? 1ULL : 0ULL;

        old = acc_mi;
        acc_mi += rmi + carry;
        carry = (acc_mi < old) ? 1ULL : 0ULL;

        acc_hi += rhi + carry;
    }

    if (lane == 0) {
        d_lo[k] = acc_lo;
        d_mi[k] = acc_mi;
        d_hi[k] = acc_hi;
    }
}

static int ll_gpu_analog(uint64_t p, int verbose) {
    size_t n  = (size_t)((p + 63) / 64);
    size_t n2 = 2 * n;

    uint64_t *d_x    = NULL;
    uint64_t *d_lo   = NULL;
    uint64_t *d_mi   = NULL;
    uint64_t *d_hi   = NULL;
    uint64_t *d_flat = NULL;
    uint8_t  *d_ovf  = NULL;
    cudaMalloc(&d_x,    n  * sizeof(uint64_t));
    cudaMalloc(&d_lo,   n2 * sizeof(uint64_t));
    cudaMalloc(&d_mi,   n2 * sizeof(uint64_t));
    cudaMalloc(&d_hi,   n2 * sizeof(uint64_t));
    cudaMalloc(&d_flat, n2 * sizeof(uint64_t));
    cudaMalloc(&d_ovf,  n2 * sizeof(uint8_t));

    uint64_t *h_flat = NULL;
    uint8_t  *h_ovf  = NULL;
    uint64_t *h_x    = NULL;
    cudaHostAlloc(&h_flat, n2 * sizeof(uint64_t), cudaHostAllocDefault);
    cudaHostAlloc(&h_ovf,  n2 * sizeof(uint8_t),  cudaHostAllocDefault);
    cudaHostAlloc(&h_x,    n  * sizeof(uint64_t), cudaHostAllocDefault);

    cudaStream_t stream_main, stream_dma;
    cudaStreamCreate(&stream_main);
    cudaStreamCreate(&stream_dma);
    cudaEvent_t ev_asm;
    cudaEventCreate(&ev_asm);

    int warp_blocks_sqr = (int)((n2 + WARP_PER_BLOCK - 1) / WARP_PER_BLOCK);
    int warp_thr_sqr    = WARP_PER_BLOCK * WARP_SIZE;
    int fld_blocks      = (int)((n2 + FOLD_THR - 1) / FOLD_THR);
    int pw = (int)(p / 64);
    int pb = (int)(p % 64);

    memset(h_x, 0, n * sizeof(uint64_t));
    h_x[0] = 4;
    cudaMemcpy(d_x, h_x, n * sizeof(uint64_t), cudaMemcpyHostToDevice);

    uint64_t iters = p - 2;
    time_t t_start_ana = time(NULL);
    time_t t_last_ana  = t_start_ana;
    for (uint64_t i = 0; i < iters; i++) {
        /* progress report */
        {
            time_t t_now = time(NULL);
            if (t_now - t_last_ana >= PROGRESS_INTERVAL) {
                double pct = 100.0 * (double)(i + 1) / (double)iters;
                long elapsed = (long)(t_now - t_start_ana);
                long eta = (elapsed > 0) ? (long)((double)elapsed * (iters - i - 1) / (double)(i + 1)) : -1;
                fprintf(stderr, "  [M_%llu analog] iter %llu/%llu  %.1f%%  elapsed %lds  eta %lds\n",
                        (unsigned long long)p, (unsigned long long)(i + 1),
                        (unsigned long long)iters, pct, elapsed, eta);
                t_last_ana = t_now;
            }
        }
        k_sqr_warp32<<<warp_blocks_sqr, warp_thr_sqr, 0, stream_main>>>(
            d_x, d_lo, d_mi, d_hi, (int)n);
        k_assemble<<<fld_blocks, FOLD_THR, 0, stream_main>>>(
            d_lo, d_mi, d_hi, d_flat, d_ovf, (int)n2);

        cudaEventRecord(ev_asm, stream_main);
        cudaStreamWaitEvent(stream_dma, ev_asm, 0);
        cudaMemcpyAsync(h_flat, d_flat, n2 * sizeof(uint64_t),
                        cudaMemcpyDeviceToHost, stream_dma);
        cudaMemcpyAsync(h_ovf,  d_ovf,  n2 * sizeof(uint8_t),
                        cudaMemcpyDeviceToHost, stream_dma);

        cudaStreamSynchronize(stream_dma);
        cpu_fold_sub2(h_flat, h_ovf, h_x, n, n2, pw, pb);

        cudaMemcpy(d_x, h_x, n * sizeof(uint64_t), cudaMemcpyHostToDevice);
    }

    cudaStreamSynchronize(stream_main);

    int result = 1;
    for (size_t k = 0; k < n; k++)
        if (h_x[k]) { result = 0; break; }

    if (verbose)
        printf("  GPU analog  n_words=%zu  iters=%llu  (32-bit multiply decomp)\n",
               n, (unsigned long long)iters);

    cudaEventDestroy(ev_asm);
    cudaStreamDestroy(stream_main);
    cudaStreamDestroy(stream_dma);
    cudaFreeHost(h_flat); cudaFreeHost(h_ovf); cudaFreeHost(h_x);
    cudaFree(d_x); cudaFree(d_lo); cudaFree(d_mi); cudaFree(d_hi);
    cudaFree(d_flat); cudaFree(d_ovf);
    return result;
}



/* ────────────────────────────────────────────────────────────────────────────
 * k_sqr_fold_limb — combined squaring + Mersenne fold in one kernel pass.
 *
 * HDGL wu-wei inline fold: thread k (0..n-1) computes position k of the
 * squaring result ALREADY folded mod 2^p - 1.
 *
 * For Mersenne modulus M_p = 2^p - 1 (p bits):
 *   The 2n-word product flat[0..2n-1] folds to:
 *     res[k] = flat[k] + flat[k+n] * 2^(n*64-p)   (shifted contribution)
 * But since p may not be a multiple of 64, the fold is at the BIT level.
 *
 * Each thread k computes both:
 *   direct  sum: S_k      = sum_{i=0}^{k}     x[i]*x[k-i]     (position k in flat)
 *   wrapped sum: S_{k+n}  = sum_{i=k-n+1}^{n-1} x[i]*x[k+n-i] (position k+n in flat)
 *
 * The "wrapped" part corresponds to flat[k+n] in the 2n product.  After
 * Mersenne fold: flat[k+n] contributes to res as:
 *   flat[k+n] shifted right by (p - k*64) bits if k*64 < p,
 *   or shifted left by (k*64 - p) bits otherwise.
 *
 * For simplicity and correctness, we store both sums in 2n words (d_lo/d_mi/d_hi
 * for the full 2n) and let k_assemble + k_carry_sub2 handle the fold.
 * The k_sqr_fold_limb path computes the FULL 2n squaring per thread and
 * reduces bandwidth by pre-accumulating both direct and wrapped contributions
 * into n output accumulators.
 *
 * Implementation: thread k (0..n-1) computes:
 *   acc_k = sum_{i+j=k, 0<=i,j<n} x[i]*x[j]       (direct, flat position k)
 *         + sum_{i+j=k+n, 0<=i,j<n} x[i]*x[j]     (wrapped, flat position k+n)
 *
 * The wrapped sum is: sum_{i=k+1}^{n-1} x[i]*x[k+n-i]  (j = k+n-i, j<n → i>k)
 *
 * Both sums are accumulated into the same 3-word accumulator.
 * The Mersenne fold then adds flat[k+n] (= acc_hi[k] contribution) back to
 * flat[k]; but since we've PRE-added it into acc[k], the fold is still needed
 * at the bit level for the pw/pb boundary.
 *
 * For the BIT-level fold at word pw with bit offset pb:
 *   - words 0..pw-1   and 0..pw-1+n are "whole words"
 *   - word pw has partial: only low pb bits belong to lo-half
 *
 * This kernel handles the WHOLE-WORD case only (assumes p % 64 != 0).
 * It writes d_folded[0..n-1] (3-word accumulator per position, 0..n-1 each).
 * ─────────────────────────────────────────────────────────────────────────── */

/* k_carry_sub2 — sequential carry + fold mod 2^p-1 + subtract-2.
 * Must be called after k_assemble with 1 block × 1 thread.
 * Reads d_flat[0..n2-1] + d_ovf[0..n2-1] (cleaned by k_assemble).
 * Writes result into d_x[0..n_words-1].
 * Re-zeros d_flat and d_ovf at the end for the next iteration. */
__global__ void k_carry_sub2(
        uint64_t * __restrict__ d_flat,
        uint8_t  * __restrict__ d_ovf,
        uint64_t * __restrict__ d_x,
        int n_words,
        int p_exp)
{
    int  n   = n_words;
    int  n2  = 2 * n;
    int  pw  = p_exp / 64;
    int  pb  = p_exp % 64;

    /* Step 1: carry propagation through the flat product.
     * d_flat[k] is already the sum of lo[k]+mi[k-1]+hi[k-2],
     * d_ovf[k] holds the overflow (0..2) into position k+1. */
    uint64_t carry = 0;
    for (int k = 0; k < n2; k++) {
        unsigned __int128 s128 = (unsigned __int128)d_flat[k] + carry;
        d_flat[k] = (uint64_t)s128;
        carry     = (uint64_t)(s128 >> 64) + d_ovf[k];
    }
    /* carry holds residual above position n2-1; feeds Mersenne fold */

    /* Step 2: Fold mod 2^p - 1 into d_x.
     * res = flat_lo + flat_hi  where flat_lo = bits[0..p-1], flat_hi = bits[p..2p] */
    for (int k = 0; k < n; k++) d_x[k] = 0;

    for (int k = 0; k < pw && k < n2; k++)
        d_x[k] = d_flat[k];
    if (pb > 0 && pw < n2 && pw < n)
        d_x[pw] = d_flat[pw] & ((1ULL << pb) - 1ULL);

    uint64_t fc = carry;
    for (int k = 0; k < n + 2; k++) {
        int bi = pw + k;
        uint64_t hi_word;
        if (pb == 0) {
            hi_word = (bi < n2) ? d_flat[bi] : 0;
        } else {
            uint64_t lo = (bi   < n2) ? d_flat[bi]   : 0;
            uint64_t hi = (bi+1 < n2) ? d_flat[bi+1] : 0;
            hi_word = (lo >> pb) | (hi << (64 - pb));
        }
        if (k >= n) { fc += hi_word; break; }
        unsigned __int128 s128 = (unsigned __int128)d_x[k] + hi_word + fc;
        d_x[k] = (uint64_t)s128;
        fc = (uint64_t)(s128 >> 64);
    }

    /* Normalize overflow */
    for (;;) {
        uint64_t over = (pb > 0) ? (d_x[n-1] >> pb) : 0;
        if (over) d_x[n-1] &= (1ULL << pb) - 1ULL;
        uint64_t c2 = fc + over;
        fc = 0;
        if (!c2) break;
        for (int k = 0; k < n && c2; k++) {
            unsigned __int128 a = (unsigned __int128)d_x[k] + c2;
            d_x[k] = (uint64_t)a;
            c2 = (uint64_t)(a >> 64);
        }
        fc = c2;
    }

    /* Canonical check: M_p ≡ 0 */
    int is_mp = 1;
    for (int k = 0; k < n && is_mp; k++) {
        uint64_t expected;
        if      (pb == 0) expected = ~0ULL;
        else if (k < pw)  expected = ~0ULL;
        else if (k == pw) expected = (1ULL << pb) - 1ULL;
        else              expected = 0ULL;
        if (d_x[k] != expected) is_mp = 0;
    }
    if (is_mp) for (int k = 0; k < n; k++) d_x[k] = 0;

    /* Subtract 2 mod M_p */
    int   small = 1;
    for (int k = n-1; k >= 1 && small; k--)
        if (d_x[k]) small = 0;
    if (small && d_x[0] >= 2) small = 0;

    if (!small) {
        uint64_t borrow = 2;
        for (int k = 0; k < n && borrow; k++) {
            if (d_x[k] >= borrow) { d_x[k] -= borrow; borrow = 0; }
            else                  { d_x[k] -= borrow; borrow = 1; }
        }
    } else {
        uint64_t val = d_x[0];
        for (int k = 0; k < n; k++) d_x[k] = ~0ULL;
        if (pb > 0) d_x[n-1] = (1ULL << pb) - 1ULL;
        uint64_t borrow = 2 - val;
        for (int k = 0; k < n && borrow; k++) {
            if (d_x[k] >= borrow) { d_x[k] -= borrow; borrow = 0; }
            else                  { d_x[k] -= borrow; borrow = 1; }
        }
    }

    /* Re-zero scratch for next iteration */
    for (int k = 0; k < n2; k++) { d_flat[k] = 0; d_ovf[k] = 0; }
}

/* ── ll_small: p <= 62 ──────────────────────────────────────────────────── */
/* Mersenne fold reduction — no __int128 % (avoids __umodti3 link symbol).  */
/* apa_multiply carry pattern from unified_bigG_fudge10_empiirical_4096bit.c */
static int ll_small(uint64_t p, int verbose) {
    if (p == 2) return 1; /* M_2 = 3, known prime */

    uint64_t mp = (p < 64) ? ((1ULL << p) - 1ULL) : ~0ULL;
    uint64_t s  = 4 % mp;

    for (uint64_t i = 0; i < p - 2; i++) {
        /* s*s via 64×64→128-bit multiply, then Mersenne fold:
         * sq = s*s;  result = (sq & mp) + (sq >> p)
         * mirrors the split in fold_mod_mp and fold26_wuwei carry logic  */
        unsigned __int128 sq  = (unsigned __int128)s * s;  /* mul only, no % */
        uint64_t sq_lo = (uint64_t)sq;                     /* low  64 bits  */
        uint64_t sq_hi = (uint64_t)(sq >> 64);             /* high 64 bits  */
        /* fold at bit p: lo_p = low p bits, hi_p = bits above p */
        uint64_t lo_p  = sq_lo & mp;                       /* bits 0..p-1   */
        uint64_t hi_p  = (sq_lo >> p) | (sq_hi << (64 - p)); /* bits p..127 */
        s = lo_p + hi_p;
        if (s >= mp) s -= mp;
        /* s -= 2 mod M_p */
        if (s >= 2) s -= 2;
        else         s += mp - 2;
    }
    (void)verbose;
    return s == 0;
}

/* ── ll_cpu: 62 < p <= CPU_TH ───────────────────────────────────────────── */
static int ll_cpu(uint64_t p, int verbose) {
    size_t n = (size_t)((p + 63) / 64);

    MPI s;
    mpi_alloc(&s, n);
    s.words[0] = 4;  /* start value */

    for (uint64_t i = 0; i < p - 2; i++) {
        mpi_sqr_mod_mp_cpu(&s, p);
        mpi_sub2_mod_mp(&s, p);
    }

    int result = mpi_is_zero(&s, n);
    if (verbose) {
        printf("  CPU schoolbook  n_words=%zu  iterations=%llu\n",
               n, (unsigned long long)(p - 2));
    }
    mpi_free_m(&s);
    return result;
}

/* ── cpu_fold_sub2: CPU carry + fold mod 2^p-1 + sub2 ───────────────────── */
static void cpu_fold_sub2(uint64_t *h_flat, const uint8_t *h_ovf,
                          uint64_t *h_x, size_t n, size_t n2,
                          int pw, int pb)
{
    /* carry propagation: merge overflow bytes into flat product */
    uint64_t carry = 0;
    for (size_t k = 0; k < n2; k++) {
        unsigned __int128 acc = (unsigned __int128)h_flat[k] + carry;
        h_flat[k] = (uint64_t)acc;
        carry     = (uint64_t)(acc >> 64) + (uint64_t)h_ovf[k];
    }
    /* fold mod 2^p - 1 → h_x */
    for (size_t k = 0; k < n; k++) h_x[k] = 0;
    for (int k = 0; k < pw && k < (int)n2; k++) h_x[k] = h_flat[k];
    if (pb > 0 && pw < (int)n2 && pw < (int)n)
        h_x[pw] = h_flat[pw] & ((1ULL << pb) - 1ULL);
    uint64_t fc = carry;
    for (int k = 0; k < (int)n + 2; k++) {
        int bi = pw + k;
        uint64_t hw;
        if (pb == 0) {
            hw = (bi < (int)n2) ? h_flat[bi] : 0;
        } else {
            uint64_t lo = (bi   < (int)n2) ? h_flat[bi]   : 0;
            uint64_t hi = (bi+1 < (int)n2) ? h_flat[bi+1] : 0;
            hw = (lo >> pb) | (hi << (64 - pb));
        }
        if (k >= (int)n) { fc += hw; break; }
        unsigned __int128 a = (unsigned __int128)h_x[k] + hw + fc;
        h_x[k] = (uint64_t)a;
        fc = (uint64_t)(a >> 64);
    }
    for (;;) {
        uint64_t over = (pb > 0) ? (h_x[n-1] >> pb) : 0;
        if (over) h_x[n-1] &= (1ULL << pb) - 1ULL;
        uint64_t c2 = fc + over;  fc = 0;
        if (!c2) break;
        for (size_t k = 0; k < n && c2; k++) {
            unsigned __int128 a = (unsigned __int128)h_x[k] + c2;
            h_x[k] = (uint64_t)a; c2 = (uint64_t)(a >> 64);
        }
        fc = c2;
    }
    /* canonical M_p ≡ 0 → zero */
    {
        int is_mp = 1;
        for (size_t k = 0; k < n && is_mp; k++) {
            uint64_t expected = (pb == 0) ? ~0ULL
                : (k < (size_t)pw) ? ~0ULL
                : (k == (size_t)pw) ? (1ULL << pb) - 1ULL
                : 0ULL;
            if (h_x[k] != expected) is_mp = 0;
        }
        if (is_mp) memset(h_x, 0, n * sizeof(uint64_t));
    }
    /* subtract 2 mod M_p */
    {
        int small = 1;
        for (size_t k = n-1; k >= 1 && small; k--)
            if (h_x[k]) small = 0;
        if (small && h_x[0] >= 2) small = 0;
        if (!small) {
            uint64_t borrow = 2;
            for (size_t k = 0; k < n && borrow; k++) {
                if (h_x[k] >= borrow) { h_x[k] -= borrow; borrow = 0; }
                else                  { h_x[k] -= borrow; borrow = 1; }
            }
        } else {
            uint64_t val = h_x[0];
            for (size_t k = 0; k < n; k++) h_x[k] = ~0ULL;
            if (pb > 0) h_x[n-1] = (1ULL << pb) - 1ULL;
            uint64_t borrow = 2 - val;
            for (size_t k = 0; k < n && borrow; k++) {
                if (h_x[k] >= borrow) { h_x[k] -= borrow; borrow = 0; }
                else                  { h_x[k] -= borrow; borrow = 1; }
            }
        }
    }
}

/* ── GPU parallel carry: wu-wei function-composition prefix scan ─────────
 *
 * Each limb position k has a carry-transfer function f_k: {0,1,2,3}→{0,1,2,3}
 *   f_k(c) = floor((d_flat[k] + c) / 2^64) + d_ovf[k]
 * packed into a uint8_t (bits [2c+1:2c] = f_k(c), c = 0..3).
 *
 * Parallel carry_in[k] = (f_{k-1} ∘ … ∘ f_0)(0) — exclusive prefix at 0.
 * Three-kernel pipeline: lscan (block prefixes) → bscan (block carry-ins,
 * <<<1,1>>>) → apply (update d_flat in-place) → fold/sub2 (<<<1,1>>>).
 *
 * Eliminates PCIe D2H+H2D round-trip (~120-200 µs/iter on WDDM) with a
 * GPU-only prefix scan (~5-10 µs/iter). Captured in a CUDA graph.
 *
 * Wu-wei principle (fold26_wuwei): each thread expreses its own carry
 * function independently; composition is associative — let the data
 * determine its path through the prefix tree.
 * ──────────────────────────────────────────────────────────────────────── */

#define CFUNC_ID  0xE4u   /* identity: f(c)=c → 0|1<<2|2<<4|3<<6        */
#define CARRY_BLK 256     /* threads per block for carry kernels          */
#define CARRY_WPB (CARRY_BLK / 32)  /* warps per block = 8                */

/* Apply packed carry function: f(c) */
static __device__ __forceinline__ uint8_t cfa(uint8_t f, uint8_t c) {
    return (f >> (c << 1)) & 3u;
}
/* Compose packed carry functions: (b∘a)(c) = b(a(c)) */
static __device__ __forceinline__ uint8_t cfc(uint8_t b, uint8_t a) {
    return (uint8_t)(  cfa(b, cfa(a, 0))
                    | (cfa(b, cfa(a, 1)) << 2)
                    | (cfa(b, cfa(a, 2)) << 4)
                    | (cfa(b, cfa(a, 3)) << 6));
}
/* Build packed carry function from (flat_k, ovf_k):
 * f_k(c) = (flat_k + c overflows 64 bits ? 1 : 0) + ovf_k          */
static __device__ __forceinline__ uint8_t cfm(uint64_t flat, uint8_t ovf) {
    /* c=0: no overflow ever (flat ≤ UINT64_MAX)                             */
    /* c=1: overflow iff flat == 0xFFFF…F                                   */
    /* c=2: overflow iff flat >= 0xFFFF…E                                   */
    /* c=3: overflow iff flat >= 0xFFFF…D                                   */
    uint8_t bc1 = (flat == 0xFFFFFFFFFFFFFFFFULL) ? 1u : 0u;
    uint8_t bc2 = (flat >= 0xFFFFFFFFFFFFFFFEULL) ? 1u : 0u;
    uint8_t bc3 = (flat >= 0xFFFFFFFFFFFFFFFDULL) ? 1u : 0u;
    return  (       ovf        & 3u)
         | (((bc1 + ovf) & 3u) << 2)
         | (((bc2 + ovf) & 3u) << 4)
         | (((bc3 + ovf) & 3u) << 6);
}

/* ─────────────────────────────────────────────────────────────────────────
 * k_carry_lscan — block-level exclusive prefix scan over carry functions.
 * Output d_lpfx[k]   : exclusive prefix within block (identity at k=blk_start).
 * Output d_bagg[blkId]: inclusive block aggregate function.
 * ───────────────────────────────────────────────────────────────────────── */
__global__ void k_carry_lscan(
        const uint64_t * __restrict__ d_flat,
        const uint8_t  * __restrict__ d_ovf,
        uint8_t        * __restrict__ d_lpfx,   /* n2 bytes out */
        uint8_t        * __restrict__ d_bagg,   /* nblocks bytes out */
        int n2)
{
    __shared__ uint8_t s_wagg[CARRY_WPB];  /* warp inclusive aggregates (8) */
    __shared__ uint8_t s_wpfx[CARRY_WPB];  /* warp exclusive prefix (8)     */

    int tid  = (int)threadIdx.x;
    int k    = (int)(blockIdx.x * blockDim.x) + tid;
    int lane = tid & 31;
    int wid  = tid >> 5;

    /* Step 1: local carry function (identity for out-of-bounds) */
    uint8_t fk = (k < n2) ? cfm(d_flat[k], d_ovf[k]) : CFUNC_ID;

    /* Step 2: warp-level inclusive scan via shuffle */
    uint8_t wincl = fk;
    for (int s = 1; s <= 16; s <<= 1) {
        uint8_t prev = (uint8_t)__shfl_up_sync(0xFFFFFFFFu, (unsigned)wincl, s);
        if (lane >= s) wincl = cfc(wincl, prev);
    }
    /* wincl = f_{wbase+lane} ∘ … ∘ f_{wbase}  (inclusive) */

    /* Exclusive per-lane: shift right by 1 */
    uint8_t wexcl = (uint8_t)__shfl_up_sync(0xFFFFFFFFu, (unsigned)wincl, 1);
    if (lane == 0) wexcl = CFUNC_ID;

    /* Step 3: store warp aggregate (inclusive of last real lane in warp) */
    if (lane == 31) s_wagg[wid] = wincl;
    __syncthreads();

    /* Step 4: warp 0 scans the 8 warp aggregates */
    if (wid == 0) {
        uint8_t wf = (lane < CARRY_WPB) ? s_wagg[lane] : CFUNC_ID;
        /* inclusive scan over 8 warps using lanes 0..7 */
        for (int s = 1; s <= 4; s <<= 1) {
            uint8_t prev = (uint8_t)__shfl_up_sync(0xFFFFFFFFu, (unsigned)wf, s);
            if (lane >= s) wf = cfc(wf, prev);
        }
        /* exclusive: shift right by 1 */
        uint8_t wepfx = (uint8_t)__shfl_up_sync(0xFFFFFFFFu, (unsigned)wf, 1);
        if (lane == 0) wepfx = CFUNC_ID;
        if (lane < CARRY_WPB) s_wpfx[lane] = wepfx;
        /* block aggregate = inclusive of last warp (lane CARRY_WPB-1) */
        if (lane == CARRY_WPB - 1) d_bagg[blockIdx.x] = wf;
    }
    __syncthreads();

    /* Step 5: combine warp exclusive prefix + per-lane exclusive = block-exclusive.
     * We want: carry_in[k] = apply(chain, 0) where chain first applies
     * s_wpfx[wid] (all warps before mine), then wexcl (my warp before my lane).
     * cfc(b,a)(c) = b(a(c)), so cfc(wexcl, s_wpfx[wid]) applies s_wpfx first. */
    uint8_t blk_excl = cfc(wexcl, s_wpfx[wid]);
    if (k < n2) d_lpfx[k] = blk_excl;
}

/* ─────────────────────────────────────────────────────────────────────────
 * k_carry_bscan — <<<1,1>>>: serial scan over block aggregates.
 * Writes per-block carry-in (uint8_t) and total top carry (1 byte).
 * ───────────────────────────────────────────────────────────────────────── */
__global__ void k_carry_bscan(
        const uint8_t * __restrict__ d_bagg,
        uint8_t       * __restrict__ d_bcarry,   /* nblocks bytes out */
        uint8_t       * __restrict__ d_topcarry, /* 1 byte out        */
        int nblocks)
{
    uint8_t carry = 0u;
    for (int b = 0; b < nblocks; b++) {
        d_bcarry[b] = carry;
        carry = cfa(d_bagg[b], carry);
    }
    d_topcarry[0] = carry;
}

/* ─────────────────────────────────────────────────────────────────────────
 * k_carry_apply — apply per-element carry-ins and update d_flat in-place.
 *   carry_in[k] = apply(d_lpfx[k], d_bcarry[blockIdx.x])
 *   d_flat[k]  += carry_in[k]   (unsigned mod 2^64 — intentional wrap)
 * ───────────────────────────────────────────────────────────────────────── */
__global__ void k_carry_apply(
        uint64_t       * __restrict__ d_flat,
        const uint8_t  * __restrict__ d_lpfx,
        const uint8_t  * __restrict__ d_bcarry,
        int n2)
{
    int k = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (k >= n2) return;
    uint8_t cin = cfa(d_lpfx[k], d_bcarry[blockIdx.x]);
    d_flat[k] += (uint64_t)cin;   /* correct mod-2^64 low word of (orig+carry) */
}

/* ─────────────────────────────────────────────────────────────────────────
 * k_fold_sub2_gpu — fold carry-propagated flat array mod 2^p-1, subtract 2.
 *
 * Launched as <<<1, CARRY_BLK, n2*sizeof(uint64_t)>>>.
 * All CARRY_BLK threads cooperatively load d_flat into shared memory
 * (parallel global→L1/shmem at ~full bandwidth).  Thread 0 then executes
 * the serial fold from shared memory at ~4-cycle-per-word shmem speed
 * (vs ~300-cycle global memory latency per sequential dependent load).
 * ───────────────────────────────────────────────────────────────────────── */
__global__ void k_fold_sub2_gpu(
        const uint64_t * __restrict__ d_flat,
        const uint8_t  * __restrict__ d_topcarry,
        uint64_t       * __restrict__ d_x,
        int n_words, int p_exp)
{
    /* Cooperative parallel load: all CARRY_BLK threads fill shmem.
     * Skipped when blockDim.x == 1 (shmem limit fallback). */
    extern __shared__ uint64_t s_flat[];
    int n  = n_words;
    int n2 = 2 * n;

    if (blockDim.x > 1) {
        for (int i = (int)threadIdx.x; i < n2; i += (int)blockDim.x)
            s_flat[i] = d_flat[i];
        __syncthreads();
    }

    /* Thread 0 folds.  Read from shmem (fast) if cooperative, d_flat otherwise. */
    if (threadIdx.x == 0) {
        const uint64_t *src = (blockDim.x > 1) ? s_flat : d_flat;
        int pw = p_exp / 64;
        int pb = p_exp % 64;
        uint64_t fc = (uint64_t)d_topcarry[0];

        /* Fold lo-half into d_x */
        for (int k = 0; k < n;  k++) d_x[k] = 0;
        for (int k = 0; k < pw && k < n2; k++) d_x[k] = src[k];
        if (pb > 0 && pw < n2 && pw < n)
            d_x[pw] = src[pw] & ((1ULL << pb) - 1ULL);

        /* Add hi-half shifted right by pb through bit boundary at pw */
        for (int k = 0; k < n + 2; k++) {
            int bi = pw + k;
            uint64_t hw;
            if (pb == 0) {
                hw = (bi < n2) ? src[bi] : 0;
            } else {
                uint64_t lo = (bi   < n2) ? src[bi]   : 0;
                uint64_t hi = (bi+1 < n2) ? src[bi+1] : 0;
                hw = (lo >> pb) | (hi << (64 - pb));
            }
            if (k >= n) { fc += hw; break; }
            unsigned __int128 s128 = (unsigned __int128)d_x[k] + hw + fc;
            d_x[k] = (uint64_t)s128;
            fc = (uint64_t)(s128 >> 64);
        }

        /* Normalize residual overflow */
        for (;;) {
            uint64_t over = (pb > 0) ? (d_x[n-1] >> pb) : 0;
            if (over) d_x[n-1] &= (1ULL << pb) - 1ULL;
            uint64_t c2 = fc + over;
            fc = 0;
            if (!c2) break;
            for (int k = 0; k < n && c2; k++) {
                unsigned __int128 a = (unsigned __int128)d_x[k] + c2;
                d_x[k] = (uint64_t)a;
                c2 = (uint64_t)(a >> 64);
            }
            fc = c2;
        }

        /* Canonical M_p → 0 */
        int is_mp = 1;
        for (int k = 0; k < n && is_mp; k++) {
            uint64_t expected;
            if      (pb == 0) expected = ~0ULL;
            else if (k < pw)  expected = ~0ULL;
            else if (k == pw) expected = (1ULL << pb) - 1ULL;
            else              expected = 0ULL;
            if (d_x[k] != expected) is_mp = 0;
        }
        if (is_mp) for (int k = 0; k < n; k++) d_x[k] = 0;

        /* Subtract 2 mod M_p */
        int small = 1;
        for (int k = n-1; k >= 1 && small; k--)
            if (d_x[k]) small = 0;
        if (small && d_x[0] >= 2) small = 0;

        if (!small) {
            uint64_t borrow = 2;
            for (int k = 0; k < n && borrow; k++) {
                if (d_x[k] >= borrow) { d_x[k] -= borrow; borrow = 0; }
                else                  { d_x[k] -= borrow; borrow = 1; }
            }
        } else {
            uint64_t val = d_x[0];
            for (int k = 0; k < n; k++) d_x[k] = ~0ULL;
            if (pb > 0) d_x[n-1] = (1ULL << pb) - 1ULL;
            uint64_t borrow = 2 - val;
            for (int k = 0; k < n && borrow; k++) {
                if (d_x[k] >= borrow) { d_x[k] -= borrow; borrow = 0; }
                else                  { d_x[k] -= borrow; borrow = 1; }
            }
        }
    }  /* end thread 0 */
}

/* ── ll_gpu: p > CPU_TH — async stream pipeline ──────────────────────────
 *
 * HDGL wu-wei streaming: GPU squaring (parallel) + CPU fold (sequential).
 * Uses a CUDA stream so cudaMemcpyAsync D2H overlaps with CPU fold work.
 *
 * Per iteration:
 *   1. k_sqr_limb + k_assemble on stream_main (parallel GPU squaring)
 *   2. Record ev_asm after assembly finishes
 *   3. stream_dma waits for ev_asm, then async D2H of d_flat (n2×8) + d_ovf (n2×1)
 *   4. CPU fold+sub2 → h_x  (overlaps with early part of DMA)
 *   5. stream_dma sync → upload h_x → d_x synchronously
 *
 * Bandwidth: ~12KB D2H (n2×9B) + ~5.6KB H2D (n×8B) vs old ~38KB/iter
 * ── */
static int ll_gpu(uint64_t p, int verbose) {
    size_t n  = (size_t)((p + 63) / 64);
    size_t n2 = 2 * n;

    uint64_t *d_x    = NULL;
    uint64_t *d_lo   = NULL;
    uint64_t *d_mi   = NULL;
    uint64_t *d_hi   = NULL;
    uint64_t *d_flat = NULL;
    uint8_t  *d_ovf  = NULL;
    cudaMalloc(&d_x,    n  * sizeof(uint64_t));
    cudaMalloc(&d_lo,   n2 * sizeof(uint64_t));
    cudaMalloc(&d_mi,   n2 * sizeof(uint64_t));
    cudaMalloc(&d_hi,   n2 * sizeof(uint64_t));
    cudaMalloc(&d_flat, n2 * sizeof(uint64_t));
    cudaMalloc(&d_ovf,  n2 * sizeof(uint8_t));

    uint64_t *h_flat = NULL;
    uint8_t  *h_ovf  = NULL;
    uint64_t *h_x    = NULL;
    cudaHostAlloc(&h_flat, n2 * sizeof(uint64_t), cudaHostAllocDefault);
    cudaHostAlloc(&h_ovf,  n2 * sizeof(uint8_t),  cudaHostAllocDefault);
    cudaHostAlloc(&h_x,    n  * sizeof(uint64_t), cudaHostAllocDefault);

    cudaStream_t stream_main, stream_dma;
    cudaStreamCreate(&stream_main);
    cudaStreamCreate(&stream_dma);
    cudaEvent_t ev_asm;
    cudaEventCreate(&ev_asm);

    int sqr_thr    = 256;
    int sqr_blocks = (int)((n2 + sqr_thr  - 1) / sqr_thr);
    int fld_blocks = (int)((n2 + FOLD_THR - 1) / FOLD_THR);
    int pw = (int)(p / 64);
    int pb = (int)(p % 64);

    /* k_sqr_warp: one warp (32 threads) per output limb */
    int warp_blocks_sqr = (int)((n2 + WARP_PER_BLOCK - 1) / WARP_PER_BLOCK);
    int warp_thr_sqr    = WARP_PER_BLOCK * WARP_SIZE;  /* = 256 */

    memset(h_x, 0, n * sizeof(uint64_t));
    h_x[0] = 4;
    cudaMemcpy(d_x, h_x, n * sizeof(uint64_t), cudaMemcpyHostToDevice);

    uint64_t iters = p - 2;
    time_t t_start_gpu = time(NULL);
    time_t t_last_gpu  = t_start_gpu;
    for (uint64_t i = 0; i < iters; i++) {
        /* progress report: fired at most once per PROGRESS_INTERVAL seconds;
         * time() is a single syscall — negligible vs the D2H round-trip.  */
        {
            time_t t_now = time(NULL);
            if (t_now - t_last_gpu >= PROGRESS_INTERVAL) {
                double pct = 100.0 * (double)(i + 1) / (double)iters;
                long elapsed = (long)(t_now - t_start_gpu);
                long eta = (elapsed > 0) ? (long)((double)elapsed * (iters - i - 1) / (double)(i + 1)) : -1;
                fprintf(stderr, "  [M_%llu] iter %llu/%llu  %.1f%%  elapsed %lds  eta %lds\n",
                        (unsigned long long)p, (unsigned long long)(i + 1),
                        (unsigned long long)iters, pct, elapsed, eta);
                t_last_gpu = t_now;
            }
        }
        /* GPU parallel squaring (warp-reduced, no atomics) + assembly */
        k_sqr_warp<<<warp_blocks_sqr, warp_thr_sqr, 0, stream_main>>>(
            d_x, d_lo, d_mi, d_hi, (int)n);
        k_assemble<<<fld_blocks, FOLD_THR, 0, stream_main>>>(
            d_lo, d_mi, d_hi, d_flat, d_ovf, (int)n2);

        /* Record event when assembly is done */
        cudaEventRecord(ev_asm, stream_main);

        /* Async D2H on stream_dma (runs while stream_main is free) */
        cudaStreamWaitEvent(stream_dma, ev_asm, 0);
        cudaMemcpyAsync(h_flat, d_flat, n2 * sizeof(uint64_t),
                        cudaMemcpyDeviceToHost, stream_dma);
        cudaMemcpyAsync(h_ovf,  d_ovf,  n2 * sizeof(uint8_t),
                        cudaMemcpyDeviceToHost, stream_dma);

        /* Wait for DMA to complete, then CPU fold+sub2 */
        cudaStreamSynchronize(stream_dma);
        cpu_fold_sub2(h_flat, h_ovf, h_x, n, n2, pw, pb);

        /* Upload new s to device (synchronous — must be done before next sqr) */
        cudaMemcpy(d_x, h_x, n * sizeof(uint64_t), cudaMemcpyHostToDevice);
    }

    cudaStreamSynchronize(stream_main);

    int result = 1;
    for (size_t k = 0; k < n; k++)
        if (h_x[k]) { result = 0; break; }

    if (verbose)
        printf("  GPU stream  n_words=%zu  iters=%llu  (2 kernels + CPU fold/sub2)\n",
               n, (unsigned long long)iters);

    cudaEventDestroy(ev_asm);
    cudaStreamDestroy(stream_main);
    cudaStreamDestroy(stream_dma);
    cudaFreeHost(h_flat); cudaFreeHost(h_ovf); cudaFreeHost(h_x);
    cudaFree(d_x); cudaFree(d_lo); cudaFree(d_mi); cudaFree(d_hi);
    cudaFree(d_flat); cudaFree(d_ovf);
    return result;
}

/* ── ll_gpu_gpucarry: schoolbook squaring + on-device parallel carry scan ─
 *
 * Same k_sqr_warp + k_assemble squaring as ll_gpu, but replaces the
 * PCIe D2H/H2D round-trip with an all-on-device carry pipeline:
 *
 *   k_sqr_warp   → d_lo, d_mi, d_hi
 *   k_assemble   → d_flat[0..n2-1], d_ovf[0..n2-1]
 *   k_carry_lscan→ d_lpfx[0..n2-1], d_bagg[0..nblks-1]  (block prefix fns)
 *   k_carry_bscan→ d_bcarry[0..nblks-1], d_topcarry[0]   (<<<1,1>>>)
 *   k_carry_apply→ d_flat updated in-place (low 64 bits of orig+carry_in)
 *   k_fold_sub2_gpu → d_x updated in-place                (<<<1,1>>>)
 *
 * All six launches captured in a CUDA graph → single cudaGraphLaunch per
 * iteration, ~1 µs overhead vs ~150 µs PCIe round-trip on Windows/WDDM.
 * No cudaMemcpy inside the iteration loop.
 * ──────────────────────────────────────────────────────────────────────── */
static int ll_gpu_gpucarry(uint64_t p, int verbose) {
    size_t n  = (size_t)((p + 63) / 64);
    size_t n2 = 2 * n;
    int pw = (int)(p / 64);
    int pb = (int)(p % 64);

    int nblks     = (int)((n2 + CARRY_BLK - 1) / CARRY_BLK);
    int warp_blks = (int)((n2 + WARP_PER_BLOCK - 1) / WARP_PER_BLOCK);
    int warp_thr  = WARP_PER_BLOCK * WARP_SIZE;
    int fld_blks  = (int)((n2 + FOLD_THR    - 1) / FOLD_THR);
    int cblks     = (int)((n2 + CARRY_BLK   - 1) / CARRY_BLK);

    /* Device buffers */
    uint64_t *d_x = NULL, *d_lo = NULL, *d_mi = NULL, *d_hi = NULL;
    uint64_t *d_flat = NULL;
    uint8_t  *d_ovf  = NULL;
    uint8_t  *d_lpfx = NULL, *d_bagg = NULL;
    uint8_t  *d_bcarry = NULL, *d_topcarry = NULL;

    cudaMalloc(&d_x,       n  * sizeof(uint64_t));
    cudaMalloc(&d_lo,      n2 * sizeof(uint64_t));
    cudaMalloc(&d_mi,      n2 * sizeof(uint64_t));
    cudaMalloc(&d_hi,      n2 * sizeof(uint64_t));
    cudaMalloc(&d_flat,    n2 * sizeof(uint64_t));
    cudaMalloc(&d_ovf,     n2 * sizeof(uint8_t));
    cudaMalloc(&d_lpfx,    n2 * sizeof(uint8_t));
    cudaMalloc(&d_bagg,    (size_t)nblks * sizeof(uint8_t));
    cudaMalloc(&d_bcarry,  (size_t)nblks * sizeof(uint8_t));
    cudaMalloc(&d_topcarry, 1  * sizeof(uint8_t));

    /* Set initial LL state: x = 4 */
    cudaMemset(d_x, 0, n * sizeof(uint64_t));
    {
        uint64_t four = 4;
        cudaMemcpy(d_x, &four, sizeof(uint64_t), cudaMemcpyHostToDevice);
    }

    /* Configure k_fold_sub2_gpu for dynamic shmem: n2 uint64_t per block.
     * sm_75 supports up to 96 KB (98304 B) per block.  For p that exceeds
     * this limit (p > ~393K), fall back to a single-thread global-memory
     * fold (slower, but correct — same kernel, blockDim.x==1 branch).    */
    size_t fold_smem = n2 * sizeof(uint64_t);
    int    fold_thr  = CARRY_BLK;   /* cooperative shmem: 256 threads */
    if (fold_smem > 98304u) { fold_smem = 0; fold_thr = 1; }  /* fallback */
    else cudaFuncSetAttribute(k_fold_sub2_gpu,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              (int)fold_smem);

    /* Capture the 6-kernel pipeline into a CUDA graph */
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaGraph_t     graph = NULL;
    cudaGraphExec_t gexec = NULL;

    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

    k_sqr_warp<<<warp_blks, warp_thr, 0, stream>>>(
        d_x, d_lo, d_mi, d_hi, (int)n);
    k_assemble<<<fld_blks, FOLD_THR, 0, stream>>>(
        d_lo, d_mi, d_hi, d_flat, d_ovf, (int)n2);
    k_carry_lscan<<<cblks, CARRY_BLK, 0, stream>>>(
        d_flat, d_ovf, d_lpfx, d_bagg, (int)n2);
    k_carry_bscan<<<1, 1, 0, stream>>>(
        d_bagg, d_bcarry, d_topcarry, nblks);
    k_carry_apply<<<cblks, CARRY_BLK, 0, stream>>>(
        d_flat, d_lpfx, d_bcarry, (int)n2);
    k_fold_sub2_gpu<<<1, fold_thr, fold_smem, stream>>>(
        d_flat, d_topcarry, d_x, (int)n, (int)p);

    cudaStreamEndCapture(stream, &graph);
    cudaGraphInstantiate(&gexec, graph, NULL, NULL, 0);
    cudaGraphDestroy(graph);

    uint64_t iters = p - 2;
    time_t t_start = time(NULL);
    time_t t_last  = t_start;

    for (uint64_t i = 0; i < iters; i++) {
        {
            time_t t_now = time(NULL);
            if (t_now - t_last >= PROGRESS_INTERVAL) {
                double pct = 100.0 * (double)(i + 1) / (double)iters;
                long elapsed = (long)(t_now - t_start);
                long eta = (elapsed > 0) ? (long)((double)elapsed * (iters - i - 1) / (double)(i + 1)) : -1;
                fprintf(stderr, "  [M_%llu gpucarry] iter %llu/%llu  %.1f%%  elapsed %lds  eta %lds\n",
                        (unsigned long long)p, (unsigned long long)(i + 1),
                        (unsigned long long)iters, pct, elapsed, eta);
                t_last = t_now;
            }
        }
        cudaGraphLaunch(gexec, stream);
    }
    cudaStreamSynchronize(stream);

    /* Read result: prime iff d_x == 0 */
    uint64_t *h_x = (uint64_t *)calloc(n, sizeof(uint64_t));
    cudaMemcpy(h_x, d_x, n * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    int result = 1;
    for (size_t k = 0; k < n; k++)
        if (h_x[k]) { result = 0; break; }
    free(h_x);

    if (verbose)
        printf("  GPU gpucarry  n_words=%zu  iters=%llu"
               "  (warp-sqr + parallel carry scan + on-device fold, CUDA graph)\n",
               n, (unsigned long long)iters);

    cudaGraphExecDestroy(gexec);
    cudaStreamDestroy(stream);
    cudaFree(d_x);    cudaFree(d_lo);   cudaFree(d_mi);  cudaFree(d_hi);
    cudaFree(d_flat); cudaFree(d_ovf);
    cudaFree(d_lpfx); cudaFree(d_bagg);
    cudaFree(d_bcarry); cudaFree(d_topcarry);
    return result;
}

/* ── ll_gpu_persistent: single kernel launch, all iterations on-device ─── */
static int ll_gpu_persistent(uint64_t p, int verbose) {
    int n     = (int)((p + 63) / 64);
    int n2    = 2 * n;
    int iters = (int)(p - 2);

    uint64_t *d_s  = NULL;
    uint64_t *d_lo = NULL;
    uint64_t *d_mi = NULL;
    uint64_t *d_hi = NULL;
    cudaMalloc(&d_s,  (size_t)n  * sizeof(uint64_t));
    cudaMalloc(&d_lo, (size_t)n2 * sizeof(uint64_t));
    cudaMalloc(&d_mi, (size_t)n2 * sizeof(uint64_t));
    cudaMalloc(&d_hi, (size_t)n2 * sizeof(uint64_t));

    /* init s = 4; zero scratch */
    cudaMemset(d_s,  0, (size_t)n  * sizeof(uint64_t));
    cudaMemset(d_lo, 0, (size_t)n2 * sizeof(uint64_t));
    cudaMemset(d_mi, 0, (size_t)n2 * sizeof(uint64_t));
    cudaMemset(d_hi, 0, (size_t)n2 * sizeof(uint64_t));
    uint64_t init_val = 4;
    cudaMemcpy(d_s, &init_val, sizeof(uint64_t), cudaMemcpyHostToDevice);

    /* shared memory: smem_s[n_words] */
    size_t smem_bytes = (size_t)n * sizeof(uint64_t);
    fprintf(stderr, "  [M_%llu persistent] running %d iters in single kernel launch — no per-iter progress\n",
            (unsigned long long)p, iters);
    k_ll_persistent_block<<<1, BLOCK_SZ, smem_bytes>>>(
            d_s, d_lo, d_mi, d_hi, n, (int)p, iters);
    cudaDeviceSynchronize();

    uint64_t *h_s = (uint64_t *)calloc((size_t)n, sizeof(uint64_t));
    cudaMemcpy(h_s, d_s, (size_t)n * sizeof(uint64_t), cudaMemcpyDeviceToHost);

    int result = 1;
    for (int k = 0; k < n; k++)
        if (h_s[k]) { result = 0; break; }

    if (verbose)
        printf("  GPU persistent  n_words=%d  iters=%d  (1 launch, all on-device)\n",
               n, iters);

    free(h_s);
    cudaFree(d_s); cudaFree(d_lo); cudaFree(d_mi); cudaFree(d_hi);
    return result;
}

/* ── top-level test ─────────────────────────────────────────────────────── */
static int ll_test(uint64_t p, int verbose) {
    if (verbose) {
        printf("M_%llu: testing...\n", (unsigned long long)p);
        report_resonance(p);
    }

    /* Resolve auto-select: NTT wins above NTT_AUTO_THRESHOLD.             *
     * Explicit --squaring flag always overrides auto.                     */
    int use_ntt;
    if      (g_squaring == 1)  use_ntt = 1;
    else if (g_squaring == 0)  use_ntt = 0;
    else                       use_ntt = (p >= NTT_AUTO_THRESHOLD);

    if (verbose && p > CPU_TH && !g_use_analog_gpu && g_precision != 32
                && !g_use_persistent && g_squaring == -1)
        printf("  [auto-select] p=%llu %s NTT_AUTO_THRESHOLD=%u -> %s\n",
               (unsigned long long)p, use_ntt ? ">=" : "<",
               NTT_AUTO_THRESHOLD, use_ntt ? "NTT" : "schoolbook");

    int result;
    if      (p <= 62 && g_squaring != 3)     result = ll_small(p, verbose);
    else if (p <= CPU_TH && g_squaring != 3) result = ll_cpu(p, verbose);
    else if (g_squaring == 3)                result = ll_analog(p, verbose);  /* v30b+Kuramoto */
    else if (use_ntt)                                 result = ll_gpu_ntt(p, verbose);
    else if (g_squaring == 2)                         result = ll_gpu_gpucarry(p, verbose);
    else if (g_squaring == 0)                         result = ll_gpu(p, verbose);
    else if (g_use_analog_gpu || g_precision == 32)  result = ll_gpu_analog(p, verbose);
    else if (g_use_persistent)   result = ll_gpu_persistent(p, verbose);
    else                         result = ll_gpu_gpucarry(p, verbose);  /* auto default */

    if (verbose)
        printf("  => %s\n\n", result ? "PRIME" : "COMPOSITE");
    return result;
}

/* ── self-test ──────────────────────────────────────────────────────────── */
static void run_selftest(void) {
    /* Known Mersenne prime exponents */
    static const uint64_t PRIMES_K[] = {
        2, 3, 5, 7, 13, 17, 19, 31, 61, 89,
        107, 127, 521, 607, 1279, 2203, 2281, 3217, 4253, 4423
    };
    /* Known composite Mersenne exponents (M_p is NOT prime) */
    static const uint64_t COMPOSITES_K[] = { 11, 23, 29, 37, 41 };

    int pass = 0, fail = 0;
    clock_t t0 = clock();

    printf("=== Lucas-Lehmer self-test ===\n");

    for (size_t i = 0; i < sizeof(PRIMES_K)/sizeof(PRIMES_K[0]); i++) {
        uint64_t p = PRIMES_K[i];
        int r = ll_test(p, 0);
        if (r) {
            printf("  M_%-5llu  PRIME    [OK]\n", (unsigned long long)p);
            pass++;
        } else {
            printf("  M_%-5llu  COMPOSITE  [FAIL - expected PRIME]\n",
                   (unsigned long long)p);
            fail++;
        }
    }

    for (size_t i = 0; i < sizeof(COMPOSITES_K)/sizeof(COMPOSITES_K[0]); i++) {
        uint64_t p = COMPOSITES_K[i];
        int r = ll_test(p, 0);
        if (!r) {
            printf("  M_%-5llu  COMPOSITE [OK]\n", (unsigned long long)p);
            pass++;
        } else {
            printf("  M_%-5llu  PRIME     [FAIL - expected COMPOSITE]\n",
                   (unsigned long long)p);
            fail++;
        }
    }

    double elapsed = (double)(clock() - t0) / CLOCKS_PER_SEC;
    printf("\nSelf-test: %d passed, %d failed  (%.2fs)\n", pass, fail, elapsed);
}

/* ── main ───────────────────────────────────────────────────────────────── */
int main(int argc, char **argv) {
    if (argc < 2) {
        printf("Usage:\n");
        printf("  ll_mpi.exe --selftest [--persistent] [--precision <32|64>] [--squaring <schoolbook|ntt>]\n");
        printf("  ll_mpi.exe <p> [--verbose] [--analog] [--persistent] [--precision <32|64>] [--squaring <schoolbook|ntt>]\n");
        printf("  ll_mpi.exe --gpu-info\n");
        printf("\n");
        printf("  --precision 64       64-bit warp squaring via __int128 (default, fastest)\n");
        printf("  --precision 32       32-bit half-multiply decomposition (same as --analog flag)\n");
        printf("  --squaring schoolbook  O(n^2) schoolbook multiply (GPU, default)\n");
        printf("  --squaring ntt         O(n log n) NTT squaring over Z/(2^64-2^32+1)\n");
        printf("  --squaring gpucarry    on-device parallel carry scan (auto default)\n");
        printf("  --squaring analog      v30b Slot4096 APA + 8D Kuramoto oscillator (CPU, CUDA-free)\n");
        return 0;
    }

    /* scan all flags first so --selftest can honour --persistent etc. */
    int do_selftest = 0;
    int do_gpuinfo  = 0;
    int verbose     = 0;
    int analog      = 0;
    int persistent  = 0;
    int precision   = 64;   /* default: 64-bit warp squaring */
    uint64_t p_arg  = 0;
    for (int i = 1; i < argc; i++) {
        if      (strcmp(argv[i], "--selftest")   == 0) do_selftest = 1;
        else if (strcmp(argv[i], "--gpu-info")   == 0) do_gpuinfo  = 1;
        else if (strcmp(argv[i], "--verbose")    == 0) verbose     = 1;
        else if (strcmp(argv[i], "--analog")     == 0) { analog = 1; precision = 32; }
        else if (strcmp(argv[i], "--persistent") == 0) persistent  = 1;
        else if (strcmp(argv[i], "--precision")  == 0) {
            if (i + 1 < argc) {
                int pv = atoi(argv[++i]);
                if (pv == 32 || pv == 64) precision = pv;
                else { fprintf(stderr, "--precision must be 32 or 64\n"); return 1; }
            } else { fprintf(stderr, "--precision requires a value (32 or 64)\n"); return 1; }
        }
        else if (strcmp(argv[i], "--squaring") == 0) {
            if (i + 1 < argc) {
                const char *sv = argv[++i];
                if      (strcmp(sv, "ntt")        == 0) g_squaring = 1;
                else if (strcmp(sv, "schoolbook") == 0) g_squaring = 0;
                else if (strcmp(sv, "auto")       == 0) g_squaring = -1;
                else if (strcmp(sv, "gpucarry")   == 0) g_squaring = 2;
                else if (strcmp(sv, "analog")      == 0) g_squaring = 3;
                else { fprintf(stderr, "--squaring must be 'auto', 'schoolbook', 'ntt', 'gpucarry', or 'analog'\n"); return 1; }
            } else { fprintf(stderr, "--squaring requires a value (auto|schoolbook|ntt|gpucarry|analog)\n"); return 1; }
        }
        else p_arg = (uint64_t)strtoull(argv[i], NULL, 10);
    }
    g_use_analog_gpu = analog;
    g_use_persistent = persistent;
    g_precision      = precision;

    if (do_selftest) {
        run_selftest();
        return 0;
    }

    if (do_gpuinfo) {
        int dev_count = 0;
        cudaGetDeviceCount(&dev_count);
        printf("CUDA devices: %d\n", dev_count);
        for (int i = 0; i < dev_count; i++) {
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, i);
            printf("  [%d] %s  sm_%d%d  %.0f MB  warp=%d\n",
                   i, prop.name,
                   prop.major, prop.minor,
                   prop.totalGlobalMem / 1048576.0,
                   prop.warpSize);
        }
        return 0;
    }

    uint64_t p = p_arg;
    if (p < 2) { fprintf(stderr, "p must be >= 2\n"); return 1; }

    int result = ll_test(p, verbose);

    if (!verbose)
        printf("M_%llu: %s\n", (unsigned long long)p,
               result ? "PRIME" : "COMPOSITE");
    return result ? 0 : 1;
}
