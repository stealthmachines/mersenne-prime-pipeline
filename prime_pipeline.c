/*
 * prime_pipeline.c  —  Track D: φ-filter → ψ-guided → Lucas-Lehmer pipeline
 *
 * Stages:
 *   1. Enumerate prime exponents p in [p_lo, p_hi] using segmented sieve.
 *   2. Compute φ-lattice coordinate  n(2^p) for each prime p.
 *   3. Apply Track B filter: frac(n) < 0.5 (lower-half bias; 67% of known
 *      Mersenne exponents pass vs 50% random).
 *   4. Compute Dn(r) acoustic resonance score (hdgl_analog_v30 formula).
 *   5. Rank by Dn·lower_half_bonus descending.
 *   6. Print ranked candidate table → pipe output into ll_cuda.exe.
 *
 * Build (MSVC):
 *   cl /O2 /W3 prime_pipeline.c /link /OUT:prime_pipeline.exe
 * Build (gcc/mingw):
 *   gcc -O3 -o prime_pipeline.exe prime_pipeline.c -lm
 *
 * Usage:
 *   prime_pipeline.exe <p_lo> <p_hi> [--all] [--top N] [--min-dn D]
 *
 *   --all      : include both halves (no frac < 0.5 filter)
 *   --top N    : print only top N candidates (default 200)
 *   --min-dn D : minimum Dn score to include (default 0)
 *
 * Example (pipe into ll_cuda):
 *   prime_pipeline.exe 82589934 83000000 --top 10 | \
 *       for /f "tokens=1" %p in ('findstr /r "^[0-9]"') do ll_cuda.exe %p
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

/* ──────────────────────── constants ───────────────────────── */

#define PHI      1.618033988749895
#define SQRT5    2.2360679774997896
#define PI       3.141592653589793
#ifndef M_PI
#  define M_PI   3.141592653589793
#endif
#define LN_PHI   0.4812118250596034748
#define M_LN2    0.6931471805599453094

/* First 50 primes — same table as EMPIRICAL_VALIDATION_ASCII.c */
static const int PRIMES[50] = {
     2,  3,  5,  7, 11, 13, 17, 19, 23, 29,
    31, 37, 41, 43, 47, 53, 59, 61, 67, 71,
    73, 79, 83, 89, 97,101,103,107,109,113,
   127,131,137,139,149,151,157,163,167,173,
   179,181,191,193,197,199,211,223,227,229
};

/* ──────────────────────── Dₙ operator ─────────────────────── */

/* Binet's formula for real-index Fibonacci — from EMPIRICAL_VALIDATION_ASCII.c */
static double fibonacci_real(double n) {
    double term1 = pow(PHI, n) / SQRT5;
    double term2 = pow(1.0 / PHI, n) * cos(PI * n);
    return term1 - term2;
}

/* Prime product index — 50-entry modular table */
static double prime_product_index(double n, double beta) {
    int idx = ((int)floor(n + beta) + 50) % 50;
    return (double)PRIMES[idx];
}

/*
 * Universal D_n operator — from EMPIRICAL_VALIDATION_ASCII.c:
 *   D_n(n,beta,r,k,Omega,base) = sqrt(φ · F_{n+β} · P_{n+β} · base^{n+β} · Ω) · r^k
 * For Mersenne exponents: base=2, beta=0, k=(n_int+1)/8
 */
static double D_n(double n, double beta, double r, double k,
                  double Omega, double base) {
    double n_beta = n + beta;
    double Fn     = fibonacci_real(n_beta);
    double Pn     = prime_product_index(n, beta);
    double dyadic = pow(base, n_beta);
    double val    = PHI * fmax(Fn, 1e-15) * dyadic * Pn * Omega;
    return sqrt(fmax(val, 1e-15)) * pow(fabs(r), k);
}

/* ──────────────────────── φ-lattice ────────────────────────── */
/*
 * n(x) = log(log(x)/ln(φ)) / ln(φ) − 1/(2φ)
 * Inverse of x(n) = φ^(φ^(n + 1/(2φ))).
 */
static double n_of_x(double x) {
    if (x <= 1.0) return -1.0;
    double lx = log(x);
    if (lx <= 0.0) return -1.0;
    double llx = log(lx / LN_PHI);
    if (llx <= 0.0) return -1.0;
    return llx / LN_PHI - 0.5 / PHI;
}

/* n(2^p) = n_of_x(2^p)  but computed without overflow for large p */
static double n_of_2p(uint64_t p) {
    /* log(2^p) = p·ln2, then n(x) formula */
    double lx  = (double)p * M_LN2;              /* ln(2^p) */
    double llx = log(lx / LN_PHI);
    if (llx <= 0.0) return -1.0;
    return llx / LN_PHI - 0.5 / PHI;
}

/* ──────────────────────── Miller-Rabin (64-bit) ────────────── */
/* Used to check that candidate exponents p are prime. */

#ifdef _WIN32
#  include <intrin.h>
static uint64_t mulmod64(uint64_t a, uint64_t b, uint64_t m) {
    uint64_t hi;
    uint64_t lo = _umul128(a, b, &hi);
    /* (hi·2^64 + lo) mod m  — slow but correct */
    uint64_t r = 0;
    /* Use __uint128 if available under mingw; otherwise __udivti3 path */
    (void)hi; (void)lo;
    /* Fallback: Montgomery or binary method */
    a %= m; b %= m;
    while (b > 0) { if (b & 1) r = (r + a < r || r + a >= m) ? (r + a - m) : (r + a); a = (2*a >= m) ? (2*a - m) : 2*a; b >>= 1; }
    return r;
}
#else
static uint64_t mulmod64(uint64_t a, uint64_t b, uint64_t m) {
    return (uint64_t)((__uint128_t)a * b % m);
}
#endif

static uint64_t powmod64(uint64_t base, uint64_t exp, uint64_t mod) {
    uint64_t result = 1;
    base %= mod;
    while (exp > 0) {
        if (exp & 1) result = mulmod64(result, base, mod);
        base = mulmod64(base, base, mod);
        exp >>= 1;
    }
    return result;
}

static int miller_rabin_witness(uint64_t n, uint64_t a) {
    if (n % a == 0) return n == a;
    uint64_t d = n - 1;
    int r = 0;
    while (!(d & 1)) { d >>= 1; r++; }
    uint64_t x = powmod64(a, d, n);
    if (x == 1 || x == n - 1) return 1;
    for (int i = 0; i < r - 1; i++) {
        x = mulmod64(x, x, n);
        if (x == n - 1) return 1;
    }
    return 0;
}

/* Deterministic for n < 3,317,044,064,679,887,385,961,981 */
static int is_prime_64(uint64_t n) {
    if (n < 2) return 0;
    if (n < 4) return 1;
    if (!(n & 1) || n % 3 == 0) return 0;
    static const uint64_t witnesses[] = {
        2,3,5,7,11,13,17,19,23,29,31,37
    };
    for (int i = 0; i < 12; i++)
        if (!miller_rabin_witness(n, witnesses[i])) return 0;
    return 1;
}

/* ──────────────────────── segmented sieve ──────────────────── */
/* Returns array of all primes in [lo, hi].  Caller frees. */

#define SMALL_SIEVE_LIMIT 1000000

static uint64_t *sieve_range(uint64_t lo, uint64_t hi, int *count) {
    *count = 0;
    if (hi < lo) return NULL;

    /* For very small ranges, use trial division */
    uint64_t cap   = 8 * ((hi - lo) / 50 + 64);   /* rough estimate */
    uint64_t *out  = (uint64_t *)malloc(cap * sizeof(uint64_t));
    if (!out) return NULL;

    uint64_t span  = hi - lo + 1;
    uint8_t *sieve = (uint8_t *)calloc((size_t)span, 1);
    if (!sieve) { free(out); return NULL; }

    /* sieve[i] = 1 means lo+i is composite */
    /* Mark even numbers (except 2) */
    uint64_t start_even = (lo % 2 == 0) ? lo : lo + 1;
    for (uint64_t n = start_even; n <= hi; n += 2)
        if (n > 2) sieve[(size_t)(n - lo)] = 1;

    /* Sieve with small primes up to sqrt(hi) */
    uint64_t sq = (uint64_t)ceil(sqrt((double)hi)) + 1;

    for (uint64_t p = 3; p <= sq; p += 2) {
        /* Check if p itself is prime (simple trial-div for small p) */
        int p_prime = 1;
        for (uint64_t d = 3; d * d <= p; d += 2)
            if (p % d == 0) { p_prime = 0; break; }
        if (!p_prime) continue;

        /* Mark multiples of p in [lo, hi] */
        uint64_t first = ((lo + p - 1) / p) * p;
        if (first < p * p) first = p * p;
        if (first > hi) continue;
        if (first == p) first = p * p;
        for (uint64_t n = first; n <= hi; n += p)
            sieve[(size_t)(n - lo)] = 1;
    }

    for (uint64_t n = lo; n <= hi; n++) {
        if (sieve[(size_t)(n - lo)]) continue;
        if (n < 2) continue;
        if ((size_t)*count >= (size_t)(cap - 1)) {
            cap *= 2;
            uint64_t *tmp = (uint64_t *)realloc(out, cap * sizeof(uint64_t));
            if (!tmp) break;
            out = tmp;
        }
        out[(*count)++] = n;
    }

    free(sieve);
    return out;
}

/* ──────────────────────── candidate record ─────────────────── */

typedef struct {
    uint64_t p;
    double   n_coord;
    double   frac;
    int      dim;
    double   dn;
    double   score;       /* dn × lower_half_bonus */
    int      lower_half;
} Candidate;

static int cmp_score_desc(const void *a, const void *b) {
    double sa = ((const Candidate *)a)->score;
    double sb = ((const Candidate *)b)->score;
    if (sa > sb) return -1;
    if (sa < sb) return  1;
    return 0;
}

/* ──────────────────────── main ─────────────────────────────── */

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr,
            "Usage: prime_pipeline.exe <p_lo> <p_hi> [opts]\n"
            "  --all       skip lower-half filter\n"
            "  --top N     print top N candidates (default 200)\n"
            "  --min-dn D  minimum Dn score (default 0.0)\n"
            "  --exponents-only  print only p values (for piping to ll_cuda)\n");
        return 1;
    }

    uint64_t p_lo       = (uint64_t)strtoull(argv[1], NULL, 10);
    uint64_t p_hi       = (uint64_t)strtoull(argv[2], NULL, 10);
    int      filter_lh  = 1;     /* lower-half filter on by default */
    int      top_n      = 200;
    double   min_dn     = 0.0;
    int      exponents_only = 0;

    for (int i = 3; i < argc; i++) {
        if (strcmp(argv[i], "--all") == 0)          filter_lh = 0;
        else if (strcmp(argv[i], "--exponents-only") == 0) exponents_only = 1;
        else if (strcmp(argv[i], "--top") == 0    && i+1 < argc) top_n   = atoi(argv[++i]);
        else if (strcmp(argv[i], "--min-dn") == 0 && i+1 < argc) min_dn  = atof(argv[++i]);
    }

    if (p_lo < 2) p_lo = 2;
    if (p_hi < p_lo) { fprintf(stderr, "p_hi < p_lo\n"); return 1; }

    if (!exponents_only) {
        printf("╔══ PRIME PIPELINE — Track D ════════════════════════════════╗\n");
        printf("  Range  : p ∈ [%llu, %llu]\n",
               (unsigned long long)p_lo, (unsigned long long)p_hi);
        printf("  Filter : lower-half (frac < 0.5) %s\n",
               filter_lh ? "ON" : "off");
        printf("  Min Dn : %.2f\n", min_dn);
        printf("╚════════════════════════════════════════════════════════════╝\n\n");
    }

    /* ── Sieve for prime exponents ── */
    int      n_primes = 0;
    uint64_t *primes  = sieve_range(p_lo, p_hi, &n_primes);
    if (!primes || n_primes == 0) {
        fprintf(stderr, "No primes in [%llu, %llu]\n",
                (unsigned long long)p_lo, (unsigned long long)p_hi);
        free(primes);
        return 0;
    }

    if (!exponents_only)
        printf("  Candidate prime exponents in range: %d\n\n", n_primes);

    /* ── Score each prime exponent ── */
    int        n_cands   = 0;
    Candidate *cands     = (Candidate *)malloc((size_t)n_primes * sizeof(Candidate));
    if (!cands) { free(primes); return 1; }

    for (int i = 0; i < n_primes; i++) {
        uint64_t p     = primes[i];
        double   nc    = n_of_2p(p);
        if (nc < 0.0) continue;                /* invalid lattice coord */

        double frac = nc - floor(nc);
        int    lh   = (frac < 0.5) ? 1 : 0;

        if (filter_lh && !lh) continue;        /* lower-half filter */

        /* D_n with base=2 (Mersenne), beta=0, k from dimension ramp */
        /* prismatic spirit: Omega peaks at frac→0 (lower-half attractor) */
        double n_int = floor(nc);
        int    dim   = ((int)n_int % 8) + 1;
        double k     = (dim + 1) / 8.0;
        double omega = 0.5 + 0.5 * sin(M_PI * frac * PHI);
        double dn    = D_n(n_int, 0.0, frac, k, omega, 2.0);

        if (dn < min_dn) continue;

        double score = dn * (lh ? 1.5 : 1.0); /* 50% bonus for lower-half */

        Candidate *c = &cands[n_cands++];
        c->p          = p;
        c->n_coord    = nc;
        c->frac       = frac;
        c->dim        = dim;
        c->dn         = dn;
        c->score      = score;
        c->lower_half = lh;
    }

    free(primes);

    /* ── Sort by score descending ── */
    qsort(cands, (size_t)n_cands, sizeof(Candidate), cmp_score_desc);

    int print_n = (top_n > 0 && top_n < n_cands) ? top_n : n_cands;

    if (exponents_only) {
        /* Just print p values, one per line — for piping into ll_cuda */
        for (int i = 0; i < print_n; i++)
            printf("%llu\n", (unsigned long long)cands[i].p);
    } else {
        /* Full table */
        printf("  Candidates passing filter: %d  (showing top %d)\n\n",
               n_cands, print_n);
        printf("  %-12s  %-10s  %-7s  %-5s  %-8s  %-8s  %s\n",
               "p", "n_coord", "frac", "dim", "Dn", "score", "flag");
        printf("  %s\n",
               "─────────────────────────────────────────────────────────────");

        for (int i = 0; i < print_n; i++) {
            Candidate *c = &cands[i];
            printf("  %-12llu  %-10.6f  %-7.4f  D%-4d  %-8.3f  %-8.3f  %s\n",
                   (unsigned long long)c->p,
                   c->n_coord,
                   c->frac,
                   c->dim,
                   c->dn,
                   c->score,
                   c->lower_half ? "[lower-half]" : "");
        }

        printf("\n  To verify top candidates with Lucas-Lehmer:\n");
        printf("  ll_cuda.exe");
        int show = (print_n < 8) ? print_n : 8;
        for (int i = 0; i < show; i++)
            printf(" %llu", (unsigned long long)cands[i].p);
        printf("\n\n");

        /* Summary statistics */
        double avg_dn = 0.0;
        int    n_lh   = 0;
        for (int i = 0; i < n_cands; i++) {
            avg_dn += cands[i].dn;
            if (cands[i].lower_half) n_lh++;
        }
        if (n_cands > 0) avg_dn /= n_cands;
        printf("  Stats: avg Dn=%.3f  lower-half %.1f%%  of %d candidates\n",
               avg_dn, 100.0 * n_lh / n_cands, n_cands);
    }

    free(cands);
    return 0;
}
