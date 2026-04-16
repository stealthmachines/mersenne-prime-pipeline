/*
 * phi_mersenne_predictor.c — Track B
 *
 * Empirically tests whether the 51 known Mersenne prime exponents
 * cluster on the phi-lattice coordinate n(p) = log_phi(log_phi(p)) - 1/(2phi)
 * and at peaks of the Dn(r) amplitude field.
 *
 * Output:
 *   1. Per-exponent table: p, n(p), frac{n(p)}, Dn amplitude, gap from prev
 *   2. Summary statistics on fractional parts (uniform null = mean~0.5, sd~0.289)
 *   3. ASCII histogram of fractional parts across [0,1)
 *   4. ASCII histogram of delta-n gaps between consecutive Mersenne lattice coords
 *   5. Next-candidate prediction table: top 20 phi-lattice points beyond n(136279841)
 *
 * Build:
 *   gcc -O2 -lm phi_mersenne_predictor.c -o phi_mersenne_predictor
 *
 * No external dependencies.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

/* ── Constants ─────────────────────────────────────────────────── */
#define PHI      1.6180339887498948482L
#define LN_PHI   0.4812118250596034748L
#define LOG10PHI 0.2090150076824960L       /* log10(phi) */

/* Dn operator parameters (from optimal-prime2.c / hdgl_analog_v30) */
#define NUM_DN 8
static const uint64_t FIB_TABLE[NUM_DN]   = {1, 1, 2, 3,  5,  8, 13, 21};
static const uint64_t PRIME_TABLE[NUM_DN] = {2, 3, 5, 7, 11, 13, 17, 19};

/* ── All 51 known Mersenne prime exponents (as of 2026-04) ──────
 *  Source: https://www.mersenne.org/primes/
 *  M1 = 2^2 - 1 through M51 = 2^136279841 - 1
 */
static const uint64_t MERSENNE_EXP[51] = {
    2ULL, 3ULL, 5ULL, 7ULL, 13ULL, 17ULL, 19ULL, 31ULL, 61ULL, 89ULL,
    107ULL, 127ULL, 521ULL, 607ULL, 1279ULL, 2203ULL, 2281ULL, 3217ULL,
    4253ULL, 4423ULL, 9689ULL, 9941ULL, 11213ULL, 19937ULL, 21701ULL,
    23209ULL, 44497ULL, 86243ULL, 110503ULL, 132049ULL, 216091ULL,
    756839ULL, 859433ULL, 1257787ULL, 1398269ULL, 2976221ULL, 3021377ULL,
    6972593ULL, 13466917ULL, 20996011ULL, 24036583ULL, 25964951ULL,
    30402457ULL, 32582657ULL, 37156667ULL, 42643801ULL, 43112609ULL,
    57885161ULL, 74207281ULL, 77232917ULL, 136279841ULL
};
static const int N_MERSENNE = 51;

/* ── phi-lattice coordinate ──────────────────────────────────── */
static long double n_of_x(long double x) {
    /* n(x) = log_phi(log_phi(x)) - 1/(2*phi)
     *       = log(log(x)/ln(phi)) / ln(phi)  - 1/(2*phi) */
    if (x <= 1.0L) return -1.0L; /* undefined */
    long double lx = logl(x);
    if (lx <= 0.0L) return -1.0L;
    long double log_phi_x = lx / LN_PHI;
    if (log_phi_x <= 0.0L) return -1.0L;
    return logl(log_phi_x) / LN_PHI - 0.5L / PHI;
}

/* Inverse: x(n) = phi^(phi^(n + 1/(2*phi)))
 * Returns log10(x) to avoid overflow for large n. */
static long double log10_x_of_n(long double n) {
    long double inner = n + 0.5L / PHI;          /* n + 1/(2*phi)          */
    long double phi_inner = expl(inner * LN_PHI); /* phi^(n+1/(2phi))       */
    return phi_inner * LOG10PHI;                  /* log10(phi^phi_inner)   */
}

/* ── Dn(r) operator ─────────────────────────────────────────── */
/* D_n(r) = sqrt(phi * F_n * P_n * 2^n * Omega) * r^k
 * We evaluate at n modulo NUM_DN, r=1, Omega=1, k=(n+1)/8 */
static long double dn_amplitude(long double n_coord) {
    /* Map continuous n to table index */
    int idx = (int)(fabsl(n_coord)) % NUM_DN;
    long double phi    = PHI;
    long double F_n    = (long double)FIB_TABLE[idx];
    long double P_n    = (long double)PRIME_TABLE[idx];
    long double two_n  = powl(2.0L, (long double)(idx + 1));
    long double Omega  = 1.0L;
    long double k      = ((long double)(idx + 1)) / 8.0L;
    long double r      = 1.0L;

    long double core = phi * F_n * P_n * two_n * Omega;
    if (core < 0.0L) return 0.0L;
    return sqrtl(core) * powl(r, k);
}

/* ── Statistics helpers ─────────────────────────────────────── */
static long double frac_part(long double v) {
    long double f = v - floorl(v);
    if (f < 0.0L) f += 1.0L;
    return f;
}

static long double dist_to_nearest_int(long double f) {
    return f < 0.5L ? f : 1.0L - f;
}

/* ── ASCII bar helper ────────────────────────────────────────── */
static void print_bar(int count, int max_count, int width) {
    int filled = (max_count > 0) ? (count * width / max_count) : 0;
    for (int i = 0; i < filled; i++) putchar('#');
    for (int i = filled; i < width; i++) putchar(' ');
}

/* ═══════════════════════════════════════════════════════════════ */
int main(void) {

    printf("╔══ phi-LATTICE MERSENNE PREDICTOR — Track B ═══════════════╗\n");
    printf("  n(p) = log_phi(log_phi(p)) - 1/(2*phi)\n");
    printf("  D_n(r) = sqrt(phi * F_n * P_n * 2^n * Omega) * r^k\n");
    printf("  Null hypothesis: frac{n(p)} ~ Uniform[0,1)  (mean=0.5, sd=0.289)\n");
    printf("╚════════════════════════════════════════════════════════════╝\n\n");

    /* ── Per-exponent table ───────────────────────────────────── */
    long double n_vals[51];
    long double frac_vals[51];
    long double dn_vals[51];
    long double delta_n[50]; /* gaps between consecutive n values */

    printf("%-4s  %-14s  %-12s  %-10s  %-10s  %-10s\n",
           "idx", "exponent p", "n(p)", "frac{n}", "dist_int", "Dn_amp");
    printf("%-4s  %-14s  %-12s  %-10s  %-10s  %-10s\n",
           "---", "-----------", "-----------", "-------", "--------", "------");

    for (int i = 0; i < N_MERSENNE; i++) {
        long double p = (long double)MERSENNE_EXP[i];
        long double n = n_of_x(p);
        long double f = frac_part(n);
        long double d = dist_to_nearest_int(f);
        long double dn = dn_amplitude(n);

        n_vals[i]    = n;
        frac_vals[i] = f;
        dn_vals[i]   = dn;

        printf("M%-3d  %-14llu  %+11.6Lf  %10.6Lf  %10.6Lf  %10.4Lf\n",
               i + 1,
               (unsigned long long)MERSENNE_EXP[i],
               n, f, d, dn);
    }

    /* ── Delta-n gaps ─────────────────────────────────────────── */
    printf("\n── Delta-n gaps between consecutive Mersenne lattice coords ──\n");
    printf("%-4s  %-14s  %-14s  %-12s\n", "pair", "M_i", "M_i+1", "delta_n");
    printf("%-4s  %-14s  %-14s  %-12s\n", "----", "---", "-----", "-------");

    long double delta_sum = 0.0L, delta_min = 1e30L, delta_max = 0.0L;
    for (int i = 0; i < N_MERSENNE - 1; i++) {
        delta_n[i] = n_vals[i + 1] - n_vals[i];
        delta_sum += delta_n[i];
        if (delta_n[i] < delta_min) delta_min = delta_n[i];
        if (delta_n[i] > delta_max) delta_max = delta_n[i];

        if (i < 20 || i >= N_MERSENNE - 5) { /* print first 20 and last 5 */
            printf("%-4d  %-14llu  %-14llu  %12.6Lf\n",
                   i + 1,
                   (unsigned long long)MERSENNE_EXP[i],
                   (unsigned long long)MERSENNE_EXP[i + 1],
                   delta_n[i]);
        } else if (i == 20) {
            printf("  ... (%d more pairs) ...\n", N_MERSENNE - 25);
        }
    }

    /* ── Summary statistics on frac{n(p)} ───────────────────── */
    long double sum = 0.0L, sum_sq = 0.0L;
    long double dist_sum = 0.0L;
    for (int i = 0; i < N_MERSENNE; i++) {
        sum    += frac_vals[i];
        sum_sq += frac_vals[i] * frac_vals[i];
        dist_sum += dist_to_nearest_int(frac_vals[i]);
    }
    long double mean = sum / N_MERSENNE;
    long double variance = (sum_sq / N_MERSENNE) - (mean * mean);
    long double sd = sqrtl(variance < 0.0L ? 0.0L : variance);
    long double mean_dist = dist_sum / N_MERSENNE;

    printf("\n── Summary statistics: frac{n(p)} across %d Mersenne exponents ──\n",
           N_MERSENNE);
    printf("  mean        = %.6Lf   (uniform null: 0.500000)\n", mean);
    printf("  std dev     = %.6Lf   (uniform null: 0.288675)\n", sd);
    printf("  mean |dist to nearest int| = %.6Lf  (0=all integers,0.25=uniform)\n",
           mean_dist);
    printf("  delta_n min = %.6Lf  max = %.6Lf  mean = %.6Lf\n",
           delta_min, delta_max, delta_sum / (N_MERSENNE - 1));

    /* ── ASCII histogram: frac{n(p)} in [0,1) with 20 bins ──── */
    printf("\n── Histogram: frac{n(p)} — 20 bins across [0.0, 1.0) ─────────\n");
    printf("  (uniform distribution = ~2-3 per bin for N=51)\n\n");

    #define NBINS 20
    int bins[NBINS];
    memset(bins, 0, sizeof(bins));
    for (int i = 0; i < N_MERSENNE; i++) {
        int b = (int)(frac_vals[i] * NBINS);
        if (b >= NBINS) b = NBINS - 1;
        bins[b]++;
    }
    int max_bin = 0;
    for (int b = 0; b < NBINS; b++) if (bins[b] > max_bin) max_bin = bins[b];

    for (int b = 0; b < NBINS; b++) {
        printf("  [%.2f-%.2f) |", (double)b / NBINS, (double)(b + 1) / NBINS);
        print_bar(bins[b], max_bin, 40);
        printf("| %2d\n", bins[b]);
    }

    /* ── ASCII histogram: delta_n gaps (50 gaps, 10 bins) ───── */
    printf("\n── Histogram: delta_n gaps — 10 bins ─────────────────────────\n");
    #define NGAPBINS 10
    int gap_bins[NGAPBINS];
    memset(gap_bins, 0, sizeof(gap_bins));
    int n_gaps = N_MERSENNE - 1;
    for (int i = 0; i < n_gaps; i++) {
        /* normalise gap into [0,1) relative to [delta_min, delta_max] */
        long double norm = (delta_max > delta_min)
            ? (delta_n[i] - delta_min) / (delta_max - delta_min)
            : 0.5L;
        int b = (int)(norm * NGAPBINS);
        if (b >= NGAPBINS) b = NGAPBINS - 1;
        gap_bins[b]++;
    }
    int max_gap_bin = 0;
    for (int b = 0; b < NGAPBINS; b++)
        if (gap_bins[b] > max_gap_bin) max_gap_bin = gap_bins[b];

    long double bin_step = (delta_max - delta_min) / NGAPBINS;
    for (int b = 0; b < NGAPBINS; b++) {
        long double lo = delta_min + b * bin_step;
        long double hi = lo + bin_step;
        printf("  [%.4Lf-%.4Lf) |", lo, hi);
        print_bar(gap_bins[b], max_gap_bin, 35);
        printf("| %2d\n", gap_bins[b]);
    }

    /* ── Prediction table: next 20 candidate exponents ──────── */
    printf("\n── Predicted next candidates beyond M51 (p=136,279,841) ──────\n");
    printf("  Method: step forward by mean delta_n on phi-lattice, report log10(p)\n");
    printf("  (These are the candidates Track C should test with Lucas-Lehmer)\n\n");

    long double mean_delta = delta_sum / (N_MERSENNE - 1);
    /* Also show median delta for robustness */
    /* simple sort of delta_n copy for median */
    long double delta_sorted[50];
    memcpy(delta_sorted, delta_n, n_gaps * sizeof(long double));
    /* bubble sort (small array) */
    for (int i = 0; i < n_gaps - 1; i++)
        for (int j = 0; j < n_gaps - 1 - i; j++)
            if (delta_sorted[j] > delta_sorted[j+1]) {
                long double tmp = delta_sorted[j];
                delta_sorted[j] = delta_sorted[j+1];
                delta_sorted[j+1] = tmp;
            }
    long double median_delta = (n_gaps % 2 == 0)
        ? (delta_sorted[n_gaps/2 - 1] + delta_sorted[n_gaps/2]) / 2.0L
        : delta_sorted[n_gaps / 2];

    printf("  mean delta_n   = %.6Lf\n", mean_delta);
    printf("  median delta_n = %.6Lf\n\n", median_delta);

    printf("  Using mean-step prediction:\n");
    printf("  %-4s  %-16s  %-14s  %-10s\n",
           "rank", "approx log10(p)", "n_pred", "Dn_amp");
    printf("  %-4s  %-16s  %-14s  %-10s\n",
           "----", "---------------", "------", "------");

    long double n_last = n_vals[N_MERSENNE - 1];
    for (int k = 1; k <= 20; k++) {
        long double n_pred = n_last + (long double)k * mean_delta;
        long double log10p = log10_x_of_n(n_pred);
        long double amp     = dn_amplitude(n_pred);
        printf("  %-4d  %16.4Lf  %14.6Lf  %10.4Lf\n",
               k, log10p, n_pred, amp);
    }

    printf("\n  Using median-step prediction:\n");
    printf("  %-4s  %-16s  %-14s  %-10s\n",
           "rank", "approx log10(p)", "n_pred", "Dn_amp");
    printf("  %-4s  %-16s  %-14s  %-10s\n",
           "----", "---------------", "------", "------");

    for (int k = 1; k <= 20; k++) {
        long double n_pred = n_last + (long double)k * median_delta;
        long double log10p = log10_x_of_n(n_pred);
        long double amp     = dn_amplitude(n_pred);
        printf("  %-4d  %16.4Lf  %14.6Lf  %10.4Lf\n",
               k, log10p, n_pred, amp);
    }

    /* ── Interpretation guide ────────────────────────────────── */
    printf("\n╔══ HOW TO READ THESE RESULTS ════════════════════════════════╗\n");
    printf("  CLUSTERING SIGNAL if:\n");
    printf("    frac mean << 0.5  OR  >> 0.5   (shift from 0.5)\n");
    printf("    frac sd   << 0.289             (tighter than uniform)\n");
    printf("    mean |dist to nearest int| << 0.25  (near-integer n values)\n");
    printf("    histogram shows 1-3 dominant bins  (spike vs flat)\n");
    printf("    delta_n histogram is narrow        (quasi-periodic spacing)\n");
    printf("\n");
    printf("  If SD < 0.20 or mean_dist < 0.15: STRONG clustering — proceed\n");
    printf("  to Track C Lucas-Lehmer with high confidence.\n");
    printf("\n");
    printf("  If SD ~ 0.289 and mean_dist ~ 0.25: null hypothesis holds —\n");
    printf("  pivot to Track A/D (general primes via psi resonance).\n");
    printf("╚════════════════════════════════════════════════════════════╝\n");

    return 0;
}
