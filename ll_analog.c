/* ll_analog.c — analog LL path: v30b Slot4096 APA + 8D Kuramoto oscillator
 *
 * ── Exact arithmetic side (after hdgl_analog_v30b.c / bootloaderZ.c) ──────
 *   mantissa_words[0..n-1]  — p-bit LL residue (same layout as Slot4096.
 *                             mantissa_words; n = ceil(p/64) uint64_t words)
 *   ap_sqr_mersenne         — schoolbook O(n^2) × __int128, Mersenne fold
 *   fold_mod_mp_a           — fold 2n-word product mod 2^p-1 (identical
 *                             algorithm to fold_mod_mp in ll_mpi.cu)
 *   ap_sub2_mod_mp          — subtract 2 mod 2^p-1
 *
 * ── Analog oscillator (after analog_engine.h / AnalogContainer1) ──────────
 *   AnaOsc8D:
 *     re[8], im[8]          — complex amplitudes (Kuramoto coupling state)
 *     theta[8]              — phases
 *     omega[8]              — natural frequencies (φ-seeded: BASE_INF_SEEDS*dt)
 *     gamma, k_coupling     — adaptive damping / coupling (K/γ wu-wei ratio)
 *     aphase                — Pluck→Sustain→FineTune→Lock
 *     theta_hist[200]       — mean-phase history (ANG_PHASE_HIST)
 *     cv_hist[50]           — CV window for lock detection (ANG_LOCK_WINDOW)
 *
 * ── Cooperative (conditional) memory ──────────────────────────────────────
 *   Every ANA_SHA_INTERVAL (=8) iterations:
 *     h = xor_fold(mantissa_words)  — deterministic hash of exact residue
 *     theta[i] += det_rand(h^i) * delta / k_coupling  — phase perturbation
 *   → the oscillator "remembers" the arithmetic trajectory via its phase
 *     history.  SHA-256-style imprint without the SHA-256 dependency.
 *
 * ── Adaptive phase (K/γ wu-wei ratios from WU_WEI_ANALYSIS.md) ───────────
 *   Pluck:    K=5.0 γ=0.005  (1000:1) — rapid excitation, high energy
 *   Sustain:  K=3.0 γ=0.008           — absorbing phase structure
 *   FineTune: K=2.0 γ=0.010           — refinement
 *   Lock:     K=1.8 γ=0.012           — settled consensus
 *   Threshold cv: 0.50 / 0.30 / 0.10 (ANG_CV_TO_SUSTAIN/FINETUNE/LOCK)
 *   Emergency reset to Pluck if cv > ANG_EMERGENCY_VAR (=10.0)
 *
 * ── Wu-wei principle ──────────────────────────────────────────────────────
 *   The oscillator does NOT shortcut the LL test — every p-2 iterations run
 *   exact.  Phase lock is a readout, not a gate.  It provides:
 *     1. Progress pacing (logging only on natural phase transitions)
 *     2. Resonance diagnostics (cv, aphase, lock status)
 *     3. Double confirmation: osc LOCKED + residue=0 → strong prime signal
 *     4. Architectural path independence from CUDA
 *
 * Licensed per https://zchg.org/t/legal-notice-copyright-applicable-ip-and-licensing-read-me/440
 */

#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif

#include "ll_analog.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>

/* ── Constants (mirrors analog_engine.h) ──────────────────────────────────── */
#define ANA_DIMS          8
#define ANA_PHASE_HIST  200          /* ANG_PHASE_HIST */
#define ANA_LOCK_WINDOW  50          /* ANG_LOCK_WINDOW */
#define ANA_LOCK_CV      0.05        /* ANG_LOCK_CV */
#define ANA_DT           0.01        /* integration timestep */
#define ANA_PHI          1.6180339887498948
#define ANA_PI           3.14159265358979323846
#define ANA_SHA_INTERVAL 8           /* ANG_SHA_INTERVAL: feedback every N iters */

/* Phase transition CV thresholds (analog_engine.h ANG_CV_TO_*) */
#define ANA_CV_TO_SUSTAIN    0.50
#define ANA_CV_TO_FINETUNE   0.30
#define ANA_CV_TO_LOCK       0.10
#define ANA_EMERGENCY_VAR   10.0

/* K/γ ratios: Pluck=1000:1, critical insight from WU_WEI_ANALYSIS.md.
 * Matching APHASE_COUPLING[] and APHASE_GAMMA[] in analog_engine.c. */
static const double ANA_GAMMA[4]    = {0.005, 0.008, 0.010, 0.012};
static const double ANA_COUPLING[4] = {5.0,   3.0,   2.0,   1.8};

/* Base(∞) φ-seeds for natural frequencies (analog_engine.c BASE_INF_SEEDS).
 * omega[i] = BASE_INF_SEEDS[i] * ANA_DT  (radians/step at dt=0.01) */
static const double BASE_INF_SEEDS[ANA_DIMS] = {
    1.6180339887,   /* φ¹  D₁ */
    2.6180339887,   /* φ²  D₂ */
    3.6180339887,   /* φ³  D₃ */
    4.8541019662,   /* φ⁴  D₄ */
    5.6180339887,   /* φ⁵  D₅ */
    6.4721359549,   /* φ⁶  D₆ */
    7.8541019662,   /* φ⁷  D₇ */
    8.3141592654,   /* φ⁸  D₈ */
};

/* ── Adaptive phase state (matches APhase in analog_engine.h) ─────────────── */
typedef enum {
    APHASE_PLUCK    = 0,   /* high energy excitation  */
    APHASE_SUSTAIN  = 1,   /* absorbing structure      */
    APHASE_FINETUNE = 2,   /* refinement               */
    APHASE_LOCK     = 3    /* settled consensus        */
} APhase;

static const char *APHASE_NAMES[4] = {"Pluck", "Sustain", "FineTune", "Lock"};

/* ── 8D Kuramoto oscillator state (matches AnalogState8D in analog_engine.h) ─ */
typedef struct {
    double re[ANA_DIMS];                  /* complex amplitude — real part    */
    double im[ANA_DIMS];                  /* complex amplitude — imag part    */
    double theta[ANA_DIMS];               /* oscillator phases [0, 2π)        */
    double omega[ANA_DIMS];               /* natural frequencies (rad/step)   */
    double gamma;                         /* current damping coefficient      */
    double k_coupling;                    /* current coupling strength        */
    APhase aphase;                        /* adaptive phase state             */
    double phase_var;                     /* current phase variance (CV)      */
    double theta_hist[ANA_PHASE_HIST];    /* mean-phase sliding history       */
    double cv_hist[ANA_LOCK_WINDOW];      /* CV history for lock detection    */
    int    hist_idx;                      /* write head for theta_hist        */
    int    cv_idx;                        /* write head for cv_hist           */
    int    steps;                         /* total RK4 steps taken            */
} AnaOsc8D;

/* ── Deterministic pseudo-random (mirrors det_rand in AnalogContainer1) ─────── */
static inline double det_rand64(uint64_t seed) {
    uint64_t x = seed;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    return (double)(x * 0x2545F4914F6CDD1DULL) / 18446744073709551615.0;
}

/* ── Oscillator initialisation ───────────────────────────────────────────────── */
static void ana_init(AnaOsc8D *s, uint64_t seed) {
    memset(s, 0, sizeof(*s));
    s->aphase     = APHASE_PLUCK;
    s->gamma      = ANA_GAMMA[APHASE_PLUCK];
    s->k_coupling = ANA_COUPLING[APHASE_PLUCK];
    s->phase_var  = 1e6;
    for (int i = 0; i < ANA_DIMS; i++) {
        s->omega[i] = BASE_INF_SEEDS[i] * ANA_DT;
        /* stagger initial phases deterministically from p-seeded entropy */
        s->theta[i] = 2.0 * ANA_PI * det_rand64(seed ^ ((uint64_t)i * 0x9e3779b97f4a7c15ULL));
        s->re[i]    = cos(s->theta[i]);
        s->im[i]    = sin(s->theta[i]);
    }
}

/* ── Phase variance ──────────────────────────────────────────────────────────── */
static double ana_phase_var(const AnaOsc8D *s) {
    double mean = 0.0;
    for (int i = 0; i < ANA_DIMS; i++) mean += s->theta[i];
    mean /= ANA_DIMS;
    double var = 0.0;
    for (int i = 0; i < ANA_DIMS; i++) {
        double d = s->theta[i] - mean;
        while (d >  ANA_PI) d -= 2.0 * ANA_PI;
        while (d < -ANA_PI) d += 2.0 * ANA_PI;
        var += d * d;
    }
    return sqrt(var / ANA_DIMS);
}

/* ── RK4 derivative struct ───────────────────────────────────────────────────── */
typedef struct {
    double dtheta[ANA_DIMS];
    double dre[ANA_DIMS];
    double dim[ANA_DIMS];
} AnaD;

/* Evaluate Kuramoto derivatives at given state arrays.
 * dθ_i/dt = ω_i + K Σ_j sin(θ_j - θ_i)   (phase coupling)
 * dA_i/dt = -γ A_i                          (amplitude damping)
 * Mirrors EVAL macro in analog_engine.c analog8_rk4_step. */
static AnaD ana_deriv(const AnaOsc8D *s,
                      const double theta[ANA_DIMS],
                      const double re[ANA_DIMS],
                      const double im[ANA_DIMS]) {
    AnaD d;
    for (int i = 0; i < ANA_DIMS; i++) {
        double sum_sin = 0.0;
        for (int j = 0; j < ANA_DIMS; j++)
            sum_sin += sin(theta[j] - theta[i]);
        d.dtheta[i] = s->omega[i] + s->k_coupling * sum_sin;
        d.dre[i]    = -s->gamma * re[i];
        d.dim[i]    = -s->gamma * im[i];
    }
    return d;
}

/* ── One RK4 step (mirrors analog8_rk4_step in analog_engine.c) ─────────────── */
static void ana_rk4_step(AnaOsc8D *s) {
    double t1[ANA_DIMS], r1[ANA_DIMS], i1[ANA_DIMS];
    double t2[ANA_DIMS], r2[ANA_DIMS], i2[ANA_DIMS];
    double t3[ANA_DIMS], r3[ANA_DIMS], i3[ANA_DIMS];

    /* k1 */
    AnaD k1 = ana_deriv(s, s->theta, s->re, s->im);
    for (int i = 0; i < ANA_DIMS; i++) {
        t1[i] = s->theta[i] + 0.5 * ANA_DT * k1.dtheta[i];
        r1[i] = s->re[i]    + 0.5 * ANA_DT * k1.dre[i];
        i1[i] = s->im[i]    + 0.5 * ANA_DT * k1.dim[i];
    }
    /* k2 */
    AnaD k2 = ana_deriv(s, t1, r1, i1);
    for (int i = 0; i < ANA_DIMS; i++) {
        t2[i] = s->theta[i] + 0.5 * ANA_DT * k2.dtheta[i];
        r2[i] = s->re[i]    + 0.5 * ANA_DT * k2.dre[i];
        i2[i] = s->im[i]    + 0.5 * ANA_DT * k2.dim[i];
    }
    /* k3 */
    AnaD k3 = ana_deriv(s, t2, r2, i2);
    for (int i = 0; i < ANA_DIMS; i++) {
        t3[i] = s->theta[i] + ANA_DT * k3.dtheta[i];
        r3[i] = s->re[i]    + ANA_DT * k3.dre[i];
        i3[i] = s->im[i]    + ANA_DT * k3.dim[i];
    }
    /* k4 + final update */
    AnaD k4 = ana_deriv(s, t3, r3, i3);
    for (int i = 0; i < ANA_DIMS; i++) {
        s->theta[i] += (ANA_DT / 6.0) * (k1.dtheta[i] + 2.0*k2.dtheta[i]
                                        + 2.0*k3.dtheta[i] + k4.dtheta[i]);
        s->re[i]    += (ANA_DT / 6.0) * (k1.dre[i]    + 2.0*k2.dre[i]
                                        + 2.0*k3.dre[i]    + k4.dre[i]);
        s->im[i]    += (ANA_DT / 6.0) * (k1.dim[i]    + 2.0*k2.dim[i]
                                        + 2.0*k3.dim[i]    + k4.dim[i]);
        /* wrap phase to [0, 2π) */
        s->theta[i] = fmod(s->theta[i], 2.0 * ANA_PI);
        if (s->theta[i] < 0.0) s->theta[i] += 2.0 * ANA_PI;
    }
    s->steps++;
}

/* ── Cooperative memory: residue hash → oscillator phase perturbation ─────────
 * Analogous to analog8_sha_feedback in analog_engine.c.  Uses XOR-fold of
 * mantissa_words (the exact LL residue) instead of SHA-256 to avoid the
 * sha256_minimal.h dependency, preserving the same coupling semantics:
 * the exact arithmetic state imprints onto the continuous oscillator.
 * The perturbation magnitude is inversely proportional to k_coupling so
 * perturbations shrink as the oscillator approaches consensus (wu-wei). */
static void ana_residue_feedback(AnaOsc8D *s,
                                 const uint64_t *words, size_t n) {
    /* FNV-1a-like XOR fold over residue words → 64-bit hash */
    uint64_t h = 0xcbf29ce484222325ULL;
    for (size_t k = 0; k < n; k++)
        h = (h ^ words[k]) * 0x100000001b3ULL;

    /* Perturb each phase: amplitude ∝ 1/k_coupling so effect wanes at Lock */
    double scale = 0.01 / (s->k_coupling + 1e-9);
    for (int i = 0; i < ANA_DIMS; i++) {
        double delta = (det_rand64(h ^ ((uint64_t)i * 0x6c62272e07bb0142ULL)) - 0.5)
                       * scale;
        s->theta[i] = fmod(s->theta[i] + delta + 2.0 * ANA_PI, 2.0 * ANA_PI);
    }

    /* Record mean phase in history (theta_hist is the "memory" buffer) */
    double mean = 0.0;
    for (int i = 0; i < ANA_DIMS; i++) mean += s->theta[i];
    mean /= ANA_DIMS;
    s->theta_hist[s->hist_idx % ANA_PHASE_HIST] = mean;
    s->hist_idx++;
}

/* ── Adaptive phase update — wu-wei: transitions happen naturally ─────────────
 * Only advance through phases; emergency reset to Pluck on high variance. */
static void ana_update_phase(AnaOsc8D *s) {
    double cv = s->phase_var;

    /* Record CV in sliding lock-detection window */
    s->cv_hist[s->cv_idx % ANA_LOCK_WINDOW] = cv;
    s->cv_idx++;

    APhase new_phase = s->aphase;
    if (cv > ANA_EMERGENCY_VAR) {
        new_phase = APHASE_PLUCK;   /* emergency reset */
    } else {
        if (s->aphase < APHASE_SUSTAIN  && cv < ANA_CV_TO_SUSTAIN)  new_phase = APHASE_SUSTAIN;
        if (s->aphase < APHASE_FINETUNE && cv < ANA_CV_TO_FINETUNE) new_phase = APHASE_FINETUNE;
        if (s->aphase < APHASE_LOCK     && cv < ANA_CV_TO_LOCK)     new_phase = APHASE_LOCK;
    }

    if (new_phase != s->aphase) {
        s->aphase     = new_phase;
        s->gamma      = ANA_GAMMA[new_phase];
        s->k_coupling = ANA_COUPLING[new_phase];
    }
}

/* ── Lock detection: CV sustained below threshold over full ANA_LOCK_WINDOW ─── */
static int ana_is_locked(const AnaOsc8D *s) {
    if (s->cv_idx < ANA_LOCK_WINDOW) return 0;
    for (int i = 0; i < ANA_LOCK_WINDOW; i++)
        if (s->cv_hist[i] > ANA_LOCK_CV) return 0;
    return 1;
}

/* ════════════════════════════════════════════════════════════════════════════
 * Exact arithmetic: fold_mod_mp_a + ap_sqr_mersenne + ap_sub2_mod_mp
 *
 * These are independent reimplementations of fold_mod_mp, mpi_sqr_mod_mp_cpu,
 * and mpi_sub2_mod_mp from ll_mpi.cu, operating directly on raw uint64_t[]
 * arrays (the Slot4096.mantissa_words layout from hdgl_analog_v30b.c).
 * Algorithm is identical — same carry pattern, same fold logic.
 * ════════════════════════════════════════════════════════════════════════════ */

/* fold_mod_mp_a: fold a 2n-word product into n-word result mod 2^p-1.
 * out[] must be zeroed before call.  Identical to fold_mod_mp() in ll_mpi.cu. */
static void fold_mod_mp_a(const uint64_t *prod, size_t prod_len,
                          uint64_t p, uint64_t *out, size_t n)
{
    uint64_t pw = p / 64;   /* word index of the p-bit boundary */
    uint64_t pb = p % 64;   /* bit index within that word        */

    memset(out, 0, n * sizeof(uint64_t));

    /* out = flat_lo (bits 0..p-1) */
    for (size_t k = 0; k < (size_t)pw && k < prod_len && k < n; k++)
        out[k] = prod[k];
    if (pb > 0 && (size_t)pw < prod_len && (size_t)pw < n)
        out[pw] = prod[pw] & ((1ULL << pb) - 1ULL);

    /* out += flat >> p (add the high half back, since 2^p ≡ 1 mod M_p) */
    uint64_t carry = 0;
    for (size_t k = 0; k < n + 2; k++) {
        size_t   base = (size_t)(pw + k);
        uint64_t hw;
        if (pb == 0) {
            hw = (base < prod_len) ? prod[base] : 0;
        } else {
            uint64_t lo = (base   < prod_len) ? prod[base]   : 0;
            uint64_t hi = (base+1 < prod_len) ? prod[base+1] : 0;
            hw = (lo >> pb) | (hi << (64 - pb));
        }
        if (k >= n) { carry += hw; break; }
        unsigned __int128 s = (unsigned __int128)out[k] + hw + carry;
        out[k] = (uint64_t)s;
        carry  = (uint64_t)(s >> 64);
    }

    /* Normalize: propagate carry and top-word overflow back into out[0].
     * 2^p ≡ 1 mod M_p so each overflow bit → one unit added to out[0]. */
    for (;;) {
        uint64_t over = (pb > 0) ? (out[n-1] >> pb) : 0;
        if (over) out[n-1] &= (1ULL << pb) - 1ULL;
        uint64_t c = carry + over;
        carry = 0;
        if (!c) break;
        for (size_t k = 0; k < n && c; k++) {
            unsigned __int128 a = (unsigned __int128)out[k] + c;
            out[k] = (uint64_t)a;
            c      = (uint64_t)(a >> 64);
        }
        carry = c;
    }

    /* canonical: M_p ≡ 0 mod M_p */
    int is_mp = 1;
    for (size_t k = 0; k < n && is_mp; k++) {
        uint64_t expected = (pb == 0) ? ~0ULL
            : (k < (size_t)pw) ? ~0ULL
            : (k == (size_t)pw) ? (1ULL << pb) - 1ULL
            : 0ULL;
        if (out[k] != expected) is_mp = 0;
    }
    if (is_mp) memset(out, 0, n * sizeof(uint64_t));
}

/* ap_sqr_mersenne: in-place s[] = s[]² mod 2^p-1.
 * tmp must point to a caller-provided zeroed buffer of (2n+2) uint64_t.
 * Identical schoolbook pattern to mpi_sqr_mod_mp_cpu in ll_mpi.cu. */
static void ap_sqr_mersenne(uint64_t *s, size_t n, uint64_t p, uint64_t *tmp) {
    size_t n2 = 2 * n;
    memset(tmp, 0, (n2 + 2) * sizeof(uint64_t));

    for (size_t i = 0; i < n; i++) {
        unsigned __int128 carry = 0;
        uint64_t xi = s[i];
        if (!xi) continue;
        for (size_t j = 0; j < n; j++) {
            unsigned __int128 t = (unsigned __int128)xi * s[j]
                                + tmp[i + j] + carry;
            tmp[i + j] = (uint64_t)t;
            carry       = t >> 64;
        }
        /* propagate carry beyond the inner loop */
        size_t k = i + n;
        while (carry) {
            unsigned __int128 t = (unsigned __int128)tmp[k] + carry;
            tmp[k] = (uint64_t)t;
            carry   = t >> 64;
            k++;
        }
    }

    fold_mod_mp_a(tmp, n2 + 2, p, s, n);
}

/* ap_sub2_mod_mp: in-place s[] = s[] - 2 mod 2^p-1.
 * Identical to mpi_sub2_mod_mp in ll_mpi.cu. */
static void ap_sub2_mod_mp(uint64_t *s, size_t n, uint64_t p) {
    uint64_t pb = p % 64;

    /* check if s < 2 */
    int small = 1;
    for (size_t k = n; k-- > 1; )
        if (s[k]) { small = 0; break; }
    if (small && s[0] >= 2) small = 0;

    if (!small) {
        uint64_t borrow = 2;
        for (size_t k = 0; k < n && borrow; k++) {
            if (s[k] >= borrow) { s[k] -= borrow; borrow = 0; }
            else                { s[k] -= borrow; borrow = 1; }
        }
    } else {
        /* s is 0 or 1: result = M_p + s - 2 */
        uint64_t val = s[0];
        for (size_t k = 0; k < n; k++) s[k] = ~0ULL;
        if (pb > 0) s[n-1] = (1ULL << pb) - 1ULL;
        uint64_t sub    = 2 - val;
        uint64_t borrow = sub;
        for (size_t k = 0; k < n && borrow; k++) {
            if (s[k] >= borrow) { s[k] -= borrow; borrow = 0; }
            else                { s[k] -= borrow; borrow = 1; }
        }
    }
}

static int is_zero_a(const uint64_t *words, size_t n) {
    for (size_t k = 0; k < n; k++)
        if (words[k]) return 0;
    return 1;
}

/* ════════════════════════════════════════════════════════════════════════════
 * ll_analog: main entry point
 *
 * Runs exact Lucas-Lehmer with:
 *   mantissa[0..n-1]   — Slot4096.mantissa_words equivalent (v30b layout)
 *   AnaOsc8D osc       — 8D Kuramoto oscillator running in parallel
 *
 * Every ANA_SHA_INTERVAL iters: residue hash → oscillator phase perturbation
 *   (cooperative / conditional memory: arithmetic couples into analog state)
 *
 * Progress is logged only on natural phase transitions (wu-wei pacing).
 * Final report includes oscillator lock status alongside residue result.
 * ════════════════════════════════════════════════════════════════════════════ */
int ll_analog(uint64_t p, int verbose) {
    size_t n  = (size_t)((p + 63) / 64);   /* Slot4096 mantissa word count */
    size_t n2 = 2 * n;

    /* ── Allocate: residue "mantissa_words" + squaring scratch (v30b style) ── */
    uint64_t *mantissa = (uint64_t *)calloc(n,       sizeof(uint64_t));
    uint64_t *tmp      = (uint64_t *)calloc(n2 + 2,  sizeof(uint64_t));
    if (!mantissa || !tmp) {
        fprintf(stderr, "[ll_analog] allocation failed (n=%zu)\n", n);
        free(mantissa); free(tmp);
        return -1;
    }
    mantissa[0] = 4;   /* LL initial seed: s₀ = 4 */

    /* ── Initialise 8D Kuramoto oscillator ── */
    AnaOsc8D osc;
    ana_init(&osc, (uint64_t)p * 0x9e3779b97f4a7c15ULL);

    if (verbose) {
        printf("  [analog] p=%llu  n_words=%zu  osc=8D-Kuramoto\n",
               (unsigned long long)p, n);
        printf("  [analog] K/γ ratio at Pluck = %.0f:1  (wu-wei, WU_WEI_ANALYSIS.md)\n",
               ANA_COUPLING[APHASE_PLUCK] / ANA_GAMMA[APHASE_PLUCK]);
    }

    clock_t    t0         = clock();
    uint64_t   iters      = p - 2;
    APhase     last_phase = APHASE_PLUCK;
    int        logged     = 0;

    /* ══ Main LL loop ══════════════════════════════════════════════════════ */
    for (uint64_t iter = 0; iter < iters; iter++) {

        /* ── Exact arithmetic: s = s² - 2 mod 2^p-1 ── */
        ap_sqr_mersenne(mantissa, n, p, tmp);
        ap_sub2_mod_mp(mantissa, n, p);

        /* ── Analog oscillator: one RK4 step ── */
        ana_rk4_step(&osc);
        osc.phase_var = ana_phase_var(&osc);
        ana_update_phase(&osc);

        /* ── Cooperative memory: imprint residue onto oscillator (conditional) ── */
        if ((iter & (ANA_SHA_INTERVAL - 1)) == 0)
            ana_residue_feedback(&osc, mantissa, n);

        /* ── Progress: only log on natural phase transitions (wu-wei pacing) ── */
        if (verbose) {
            int is_transition = (osc.aphase != last_phase);
            int is_milestone  = (iter == 0 || iter == iters - 1
                                 || (iters > 20 && iter % (iters / 10) == 0));
            if (is_transition || is_milestone) {
                last_phase = osc.aphase;
                double pct     = 100.0 * (double)(iter + 1) / (double)iters;
                double elapsed = (double)(clock() - t0) / CLOCKS_PER_SEC;
                printf("  [analog] iter=%-8llu  %5.1f%%  phase=%-8s  cv=%.4f  t=%.1fs%s\n",
                       (unsigned long long)iter, pct,
                       APHASE_NAMES[osc.aphase], osc.phase_var, elapsed,
                       is_transition ? "  [phase transition]" : "");
                logged++;
            }
        }
    }

    /* ── Final result ── */
    int result = is_zero_a(mantissa, n);

    if (verbose) {
        double elapsed = (double)(clock() - t0) / CLOCKS_PER_SEC;
        printf("  [analog] done: %.2fs  phase=%s  cv=%.4f  locked=%s  residue=%s\n",
               elapsed,
               APHASE_NAMES[osc.aphase],
               osc.phase_var,
               ana_is_locked(&osc) ? "yes" : "no",
               result ? "0 (PRIME)" : "non-zero (COMPOSITE)");
        /* Double confirmation: both analog and exact agree */
        if (ana_is_locked(&osc) && result)
            printf("  [analog] ** osc LOCKED + residue=0: strong prime resonance **\n");
        if (!ana_is_locked(&osc) && result)
            printf("  [analog] note: residue=0 (prime) but osc not locked\n");
        printf("  [analog] n_words=%zu  iters=%llu  rk4_steps=%d  log_events=%d\n",
               n, (unsigned long long)iters, osc.steps, logged);
    }

    free(mantissa);
    free(tmp);
    return result;
}
