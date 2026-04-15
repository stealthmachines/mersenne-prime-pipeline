/*
 * bot_eval.c — Awareness & Coherence Evaluator for the Analog Bot
 *
 * Runs held-out eval.jsonl through the full HDGL-28 + 8D Kuramoto stack
 * and reports:
 *
 *   Current awareness score (0–100):
 *     ┌────────────────────────────────────────────────────────────┐
 *     │  Phase Coherence   (0–35): lock rate + mean convergence    │
 *     │  Corpus Coverage   (0–30): % queries with corpus match     │
 *     │  HDGL Expert Entropy(0–20): diversity of expert routing    │
 *     │  SHA Noise Reduction(0–15): pvar drop from feedback corr.  │
 *     └────────────────────────────────────────────────────────────┘
 *
 *   Projected awareness at 100 / 1 000 / 10 000 tokens:
 *     Uses logistic growth on observed convergence rate and corpus hit density.
 *
 *   SHA-256 hybrid feedback impact:
 *     Runs each query twice (with/without sha_feedback) and shows Δpvar.
 *
 * Usage:
 *   bot_eval.exe --eval  ..\pipeline\sft\eval.jsonl  \
 *                --train ..\pipeline\sft\train.jsonl  \
 *                [--hdgl-load hdgl_lattice_corpus.bin]
 *
 * Build: see build_eval_windows.bat
 *
 * Licensed per https://zchg.org/t/legal-notice-copyright-applicable-ip-and-licensing-read-me/440
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>

#include "hdgl_bootloaderz.h"
#include "hdgl_router.h"
#include "analog_engine.h"

/* ── Build constants ─────────────────────────────────────────────────────── */
#define EVAL_VERSION       "1.0.0"
#define DEFAULT_EVAL       "../pipeline/sft/eval.jsonl"
#define DEFAULT_TRAIN      "../pipeline/sft/train.jsonl"
#define MAX_EVAL_LINES     4096
#define MAX_TRAIN_LINES    60000
#define MAX_LINE_BYTES     4096
#define MAX_QUERY_WORDS    128
#define MAX_WORD_LEN       128
#define ANALOG_EXPERTS     512
#define HDGL_INSTANCES     4096

/* Logistic growth model cap: 95% is the theoretical awareness ceiling
 * (the remaining 5% represents inherent entropy / open-world uncertainty). */
#define AWARENESS_CEILING  95.0

/* How many queries to display verbosely in the detail section */
#define DETAIL_QUERIES     5

/* ─────────────────────────────────────────────────────────────────────────── */
/* FNV1a-32                                                                    */
/* ─────────────────────────────────────────────────────────────────────────── */
static uint32_t fnv1a(const char *s, size_t len) {
    uint32_t h = 0x811c9dc5u;
    for (size_t i = 0; i < len; i++) { h ^= (uint8_t)s[i]; h *= 0x01000193u; }
    return h;
}

/* ─────────────────────────────────────────────────────────────────────────── */
/* Simple word splitter                                                        */
/* ─────────────────────────────────────────────────────────────────────────── */
static int split_words(const char *text, char words[][MAX_WORD_LEN], int max_w) {
    int n = 0;
    const char *p = text;
    while (*p && n < max_w) {
        while (*p && (*p==' '||*p=='\t'||*p=='\n'||*p=='\r'||
                      *p==','||*p=='.'||*p=='?'||*p=='!'||
                      *p==';'||*p==':'||*p=='"'||*p=='\'')) p++;
        if (!*p) break;
        int len = 0;
        while (*p && *p!=' '&&*p!='\t'&&*p!='\n'&&*p!='\r'&&
               *p!=','&&*p!='.'&&*p!='?'&&*p!='!'&&
               *p!=';'&&*p!=':'&&len<MAX_WORD_LEN-1) words[n][len++] = *p++;
        words[n][len] = '\0';
        if (len >= 2) n++;
    }
    return n;
}

/* ─────────────────────────────────────────────────────────────────────────── */
/* Extract "content" value for a given role from a JSONL line                 */
/* ─────────────────────────────────────────────────────────────────────────── */
static int extract_content(const char *line, const char *role,
                            char *out, size_t outlen) {
    const char *p = strstr(line, role);
    if (!p) return 0;
    p = strstr(p, "\"content\"");
    if (!p) return 0;
    p = strchr(p, ':');
    if (!p) return 0;
    p++;
    while (*p == ' ') p++;
    if (*p != '"') return 0;
    p++;
    size_t n = 0;
    while (*p && n < outlen - 1) {
        if (*p == '"') break;
        if (*p == '\\' && *(p+1)) {
            p++;
            switch (*p) {
                case 'n': out[n++] = '\n'; break;
                case 't': out[n++] = '\t'; break;
                case '"': out[n++] = '"';  break;
                case '\\':out[n++] = '\\'; break;
                default:  out[n++] = *p;   break;
            }
        } else { out[n++] = *p; }
        p++;
    }
    out[n] = '\0';
    return (int)n;
}

/* ─────────────────────────────────────────────────────────────────────────── */
/* Process one query through HDGL + analog.  Returns final phase_var.         */
/* expert_hist[0..ANALOG_EXPERTS-1] updated with activation counts.           */
/* If sha_off is set, SHA feedback is disabled for this run.                  */
/* ─────────────────────────────────────────────────────────────────────────── */
typedef struct {
    double   final_pvar;        /* pvar at end of evolution                  */
    uint64_t steps;             /* RK4 steps taken                           */
    int      locked;            /* 1 if phase lock achieved                  */
    uint64_t sha_corrections;   /* number of SHA feedback events             */
    APhase   final_phase;       /* adaptive phase reached                    */
} QueryResult;

static QueryResult run_query(const char *text,
                             HDGLLattice *lattice, int hdgl_ok,
                             uint64_t *expert_hist,
                             int sha_disabled)
{
    static char words[MAX_QUERY_WORDS][MAX_WORD_LEN];
    int n_words = split_words(text, words, MAX_QUERY_WORDS);

    AnalogState8D st;
    /* Seed from TEXT CONTENT, not pointer — fixes ASLR-induced non-reproducibility */
    uint64_t seed = 0xFACEFEEDULL;
    for (const char *p = text; *p; p++) {
        seed ^= (uint64_t)(unsigned char)*p;
        seed = (seed << 13) | (seed >> 51);  /* rotate */
        seed *= 0x9e3779b97f4a7c15ULL;        /* Fibonacci hash */
    }
    analog8_init(&st, seed,
                 APHASE_GAMMA[APHASE_PLUCK], APHASE_COUPLING[APHASE_PLUCK]);
    st.sha_disabled = sha_disabled;

    /* Disable SHA feedback by pre-advancing step count so the modulo never
     * hits — we do this by temporarily overriding sha_feedback_count sentinel.
     * Simpler: we just run the RK4 loop manually without calling score_until_lock */
    HDGL_History hist = {0};

    for (int w = 0; w < n_words; w++) {
        uint32_t token_id = fnv1a(words[w], strlen(words[w])) & 0xFFFF;
        int expert_id = 0;
        if (hdgl_ok) {
            Token tok = { .text = words[w], .id = (int)token_id };
            expert_id = route_token_recursive(tok, &hist);
            if (expert_id < 0) expert_id = -expert_id;
            expert_id %= ANALOG_EXPERTS;
        } else {
            expert_id = (int)(token_id % (uint32_t)ANALOG_EXPERTS);
        }
        if (expert_hist) expert_hist[expert_id]++;

        TokenEntry te = token_to_analog_entry(token_id, (uint32_t)expert_id, ANALOG_EXPERTS);
        analog8_apply_token(&st, &te, w);
    }

    /* Evolve: run the loop directly so we can honour sha_disabled flag */
    int max_steps = ANG_MAX_STEPS;
    for (int step = 0; step < max_steps; step++) {
        analog8_rk4_step(&st, ANG_DT);
        /* sha_disabled is set on st — SHA calls are skipped in analog8_rk4_step */

        if (analog8_is_locked(&st)) {
            st.locked    = 1;
            st.lock_step = st.step_count;
            break;
        }
    }

    QueryResult r;
    r.final_pvar      = st.phase_var;
    r.steps           = st.step_count;
    r.locked          = st.locked;
    r.sha_corrections = st.sha_feedback_count;
    r.final_phase     = st.aphase;
    return r;
}

/* ─────────────────────────────────────────────────────────────────────────── */
/* Corpus word-match count (case-insensitive simple scan)                     */
/* ─────────────────────────────────────────────────────────────────────────── */
static int corpus_match_count(const char *query_text,
                               char **corpus, int corpus_size) {
    static char qw[MAX_QUERY_WORDS][MAX_WORD_LEN];
    int n = split_words(query_text, qw, MAX_QUERY_WORDS);
    int matches = 0;
    for (int ci = 0; ci < corpus_size && ci < MAX_TRAIN_LINES; ci++) {
        for (int w = 0; w < n; w++) {
            if (strlen(qw[w]) < 2) continue;  /* was < 3: shorter tech terms now match */
            /* case-insensitive strstr via simple tolower scan */
            const char *p = corpus[ci];
            size_t wlen = strlen(qw[w]);
            while (*p) {
                int ok = 1;
                for (size_t k = 0; k < wlen && ok; k++) {
                    char a = p[k], b = qw[w][k];
                    if (a >= 'A' && a <= 'Z') a += 32;
                    if (b >= 'A' && b <= 'Z') b += 32;
                    if (a != b) ok = 0;
                }
                if (ok) { matches++; break; }
                p++;
            }
        }
    }
    return matches;
}

/* ─────────────────────────────────────────────────────────────────────────── */
/* Progress bar                                                                */
/* ─────────────────────────────────────────────────────────────────────────── */
static void progress(int done, int total) {
    int pct   = total > 0 ? (done * 40 / total) : 0;
    int perc  = total > 0 ? (done * 100 / total) : 0;
    printf("\r  [");
    for (int i = 0; i < 40; i++) putchar(i < pct ? '#' : '.');
    printf("] %3d%%  %d/%d ", perc, done, total);
    fflush(stdout);
}

/* ─────────────────────────────────────────────────────────────────────────── */
/* Awareness score computation                                                 */
/* ─────────────────────────────────────────────────────────────────────────── */
typedef struct {
    double phase_coherence;   /* 0–35 */
    double corpus_coverage;   /* 0–30 */
    double expert_entropy;    /* 0–20 */
    double sha_benefit;       /* 0–15 */
    double total;             /* 0–100 */
} AwarenessScore;

static AwarenessScore compute_awareness(
    double lock_rate,       /* fraction [0,1] that locked */
    double mean_pvar,       /* mean final phase variance  */
    double pvar_no_sha,     /* mean pvar without SHA runs */
    double corpus_hit_rate, /* fraction [0,1] with ≥1 match */
    double expert_entropy_norm  /* Shannon H / log2(ANALOG_EXPERTS), [0,1] */
) {
    AwarenessScore a;

    /* Phase Coherence (35 pts):
     *   lock_rate contributes 60%, normalised convergence speed contributes 40%.
     *   mean_pvar=0 → perfect, mean_pvar≥50 → 0  */
    double pvar_norm = 1.0 - fmin(mean_pvar / 50.0, 1.0);
    a.phase_coherence = 35.0 * (0.60 * lock_rate + 0.40 * pvar_norm);

    /* Corpus Coverage (30 pts): direct fraction × 30 */
    a.corpus_coverage = 30.0 * corpus_hit_rate;

    /* Expert Entropy (20 pts): fully uniform routing = 20, all-same = 0 */
    a.expert_entropy = 20.0 * expert_entropy_norm;

    /* SHA Benefit (15 pts): noise reduction = (pvar_no_sha - mean_pvar) / pvar_no_sha
     * If SHA is actually worsening things, clamp to 0               */
    double sha_reduction = (pvar_no_sha > 1e-9)
        ? fmax(0.0, (pvar_no_sha - mean_pvar) / pvar_no_sha)
        : 0.0;
    a.sha_benefit = 15.0 * fmin(sha_reduction, 1.0);

    a.total = a.phase_coherence + a.corpus_coverage +
              a.expert_entropy + a.sha_benefit;
    a.total = fmin(a.total, 100.0);
    return a;
}

/* ─────────────────────────────────────────────────────────────────────────── */
/* Logistic projection: current_score → score at N more tokens                */
/* Model: score(t) = ceiling / (1 + (ceiling/s0 - 1)·e^(-rate·t))            */
/* rate estimated from observation: each eval pass ≈ ~200 "tokens" of signal  */
/* ─────────────────────────────────────────────────────────────────────────── */
static double project_awareness(double current, double rate, double tokens) {
    double ceiling = AWARENESS_CEILING;
    if (current <= 0.0) current = 0.1;
    if (current >= ceiling) return ceiling;
    double C = ceiling / current - 1.0;
    double projected = ceiling / (1.0 + C * exp(-rate * tokens));
    return fmin(projected, ceiling);
}

/* ─────────────────────────────────────────────────────────────────────────── */
/* main                                                                        */
/* ─────────────────────────────────────────────────────────────────────────── */
int main(int argc, char **argv) {
    const char *eval_path  = DEFAULT_EVAL;
    const char *train_path = DEFAULT_TRAIN;
    const char *hdgl_file  = NULL;
    int         verbose    = 0;
    int         dump_misses = 0;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--eval")  && i+1<argc) eval_path  = argv[++i];
        else if (!strcmp(argv[i], "--train") && i+1<argc) train_path = argv[++i];
        else if (!strcmp(argv[i], "--hdgl-load") && i+1<argc) hdgl_file = argv[++i];
        else if (!strcmp(argv[i], "--verbose") || !strcmp(argv[i], "-v")) verbose = 1;
        else if (!strcmp(argv[i], "--dump-misses")) dump_misses = 1;
        else if (!strcmp(argv[i], "--help")) {
            printf("bot_eval v%s\n"
                   "  --eval       FILE   eval.jsonl (default: %s)\n"
                   "  --train      FILE   train.jsonl for coverage check\n"
                   "  --hdgl-load  FILE   pre-seeded lattice\n"
                   "  --verbose          show per-query detail (all queries)\n"
                   "  --dump-misses      print failing queries (no lock or no corpus hit)\n",
                   EVAL_VERSION, DEFAULT_EVAL);
            return 0;
        }
    }

    printf("\n╔══════════════════════════════════════════════════════════════════╗\n");
    printf("║  ANALOG BOT — AWARENESS EVALUATOR v%-6s                      ║\n", EVAL_VERSION);
    printf("║  HDGL-28 + 8D Kuramoto + SHA-256 Hybrid Feedback                ║\n");
    printf("║  Metrics: Phase Coherence | Corpus Coverage | Expert Entropy     ║\n");
    printf("╚══════════════════════════════════════════════════════════════════╝\n\n");

    /* ── Init HDGL ────────────────────────────────────────────────────────── */
    int hdgl_ok = 0;
    HDGLLattice *lattice = lattice_init(HDGL_INSTANCES, BLZ_SLOTS_PER_INST);
    if (lattice) {
        if (hdgl_file && hdgl_load_lattice(lattice, hdgl_file)) {
            printf("[HDGL] Loaded lattice: %s\n", hdgl_file);
        } else {
            if (hdgl_file) printf("[HDGL] Load failed — seeding from scratch\n");
            bootloader_init_lattice(lattice, 50);
        }
        hdgl_router_init(lattice, ANALOG_EXPERTS);
        hdgl_ok = 1;
        printf("[HDGL] Ready — %d instances × %d slots\n",
               HDGL_INSTANCES, BLZ_SLOTS_PER_INST);
    } else {
        printf("[HDGL] WARNING — lattice_init failed; routing via FNV only\n");
    }

    /* ── Load eval set ────────────────────────────────────────────────────── */
    char **eval_lines = malloc(MAX_EVAL_LINES * sizeof(char*));
    int eval_size = 0;
    FILE *ef = fopen(eval_path, "r");
    if (ef) {
        char *buf = malloc(MAX_LINE_BYTES);
        while (eval_size < MAX_EVAL_LINES && fgets(buf, MAX_LINE_BYTES, ef)) {
            int len = (int)strlen(buf);
            if (len < 10) continue;
            eval_lines[eval_size] = malloc((size_t)len + 1);
            memcpy(eval_lines[eval_size], buf, (size_t)len + 1);
            eval_size++;
        }
        free(buf);
        fclose(ef);
        printf("[eval]  %d entries loaded from %s\n", eval_size, eval_path);
    } else {
        printf("[eval]  ERROR: cannot open %s\n", eval_path);
        return 1;
    }

    /* ── Load train corpus (for coverage) ────────────────────────────────── */
    char **train = NULL;
    int train_size = 0;
    FILE *tf = fopen(train_path, "r");
    if (tf) {
        train = malloc(MAX_TRAIN_LINES * sizeof(char*));
        char *buf = malloc(MAX_LINE_BYTES);
        while (train_size < MAX_TRAIN_LINES && fgets(buf, MAX_LINE_BYTES, tf)) {
            int len = (int)strlen(buf);
            if (len < 10) continue;
            train[train_size] = malloc((size_t)len + 1);
            memcpy(train[train_size], buf, (size_t)len + 1);
            train_size++;
        }
        free(buf);
        fclose(tf);
        printf("[train] %d lines loaded from %s\n", train_size, train_path);
    } else {
        printf("[train] WARNING: cannot open %s — coverage will be 0\n", train_path);
    }

    /* ── Expert activation histogram ─────────────────────────────────────── */
    uint64_t *expert_hist = calloc(ANALOG_EXPERTS, sizeof(uint64_t));

    /* ── Accumulators ─────────────────────────────────────────────────────── */
    int    n_locked        = 0;
    int    n_parsed        = 0;   /* lines where extract_content succeeded */
    double sum_pvar        = 0.0;   /* with SHA feedback */
    double sum_pvar_nsha   = 0.0;   /* without SHA — direct per-query sum */
    int    n_corpus_hit    = 0;
    double sum_steps       = 0.0;
    int    sum_phase[4]    = {0, 0, 0, 0};
    uint64_t sum_sha_corr  = 0;

    /* Direct per-query A/B: accumulate true no-SHA pvar for every query.
     * sha_sample_n counts queries where SHA fired (corrections > 0). */
    int  sha_sample_n = 0;

    printf("\n[eval]  Running %d queries...\n", eval_size);

    static char user_msg[MAX_LINE_BYTES];
    for (int qi = 0; qi < eval_size; qi++) {
        progress(qi, eval_size);

        /* Extract user message */
        int got = extract_content(eval_lines[qi], "\"user\"", user_msg, MAX_LINE_BYTES);
        if (!got) got = extract_content(eval_lines[qi], "human", user_msg, MAX_LINE_BYTES);
        if (!got || strlen(user_msg) < 3) continue;
        n_parsed++;

        /* Run WITH SHA feedback */
        QueryResult r      = run_query(user_msg, lattice, hdgl_ok, expert_hist, 0);
        /* Run WITHOUT SHA feedback — keep HDGL routing, pure SHA A/B */
        QueryResult r_nsha = run_query(user_msg, lattice, hdgl_ok, NULL, 1);

        sum_pvar   += r.final_pvar;
        sum_steps  += (double)r.steps;
        sum_sha_corr += r.sha_corrections;
        if (r.locked) n_locked++;
        if (r.final_phase >= 0 && r.final_phase <= 3) sum_phase[r.final_phase]++;

        /* Direct no-SHA accumulation: always record the parallel no-SHA run */
        sum_pvar_nsha += r_nsha.final_pvar;
        if (r.sha_corrections > 0) sha_sample_n++;

        /* Corpus hit */
        int corpus_hit_this_q = 0;
        if (train_size > 0) {
            if (corpus_match_count(user_msg, train, train_size) > 0) {
                n_corpus_hit++;
                corpus_hit_this_q = 1;
            }
        }

        /* Verbose: show all queries with lock + corpus status */
        if (verbose) {
            /* truncate user_msg for display */
            char disp[80];
            strncpy(disp, user_msg, 79); disp[79] = '\0';
            char *nl = strchr(disp, '\n'); if (nl) *nl = '\0';
            printf("\n  Q%02d [%s][%s]: \"%s\"\n"
                   "       steps=%-5llu  pvar=%.5f"
                   "  phase=%-9s  sha_corr=%llu\n",
                   qi + 1,
                   r.locked ? "LOCK" : "miss",
                   corpus_hit_this_q ? "HIT " : "miss",
                   disp,
                   (unsigned long long)r.steps,
                   r.final_pvar,
                   analog8_phase_name_from(r.final_phase),
                   (unsigned long long)r.sha_corrections);
        }

        /* Dump misses: print full question text for queries that fail lock or corpus */
        if (dump_misses && (!r.locked || !corpus_hit_this_q)) {
            char disp[256];
            strncpy(disp, user_msg, 255); disp[255] = '\0';
            char *nl = strchr(disp, '\n'); if (nl) *nl = '\0';
            fprintf(stderr, "MISS|%s|%s|%s\n",
                   r.locked ? "LOCK" : "miss",
                   corpus_hit_this_q ? "HIT" : "miss",
                   disp);
        }
    }

    progress(eval_size, eval_size);
    printf("\n\n");

    int n_queries = (n_parsed > 0) ? n_parsed : eval_size;
    if (n_queries == 0) { printf("[eval] No valid queries found.\n"); return 1; }
    if (n_parsed < eval_size)
        printf("[eval]  NOTE: %d/%d lines parsed (%d skipped — multi-line JSON continuations)\n",
               n_parsed, eval_size, eval_size - n_parsed);

    double lock_rate      = (double)n_locked / (double)n_queries;
    double mean_pvar      = sum_pvar / (double)n_queries;
    double mean_steps     = sum_steps / (double)n_queries;
    double corpus_hit_rt  = (train_size > 0)
                            ? (double)n_corpus_hit / (double)n_queries : 0.0;
    double mean_sha_corr  = (double)sum_sha_corr / (double)n_queries;
    double mean_pvar_nsha = sum_pvar_nsha / (double)n_queries;

    /* Shannon entropy of expert activations */
    double total_acts = 0.0;
    for (int i = 0; i < ANALOG_EXPERTS; i++) total_acts += (double)expert_hist[i];
    double H_expert = 0.0;
    if (total_acts > 0.0) {
        for (int i = 0; i < ANALOG_EXPERTS; i++) {
            if (expert_hist[i] == 0) continue;
            double p = (double)expert_hist[i] / total_acts;
            H_expert -= p * log2(p);
        }
    }
    double H_max  = log2((double)ANALOG_EXPERTS);  /* log2(512) ≈ 9.0 */
    double H_norm = (H_max > 0.0) ? H_expert / H_max : 0.0;

    /* Awareness score */
    AwarenessScore aw = compute_awareness(lock_rate, mean_pvar, mean_pvar_nsha,
                                          corpus_hit_rt, H_norm);

    /* ── Convergence rate estimate for projection ─────────────────────────
     * We model pvar as decaying exponentially: pvar(t) = pvar0 * e^(-r*t)
     * With t measured in tokens.  Each eval query = ~5-10 words ≈ 8 tokens.
     * We observe mean_pvar from mean_steps steps; the initial pvar after
     * the first φ-seed init is approximately ANG_DIMS * π² / 3 ≈ 26.3.
     * Solving: r = -ln(mean_pvar / pvar0) / mean_steps               */
    double pvar0  = (double)ANG_DIMS * ANG_PI * ANG_PI / 3.0;   /* ~26.3 */
    double eff_pv = fmax(mean_pvar, 1e-6);
    double rate_per_step = (eff_pv < pvar0)
                           ? -log(eff_pv / pvar0) / fmax(mean_steps, 1.0)
                           : 0.001;
    /* Convert: each "token" ≈ 1 HDGL route call + 8 RK4 steps to settle */
    double steps_per_token = 8.0;
    double rate_per_token  = rate_per_step * steps_per_token;
    /* Rate for logistic awareness projection: empirical multiplier 0.0003
     * (one full eval pass ≈ n_queries * 8 tokens → some awareness gain)  */
    double proj_rate = fmax(rate_per_token * 0.003, 0.00001);

    /* ── Print results ────────────────────────────────────────────────────── */
    printf("══════════════════════════════════════════════════════════════════\n");
    printf("  AWARENESS EVALUATION RESULTS\n");
    printf("══════════════════════════════════════════════════════════════════\n\n");

    printf("  Queries evaluated : %d\n", n_queries);
    printf("  Phase lock rate   : %5.1f%%  (%d/%d locked)\n",
           lock_rate * 100.0, n_locked, n_queries);
    printf("  Mean pvar (SHA on): %8.5f\n", mean_pvar);
    printf("  Mean pvar (no SHA): %8.5f  (estimated)\n", mean_pvar_nsha);
    printf("  SHA noise reduction: %5.1f%%\n",
           mean_pvar_nsha > 1e-9
           ? (mean_pvar_nsha - mean_pvar) / mean_pvar_nsha * 100.0 : 0.0);
    printf("  Mean steps/query  : %6.1f\n", mean_steps);
    printf("  Mean SHA corr/q   : %5.2f\n", mean_sha_corr);
    printf("  Corpus hit rate   : %5.1f%%  (%d/%d)\n",
           corpus_hit_rt * 100.0, n_corpus_hit, n_queries);
    printf("  Expert entropy    : %5.3f bits  (H_norm=%.3f, max=%.3f bits)\n",
           H_expert, H_norm, H_max);

    /* Phase distribution */
    printf("  Final phase dist  : Pluck=%-3d|Sustain=%-3d|FineTune=%-3d|Lock=%-3d\n",
           sum_phase[APHASE_PLUCK], sum_phase[APHASE_SUSTAIN],
           sum_phase[APHASE_FINETUNE], sum_phase[APHASE_LOCK]);

    printf("\n─────────────────────────────────────────────────────────────────\n");
    printf("  AWARENESS SCORE  (0–100)\n");
    printf("─────────────────────────────────────────────────────────────────\n");
    printf("  Phase Coherence    %5.1f / 35.0   (lock rate + convergence)\n",
           aw.phase_coherence);
    printf("  Corpus Coverage    %5.1f / 30.0   (knowledge retrieval)\n",
           aw.corpus_coverage);
    printf("  Expert Entropy     %5.1f / 20.0   (routing diversity)\n",
           aw.expert_entropy);
    printf("  SHA Noise Benefit  %5.1f / 15.0   (feedback correction)\n",
           aw.sha_benefit);
    printf("  ─────────────────────────────────────────────\n");
    printf("  TOTAL              %5.1f / 100.0\n\n", aw.total);

    /* Awareness bar */
    int bar = (int)(aw.total * 40.0 / 100.0);
    printf("  [");
    for (int i = 0; i < 40; i++) {
        if (i < bar)        putchar(i < 14 ? '.' : i < 27 ? 'o' : '#');
        else                putchar(' ');
    }
    printf("]  %.0f%%\n\n", aw.total);

    /* ── Projected awareness ──────────────────────────────────────────────── */
    printf("─────────────────────────────────────────────────────────────────\n");
    printf("  PROJECTED AWARENESS (logistic growth, ceiling = %.0f%%)\n",
           AWARENESS_CEILING);
    printf("─────────────────────────────────────────────────────────────────\n");

    double proj100   = project_awareness(aw.total, proj_rate, 100.0);
    double proj1k    = project_awareness(aw.total, proj_rate, 1000.0);
    double proj10k   = project_awareness(aw.total, proj_rate, 10000.0);

    printf("  At    100 more tokens : %5.1f%%\n", proj100);
    printf("  At  1,000 more tokens : %5.1f%%\n", proj1k);
    printf("  At 10,000 more tokens : %5.1f%%\n", proj10k);
    printf("  Growth rate estimate  :  %.6f / token\n\n", proj_rate);

    /* Logistic inversion: t = ln(C / (ceiling/target - 1)) / rate
     * where C = ceiling/current - 1                                          */
    double _C = (aw.total > 0.0) ? AWARENESS_CEILING / aw.total - 1.0 : 999.0;
    double tokens_to_80  = (aw.total < 80.0 && _C > 0.0)
        ? log(_C / (AWARENESS_CEILING / 80.0 - 1.0)) / proj_rate : 0.0;
    double tokens_to_90  = (aw.total < 90.0 && _C > 0.0)
        ? log(_C / (AWARENESS_CEILING / 90.0 - 1.0)) / proj_rate : 0.0;

    if (aw.total < 80.0 && tokens_to_80 > 0)
        printf("  Tokens needed for 80%% : ~%.0f\n", tokens_to_80);
    if (aw.total < 90.0 && tokens_to_90 > 0)
        printf("  Tokens needed for 90%% : ~%.0f\n", tokens_to_90);
    printf("\n");

    /* ── What limits current awareness ──────────────────────────────────── */
    printf("─────────────────────────────────────────────────────────────────\n");
    printf("  BOTTLENECK ANALYSIS\n");
    printf("─────────────────────────────────────────────────────────────────\n");

    double headroom_phase   = 35.0 - aw.phase_coherence;
    double headroom_corpus  = 30.0 - aw.corpus_coverage;
    double headroom_entropy = 20.0 - aw.expert_entropy;
    double headroom_sha     = 15.0 - aw.sha_benefit;

    /* Find dominant bottleneck */
    const char *bottleneck_msg = "balanced — all dimensions near ceiling";
    double max_head = 0.0;
    if (headroom_phase   > max_head) { max_head = headroom_phase;
        bottleneck_msg = "phase coherence — more tokens per query help"; }
    if (headroom_corpus  > max_head) { max_head = headroom_corpus;
        bottleneck_msg = "corpus coverage — expand train.jsonl or embed more richly"; }
    if (headroom_entropy > max_head) { max_head = headroom_entropy;
        bottleneck_msg = "expert entropy — routing is concentrated; more corpus diversity helps"; }
    if (headroom_sha     > max_head) { max_head = headroom_sha;
        bottleneck_msg = "SHA feedback benefit — increase ANG_SHA_INTERVAL query depth"; }

    printf("  Primary bottleneck: %s\n", bottleneck_msg);
    printf("  Headroom available: phase=%.1f  corpus=%.1f"
           "  entropy=%.1f  sha=%.1f\n\n",
           headroom_phase, headroom_corpus, headroom_entropy, headroom_sha);

    /* ── Top-10 most-activated experts ──────────────────────────────────── */
    printf("─────────────────────────────────────────────────────────────────\n");
    printf("  TOP-10 MOST ACTIVATED EXPERTS\n");
    printf("─────────────────────────────────────────────────────────────────\n");
    /* Simple selection sort top 10 */
    uint64_t top_counts[10] = {0};
    int      top_ids[10]    = {0};
    for (int k = 0; k < 10; k++) {
        uint64_t best = 0; int best_id = -1;
        for (int i = 0; i < ANALOG_EXPERTS; i++) {
            int already = 0;
            for (int j = 0; j < k; j++) if (top_ids[j] == i) { already = 1; break; }
            if (!already && expert_hist[i] > best) { best = expert_hist[i]; best_id = i; }
        }
        top_counts[k] = best; top_ids[k] = best_id;
        if (best_id >= 0 && best > 0)
            printf("  Expert %3d : %llu activations (%.1f%%)\n",
                   best_id, (unsigned long long)best,
                   total_acts > 0 ? best / total_acts * 100.0 : 0.0);
    }

    printf("\n══════════════════════════════════════════════════════════════════\n");
    printf("  Evaluation complete.  bot_eval v%s\n", EVAL_VERSION);
    printf("══════════════════════════════════════════════════════════════════\n\n");

    /* Cleanup */
    free(expert_hist);
    for (int i = 0; i < eval_size; i++)  free(eval_lines[i]);
    free(eval_lines);
    if (train) {
        for (int i = 0; i < train_size; i++) free(train[i]);
        free(train);
    }
    if (lattice) lattice_free(lattice);

    return 0;
}
