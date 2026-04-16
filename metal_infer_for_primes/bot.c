/*
 * bot.c — C-native analog forum intelligence bot
 *
 * Architecture (no Python, no Ollama, no Docker):
 *
 *   Input text
 *     ↓ word-level tokenisation (FNV1a hash → token_id)
 *     ↓ HDGL-28 route_token_recursive → expert_id per word
 *     ↓ token_to_analog_entry → TokenEntry{k, γ, φ}
 *     ↓ analog8_apply_token on 8D Kuramoto state (Spiral8, 8 dims)
 *     ↓ analog8_score_until_lock (RK4 → consensus phase lock)
 *     ↓ JSONL search + analog resonance scoring (train.jsonl)
 *     ↓ Best-match assistant response
 *   Output text
 *
 * Engines:
 *   — HDGL-28 (hdgl_bootloaderz.c / hdgl_router.c)  — routing prior
 *   — AnalogContainer1 8D Kuramoto RK4 (analog_engine.c) — resonance scoring
 *   — vector_container.c (VectorContext + φ-Fourier)  — conversation state
 *   — train.jsonl (pipeline/sft/train.jsonl)          — forum knowledge
 *
 * References:
 *   https://github.com/stealthmachines/AnalogContainer1
 *   https://github.com/stealthmachines/spiral8plus
 *   https://zchg.org
 *
 * Licensed per https://zchg.org/t/legal-notice-copyright-applicable-ip-and-licensing-read-me/440
 *
 * Build: see build_bot_windows.bat / build_bot_windows.ps1
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#ifdef _WIN32
#  include <windows.h>  /* FlushConsoleInputBuffer */
#endif
#ifndef MAX_PATH
#define MAX_PATH 260
#endif

#include "hdgl_bootloaderz.h"
#include "hdgl_router.h"
#include "analog_engine.h"
#include "vector_container.h"

/* ── Build-time defaults ─────────────────────────────────────────────────── */
#define BOT_VERSION        "1.0.0"
#define DEFAULT_CORPUS     "../pipeline/sft/train.jsonl"
#define LLM_SCRIPT         "../local_tiny/brainwave_assist.py"
#define MAX_QUERY_WORDS    256
#define MAX_WORD_LEN       128
#define MAX_CORPUS_LINES   60000
#define MAX_LINE_BYTES     4096       /* cap per JSONL line (memory bound)    */
#define TOP_K_SEARCH       50         /* lines scored by analog after word-match */
#define ANALOG_EXPERTS     512        /* expert count passed to HDGL router   */
#define HDGL_INSTANCES     4096

/* ─────────────────────────────────────────────────────────────────────────── */
/* FNV1a-32: word → token_id                                                  */
/* ─────────────────────────────────────────────────────────────────────────── */
static uint32_t fnv1a(const char *s, size_t len) {
    uint32_t h = 0x811c9dc5u;
    for (size_t i = 0; i < len; i++) {
        h ^= (uint8_t)s[i];
        h *= 0x01000193u;
    }
    return h;
}

/* ─────────────────────────────────────────────────────────────────────────── */
/* Word splitter (space / punctuation delimiters)                              */
/* ─────────────────────────────────────────────────────────────────────────── */
static int split_words(const char *text, char words[][MAX_WORD_LEN], int max_words) {
    int   n = 0;
    const char *p = text;
    while (*p && n < max_words) {
        /* skip whitespace and punctuation */
        while (*p && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r' ||
                      *p == ',' || *p == '.' || *p == '?' || *p == '!' ||
                      *p == ';' || *p == ':' || *p == '"' || *p == '\''))
            p++;
        if (!*p) break;
        int len = 0;
        while (*p && *p != ' ' && *p != '\t' && *p != '\n' && *p != '\r' &&
               *p != ',' && *p != '.' && *p != '?' && *p != '!' &&
               *p != ';' && *p != ':' && len < MAX_WORD_LEN - 1) {
            words[n][len++] = *p++;
        }
        words[n][len] = '\0';
        if (len >= 2) n++;   /* ignore single-char tokens */
    }
    return n;
}

/* ─────────────────────────────────────────────────────────────────────────── */
/* JSON helpers — extract "content" value from a role block                    */
/* ─────────────────────────────────────────────────────────────────────────── */
static int extract_content_after(const char *haystack, const char *role_tag,
                                  char *out, size_t outlen) {
    const char *p = strstr(haystack, role_tag);
    if (!p) return 0;
    p = strstr(p, "\"content\"");
    if (!p) return 0;
    p = strchr(p, ':');
    if (!p) return 0;
    p++;
    while (*p == ' ') p++;
    if (*p != '"') return 0;
    p++;  /* skip opening quote */

    size_t n = 0;
    while (*p && n < outlen - 1) {
        if (*p == '"') break;
        if (*p == '\\' && *(p + 1)) {
            p++;
            switch (*p) {
                case 'n':  out[n++] = '\n'; break;
                case 't':  out[n++] = '\t'; break;
                case '"':  out[n++] = '"';  break;
                case '\\': out[n++] = '\\'; break;
                case 'r':                   break; /* skip \r */
                default:   out[n++] = *p;   break;
            }
        } else {
            out[n++] = *p;
        }
        p++;
    }
    out[n] = '\0';
    return (int)n;
}

/* ─────────────────────────────────────────────────────────────────────────── */
/* Print a response string wrapped to ~79 columns                              */
/* ─────────────────────────────────────────────────────────────────────────── */
static void print_wrapped(const char *text) {
    int col = 0;
    for (const char *c = text; *c; c++) {
        putchar(*c);
        col++;
        if (*c == '\n') { col = 0; continue; }
        if (col > 79 && *c == ' ') { putchar('\n'); col = 0; }
    }
    if (col > 0) putchar('\n');
}

/* ─────────────────────────────────────────────────────────────────────────── */
/* Print banner                                                                 */
/* ─────────────────────────────────────────────────────────────────────────── */
static void print_banner(int hdgl_ok, int corpus_lines, const char *corpus_path, int use_llm_flag) {
    printf("\n╔══════════════════════════════════════════════════════════════════╗\n");
    printf("║  ANALOG BOT v%-7s  C-native forum intelligence            ║\n", BOT_VERSION);
    printf("║  8D Kuramoto RK4 + HDGL-28 + Spiral8  |  AnalogContainer1      ║\n");
    printf("║  https://github.com/stealthmachines/AnalogContainer1            ║\n");
    printf("║  https://github.com/stealthmachines/spiral8plus  |  zchg.org   ║\n");
    printf("╚══════════════════════════════════════════════════════════════════╝\n");
    printf("  HDGL lattice : %s (%d instances x %d slots)\n",
           hdgl_ok ? "ready" : "FAILED", HDGL_INSTANCES, BLZ_SLOTS_PER_INST);
    printf("  Corpus        : %d lines  (%s)\n", corpus_lines, corpus_path);
    printf("  Analog dims   : %d (Spiral8 geometry)\n", ANG_DIMS);
    printf("  LLM assist    : %s\n", use_llm_flag ? "enabled (brainwave_assist.py)" : "off  (--llm to enable)");
    printf("  Commands      : /quit  /verbose  /status  /glyph  /help\n\n");
}

/* ─────────────────────────────────────────────────────────────────────────── */
/* Print help                                                                   */
/* ─────────────────────────────────────────────────────────────────────────── */
static void print_help(void) {
    printf("  /quit     — exit\n"
           "  /verbose  — toggle routing diagnostics\n"
           "  /status   — show analog state & HDGL stats\n"
           "  /glyph    — show current VectorContext holographic glyph\n"
           "  /help     — this message\n\n"
           "  Ask any question. The bot routes it through HDGL-28 expert\n"
           "  coils, evolves the 8D Kuramoto oscillator to phase-lock,\n"
           "  then retrieves the best-resonating forum answer.\n\n");
}

/* ─────────────────────────────────────────────────────────────────────────── */
/* LLM assist — pipe query + context through brainwave_assist.py               */
/* ─────────────────────────────────────────────────────────────────────────── */
static void llm_assist(const char *query, const char *context, const char *band) {
    /* Write inputs to a temp file, invoke Python script, print output */
    char tmp_path[MAX_PATH + 32];
    char *tmp = getenv("TEMP");
    if (!tmp) tmp = ".";
    snprintf(tmp_path, sizeof(tmp_path), "%s\\bot_llm_input.txt", tmp);

    FILE *fp = fopen(tmp_path, "w");
    if (!fp) { printf("[llm] ERROR: cannot write temp file\n"); return; }
    fprintf(fp, "QUERY: %s\n", query);
    fprintf(fp, "BAND: %s\n", band);
    fprintf(fp, "---CONTEXT---\n%s\n", context);
    fclose(fp);

    char cmd[4096];
    snprintf(cmd, sizeof(cmd),
             "python \"%s\" --input-file \"%s\" 2>&1",
             LLM_SCRIPT, tmp_path);
    FILE *pipe = _popen(cmd, "r");
    if (!pipe) { printf("[llm] ERROR: cannot launch python script\n"); return; }
    char line[512];
    while (fgets(line, sizeof(line), pipe))
        fputs(line, stdout);
    _pclose(pipe);
}

/* ─────────────────────────────────────────────────────────────────────────── */
/* HDGL-driven bigram generation                                               */
/* ─────────────────────────────────────────────────────────────────────────── */
#define NGRAM_EXPERTS  512
#define NGRAM_SLOTS    64
#define NGRAM_KEYLEN   48

typedef struct {
    char     key[NGRAM_KEYLEN];   /* predecessor word  */
    char     val[NGRAM_KEYLEN];   /* successor word    */
    uint32_t count;               /* co-occurrence freq */
} NgramRecord;

/* File-scope: ~3.1 MB in BSS — fine for a long-lived binary */
static NgramRecord bot_ngrams[NGRAM_EXPERTS][NGRAM_SLOTS];
static int         bot_ngrams_ok = 0;

/* djb2 — same hash used by hdgl_corpus_seeder's ngram_expert() */
static uint32_t ngram_djb2(const char *s) {
    uint32_t h = 5381;
    while (*s) h = h * 33 + (uint8_t)(*s++);
    return h;
}

/* Walk the per-expert bigram table to compose a short phrase.
 * last_expert : HDGL expert that fired last for this query
 * lock_phase  : query_state.phases[0] at Kuramoto lock
 * max_words   : upper bound on output word count
 * out / outlen: caller-supplied buffer                                        */
/* Autocatalytic lattice generation.
 * Each emitted word is routed back through route_token_recursive(), updating
 * hist->primary_phase / mirror_phase / last_expert_id.  The spiral strand
 * phases evolve with each token, selecting the next expert from actual lattice
 * geometry rather than a static hash.  The trajectory sustains itself as long
 * as the lattice has ngram paths; it terminates naturally when the attractor
 * dissolves (no successor found in the lattice-selected expert bucket).
 */
static void hdgl_generate(const NgramRecord ngrams[][NGRAM_SLOTS],
                           HDGL_History *hist, double seed_phase,
                           int max_words, char *out, size_t outlen) {
    out[0] = '\0';
    size_t pos = 0;

    uint32_t ex = (uint32_t)(hist->last_expert_id < 0
                              ? -hist->last_expert_id
                              :  hist->last_expert_id)
                  % NGRAM_EXPERTS;

    /* Seed slot: phase fraction × NGRAM_SLOTS, walk forward until non-empty */
    int start = (int)(fabs(seed_phase) / (2.0 * 3.14159265358979) * NGRAM_SLOTS)
                % NGRAM_SLOTS;
    int seed_slot = -1;
    for (int i = 0; i < NGRAM_SLOTS; i++) {
        int s = (start + i) % NGRAM_SLOTS;
        if (ngrams[ex][s].key[0]) { seed_slot = s; break; }
    }
    if (seed_slot < 0) return;  /* expert bucket empty */

    char cur[NGRAM_KEYLEN];
    strncpy(cur, ngrams[ex][seed_slot].key, NGRAM_KEYLEN - 1);
    cur[NGRAM_KEYLEN - 1] = '\0';

    /* Longer seen-word ring: lattice routing changes experts dynamically,
     * reducing simple repetition, so allow a longer trajectory before halting. */
    char seen[16][NGRAM_KEYLEN];
    int  n_seen = 0;

    for (int w = 0; w < max_words && cur[0] && pos < outlen - 2; w++) {
        /* Detect cycle */
        for (int si = 0; si < n_seen; si++) {
            if (strncmp(seen[si], cur, NGRAM_KEYLEN - 1) == 0) goto done;
        }
        if (n_seen < 16) strncpy(seen[n_seen++], cur, NGRAM_KEYLEN - 1);

        size_t wl = strlen(cur);
        if (pos + wl + 1 >= outlen) break;
        if (pos > 0) out[pos++] = ' ';
        memcpy(out + pos, cur, wl);
        pos += wl;

        /* ── Autocatalytic step ──────────────────────────────────────────────
         * Route the emitted word back through the lattice.  This updates
         * hist->primary_phase, mirror_phase, and last_expert_id in place,
         * so the next lookup uses the trajectory's evolved attractor state
         * rather than a frozen hash.  The lattice sustains the sequence. */
        Token emit_tok = { cur, w };
        int next_expert = route_token_recursive(emit_tok, hist);
        ex = (uint32_t)(next_expert < 0 ? -next_expert : next_expert) % NGRAM_EXPERTS;

        /* Find highest-count successor for cur in the lattice-selected expert */
        const NgramRecord *best = NULL;
        uint32_t best_cnt = 0;
        for (int i = 0; i < NGRAM_SLOTS; i++) {
            if (strncmp(ngrams[ex][i].key, cur, NGRAM_KEYLEN - 1) == 0
                && ngrams[ex][i].count > best_cnt) {
                best_cnt = ngrams[ex][i].count;
                best = &ngrams[ex][i];
            }
        }
        /* Attractor dissolved: lattice has no path forward — natural stop */
        if (!best || !best->val[0]) break;

        strncpy(cur, best->val, NGRAM_KEYLEN - 1);
        cur[NGRAM_KEYLEN - 1] = '\0';
    }
done:
    out[pos] = '\0';
}

/* ═══════════════════════════════════════════════════════════════════════════ */
int main(int argc, char **argv) {
    const char *hdgl_file   = NULL;
    const char *ngrams_file = NULL;   /* hdgl_lattice_corpus_ngrams.bin */
    const char *corpus_path = DEFAULT_CORPUS;
    int         verbose     = 0;
    int         use_llm     = 0;
    double      band_freq   = 0.0;  /* 0 = default (phi-seeded Beta) */

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--hdgl-load") && i + 1 < argc)
            hdgl_file = argv[++i];
        else if (!strcmp(argv[i], "--ngrams-load") && i + 1 < argc)
            ngrams_file = argv[++i];
        else if (!strcmp(argv[i], "--corpus") && i + 1 < argc)
            corpus_path = argv[++i];
        else if (!strcmp(argv[i], "--verbose") || !strcmp(argv[i], "-v"))
            verbose = 1;
        else if (!strcmp(argv[i], "--llm"))
            use_llm = 1;
        else if (!strcmp(argv[i], "--alpha-mode"))
            band_freq = 10.5;  /* Alpha: 8-13 Hz, centre 10.5 Hz */
        else if (!strcmp(argv[i], "--theta-mode"))
            band_freq = 6.0;   /* Theta: 4-8 Hz,  centre 6.0 Hz  */
        else if (!strcmp(argv[i], "--gamma-mode"))
            band_freq = 40.0;  /* Gamma: 30+ Hz,  centre 40 Hz   */
        else if (!strcmp(argv[i], "--band-mode") && i + 1 < argc) {
            const char *bm = argv[++i];
            if      (!strcmp(bm, "alpha") || !strcmp(bm, "Alpha")) band_freq = 10.5;
            else if (!strcmp(bm, "theta") || !strcmp(bm, "Theta")) band_freq = 6.0;
            else if (!strcmp(bm, "gamma") || !strcmp(bm, "Gamma")) band_freq = 40.0;
            else if (!strcmp(bm, "beta")  || !strcmp(bm, "Beta"))  band_freq = 21.5;
            else if (!strcmp(bm, "delta") || !strcmp(bm, "Delta")) band_freq = 2.0;
        }
        else if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h")) {
            printf("analog bot v%s\n"
                   "  --hdgl-load FILE      pre-seeded lattice (hdgl_lattice_corpus.bin)\n"
                   "  --ngrams-load FILE    per-expert bigrams (hdgl_lattice_corpus_ngrams.bin)\n"
                   "  --corpus    FILE      train.jsonl path (default: " DEFAULT_CORPUS ")\n"
                   "  --verbose             show per-word routing stats\n"
                   "  --llm                 pipe responses through brainwave_assist.py\n"
                   "  --alpha-mode          tune oscillator to Alpha band (10.5 Hz)\n"
                   "  --gamma-mode          tune oscillator to Gamma band (40 Hz)\n"
                   "  --band-mode BAND      alpha/theta/beta/gamma/delta\n", BOT_VERSION);
            return 0;
        }
    }

    /* ── Init HDGL-28 lattice ─────────────────────────────────────────────── */
    int hdgl_ok = 0;
    HDGLLattice *lattice = lattice_init(HDGL_INSTANCES, BLZ_SLOTS_PER_INST);
    if (lattice) {
        if (hdgl_file && hdgl_load_lattice(lattice, hdgl_file)) {
            printf("[HDGL] Loaded lattice from %s\n", hdgl_file);
        } else {
            if (hdgl_file) printf("[HDGL] Load failed, seeding from scratch\n");
            bootloader_init_lattice(lattice, 50);
        }
        hdgl_router_init(lattice, ANALOG_EXPERTS);
        hdgl_ok = 1;
    } else {
        fprintf(stderr, "[HDGL] lattice_init failed — running without routing\n");
    }

    /* ── Init VectorContext (conversation-level state) ───────────────────── */
    FrameworkContainer *container = container_create("bot-session");
    container_initialize(container);

    /* ── Load corpus (train.jsonl) ───────────────────────────────────────── */
    char  **corpus_lines = NULL;
    int     corpus_size  = 0;

    FILE *cf = fopen(corpus_path, "r");
    if (cf) {
        corpus_lines = malloc(MAX_CORPUS_LINES * sizeof(char *));
        char *linebuf = malloc(MAX_LINE_BYTES);
        while (corpus_size < MAX_CORPUS_LINES && fgets(linebuf, MAX_LINE_BYTES, cf)) {
            int len = (int)strlen(linebuf);
            if (len < 10) continue;
            /* Cap line to MAX_LINE_BYTES */
            corpus_lines[corpus_size] = malloc((size_t)len + 1);
            memcpy(corpus_lines[corpus_size], linebuf, (size_t)len + 1);
            corpus_size++;
        }
        free(linebuf);
        fclose(cf);
    } else {
        fprintf(stderr, "[corpus] Not found: %s\n", corpus_path);
        fprintf(stderr, "         Run  python pipeline/build_index.py  first,\n");
        fprintf(stderr, "         or   build_hdgl_corpus_windows.bat   to seed lattice.\n");
    }

    /* ── Load per-expert bigram table (optional) ─────────────────────────── */
    /* Auto-detect: if --hdgl-load path given and --ngrams-load not given,
     * try <hdgl-load-stem>_ngrams.bin alongside the lattice file.            */
    if (!ngrams_file && hdgl_file) {
        static char auto_ngrams_path[MAX_PATH + 16];
        strncpy(auto_ngrams_path, hdgl_file, MAX_PATH);
        auto_ngrams_path[MAX_PATH] = '\0';
        char *dot = strrchr(auto_ngrams_path, '.');
        if (dot) strcpy(dot, "_ngrams.bin");
        else      strcat(auto_ngrams_path, "_ngrams.bin");
        /* Only use if the file exists */
        FILE *probe = fopen(auto_ngrams_path, "rb");
        if (probe) { fclose(probe); ngrams_file = auto_ngrams_path; }
    }
    if (ngrams_file) {
        FILE *nf = fopen(ngrams_file, "rb");
        if (!nf) {
            fprintf(stderr, "[ngrams] Not found: %s\n", ngrams_file);
        } else {
            char hdr[6] = {0};
            uint32_t ne = 0, ns = 0;
            fread(hdr, 1, 5, nf);
            fread(&ne, 4, 1, nf);
            fread(&ns, 4, 1, nf);
            if (strcmp(hdr, "NGRAM") == 0
                && ne == NGRAM_EXPERTS && ns == NGRAM_SLOTS) {
                fread(bot_ngrams, sizeof(bot_ngrams), 1, nf);
                bot_ngrams_ok = 1;
                printf("[ngrams] Loaded: %s\n", ngrams_file);
            } else {
                fprintf(stderr, "[ngrams] Bad header in %s (ne=%u ns=%u)\n",
                        ngrams_file, ne, ns);
            }
            fclose(nf);
        }
    }

    print_banner(hdgl_ok, corpus_size, corpus_path, use_llm);

    /* ── Analog state (reset per query) ──────────────────────────────────── */
    AnalogState8D query_state;
    analog8_init(&query_state, 42ULL, HDGL_GAMMA, HDGL_K_COUPLING);
    if (band_freq > 0.0) {
        analog8_tune_band(&query_state, band_freq);
        printf("[band-mode] Frequencies tuned to %.1f Hz (%s)\n",
               band_freq, analog8_bw_band_name(&query_state));
    }

    /* ── Per-session HDGL history (persists across tokens in one query) ──── */
    HDGL_History hdgl_hist = {0};

    /* ── Query word scratch space ─────────────────────────────────────────── */
    static char words[MAX_QUERY_WORDS][MAX_WORD_LEN];
    char   input[4096];

    /* ── Search scratch ──────────────────────────────────────────────────── */
    typedef struct { int  line_idx; int word_score; double analog_score; } SResult;
    SResult *results = malloc(MAX_CORPUS_LINES * sizeof(SResult));

    /* ── Chat loop ────────────────────────────────────────────────────────── */
#ifdef _WIN32
    /* Flush any stale CR/LF left in the console input buffer at startup.
     * Without this the user must press Enter once before the prompt accepts
     * input (classic Windows console cooked-mode quirk). */
    FlushConsoleInputBuffer(GetStdHandle(STD_INPUT_HANDLE));
#endif
    printf("You: ");
    fflush(stdout);

    while (fgets(input, sizeof(input), stdin)) {
        /* Trim trailing newline */
        int ilen = (int)strlen(input);
        while (ilen > 0 && (input[ilen-1] == '\n' || input[ilen-1] == '\r'))
            input[--ilen] = '\0';
        if (ilen == 0) { printf("You: "); fflush(stdout); continue; }

        /* ── Built-in commands ─────────────────────────────────────────── */
        if (!strcmp(input, "/quit") || !strcmp(input, "/exit")) break;
        if (!strcmp(input, "/help")) { print_help(); printf("You: "); fflush(stdout); continue; }

        if (!strcmp(input, "/verbose")) {
            verbose = !verbose;
            printf("[verbose %s]\n", verbose ? "on" : "off");
            printf("You: "); fflush(stdout); continue;
        }

        if (!strcmp(input, "/status")) {
            printf("[analog]  steps=%llu  locked=%d  pvar=%.6f\n",
                   (unsigned long long)query_state.step_count,
                   query_state.locked, query_state.phase_var);
            printf("[phase]   %s  (gamma=%.4f  K=%.3f  K/gamma=%.0f:1)\n",
                   analog8_phase_name(&query_state),
                   query_state.gamma, query_state.k_coupling,
                   query_state.k_coupling / query_state.gamma);
            printf("[brainwave] band=%s  mean_freq=%.4f rad/step  (~%.1f Hz)\n",
                   analog8_bw_band_name(&query_state),
                   query_state.mean_freq,
                   query_state.mean_freq / (2.0 * 3.14159265 * 0.01));
            printf("[corpus]  %d lines loaded\n", corpus_size);
            printf("[HDGL]    %s  last_expert=%d  primary_phase=%.4f\n",
                   hdgl_ok ? "ok" : "unavailable",
                   hdgl_hist.last_expert_id, hdgl_hist.primary_phase);
            printf("You: "); fflush(stdout); continue;
        }

        if (!strcmp(input, "/glyph")) {
            HolographicGlyph g = glyph_generate(
                &container->context, (uint32_t)time(NULL), (uint64_t)time(NULL));
            printf("[glyph]  char=%c  DNA=%s  ternary=%s  phase=%.4f\n",
                   g.projected_char, g.dna_sequence,
                   g.ternary_state == TERNARY_NEGATIVE ? "NEGATIVE" :
                   g.ternary_state == TERNARY_NEUTRAL  ? "NEUTRAL"  : "POSITIVE",
                   g.breathing_phase);
            printf("You: "); fflush(stdout); continue;
        }

        /* ── Tokenise query ─────────────────────────────────────────────── */
        int n_words = split_words(input, words, MAX_QUERY_WORDS);
        if (n_words == 0) { printf("You: "); fflush(stdout); continue; }

        /* ── Reset analog state for this query ──────────────────────────── */
        analog8_init(&query_state, (uint64_t)time(NULL), HDGL_GAMMA, HDGL_K_COUPLING);
        if (band_freq > 0.0) analog8_tune_band(&query_state, band_freq);
        memset(&hdgl_hist, 0, sizeof(hdgl_hist));

        /* ── Route each word: HDGL → analog state ───────────────────────── */
        for (int w = 0; w < n_words; w++) {
            uint32_t token_id = fnv1a(words[w], strlen(words[w])) & 0xFFFF;

            int expert_id = 0;
            if (hdgl_ok) {
                Token tok = { .text = words[w], .id = (int)token_id };
                expert_id = route_token_recursive(tok, &hdgl_hist);
                if (expert_id < 0) expert_id = -expert_id;
                expert_id %= ANALOG_EXPERTS;
            } else {
                expert_id = (int)(token_id % (uint32_t)ANALOG_EXPERTS);
            }

            TokenEntry te = token_to_analog_entry(token_id, (uint32_t)expert_id, ANALOG_EXPERTS);
            analog8_apply_token(&query_state, &te, w);

            if (verbose)
                printf("  [%s] tok=%u exp=%d k=%.3f γ=%.4f φ=%.3f\n",
                       words[w], token_id, expert_id, te.k, te.gamma, te.phase);
        }

        /* ── Evolve analog state to consensus lock ───────────────────────── */
        double lock_score = analog8_score_until_lock(&query_state, ANG_MAX_STEPS);

        if (verbose)
            printf("[analog] %llu steps  locked=%d  pvar=%.6f\n",
                   (unsigned long long)query_state.step_count,
                   query_state.locked, lock_score);

        /* ── Update conversation VectorContext with query trajectory ─────── */
        if (corpus_size > 0) {
            /* Use phase_history as the k-trajectory sample for Fourier encoding */
            int nhist = (query_state.hist_count < ANG_PHASE_HIST)
                        ? query_state.hist_count : ANG_PHASE_HIST;
            if (nhist >= 3) {
                double *ph_k    = malloc(nhist * sizeof(double));
                double *ph_g    = malloc(nhist * sizeof(double));
                double *ph_p    = malloc(nhist * sizeof(double));
                for (int i = 0; i < nhist; i++) {
                    int idx = (query_state.hist_idx - nhist + i + ANG_PHASE_HIST) % ANG_PHASE_HIST;
                    ph_k[i] = query_state.phase_history[idx] * ANG_PHI;
                    ph_g[i] = sin(ph_k[i]) * ANG_INV_PHI;
                    ph_p[i] = query_state.phases[i % ANG_DIMS];
                }
                context_set_trajectory(&container->context,
                                       ph_k, nhist, ph_g, nhist, ph_p, nhist);
                breathing_perform_cycle(&container->context, 3);
                free(ph_k); free(ph_g); free(ph_p);
            }
        }

        /* ── Search corpus ───────────────────────────────────────────────── */
        printf("\nBot: ");

        if (corpus_size == 0) {
            /* No corpus — show analog routing report only */
            printf("[analog routing report — no corpus loaded]\n");
            printf("  Query routed through %d HDGL experts.  8D Kuramoto: %s at step %llu\n",
                   n_words, query_state.locked ? "LOCKED" : "no lock",
                   (unsigned long long)query_state.lock_step);
            printf("  Load a corpus:  --corpus pipeline/sft/train.jsonl\n");
            printf("\nYou: "); fflush(stdout); continue;
        }

        /* Phase 1: word-match scoring — O(corpus × words) strstr scan */
        int n_matches = 0;
        for (int i = 0; i < corpus_size; i++) {
            int score = 0;
            for (int w = 0; w < n_words; w++) {
                if (strlen(words[w]) >= 2 && strstr(corpus_lines[i], words[w]))
                    score++;
            }
            if (score > 0) { /* min word len >=2 (changed from >=3) */
                results[n_matches].line_idx    = i;
                results[n_matches].word_score  = score;
                results[n_matches].analog_score = 0.0;
                n_matches++;
            }
        }

        /* Phase 2: analog resonance scoring — top-K by word score */
        /* Sort top TOP_K_SEARCH by word_score (selection sort, small K) */
        int to_score = (n_matches < TOP_K_SEARCH) ? n_matches : TOP_K_SEARCH;
        for (int i = 0; i < to_score; i++) {
            int best = i;
            for (int j = i + 1; j < n_matches; j++)
                if (results[j].word_score > results[best].word_score) best = j;
            if (best != i) { SResult tmp = results[i]; results[i] = results[best]; results[best] = tmp; }
        }

        for (int i = 0; i < to_score; i++) {
            /* Clone query state and nudge with corpus line hash */
            AnalogState8D cand = query_state;
            uint32_t lhash = fnv1a(corpus_lines[results[i].line_idx],
                                   strlen(corpus_lines[results[i].line_idx]));
            /* Nudge phase of D0 toward line hash */
            cand.phases[0] = fmod(cand.phases[0] +
                                  2.0 * ANG_PI * ((double)(lhash & 0xFFFF) / 65536.0) * 0.1,
                                  2.0 * ANG_PI);
            double cand_pvar = analog8_score_until_lock(&cand, 512);
            /* Resonance = improvement in phase_var (lower absolute value = better) */
            double improvement = lock_score - cand_pvar;
            results[i].analog_score = (double)results[i].word_score * (1.0 + fabs(improvement));
        }

        if (to_score == 0) {
            printf("No matching knowledge found.\n");
            printf("  (HDGL locked=%d at step %llu, pvar=%.4f, band=%s)\n",
                   query_state.locked,
                   (unsigned long long)query_state.lock_step,
                   query_state.phase_var,
                   analog8_bw_band_name(&query_state));
            if (bot_ngrams_ok) {
                char gen_buf[512];
                hdgl_generate(bot_ngrams, &hdgl_hist,
                               query_state.phases[0], 20, gen_buf, sizeof(gen_buf));
                if (gen_buf[0]) {
                    printf("[autocatal | expert=%d | %s]\n",
                           hdgl_hist.last_expert_id, analog8_bw_band_name(&query_state));
                    print_wrapped(gen_buf);
                } else {
                    printf("  Try rephrasing, or check that train.jsonl is populated.\n");
                }
            } else {
                printf("  Try rephrasing, or check that train.jsonl is populated.\n");
            }
        } else {
            /* Try candidates in descending analog_score order until one has
             * an extractable assistant response.  This avoids dead-ends on
             * user-only corpus entries (no assistant turn in the JSONL). */
            static const char *role_tags[] = {
                "\"assistant\"", "\"gpt\"", "\"bot\"", "\"model\"", NULL
            };
            char resp_buf[MAX_LINE_BYTES];
            int found = 0;

            for (int attempt = 0; attempt < to_score && !found; attempt++) {
                /* Selection-sort next best into position `attempt` */
                int best = attempt;
                for (int i = attempt + 1; i < to_score; i++)
                    if (results[i].analog_score > results[best].analog_score) best = i;
                if (best != attempt) {
                    SResult tmp = results[attempt];
                    results[attempt] = results[best];
                    results[best] = tmp;
                }

                int rlen = 0;
                for (int ri = 0; role_tags[ri] && !rlen; ri++)
                    rlen = extract_content_after(corpus_lines[results[attempt].line_idx],
                                                 role_tags[ri], resp_buf, sizeof(resp_buf));

                if (rlen > 0) {
                    found = 1;
                    printf("[%s | %s | pvar=%.4f]\n",
                           analog8_phase_name(&query_state),
                           analog8_bw_band_name(&query_state),
                           query_state.phase_var);
                    print_wrapped(resp_buf);
                    if (use_llm) {
                        printf("\n[LLM assist — %s band]\n", analog8_bw_band_name(&query_state));
                        llm_assist(input, resp_buf, analog8_bw_band_name(&query_state));
                    }
                    if (verbose) {
                        char qbuf[256];
                        extract_content_after(corpus_lines[results[attempt].line_idx],
                                              "\"user\"", qbuf, sizeof(qbuf));
                        printf("  ── source ──────────────────────────────────────────\n");
                        printf("  Q: %s\n", qbuf);
                        printf("  word_score=%d  analog_resonance=%.4f  lock_step=%llu\n",
                               results[attempt].word_score, results[attempt].analog_score,
                               (unsigned long long)query_state.lock_step);
                        if (attempt > 0)
                            printf("  (skipped %d user-only entries)\n", attempt);
                    }
                }
            }

            if (!found) {
                printf("No extractable assistant response in top %d matches.\n", to_score);
                printf("  (HDGL locked=%d, pvar=%.4f, band=%s)\n",
                       query_state.locked, query_state.phase_var,
                       analog8_bw_band_name(&query_state));
                if (bot_ngrams_ok) {
                    char gen_buf[512];
                    hdgl_generate(bot_ngrams, &hdgl_hist,
                                   query_state.phases[0], 20, gen_buf, sizeof(gen_buf));
                    if (gen_buf[0]) {
                        printf("[autocatal | expert=%d | %s]\n",
                               hdgl_hist.last_expert_id, analog8_bw_band_name(&query_state));
                        print_wrapped(gen_buf);
                    }
                }
            }
        }

        printf("\nYou: ");
        fflush(stdout);
    }

    /* ── Cleanup ─────────────────────────────────────────────────────────── */
    printf("\n[analog bot] session ended  (steps=%llu  glyph=%c)\n",
           (unsigned long long)query_state.step_count,
           container->context.glyph.projected_char);

    free(results);
    container_destroy(container);
    if (corpus_lines) {
        for (int i = 0; i < corpus_size; i++) free(corpus_lines[i]);
        free(corpus_lines);
    }
    if (lattice) lattice_free(lattice);
    free_apa_constants();

    return 0;
}
