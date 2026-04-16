// hdgl_corpus_seeder.c — HDGL-28 corpus-seeded lattice generator
//
// Reads pipeline/sft/train.jsonl, extracts the token frequency distribution
// across all conversation turns, φ-Fourier encodes the distribution
// (derived from AnalogContainer1/vector_container.c by the same author),
// then uses the resulting spectral coefficients to bias the HDGL lattice
// slot charges, phases, and frequencies before writing the final bin.
//
// Output: hdgl_lattice_corpus.bin — binary-compatible with --hdgl-load.
//
// The "small model" is not a language model.  It is a corpus-seeded lattice
// state: a VectorContext in the AnalogContainer1 sense, expressed as
// Slot4096 charge/phase/freq fields that encode the semantic topology of the
// forum data into the routing prior.
//
// Build (Windows — requires LLVM/clang):
//   clang -O2 -Wall -I. -D_CRT_SECURE_NO_WARNINGS             ^
//         hdgl_corpus_seeder.c hdgl_bootloaderz.c hdgl_router.c ^
//         -o hdgl_corpus_seeder.exe -lm
//   (or use build_hdgl_corpus_windows.bat)
//
// Build (Linux / macOS):
//   gcc -O2 -Wall -I. hdgl_corpus_seeder.c hdgl_bootloaderz.c hdgl_router.c \
//       -lm -o hdgl_corpus_seeder
//
// Usage:
//   hdgl_corpus_seeder --corpus ..\pipeline\sft\train.jsonl
//   hdgl_corpus_seeder --corpus ..\pipeline\sft\train.jsonl \
//                      --output hdgl_lattice_corpus.bin      \
//                      --instances 4096 --steps 50 --alpha 0.30
//
// ZCHG License: https://zchg.org/t/legal-notice-copyright-applicable-ip-and-licensing-read-me/440

// Expose M_PI on MSVC / Windows clang before any system headers are pulled in
#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include "hdgl_bootloaderz.h"
#include "hdgl_router.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <inttypes.h>
#include <math.h>
#include <time.h>
#include <ctype.h>

// -------------------------------------------------------------------------
// φ-Fourier encoding
// Derived from AnalogContainer1/vector_container.c (same author/license).
// State is not a file — it is a continuous function encoded as
// φ-harmonic Fourier coefficients.
// -------------------------------------------------------------------------

#define PHI_CORP       1.6180339887498948
#define INV_PHI_CORP   0.6180339887498948
#define FOURIER_N_CORP 12   // φ-scaled harmonics (matches AnalogContainer1)

typedef struct {
    double cos_b[FOURIER_N_CORP];
    double sin_b[FOURIER_N_CORP];
    double mean;
    double scale;
    double temporal_phase;  // φ-modulated phase, seeds slot->phase bias
} PhiSpectrum;

static void phi_fourier_encode(const double *samples, size_t count,
                               PhiSpectrum *out) {
    double sum = 0.0;
    size_t i;
    for (i = 0; i < count; i++) sum += samples[i];
    out->mean = sum / (double)count;

    double max_amp = 0.0;
    for (int n = 0; n < FOURIER_N_CORP; n++) {
        double sc = 0.0, ss = 0.0;
        /* URF §1: use (n + β) not integer n.  β=0.1 is the h-domain fractional
         * coordinate from the joint-optimised fit table; consistent with
         * prismatic_recursion's n_cont = (idx%16) + 0.1 in hdgl_bootloaderz.c. */
        double freq = ((double)n + 0.1) * PHI_CORP;  /* φ-scaled (n+β) frequency */
        for (i = 0; i < count; i++) {
            double angle = 2.0 * M_PI * freq * (double)i / (double)count;
            sc += samples[i] * cos(angle);
            ss += samples[i] * sin(angle);
        }
        out->cos_b[n] = sc / (double)count;
        out->sin_b[n] = ss / (double)count;
        double amp = sqrt(sc * sc + ss * ss) / (double)count;
        if (amp > max_amp) max_amp = amp;
    }
    out->scale         = max_amp;
    out->temporal_phase = fmod(out->cos_b[0] * M_PI * PHI_CORP, 2.0 * M_PI);
}

// -------------------------------------------------------------------------
// Token hash table (open-address, djb2)
// -------------------------------------------------------------------------

#define TBL_BITS   20
#define TBL_SIZE   (1 << TBL_BITS)  // 1 M buckets — well beyond any forum vocab
#define TBL_MASK   (TBL_SIZE - 1)
#define MAX_TOKLEN 64

typedef struct {
    char     key[MAX_TOKLEN];
    uint32_t hash;
    uint64_t count;
} TokenEntry;

static TokenEntry *g_table = NULL;

static uint32_t djb2(const char *s, size_t len) {
    uint32_t h = 5381;
    for (size_t i = 0; i < len; i++)
        h = h * 33u + (unsigned char)s[i];
    return h;
}

// Returns table index, or -1 if table is full (should not happen at 1M buckets).
static int tbl_find_or_insert(const char *tok, size_t len) {
    if (len == 0 || len >= MAX_TOKLEN) return -1;
    uint32_t h   = djb2(tok, len);
    uint32_t idx = h & TBL_MASK;
    for (uint32_t probe = 0; probe < TBL_SIZE; probe++) {
        uint32_t i = (idx + probe) & TBL_MASK;
        if (g_table[i].key[0] == '\0') {
            // Empty slot — insert
            memcpy(g_table[i].key, tok, len);
            g_table[i].key[len]  = '\0';
            g_table[i].hash  = h;
            g_table[i].count = 0;
            return (int)i;
        }
        if (g_table[i].hash == h &&
            strncmp(g_table[i].key, tok, len) == 0 &&
            g_table[i].key[len] == '\0')
            return (int)i;
    }
    return -1;
}

// -------------------------------------------------------------------------
// Token counting — lowercased alpha-numeric runs only (no external lib)
// -------------------------------------------------------------------------

static char g_lower_buf[MAX_TOKLEN];

static void count_word(const char *tok, size_t len) {
    if (len == 0 || len >= MAX_TOKLEN) return;
    for (size_t i = 0; i < len; i++)
        g_lower_buf[i] = (char)tolower((unsigned char)tok[i]);
    int idx = tbl_find_or_insert(g_lower_buf, len);
    if (idx >= 0) g_table[idx].count++;
}

static void tokenize_text(const char *text, size_t len) {
    size_t i = 0;
    while (i < len) {
        while (i < len && !isalnum((unsigned char)text[i])) i++;
        size_t start = i;
        while (i < len && isalnum((unsigned char)text[i])) i++;
        if (i > start) count_word(text + start, i - start);
    }
}

// -------------------------------------------------------------------------
// Per-expert bigram table
//
// Layout: NGRAM_EXPERTS experts × NGRAM_SLOTS bigram slots each.
// Each slot: predecessor word (key), successor word (val), hit count.
// Expert assignment: djb2(word) % NGRAM_EXPERTS — same hash the router uses
// (hdgl_router_key_to_slot calls djb2 internally, but we replicate mod here
//  to avoid calling the full router during corpus scan which needs a live
//  lattice object; the router guarantees same slot for same key, so this is
//  structurally equivalent at NGRAM_EXPERTS resolution).
//
// On disk (hdgl_ngrams.bin):
//   "NGRAM" (5 bytes)
//   uint32_t NGRAM_EXPERTS
//   uint32_t NGRAM_SLOTS
//   For each expert × slot: NgramRecord { key[48], val[48], uint32_t count }
// -------------------------------------------------------------------------

#define NGRAM_EXPERTS  512
#define NGRAM_SLOTS    64    /* bigram successors stored per expert */
#define NGRAM_KEYLEN   48

typedef struct {
    char     key[NGRAM_KEYLEN];   /* predecessor word */
    char     val[NGRAM_KEYLEN];   /* successor word */
    uint32_t count;
} NgramRecord;

static NgramRecord g_ngrams[NGRAM_EXPERTS][NGRAM_SLOTS];

static uint32_t ngram_expert(const char *word) {
    /* Replicate djb2 mod — same mapping as hdgl_router_key_to_slot at 512 res */
    uint32_t h = 5381;
    for (const char *p = word; *p; p++) h = h * 33u + (unsigned char)*p;
    return h % (uint32_t)NGRAM_EXPERTS;
}

/* Record predecessor → successor bigram under predecessor's expert bucket */
static void ngram_record(const char *pred, const char *succ) {
    if (!pred[0] || !succ[0]) return;
    uint32_t ex = ngram_expert(pred);
    NgramRecord *slots = g_ngrams[ex];
    /* Linear probe: find existing pred→succ pair or empty slot */
    for (int i = 0; i < NGRAM_SLOTS; i++) {
        if (slots[i].key[0] == '\0') {
            /* Empty — insert */
            strncpy(slots[i].key, pred, NGRAM_KEYLEN - 1);
            strncpy(slots[i].val, succ, NGRAM_KEYLEN - 1);
            slots[i].count = 1;
            return;
        }
        if (strncmp(slots[i].key, pred, NGRAM_KEYLEN) == 0 &&
            strncmp(slots[i].val, succ, NGRAM_KEYLEN) == 0) {
            slots[i].count++;
            return;
        }
    }
    /* Bucket full — evict lowest-count slot (keeps high-frequency bigrams) */
    int min_i = 0;
    for (int i = 1; i < NGRAM_SLOTS; i++)
        if (slots[i].count < slots[min_i].count) min_i = i;
    strncpy(slots[min_i].key, pred, NGRAM_KEYLEN - 1);
    strncpy(slots[min_i].val, succ, NGRAM_KEYLEN - 1);
    slots[min_i].key[NGRAM_KEYLEN - 1] = '\0';
    slots[min_i].val[NGRAM_KEYLEN - 1] = '\0';
    slots[min_i].count = 1;
}

/* Tokenise text and record all consecutive bigrams */
#define MAX_LINE_WORDS 512
static void tokenize_and_bigram(const char *text, size_t len) {
    char words[MAX_LINE_WORDS][NGRAM_KEYLEN];
    int  nw = 0;
    size_t i = 0;
    while (i < len && nw < MAX_LINE_WORDS) {
        while (i < len && !isalnum((unsigned char)text[i])) i++;
        size_t start = i;
        while (i < len && isalnum((unsigned char)text[i])) i++;
        size_t wlen = i - start;
        if (wlen == 0 || wlen >= NGRAM_KEYLEN) continue;
        /* lowercase */
        for (size_t j = 0; j < wlen; j++)
            words[nw][j] = (char)tolower((unsigned char)text[start + j]);
        words[nw][wlen] = '\0';
        /* also count for freq table */
        count_word(words[nw], wlen);
        nw++;
    }
    for (int wi = 0; wi + 1 < nw; wi++)
        ngram_record(words[wi], words[wi + 1]);
}

// -------------------------------------------------------------------------
// JSONL content extraction — finds all "content":"..." blocks per line.
// No JSON library dependency; handles basic \" escaping.
// -------------------------------------------------------------------------

/* Decode a "content":"..." value into buf (max outlen-1 chars), return length */
static size_t decode_content(const char *start, const char *end, char *buf, size_t outlen) {
    size_t n = 0;
    for (const char *p = start; p < end && n < outlen - 1; p++) {
        if (*p == '\\' && p + 1 < end) {
            p++;
            switch (*p) {
                case 'n':  buf[n++] = '\n'; break;
                case 't':  buf[n++] = '\t'; break;
                case '"':  buf[n++] = '"';  break;
                case '\\': buf[n++] = '\\'; break;
                default:   buf[n++] = *p;   break;
            }
        } else {
            buf[n++] = *p;
        }
    }
    buf[n] = '\0';
    return n;
}

// -------------------------------------------------------------------------
// Corpus scan
// -------------------------------------------------------------------------

#define LINE_BUF_SZ (1 << 20)  // 1 MB — handles the largest forum posts

static char g_content_buf[LINE_BUF_SZ];

/* Extract and encode all occurrences of one named string field from a JSONL line.
   field  — e.g. "\"content\":" or "\"body\":"
   flen   — strlen(field)
   Handles both "key":"value" and "key": "value" (space-tolerant).
   Covers forum SFT messages ("content":) and raw Discourse posts ("body":). */
static void extract_field(const char *line, const char *field, size_t flen) {
    const char *p = line;
    while (*p) {
        const char *found = strstr(p, field);
        if (!found) break;
        const char *start = found + flen;
        while (*start == ' ') start++;
        if (*start != '"') { p = start; continue; }
        start++;  /* skip opening quote */
        const char *q = start;
        while (*q && !(*q == '"' && (q == start || *(q - 1) != '\\'))) q++;
        if (q > start) {
            size_t clen = decode_content(start, q, g_content_buf, LINE_BUF_SZ);
            tokenize_text(g_content_buf, clen);           /* freq counts */
            tokenize_and_bigram(g_content_buf, clen);     /* bigrams     */
        }
        p = (*q == '"') ? q + 1 : q;
    }
}

/* Scan one JSONL line for all recognisable text fields.
   - "content": — SFT train/eval Q&A turns (messages[] format)
   - "body":    — raw Discourse forum post bodies (corpus_public.jsonl format)
   Both are encoded together; the lattice reflects the full semantic pool. */
static void extract_and_count(const char *line) {
    extract_field(line, "\"content\":", 10);  /* SFT format  */
    extract_field(line, "\"body\":",     7);  /* Discourse format */
}

static uint64_t scan_corpus(const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "[hdgl_corpus_seeder] WARN: cannot open corpus: %s\n", path);
        return 0;
    }
    char *line = (char *)malloc(LINE_BUF_SZ);
    if (!line) { fclose(f); return 0; }
    uint64_t n = 0;
    while (fgets(line, LINE_BUF_SZ, f)) {
        extract_and_count(line);
        n++;
        if (n % 1000 == 0) {
            printf("[hdgl_corpus_seeder]   ... %" PRIu64 " lines\r", n);
            fflush(stdout);
        }
    }
    if (n >= 1000) printf("\n");
    free(line);
    fclose(f);
    return n;
}

// -------------------------------------------------------------------------
// Caption scan — reads image_captions.jsonl produced by image_caption.exe
//
// Each line: {"hash":"BASENAME.ext","caption":"TEXT..."}
// We tokenise and bigram only the "caption" field — formulas and descriptions
// from Discourse forum images.  The hash (filename) is not corpus text.
//
// Captions feed into the same shared frequency table and bigram graph as
// the text corpus, so the lattice slot biases reflect the full knowledge pool:
// SFT Q&A + raw forum posts + image formula/description vocabulary.
// -------------------------------------------------------------------------

static uint64_t scan_captions(const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "[hdgl_corpus_seeder] WARN: cannot open captions: %s\n", path);
        return 0;
    }
    char *line = (char *)malloc(LINE_BUF_SZ);
    if (!line) { fclose(f); return 0; }
    uint64_t n = 0;
    while (fgets(line, LINE_BUF_SZ, f)) {
        /* Extract only the "caption" field — not "hash" (filename, not text) */
        extract_field(line, "\"caption\":", 10);
        n++;
        if (n % 500 == 0) {
            printf("[hdgl_corpus_seeder]   ... %" PRIu64 " captions\r", n);
            fflush(stdout);
        }
    }
    if (n >= 500) printf("\n");
    free(line);
    fclose(f);
    return n;
}

// -------------------------------------------------------------------------
// Sort helper
// -------------------------------------------------------------------------

static int cmp_by_count_desc(const void *a, const void *b) {
    const TokenEntry *ta = (const TokenEntry *)a;
    const TokenEntry *tb = (const TokenEntry *)b;
    if (tb->count > ta->count) return  1;
    if (tb->count < ta->count) return -1;
    return 0;
}

// -------------------------------------------------------------------------
// On-disk slot record — identical layout to hdgl_lattice_generator.c so
// --hdgl-load can consume either generator's output transparently.
// -------------------------------------------------------------------------

typedef struct {
    uint64_t mantissa_word0;
    int64_t  exponent;
    double   phase;
    double   freq;
    uint32_t state_flags;
    uint32_t strand_idx;
} SlotRecord;

static void write_u32(FILE *f, uint32_t v) { fwrite(&v, 4, 1, f); }
static void write_f64(FILE *f, double   v) { fwrite(&v, 8, 1, f); }

// -------------------------------------------------------------------------
// URF prime list — first 50 primes; accessed as PRIMES_URF[p_idx] in v3
// phase-entropy term: (1 + ln P_{n+β} / φ^{n+β})
// Source: Unified Recursive Framework constant table (D-operator spec).
// -------------------------------------------------------------------------
static const int PRIMES_URF[50] = {
    2,  3,  5,  7, 11, 13, 17, 19, 23, 29,
   31, 37, 41, 43, 47, 53, 59, 61, 67, 71,
   73, 79, 83, 89, 97,101,103,107,109,113,
  127,131,137,139,149,151,157,163,167,173,
  179,181,191,193,197,199,211,223,227,229
};

// -------------------------------------------------------------------------
// main
// -------------------------------------------------------------------------

#define MAX_CORPUS_FILES   16
#define MAX_CAPTION_FILES  16

int main(int argc, char **argv) {
    const char *corpus_files[MAX_CORPUS_FILES];
    const char *caption_files[MAX_CAPTION_FILES];
    int         n_corpus     = 0;
    int         n_captions   = 0;
    const char *outfile      = "hdgl_lattice_corpus.bin";
    int         num_instances = 4096;
    int         slots_per    = BLZ_SLOTS_PER_INST;
    int         base_steps   = 50;   // standard bootloader pre-seeding steps
    double      bias_alpha   = 0.30; // corpus weight vs base lattice [0..1]

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--corpus") && i+1 < argc) {
            if (n_corpus < MAX_CORPUS_FILES)
                corpus_files[n_corpus++] = argv[++i];
            else { fprintf(stderr, "Too many --corpus files (max %d)\n", MAX_CORPUS_FILES); return 1; }
        }
        else if (!strcmp(argv[i], "--captions") && i+1 < argc) {
            if (n_captions < MAX_CAPTION_FILES)
                caption_files[n_captions++] = argv[++i];
            else { fprintf(stderr, "Too many --captions files (max %d)\n", MAX_CAPTION_FILES); return 1; }
        }
        else if (!strcmp(argv[i], "--output")    && i+1 < argc) outfile       = argv[++i];
        else if (!strcmp(argv[i], "--instances") && i+1 < argc) num_instances = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--steps")     && i+1 < argc) base_steps    = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--alpha")     && i+1 < argc) bias_alpha    = atof(argv[++i]);
        else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            return 1;
        }
    }

    if (n_corpus == 0 && n_captions == 0) {
        fprintf(stderr,
            "Usage: hdgl_corpus_seeder --corpus <file.jsonl> [options]\n"
            "  --corpus FILE       SFT/Discourse JSONL (\"content\" or \"body\" fields)\n"
            "  --captions FILE     image_captions.jsonl from image_caption.exe\n"
            "  Multiple --corpus and --captions flags accepted; all merged.\n"
            "  --output FILE       output lattice bin (default: hdgl_lattice_corpus.bin)\n"
            "  --instances N       APA instances == hidden_size (default: 4096)\n"
            "  --steps N           base bootloader seeding steps (default: 50)\n"
            "  --alpha F           corpus bias weight 0..1 (default: 0.30)\n"
            "\n"
            "Output is binary-compatible with --hdgl-load.\n");
        return 1;
    }

    // 1. Allocate token hash table (~64 MB; freed before exit)
    g_table = (TokenEntry *)calloc(TBL_SIZE, sizeof(TokenEntry));
    if (!g_table) { fprintf(stderr, "OOM: token table\n"); return 1; }

    // 2. Scan all corpus files — accumulate token frequency counts across all sources.
    //    Each file's "content": (SFT) and "body": (Discourse) fields are merged into
    //    one shared frequency table + bigram graph before phi-Fourier encoding.
    //    This gives the lattice a view of the FULL semantic pool, not a single slice.
    printf("[hdgl_corpus_seeder] v%s\n", HDGL_VERSION_STR);
    uint64_t lines = 0;
    for (int ci = 0; ci < n_corpus; ci++) {
        printf("[hdgl_corpus_seeder] Corpus %d/%d: %s\n", ci+1, n_corpus, corpus_files[ci]);
        uint64_t file_lines = scan_corpus(corpus_files[ci]);
        printf("[hdgl_corpus_seeder]   -> %" PRIu64 " lines\n", file_lines);
        lines += file_lines;
    }
    printf("[hdgl_corpus_seeder] Total corpus lines: %" PRIu64 "\n", lines);

    // 2b. Scan caption files — image formula/description vocabulary from image_caption.exe.
    //     Caption tokens merge into the same shared frequency table and bigram graph,
    //     so the lattice slot biases reflect image knowledge alongside text knowledge.
    uint64_t captions = 0;
    for (int ci = 0; ci < n_captions; ci++) {
        printf("[hdgl_corpus_seeder] Captions %d/%d: %s\n", ci+1, n_captions, caption_files[ci]);
        uint64_t file_captions = scan_captions(caption_files[ci]);
        printf("[hdgl_corpus_seeder]   -> %" PRIu64 " captions\n", file_captions);
        captions += file_captions;
    }
    if (n_captions > 0)
        printf("[hdgl_corpus_seeder] Total captions : %" PRIu64 "\n", captions);

    // 3. Collect non-zero entries and sort by descending frequency
    size_t n_tokens = 0;
    for (size_t i = 0; i < TBL_SIZE; i++)
        if (g_table[i].key[0] != '\0' && g_table[i].count > 0)
            n_tokens++;

    printf("[hdgl_corpus_seeder] Unique tokens: %zu\n", n_tokens);

    if (n_tokens == 0) {
        fprintf(stderr, "[hdgl_corpus_seeder] WARN: no tokens found in corpus\n");
        fprintf(stderr, "  Check that files contain \"content\":\"...\" (SFT) or \"body\":\"...\" (Discourse) fields.\n");
    }

    TokenEntry *sorted = (TokenEntry *)calloc(n_tokens + 1, sizeof(TokenEntry));
    if (!sorted) { fprintf(stderr, "OOM: sorted array\n"); free(g_table); return 1; }

    size_t k = 0;
    for (size_t i = 0; i < TBL_SIZE; i++)
        if (g_table[i].key[0] != '\0' && g_table[i].count > 0)
            sorted[k++] = g_table[i];
    qsort(sorted, k, sizeof(TokenEntry), cmp_by_count_desc);

    // 4. Build frequency time-series: 256 log-normalised samples of the sorted
    //    distribution.  Treating rank order as time makes the distribution a
    //    continuous analog signal that the φ-Fourier transform can encode.
    size_t series_len = (k < 256) ? k : 256;
    double series[256];
    memset(series, 0, sizeof(series));
    double max_c = (k > 0) ? (double)sorted[0].count : 1.0;
    for (size_t i = 0; i < series_len; i++)
        series[i] = log1p((double)sorted[i].count) / log1p(max_c);

    // 5. φ-Fourier encode the global distribution
    PhiSpectrum global_spec;
    phi_fourier_encode(series, 256, &global_spec);
    printf("[hdgl_corpus_seeder] Spectrum: DC=%.4f  scale=%.4f  tphase=%.4f\n",
           global_spec.mean, global_spec.scale, global_spec.temporal_phase);

    // 6. Initialise base lattice with standard bootloader seeding
    printf("[hdgl_corpus_seeder] Base seeding (%d steps)...\n", base_steps);
    srand((unsigned int)time(NULL));
    HDGLLattice *lat = lattice_init(num_instances, slots_per);
    if (!lat) {
        fprintf(stderr, "ERROR: lattice_init failed\n");
        free(sorted); free(g_table); return 1;
    }
    g_hdgl_lattice = lat;
    bootloader_init_lattice(lat, base_steps);

    // 7. Apply corpus bias
    //    Each token is mapped to a lattice slot via φ-hash projection and its
    //    normalised frequency is used to:
    //      a) add a charge delta to mantissa_words[0]
    //      b) rotate slot->phase toward the token's φ-phase + spectral phase
    //      c) nudge slot->freq by the token's rank position in the distribution
    printf("[hdgl_corpus_seeder] Applying corpus bias (alpha=%.2f)...\n", bias_alpha);
    int    total_slots = num_instances * slots_per;
    size_t n_bias      = (k < (size_t)total_slots) ? k : (size_t)total_slots;

    // Saturation guard: common words (e.g. "the", "this", "how") hammer the same
    // slots via djb2 with no upper bound, flooding those cells and diluting
    // distinctive content.  CHARGE_SAT_LIMIT caps each slot at 75% of uint64 max.
    // When a slot is saturated, half the charge delta routes to an alternate slot
    // (next-prime-step neighbour) so the energy propagates rather than being lost.
#define CHARGE_SAT_LIMIT 0xBFFFFFFFFFFFFFFFULL   /* 75% of UINT64_MAX */
    int sat_count = 0;

    for (size_t ti = 0; ti < n_bias; ti++) {
        // Hebb: phi_tau + Spiral8 cold-start projection is the same path the
        // router takes on first encounter of this token (H.primary_phase = 0).
        // Slot biased here == slot router visits → fires together, wires together.
        // Structurally guarantees ∂p/∂α > 0 at inference time.
        int slot_idx = hdgl_router_key_to_slot(sorted[ti].key, total_slots);

        Slot4096 *s = lattice_get_slot(lat, slot_idx);
        if (!s) continue;

        double norm_freq = log1p((double)sorted[ti].count) / log1p(max_c);

        // URF rank coordinate: (n, β) = log_φ(rank+1) split at floor.
        // Maps each token to a (n, β) node in the golden lattice — corpus
        // becomes a proper URF domain with its own (n, β, Ω_corpus) triple.
        double nf      = log1p((double)ti) / log(PHI_CORP);  // log_φ(rank+1)
        double beta_nf = nf - floor(nf);                     // fractional β
        double phi_nf  = pow(PHI_CORP, nf);                  // φ^(n+β)

        // URF v3 phase-entropy modulation:
        //   cos_term     = |cos(πβφ)|                  — phase gate
        //   entropy_term = 1 + ln(P_{n+β}) / φ^{n+β}  — prime entropy weight
        double cos_term     = fabs(cos(M_PI * beta_nf * PHI_CORP));
        int    p_idx        = ((int)fabs(floor(nf)) + 50) % 50;
        double entropy_term = 1.0 + log((double)PRIMES_URF[p_idx]) / phi_nf;

        // (a) Charge: frequency-weighted URF amplitude (cos × entropy modulated)
        //     Saturation guard: if slot is already at 75% capacity, route the
        //     overflow to the next-prime-step neighbour at half strength,
        //     propagating energy into cold zones rather than re-flooding hot ones.
        if (s->mantissa_words) {
            double charge_delta = norm_freq * bias_alpha
                                  * cos_term * entropy_term;
            uint64_t delta64 = (uint64_t)(charge_delta * (double)0xFFFFFFFFULL);
            if (s->mantissa_words[0] <= CHARGE_SAT_LIMIT) {
                s->mantissa_words[0] += delta64;
            } else {
                /* Saturated: route half-delta to next-prime-step neighbour slot */
                int alt_idx = (slot_idx + (int)PRIMES_URF[p_idx]) % total_slots;
                Slot4096 *s2 = lattice_get_slot(lat, alt_idx);
                if (s2 && s2->mantissa_words
                        && s2->mantissa_words[0] <= CHARGE_SAT_LIMIT) {
                    s2->mantissa_words[0] += delta64 / 2;
                }
                sat_count++;
            }
        }

        // (b) Phase: global spectral phase gated by v3 cosine term
        s->phase = fmod(s->phase
                        + bias_alpha * global_spec.temporal_phase * cos_term,
                        2.0 * M_PI);

        // (c) Freq: β × φ rank nudge weighted by prime entropy
        s->freq = fmod(s->freq
                       + bias_alpha * beta_nf * PHI_CORP * entropy_term,
                       2.0 * M_PI);
    }

    // Print top-10 for verification (slot via router projection — same as inference)
    if (sat_count > 0)
        printf("[hdgl_corpus_seeder] Saturation: %d/%zu slots hit cap (%.1f%%) "
               "— charge overflow routed to prime-step neighbours\n",
               sat_count, n_bias, 100.0 * sat_count / (double)n_bias);
    printf("[hdgl_corpus_seeder] Top-10 corpus tokens:\n");
    for (size_t i = 0; i < k && i < 10; i++) {
        int slot_idx = hdgl_router_key_to_slot(sorted[i].key, total_slots);
        printf("  [%2zu] %-30s  count=%-8" PRIu64 "  slot=%d\n",
               i, sorted[i].key, sorted[i].count, slot_idx);
    }

    // 8. Serialise — same binary format as hdgl_lattice_generator.c
    FILE *out = fopen(outfile, "wb");
    if (!out) {
        fprintf(stderr, "ERROR: cannot open output: %s\n", outfile);
        lattice_free(lat); free(sorted); free(g_table); return 1;
    }

    fwrite("HDGL", 1, 4, out);
    write_u32(out, 0x00020000u);              // VERSION
    write_u32(out, (uint32_t)num_instances);
    write_u32(out, (uint32_t)slots_per);
    write_f64(out, lat->time);
    write_f64(out, lat->omega);
    write_f64(out, lat->phase_var);

    int written = 0, skipped = 0;
    for (int i = 0; i < total_slots; i++) {
        Slot4096  *s   = lattice_get_slot(lat, i);
        SlotRecord rec = {0};
        if (s && s->mantissa_words) {
            rec.mantissa_word0 = s->mantissa_words[0];
            rec.exponent       = s->exponent;
            rec.phase          = s->phase;
            rec.freq           = s->freq;
            rec.state_flags    = s->state_flags;
            rec.strand_idx     = (uint32_t)(i % SPIRAL8_GEOMETRIES);
            written++;
        } else {
            skipped++;
        }
        fwrite(&rec, sizeof(rec), 1, out);
    }
    fclose(out);

    printf("[hdgl_corpus_seeder] Written : %d slots (%d skipped)\n",
           written, skipped);
    printf("[hdgl_corpus_seeder] Output  : %s  (%.2f MB)\n", outfile,
           (double)(total_slots * (int)sizeof(SlotRecord) + 36)
           / (1024.0 * 1024.0));
    printf("[hdgl_corpus_seeder] Load via: --hdgl-load %s\n", outfile);

    // 9. Write hdgl_ngrams.bin — per-expert bigram table for HDGL generation
    //    Format: "NGRAM" | uint32 experts | uint32 slots | NgramRecord[][]
    {
        // Derive ngrams output path from lattice output path
        char ngram_out[4096];
        strncpy(ngram_out, outfile, sizeof(ngram_out) - 16);
        ngram_out[sizeof(ngram_out) - 16] = '\0';
        // Replace extension or append _ngrams.bin
        char *dot = strrchr(ngram_out, '.');
        if (dot) strcpy(dot, "_ngrams.bin");
        else      strcat(ngram_out, "_ngrams.bin");

        FILE *nout = fopen(ngram_out, "wb");
        if (!nout) {
            fprintf(stderr, "[hdgl_corpus_seeder] WARN: cannot write ngrams: %s\n", ngram_out);
        } else {
            fwrite("NGRAM", 1, 5, nout);
            uint32_t ne = NGRAM_EXPERTS, ns = NGRAM_SLOTS;
            fwrite(&ne, 4, 1, nout);
            fwrite(&ns, 4, 1, nout);
            fwrite(g_ngrams, sizeof(g_ngrams), 1, nout);
            fclose(nout);
            /* Count non-empty bigrams for display */
            int nb = 0;
            for (int e = 0; e < NGRAM_EXPERTS; e++)
                for (int s2 = 0; s2 < NGRAM_SLOTS; s2++)
                    if (g_ngrams[e][s2].key[0]) nb++;
            printf("[hdgl_corpus_seeder] Ngrams  : %d bigrams → %s\n", nb, ngram_out);
        }
    }

    g_hdgl_lattice = NULL;
    lattice_free(lat);
    free(sorted);
    free(g_table);
    return 0;
}
