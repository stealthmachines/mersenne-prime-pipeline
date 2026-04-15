/*
 * analog_batch.cu — GPU-parallel 8D Kuramoto RK4 batch evaluator
 *
 * Replaces 76 sequential bot.exe Kuramoto calls with a single GPU kernel launch.
 * One CUDA thread per question — RTX 2060 (1920 cores) handles 76 in parallel,
 * each running up to 4096 RK4 steps.
 *
 * Architecture:
 *   CPU: tokenise questions (FNV1a) → compute initial AnalogState8D per question
 *   GPU: parallel analog8_score_until_lock for all questions simultaneously
 *   CPU: collect results → print JSON (one line per question)
 *
 * Output (stdout, one JSON per line):
 *   {"q_idx":0,"pvar":1.234,"locked":1,"steps":892,"band":"Beta","phase":"Lock"}
 *
 * Usage:
 *   echo "Question one" | analog_batch.exe
 *   analog_batch.exe < questions.txt
 *   analog_batch.exe --alpha-mode < questions.txt
 *
 * Build: build_analog_cuda.bat
 *
 * Licensed per https://zchg.org/t/legal-notice-copyright-applicable-ip-and-licensing-read-me/440
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

/* Windows MSVCRT compatibility */
#if defined(_WIN32) && !defined(strtok_r)
#  define strtok_r strtok_s
#endif
#ifndef _CRT_SECURE_NO_WARNINGS
#  define _CRT_SECURE_NO_WARNINGS
#endif

/* ── Constants (mirror analog_engine.h exactly) ──────────────────────────── */
#define ANG_DIMS         8
#define ANG_PHASE_HIST   200
#define ANG_LOCK_WINDOW  50
#define ANG_LOCK_CV      0.05
#define ANG_MAX_STEPS    4096
#define ANG_DT           0.01
#define ANG_SAT          1e6
#define ANG_PHI          1.6180339887498948
#define ANG_INV_PHI      0.6180339887498948
#define ANG_PI           3.14159265358979323846

#define ANG_CV_TO_SUSTAIN   0.50
#define ANG_CV_TO_FINETUNE  0.30
#define ANG_CV_TO_LOCK      0.10
#define ANG_EMERGENCY_VAR   10.0
#define ANG_SHA_INTERVAL    100

#define MAX_QUESTIONS    1024
#define MAX_WORDS_PER_Q  256
#define MAX_Q_LEN        2048

/* Adaptive phase params */
static __constant__ double d_APHASE_GAMMA[4]    = {0.005, 0.008, 0.010, 0.012};
static __constant__ double d_APHASE_COUPLING[4] = {5.0,   3.0,   2.0,   1.8};

static const double H_APHASE_GAMMA[4]    = {0.005, 0.008, 0.010, 0.012};
static const double H_APHASE_COUPLING[4] = {5.0,   3.0,   2.0,   1.8};

static const double BASE_INF_SEEDS[8] = {
    1.6180339887, 2.6180339887, 3.6180339887, 4.8541019662,
    5.6180339887, 6.4721359549, 7.8541019662, 8.3141592654,
};

/* ── GPU-side state (all fields needed for RK4 + lock detection) ─────────── */
typedef struct {
    double re[ANG_DIMS], im[ANG_DIMS];
    double freqs[ANG_DIMS], phases[ANG_DIMS];
    double gamma, k_coupling;
    double phase_var, prev_phase_var;
    double mean_freq;
    double phase_history[ANG_PHASE_HIST];
    uint64_t step_count;
    int hist_idx, hist_count;
    int aphase;     /* 0=Pluck 1=Sustain 2=FineTune 3=Lock */
    int bw_band;    /* 0=Delta 1=Theta 2=Alpha 3=Beta 4=Gamma */
    int locked;
    uint64_t lock_step;
} GpuAnalogState;

/* ── GPU result ──────────────────────────────────────────────────────────── */
typedef struct {
    double pvar;
    int    locked;
    uint64_t steps;
    int    band;
    int    aphase;
} GpuResult;

/* ── GPU: phase variance ─────────────────────────────────────────────────── */
__device__ double gpu_phase_var(const GpuAnalogState *s) {
    double sum = 0.0;
    for (int i = 0; i < ANG_DIMS; i++) sum += s->phases[i];
    double mean = sum / (double)ANG_DIMS;
    double var = 0.0;
    for (int i = 0; i < ANG_DIMS; i++) {
        double d = s->phases[i] - mean;
        var += d * d;
    }
    return var / (double)ANG_DIMS;
}

/* ── GPU: lock check ─────────────────────────────────────────────────────── */
__device__ int gpu_is_locked(const GpuAnalogState *s) {
    if (s->hist_count < ANG_LOCK_WINDOW) return 0;
    double sum = 0.0;
    for (int i = 0; i < ANG_LOCK_WINDOW; i++) {
        int idx = (s->hist_idx - 1 - i + ANG_PHASE_HIST) % ANG_PHASE_HIST;
        sum += s->phase_history[idx];
    }
    double mean = sum / (double)ANG_LOCK_WINDOW;
    if (mean < 1e-12) return 0;
    double sq = 0.0;
    for (int i = 0; i < ANG_LOCK_WINDOW; i++) {
        int idx = (s->hist_idx - 1 - i + ANG_PHASE_HIST) % ANG_PHASE_HIST;
        double d = s->phase_history[idx] - mean;
        sq += d * d;
    }
    return sqrt(sq / (double)ANG_LOCK_WINDOW) / mean < ANG_LOCK_CV;
}

/* ── GPU: lightweight phase feedback (replaces SHA-256 — same purpose) ───── */
__device__ void gpu_phase_feedback(GpuAnalogState *s) {
    uint64_t h = 0xcbf29ce484222325ULL;
    for (int i = 0; i < ANG_DIMS; i++) {
        uint32_t ri, ii;
        float rf = (float)s->re[i], imf = (float)s->im[i];
        memcpy(&ri, &rf, 4); memcpy(&ii, &imf, 4);
        h ^= (uint64_t)ri; h *= 0x100000001b3ULL;
        h ^= (uint64_t)ii; h *= 0x100000001b3ULL;
    }
    for (int i = 0; i < ANG_DIMS; i++) {
        h ^= h >> 17; h ^= h << 31; h ^= h >> 8;
        double p = (double)(h & 0xFFFFFFu) / (double)0x1000000u * 0.2 - 0.1;
        s->phases[i] = fmod(s->phases[i] + p + 2.0 * ANG_PI, 2.0 * ANG_PI);
    }
}

/* ── GPU: single RK4 step (full port of analog8_rk4_step) ───────────────── */
__device__ void gpu_rk4_step(GpuAnalogState *s) {
    const double dt = ANG_DT;
    double k1r[ANG_DIMS], k1i[ANG_DIMS];
    double k2r[ANG_DIMS], k2i[ANG_DIMS];
    double k3r[ANG_DIMS], k3i[ANG_DIMS];
    double k4r[ANG_DIMS], k4i[ANG_DIMS];
    double tr[ANG_DIMS],  ti[ANG_DIMS];

#define EVAL_GPU(sr, si, or_, oi) \
    for (int _i = 0; _i < ANG_DIMS; _i++) { \
        double _c = 0.0; \
        for (int _j = 0; _j < ANG_DIMS; _j++) { \
            if (_j != _i) { \
                double _dr = (sr)[_j]-(sr)[_i], _di = (si)[_j]-(si)[_i]; \
                _c += s->k_coupling * sqrt(_dr*_dr + _di*_di); \
            } \
        } \
        double _cp = cos(s->phases[_i]), _sp = sin(s->phases[_i]); \
        (or_)[_i] = s->freqs[_i]*_cp + _c - s->gamma*(sr)[_i]; \
        (oi)[_i]  = s->freqs[_i]*_sp + _c - s->gamma*(si)[_i]; \
        double _m = sqrt((or_)[_i]*(or_)[_i]+(oi)[_i]*(oi)[_i]); \
        if (_m > ANG_SAT) { (or_)[_i] *= ANG_SAT/_m; (oi)[_i] *= ANG_SAT/_m; } \
    }

    EVAL_GPU(s->re, s->im, k1r, k1i);
    for (int i=0;i<ANG_DIMS;i++){tr[i]=s->re[i]+0.5*dt*k1r[i]; ti[i]=s->im[i]+0.5*dt*k1i[i];}
    EVAL_GPU(tr, ti, k2r, k2i);
    for (int i=0;i<ANG_DIMS;i++){tr[i]=s->re[i]+0.5*dt*k2r[i]; ti[i]=s->im[i]+0.5*dt*k2i[i];}
    EVAL_GPU(tr, ti, k3r, k3i);
    for (int i=0;i<ANG_DIMS;i++){tr[i]=s->re[i]+dt*k3r[i]; ti[i]=s->im[i]+dt*k3i[i];}
    EVAL_GPU(tr, ti, k4r, k4i);
#undef EVAL_GPU

    for (int i = 0; i < ANG_DIMS; i++) {
        s->re[i] += (k1r[i]+2.0*k2r[i]+2.0*k3r[i]+k4r[i]) * dt / 6.0;
        s->im[i] += (k1i[i]+2.0*k2i[i]+2.0*k3i[i]+k4i[i]) * dt / 6.0;
        s->phases[i] = fmod(s->phases[i] + s->freqs[i]*dt, 2.0*ANG_PI);
    }

    s->step_count++;
    s->prev_phase_var = s->phase_var;
    s->phase_var = gpu_phase_var(s);

    /* Band */
    {
        double mf = 0.0;
        for (int i=0;i<ANG_DIMS;i++) mf += s->freqs[i];
        mf /= (double)ANG_DIMS; s->mean_freq = mf;
        if      (mf < 0.2513) s->bw_band = 0;
        else if (mf < 0.5027) s->bw_band = 1;
        else if (mf < 0.8168) s->bw_band = 2;
        else if (mf < 1.8850) s->bw_band = 3;
        else                  s->bw_band = 4;
    }

    /* History */
    s->phase_history[s->hist_idx] = s->phase_var;
    s->hist_idx = (s->hist_idx + 1) % ANG_PHASE_HIST;
    if (s->hist_count < ANG_PHASE_HIST) s->hist_count++;

    /* Emergency damping */
    if (s->phase_var > ANG_EMERGENCY_VAR) {
        s->gamma = 0.040; s->k_coupling = 0.5;
    }

    /* Adaptive phase advance */
    if (s->hist_count >= ANG_LOCK_WINDOW && s->aphase < 3) {
        double sum = 0.0;
        for (int i=0;i<ANG_LOCK_WINDOW;i++) {
            int idx = (s->hist_idx-1-i+ANG_PHASE_HIST)%ANG_PHASE_HIST;
            sum += s->phase_history[idx];
        }
        double mean = sum / (double)ANG_LOCK_WINDOW;
        if (mean > 1e-12) {
            double sq = 0.0;
            for (int i=0;i<ANG_LOCK_WINDOW;i++) {
                int idx=(s->hist_idx-1-i+ANG_PHASE_HIST)%ANG_PHASE_HIST;
                double d = s->phase_history[idx]-mean; sq+=d*d;
            }
            double cv = sqrt(sq/(double)ANG_LOCK_WINDOW)/mean;
            double trend = s->phase_var - s->prev_phase_var;
            double thresholds[3] = {ANG_CV_TO_SUSTAIN, ANG_CV_TO_FINETUNE, ANG_CV_TO_LOCK};
            if (cv < thresholds[s->aphase] && trend <= 0.0) {
                s->aphase++;
                s->gamma      = d_APHASE_GAMMA[s->aphase];
                s->k_coupling = d_APHASE_COUPLING[s->aphase];
            } else if (trend > 0.0 && s->phase_var > 0.1 && s->aphase > 0) {
                s->aphase--;
                s->gamma      = d_APHASE_GAMMA[s->aphase];
                s->k_coupling = d_APHASE_COUPLING[s->aphase];
            }
        }
    }

    /* Phase feedback every SHA_INTERVAL steps */
    if (s->step_count % ANG_SHA_INTERVAL == 0)
        gpu_phase_feedback(s);
}

/* ── GPU kernel: one thread per question ────────────────────────────────── */
__global__ void analog_batch_kernel(
    const GpuAnalogState *__restrict__ states_in,
    GpuResult *__restrict__ results,
    int n_questions,
    int max_steps
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_questions) return;

    /* Copy state into local registers/stack — fast private memory */
    GpuAnalogState s = states_in[idx];

    for (int step = 0; step < max_steps; step++) {
        gpu_rk4_step(&s);
        if (gpu_is_locked(&s)) {
            s.locked    = 1;
            s.lock_step = s.step_count;
            break;
        }
    }

    results[idx].pvar   = s.phase_var;
    results[idx].locked = s.locked;
    results[idx].steps  = s.step_count;
    results[idx].band   = s.bw_band;
    results[idx].aphase = s.aphase;
}

/* ── Host: FNV1a tokeniser (mirror bot.c) ────────────────────────────────── */
static uint32_t fnv1a(const char *s, size_t len) {
    uint32_t h = 2166136261u;
    for (size_t i = 0; i < len; i++) { h ^= (uint8_t)s[i]; h *= 16777619u; }
    return h;
}

/* ── Host: deterministic pseudo-random (mirror analog_engine.c) ─────────── */
static double det_rand(uint64_t seed) {
    uint64_t x = seed;
    x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
    return (double)(x * 0x2545F4914F6CDD1DULL) / (double)UINT64_MAX;
}

/* ── Host: initialise state (mirror analog8_init) ───────────────────────── */
static void host_analog8_init(GpuAnalogState *s, uint64_t seed) {
    memset(s, 0, sizeof(*s));
    s->aphase     = 0; /* PLUCK */
    s->gamma      = H_APHASE_GAMMA[0];
    s->k_coupling = H_APHASE_COUPLING[0];
    for (int i = 0; i < ANG_DIMS; i++) {
        s->re[i]     = BASE_INF_SEEDS[i];
        s->im[i]     = ANG_INV_PHI * det_rand(seed + (uint64_t)i * 7ULL);
        s->freqs[i]  = 1.0 + 0.5 * det_rand(seed + (uint64_t)i);
        s->phases[i] = 2.0 * ANG_PI * det_rand(seed + 100ULL + (uint64_t)i);
    }
}

/* ── Host: apply token (mirror analog8_apply_token) ─────────────────────── */
static void host_apply_token(GpuAnalogState *s, uint32_t token_id, uint32_t expert_id, int tok_idx) {
    /* token_to_analog_entry */
    double t  = (double)(token_id & 0xFFFF) / 65536.0;
    double k  = 1.5 + sin(t * 2.0 * ANG_PI * ANG_PHI);
    double g  = 0.010 + 0.040 * ((double)(expert_id % 512u) / 512.0);
    uint64_t h = (uint64_t)token_id * 0xDEADBEEFCAFEBABEULL;
    h ^= h>>17; h ^= h<<31; h ^= h>>8;
    double ph = 2.0 * ANG_PI * ((double)(h & 0xFFFFFFu) / (double)0x1000000u);

    /* analog8_apply_token */
    int dim = ((unsigned)tok_idx) % ANG_DIMS;
    s->freqs[dim] += ANG_INV_PHI * (k - s->freqs[dim]);
    s->gamma      += 0.05 * ANG_INV_PHI * (g - s->gamma);
    s->phases[dim] = fmod(s->phases[dim] + ph * 0.1, 2.0 * ANG_PI);
}

/* ── Host: band/phase name helpers ──────────────────────────────────────── */
static const char *band_name(int b) {
    static const char *n[] = {"Delta","Theta","Alpha","Beta","Gamma"};
    return (b>=0&&b<5) ? n[b] : "?";
}
static const char *phase_name(int p) {
    static const char *n[] = {"Pluck","Sustain","FineTune","Lock"};
    return (p>=0&&p<4) ? n[p] : "?";
}

/* ── Band tuning (mirror analog8_tune_band) ────────────────────────────── */
static void host_tune_band(GpuAnalogState *s, double target_hz) {
    double target_rad = target_hz * 2.0 * ANG_PI * 0.01;
    double mean = 0.0;
    for (int i=0;i<ANG_DIMS;i++) mean += s->freqs[i];
    mean /= (double)ANG_DIMS;
    if (mean < 1e-9) mean = 1.0;
    double ratio = target_rad / mean;
    for (int i=0;i<ANG_DIMS;i++) s->freqs[i] *= ratio;
    s->mean_freq = target_rad;
}

/* ── Main ────────────────────────────────────────────────────────────────── */
int main(int argc, char **argv) {
    double band_hz = 10.5; /* Alpha default */
    for (int a = 1; a < argc; a++) {
        if (strcmp(argv[a], "--alpha-mode") == 0) band_hz = 10.5;
        else if (strcmp(argv[a], "--beta-mode")  == 0) band_hz = 20.0;
        else if (strcmp(argv[a], "--gamma-mode") == 0) band_hz = 40.0;
        else if (strcmp(argv[a], "--theta-mode") == 0) band_hz =  6.0;
        else if (strcmp(argv[a], "--delta-mode") == 0) band_hz =  2.0;
    }

    /* Read questions from stdin (one per line, blank lines skipped) */
    static char questions[MAX_QUESTIONS][MAX_Q_LEN];
    static GpuAnalogState h_states[MAX_QUESTIONS];
    int n_q = 0;

    char line[MAX_Q_LEN];
    while (fgets(line, sizeof(line), stdin) && n_q < MAX_QUESTIONS) {
        size_t len = strlen(line);
        while (len > 0 && (line[len-1] == '\n' || line[len-1] == '\r')) line[--len] = '\0';
        if (len == 0) continue;
        strncpy(questions[n_q], line, MAX_Q_LEN-1);

        /* Build initial analog state for this question */
        host_analog8_init(&h_states[n_q], (uint64_t)fnv1a(line, len));
        host_tune_band(&h_states[n_q], band_hz);

        /* Tokenise: split on spaces/punctuation, apply each word */
        char tmp[MAX_Q_LEN];
        strncpy(tmp, line, MAX_Q_LEN-1);
        char *save = NULL;
        char *word = strtok_r(tmp, " \t\r\n.,!?;:'\"()-", &save);
        int tok_idx = 0;
        while (word && tok_idx < MAX_WORDS_PER_Q) {
            size_t wlen = strlen(word);
            if (wlen >= 3) {
                uint32_t token_id  = fnv1a(word, wlen);
                /* Simple deterministic expert: fnv1a of word → mod 512 */
                uint32_t expert_id = (token_id ^ (token_id >> 16)) % 512u;
                host_apply_token(&h_states[n_q], token_id, expert_id, tok_idx);
                tok_idx++;
            }
            word = strtok_r(NULL, " \t\r\n.,!?;:'\"()-", &save);
        }
        n_q++;
    }

    if (n_q == 0) {
        fprintf(stderr, "[analog_batch] no questions on stdin\n");
        return 1;
    }

    /* Allocate GPU memory */
    GpuAnalogState *d_states = NULL;
    GpuResult      *d_results = NULL;
    size_t states_bytes  = (size_t)n_q * sizeof(GpuAnalogState);
    size_t results_bytes = (size_t)n_q * sizeof(GpuResult);

    cudaError_t err;
    err = cudaMalloc(&d_states,  states_bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "[analog_batch] cudaMalloc states failed: %s\n", cudaGetErrorString(err));
        return 1;
    }
    err = cudaMalloc(&d_results, results_bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "[analog_batch] cudaMalloc results failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_states); return 1;
    }

    /* Copy states to GPU */
    cudaMemcpy(d_states, h_states, states_bytes, cudaMemcpyHostToDevice);

    /* Launch: 1 thread per question, up to 256 threads/block */
    int block = (n_q < 256) ? n_q : 256;
    int grid  = (n_q + block - 1) / block;
    analog_batch_kernel<<<grid, block>>>(d_states, d_results, n_q, ANG_MAX_STEPS);

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "[analog_batch] kernel failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_states); cudaFree(d_results); return 1;
    }

    /* Copy results back */
    static GpuResult h_results[MAX_QUESTIONS];
    cudaMemcpy(h_results, d_results, results_bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_states);
    cudaFree(d_results);

    /* Print JSON lines to stdout */
    for (int i = 0; i < n_q; i++) {
        printf("{\"q_idx\":%d,\"pvar\":%.4f,\"locked\":%d,\"steps\":%llu,\"band\":\"%s\",\"phase\":\"%s\"}\n",
               i,
               h_results[i].pvar,
               h_results[i].locked,
               (unsigned long long)h_results[i].steps,
               band_name(h_results[i].band),
               phase_name(h_results[i].aphase));
    }
    fflush(stdout);
    return 0;
}
