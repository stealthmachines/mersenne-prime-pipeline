// image_caption.c — HDGL-28 image captioner (pure C, Windows, zero Python at runtime)
//
// Walks pipeline\images\ (images extracted from the Discourse backup),
// sends each image as base64 to a local Ollama vision model via raw
// Winsock2 TCP, and writes:
//   {"hash":"BASENAME.ext","caption":"TEXT"}
// to image_captions.jsonl for ingestion by hdgl_corpus_seeder.exe.
//
// The "hash" key matches the Discourse upload:// token basename exactly:
//   upload://nUMMm7tH5.jpeg  →  {"hash":"nUMMm7tH5.jpeg", ...}
// so image_link.py or a future C linker can resolve the connection.
//
// Formula-first prompt: zchg.org images are primarily equations/diagrams.
// The Ollama prompt explicitly requests equation transcription before description.
//
// SETUP (one-time):   python pipeline\extract_images.py
// BUILD:              build_image_caption_windows.bat
// RUN:                image_caption.exe [options]
// RESUME:             run again — already-captioned hashes are skipped.
//
// ZCHG License: https://zchg.org/t/legal-notice-copyright-applicable-ip-and-licensing-read-me/440

#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0600
#endif
#define WIN32_LEAN_AND_MEAN

#include <windows.h>
#include <winsock2.h>
#include <ws2tcpip.h>

#include "jsmn.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <ctype.h>

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

#define DEFAULT_IMAGES_DIR  "..\\pipeline\\images"
#define DEFAULT_OUT_JSONL   "..\\pipeline\\sft\\image_captions.jsonl"
#define DEFAULT_MODEL       "moondream"
#define DEFAULT_HOST        "127.0.0.1"
#define DEFAULT_PORT        11434
#define DEFAULT_DELAY_MS    0

// Image size cap: images beyond this are read truncated.
// 512 KB → ~683 KB base64 — fits well within a typical Ollama request.
#define MAX_IMAGE_BYTES     (512 * 1024)
#define MAX_B64_BYTES       (700 * 1024)   // base64 overhead ≈ 1.37×
#define MAX_PAYLOAD_BYTES   (810 * 1024)   // JSON body (model + prompt + b64)
#define MAX_REQUEST_BYTES   (820 * 1024)   // HTTP headers + body
#define MAX_RESPONSE_BYTES  (256 * 1024)   // Ollama JSON response
#define MAX_CAPTION_LEN      16384          // max caption chars stored
#define MAX_HASH_LEN           512          // filename/hash max length

// Image extensions accepted — lowercase, including dot
static const char *IMAGE_EXTS[] = { ".jpg", ".jpeg", ".png", ".gif", ".webp", NULL };

// Ollama description prompt — kept short for moondream (1B phi2, limited instruction-follow)
static const char *OLLAMA_PROMPT = "Describe this image.";

// ---------------------------------------------------------------------------
// Global buffers — static to avoid stack overflow on large images
// ---------------------------------------------------------------------------

static unsigned char g_imgbuf[MAX_IMAGE_BYTES];
static char          g_b64buf[MAX_B64_BYTES];
static char          g_payload[MAX_PAYLOAD_BYTES];
static char          g_req[MAX_REQUEST_BYTES];
static char          g_rsp[MAX_RESPONSE_BYTES];

// ---------------------------------------------------------------------------
// Checkpoint: open-address hash set of already-captioned filenames
// Avoids re-calling Ollama for images processed in a prior run.
// ---------------------------------------------------------------------------

#define SEEN_BITS  14
#define SEEN_SIZE  (1 << SEEN_BITS)   // 16384 buckets — ample for 8696 images
#define SEEN_MASK  (SEEN_SIZE - 1)

typedef struct { char key[MAX_HASH_LEN]; } SeenEntry;
static SeenEntry g_seen[SEEN_SIZE];
static int       g_n_seen = 0;

static uint32_t seen_djb2(const char *s) {
    uint32_t h = 5381;
    while (*s) h = h * 33u + (unsigned char)*s++;
    return h;
}

static int seen_contains(const char *key) {
    uint32_t idx = seen_djb2(key) & SEEN_MASK;
    for (uint32_t p = 0; p < SEEN_SIZE; p++) {
        uint32_t i = (idx + p) & SEEN_MASK;
        if (!g_seen[i].key[0]) return 0;
        if (strcmp(g_seen[i].key, key) == 0) return 1;
    }
    return 0;
}

static void seen_insert(const char *key) {
    if (strlen(key) >= MAX_HASH_LEN) return;
    uint32_t idx = seen_djb2(key) & SEEN_MASK;
    for (uint32_t p = 0; p < SEEN_SIZE; p++) {
        uint32_t i = (idx + p) & SEEN_MASK;
        if (!g_seen[i].key[0] || strcmp(g_seen[i].key, key) == 0) {
            strncpy(g_seen[i].key, key, MAX_HASH_LEN - 1);
            g_seen[i].key[MAX_HASH_LEN - 1] = '\0';
            g_n_seen++;
            return;
        }
    }
}

// Load existing image_captions.jsonl and populate seen set.
// Called at startup to enable safe resume after interruption.
static void load_checkpoint(const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) return;  // file doesn't exist yet — first run
    char line[MAX_HASH_LEN + 64];
    while (fgets(line, sizeof(line), f)) {
        const char *p = strstr(line, "\"hash\":\"");
        if (!p) continue;
        p += 8;  // skip past "hash":"
        const char *q = strchr(p, '"');
        if (!q || q - p >= MAX_HASH_LEN) continue;
        char key[MAX_HASH_LEN];
        size_t len = (size_t)(q - p);
        memcpy(key, p, len);
        key[len] = '\0';
        seen_insert(key);
    }
    fclose(f);
    if (g_n_seen > 0)
        printf("[image_caption] Checkpoint: %d already captioned — resuming.\n", g_n_seen);
}

// ---------------------------------------------------------------------------
// Base64 encoder (RFC 4648, no line breaks)
// LUT-based, processes 3 bytes → 4 chars. Returns encoded length.
// ---------------------------------------------------------------------------

static const char B64_TABLE[] =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

static size_t base64_encode(const unsigned char *in, size_t inlen, char *out) {
    size_t o = 0, i = 0;
    for (; i + 2 < inlen; i += 3) {
        out[o++] = B64_TABLE[(in[i]   >> 2) & 0x3F];
        out[o++] = B64_TABLE[((in[i]   & 3) << 4) | (in[i+1] >> 4)];
        out[o++] = B64_TABLE[((in[i+1] & 0xF) << 2) | (in[i+2] >> 6)];
        out[o++] = B64_TABLE[in[i+2] & 0x3F];
    }
    if (i < inlen) {
        out[o++] = B64_TABLE[(in[i] >> 2) & 0x3F];
        if (i + 1 < inlen) {
            out[o++] = B64_TABLE[((in[i] & 3) << 4) | (in[i+1] >> 4)];
            out[o++] = B64_TABLE[(in[i+1] & 0xF) << 2];
        } else {
            out[o++] = B64_TABLE[(in[i] & 3) << 4];
            out[o++] = '=';
        }
        out[o++] = '=';
    }
    out[o] = '\0';
    return o;
}

// ---------------------------------------------------------------------------
// JSON string escaper
// Writes JSON-safe escaped version of src into dst. Returns written bytes.
// dst must be at least 2× src in size to guarantee no overflow.
// ---------------------------------------------------------------------------

static size_t json_escape(const char *src, char *dst, size_t dst_max) {
    size_t o = 0;
    for (const char *p = src; *p && o + 6 < dst_max; p++) {
        unsigned char c = (unsigned char)*p;
        if      (c == '"')  { dst[o++] = '\\'; dst[o++] = '"';  }
        else if (c == '\\') { dst[o++] = '\\'; dst[o++] = '\\'; }
        else if (c == '\n') { dst[o++] = '\\'; dst[o++] = 'n';  }
        else if (c == '\r') { dst[o++] = '\\'; dst[o++] = 'r';  }
        else if (c == '\t') { dst[o++] = '\\'; dst[o++] = 't';  }
        else if (c < 0x20)  { dst[o++] = ' '; }  // strip other control chars
        else                 { dst[o++] = (char)c; }
    }
    dst[o] = '\0';
    return o;
}

// ---------------------------------------------------------------------------
// Winsock2 HTTP POST to Ollama /api/chat
//
// Transport: raw TCP, HTTP/1.1, blocking recv, connection-close per request.
// stream:false — Ollama sends one complete JSON blob with Content-Length.
// Response parsing: jsmn tokenizer, finds message.content.
//
// Returns 1 on success with caption_out populated; 0 on any failure.
// ---------------------------------------------------------------------------

static int ollama_chat(const char *host, int port,
                       const char *json_body, size_t body_len,
                       char *caption_out, size_t cap_max) {
    // Build HTTP/1.1 headers
    int hdr_len = snprintf(g_req, 512,
        "POST /api/chat HTTP/1.1\r\n"
        "Host: %s:%d\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: %zu\r\n"
        "Connection: close\r\n"
        "\r\n",
        host, port, body_len);

    if (hdr_len < 0 || (size_t)hdr_len + body_len >= MAX_REQUEST_BYTES) {
        fprintf(stderr, "[image_caption] Request too large (%zu bytes)\n",
                (size_t)hdr_len + body_len);
        return 0;
    }
    memcpy(g_req + hdr_len, json_body, body_len);
    size_t total_req = (size_t)hdr_len + body_len;

    // Resolve and connect
    struct sockaddr_in sa;
    memset(&sa, 0, sizeof(sa));
    sa.sin_family = AF_INET;
    sa.sin_port   = htons((unsigned short)port);
    if (inet_pton(AF_INET, host, &sa.sin_addr) != 1) {
        fprintf(stderr, "[image_caption] Invalid host: %s\n", host);
        return 0;
    }

    SOCKET sock = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (sock == INVALID_SOCKET) {
        fprintf(stderr, "[image_caption] socket() failed: %d\n", WSAGetLastError());
        return 0;
    }

    // 300-second receive timeout — vision inference slow on first model load
    DWORD timeout_ms = 300000;
    setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, (const char*)&timeout_ms, sizeof(timeout_ms));

    if (connect(sock, (struct sockaddr*)&sa, sizeof(sa)) == SOCKET_ERROR) {
        fprintf(stderr, "[image_caption] connect() failed: %d — is Ollama running?\n",
                WSAGetLastError());
        closesocket(sock);
        return 0;
    }

    // Send (loop handles partial sends on large base64 payloads)
    size_t sent = 0;
    while (sent < total_req) {
        int n = send(sock, g_req + sent, (int)(total_req - sent), 0);
        if (n == SOCKET_ERROR || n <= 0) {
            fprintf(stderr, "[image_caption] send() error: %d\n", WSAGetLastError());
            closesocket(sock);
            return 0;
        }
        sent += (size_t)n;
    }

    // Receive until connection closes (stream:false → single blob + close)
    size_t rcvd = 0;
    while (rcvd < MAX_RESPONSE_BYTES - 1) {
        int n = recv(sock, g_rsp + rcvd, (int)(MAX_RESPONSE_BYTES - 1 - rcvd), 0);
        if (n == 0) break;  // server closed — response complete
        if (n == SOCKET_ERROR) {
            int err = WSAGetLastError();
            if (err == WSAETIMEDOUT)
                fprintf(stderr, "[image_caption] recv() timeout after 120s\n");
            else
                fprintf(stderr, "[image_caption] recv() error: %d\n", err);
            break;
        }
        rcvd += (size_t)n;
    }
    closesocket(sock);
    g_rsp[rcvd] = '\0';

    if (rcvd == 0) {
        fprintf(stderr, "[image_caption] Empty response from Ollama\n");
        return 0;
    }

    // Skip HTTP headers — find \r\n\r\n separator
    const char *body_start = strstr(g_rsp, "\r\n\r\n");
    if (!body_start) {
        fprintf(stderr, "[image_caption] No HTTP body separator in response\n");
        return 0;
    }
    body_start += 4;

    // Parse JSON with jsmn — find message.content
    // Expected Ollama response shape (stream:false):
    //   {"model":"...","message":{"role":"assistant","content":"CAPTION"},"done":true}
    jsmn_parser jp;
    jsmntok_t   toks[512];
    jsmn_init(&jp);
    size_t body_len_actual = (size_t)((g_rsp + rcvd) - body_start);
    int r = jsmn_parse(&jp, body_start, body_len_actual, toks, 512);
    if (r < 0) {
        fprintf(stderr, "[image_caption] jsmn parse error %d (body=%zu bytes)\n",
                r, body_len_actual);
        return 0;
    }

    // Walk tokens in document order.
    // Also detect top-level "error" key so we can surface Ollama error strings
    // (e.g. "model not found") rather than a generic parse-failure message.
    int in_message = 0;
    for (int ti = 0; ti < r - 1; ti++) {
        jsmntok_t *t  = &toks[ti];
        jsmntok_t *tn = &toks[ti + 1];
        if (t->type != JSMN_STRING) continue;

        size_t      klen = (size_t)(t->end - t->start);
        const char *k    = body_start + t->start;

        // Detect Ollama error field — model not found, OOM, etc.
        if (klen == 5 && strncmp(k, "error", 5) == 0 && tn->type == JSMN_STRING) {
            size_t vlen = (size_t)(tn->end - tn->start);
            if (vlen > 256) vlen = 256;
            char errbuf[257];
            memcpy(errbuf, body_start + tn->start, vlen);
            errbuf[vlen] = '\0';
            fprintf(stderr, "[image_caption] Ollama error: %s\n", errbuf);
            return 0;
        }

        if (klen == 7 && strncmp(k, "message", 7) == 0 && tn->type == JSMN_OBJECT) {
            in_message = 1;
            continue;
        }
        if (in_message && klen == 7 && strncmp(k, "content", 7) == 0
                && tn->type == JSMN_STRING) {
            size_t vlen = (size_t)(tn->end - tn->start);
            if (vlen >= cap_max) vlen = cap_max - 1;
            memcpy(caption_out, body_start + tn->start, vlen);
            caption_out[vlen] = '\0';
            return 1;
        }
    }

    // Fallback: print the first 256 bytes of the response body for diagnosis
    char snippet[257];
    size_t snip_len = body_len_actual > 256 ? 256 : body_len_actual;
    memcpy(snippet, body_start, snip_len);
    snippet[snip_len] = '\0';
    fprintf(stderr, "[image_caption] Unexpected response (no message.content): %.256s\n", snippet);
    return 0;
}

// ---------------------------------------------------------------------------
// Directory walker + per-image processor
// Uses Win32 FindFirstFileA/FindNextFileA for OS-native directory iteration.
// ---------------------------------------------------------------------------

static int is_image_ext(const char *filename) {
    const char *dot = strrchr(filename, '.');
    if (!dot) return 0;
    char ext[16];
    size_t i;
    for (i = 0; dot[i] && i < 15; i++) ext[i] = (char)tolower((unsigned char)dot[i]);
    ext[i] = '\0';
    for (const char **e = IMAGE_EXTS; *e; e++)
        if (strcmp(ext, *e) == 0) return 1;
    return 0;
}

typedef struct {
    const char *host;
    int         port;
    const char *model;
    int         limit;
    int         delay_ms;
    FILE       *out_f;
    int         n_processed;
    int         n_skipped;
    int         n_errors;
} WalkState;

static void process_image(const char *fullpath, const char *basename, WalkState *st) {
    // Checkpoint skip
    if (seen_contains(basename)) {
        st->n_skipped++;
        return;
    }

    // Read image bytes (up to MAX_IMAGE_BYTES)
    FILE *img_f = fopen(fullpath, "rb");
    if (!img_f) {
        fprintf(stderr, "[image_caption] Cannot open: %s\n", fullpath);
        st->n_errors++;
        return;
    }
    size_t img_sz = fread(g_imgbuf, 1, MAX_IMAGE_BYTES, img_f);
    fclose(img_f);
    if (img_sz == 0) {
        fprintf(stderr, "[image_caption] Empty file: %s\n", fullpath);
        st->n_errors++;
        return;
    }

    // Base64 encode
    size_t b64_len = base64_encode(g_imgbuf, img_sz, g_b64buf);

    // Build Ollama JSON payload
    // Format: {"model":"NAME","messages":[{"role":"user","content":"PROMPT","images":["B64"]}],"stream":false}
    int payload_len = snprintf(g_payload, MAX_PAYLOAD_BYTES,
        "{\"model\":\"%s\","
        "\"messages\":[{"
            "\"role\":\"user\","
            "\"content\":\"%s\","
            "\"images\":[\"%.*s\"]"
        "}],"
        "\"stream\":false}",
        st->model,
        OLLAMA_PROMPT,
        (int)b64_len, g_b64buf);

    if (payload_len <= 0 || payload_len >= MAX_PAYLOAD_BYTES) {
        fprintf(stderr, "[image_caption] Payload overflow for: %s (%zu bytes raw)\n",
                basename, img_sz);
        st->n_errors++;
        return;
    }

    printf("[image_caption] [%d] %s (%zu KB) ... ",
           st->n_processed + 1 + st->n_skipped,
           basename, img_sz / 1024);
    fflush(stdout);

    // Call Ollama
    char caption[MAX_CAPTION_LEN];
    if (!ollama_chat(st->host, st->port, g_payload, (size_t)payload_len,
                     caption, MAX_CAPTION_LEN)) {
        fprintf(stderr, "[image_caption] FAILED: %s\n", basename);
        st->n_errors++;
        return;
    }

    // Escape both hash (filename) and caption for JSON output
    char esc_hash[MAX_HASH_LEN * 2];
    char esc_cap[MAX_CAPTION_LEN * 2];
    json_escape(basename, esc_hash, sizeof(esc_hash));
    json_escape(caption,  esc_cap,  sizeof(esc_cap));

    // Append JSONL line — fflush immediately so partial runs are safe to read
    fprintf(st->out_f, "{\"hash\":\"%s\",\"caption\":\"%s\"}\n", esc_hash, esc_cap);
    fflush(st->out_f);

    // Mark as done
    seen_insert(basename);
    st->n_processed++;

    printf("OK (%zu chars)\n", strlen(caption));

    if (st->delay_ms > 0) Sleep((DWORD)st->delay_ms);
}

static void walk_dir(const char *dir_path, WalkState *st) {
    if (st->limit > 0 && st->n_processed >= st->limit) return;

    char search[MAX_PATH];
    snprintf(search, sizeof(search), "%s\\*", dir_path);

    WIN32_FIND_DATAA fd;
    HANDLE h = FindFirstFileA(search, &fd);
    if (h == INVALID_HANDLE_VALUE) return;

    do {
        if (fd.cFileName[0] == '.') continue;  // skip . and ..

        char fullpath[MAX_PATH];
        snprintf(fullpath, sizeof(fullpath), "%s\\%s", dir_path, fd.cFileName);

        if (fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) {
            walk_dir(fullpath, st);
        } else if (is_image_ext(fd.cFileName)) {
            if (st->limit > 0 && st->n_processed >= st->limit) break;
            process_image(fullpath, fd.cFileName, st);
        }
    } while (FindNextFileA(h, &fd));

    FindClose(h);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char **argv) {
    const char *images_dir = DEFAULT_IMAGES_DIR;
    const char *out_path   = DEFAULT_OUT_JSONL;
    const char *model      = DEFAULT_MODEL;
    const char *host       = DEFAULT_HOST;
    int         port       = DEFAULT_PORT;
    int         limit      = 0;
    int         delay_ms   = DEFAULT_DELAY_MS;

    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "--images")   && i+1 < argc) images_dir = argv[++i];
        else if (!strcmp(argv[i], "--out")      && i+1 < argc) out_path   = argv[++i];
        else if (!strcmp(argv[i], "--model")    && i+1 < argc) model      = argv[++i];
        else if (!strcmp(argv[i], "--host")     && i+1 < argc) host       = argv[++i];
        else if (!strcmp(argv[i], "--port")     && i+1 < argc) port       = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--limit")    && i+1 < argc) limit      = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--delay-ms") && i+1 < argc) delay_ms   = atoi(argv[++i]);
        else {
            fprintf(stderr,
                "image_caption.exe — HDGL-28 Discourse image captioner\n"
                "\n"
                "Usage: image_caption.exe [options]\n"
                "  --images DIR      directory of extracted images\n"
                "                    (default: ..\\pipeline\\images)\n"
                "  --out FILE        output JSONL path\n"
                "                    (default: ..\\pipeline\\sft\\image_captions.jsonl)\n"
                "  --model NAME      Ollama vision model (default: qwen2.5vl:3b)\n"
                "  --host ADDR       Ollama host (default: 127.0.0.1)\n"
                "  --port N          Ollama port (default: 11434)\n"
                "  --limit N         stop after N images; 0 = all (default: 0)\n"
                "  --delay-ms N      ms to wait between requests (default: 0)\n"
                "\n"
                "Setup: run pipeline\\extract_images.py once to populate --images dir.\n"
                "Safe to interrupt and resume: already-captioned images are skipped.\n");
            return 1;
        }
    }

    printf("[image_caption] HDGL-28 Image Captioner\n");
    printf("[image_caption] Images dir : %s\n", images_dir);
    printf("[image_caption] Output     : %s\n", out_path);
    printf("[image_caption] Model      : %s\n", model);
    printf("[image_caption] Limit      : %s\n", limit > 0 ? "yes" : "all");

    // Initialise Winsock2
    WSADATA wsa;
    if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0) {
        fprintf(stderr, "[image_caption] WSAStartup failed\n");
        return 1;
    }

    // Load checkpoint — populate seen set from existing JSONL
    load_checkpoint(out_path);
    printf("[image_caption] Already done: %d\n", g_n_seen);

    // Open output in append mode — safe for resume
    FILE *out_f = fopen(out_path, "a");
    if (!out_f) {
        fprintf(stderr, "[image_caption] Cannot open output for writing: %s\n", out_path);
        WSACleanup();
        return 1;
    }

    WalkState st = {
        .host        = host,
        .port        = port,
        .model       = model,
        .limit       = limit,
        .delay_ms    = delay_ms,
        .out_f       = out_f,
        .n_processed = 0,
        .n_skipped   = g_n_seen,
        .n_errors    = 0,
    };

    walk_dir(images_dir, &st);
    fclose(out_f);

    printf("\n[image_caption] Complete.\n");
    printf("[image_caption]   Processed : %d\n", st.n_processed);
    printf("[image_caption]   Skipped   : %d (already done)\n", st.n_skipped);
    printf("[image_caption]   Errors    : %d\n", st.n_errors);
    printf("[image_caption]   Output    : %s\n", out_path);
    printf("[image_caption] Next step   : hdgl_corpus_seeder.exe --captions %s\n", out_path);

    WSACleanup();
    return 0;
}
