#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <unistd.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// --- System Constants ---
#define PHI 1.6180339887498948
#define MAX_INSTANCES 8388608
#define SLOTS_PER_INSTANCE 4
#define MAX_SLOTS (MAX_INSTANCES * SLOTS_PER_INSTANCE)
#define CHUNK_SIZE 1048576
#define MSB_MASK (1ULL << 63)

// --- Analog Constants (Tuned) ---
#define GAMMA 0.02
#define LAMBDA 0.05
#define SAT_LIMIT 1e6
#define NOISE_SIGMA 0.01
#define CONSENSUS_EPS 1e-6
#define CONSENSUS_N 100
#define ADAPT_THRESH 0.8
#define K_COUPLING 1.0

// --- Checkpoint Constants ---
#define CHECKPOINT_INTERVAL 100
#define SNAPSHOT_MAX 10
#define SNAPSHOT_DECAY 0.95

// --- Dₙ(r) Lattice Constants ---
#define NUM_DN 8
static const uint64_t FIB_TABLE[NUM_DN] = {1, 1, 2, 3, 5, 8, 13, 21};
static const uint64_t PRIME_TABLE[NUM_DN] = {2, 3, 5, 7, 11, 13, 17, 19};

// --- Base(∞) Numeric Lattice ---
typedef struct {
    double upper_field[7];
    double analog_dims[13];
    double void_state;
    double lower_field[8];
    double sibling_harmonics[8];
    double inf_layer[4];
    double choke_layer[4];
    double *base_infinity_seeds;
    size_t num_seeds;
} NumericLattice;

// --- MPI Stub ---
#define MPI_REAL 0
#if MPI_REAL
#include <mpi.h>
#define MPI_BCAST(buf, cnt, type, root, comm) MPI_Bcast(buf, cnt, type, root, MPI_COMM_WORLD)
#define MPI_REDUCE(buf, res, cnt, type, op, root, comm) MPI_Reduce(buf, res, cnt, type, op, root, MPI_COMM_WORLD)
#else
#define MPI_BCAST(buf, cnt, type, root, comm)
#define MPI_REDUCE(buf, res, cnt, type, op, root, comm)
#define MPI_SUM 0
#endif

// --- Timing ---
#ifdef USE_DS3231
#include <i2c/smbus.h>
#define DS3231_ADDR 0x68
static int i2c_fd = -1;
#endif

double get_normalized_rand() {
    return (double)rand() / RAND_MAX;
}

uint64_t det_rand(uint64_t seed) {
    seed ^= seed << 13;
    seed ^= seed >> 7;
    seed ^= seed << 17;
    return seed;
}

#define GET_RANDOM_UINT64() (((uint64_t)rand() << 32) | rand())

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Timing Primitives
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

int64_t get_rtc_ns() {
#ifdef USE_DS3231
    if (i2c_fd >= 0) {
        uint8_t data[7];
        if (i2c_smbus_read_i2c_block_data(i2c_fd, DS3231_ADDR, 0x00, 7, data) == 7) {
            int sec = ((data[0] >> 4) * 10) + (data[0] & 0x0F);
            int min = ((data[1] >> 4) * 10) + (data[1] & 0x0F);
            int hr = ((data[2] >> 4) * 10) + (data[2] & 0x0F);
            return (int64_t)(hr * 3600 + min * 60 + sec) * 1000000000LL;
        }
    }
#endif
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

void rtc_sleep_until(int64_t target_ns) {
    int64_t now = get_rtc_ns();
    if (target_ns <= now) return;
    struct timespec req = {
        .tv_sec = (target_ns - now) / 1000000000LL,
        .tv_nsec = (target_ns - now) % 1000000000LL
    };
    nanosleep(&req, NULL);
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Numeric Lattice Initialization
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

void init_numeric_lattice(NumericLattice *nl) {
    // Upper Field
    nl->upper_field[0] = 170.6180339887;
    nl->upper_field[1] = 150.9442719100;
    nl->upper_field[2] = 12.6180339887;
    nl->upper_field[3] = 8.8541019662;
    nl->upper_field[4] = 4.2360679775;
    nl->upper_field[5] = 3.6180339887;
    nl->upper_field[6] = 1.6180339887;

    // Analog Dimensionality
    nl->analog_dims[0] = 8.3141592654;
    nl->analog_dims[1] = 7.8541019662;
    nl->analog_dims[2] = 6.4721359549;
    nl->analog_dims[3] = 5.6180339887;
    nl->analog_dims[4] = 4.8541019662;
    nl->analog_dims[5] = 3.6180339887;
    nl->analog_dims[6] = 2.6180339887;
    nl->analog_dims[7] = 1.6180339887;
    nl->analog_dims[8] = 1.0000000000;
    nl->analog_dims[9] = 7.8541019662;
    nl->analog_dims[10] = 11.0901699437;
    nl->analog_dims[11] = 17.9442719100;
    nl->analog_dims[12] = 29.0344465435;

    // The Void
    nl->void_state = 0.0;

    // Lower Field
    nl->lower_field[0] = 0.0000000001;
    nl->lower_field[1] = 0.0344465435;
    nl->lower_field[2] = 0.0557280900;
    nl->lower_field[3] = 0.0901699437;
    nl->lower_field[4] = 0.1458980338;
    nl->lower_field[5] = 0.2360679775;
    nl->lower_field[6] = 0.3819660113;
    nl->lower_field[7] = 0.6180339887;

    // Sibling Harmonics
    nl->sibling_harmonics[0] = 0.0901699437;
    nl->sibling_harmonics[1] = 0.1458980338;
    nl->sibling_harmonics[2] = 0.2360679775;
    nl->sibling_harmonics[3] = 0.3090169944;
    nl->sibling_harmonics[4] = 0.3819660113;
    nl->sibling_harmonics[5] = 0.4721359549;
    nl->sibling_harmonics[6] = 0.6545084972;
    nl->sibling_harmonics[7] = 0.8729833462;

    // Infinity and Choke Layers
    for (int i = 0; i < 4; i++) {
        nl->inf_layer[i] = INFINITY;
        nl->choke_layer[i] = 1.7976931348623157e+308;
    }

    // Base(∞) Seeds - allocate and initialize
    nl->num_seeds = 64;
    nl->base_infinity_seeds = malloc(nl->num_seeds * sizeof(double));

    double seeds[] = {
        0.6180339887, 1.6180339887, 2.6180339887, 3.6180339887, 4.8541019662,
        5.6180339887, 6.4721359549, 7.8541019662, 8.3141592654, 0.0901699437,
        0.1458980338, 0.2360679775, 0.3090169944, 0.3819660113, 0.4721359549,
        0.6545084972, 0.8729833462, 1.0000000000, 1.2360679775, 1.6180339887,
        2.2360679775, 2.6180339887, 3.1415926535, 3.6180339887, 4.2360679775,
        4.8541019662, 5.6180339887, 6.4721359549, 7.2360679775, 7.8541019662,
        8.6180339887, 9.2360679775, 9.8541019662, 10.6180339887, 11.0901699437,
        11.9442719100, 12.6180339887, 13.6180339887, 14.2360679775, 14.8541019662,
        15.6180339887, 16.4721359549, 17.2360679775, 17.9442719100, 18.6180339887,
        19.2360679775, 19.8541019662, 20.6180339887, 21.0901699437, 21.9442719100,
        22.6180339887, 23.6180339887, 24.2360679775, 24.8541019662, 25.6180339887,
        26.4721359549, 27.2360679775, 27.9442719100, 28.6180339887, 29.0344465435,
        29.6180339887, 30.2360679775, 30.8541019662, 31.6180339887
    };

    memcpy(nl->base_infinity_seeds, seeds, nl->num_seeds * sizeof(double));
}

void free_numeric_lattice(NumericLattice *nl) {
    if (nl->base_infinity_seeds) {
        free(nl->base_infinity_seeds);
        nl->base_infinity_seeds = NULL;
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Dₙ(r) Calculation - Core Formula
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

double compute_Dn_r(int n, double r, double omega) {
    if (n < 1 || n > NUM_DN) return 0.0;
    int idx = n - 1;

    // Dₙ(r) = √(ϕ · Fₙ · 2ⁿ · Pₙ · Ω) · r^k
    // where k = (n+1)/8 for progressive dimensionality
    double phi = PHI;
    double F_n = (double)FIB_TABLE[idx];
    double two_n = pow(2.0, n);
    double P_n = (double)PRIME_TABLE[idx];
    double k = (double)(n + 1) / 8.0;

    double base = sqrt(phi * F_n * two_n * P_n * omega);
    double r_power = pow(fabs(r), k);

    return base * r_power;
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// MPI (Multi-Word Integer) Structure - Unchanged
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

typedef struct {
    uint64_t *words;
    size_t num_words;
    uint8_t sign;
} MPI;

#define APA_FLAG_SIGN_NEG (1 << 0)
#define APA_FLAG_IS_NAN   (1 << 1)
#define APA_FLAG_GOI      (1 << 2)
#define APA_FLAG_GUZ      (1 << 3)
#define APA_FLAG_CONSENSUS (1 << 4)

void mpi_init(MPI *m, size_t initial_words) {
    m->words = calloc(initial_words, sizeof(uint64_t));
    m->num_words = initial_words;
    m->sign = 0;
}

void mpi_free(MPI *m) {
    if (m->words) free(m->words);
    m->words = NULL;
    m->num_words = 0;
}

void mpi_copy(MPI *dest, const MPI *src) {
    mpi_free(dest);
    dest->num_words = src->num_words;
    dest->words = malloc(src->num_words * sizeof(uint64_t));
    if (src->words && dest->words) {
        memcpy(dest->words, src->words, src->num_words * sizeof(uint64_t));
    }
    dest->sign = src->sign;
}

void mpi_set_value(MPI *m, uint64_t value, uint8_t sign) {
    if (m->words) m->words[0] = value;
    m->sign = sign;
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Enhanced Slot4096 with Dₙ(r) Integration
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

typedef struct {
    uint64_t *mantissa_words;
    MPI num_words_mantissa;
    MPI exponent_mpi;
    uint16_t exponent_base;
    uint32_t state_flags;
    MPI source_of_infinity;
    size_t num_words;
    int64_t exponent;
    float base;
    int bits_mant;
    int bits_exp;

    // Complex phase state
    double phase;
    double phase_vel;
    double freq;
    double amp_im;

    // NEW: Dₙ(r) state
    int dimension_n;      // Which Dₙ (1-8)
    double r_value;       // Radial position (0-1)
    double Dn_amplitude;  // Current Dₙ(r) value
    double wave_mode;     // -1, 0, +1 for wave phase
} Slot4096;

// Forward declarations
void ap_normalize_legacy(Slot4096 *slot);
void ap_add_legacy(Slot4096 *A, const Slot4096 *B);
void ap_free(Slot4096 *slot);
void ap_copy(Slot4096 *dest, const Slot4096 *src);
double ap_to_double(const Slot4096 *slot);
Slot4096* ap_from_double(double value, int bits_mant, int bits_exp);
void ap_shift_right_legacy(uint64_t *mantissa_words, size_t num_words, int64_t shift_amount);

Slot4096 slot_init_apa_with_Dn(int bits_mant, int bits_exp, int dim_n, double r_val, double omega) {
    Slot4096 slot = {0};
    slot.bits_mant = bits_mant;
    slot.bits_exp = bits_exp;
    slot.num_words = (bits_mant + 63) / 64;
    slot.mantissa_words = calloc(slot.num_words, sizeof(uint64_t));

    mpi_init(&slot.exponent_mpi, 1);
    mpi_init(&slot.num_words_mantissa, 1);
    mpi_init(&slot.source_of_infinity, 1);

    if (!slot.mantissa_words) {
        fprintf(stderr, "Error: Failed to allocate mantissa.\n");
        return slot;
    }

    // Initialize Dₙ(r) state
    slot.dimension_n = dim_n;
    slot.r_value = r_val;
    slot.Dn_amplitude = compute_Dn_r(dim_n, r_val, omega);

    // Wave mode based on dimension
    if (dim_n % 3 == 1) slot.wave_mode = 1.0;
    else if (dim_n % 3 == 2) slot.wave_mode = 0.0;
    else slot.wave_mode = -1.0;

    // Set mantissa based on Dₙ(r)
    if (slot.num_words > 0) {
        slot.mantissa_words[0] = (uint64_t)(fabs(slot.Dn_amplitude) * UINT64_MAX / 1000.0);
        slot.mantissa_words[0] |= MSB_MASK;
    }

    int64_t exp_range = 1LL << bits_exp;
    int64_t exp_bias = 1LL << (bits_exp - 1);
    slot.exponent = (rand() % exp_range) - exp_bias;
    slot.base = PHI + get_normalized_rand() * 0.01;
    slot.exponent_base = 4096;

    // Initialize phase state
    slot.phase = 2.0 * M_PI * get_normalized_rand();
    slot.phase_vel = 0.0;
    slot.freq = 1.0 + 0.5 * get_normalized_rand();
    slot.amp_im = 0.1 * get_normalized_rand();

    mpi_set_value(&slot.exponent_mpi, (uint64_t)llabs(slot.exponent), slot.exponent < 0 ? 1 : 0);
    mpi_set_value(&slot.num_words_mantissa, (uint64_t)slot.num_words, 0);

    return slot;
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Analog Communication with Dₙ(r) coupling
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

typedef struct {
    double charge;
    double charge_im;
    double tension;
    double potential;
    double coupling;
    double Dn_coupling;  // NEW: Dₙ-based coupling strength
} AnalogLink;

void exchange_analog_links(AnalogLink *links, int rank, int size, int num_links) {
#if MPI_REAL
    MPI_BCAST(links, num_links * sizeof(AnalogLink), MPI_BYTE, rank, MPI_COMM_WORLD);
    AnalogLink *reduced = calloc(num_links, sizeof(AnalogLink));
    MPI_REDUCE(links, reduced, num_links * sizeof(AnalogLink), MPI_BYTE, MPI_SUM, 0, MPI_COMM_WORLD);
    for (int i = 0; i < num_links; i++) {
        links[i].charge = reduced[i].charge / size;
        links[i].charge_im = reduced[i].charge_im / size;
        links[i].tension *= 0.9;
        links[i].Dn_coupling *= 0.95;
    }
    free(reduced);
#else
    for (int i = 0; i < num_links; i++) {
        links[i].charge *= 0.95;
        links[i].charge_im *= 0.95;
        links[i].Dn_coupling *= 0.98;
    }
#endif
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// APA Implementation (continued from original)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

void ap_free(Slot4096 *slot) {
    if (slot) {
        if (slot->mantissa_words) {
            free(slot->mantissa_words);
            slot->mantissa_words = NULL;
        }
        mpi_free(&slot->exponent_mpi);
        mpi_free(&slot->num_words_mantissa);
        mpi_free(&slot->source_of_infinity);
        slot->num_words = 0;
    }
}

void ap_copy(Slot4096 *dest, const Slot4096 *src) {
    ap_free(dest);
    memcpy(dest, src, sizeof(Slot4096));
    dest->mantissa_words = malloc(src->num_words * sizeof(uint64_t));
    if (!dest->mantissa_words) {
        fprintf(stderr, "Error: Copy allocation failed.\n");
        dest->num_words = 0;
        return;
    }
    memcpy(dest->mantissa_words, src->mantissa_words, src->num_words * sizeof(uint64_t));
    mpi_copy(&dest->exponent_mpi, &src->exponent_mpi);
    mpi_copy(&dest->num_words_mantissa, &src->num_words_mantissa);
    mpi_copy(&dest->source_of_infinity, &src->source_of_infinity);
}

double ap_to_double(const Slot4096 *slot) {
    if (!slot || slot->num_words == 0 || !slot->mantissa_words) return 0.0;
    double mantissa_double = (double)slot->mantissa_words[0] / (double)UINT64_MAX;
    return mantissa_double * pow(2.0, (double)slot->exponent);
}

Slot4096* ap_from_double(double value, int bits_mant, int bits_exp) {
    Slot4096 temp_slot = slot_init_apa_with_Dn(bits_mant, bits_exp, 1, 0.5, 1.0);
    Slot4096 *slot = malloc(sizeof(Slot4096));
    if (!slot) { ap_free(&temp_slot); return NULL; }
    *slot = temp_slot;
    if (value == 0.0) return slot;
    int exp_offset;
    double mant_val = frexp(value, &exp_offset);
    slot->mantissa_words[0] = (uint64_t)(fabs(mant_val) * (double)UINT64_MAX);
    slot->exponent = (int64_t)exp_offset;
    if (value < 0) slot->state_flags |= APA_FLAG_SIGN_NEG;
    mpi_set_value(&slot->exponent_mpi, (uint64_t)llabs(slot->exponent), slot->exponent < 0 ? 1 : 0);
    return slot;
}

void ap_shift_right_legacy(uint64_t *mantissa_words, size_t num_words, int64_t shift_amount) {
    if (shift_amount <= 0 || num_words == 0) return;
    if (shift_amount >= (int64_t)(num_words * 64)) {
        memset(mantissa_words, 0, num_words * sizeof(uint64_t));
        return;
    }
    int64_t word_shift = shift_amount / 64;
    int bit_shift = (int)(shift_amount % 64);
    if (word_shift > 0) {
        for (int64_t i = num_words - 1; i >= word_shift; i--) {
            mantissa_words[i] = mantissa_words[i - word_shift];
        }
        memset(mantissa_words, 0, word_shift * sizeof(uint64_t));
    }
    if (bit_shift > 0) {
        int reverse_shift = 64 - bit_shift;
        for (size_t i = num_words - 1; i > 0; i--) {
            uint64_t upper_carry = mantissa_words[i - 1] << reverse_shift;
            mantissa_words[i] = (mantissa_words[i] >> bit_shift) | upper_carry;
        }
        mantissa_words[0] >>= bit_shift;
    }
}

void ap_normalize_legacy(Slot4096 *slot) {
    if (slot->num_words == 0) return;
    while (!(slot->mantissa_words[0] & MSB_MASK)) {
        if (slot->exponent <= -(1LL << (slot->bits_exp - 1))) {
            slot->state_flags |= APA_FLAG_GUZ;
            break;
        }
        uint64_t carry = 0;
        for (size_t i = slot->num_words - 1; i != (size_t)-1; i--) {
            uint64_t next_carry = (slot->mantissa_words[i] & MSB_MASK) ? 1 : 0;
            slot->mantissa_words[i] = (slot->mantissa_words[i] << 1) | carry;
            carry = next_carry;
        }
        slot->exponent--;
    }
    if (slot->mantissa_words[0] == 0) slot->exponent = 0;
}

void ap_add_legacy(Slot4096 *A, const Slot4096 *B) {
    if (A->num_words != B->num_words) {
        fprintf(stderr, "Error: Unaligned word counts.\n");
        return;
    }
    Slot4096 B_aligned;
    ap_copy(&B_aligned, B);
    int64_t exp_diff = A->exponent - B_aligned.exponent;
    if (exp_diff > 0) {
        ap_shift_right_legacy(B_aligned.mantissa_words, B_aligned.num_words, exp_diff);
        B_aligned.exponent = A->exponent;
    } else if (exp_diff < 0) {
        ap_shift_right_legacy(A->mantissa_words, A->num_words, -exp_diff);
        A->exponent = B_aligned.exponent;
    }
    uint64_t carry = 0;
    for (size_t i = A->num_words - 1; i != (size_t)-1; i--) {
        uint64_t sum = A->mantissa_words[i] + B_aligned.mantissa_words[i] + carry;
        carry = (sum < A->mantissa_words[i] || (sum == A->mantissa_words[i] && carry)) ? 1 : 0;
        A->mantissa_words[i] = sum;
    }
    if (carry) {
        if (A->exponent >= (1LL << (A->bits_exp - 1))) {
            A->state_flags |= APA_FLAG_GOI;
        } else {
            A->exponent += 1;
            ap_shift_right_legacy(A->mantissa_words, A->num_words, 1);
            A->mantissa_words[0] |= MSB_MASK;
        }
    }
    ap_normalize_legacy(A);
    mpi_set_value(&A->exponent_mpi, (uint64_t)llabs(A->exponent), A->exponent < 0 ? 1 : 0);
    ap_free(&B_aligned);
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Enhanced RK4 Evolution with Dₙ(r) Integration
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

typedef struct {
    double A_re, A_im;
    double phase, phase_vel;
    double Dn_val;
} ComplexState;

ComplexState compute_derivatives_Dn(ComplexState state, double omega, const AnalogLink *neighbors, int num_neigh, int dim_n, double wave_mode) {
    ComplexState deriv = {0};
    double A = sqrt(state.A_re * state.A_re + state.A_im * state.A_im);

    // Amplitude dynamics with Dₙ(r) modulation
    deriv.A_re = -GAMMA * state.A_re + 0.1 * state.Dn_val * cos(state.phase);
    deriv.A_im = -GAMMA * state.A_im + 0.1 * state.Dn_val * sin(state.phase);

    // Phase coupling with wave mode
    double sum_sin = 0.0;
    for (int k = 0; k < num_neigh; k++) {
        double delta_phi = neighbors[k].potential - state.phase;
        sum_sin += sin(delta_phi);

        // Dₙ-modulated coupling
        double Dn_factor = neighbors[k].Dn_coupling / (1.0 + fabs(state.Dn_val));
        deriv.A_re += K_COUPLING * Dn_factor * neighbors[k].charge * cos(delta_phi);
        deriv.A_im += K_COUPLING * Dn_factor * neighbors[k].charge_im * sin(delta_phi);
    }

    // Wave mode influence on phase velocity
    deriv.phase_vel = omega + K_COUPLING * sum_sin + 0.3 * wave_mode;
    deriv.phase = state.phase_vel;

    // Dₙ evolution (slow drift based on amplitude)
    deriv.Dn_val = -0.01 * (state.Dn_val - A);

    return deriv;
}

void rk4_step_Dn(Slot4096 *slot, double t, double dt, const AnalogLink *neighbors, int num_neigh, double omega) {
    ComplexState state = {
        .A_re = ap_to_double(slot),
        .A_im = slot->amp_im,
        .phase = slot->phase,
        .phase_vel = slot->phase_vel,
        .Dn_val = slot->Dn_amplitude
    };

    // RK4 stages
    ComplexState k1 = compute_derivatives_Dn(state, omega, neighbors, num_neigh, slot->dimension_n, slot->wave_mode);

    ComplexState temp = state;
    temp.A_re += dt * k1.A_re / 2.0;
    temp.A_im += dt * k1.A_im / 2.0;
    temp.phase += dt * k1.phase / 2.0;
    temp.phase_vel += dt * k1.phase_vel / 2.0;
    temp.Dn_val += dt * k1.Dn_val / 2.0;
    ComplexState k2 = compute_derivatives_Dn(temp, omega, neighbors, num_neigh, slot->dimension_n, slot->wave_mode);

    temp = state;
    temp.A_re += dt * k2.A_re / 2.0;
    temp.A_im += dt * k2.A_im / 2.0;
    temp.phase += dt * k2.phase / 2.0;
    temp.phase_vel += dt * k2.phase_vel / 2.0;
    temp.Dn_val += dt * k2.Dn_val / 2.0;
    ComplexState k3 = compute_derivatives_Dn(temp, omega, neighbors, num_neigh, slot->dimension_n, slot->wave_mode);

    temp = state;
    temp.A_re += dt * k3.A_re;
    temp.A_im += dt * k3.A_im;
    temp.phase += dt * k3.phase;
    temp.phase_vel += dt * k3.phase_vel;
    temp.Dn_val += dt * k3.Dn_val;
    ComplexState k4 = compute_derivatives_Dn(temp, omega, neighbors, num_neigh, slot->dimension_n, slot->wave_mode);

    // Update state
    state.A_re += dt / 6.0 * (k1.A_re + 2*k2.A_re + 2*k3.A_re + k4.A_re);
    state.A_im += dt / 6.0 * (k1.A_im + 2*k2.A_im + 2*k3.A_im + k4.A_im);
    state.phase += dt / 6.0 * (k1.phase + 2*k2.phase + 2*k3.phase + k4.phase);
    state.phase_vel += dt / 6.0 * (k1.phase_vel + 2*k2.phase_vel + 2*k3.phase_vel + k4.phase_vel);
    state.Dn_val += dt / 6.0 * (k1.Dn_val + 2*k2.Dn_val + 2*k3.Dn_val + k4.Dn_val);

    // Entropy dampers
    double A = sqrt(state.A_re * state.A_re + state.A_im * state.A_im);
    A *= exp(-LAMBDA * dt);
    if (A > SAT_LIMIT) A = SAT_LIMIT;
    A += NOISE_SIGMA * (2.0 * get_normalized_rand() - 1.0);

    // Normalize
    double norm = sqrt(state.A_re * state.A_re + state.A_im * state.A_im);
    if (norm > 1e-10) {
        state.A_re = (state.A_re / norm) * A;
        state.A_im = (state.A_im / norm) * A;
    }

    // Wrap phase
    state.phase = fmod(state.phase, 2.0 * M_PI);
    if (state.phase < 0) state.phase += 2.0 * M_PI;

    // Clamp Dₙ value
    if (state.Dn_val < 0) state.Dn_val = 0;
    if (state.Dn_val > 1000.0) state.Dn_val = 1000.0;

    Slot4096 *new_amp = ap_from_double(state.A_re, slot->bits_mant, slot->bits_exp);
    if (new_amp) {
        ap_copy(slot, new_amp);
        ap_free(new_amp);
        free(new_amp);
    }
    slot->amp_im = state.A_im;
    slot->phase = state.phase;
    slot->phase_vel = state.phase_vel;
    slot->Dn_amplitude = state.Dn_val;
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// HDGL Lattice with Dₙ(r) Integration
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

typedef struct {
    Slot4096 *slots;
    size_t allocated;
} HDGLChunk;

typedef struct {
    HDGLChunk **chunks;
    int num_chunks;
    int num_instances;
    int slots_per_instance;
    double omega;
    double time;
    int consensus_steps;
    double phase_var;
    int64_t last_checkpoint_ns;
    NumericLattice *numeric_lattice;
} HDGLLattice;

HDGLLattice* lattice_init(int num_instances, int slots_per_instance) {
    HDGLLattice *lat = malloc(sizeof(HDGLLattice));
    if (!lat) return NULL;
    lat->num_instances = num_instances;
    lat->slots_per_instance = slots_per_instance;
    lat->omega = 1.0;
    lat->time = 0.0;
    lat->consensus_steps = 0;
    lat->phase_var = 1e6;
    lat->last_checkpoint_ns = get_rtc_ns();

    // Initialize numeric lattice
    lat->numeric_lattice = malloc(sizeof(NumericLattice));
    init_numeric_lattice(lat->numeric_lattice);

    int total_slots = num_instances * slots_per_instance;
    lat->num_chunks = (total_slots + CHUNK_SIZE - 1) / CHUNK_SIZE;
    lat->chunks = calloc(lat->num_chunks, sizeof(HDGLChunk*));
    if (!lat->chunks) {
        free_numeric_lattice(lat->numeric_lattice);
        free(lat->numeric_lattice);
        free(lat);
        return NULL;
    }
    return lat;
}

HDGLChunk* lattice_get_chunk(HDGLLattice *lat, int chunk_idx) {
    if (chunk_idx >= lat->num_chunks) return NULL;
    if (!lat->chunks[chunk_idx]) {
        HDGLChunk *chunk = malloc(sizeof(HDGLChunk));
        if (!chunk) return NULL;
        chunk->allocated = CHUNK_SIZE;
        chunk->slots = malloc(CHUNK_SIZE * sizeof(Slot4096));
        if (!chunk->slots) { free(chunk); return NULL; }
        for (int i = 0; i < CHUNK_SIZE; i++) {
            int bits_mant = 4096 + (i % 8) * 64;
            int bits_exp = 16 + (i % 8) * 2;
            int dim_n = (i % NUM_DN) + 1;
            double r_val = (double)(i % 256) / 256.0;
            chunk->slots[i] = slot_init_apa_with_Dn(bits_mant, bits_exp, dim_n, r_val, lat->omega);
        }
        lat->chunks[chunk_idx] = chunk;
    }
    return lat->chunks[chunk_idx];
}

Slot4096* lattice_get_slot(HDGLLattice *lat, int idx) {
    int chunk_idx = idx / CHUNK_SIZE;
    int local_idx = idx % CHUNK_SIZE;
    HDGLChunk *chunk = lattice_get_chunk(lat, chunk_idx);
    if (!chunk) return NULL;
    return &chunk->slots[local_idx];
}

void detect_harmonic_consensus(HDGLLattice *lat) {
    int total_slots = lat->num_instances * lat->slots_per_instance;
    double sum_var = 0.0, mean_phase = 0.0;
    int count = 0;

    for (int i = 0; i < total_slots; i++) {
        Slot4096 *slot = lattice_get_slot(lat, i);
        if (slot && !(slot->state_flags & APA_FLAG_CONSENSUS)) {
            mean_phase += slot->phase;
            count++;
        }
    }
    if (count == 0) return;
    mean_phase /= count;

    for (int i = 0; i < total_slots; i++) {
        Slot4096 *slot = lattice_get_slot(lat, i);
        if (slot && !(slot->state_flags & APA_FLAG_CONSENSUS)) {
            double diff = slot->phase - mean_phase;
            if (diff > M_PI) diff -= 2.0 * M_PI;
            if (diff < -M_PI) diff += 2.0 * M_PI;
            sum_var += diff * diff;
        }
    }
    lat->phase_var = sqrt(sum_var / count);

    if (lat->phase_var < CONSENSUS_EPS) {
        lat->consensus_steps++;
        if (lat->consensus_steps >= CONSENSUS_N) {
            printf("[CONSENSUS] Domain locked at t=%.4f (var=%.6f)!\n", lat->time, lat->phase_var);
            for (int i = 0; i < total_slots; i++) {
                Slot4096 *slot = lattice_get_slot(lat, i);
                if (slot && !(slot->state_flags & APA_FLAG_CONSENSUS)) {
                    slot->state_flags |= APA_FLAG_CONSENSUS;
                    slot->phase_vel = 0.0;
                }
            }
            lat->consensus_steps = 0;
        }
    } else {
        lat->consensus_steps = 0;
    }
}

void lattice_integrate_rk4(HDGLLattice *lat, double dt_base) {
    int total_slots = lat->num_instances * lat->slots_per_instance;

    for (int i = 0; i < total_slots; i++) {
        Slot4096 *slot = lattice_get_slot(lat, i);
        if (!slot || (slot->state_flags & (APA_FLAG_GOI | APA_FLAG_IS_NAN | APA_FLAG_CONSENSUS))) {
            continue;
        }

        // Build neighbor links with Dₙ coupling
        AnalogLink neighbors[8] = {0};
        int neigh_indices[] = {
            (i - 1 + total_slots) % total_slots,
            (i + 1) % total_slots,
            (i - lat->slots_per_instance + total_slots) % total_slots,
            (i + lat->slots_per_instance) % total_slots,
            (i - lat->slots_per_instance - 1 + total_slots) % total_slots,
            (i - lat->slots_per_instance + 1 + total_slots) % total_slots,
            (i + lat->slots_per_instance - 1 + total_slots) % total_slots,
            (i + lat->slots_per_instance + 1) % total_slots
        };

        for (int j = 0; j < 8; j++) {
            Slot4096 *neigh = lattice_get_slot(lat, neigh_indices[j]);
            if (neigh) {
                neighbors[j].charge = ap_to_double(neigh);
                neighbors[j].charge_im = neigh->amp_im;
                neighbors[j].tension = (ap_to_double(neigh) - ap_to_double(slot)) / dt_base;
                neighbors[j].potential = neigh->phase - slot->phase;

                // Dₙ-based coupling
                double Dn_correlation = fabs(neigh->Dn_amplitude - slot->Dn_amplitude);
                neighbors[j].Dn_coupling = neigh->Dn_amplitude * exp(-Dn_correlation);

                double amp_correlation = fabs(ap_to_double(neigh)) / (fabs(ap_to_double(slot)) + 1e-10);
                neighbors[j].coupling = K_COUPLING * exp(-fabs(1.0 - amp_correlation));
            }
        }

        exchange_analog_links(neighbors, i % lat->num_instances, lat->num_instances, 8);
        rk4_step_Dn(slot, lat->time, dt_base, neighbors, 8, lat->omega);

        // φ-Adaptive time step
        double amp = ap_to_double(slot);
        if (fabs(amp) > ADAPT_THRESH) {
            dt_base *= PHI;
        } else if (fabs(amp) < ADAPT_THRESH / PHI) {
            dt_base /= PHI;
        }

        if (dt_base < 1e-6) dt_base = 1e-6;
        if (dt_base > 0.1) dt_base = 0.1;
    }

    detect_harmonic_consensus(lat);
    lat->omega += 0.01 * dt_base;
    lat->time += dt_base;
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Checkpoint Management
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

typedef struct {
    int evolution;
    int64_t timestamp_ns;
    double phase_var;
    double omega;
    double weight;
} CheckpointMeta;

typedef struct {
    CheckpointMeta *snapshots;
    int count;
    int capacity;
} CheckpointManager;

CheckpointManager* checkpoint_init() {
    CheckpointManager *mgr = malloc(sizeof(CheckpointManager));
    mgr->snapshots = malloc(SNAPSHOT_MAX * sizeof(CheckpointMeta));
    mgr->count = 0;
    mgr->capacity = SNAPSHOT_MAX;
    return mgr;
}

void checkpoint_add(CheckpointManager *mgr, int evo, HDGLLattice *lat) {
    if (mgr->count >= mgr->capacity) {
        int min_idx = 0;
        double min_weight = mgr->snapshots[0].weight;
        for (int i = 1; i < mgr->count; i++) {
            if (mgr->snapshots[i].weight < min_weight) {
                min_weight = mgr->snapshots[i].weight;
                min_idx = i;
            }
        }
        for (int i = min_idx; i < mgr->count - 1; i++) {
            mgr->snapshots[i] = mgr->snapshots[i + 1];
        }
        mgr->count--;
    }

    CheckpointMeta meta = {
        .evolution = evo,
        .timestamp_ns = get_rtc_ns(),
        .phase_var = lat->phase_var,
        .omega = lat->omega,
        .weight = 1.0
    };
    mgr->snapshots[mgr->count++] = meta;

    for (int i = 0; i < mgr->count - 1; i++) {
        mgr->snapshots[i].weight *= SNAPSHOT_DECAY;
    }

    printf("[Checkpoint] Saved evo %d (total: %d, var=%.6f)\n", evo, mgr->count, lat->phase_var);
}

void checkpoint_free(CheckpointManager *mgr) {
    if (mgr) {
        free(mgr->snapshots);
        free(mgr);
    }
}

void lattice_free(HDGLLattice *lat) {
    if (!lat) return;
    for (int i = 0; i < lat->num_chunks; i++) {
        if (lat->chunks[i]) {
            for (size_t j = 0; j < CHUNK_SIZE; j++) {
                ap_free(&lat->chunks[i]->slots[j]);
            }
            free(lat->chunks[i]->slots);
            free(lat->chunks[i]);
        }
    }
    free(lat->chunks);
    if (lat->numeric_lattice) {
        free_numeric_lattice(lat->numeric_lattice);
        free(lat->numeric_lattice);
    }
    free(lat);
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Bootloader
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

void bootloader_init_lattice(HDGLLattice *lat, int steps, CheckpointManager *ckpt_mgr) {
    printf("[Bootloader] Initializing HDGL Analog Mainnet V3.0 with Dₙ(r) Engine...\n");
    if (!lat) {
        printf("[Bootloader] ERROR: Lattice allocation failed.\n");
        return;
    }

    printf("[Bootloader] %d instances, %d total slots\n", lat->num_instances, lat->num_instances * lat->slots_per_instance);
    printf("[Bootloader] Numeric Lattice loaded with %zu Base(∞) seeds\n", lat->numeric_lattice->num_seeds);

    double dt = 1.0 / 32768.0;
    int64_t step_ns = 30517;
    int64_t next_step_ns = get_rtc_ns() + step_ns;

    for (int i = 0; i < steps; i++) {
        lattice_integrate_rk4(lat, dt);

        if (i % CHECKPOINT_INTERVAL == 0 && i > 0) {
            checkpoint_add(ckpt_mgr, i, lat);
        }

        rtc_sleep_until(next_step_ns);
        next_step_ns += step_ns;
    }

    printf("[Bootloader] Lattice seeded with %d RK4 steps\n", steps);
    printf("[Bootloader] Omega: %.6f, Time: %.6f, PhaseVar: %.6f\n", lat->omega, lat->time, lat->phase_var);
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Main
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

int main(int argc, char *argv[]) {
    srand(time(NULL));

#ifdef USE_DS3231
    i2c_fd = i2c_open("/dev/i2c-1");
    if (i2c_fd >= 0) {
        i2c_smbus_write_byte_data(i2c_fd, DS3231_ADDR, 0x0E, 0x00);
        printf("[RTC] DS3231 initialized on I2C-1\n");
    } else {
        printf("[RTC] Using software fallback (CLOCK_MONOTONIC)\n");
    }
#else
    printf("[RTC] Using software fallback (CLOCK_MONOTONIC)\n");
#endif

    printf("=== HDGL Analog Mainnet V3.0: Dₙ(r) Engine Ready ===\n\n");

    HDGLLattice *lat = lattice_init(4096, 4);
    if (!lat) {
        fprintf(stderr, "Fatal: Could not initialize lattice.\n");
        return 1;
    }

    CheckpointManager *ckpt_mgr = checkpoint_init();
    bootloader_init_lattice(lat, 500, ckpt_mgr);

    printf("\nNumeric Lattice Summary:\n");
    printf("  Upper Field[0]: %.10f\n", lat->numeric_lattice->upper_field[0]);
    printf("  Analog D₈: %.10f\n", lat->numeric_lattice->analog_dims[0]);
    printf("  The Void: %.10f\n", lat->numeric_lattice->void_state);
    printf("  Lower Field[7]: %.10f\n", lat->numeric_lattice->lower_field[7]);
    printf("  Base(∞) Seeds: %zu total\n", lat->numeric_lattice->num_seeds);

    printf("\nFirst 8 slots (post-evolution with Dₙ(r)):\n");
    for (int i = 0; i < 8; i++) {
        Slot4096 *slot = lattice_get_slot(lat, i);
        if (slot) {
            double amp = sqrt(pow(ap_to_double(slot), 2) + pow(slot->amp_im, 2));
            printf("  D%d: |A|=%.6e φ=%.3f Dₙ=%.3f wave=%.1f r=%.3f\n",
                   i+1, amp, slot->phase, slot->Dn_amplitude, slot->wave_mode, slot->r_value);
        }
    }

    checkpoint_free(ckpt_mgr);
    lattice_free(lat);

#ifdef USE_DS3231
    if (i2c_fd >= 0) i2c_close(i2c_fd);
#endif

    printf("\n=== HDGL V3.0 OPERATIONAL ===\n");
    return 0;
}