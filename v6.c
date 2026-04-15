#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>

// --- System Constants ---
#define PHI 1.6180339887
#define MAX_INSTANCES 8388608
#define SLOTS_PER_INSTANCE 4
#define MAX_SLOTS (MAX_INSTANCES * SLOTS_PER_INSTANCE)
#define CHUNK_SIZE 1048576 // 1M slots per chunk
#define MSB_MASK (1ULL << 63) // Mask for the Most Significant Bit of a uint64_t

// --- Future MPI and Base-4096 Constants (V3 ADDITIONS) ---
#define BASE_4096_BPC 12        // Bits per Character (4096 = 2^12)
#define MPI_INITIAL_WORDS 1     // Initial allocation size for MPI structures
#define MPI_ZERO_WORD 0ULL      // Canonical zero for MPI operations

// MPI Bit Extraction Helpers
#define MPI_GET_MSB(word) ((word) >> 63)

// Fibonacci and prime tables
static const float fib_table[] = {1,1,2,3,5,8,13,21,34,55,89,144,233,377,610,987};
static const float prime_table[] = {2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53};
static const int fib_len = 16;
static const int prime_len = 16;

// Helper for generating normalized random double
double get_normalized_rand() {
    return (double)rand() / RAND_MAX;
}

// Macro for generating 64-bit random seed (Placeholder for get_random_bytes in kernel)
#define GET_RANDOM_UINT64() (((uint64_t)rand() << 32) | rand())

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// FUTURE-PROOF MPI (Multi-Word Integer) and State Flags
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

// Multi-Word Integer (MPI) Structure for arbitrarily large metadata (Exponent, Word Count)
typedef struct {
    uint64_t *words;        // Array of 64-bit words for the number
    size_t num_words;       // Number of words currently allocated
    uint8_t sign;           // 0: Positive, 1: Negative (Magnitude is always stored in words)
} MPI;

// State Flags Definitions for Implied Precision
#define APA_FLAG_SIGN_NEG           (1 << 0) // Mantissa is negative
#define APA_FLAG_IS_NAN             (1 << 1) // Not a Number
#define APA_FLAG_GOI                (1 << 2) // Gradual Overflow Infinity
#define APA_FLAG_GUZ                (1 << 3) // Gradual Underflow Zero


// --- V6.0 Addition: Decoupled Communication State ---
// This structure holds state specific to asynchronous communication and remote exponent synchronization.
typedef struct {
    MPI remote_exponent_delta; // Change derived from remote synchronization (V6.0)
    uint8_t status_flags;      // Flags for async operations (e.g., AWAITING_SYNC, SYNC_READY) (V6.0)
    int comm_handle_id;        // Unique ID for tracking async tasks (Placeholder for MPI_Request)
} CommLayerState;

// CommLayer Status Flags (Simulating MPI non-blocking state)
#define COMM_FLAG_AWAITING_SYNC (1 << 0) // Exponent needs external sync/wait
#define COMM_FLAG_SYNC_READY    (1 << 1) // Remote delta has arrived and is ready to apply
#define COMM_FLAG_RMA_PENDING   (1 << 2) // Remote Memory Access is pending completion

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Slot4096: Arbitrary Precision Architecture (APA) Structure (FUTURE-PROOF)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

typedef struct {
    // --- Core Precision Fields ---
    uint64_t *mantissa_words;   // Multi-word array for high-precision mantissa

    // FUTURE-PROOF FIELDS (Conceptual, for 4096^4096+ scaling)
    MPI num_words_mantissa;     // The COUNT of mantissa_words (Arbitrarily wide, V3)
    MPI exponent_mpi;           // The exponent value (Arbitrarily wide, V3)

    // V6.0: Exponent Decoupling Structure
    CommLayerState comm_state;  // Dedicated state for asynchronous communication

    // State and Base Control Fields
    uint16_t exponent_base;     // Base of the floating-point system (e.g., 2 or 4096)
    uint32_t state_flags;       // Flags for NaN, Sign, GOI, GUZ
    MPI source_of_infinity;     // Records magnitude for Gradual Overflow Infinity (GOI)

    // LEGACY FIELDS (Kept for compatibility and temporary transition)
    size_t num_words;           // Legacy Count of 64-bit words allocated
    int64_t exponent;           // Legacy Signed exponent (Base 2)
    float base;                 // Dynamic base (φ-scaled)
    int bits_mant;              // Actual software-managed bit width (e.g., 4096)
    int bits_exp;               // Exponent bit width

} Slot4096;

// -----------------------------------------------------------------------------
// GLOBAL HIGH-PRECISION CONSTANT SLOTS
// -----------------------------------------------------------------------------
static Slot4096 APA_CONST_PHI;  // Target slot for full precision Golden Ratio
static Slot4096 APA_CONST_PI;   // Target slot for full precision Pi


// Forward Declarations for APA Operations
void ap_normalize(Slot4096 *slot);
void ap_free(Slot4096 *slot);
void ap_copy(Slot4096 *dest, const Slot4096 *src);
double ap_to_double(const Slot4096 *slot);
Slot4096* ap_from_double(double value, int bits_mant, int bits_exp);
void ap_shift_right(uint64_t *mantissa_words, size_t num_words, int64_t shift_amount);

// --- Mantissa Magnitude Operations (V4) ---
int apa_mantissa_compare_magnitude(const uint64_t *A, const uint64_t *B, size_t num_words);
uint64_t apa_mantissa_add_magnitude(uint64_t *A, const uint64_t *B, size_t num_words);
void apa_mantissa_subtract_magnitude(uint64_t *A, const uint64_t *B, size_t num_words);
void ap_add(Slot4096 *A, const Slot4096 *B); // Full sign-magnitude V5 implementation

// --- MPI Core Functions (V5 Additions) ---
int mpi_compare_signed(const MPI *A, const MPI *B);
void mpi_update_from_legacy(MPI *m, int64_t value);
void mpi_init(MPI *m, size_t initial_words);
void mpi_free(MPI *m);
void mpi_copy(MPI *dest, const MPI *src);
void mpi_resize(MPI *m, size_t new_words);
void mpi_set_value(MPI *m, uint64_t value, uint8_t sign);
int mpi_compare(const MPI *A, const MPI *B);
void mpi_add(MPI *A, const MPI *B);
void mpi_subtract(MPI *A, const MPI *B);
size_t mpi_get_effective_words(const MPI *m);

// --- V6.0 Comm Layer Functions ---
void comm_layer_init(CommLayerState *c);
void comm_layer_free(CommLayerState *c);
void comm_layer_poll_sync(Slot4096 *slot); // V6.0: The actual decoupling logic


// -----------------------------------------------------------------------------
// MPI FUNCTION IMPLEMENTATIONS (V5 Functional Core)
// -----------------------------------------------------------------------------

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
    dest->words = (uint64_t*)malloc(src->num_words * sizeof(uint64_t));
    if (src->words && dest->words) {
        memcpy(dest->words, src->words, src->num_words * sizeof(uint64_t));
    }
    dest->sign = src->sign;
}
void mpi_resize(MPI *m, size_t new_words) {
    if (new_words == m->num_words) return;
    if (new_words == 0) { mpi_free(m); return; }

    size_t old_words = m->num_words;
    m->words = (uint64_t*)realloc(m->words, new_words * sizeof(uint64_t));

    if (new_words > old_words) {
        memset(m->words + old_words, 0, (new_words - old_words) * sizeof(uint64_t));
    }
    m->num_words = new_words;
}
// Helper to update MPI struct from legacy int64_t value (V5: Exponent Sync)
void mpi_update_from_legacy(MPI *m, int64_t value) {
    uint8_t sign = 0;
    uint64_t magnitude = (uint64_t)value;

    if (value < 0) {
        sign = 1;
        magnitude = (uint64_t)llabs(value);
    }

    // Ensure we have at least 1 word
    if (m->num_words < 1) mpi_resize(m, 1);

    m->words[0] = magnitude;
    m->sign = sign;

    // Zero out higher words if necessary
    if (m->num_words > 1) {
        memset(m->words + 1, 0, (m->num_words - 1) * sizeof(uint64_t));
    }
}

void mpi_set_value(MPI *m, uint64_t value, uint8_t sign) {
    if (m->num_words < 1) mpi_resize(m, 1);
    m->words[0] = value;
    m->sign = sign;
    if (m->num_words > 1) {
        memset(m->words + 1, 0, (m->num_words - 1) * sizeof(uint64_t));
    }
}
size_t mpi_get_effective_words(const MPI *m) {
    if (!m->words) return 0;
    size_t i = m->num_words;
    while (i > 0 && m->words[i - 1] == 0) {
        i--;
    }
    return i;
}

// Magnitude comparison: |A| vs |B|
int mpi_compare(const MPI *A, const MPI *B) {
    size_t len_A = mpi_get_effective_words(A);
    size_t len_B = mpi_get_effective_words(B);

    if (len_A > len_B) return 1;
    if (len_A < len_B) return -1;

    for (size_t i = len_A; i-- > 0; ) {
        if (A->words[i] > B->words[i]) return 1;
        if (A->words[i] < B->words[i]) return -1;
    }

    return 0;
}

// Signed comparison (V5 New): A vs B
int mpi_compare_signed(const MPI *A, const MPI *B) {
    if (A->sign != B->sign) {
        // POS (sign 0) is greater than NEG (sign 1)
        return A->sign == 0 ? 1 : -1;
    }
    // Signs are the same. Compare magnitudes.
    int mag_cmp = mpi_compare(A, B);
    if (A->sign == 0) { // Both positive
        return mag_cmp;
    } else { // Both negative: larger magnitude means smaller (more negative) number
        return -mag_cmp;
    }
}

void mpi_add(MPI *A, const MPI *B) {
    size_t len_A = mpi_get_effective_words(A);
    size_t len_B = mpi_get_effective_words(B);
    size_t max_len = len_A > len_B ? len_A : len_B;

    if (A->num_words < max_len + 1) {
        mpi_resize(A, max_len + 1);
    }

    uint64_t carry = 0;

    for (size_t i = 0; i < max_len; ++i) {
        uint64_t word_A = (i < len_A) ? A->words[i] : 0ULL;
        uint64_t word_B = (i < len_B) ? B->words[i] : 0ULL;

        uint64_t sum = word_A + word_B;
        uint64_t carry1 = (sum < word_A);
        sum += carry;
        uint64_t carry2 = (sum < carry);

        A->words[i] = sum;
        carry = carry1 | carry2;
    }

    if (carry) {
        A->words[max_len] = carry;
    } else if (A->num_words > max_len + 1) {
        memset(A->words + max_len, 0, (A->num_words - max_len) * sizeof(uint64_t));
    }
}
void mpi_subtract(MPI *A, const MPI *B) {
    size_t len_A = mpi_get_effective_words(A);
    size_t len_B = mpi_get_effective_words(B);

    if (len_A < len_B) { return; }

    uint64_t borrow = 0;

    for (size_t i = 0; i < len_A; ++i) {
        uint64_t word_B = (i < len_B) ? B->words[i] : 0ULL;

        uint64_t diff = A->words[i] - word_B - borrow;
        borrow = (diff > A->words[i]) || (A->words[i] == diff && (word_B | borrow));

        A->words[i] = diff;
    }

    size_t new_len = mpi_get_effective_words(A);
    if (new_len < A->num_words) {
        // Shrink the array to canonical size (or 1 if it's zero)
        mpi_resize(A, new_len > 0 ? new_len : 1);
    }

    if (new_len == 0) mpi_set_value(A, 0ULL, 0); // Canonical zero
}


// -----------------------------------------------------------------------------
// V6.0: COMMUNICATION LAYER IMPLEMENTATIONS (Exponent Decoupling)
// -----------------------------------------------------------------------------

void comm_layer_init(CommLayerState *c) {
    // Initialize the multi-word delta to zero
    mpi_init(&c->remote_exponent_delta, MPI_INITIAL_WORDS);
    c->status_flags = 0;
    c->comm_handle_id = 0; // Start with a clean handle
}

void comm_layer_free(CommLayerState *c) {
    mpi_free(&c->remote_exponent_delta);
}

/**
 * V6.0 Decoupling Check: Polls the communication layer for remote updates.
 * This simulates the non-blocking check for MPI_Isend/MPI_Irecv completion,
 * ensuring the core arithmetic logic is decoupled from communication latency.
 */
void comm_layer_poll_sync(Slot4096 *slot) {
    if (slot->comm_state.status_flags & COMM_FLAG_AWAITING_SYNC) {
        // Simulate completion (50% chance per poll)
        if (get_normalized_rand() < 0.5) {
            slot->comm_state.status_flags &= ~COMM_FLAG_AWAITING_SYNC;
            slot->comm_state.status_flags |= COMM_FLAG_SYNC_READY;

            // --- Simulate Delta Update from a Remote Rank ---
            // In a real system, this delta comes from the MPI layer.
            int delta_val = (slot->comm_state.comm_handle_id % 2 == 0) ? 1 : -1;
            MPI delta_mpi;
            mpi_init(&delta_mpi, 1);
            mpi_update_from_legacy(&delta_mpi, (int64_t)delta_val);

            // V6.0: Apply the delta to the core APA fields.
            slot->exponent += delta_val; // Update legacy exponent

            // Update the authoritative MPI exponent based on the delta sign
            if (delta_val > 0) mpi_add(&slot->exponent_mpi, &delta_mpi);
            else mpi_subtract(&slot->exponent_mpi, &delta_mpi);

            mpi_free(&delta_mpi);

            // Clear the ready flag after application
            slot->comm_state.status_flags &= ~COMM_FLAG_SYNC_READY;

            // Uncomment for verbose logging to demonstrate decoupling
            // printf("[CommLayer] Slot %d synchronized (Delta: %d).\n", slot->comm_state.comm_handle_id, delta_val);
        }
    }
}


// -----------------------------------------------------------------------------
// APA UTILITY FUNCTIONS
// -----------------------------------------------------------------------------

// Fixed-width multi-word magnitude comparison
int apa_mantissa_compare_magnitude(const uint64_t *A, const uint64_t *B, size_t num_words) {
    for (size_t i = num_words; i-- > 0; ) {
        if (A[i] > B[i]) return 1;
        if (A[i] < B[i]) return -1;
    }
    return 0; // Equal
}

// Fixed-width multi-word magnitude addition
uint64_t apa_mantissa_add_magnitude(uint64_t *A, const uint64_t *B, size_t num_words) {
    uint64_t carry = 0;

    for (size_t i = num_words; i-- > 0; ) {
        uint64_t sum = A[i] + B[i];
        uint64_t carry1 = (sum < A[i]);

        sum += carry;
        uint64_t carry2 = (sum < carry);

        A[i] = sum;
        carry = carry1 | carry2;
    }
    return carry;
}

// Fixed-width multi-word magnitude subtraction (A = A - B). Assumes |A| >= |B|.
void apa_mantissa_subtract_magnitude(uint64_t *A, const uint64_t *B, size_t num_words) {
    uint64_t borrow = 0;

    for (size_t i = num_words; i-- > 0; ) {
        uint64_t diff = A[i] - B[i] - borrow;
        borrow = (diff > A[i]) || (A[i] == diff && B[i] != 0 && borrow);

        A[i] = diff;
    }
}

// Initialize slot with dynamic precision and APA allocation
Slot4096 slot_init_apa(int bits_mant, int bits_exp) {
    Slot4096 slot = {0};
    slot.bits_mant = bits_mant;
    slot.bits_exp = bits_exp;

    slot.num_words = (bits_mant + 63) / 64;
    slot.mantissa_words = (uint64_t*)calloc(slot.num_words, sizeof(uint64_t));

    // FUTURE-PROOF MPI INITIALIZATION (V3 Functional)
    mpi_init(&slot.exponent_mpi, MPI_INITIAL_WORDS);
    mpi_init(&slot.num_words_mantissa, MPI_INITIAL_WORDS);
    mpi_init(&slot.source_of_infinity, MPI_INITIAL_WORDS);

    // V6.0: Comm Layer Initialization
    comm_layer_init(&slot.comm_state);
    slot.comm_state.comm_handle_id = rand() % 1000; // Assign a random handle ID
    slot.comm_state.status_flags |= COMM_FLAG_AWAITING_SYNC; // Simulate remote dependency

    if (!slot.mantissa_words) {
        fprintf(stderr, "Error: Failed to allocate multi-word mantissa.\n");
        return slot;
    }

    if (slot.num_words > 0) {
        slot.mantissa_words[0] = GET_RANDOM_UINT64();
        slot.mantissa_words[0] |= MSB_MASK;
    }

    int64_t exp_range = 1LL << bits_exp;
    int64_t exp_bias = 1LL << (bits_exp - 1);
    slot.exponent = (rand() % exp_range) - exp_bias;
    slot.base = PHI + get_normalized_rand() * 0.01;

    // Set the target base for future MPI operations
    slot.exponent_base = 4096;

    // Synchronize Legacy with MPI (V5: Use new helper)
    mpi_update_from_legacy(&slot.exponent_mpi, slot.exponent);
    mpi_set_value(&slot.num_words_mantissa, (uint64_t)slot.num_words, 0);

    return slot;
}

// Helper to free single APA slot's dynamic members
void ap_free(Slot4096 *slot) {
    if (slot) {
        if (slot->mantissa_words) {
            free(slot->mantissa_words);
            slot->mantissa_words = NULL;
        }
        // FUTURE-PROOF MPI CLEANUP
        mpi_free(&slot->exponent_mpi);
        mpi_free(&slot->num_words_mantissa);
        mpi_free(&slot->source_of_infinity);
        // V6.0: Comm Layer Cleanup
        comm_layer_free(&slot->comm_state);
        slot->num_words = 0;
    }
}

// Deep copy of APA slot
void ap_copy(Slot4096 *dest, const Slot4096 *src) {
    ap_free(dest);

    *dest = *src; // Shallow copy of struct members

    // Deep copy mantissa
    dest->mantissa_words = (uint64_t*)malloc(src->num_words * sizeof(uint64_t));
    if (!dest->mantissa_words) {
        fprintf(stderr, "Error: Failed deep copy allocation.\n");
        dest->num_words = 0;
        return;
    }
    memcpy(dest->mantissa_words, src->mantissa_words, src->num_words * sizeof(uint64_t));

    // Deep copy MPI structs
    mpi_copy(&dest->exponent_mpi, &src->exponent_mpi);
    mpi_copy(&dest->num_words_mantissa, &src->num_words_mantissa);
    mpi_copy(&dest->source_of_infinity, &src->source_of_infinity);

    // V6.0: Deep copy Comm Layer MPI struct
    mpi_copy(&dest->comm_state.remote_exponent_delta, &src->comm_state.remote_exponent_delta);
}

// Converts double to APA slot
Slot4096* ap_from_double(double value, int bits_mant, int bits_exp) {
    Slot4096 temp_slot = slot_init_apa(bits_mant, bits_exp);
    Slot4096 *slot = (Slot4096*)malloc(sizeof(Slot4096));
    if (!slot) { ap_free(&temp_slot); return NULL; }
    *slot = temp_slot;

    // Clear communication flags for value initialization, as it is a local conversion
    slot->comm_state.status_flags = 0;

    if (value == 0.0) return slot;

    if (value < 0) {
        slot->state_flags |= APA_FLAG_SIGN_NEG;
        value = -value;
    } else {
        slot->state_flags &= ~APA_FLAG_SIGN_NEG;
    }

    int exp_offset;
    double mant_val = frexp(value, &exp_offset);

    slot->mantissa_words[0] = (uint64_t)(mant_val * (double)UINT64_MAX);
    slot->exponent = (int64_t)exp_offset;

    // Synchronize Legacy with MPI (V5: Use new helper)
    mpi_update_from_legacy(&slot->exponent_mpi, slot->exponent);

    return slot;
}

/**
 * Multi-word right shift (V4 Logic)
 */
void ap_shift_right(uint64_t *mantissa_words, size_t num_words, int64_t shift_amount) {
    if (shift_amount <= 0 || num_words == 0) return;

    if (shift_amount >= (int64_t)num_words * 64) {
        memset(mantissa_words, 0, num_words * sizeof(uint64_t));
        return;
    }

    int64_t word_shift = shift_amount / 64;
    int bit_shift = (int)(shift_amount % 64);

    if (word_shift > 0) {
        // Shift words right by word_shift positions
        memmove(mantissa_words, mantissa_words + word_shift,
                (num_words - word_shift) * sizeof(uint64_t));
        // Zero out the highest 'word_shift' words
        memset(mantissa_words + num_words - word_shift, 0, word_shift * sizeof(uint64_t));
    }

    if (bit_shift > 0) {
        int reverse_shift = 64 - bit_shift;

        // Shift bits right, propagating carry (upper bits) down
        for (size_t i = num_words; i-- > 0; ) {
            uint64_t upper_carry = 0;
            if (i > 0) {
                // Carry from the word below (higher index) in memory, which is higher magnitude
                upper_carry = mantissa_words[i - 1] << reverse_shift;
            }
            mantissa_words[i] = (mantissa_words[i] >> bit_shift) | upper_carry;
        }
    }
}


// -----------------------------------------------------------------------------
// APA CORE ARITHMETIC FUNCTIONS (V6.0: Decoupling Implementation)
// -----------------------------------------------------------------------------

// Performs multi-word normalization to maintain canonical range [0.5, 1.0)
void ap_normalize(Slot4096 *slot) {
    if (slot->num_words == 0) return;

    // --- Check for Zero Mantissa ---
    int is_zero = 1;
    for (size_t i = 0; i < slot->num_words; i++) {
        if (slot->mantissa_words[i] != 0) {
            is_zero = 0;
            break;
        }
    }
    if (is_zero) {
        slot->exponent = 0;
        slot->state_flags &= ~APA_FLAG_SIGN_NEG; // Canonical zero is positive
        mpi_update_from_legacy(&slot->exponent_mpi, slot->exponent); // V5: Sync MPI
        return;
    }

    // --- Shift Left (Underflow Correction: Mantissa < 0.5) ---
    while (!(slot->mantissa_words[0] & MSB_MASK)) {
        if (slot->exponent <= -(1LL << (slot->bits_exp - 1))) {
            slot->state_flags |= APA_FLAG_GUZ;
            break;
        }

        uint64_t carry = 0;
        for (size_t i = slot->num_words; i-- > 0; ) {
            uint64_t next_carry = (slot->mantissa_words[i] & MSB_MASK) ? 1 : 0;
            slot->mantissa_words[i] = (slot->mantissa_words[i] << 1) | carry;
            carry = next_carry;
        }
        slot->exponent--;
    }

    // V5: Synchronize MPI after all shifting is complete
    mpi_update_from_legacy(&slot->exponent_mpi, slot->exponent);
}


// Full APA addition with sign-magnitude handling (V6.0)
void ap_add(Slot4096 *A, const Slot4096 *B) {
    if (A->num_words != B->num_words) {
        fprintf(stderr, "Error: APA addition failed due to unaligned word counts.\n");
        return;
    }

    Slot4096 B_aligned;
    ap_copy(&B_aligned, B);

    // --- V6.0: Communication Layer Synchronization Point ---
    // Enforce decoupling: If the exponent is dependent on a remote operation, poll for completion
    comm_layer_poll_sync(A);
    comm_layer_poll_sync(&B_aligned);
    // After this point, the core logic relies on the now-synchronized MPI exponents.

    // --- 1. Exponent Alignment (V5/V6.0: Use MPI comparison for decision) ---

    // Determine which MPI exponent is larger (signed comparison)
    int exp_cmp = mpi_compare_signed(&A->exponent_mpi, &B_aligned.exponent_mpi);
    int64_t exp_diff = A->exponent - B_aligned.exponent; // Use legacy field for the shift amount calculation

    if (exp_cmp > 0) { // A.exp > B.exp (A is larger/more positive)
        // Shift B right. Shift amount is the difference (A.exp - B.exp).
        int64_t shift_amount = exp_diff;
        ap_shift_right(B_aligned.mantissa_words, B_aligned.num_words, shift_amount);

        // Update B_aligned's exponent to match A's
        B_aligned.exponent = A->exponent;
        mpi_copy(&B_aligned.exponent_mpi, &A->exponent_mpi); // V5: Sync B's MPI

    } else if (exp_cmp < 0) { // B.exp > A.exp (B is larger/more positive)
        // Shift A right. Shift amount is the difference (B.exp - A.exp).
        int64_t shift_amount = -exp_diff;
        ap_shift_right(A->mantissa_words, A->num_words, shift_amount);

        // Update A's exponent to match B's
        A->exponent = B_aligned.exponent;
        mpi_copy(&A->exponent_mpi, &B_aligned.exponent_mpi); // V5: Sync A's MPI
    }
    // If exp_cmp == 0, exponents are already aligned.


    // --- 2. Sign-Magnitude Decision (V4 Core Logic) ---
    int sign_A = (A->state_flags & APA_FLAG_SIGN_NEG) ? 1 : 0;
    int sign_B = (B_aligned.state_flags & APA_FLAG_SIGN_NEG) ? 1 : 0;
    uint64_t final_carry = 0;

    if (sign_A == sign_B) {
        // Case 1: Same Signs -> Add Magnitudes
        final_carry = apa_mantissa_add_magnitude(A->mantissa_words, B_aligned.mantissa_words, A->num_words);

    } else {
        // Case 2: Different Signs -> Subtract Magnitudes
        int cmp_result = apa_mantissa_compare_magnitude(A->mantissa_words, B_aligned.mantissa_words, A->num_words);

        if (cmp_result >= 0) {
            // |A| >= |B|. Result sign is A's sign. (A = A - B)
            apa_mantissa_subtract_magnitude(A->mantissa_words, B_aligned.mantissa_words, A->num_words);

        } else {
            // |A| < |B|. Result sign is B's sign. (A = B - A)
            uint64_t *temp_words = (uint64_t*)malloc(A->num_words * sizeof(uint64_t));
            memcpy(temp_words, B_aligned.mantissa_words, A->num_words * sizeof(uint64_t));

            apa_mantissa_subtract_magnitude(temp_words, A->mantissa_words, A->num_words);

            memcpy(A->mantissa_words, temp_words, A->num_words * sizeof(uint64_t));
            free(temp_words);

            // Set A's sign to B's sign
            if (sign_B) A->state_flags |= APA_FLAG_SIGN_NEG;
            else A->state_flags &= ~APA_FLAG_SIGN_NEG;
        }
    }

    // --- 3. Final Carry Handling (Only for magnitude addition) ---
    if (final_carry) {
        // Overflow due to addition (result >= 2.0 * 2^exp). Renormalize.
        if (A->exponent >= (1LL << (A->bits_exp - 1))) {
            A->state_flags |= APA_FLAG_GOI;
        } else {
            A->exponent += 1; // Increment legacy exponent

            // Right shift all words by 1 bit, propagating carry down
            uint64_t next_carry = final_carry << 63;
            for (size_t i = 0; i < A->num_words; i++) {
                uint64_t current_carry_bit = A->mantissa_words[i] & 1;
                A->mantissa_words[i] = (A->mantissa_words[i] >> 1) | next_carry;
                next_carry = current_carry_bit << 63;
            }
        }
        // V5: Sync A's MPI exponent after carry update
        mpi_update_from_legacy(&A->exponent_mpi, A->exponent);
    }

    // --- 4. Final Normalization and Sync ---
    ap_normalize(A);
    ap_free(&B_aligned);
}


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// HDGL Lattice (Unchanged from V4/V5)
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
} HDGLLattice;

// Initialize lattice
HDGLLattice* lattice_init(int num_instances, int slots_per_instance) {
    HDGLLattice *lat = malloc(sizeof(HDGLLattice));
    if (!lat) return NULL;

    lat->num_instances = num_instances;
    lat->slots_per_instance = slots_per_instance;
    lat->omega = 0.0;
    lat->time = 0.0;

    int total_slots = num_instances * slots_per_instance;
    lat->num_chunks = (total_slots + CHUNK_SIZE - 1) / CHUNK_SIZE;
    lat->chunks = calloc(lat->num_chunks, sizeof(HDGLChunk*));
    if (!lat->chunks) { free(lat); return NULL; }

    return lat;
}

// Get chunk, allocate if needed
HDGLChunk* lattice_get_chunk(HDGLLattice *lat, int chunk_idx) {
    if (chunk_idx >= lat->num_chunks) return NULL;

    if (!lat->chunks[chunk_idx]) {
        HDGLChunk *chunk = malloc(sizeof(HDGLChunk));
        if (!chunk) return NULL;

        chunk->allocated = CHUNK_SIZE;
        chunk->slots = (Slot4096*)malloc(CHUNK_SIZE * sizeof(Slot4096));
        if (!chunk->slots) { free(chunk); return NULL; }

        for (int i = 0; i < CHUNK_SIZE; i++) {
            // Dynamic precision allocation: 4096 bits minimum, up to 4576 bits
            int bits_mant = 4096 + (i % 8) * 64;
            int bits_exp = 16 + (i % 8) * 2;
            chunk->slots[i] = slot_init_apa(bits_mant, bits_exp);
        }
        lat->chunks[chunk_idx] = chunk;
    }

    return lat->chunks[chunk_idx];
}

// Get slot pointer
Slot4096* lattice_get_slot(HDGLLattice *lat, int idx) {
    int chunk_idx = idx / CHUNK_SIZE;
    int local_idx = idx % CHUNK_SIZE;
    HDGLChunk *chunk = lattice_get_chunk(lat, chunk_idx);
    if (!chunk) return NULL;
    return &chunk->slots[local_idx];
}

// Prismatic recursion function (MUST use lossy double for calculation input)
double prismatic_recursion(HDGLLattice *lat, int idx, double val) {
    double phi_harm = pow(PHI, (double)(idx % 16));
    double fib_harm = fib_table[idx % fib_len];
    double dyadic = (double)(1 << (idx % 16));
    double prime_harm = prime_table[idx % prime_len];
    double omega_val = 0.5 + 0.5 * sin(lat->time + idx * 0.01);
    double r_dim = pow(val, (double)((idx % 7) + 1));

    return sqrt(phi_harm * fib_harm * dyadic * prime_harm * omega_val) * r_dim;
}

// Converts APA to double (Lossy, for display/recursion input)
double ap_to_double(const Slot4096 *slot) {
    if (!slot || slot->num_words == 0 || !slot->mantissa_words) return 0.0;

    double mantissa_double = (double)slot->mantissa_words[0] / (double)UINT64_MAX;

    double result = mantissa_double * pow(2.0, (double)slot->exponent);

    if (slot->state_flags & APA_FLAG_SIGN_NEG) {
        result *= -1.0;
    }

    return result;
}

// CPU step with APA processing
void lattice_step_cpu(HDGLLattice *lat, double tick) {
    int total_slots = lat->num_instances * lat->slots_per_instance;

    for (int i = 0; i < total_slots; i++) {
        Slot4096 *slot = lattice_get_slot(lat, i);
        if (!slot) continue;

        // Check for Implied Precision State
        if (slot->state_flags & (APA_FLAG_GOI | APA_FLAG_IS_NAN)) {
            // Skip computation if state is implied infinity or NaN
            continue;
        }

        double val = ap_to_double(slot);
        double r = prismatic_recursion(lat, i, fabs(val));

        // Ensure increment value can be negative to test sign-magnitude arithmetic
        double increment_value = (pow(slot->base, (double)slot->exponent) * tick + 0.05 * r) * (i % 2 == 0 ? 1.0 : -1.0);

        Slot4096 *increment_apa = ap_from_double(
            increment_value, slot->bits_mant, slot->bits_exp
        );
        if (!increment_apa) continue;

        ap_add(slot, increment_apa);

        ap_free(increment_apa);
        free(increment_apa);
    }

    lat->omega += 0.01 * tick;
    lat->time += tick;
}

// Prismatic folding: double instance count
void lattice_fold(HDGLLattice *lat) {
    int old_instances = lat->num_instances;
    int new_instances = old_instances * 2;
    if (new_instances > MAX_INSTANCES) return;

    int old_total = old_instances * lat->slots_per_instance;
    int new_total = new_instances * lat->slots_per_instance;
    int old_chunks = lat->num_chunks;
    int new_chunks = (new_total + CHUNK_SIZE - 1) / CHUNK_SIZE;

    HDGLChunk **new_chunks_ptr = realloc(lat->chunks, new_chunks * sizeof(HDGLChunk*));
    if (!new_chunks_ptr) {
        fprintf(stderr, "Failed to allocate memory for folding\n");
        return;
    }
    lat->chunks = new_chunks_ptr;

    for (int i = old_chunks; i < new_chunks; i++) {
        lat->chunks[i] = NULL;
    }

    for (int i = 0; i < old_total; i++) {
        Slot4096 *old_slot = lattice_get_slot(lat, i);
        Slot4096 *new_slot = lattice_get_slot(lat, old_total + i);

        if (old_slot && new_slot) {
            ap_copy(new_slot, old_slot);

            double perturbation = fib_table[i % fib_len] * 0.01;
            Slot4096 *pert_apa = ap_from_double(perturbation, new_slot->bits_mant, new_slot->bits_exp);

            if (pert_apa) {
                ap_add(new_slot, pert_apa);
                ap_free(pert_apa);
                free(pert_apa);
            }

            new_slot->base += get_normalized_rand() * 0.001;
        }
    }

    lat->num_instances = new_instances;
    lat->num_chunks = new_chunks;
}

// Free lattice: MUST handle freeing multi-word mantissa in every slot
void lattice_free(HDGLLattice *lat) {
    if (!lat) return;

    for (int i = 0; i < lat->num_chunks; i++) {
        if (lat->chunks[i])