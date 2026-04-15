# Mersenne Prime Search Pipeline

A 4-track GPU-accelerated pipeline for finding Mersenne prime candidates and
verifying them with Lucas-Lehmer, running on an **RTX 2060 12 GB** (CUDA sm_75).

## What Is It?

This is an end-to-end system in beta for eventually hunting the next world-record Mersenne prime.
It combines two original mathematical ideas — a **phi-lattice resonance scoring
function** and the **HDGL wu-wei data-flow model** — with a hand-optimised,
exact-integer Lucas-Lehmer GPU verifier.

A Mersenne prime has the form M_p = 2^p − 1.  There are only 52 known.  The
largest (M_82589933, found 2018, ~24.8 million digits) took weeks on specialized
hardware.  This pipeline fits entirely on a single consumer GPU and can verify
exponents up to ~130 000 in minutes.

## What's Cool About It?

**1. No floating-point in the modular arithmetic.**
Most GPU Lucas-Lehmer implementations use cuFFT to multiply big numbers in O(n log n).
This one uses a pure-integer schoolbook squaring kernel — exact,
zero rounding error, no external library, no cuFFT, no DWT.

**2. Warp-parallel schoolbook squaring (k_sqr_warp).**
Classical GPU LL assigns one thread per output limb of the squaring (O(n) inner
loop per thread → only ~1400 threads active for p=44497, starving 96% of the GPU).
`k_sqr_warp` assigns a full **warp of 32 threads** to each output limb.  Threads
split the inner-product terms across lanes and recombine with `__shfl_down_sync`
warp shuffle — no atomics, no shared memory, no memset between iterations.
This fills the GPU at any exponent size and delivers 1.2–2.5× speedup.

**3. HDGL wu-wei hybrid pipeline.**
The HDGL (Harmonics-of-Digital-Geometric-Lattice) philosophy: let each part of the
computation run where it is fastest.  The GPU does parallel squaring.  The CPU does
sequential carry-propagation and Mersenne fold (x86 -O2 is 10–12× faster than a
single GPU thread for O(n) sequential work).  Pinned host memory and an async
CUDA stream connect the two sides with minimal stall.

**4. phi-lattice candidate ranking.**
The phi-lattice coordinate n(2^p) = log(p·ln2 / ln(φ)) / ln(φ) − 1/(2φ)
maps known Mersenne exponents onto a number line.  67% of the 51 known
exponents land in the lower half of each unit interval (vs. 50% random).
The pipeline filters and scores new candidates by this resonance before
committing GPU time to Lucas-Lehmer verification.

**5. Analog 32-bit warp squaring path (`k_sqr_warp32`, `--precision 32` / `--analog`).**
An alternative squaring kernel that decomposes each 64-bit limb into 32-bit halves
before warp-shuffle reduction.  Selectable at runtime with `--precision 32` (or the
legacy `--analog` alias); produces identical results to the default path.

**6. Persistent on-device loop (`k_ll_persistent_block`, `--persistent`).**
Runs all p−2 squaring iterations inside a single kernel launch: shared memory holds
the current `s` vector (`n×8` bytes, fits 48 KB for p up to ~460 000), and
`d_lo/d_mi/d_hi` scratch buffers hold the squaring product.  Zero host round-trips
after the initial upload.  Reference path and correctness cross-check; for large p
the multi-block stream path is faster because it saturates all GPU SMs in parallel.

**7. Riemann psi scanner as a secondary filter.**
The GPU Track A scanner evaluates the Riemann explicit formula with up to 10 000
zeta zeros to assign each candidate a primality confidence score, eliminating
composites before the expensive squaring step.

---

## Architecture

```
prime_pipeline.exe <p_lo> <p_hi> --exponents-only
        |
        |  prime exponents ranked by phi-lattice D_n score
        v
ll_mpi.exe <p>
        |
        |  PRIME or COMPOSITE  (exact integer, schoolbook GPU)
        v
   world-record candidate
```

| Track | File | Purpose |
|-------|------|---------|
| A | psi_scanner_cuda.cu  | GPU Riemann-psi scanner (up to 10 000 zeta zeros) |
| B | phi_mersenne_predictor.c | phi-lattice analysis of all 51 known exponents |
| C | ll_mpi.cu | Lucas-Lehmer verifier — exact integer, no DWT, no cuFFT |
| D | prime_pipeline.c | phi-filter + D_n ranker + sieve |

---

## Track C — Lucas-Lehmer Verifier (ll_mpi.cu)

### How It Works

Exact-integer Lucas-Lehmer: M_p is prime iff s_{p−2} ≡ 0 (mod M_p),
where s_0 = 4, s_{i+1} = s_i² − 2 (mod M_p).

Each squaring step:

```
CPU host:                          GPU device:
                                   ┌─ k_sqr_warp ──────────────────────────┐
  d_x ──────────────────────────►  │  32 threads per output limb k (0..2n-1) │
  (n 64-bit limbs, n=ceil(p/64))   │  lane l sums x[i]*x[k-i] for i≡l mod 32│
                                   │  warp shuffle reduction → d_lo/mi/hi[k] │
                                   └───────────────────────────────────────┘
                                   ┌─ k_assemble ──────────────────────────┐
                                   │  1 thread per position k               │
                                   │  d_flat[k] = lo[k]+mi[k-1]+hi[k-2]    │
                                   │  overflow byte in d_ovf[k]             │
                                   └───────────────────────────────────────┘
  pinned h_flat  ◄── async D2H ──  d_flat  (n2×8 bytes, single transfer)
  pinned h_ovf   ◄── async D2H ──  d_ovf   (n2×1 bytes)

CPU carry propagation:
  for k in 0..2n-1:
    acc = h_flat[k] + carry;  h_flat[k] = acc & mask64;  carry = acc>>64 + h_ovf[k]

CPU Mersenne fold mod 2^p-1:
  lo-half [0..pw] preserved; hi-half [pw..2n] right-shifted by pb bits and added back

CPU sub-2 mod M_p → h_x
  d_x ◄──── H2D upload ─── h_x
```

### Five Dispatch Paths

| Path | Range | Flag | Notes |
|------|-------|------|-------|
| `ll_small` | p ≤ 62 | — | `unsigned __int128`, direct fold |
| `ll_cpu` | 62 < p ≤ 20 000 | — | Schoolbook MPI, `__int128` carry |
| `ll_gpu` | p > 20 000 | (default) | `k_sqr_warp` 64-bit warp shuffle + CPU fold |
| `ll_gpu_analog` | p > 20 000 | `--analog` / `--precision 32` | `k_sqr_warp32` 32-bit decomposition variant |
| `ll_gpu_persistent` | any p > 20 000 | `--persistent` | single kernel launch — all p−2 iterations on-device, no host round-trips |

### Benchmarks (RTX 2060, sm_75, April 2026)

**Default stream path (`ll_gpu` — `k_sqr_warp` + CPU fold):**

| Exponent p | Words n | Iterations | Time | vs limb baseline |
|------------|---------|-----------|------|------------------|
| 21 701 | 340 | 21 699 | **2.28 s** | 1.21× faster |
| 44 497 | 696 | 44 495 | **4.25 s** | 1.63× faster |
| 86 243 | 1 348 | 86 241 | **10.2 s** | 2.08× faster |
| 110 503 | 1 727 | 110 501 | **21.3 s** | 2.54× faster |

Speedup grows with p because larger n means longer warp inner loops — the 32-lane
warp reduction provides a larger multiplier on the serial inner-product bottleneck.

**Persistent path (`--persistent` — single kernel launch):**

| Exponent p | Words n | Time | vs default stream |
|------------|---------|------|-------------------|
| 110 503 | 1 727 | **144.6 s** | 6.8× slower |

The persistent kernel runs all p−2 iterations in one block of 1 024 threads with no
host round-trips.  At n=1 727 the per-iteration computation dominates; the multi-block
stream path wins because it saturates all SMs in parallel.  The persistent path is
worthwhile only at very small p where kernel-launch overhead would itself be the
bottleneck.

All exponents above are verified Mersenne primes (PRIME result, 25/25 selftest pass
on both default and persistent paths).

**Benchmark methodology note — precision vs algorithm:**

These numbers are not directly comparable to GpuOwl or Prime95 because the two
engines sit at different points on two independent axes:

| Axis | This engine | GpuOwl / Prime95 |
|------|-------------|------------------|
| Arithmetic | **Exact schoolbook integer** — every bit correct by construction | FP-NTT (FP64, ~20-bit limbs) — bounded rounding error, Gerbicz-checked |
| Complexity | O(n²) per iteration | O(n log n) per iteration |

A floating-point squaring kernel (`--precision fp`) would require changing the limb
representation: our 64-bit limbs produce 128-bit products that overflow the FP64
53-bit mantissa.  Correct FP multiplication requires limbs ≤ 26 bits — a full
data-layout refactor.  Even after that change, the FP speedup would be ~2–3× on
raw multiply throughput.  The remaining ~250–300× gap to GpuOwl at p = 1 M is
purely algorithmic (O(n²) schoolbook vs O(n log n) NTT) and unaffected by
precision tier.

The engine is intentionally a **provably-exact reference verifier**, not a speed
competitor.

**Build:** `build_ll.bat`  (requires clang + CUDA 13.2)

```bat
build_ll.bat
```

**Usage:**
```
ll_mpi.exe <p>                          # test M_p, print PRIME / COMPOSITE
ll_mpi.exe --selftest                   # 25 known cases (CPU + GPU), ~0.1 s total
ll_mpi.exe --selftest --precision 32    # selftest on 32-bit decomposition path
ll_mpi.exe --selftest --persistent      # selftest on persistent single-launch path
ll_mpi.exe <p> --verbose                # timing + resonance report
ll_mpi.exe <p> --precision 64           # 64-bit warp squaring via __int128 (default)
ll_mpi.exe <p> --precision 32           # 32-bit half-multiply decomposition
ll_mpi.exe <p> --analog                 # legacy alias for --precision 32
ll_mpi.exe <p> --persistent             # single kernel, all iterations on-device
ll_mpi.exe --gpu-info                   # list CUDA devices
```

**`--precision` values:**

| Value | Kernel | Inner multiply | Notes |
|-------|--------|---------------|-------|
| `64` | `k_sqr_warp` | `__int128` (64×64→128) | Default — fastest |
| `32` | `k_sqr_warp32` | 32-bit half-multiply (32×32→64 ×4) | Same result, ~15% slower; `--analog` is an alias |

All flags scan the full `argv` array; order relative to `<p>` does not matter.
Flags may be freely combined (`--precision 32 --verbose`, `--selftest --precision 32`, etc.).

---

## Track A — Riemann psi Scanner

GPU-accelerated prime scanner using the Riemann explicit formula:

    Δψ(x) = x − Σ_k 2·Re(x^ρ_k / ρ_k) − log(2π)

Three-pass adaptive pipeline: 500 → 5000 → 10 000 zeta zeros, then Miller-Rabin.
Requires `zeta_zeros_10k.json` in working directory.

**Build:** `nvcc -O3 -arch=sm_75 -o psi_scanner_cuda.exe psi_scanner_cuda.cu`
**Usage:** `psi_scanner_cuda.exe <x_start> <x_end> [--mersenne]`

---

## Track B — phi-Lattice Predictor

Validates and exploits the phi-lattice hypothesis:

    n(2^p) = log( p·ln2 / ln(φ) ) / ln(φ) − 1/(2φ)

67% of known Mersenne exponents have frac(n) < 0.5 (vs. 50% for random primes).
Produces top-20 next-candidate predictions beyond M_51 (p = 136 279 841).

**Build:** `clang -O2 -D_CRT_SECURE_NO_WARNINGS phi_mersenne_predictor.c -o phi_mersenne_predictor.exe`

---

## Track D — Prime Pipeline

Segmented sieve + phi-lattice D_n scoring:

1. Sieve [p_lo, p_hi] for prime exponents
2. Compute n(2^p) — flag lower-half (frac < 0.5) for 1.5× score bonus
3. Prismatic Ω = 0.5 + 0.5·sin(π·frac·φ)
4. D_n = √(φ · F_n · P_n · 2^n · Ω) · r^k
5. Sort descending, emit top-N

**Build:** `clang -O2 -D_CRT_SECURE_NO_WARNINGS prime_pipeline.c -o prime_pipeline.exe`

**End-to-end (PowerShell):**
```powershell
.\prime_pipeline.exe 21000 22000 --top 10 --exponents-only |
  ForEach-Object { .\ll_mpi.exe $_ }
```

---

## Requirements

| Component | Version |
|-----------|---------|
| GPU | NVIDIA sm_75+ (RTX 2060 / 2070 / 2080 / 3xxx / 4xxx) |
| CUDA Toolkit | 13.2 |
| clang | 14+ with CUDA target support |
| OS | Windows 10/11 (Linux: remove `-D_CRT_SECURE_NO_WARNINGS`) |

---

## Mathematical Foundation

**phi-lattice coordinate**

    n(x) = log( log(x)/ln(φ) ) / ln(φ) − 1/(2φ)

**D_n resonance operator**

    D_n = √( φ · F_n · P_n · base^n · Ω ) · r^k

where F_n is continuous Binet Fibonacci, P_n is the nearest entry in a 50-prime
table, and Ω = 0.5 + 0.5·sin(π·frac(n)·φ) is the prismatic phase term.

**Mersenne fold identity (2^p ≡ 1 mod M_p)**

    a·2^p + b  ≡  a + b  (mod M_p)

The 2n-word squaring product is split at bit p; the upper half is right-shifted
by p bits and added back to the lower half.  No floating-point, no convolution.

**k_sqr_warp inner-product decomposition**

For output limb k of x² (x has n 64-bit limbs):

    flat[k] = Σ_{i=i0}^{i1}  x[i] · x[k−i]

Thread lane l (0..31) computes the sub-sum for i = i0+l, i0+l+32, i0+l+64, …
A log₂(32)-stage butterfly with `__shfl_down_sync` reduces the 32 partial 192-bit
sums to the single answer in lane 0, written to d_lo[k]/d_mi[k]/d_hi[k].
Total active threads: 32 × 2n (vs. 2n for k_sqr_limb) — full SM utilisation.
