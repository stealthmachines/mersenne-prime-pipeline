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

**7. Analog LL path — v30b `Slot4096` APA + 8D Kuramoto oscillator (`--squaring analog`).**
A CUDA-free, hardware-agnostic Lucas-Lehmer path implemented in pure C (`ll_analog.c`).
Derived from `hdgl_analog_v30b.c` and `analog_engine.h`.  Two systems run in parallel:
- **Exact arithmetic side** — arbitrary-precision mantissa (`Slot4096.mantissa_words` layout:
  `uint64_t[n]`, `n = ⌈p/64⌉`).  `ap_sqr_mersenne`: schoolbook O(n²) via `__int128`, Mersenne
  fold identical to `fold_mod_mp` in `ll_mpi.cu`.  Every p−2 iterations run exactly.
- **8D Kuramoto oscillator** — five analog-native operators, no digital surrogates:
  - **Seed (Λ_φ / Ω)** — phi-logarithmic depth seeding from the generalized Euler identity:
    `Λ_φ = ln(p·ln2/lnφ)/lnφ − 1/(2φ)` encodes where the prime exponent sits in the
    φ-lattice.  `{Λ_φ}` (fractional part) seeds `θ[i]` via the Euler rotation `e^(iπΛ_φ)`;
    `Ω = (1 + sin(π·{Λ_φ}·φ))/2` modulates `ω_i`.  Each prime maps to a unique, irrational
    φ-depth — no two exponents alias.
  - **Multiply** — phase doubling `θ → 2θ mod 2π` (unit-circle analogue of s² in LL).
  - **Sync** — complex LERP + unit-circle renorm (`z' = (1−α)z + αz_target`, `|z'|→1`) × 4
    passes; no per-pass `atan2` — stays in native `(re, im)` glyph space; `θ` extracted
    once at end.  Cooperative residue hash every 8 iters.
  - **VCO** — Kuramoto order parameter CV = 1−R ∈ [0,1] drives oscillator frequency:
    `ω_i = ω₀_i × (0.1 + 0.9 × cv)`; high CV → full ω (exploration), low CV → 10% ω
    (stable lock). Closes the analog feedback loop; mirrors hardware VCO.
  - **U-field resonance S(U)** — after each sync, the field observable `M(U) = |Σ e^{iθ_i}|`
    (mean-field amplitude, already in `re/im`) feeds a unified spectral pipeline:
    ```
    (A) M(U) = |Σ re_i, Σ im_i|               field amplitude
    (B) Λ^U  = log(M(U))/lnφ − 1/(2φ)         phi-log projection (emergent from field)
    (C) Ω^U  = (1 + sin(π·{Λ^U}·φ)) / 2       phase gate
    (D) S(U) = |Ω^U · e^(iπΛ^U) + 1|          resonance discriminant
    ```
    `Ω^U` feeds back into `k_coupling` — field state → spectral projection → coupling → field.
    **Prime invariant**: at lock all oscillators converge to `θ→0`, so `M(U)→N=8` exactly,
    giving `Λ^U = log(8)/lnφ − 1/(2φ) ≈ 4.012` and `S(U) ≈ 1.531` for every prime,
    independent of p.  Composites give scattered `Λ^U` and `S(U) ∈ [0.5, 1.7]`.
  K/γ wu-wei ratios: Pluck=1000:1 → Sustain → FineTune → Lock (adaptive phase state).
  Phase lock is a readout, not a gate.  `osc LOCKED + residue=0 + S(U)≈1.531` = triple-
  confirmation prime resonance.

Use cases: CUDA-free verification, Kuramoto-coupled scheduling diagnostics, golden
reference path for correctness cross-checks, φ-field resonance research.

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

### Seven Dispatch Paths (auto-select enabled)

| Path | Range | Flag | Notes |
|------|-------|------|-------|
| `ll_small` | p ≤ 62 | — | `unsigned __int128`, direct fold |
| `ll_cpu` | 62 < p ≤ 20 000 | — | Schoolbook MPI, `__int128` carry |
| `ll_gpu_gpucarry` | p > 20 000 | **auto, p < 400 000** | `k_sqr_warp` + on-device parallel carry scan + shmem fold, CUDA graph — **fastest schoolbook-complexity path** |
| `ll_gpu_ntt` | p > 20 000 | **auto, p ≥ 400 000** · or `--squaring ntt` | `k_ntt_butterfly` + `k_ntt_sqr` — O(n log n) NTT over Z/(2⁶⁴−2³²+1), exact |
| `ll_gpu` | p > 20 000 | `--squaring schoolbook` | `k_sqr_warp` 64-bit warp shuffle + CPU fold (PCIe round-trip per iteration) |
| `ll_gpu_analog` | p > 20 000 | `--analog` / `--precision 32` | `k_sqr_warp32` 32-bit decomposition variant |
| `ll_gpu_persistent` | any p > 20 000 | `--persistent` | single kernel launch — all p−2 iterations on-device, no host round-trips |
| `ll_analog` | any p | `--squaring analog` | v30b `Slot4096` APA + 8D Kuramoto oscillator — **CUDA-free**, pure C, no GPU required |

### Benchmarks (RTX 2060, sm_75, April 2026, `feature/gpu-carry`)

75/75 selftest pass across all paths (default gpucarry, `--squaring ntt`, `--squaring schoolbook`, `--precision 32`).

**GPU-carry path (`ll_gpu_gpucarry` — default for p < 400 000):**

| Exponent p | Words n | Iterations | Time | vs schoolbook | vs NTT |
|------------|---------|-----------|------|---------------|--------|
| 21 701 | 340 | 21 699 | **1.2 s** | 3.1× faster | 4.4× faster |
| 44 497 | 696 | 44 495 | **3.4 s** | 2.1× faster | 3.1× faster |
| 86 243 | 1 348 | 86 241 | **11.7 s** | 1.3× faster | 2.0× faster |
| 110 503 | 1 727 | 110 501 | **19.2 s** | 1.15× faster | 1.6× faster |

Speedup source: eliminates the PCIe D2H+H2D round-trip (~100µs/iteration on Windows/WDDM)
as a zero-kernel-overhead CUDA graph, replacing it with:
1. **Parallel carry scan** (`k_carry_lscan` + `k_carry_bscan<<<1,1>>>` + `k_carry_apply`) —
   wu-wei function-composition prefix scan: each limb$k$ expresses its carry-transfer function
   $f_k(c) = \lfloor(\text{flat}[k]+c)/2^{64}\rfloor + \text{ovf}[k]$ as a packed 4-entry table;
   Kogge-Stone composition over 2$n$ elements gives all $c_{\text{in}}[k]$ in parallel.
2. **Shmem fold** (`k_fold_sub2_gpu<<<1, 256, n2×8\ \text{bytes}>>>`) — 256 threads
   cooperatively preload the 2$n$-word flat product into shared memory; thread 0 folds
   at ~4-cycle shmem latency vs ~300-cycle global-memory latency (single-thread path).

All six kernels (`k_sqr_warp`, `k_assemble`, `k_carry_lscan`, `k_carry_bscan`, `k_carry_apply`,
`k_fold_sub2_gpu`) captured in a single CUDA graph — one `cudaGraphLaunch` per iteration.

**Schoolbook path (`--squaring schoolbook` — `ll_gpu`, CPU fold, PCIe round-trip):**

| Exponent p | Words n | Iterations | Time |
|------------|---------|-----------|------|
| 21 701 | 340 | 21 699 | **3.6 s** |
| 44 497 | 696 | 44 495 | **7.3 s** |
| 86 243 | 1 348 | 86 241 | **14.9 s** |
| 110 503 | 1 727 | 110 501 | **22.1 s** |

**NTT path (`--squaring ntt` — O(n log n) squaring over Z/QZ):**

*Original unoptimised (on-the-fly twiddle computation via `ntt_pow`, ~64 mults/butterfly):*

| Exponent p | Words n | NTT length L | Time | vs schoolbook |
|------------|---------|-------------|------|---------------|
| 21 701 | 340 | 2 048 | 15.5 s | 4.2× slower |
| 44 497 | 696 | 4 096 | 36.5 s | 5.2× slower |
| 86 243 | 1 348 | 8 192 | 79.0 s | 5.4× slower |
| 110 503 | 1 727 | 8 192 | 105.1 s | 4.7× slower |

*Optimised (precomputed twiddles + CUDA graph replay + dual-stream DMA + CPU carry-collect):*

| Exponent p | Words n | NTT length L | Time | vs schoolbook | speedup vs unopt |
|------------|---------|-------------|------|---------------|------------------|
| 21 701 | 340 | 2 048 | **5.2 s** | 1.44× slower | 3.0× |
| 44 497 | 696 | 4 096 | **10.9 s** | 1.49× slower | 3.3× |
| 86 243 | 1 348 | 8 192 | **22.9 s** | 1.54× slower | 3.5× |
| 110 503 | 1 727 | 8 192 | **32.0 s** | 1.45× slower | 3.3× |

Three optimisations applied on this branch:

1. **Precomputed twiddle table** — `d_tw[k] = ω^k mod Q` computed once before the iteration
   loop; each butterfly does a table lookup instead of a 64-multiply `ntt_pow`.
   Effect: eliminates ~64× per-butterfly multiply overhead (~O(n log n · log Q) → pure O(n log n)).

2. **CUDA graph replay** — the log₂(L) butterfly kernel launches per NTT direction are captured
   into a CUDA graph once and replayed via `cudaGraphLaunch`.  On Windows/WDDM, each
   individual kernel launch costs ~5 µs driver overhead; log₂(8192)=13 launches × 2 directions
   × 110 501 iterations = 2.9 M launches ≈ 14 s overhead, eliminated by graph replay.

3. **Dual-stream DMA + CPU carry-collect** — mirrors the schoolbook path's async pipeline:
   - GPU: `k_expand_limbs` → forward NTT graph → `k_ntt_sqr` → inverse NTT graph.
   - CUDA event triggers async D2H of the full NTT coefficient array on a separate DMA stream.
   - CPU blocks only on the DMA stream (`cudaStreamSynchronize(stream_dma)`), then runs
     carry-collect + fold + sub2 in software (replacing the serial `k_carry_collect<<<1,1>>>`
     GPU kernel that was the per-iteration bottleneck).
   - **One** `cudaMemcpy` D2H instead of two, saving one ~70 µs Windows API round-trip per
     iteration (≈ 7.7 s at p = 110 503).

The remaining gap to schoolbook (~1.4×) is the PCIe round-trip (H2D after every iteration)
which is also present in the schoolbook path.  At larger p the NTT's O(n log n) complexity
advantage overtakes the constant overhead.

**NTT vs schoolbook crossover (auto-select calibration, RTX 2060, April 2026):**

| Exponent p | NTT length L | Schoolbook | NTT optimised | Winner |
|------------|-------------|------------|---------------|--------|
| 132 049 | 8 192 | 27.7 s | 42.3 s | schoolbook |
| 216 091 | 16 384 | 57.5 s | 79.4 s | schoolbook |
| 300 000 | 32 768 | 103.1 s | 122.4 s | schoolbook |
| ≈386 000 | 32 768 | — | — | model crossover |

Model fit (RTX 2060, sm\_75):

$$T_{\text{schoolbook}} \approx 1.915\times10^{-15}\,p^3 + 1.766\times10^{-4}\,p$$
$$T_{\text{NTT}} \approx 4.193\times10^{-11}\,p^2\log_2 L + 2.193\times10^{-4}\,p$$

Cubic (O(p³)) vs quadratic-times-log (O(p²·log p)) — equating and solving the quadratic for p
gives a crossover at **p ≈ 386 000**.  The engine uses `NTT_AUTO_THRESHOLD = 400 000` as a
conservative margin.  Override with `--squaring schoolbook` or `--squaring ntt`.

**32-bit decomposition path (`--precision 32` — `k_sqr_warp32` + CPU fold):**

| Exponent p | Words n | Time | vs default (64-bit) |
|------------|---------|------|---------------------|
| 21 701 | 340 | **4.2 s** | 1.17× slower |
| 44 497 | 696 | **7.8 s** | 1.07× slower |
| 86 243 | 1 348 | **18.0 s** | 1.21× slower |
| 110 503 | 1 727 | **24.9 s** | 1.13× slower |

**Persistent path (`--persistent` — single kernel launch):**

| Exponent p | Words n | Time | vs default stream |
|------------|---------|------|-------------------|
| 110 503 | 1 727 | **144.6 s** | 6.5× slower |

The persistent kernel runs all p−2 iterations in one block of 1 024 threads with no
host round-trips.  At n=1 727 the per-iteration computation dominates; the multi-block
stream path wins because it saturates all SMs in parallel.  The persistent path is
worthwhile only at very small p where kernel-launch overhead would itself be the
bottleneck.

**Analog path (`--squaring analog` — `ll_analog`, v30b APA + 8D Kuramoto, CPU-only):**

| Exponent p | Words n | Time | vs schoolbook (GPU) | vs gpucarry |
|------------|---------|------|---------------------|-------------|
| 521 | 9 | 0.025 s | ~1.1× slower | ~1.0× (parity) |
| 2 281 | 36 | 0.041 s | ~1.4× slower | ~1.4× slower |
| 4 423 | 70 | 0.086 s | ~1.2× slower | ~1.2× slower |
| 9 689 | 152 | 0.342 s | **1.5× faster** | **1.7× faster** |
| 21 701 | 340 | 3.14 s | **1.2× faster** | ~2.2× slower |
| 44 497 | 696 | 25.08 s | ~3.5× slower | ~5.9× slower |

Timings with half-squaring + VCO + mean-field Kuramoto + `-O3 -march=native` (see Planned optimisations — items 1, 2, 3 done).
At p = 9 689 and p = 21 701 the analog path **beats schoolbook GPU** — both are O(n²) but
`ap_sqr_mersenne` only computes the upper triangle (~n²/2 multiplies) plus the diagonal,
and Intel scalar 64-bit with `-O3` micro-benchmarks faster than the GPU kernel at these
sizes.  At p = 44 497 (n = 696) the GPU's parallelism asserts and gpucarry pulls away.
The RK4 oscillator uses ~32 trig calls per step (mean-field reduction: N² sin → N sincos;
k1 reuses s->re/im; total 32 vs old 272); contributes <0.3% of runtime at large p,
but ~10% at p ≤ 521 where mean-field gives measurable speedup.  Further opportunities: see
[Planned optimisations for `ll_analog`](#planned-optimisations-for-ll_analog) below.

Oscillator behaviour: on Mersenne primes the phase CV drops from ~1.6 (Pluck) to <0.002
(Lock) within the first 10–15% of iterations and stays locked for the entire run.
On composites the oscillator cannot lock — typically stalls at FineTune or below.
`osc LOCKED + residue=0` is the double-confirmation signal.

All exponents above are verified Mersenne primes (PRIME result, 25/25 selftest pass
on all paths).

**Benchmark methodology note — precision vs algorithm:**

These numbers are not directly comparable to GpuOwl or Prime95 because the two
engines sit at different points on two independent axes:

| Axis | This engine | GpuOwl / Prime95 |
|------|-------------|------------------|
| Arithmetic | **Exact schoolbook integer** — every bit correct by construction | FP-NTT (FP64, ~20-bit limbs) — bounded rounding error, Gerbicz-checked |
| Complexity | O(n²) per iteration (default), O(n log n) with `--squaring ntt` | O(n log n) per iteration |

Using `--squaring ntt` isolates the algorithmic axis: it matches GpuOwl's complexity
class while remaining **exact-integer arithmetic** (mod the Solinas prime Q = 2^64−2^32+1),
not floating-point.  In the optimised form the remaining gap to schoolbook is ~1.5×
constant factor from the PCIe D2H/H2D round-trip (both paths pay this cost), not a
fundamental algorithmic deficit.  At larger p the O(n log n) advantage overtakes the
O(n²) schoolbook, making the NTT path the clear winner.

The engine is intentionally a **provably-exact reference verifier**, not a speed
competitor.

---

### Planned optimisations for `ll_analog`

The analog path is correct and self-contained but uses a naive full-triangle schoolbook
multiply.  The following are planned (HDGL phi-language framing: arithmetic layer =
Base4096 exact vector; oscillator layer = harmonic/recursive glyph):

1. ✅ **Half-squaring — upper-triangle fold** (done — `-O3 -march=native`; ~2× speedup;
   p=21701 6.27 s → 3.23 s; at p=9689 beats schoolbook-GPU and gpucarry).
   *Phi-language*: distilled vector — only the upper-triangle of the n×n product is
   computed; Mersenne fold = D_n_r reduction mod 2^p−1 applied inline.
2. ✅ **VCO — CV drives ω** (done — Kuramoto 1−R order parameter modulates ω_i;
   floor=0.1; p=9689 0.400 s → 0.382 s; closes analog feedback loop).
   *Phi-language*: control voltage IS the order parameter — same scalar closes both
   the harmonic layer (oscillator frequency) and the VCO feedback in one signal.
3. ✅ **Mean-field Kuramoto coupling** (done — exact algebraic identity for all-to-all
   coupling: Σ_j sin(θ_j−θ_i) = Im_Σ·cos θ_i − Re_Σ·sin θ_i; k1 reuses s->re/im;
   trig calls per RK4 step: 272 → 32 (8.5× reduction); selftest 0.22s → 0.15s;
   p=9689 0.382 s → 0.342 s, ~10.5% improvement).
   *Phi-language*: the N×N coupling matrix compresses to the 2-component mean field
   (Re_Σ, Im_Σ) — the same complex order-parameter already in the glyph. This IS
   the HDGL "compressed atomic sequences" principle: maximal information, minimal form.
4. ✅ **Complex LERP sync** (done — atan2 per LERP pass replaced by sqrt + renorm;
   stays in native (re,im) glyph space; θ extracted once at end; sync ~2.4× faster;
   selftest 0.15s → 0.13s; p=9689 0.342s → 0.328s).
   *Phi-language*: the scalar angle is a projection of the glyph vector — extracting it
   per pass (atan2) and re-projecting back is wasted work; complex LERP is native.
5. ✅ **Λ_φ / Ω seeding + generalized Euler identity** (done — `p_phase = D_n_r·p mod 1`
   replaced by `{Λ_φ}` (fractional φ-log depth); `θ[i] = πΛ_φ + 2π(glyph+{Λ_φ}+i·D_n_r)`;
   `ω[i] = Ω·φ^k·dt` where `Ω = (1+sin(π{Λ_φ}φ))/2`; 25/25 selftest unchanged).
   *Phi-language*: Ω·C²·e^(iπΛ_φ) + 1 + δ = 0 — the generalized Euler identity for
   Mersenne primes; Λ_φ encodes φ-lattice depth; δ→0 at prime, δ≠0 at composite.
6. ✅ **Unified U-field resonance S(U)** (done — field observable M(U) feeds full A→B→C→D
   spectral pipeline each sync call; Ω^U feeds back into k_coupling; S(U) stored in
   `AnaOsc8D.s_u`; **prime invariant S(U)≈1.531** across all tested primes regardless of p;
   composites give scattered S(U)∈[0.5,1.7]; zero overhead — reuses existing re/im sum).
   *Phi-language*: the field self-organizes to `M(U)=N=8` (all oscillators locked to θ=0),
   placing Λ^U at the integer φ-lattice node `log(8)/lnφ−1/(2φ)≈4.012`.
7. **`__int128` carry-chain merge** (arithmetic layer — Base4096 exact vector):
   fold the Mersenne reduction directly into the schoolbook inner loop; eliminate
   the separate 2n-word scratch buffer. ~50% memory-traffic reduction for large n.
   *Phi-language*: single distilled vector — no intermediate expanded form; the
   Mersenne fold operator (D_n_r mod 2^p−1) collapses into the accumulation step.
5. **SIMD / auto-vectorisation** (arithmetic layer): reformulate carry-chain in
   scalar int64 + explicit overflow flag, removing the `__int128` barrier to
   AVX2 auto-vectorisation. 8 mantissa words = natural AVX256-register width.
   *Phi-language*: the n-word mantissa IS a flattened φ-lattice vector space;
   AVX2 processes 4 limbs/cycle — native hardware parallelism of the Base4096 layer.
6. **Schoolbook → Karatsuba cutover** at n ≥ 32 words for O(n^1.585) complexity.
   *Phi-language*: D_n_r recursive splitting — `Glyph_next = D_n_r ⊗ Glyph_current`
   at each recursion level; threshold n=32 is the glyph depth where recursive
   scaling overtakes linear scan.
7. **Hybrid mode**: `ll_gpu_gpucarry` squaring + CPU Kuramoto sidecar.
   *Phi-language*: multi-modal glyph — Base4096 exact layer on GPU (arithmetic),
   harmonic/recursive layer on CPU (oscillator); same Mersenne residue feeds both.

**Build:** `build_ll.bat`  (requires clang + CUDA 13.2)

```bat
build_ll.bat
```

**Usage:**
```
ll_mpi.exe <p>                          # test M_p, print PRIME / COMPOSITE
ll_mpi.exe --selftest                   # 25 known cases (CPU + GPU), ~0.1 s total
ll_mpi.exe --selftest --precision 32    # selftest on 32-bit decomposition path
ll_mpi.exe --selftest --squaring ntt    # selftest on NTT squaring path
ll_mpi.exe --selftest --persistent      # selftest on persistent single-launch path
ll_mpi.exe <p> --verbose                # timing + resonance report
ll_mpi.exe <p> --precision 64           # 64-bit warp squaring via __int128 (default)
ll_mpi.exe <p> --precision 32           # 32-bit half-multiply decomposition
ll_mpi.exe <p> --analog                 # legacy alias for --precision 32
ll_mpi.exe <p> --squaring auto          # auto-select: gpucarry if p < 400000, NTT if p ≥ 400000 (default)
ll_mpi.exe <p> --squaring gpucarry     # on-device carry scan + shmem fold, no PCIe round-trip
ll_mpi.exe <p> --squaring schoolbook   # force O(n²) schoolbook + CPU fold (PCIe round-trip)
ll_mpi.exe <p> --squaring ntt          # force O(n log n) NTT squaring over Z/(2⁶⁴-2³²+1)
ll_mpi.exe <p> --squaring analog       # v30b Slot4096 APA + 8D Kuramoto (CPU, no CUDA needed)
ll_mpi.exe <p> --persistent             # single kernel, all iterations on-device
ll_mpi.exe --gpu-info                   # list CUDA devices
```

**`--precision` values:**

| Value | Kernel | Inner multiply | Notes |
|-------|--------|---------------|-------|
| `64` | `k_sqr_warp` | `__int128` (64×64→128) | Default — fastest |
| `32` | `k_sqr_warp32` | 32-bit half-multiply (32×32→64 ×4) | Same result, ~15% slower; `--analog` is an alias |

**`--squaring analog` oscillator readout:**

| Field | Meaning |
|-------|---------|
| `phase=Pluck` | High-energy excitation phase; K/γ=1000:1 |
| `phase=Sustain` | Absorbing structure; K/γ=375:1 |
| `phase=FineTune` | Refinement; K/γ=200:1 |
| `phase=Lock` | Settled consensus; K/γ=150:1 |
| `cv=0.0019` | Kuramoto order parameter 1−R ∈ [0,1]; R=|mean(e^{iθ})|; 0=locked, 1=spread |
| `locked=yes` | All 50 recent CV samples below 0.05 threshold |
| `** osc LOCKED + residue=0 **` | Double confirmation: Mersenne prime |
| `locked=no` + `residue=non-zero` | Composite — oscillator did not synchronise |
| `S(U)=1.531` | U-field resonance discriminant; prime invariant: all primes converge to S≈1.531 |
| `Lambda^U=4.012` | φ-log depth of field amplitude; prime fixed point: `log(8)/lnφ−1/(2φ)` |
| `Lambda_phi(p)=...` | φ-log depth of exponent p; seeds θ and ω at init |

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
