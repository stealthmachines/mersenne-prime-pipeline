"""
c_is_better_local.py  —  Windows-portable version of c_is_better.py
Drops the /home/claude/zeta_zeros_10k.json dependency.
Uses the same 60 hardcoded Riemann zeros as optimal-prime2.c + adds up to B=500
via the mpmath library if available, else falls back to 60 zeros.

Benchmarks:
  1. phi-lattice psi scanner (same algorithm as both optimal-prime2.c and c_is_better.py)
  2. Pure-Python schoolbook Lucas-Lehmer  (traditional exact Mersenne verifier)

Usage:
  python c_is_better_local.py
"""

import math, time
import numpy as np

# ── 60 hardcoded Riemann zeros (same as optimal-prime2.c) ─────────────────
ZETA_ZEROS_60 = np.array([
    14.134725141734694,21.022039638771555,25.010857580145688,
    30.424876125859513,32.935061587739189,37.586178158825671,
    40.918719012147495,43.327073280914999,48.005150881167159,
    49.773832477672302,52.970321477714460,56.446247697063246,
    59.347044002602353,60.831778524609809,65.112544048081652,
    67.079810529494173,69.546401711173977,72.067157674481907,
    75.704690699083933,77.144840069745199,79.337375020249367,
    82.910380854086030,84.735492980517050,87.425274613125229,
    88.809111207634465,92.491899271363504,94.651344040519840,
    95.870634228245309,98.831194218193198,101.31785100573140,
    103.72553804047830,105.44662305232611,107.16861118427640,
    111.02953554316510,111.87465917699263,114.32022091545400,
    116.22668032085762,118.79078286597578,121.37012500242042,
    122.94682929355241,124.25681855434580,127.51668387959577,
    129.57870419995625,131.08768853093250,133.49773720561993,
    134.75650975337150,138.11600798837089,139.73620895212138,
    141.12370740402112,143.11184580762272,146.00098247981454,
    147.42276952571063,150.05352820956492,150.92525703791911,
    153.02469757139972,156.11290893100908,157.59759196166191,
    158.84999978364746,161.18896390679989,163.03057065534800,
], dtype=np.float64)

# Try to extend to B=500 via mpmath
_zeros_array = None

def _get_zeros(B: int) -> np.ndarray:
    global _zeros_array
    if _zeros_array is not None and len(_zeros_array) >= B:
        return _zeros_array[:B]
    try:
        from mpmath import zetazero
        n = max(B, 60)
        z = np.array([float(zetazero(k).imag) for k in range(1, n+1)])
        _zeros_array = z
        return z[:B]
    except Exception:
        pass
    # fallback
    return ZETA_ZEROS_60[:min(B, 60)]


def psi_vec(x_arr, B: int) -> np.ndarray:
    zeros = _get_zeros(B)
    B = len(zeros)
    x   = np.asarray(x_arr, dtype=np.float64)
    lx  = np.log(np.maximum(x, 2.0))
    res = x.copy()
    # batch to avoid huge temp arrays
    for i in range(0, B, 200):
        zb  = zeros[i:i+200][None, :]   # (1, batch)
        lxc = lx[:, None]                # (N, 1)
        mag = np.exp(0.5 * lxc)
        corr = (2 * mag * (0.5*np.cos(zb*lxc) + zb*np.sin(zb*lxc)) / (0.25 + zb**2)).sum(1)
        res -= corr
    return res - math.log(2 * math.pi)


def miller_rabin(n: int) -> bool:
    if n < 2: return False
    for p in (2,3,5,7,11,13):
        if n == p: return True
    if n % 2 == 0 or n % 3 == 0: return False
    d, r = n - 1, 0
    while d % 2 == 0: d //= 2; r += 1
    for a in [2,3,5,7,11,13,17,19,23,29,31,37]:
        if a >= n: continue
        x = pow(a, d, n)
        if x == 1 or x == n - 1: continue
        for _ in range(r - 1):
            x = pow(x, 2, n)
            if x == n - 1: break
        else:
            return False
    return True


def phi_prime_scanner(xs: float, xe: float, B: int = 60, chunk: int = 5000) -> list:
    """Scan [xs, xe] for primes using phi-lattice psi + MR confirmation."""
    x_vals = np.arange(max(2, int(xs)), int(xe)+1, dtype=np.float64)
    found = []
    for ci in range(0, len(x_vals), chunk):
        batch = x_vals[ci:ci+chunk]
        psi   = psi_vec(batch, B)
        jumps = np.diff(psi)
        idxs  = np.where(jumps > 0.3)[0] + 1
        for idx in idxs:
            xr = int(batch[idx])
            if miller_rabin(xr):
                found.append(xr)
    return sorted(set(found))


# ── Pure-Python schoolbook Lucas-Lehmer ───────────────────────────────────
def lucas_lehmer_py(p: int) -> bool:
    """Exact Lucas-Lehmer test for M_p = 2^p - 1. Pure Python bigint."""
    if p == 2: return True
    M = (1 << p) - 1
    s = 4
    for _ in range(p - 2):
        s = (s * s - 2) % M
    return s == 0


# ══════════════════════════════════════════════════════════════════════════
# BENCHMARK 1: phi-lattice scanner  (apples-to-apples vs optimal-prime2.c)
# ══════════════════════════════════════════════════════════════════════════
print("=" * 65)
print(" BENCHMARK A: phi-lattice psi scanner (c_is_better_local.py)")
print("=" * 65)

scan_ranges = [
    (2,      500,    60,  "B=60"),
    (2,     1000,    60,  "B=60"),
    (2,    10000,    60,  "B=60"),
    (10000, 20000,   60,  "B=60"),
]

for (xs, xe, B, blabel) in scan_ranges:
    t0 = time.perf_counter()
    found = phi_prime_scanner(xs, xe, B=B)
    dt = time.perf_counter() - t0
    # reference count via sympy
    try:
        from sympy import primerange
        known = list(primerange(xs, xe+1))
        missed = [p for p in known if p not in set(found)]
        fp     = [x for x in found if not miller_rabin(x)]
        prec   = 100.0 * len(found) / max(1, len(found)+len(fp))
        rec    = 100.0 * (len(known)-len(missed)) / max(1, len(known))
        note   = f"prec={prec:.2f}% recall={rec:.2f}% missed={len(missed)} FP={len(fp)}"
    except Exception:
        note = f"found={len(found)}"
    print(f"  [{xs:>6}, {xe:>6}] {blabel} | {note} | {dt*1000:.1f} ms")

# Try B=500 if mpmath available
print()
print("  Testing with B=500 zeros (if mpmath available):")
try:
    from mpmath import zetazero as _zz
    _zz(1)  # trigger load
    t0 = time.perf_counter()
    found500 = phi_prime_scanner(2, 500, B=500)
    dt500 = time.perf_counter() - t0
    from sympy import primerange as _pr
    known500 = list(_pr(2, 501))
    missed500 = [p for p in known500 if p not in set(found500)]
    print(f"  [2, 500] B=500 | primes found={len(found500)}/{len(known500)}"
          f" missed={len(missed500)} | {dt500*1000:.1f} ms")
except Exception as e:
    print(f"  mpmath unavailable ({e}); skipping B=500 test")


# ══════════════════════════════════════════════════════════════════════════
# BENCHMARK B: Pure-Python Lucas-Lehmer  (traditional exact LL)
# ══════════════════════════════════════════════════════════════════════════
print()
print("=" * 65)
print(" BENCHMARK B: Pure-Python schoolbook Lucas-Lehmer")
print("=" * 65)

mersenne_exponents = [2, 3, 5, 7, 13, 17, 19, 31, 61, 89, 107, 127,
                      521, 607, 1279, 2203, 2281, 3217, 4253, 4423,
                      9689, 9941, 11213]

print(f"  {'p':>6}  {'result':9}  {'time_ms':>10}  {'digits':>8}")
print(f"  {'-'*6}  {'-'*9}  {'-'*10}  {'-'*8}")
for p in mersenne_exponents:
    t0 = time.perf_counter()
    result = lucas_lehmer_py(p)
    dt = (time.perf_counter() - t0) * 1000
    label = "PRIME" if result else "COMPOSITE"
    digits = len(str((1 << p) - 1))
    print(f"  {p:>6}  {label:9}  {dt:>10.1f}  {digits:>8}")
    if dt > 120_000:  # 2 min cap
        print("  (stopping — too slow beyond this point)")
        break
