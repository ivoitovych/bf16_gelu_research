# BFloat16 GELU Approximation Research

Systematic ULP (Units in Last Place) error analysis of GELU activation function approximations for bfloat16 floating-point arithmetic, targeting ML accelerator frameworks.

## Overview

This project implements and evaluates multiple GELU approximation strategies optimized for **bfloat16** precision, measuring accuracy using **ULP error** rather than MSE. Unlike typical analyses that focus on common activation ranges like [-8, 8], this research covers the **entire bfloat16 range** (65,280 valid values, approximately ±3.4×10³⁸).

### Key Achievements

**16 methods achieve Max ULP ≤ 88** with 6 Pure methods at Max ULP ≤ 35. **R5 Pure achieves Max ULP = 2** (best overall, Mean ULP = 0.002) after fixing subnormal handling in exp approximation. Deep tail accuracy achieved via asymptotic expansion `GELU(x) ≈ -φ(x)·(1 - 1/x² + 3/x⁴ - 15/x⁶)` and erfc-based reference to avoid catastrophic cancellation.

### Complete Results Table (All 38 Methods)

Sorted by Max ULP. Region definitions: **nz** = near_zero (|x| < 0.5), **cp** = core_pos (0.5 ≤ x < 3), **cn** = core_neg (-3 ≤ x < -0.5), **tn** = tail_neg (x < -3).

> **Note**: The tail_pos region achieves **exactly 0 ULP for all methods** because for bf16 precision, GELU(x) = x·Φ(x) becomes indistinguishable from x when Φ(x) rounds to 1.0. Exhaustive enumeration shows the exact saturation threshold is **x ≥ 2.78125** (implementations use x ≥ 3 conservatively).

| Method | *Mean* | Max | *nz Mean* | nz Max | *cp Mean* | cp Max | *cn Mean* | cn Max | *tn Mean* | tn Max |
|--------|--------|-----|-----------|--------|-----------|--------|-----------|--------|-----------|--------|
| **R5 Pure** | ***0.002*** | **2** | *0.00* | 1 | *0.00* | 0 | *0.03* | 1 | *0.00* | 2 |
| **B3 Pure** | ***0.01*** | **23** | *0.00* | 0 | *0.04* | 1 | *2.03* | 23 | *0.00* | 2 |
| **D2 Pure** | ***0.01*** | **23** | *0.00* | 0 | *0.04* | 1 | *2.03* | 23 | *0.00* | 2 |
| **F3 Pure** | ***0.02*** | **28** | *0.00* | 0 | *0.40* | 3 | *4.16* | 28 | *0.00* | 2 |
| **R4 Pure** | ***0.01*** | **29** | *0.00* | 1 | *0.03* | 1 | *1.75* | 29 | *0.00* | 2 |
| **C1 Pure** | ***0.03*** | **35** | *0.00* | 1 | *0.07* | 1 | *4.09* | 12 | *0.03* | 35 |
| E4 Hermite | *0.05* | 58 | *0.00* | 0 | *0.04* | 1 | *2.03* | 23 | *0.14* | 58 |
| E4v3 Quintic | *0.04* | 61 | *0.00* | 0 | *0.04* | 1 | *2.03* | 23 | *0.13* | 61 |
| E4v2 Wide | *0.08* | 81 | *0.00* | 0 | *0.04* | 1 | *1.72* | 17 | *0.30* | 81 |
| R5 LUT | *0.07* | 87 | *0.00* | 1 | *0.00* | 0 | *0.03* | 1 | *0.29* | 87 |
| B3 Erf Poly | *0.11* | 87 | *0.00* | 0 | *0.04* | 1 | *2.03* | 23 | *0.40* | 87 |
| C1 Spline | *0.10* | 87 | *0.00* | 1 | *0.07* | 1 | *4.09* | 12 | *0.31* | 87 |
| D2 LUT+Erf | *0.11* | 87 | *0.00* | 0 | *0.04* | 1 | *2.03* | 23 | *0.40* | 87 |
| F2 Quadrature | *0.11* | 87 | *0.00* | 0 | *0.12* | 1 | *1.94* | 23 | *0.40* | 87 |
| F3 CF Erf | *0.12* | 87 | *0.00* | 0 | *0.40* | 3 | *3.90* | 23 | *0.40* | 87 |
| D4 Non-uniform | *0.60* | 88 | *0.74* | 88 | *1.45* | 7 | *29.16* | 62 | *0.35* | 87 |
| R4 Tanh | *0.11* | 166 | *0.00* | 1 | *0.03* | 1 | *1.75* | 29 | *0.42* | 166 |
| B4 Rational | *0.56* | 535 | *0.00* | 1 | *2.07* | 5 | *54.32* | 314 | *1.13* | 535 |
| B1v2 Sigmoid | *1.47* | 625 | *0.93* | 106 | *13.91* | 40 | *129.44* | 352 | *1.25* | 625 |
| B1 Pure | *1.12* | 759 | *0.50* | 35 | *10.81* | 24 | *166.64* | 759 | *0.00* | 2 |
| D3 LUT+Corr | *1.82* | 830 | *0.82* | 94 | *23.27* | 53 | *185.06* | 440 | *1.55* | 830 |
| R3 PWL | *36.82* | 832 | *72.59* | 98 | *1.50* | 6 | *112.68* | 518 | *1.58* | 832 |
| H3 Pure | *0.67* | 861 | *0.00* | 1 | *4.96* | 28 | *130.29* | 861 | *0.00* | 2 |
| C2 Piecewise | *1.15* | 881 | *0.00* | 1 | *4.35* | 27 | *135.89* | 881 | *1.86* | 870 |
| B1 Sigmoid | *1.64* | 1068 | *0.50* | 35 | *10.81* | 24 | *166.64* | 759 | *2.10* | 1068 |
| E3 Range-Scale | *1.72* | 1130 | *0.07* | 21 | *5.98* | 15 | *225.87* | 822 | *2.22* | 1130 |
| H2 GELU-Softmax | *1.31* | 1130 | *0.25* | 18 | *1.91* | 4 | *126.82* | 822 | *2.22* | 1130 |
| R2 Rational | *1.19* | 1139 | *0.00* | 1 | *5.02* | 17 | *126.52* | 775 | *2.18* | 1139 |
| A4 Cont.Frac | *1.71* | 1206 | *0.02* | 7 | *18.07* | 36 | *208.65* | 866 | *2.34* | 1206 |
| A3 Chebyshev | *2.54* | 1207 | *0.57* | 62 | *47.73* | 65 | *292.26* | 900 | *2.37* | 1207 |
| A1 Pure | *2.83* | 1211 | *0.88* | 122 | *10.87* | 29 | *475.34* | 1211 | *0.00* | 2 |
| E9 Remez BF16 | *2.83* | 1211 | *0.88* | 122 | *10.80* | 28 | *476.64* | 1211 | *0.00* | 2 |
| H3 SoftEx | *1.26* | 1247 | *0.00* | 1 | *4.96* | 28 | *130.29* | 861 | *2.38* | 1247 |
| R1 Poly-9 | *1.40* | 1312 | *0.00* | 1 | *7.38* | 40 | *150.88* | 926 | *2.50* | 1312 |
| A1 Poly-9 | *3.56* | 1404 | *0.88* | 122 | *10.87* | 29 | *475.34* | 1211 | *2.94* | 1404 |
| A1 Poly-7 | *3.58* | 1547 | *0.88* | 122 | *10.81* | 28 | *476.75* | 1211 | *3.02* | 1547 |
| **TT Accurate*** | *3224.82* | 14330 | *6482.37* | 14330 | *0.03* | 1 | *0.39* | 4 | *87.73* | 13245 |
| **TT Fast*** | *15782.90* | 32639 | *19710.55* | 28802 | *0.64* | 5 | *477.71* | 1211 | *24357.40* | 32639 |

*\*TT Accurate/Fast are Tenstorrent hardware reference benchmarks (not optimized for full bf16 range)*

### Key Observations

1. **tail_pos is trivial**: All methods achieve 0 ULP (exact saturation at x ≥ 2.78125, implementations use x ≥ 3)
2. **core_neg is the bottleneck**: Most high-ULP methods fail at x ≈ -3.5 (TAIL_START boundary)
3. **Six Pure methods achieve Max ULP ≤ 35**: R5 Pure (**2**, Mean 0.002), B3 Pure (23, Mean 0.01), D2 Pure (23, Mean 0.01), F3 Pure (28, Mean 0.02), R4 Pure (29, Mean 0.01), C1 Pure (35, Mean 0.03)
4. **Pure methods eliminate shared tail dependency**: All Pure methods use independent asymptotic expansion for deep tail, achieving Max ULP = 2-35 vs 87 for shared-tail versions
5. **LUT-based methods plateau at 87**: Shared tail LUT limited by interpolation error at x ≈ -7.65; Pure versions avoid this via asymptotic tail
6. **E4 Hermite blending achieves Max ULP 58**: Smooth transition between polynomial core and asymptotic tail reduces discontinuity error

38 methods implemented: 36 research methods (23 original + 9 Pure variants + 4 engineering variants) across 8 categories from FinalLists.md taxonomy, plus 2 Tenstorrent hardware reference benchmarks.

## Background

### GELU Activation Function

The Gaussian Error Linear Unit (GELU) is defined as:

```
GELU(x) = x · Φ(x)

where Φ(x) = 0.5 × (1 + erf(x/√2))
```

GELU is widely used in transformer architectures (BERT, GPT, etc.) as a smooth alternative to ReLU. The challenge is approximating it efficiently without transcendental functions.

### BFloat16 Format

BFloat16 (Brain Floating Point) uses 16 bits:
- 1 sign bit
- 8 exponent bits (same as float32)
- 7 mantissa bits (vs 23 in float32)

This provides the same dynamic range as float32 but with reduced precision, making it ideal for ML training where memory bandwidth often bottlenecks performance.

### ULP Error Metric

ULP (Units in Last Place) measures the distance between two floating-point values in terms of representable numbers:

```
ULP(a, b) = |index(a) - index(b)|
```

where `index(x)` assigns a sequential integer to each representable value. ULP is preferred over MSE because:
- It's scale-invariant across the dynamic range
- It directly measures precision loss in fixed-point terms
- It reveals worst-case errors that MSE averages away

## Installation

### Requirements

- **GCC 13+** with C++23 support
- Linux/WSL environment
- Standard C++ library with `<stdfloat>` header

### Build

```bash
# Clone the repository
git clone <repository-url>
cd bf16_gelu_research

# Build the main analysis tool
g++ -std=c++23 -O3 -march=native -o gelu_analysis gelu_implementations.cpp -lm

# Build the ULP calculator (standalone)
g++ -std=c++23 -O2 -o ulp_calculator ulp_calculator.cpp

# Verify bfloat16 support
g++ -std=c++23 -o test_bfloat16 test_bfloat16.cpp && ./test_bfloat16
```

### Cross-Platform Verification

Results are **identical** across different GCC versions and platforms:

| Platform | Compiler | Results |
|----------|----------|---------|
| WSL (Ubuntu 24.04) | GCC 13.3.0 | ✓ Verified |
| MinGW-w64 (MSYS2 UCRT64) | GCC 15.2.0 | ✓ Verified |

All 38 methods produce byte-for-byte identical ULP analysis output on both platforms (after normalizing line endings). This confirms:
- No compiler-specific floating-point behavior differences
- Consistent `std::bfloat16_t` implementation across GCC versions
- Deterministic results for reproducible research

**Note**: On Windows/MinGW, add `-D_USE_MATH_DEFINES` for M_PI definition.

## Usage

### Running Analysis

```bash
# Full ULP analysis over entire bfloat16 range (38 methods)
./gelu_analysis --analyze

# Diagnostic mode with specific test points
./gelu_analysis --diagnose

# Show reference GELU values at key points
./gelu_analysis --reference

# Analyze saturation boundary behavior
./gelu_analysis --saturation

# Compute correct tail LUT calibration values
./gelu_analysis --calibrate

# Run G7/G8 regression suite (50 adversarial test points)
./gelu_analysis --regression

# Test G4 backward pass (GELU derivative)
./gelu_analysis --derivative

# Debug C1 cubic spline knot values
./gelu_analysis --verify-knots

# E2: Analyze coefficient quantization to BF16
./gelu_analysis --quantization

# E6/G2: Compare FMA vs non-FMA evaluation (Horner vs Estrin)
./gelu_analysis --fma

# E7: Test coefficient sensitivity to perturbations
./gelu_analysis --sensitivity

# G5: Display cost model (ops count, vectorizability)
./gelu_analysis --cost-model

# C5: EPSS knot refinement analysis
./gelu_analysis --epss

# E3: Range-scaled approximation analysis
./gelu_analysis --range-scale

# E5/E8: Denormal and FTZ policy testing
./gelu_analysis --denormal

# H2: GELU-Softmax combined unit analysis
./gelu_analysis --softmax-unit

# ULP measurement sanity check (verify framework correctness)
./gelu_analysis --sanity

# Run all analysis modes
./gelu_analysis --all
```

### Example Output

```
================================================================
   GELU Approximation ULP Analysis - Entire BFloat16 Range
================================================================

Building ULP lookup table...
Done! Valid bfloat16 values: 65280

--- B3 Pure: Erf (no LUT) ---
Samples:    65280
Max ULP:    33
Mean ULP:   0.0119
P50 ULP:    0
P90 ULP:    0
P99 ULP:    0
Worst input: -13.2500 (0xc154)

  Region Analysis (G3):
  +---------------+-------+----------+-----------+
  | Region        | Count | Max ULP  | Mean ULP  |
  +---------------+-------+----------+-----------+
  | near_zero     | 32256 |        0 |      0.00 |
  | core_pos      |   320 |        1 |      0.04 |
  | core_neg      |   321 |       23 |      2.03 |
  | tail_pos      | 16192 |        0 |      0.00 |
  | tail_neg      | 16191 |       33 |      0.01 |
  +---------------+-------+----------+-----------+
```

## Implemented Approximations

### Approximation Constraints

**Hardware Motivation**: These constraints target **Tenstorrent Wormhole/Blackhole ML accelerators**, where the tensor core (FPU/Tensix) supports only basic arithmetic operations. Transcendental functions like `exp()`, `erf()`, and `tanh()` require the slower SFPU (Special Function Processing Unit) or must be approximated using basic ops. Efficient GELU implementation must avoid SFPU calls entirely.

All implementations avoid standard library transcendental function calls:
- **Allowed**: `+`, `-`, `*`, `/`, `|x|`, `sign()`, comparison, bit manipulation, polynomial evaluation
- **Prohibited**: `std::erf()`, `std::tanh()`, `std::exp()`, `std::log()` (except in reference implementation)

**Arithmetic-Only Policy Clarification**: The "Pure" methods use `fast_exp_neg()` for the asymptotic tail, which computes `exp(-u)` via IEEE754 bit manipulation and polynomial refinement—not via `std::exp()`. This technique maps efficiently to hardware multipliers and integer ALUs, avoiding expensive SFPU transcendental operations. See: Schraudolph, N.N. (1999) "A Fast, Compact Approximation of the Exponential Function" Neural Computation 11(4), 853-862.

### Mathematical Foundation

The GELU function and its key properties:

```
GELU(x) = x · Φ(x)

where Φ(x) = ½(1 + erf(x/√2))  is the CDF of standard normal distribution

Key properties:
  • Φ(0) = 0.5, so GELU(0) = 0
  • Φ(x) → 1 as x → +∞, so GELU(x) → x
  • Φ(x) → 0 as x → -∞, so GELU(x) → 0
  • GELU has a minimum at x ≈ -0.75 where GELU ≈ -0.17
  • GELU'(x) = Φ(x) + x·φ(x), where φ(x) = exp(-x²/2)/√(2π)
```

The approximation challenge is computing `erf(z)` without transcendental functions.

---

### Methods

#### B3 Pure: Erf Approximation with Asymptotic Tail ⭐ Best Method

**Mathematical basis**: Approximate `erf(z)` piecewise, then use asymptotic expansion for deep tail.

**Core region (|x| ≤ 8.3125)**:
```
For z = x/√2:

If |z| < 1:  Use Taylor series
  erf(z) ≈ (2/√π) · z · (1 - z²/3 + z⁴/10 - z⁶/42 + z⁸/216)

If |z| ≥ 1:  Use Abramowitz-Stegun rational (7.1.26)
  erf(z) ≈ 1 - (a₁t + a₂t² + a₃t³ + a₄t⁴) · exp(-z²)
  where t = 1/(1 + p|z|), p = 0.3275911

Then: Φ(x) = 0.5 · (1 + erf(z))   for x ≥ 0
      Φ(x) = 0.5 · erfc(-z)       for x < 0  (avoids cancellation)
```

**Deep tail (x < -8.3125)**: Asymptotic expansion derived from Mills ratio:
```
For large negative x, Φ(x) has the asymptotic expansion:
  Φ(x) ≈ φ(x)/|x| · (1 - 1/x² + 3/x⁴ - 15/x⁶ + 105/x⁸ - ...)

Since GELU(x) = x·Φ(x) and x < 0:
  GELU(x) ≈ -φ(x) · (1 - 1/x² + 3/x⁴ - 15/x⁶)

where φ(x) = exp(-x²/2) / √(2π) is computed via:
  exp(-u) = 2^(-u/ln2) using IEEE754 bit manipulation
```

**Why it works**: This Poincaré-type asymptotic series provides rapidly decreasing term magnitudes for |x| > 3, with each successive term O(1/x²) smaller. Truncating at x⁻⁶ gives sufficient precision for bf16's 7-bit mantissa.

**Result**: Mean ULP 0.01, Max ULP 33. No LUT required.

---

#### R5: LUT + Linear Interpolation

**Mathematical basis**: Precompute exact Φ(x) values, interpolate between them.

```
512-entry table storing Φ(xᵢ) for xᵢ ∈ [-9, 3] at step 0.0234375

For input x:
  1. Find interval: xᵢ ≤ x < xᵢ₊₁
  2. Compute fraction: t = (x - xᵢ) / (xᵢ₊₁ - xᵢ)
  3. Interpolate: Φ(x) ≈ (1-t)·Φ(xᵢ) + t·Φ(xᵢ₊₁)
  4. Return: GELU(x) = x · Φ(x)

For x < -3.5: Use two-tier tail LUT (see Tail Handling section)
```

**Why it works**: Linear interpolation error is O(h²·f'') where h is step size. For smooth Φ(x), this gives excellent accuracy with modest memory.

**Result**: Mean ULP 0.07, Max ULP 87 (limited by tail LUT interpolation).

---

#### R5 Pure: LUT + Asymptotic Tail ⭐ Best Mean ULP

**Mathematical basis**: Same as R5 LUT but uses asymptotic expansion for deep tail instead of tail LUT.

```
Core region [-3, 3]: 512-entry LUT with linear interpolation (same as R5 LUT)

Negative tail (x < -3): Asymptotic expansion
  GELU(x) ≈ -φ(x) · (1 - 1/x² + 3/x⁴ - 15/x⁶)
  where φ(x) = exp(-x²/2) / √(2π)

Positive saturation (x ≥ 3): GELU(x) = x
```

**Why it works**: The asymptotic series provides rapidly decreasing term magnitudes for |x| > 3, avoiding the interpolation error that limits R5 LUT to Max ULP 87. Combines the accuracy of LUT interpolation in the core region with the precision of analytical asymptotic expansion in the tail.

**Result**: Mean ULP 0.003, Max ULP 33. Ties with B3 Pure for best Max ULP but has 3× better Mean ULP.

---

#### C1: Cubic Hermite Spline

**Mathematical basis**: Piecewise cubic polynomials with continuous first derivatives.

```
Hermite cubic on interval [xᵢ, xᵢ₊₁]:
  H(t) = h₀₀(t)·yᵢ + h₁₀(t)·h·mᵢ + h₀₁(t)·yᵢ₊₁ + h₁₁(t)·h·mᵢ₊₁

where t = (x - xᵢ)/h, h = xᵢ₊₁ - xᵢ, and:
  h₀₀(t) = 2t³ - 3t² + 1      (value at left)
  h₁₀(t) = t³ - 2t² + t       (slope at left)
  h₀₁(t) = -2t³ + 3t²         (value at right)
  h₁₁(t) = t³ - t²            (slope at right)

Knots: {-4, -3, -2, -1, -0.5, 0, 0.5, 1, 2, 4}
Values yᵢ = GELU(xᵢ), slopes mᵢ = GELU'(xᵢ)
```

**Monotonicity preservation** (Fritsch-Carlson):
```
For monotone data, derivatives must satisfy:
  α² + β² ≤ 9  where α = m₀/δ, β = m₁/δ, δ = (y₁-y₀)/h

If violated, clamp: τ = 3/√(α² + β²), m₀ ← τ·α·δ, m₁ ← τ·β·δ
```

**Near-zero refinement**: For |x| < 0.125, use Taylor:
```
GELU(x) ≈ x · (0.5 + x/√(2π)) = 0.5x + 0.3989x²
```

**Result**: Mean ULP 0.10, Max ULP 87.

---

#### C1 Pure: Cubic Hermite Spline with Asymptotic Tail ⭐

**Mathematical basis**: Same as C1 but with independent asymptotic tail handling.

**Difference from C1**: Replaces shared tail LUT with direct asymptotic expansion for x < -3.5:
```
GELU(x) ≈ -φ(x) · (1 - 1/x² + 3/x⁴ - 15/x⁶)
where φ(x) = exp(-x²/2) / √(2π)
```

**Why Pure**: Eliminates shared implementation dependency, achieving methodological independence. The asymptotic expansion avoids LUT interpolation error that limits original C1 to Max ULP = 87.

**Result**: Mean ULP 0.03, Max ULP 35 (improved from 87).

---

#### R4: Tanh-Form GELU with Rational Tanh

**Mathematical basis**: Hendrycks & Gimpel's tanh approximation of GELU.

```
GELU(x) ≈ 0.5x · (1 + tanh(√(2/π) · (x + 0.044715x³)))

Let z = √(2/π) · (x + 0.044715x³) ≈ 0.7978x + 0.0356x³

Approximate tanh(z) via [3,3] Padé:
  tanh(z) ≈ z · (a₀ + a₁z² + a₂z⁴ + a₃z⁶) / (b₀ + b₁z² + b₂z⁴ + b₃z⁶)

Coefficients derived from tanh Taylor: tanh(z) = z - z³/3 + 2z⁵/15 - ...
```

**Why the cubic term**: The 0.044715x³ correction improves accuracy near x = ±1 where the tanh form otherwise deviates from true GELU.

**Result**: Mean ULP 0.11, Max ULP 166 (error peaks at x ≈ -3.5 boundary).

---

#### B1/B1v2: Sigmoid-Based GELU

**Mathematical basis**: GELU resembles x·σ(kx) for appropriate k and sigmoid σ.

```
Observation: Φ(x) ≈ σ(1.702x) where σ is the logistic sigmoid

B1 (simple rational sigmoid):
  σ(z) ≈ 0.5 + z / (2(1 + |z|))
  GELU(x) ≈ x · σ(1.702x)

B1v2 (quadratic sigmoid):
  σ(z) ≈ 0.5 + 0.5z / √(1 + z²)
  GELU(x) ≈ x · σ(1.702x)
```

**Derivation of k=1.702**: Minimize ∫|Φ(x) - σ(kx)|² dx over typical range.

**Why B1v2 is better**: The √(1+z²) form has the correct asymptotic behavior σ(z) → 1 as z → ∞, while the simple rational saturates more slowly.

**Result**: B1 Mean 1.64, Max 1068; B1v2 Mean 1.47, Max 625.

---

#### A3: Chebyshev Polynomial Approximation

**Mathematical basis**: Chebyshev polynomials minimize maximum error (near-minimax).

```
Map x ∈ [-4, 4] to u ∈ [-1, 1]: u = x/4

Chebyshev expansion of Φ(4u):
  Φ(4u) ≈ Σₙ cₙ · Tₙ(u)

where Tₙ are Chebyshev polynomials: T₀=1, T₁=u, Tₙ₊₁ = 2u·Tₙ - Tₙ₋₁

Clenshaw recurrence for stable evaluation:
  bₙ₊₁ = bₙ₊₂ = 0
  bₖ = cₖ + 2u·bₖ₊₁ - bₖ₊₂  (for k = n, n-1, ..., 0)
  result = b₀ - u·b₁
```

**Why Chebyshev**: The equioscillation theorem guarantees Chebyshev expansion is within factor 2 of the true minimax polynomial. Error oscillates uniformly rather than growing at boundaries.

**Result**: Mean ULP 2.54, Max ULP 1207.

---

#### A4/F3: Continued Fraction Methods

**Mathematical basis**: Continued fractions often converge faster than power series.

```
A4 - Direct CF for Φ(x):
  Φ(x) ≈ 0.5 + x · a₀/(1 + x²·a₁/(1 + x²·a₂/(1 + ...)))

F3 - CF for erf(z) (Lentz algorithm):
  erf(z) = (2z/√π) · 1/(1 + z²·1/(3 + z²·2/(5 + z²·3/(7 + ...))))

  Convergents computed iteratively:
    fₙ = fₙ₋₁ · Dₙ · Cₙ
  where Cₙ, Dₙ are correction factors from partial numerators/denominators
```

**Convergence**: CF converges in the cut plane, complementing power series which converge in disks. For erf, CF is effective for |z| > 0.5.

**Result**: A4 Mean 1.71, Max 1206; F3 Mean 0.12, Max 87.

---

#### F3 Pure: Continued Fraction Erf with Asymptotic Tail ⭐

**Mathematical basis**: Same CF-based erf as F3 but with independent asymptotic tail.

**Difference from F3**: Replaces shared tail LUT/fallback with direct asymptotic expansion for x < -2.0:
```
GELU(x) ≈ -φ(x) · (1 - 1/x² + 3/x⁴ - 15/x⁶)
where φ(x) = exp(-x²/2) / √(2π)
```

**Why Pure**: The continued fraction approach is mathematically elegant and deserves independent implementation. F3 Pure eliminates reliance on B3-style erf fallback, maintaining pure CF methodology throughout.

**Result**: Mean ULP 0.02, Max ULP 33 (improved from 87).

---

#### R2: Rational Padé Approximation

**Mathematical basis**: Padé approximants match more Taylor terms than polynomials of same degree.

```
Approximate (Φ(x) - 0.5)/x as rational function in x²:

  (Φ(x) - 0.5)/x ≈ P(x²)/Q(x²)

where P(u) = p₀ + p₁u + p₂u² + p₃u³
      Q(u) = 1 + q₁u + q₂u² + q₃u³

Coefficients from matching Taylor: Φ(x) = 0.5 + x/√(2π) - x³/(6√(2π)) + ...
```

**Why rational**: Near x=0, both polynomial and rational match Taylor. As |x| → ∞, rational P/Q → p₃/q₃ (constant), better matching Φ → 0 or 1 than polynomial which diverges.

**Result**: Mean ULP 1.19, Max ULP 1139.

---

#### D4: Non-uniform LUT with Variable Spacing

**Mathematical basis**: Concentrate breakpoints where function curvature is highest.

```
Breakpoints: {-4, -3, -2.5, -2, -1.5, -1, -0.75, -0.5, -0.25, 0,
              0.25, 0.5, 0.75, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 8}

Spacing rationale:
  • Dense near x=0: GELU'' is large, linear interpolation error ∝ h²·f''
  • Dense near x=-1: GELU has inflection point
  • Sparse for |x|>3: GELU is nearly linear (saturating)

Near-zero fix: For |x| < 0.125, use Taylor instead of LUT
  GELU(x) ≈ 0.5x + 0.3989x²
```

**Result**: Mean ULP 0.60, Max ULP 88.

---

#### F2: Gaussian Quadrature

**Mathematical basis**: Numerically integrate the GELU definition.

```
GELU(x) = x · Φ(x) = x · ∫_{-∞}^{x} φ(t) dt

Transform to finite interval [0,1] via t = x - s/(1-s):
  Φ(x) = ∫₀¹ φ(x - s/(1-s)) · 1/(1-s)² ds

Apply 8-point Gauss-Legendre quadrature:
  Φ(x) ≈ Σᵢ wᵢ · φ(x - sᵢ/(1-sᵢ)) · 1/(1-sᵢ)²

where (sᵢ, wᵢ) are Gauss-Legendre nodes and weights on [0,1]
```

**Limitation**: Requires exp(-t²/2) evaluation. Uses B3-style erf fallback for x < -2 where exp approximation fails.

**Result**: Mean ULP 0.11, Max ULP 87.

---

#### H2: GELU-Softmax Combined Unit

**Mathematical basis**: Share exp() approximation between GELU and softmax.

```
Both GELU (via tanh) and softmax need exp():
  tanh(z) = (e^{2z} - 1)/(e^{2z} + 1)
  softmax(xᵢ) = e^{xᵢ} / Σⱼ e^{xⱼ}

PWL exp approximation (8 segments on [-4, 4]):
  exp(x) ≈ mᵢ·x + bᵢ  for x ∈ [xᵢ, xᵢ₊₁]

Hardware benefit: Same multiply-add units compute both activations
```

**Result**: Mean ULP 1.31, Max ULP 1130.

---

#### H1: Inverted GELU (Newton-Raphson)

**Mathematical basis**: Given y, find x such that GELU(x) = y.

```
Newton-Raphson iteration:
  xₙ₊₁ = xₙ - (GELU(xₙ) - y) / GELU'(xₙ)

where GELU'(x) = Φ(x) + x·φ(x)

Initial guess:
  x₀ = y           if y ≥ 0
  x₀ = y - 0.5     if y < 0  (accounts for GELU minimum)

Convergence: 4-6 iterations for bf16 precision
```

**Application**: Memory-efficient backpropagation - store y = GELU(x) instead of x, recover x when needed.

---

#### R1: Saturation + Polynomial Core

**Mathematical basis**: Use identity/zero for saturated regions, polynomial for core.

```
Saturation thresholds:
  x ≥ 3:   GELU(x) = x        (Φ(3) ≈ 0.99865, close enough to 1)
  x ≤ -9:  GELU(x) = 0        (Φ(-9) ≈ 10⁻²⁰, effectively 0)

Core region (-9 < x < 3):
  Φ(x) ≈ 0.5 + x·(a₁ + a₃x² + a₅x⁴ + a₇x⁶ + a₉x⁸)

Odd polynomial form exploits Φ(x) - 0.5 being an odd function.
Coefficients fitted via minimax optimization over [-4, 4].
```

**Result**: Mean ULP 1.40, Max ULP 1312.

---

#### R3: Piecewise Linear (PWL)

**Mathematical basis**: Simplest approximation - connect points with straight lines.

```
Breakpoints at power-of-2 values: {-4, -2, -1, -0.5, 0, 0.5, 1, 2, 4}

For each segment [xᵢ, xᵢ₊₁]:
  GELU(x) ≈ GELU(xᵢ) + slope · (x - xᵢ)
  where slope = (GELU(xᵢ₊₁) - GELU(xᵢ)) / (xᵢ₊₁ - xᵢ)

Power-of-2 breakpoints enable efficient segment lookup via exponent bits.
```

**Trade-off**: Very fast (1 multiply + 1 add per segment) but inherently limited accuracy since GELU is curved.

**Result**: Mean ULP 36.82, Max ULP 832.

---

#### B3: Erf Polynomial (with Tail LUT)

**Mathematical basis**: Same as B3 Pure but uses LUT for tail instead of asymptotic.

```
Core: Piecewise erf approximation (Taylor + A-S rational)
Tail: For x < -3.5, use precomputed tail LUT with linear interpolation
```

**Difference from B3 Pure**: Relies on LUT for deep tail, limiting Max ULP to 87 (interpolation error) vs 33 (asymptotic).

**Result**: Mean ULP 0.11, Max ULP 87.

---

#### B4: Rational Erf with Range Reduction

**Mathematical basis**: Different rational approximations for different |z| ranges.

```
For z = x/√2:

If |z| < 1:
  erf(z) ≈ z · (p₀ + p₁z² + p₂z⁴) / (1 + q₁z² + q₂z⁴)

If |z| ≥ 1:
  erf(z) ≈ sign(z) · (1 - R(|z|) · exp(-z²))
  where R(z) is Abramowitz-Stegun rational
```

**Why range reduction**: Near z=0, erf is smooth and polynomial-like. For |z| > 1, erf approaches ±1 exponentially, needing different form.

**Result**: Mean ULP 0.56, Max ULP 535.

---

#### C2: Piecewise Rational

**Mathematical basis**: Different Padé approximants per region.

```
Segments: [-3.5, -1.5], [-1.5, 0], [0, 1.5], [1.5, 3.5]

Each segment uses [2/2] Padé:
  Φ(x) ≈ (p₀ + p₁x + p₂x²) / (1 + q₁x + q₂x²)

Coefficients fitted separately per segment for local accuracy.
Boundary continuity enforced by matching function values at knots.
```

**Advantage over polynomial**: Rational forms better capture asymptotic behavior (Φ → 0 or 1) than polynomials which diverge.

**Result**: Mean ULP 1.15, Max ULP 881.

---

#### D2: LUT Tails + B3-style Erf Center

**Mathematical basis**: Hybrid - LUT where needed, arithmetic where effective.

```
Region handling:
  x ≥ 3:      Return x (positive saturation)
  x < -3.5:   Use tail LUT with linear interpolation
  |x| ≤ 3.5:  Use B3-style piecewise erf (Taylor + A-S rational)
```

**Rationale**: The B3 erf approximation is accurate in core but fails in deep tail. LUT is accurate but memory-expensive. Combine strengths of both.

**Result**: Mean ULP 0.11, Max ULP 87.

---

#### D2 Pure: Hybrid LUT+Erf with Asymptotic Tail ⭐

**Mathematical basis**: Same hybrid approach as D2 but with independent asymptotic tail.

**Difference from D2**: Replaces shared tail LUT with direct asymptotic expansion for x < -3.0:
```
GELU(x) ≈ -φ(x) · (1 - 1/x² + 3/x⁴ - 15/x⁶)
where φ(x) = exp(-x²/2) / √(2π)
```

**Why Pure**: The original D2 hybrid strategy (LUT for tail, erf for core) is sound, but shared tail LUT creates implementation dependency. D2 Pure maintains the hybrid philosophy while achieving complete methodological independence.

**Result**: Mean ULP 0.01, Max ULP 33 (improved from 87).

---

#### D3: LUT + Polynomial Correction

**Mathematical basis**: Coarse LUT with polynomial refinement.

```
32-entry LUT at 0.25 step from -4 to +4

For input x in [xᵢ, xᵢ₊₁]:
  base = lerp(LUT[i], LUT[i+1], t)     // Linear interpolation
  correction = c₁·δ + c₂·δ²            // Polynomial correction
  result = base + correction

where δ = x - (xᵢ + xᵢ₊₁)/2 (distance from segment center)
```

**Trade-off**: Less memory than fine LUT, more computation than pure LUT.

**Result**: Mean ULP 1.82, Max ULP 830.

---

#### E3: Range-Scaled Approximation

**Mathematical basis**: Scale input to reduce coefficient magnitude.

```
Scale factor s = 2 (aligned with bf16 exponent boundaries)

Compute GELU(x) as:
  For x ≥ 0: Use polynomial in (x/s)
  For x < 0: Use B3-style erf fallback

Polynomial: Φ(x) ≈ 0.5 + (x/s)·P((x/s)²) where P has smaller coefficients
```

**Purpose**: When coefficients span many orders of magnitude, bf16 quantization loses precision. Scaling reduces dynamic range.

**Result**: Mean ULP 1.72, Max ULP 1130.

---

#### H3: SoftEx Tanh

**Mathematical basis**: Approximate exp() to enable tanh-form GELU.

```
Padé [2/2] approximation for exp(x):
  exp(x) ≈ (1 + x/2 + x²/12) / (1 - x/2 + x²/12)

Then compute tanh via:
  tanh(z) = (exp(2z) - 1) / (exp(2z) + 1)

And GELU via Hendrycks form:
  GELU(x) ≈ 0.5x · (1 + tanh(0.7978·(x + 0.0356x³)))
```

**Limitation**: Padé exp is accurate for |x| < 2 but diverges beyond. Uses fallback for |x| > 2.

**Result**: Mean ULP 1.26, Max ULP 1247.

---

### Tenstorrent Hardware Reference Benchmarks

**These are REFERENCE IMPLEMENTATIONS ONLY** - exact reproductions of Tenstorrent Wormhole/Blackhole hardware GELU for precision comparison. They are NOT optimized for the full bf16 range and should NOT be modified.

#### TT Accurate: Chebyshev-15
```
15th-degree Chebyshev polynomial (default mode in tt-train)
- x == 0: return 0
- x >= 3: return x (identity saturation)
- x < -5.5: return 0 (negative saturation)
- Otherwise: sign(x) * |POLYVAL15(coefficients, x)|
```
Designed for typical activation range [-5.5, 3]. Excellent in core regions (Max ULP 1-4), but has "floor value bug" where tiny inputs (~1e-38 to ~1e-10) return constant ~2.98e-05 due to c0 coefficient dominating.

**Source**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_gelu.h`

#### TT Fast: 6-Piece PWL
```
6-segment piecewise linear LUT (fast/approximate mode)
GELU(x) = 0.5*x + sign(x) * (A[segment]*|x| + B[segment])

Segments by |x|: [0,0.5), [0.5,1), [1,1.5), [1.5,2), [2,3), [3,∞)
```
Fast approximation using 6 linear segments. Good in core_pos (Max ULP 5), but no negative saturation handling - returns x instead of ~0 for large negative inputs.

**Source**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_gelu.h`

### Tail Handling

Methods use either **LUT-based** or **asymptotic** tail handling:

#### LUT-Based (Most methods)
Extended tail LUT for x ∈ [-8.3125, -3.5]:

```cpp
namespace tail_lut {
    constexpr float GELU_N3_50 = -8.14202e-04f;   // x = -3.50
    constexpr float GELU_N3_75 = -3.31556e-04f;   // x = -3.75 (calibrated)
    // ... 19 entries at 0.25 step from -3.5 to -8.0 ...
    // ... 6 fine entries at 0.0625 step from -8.0 to -8.3125 ...
    constexpr float GELU_N8_25 = -4.57967e-16f;   // x = -8.25
    constexpr float GELU_N8_3125 = -4.61436e-16f; // x = -8.3125, last non-zero bf16
}
```
Max ULP = 87 (limited by linear interpolation in exponential decay region).

#### Asymptotic (B3 Pure - Best Method)
For x < -8.3125, use the asymptotic expansion:

```cpp
// GELU(x) ≈ -φ(x) * (1 - 1/x² + 3/x⁴ - 15/x⁶)
// where φ(x) = exp(-x²/2) / √(2π)
float gelu_asymptotic(float x) {
    float x2 = x * x;
    float exp_val = fast_exp_neg(x2 * 0.5f);  // via 2^x bit manipulation
    float phi_x = exp_val * INV_SQRT_2PI;
    float inv_x2 = 1.0f / x2;
    float correction = 1.0f - inv_x2 + 3.0f*inv_x2*inv_x2 - 15.0f*inv_x2*inv_x2*inv_x2;
    return -phi_x * correction;
}
```
**Max ULP = 33** (best overall). No LUT required.

Note: The asymptotic expansion covers x ∈ [-13.5625, -8.3125] where bf16 GELU values are tiny subnormals. At x ≤ -13.5625, bf16(GELU(x)) rounds to exactly 0.

## Project Structure

| File | Description |
|------|-------------|
| `gelu_implementations.cpp` | All GELU approximations + analysis framework |
| `ulp_calculator.cpp` | Standalone ULP calculator with lookup table |
| `saturation_analysis.cpp` | Standalone bf16 saturation threshold finder |
| `debug_tools.cpp` | Exploratory debugging tools for tail/exp analysis |
| `test_bfloat16.cpp` | Compiler bfloat16 support verification |
| `FinalLists.md` | Strategy taxonomy (40 methods, 8 categories) |
| `SATURATION.md` | Exact bf16 GELU saturation threshold analysis |
| `CLAUDE.md` | Development instructions and status |
| `HISTORY.md` | Development history and session notes |
| `README.md` | This file |

## Technical Details

### Saturation Thresholds

**Exact bf16 saturation thresholds** (determined by exhaustive enumeration of all 65,280 valid bf16 values):

| Tail | Exact Threshold | Hex | Condition |
|------|-----------------|-----|-----------|
| **Positive** | x ≥ **2.78125** | 0x4032 | bf16(GELU(x)) == x |
| **Negative** | x ≤ **-13.5625** | 0xc159 | bf16(GELU(x)) == 0 |

**Implementation thresholds** (conservative approximations used in code):
```
Positive: x ≥ 3   → GELU(x) = x   (margin for approximation error)
Negative: x ≤ -9  → handled by tail LUT/asymptotic
```

See [SATURATION.md](SATURATION.md) for detailed transition analysis and the standalone `saturation_analysis.cpp` tool.

### Multi-Region Analysis (G3)

The codebase divides the input range into regions for detailed error analysis:

| Region | Range | Description |
|--------|-------|-------------|
| near_zero | \|x\| < 0.5 | High relative sensitivity |
| core_pos | 0.5 ≤ x < 3 | Main positive operational range |
| core_neg | -3 ≤ x < -0.5 | Main negative operational range |
| tail_pos | x ≥ 3 | Positive saturation region |
| tail_neg | x < -3 | Negative decay/saturation region |

### Max ULP Analysis

After implementing asymptotic expansion for the deep tail, **B3 Pure achieves Max ULP = 33** (best overall). LUT-based methods achieve Max ULP = 87.

**Error sources by region:**

| Region | Best Max ULP | Cause |
|--------|-------------|-------|
| near_zero | 0 (B3 Pure) | Taylor series is exact near zero |
| core_pos | 0 (R5) | LUT interpolation is accurate |
| core_neg | 1 (R5) | LUT interpolation is accurate |
| tail_pos | 0 (all) | Saturation bf16-rounding exact |
| tail_neg | 33 (B3 Pure) | Asymptotic expansion matches reference |

**Remaining error sources:**
1. **x ≈ -7.65** (LUT methods): Linear interpolation doesn't match exponential decay (87 ULP)
2. **x ≈ -3.5** (polynomial methods): Transition boundary between core and tail handler (up to 1547 ULP)
3. **x ≈ -13.25** (Pure methods): Near the exact zero-saturation threshold (-13.5625), asymptotic series truncation matters (33 ULP)

## Methodology

### ULP Calculation

```cpp
int64_t ulp_distance(bfloat16_t a, bfloat16_t b) {
    // Build ordered index of all valid bf16 values
    // Return |index(a) - index(b)|
}
```

Key considerations:
- +0 and -0 share the same ULP index
- NaN and Inf values are excluded (65,280 valid values remain)
- Type punning uses `std::memcpy()` for defined behavior

### Reference Implementation

**ULP Measurement Chain:**
1. Calculate GELU(x) in float64 using erfc-based formula (below)
2. Round result to nearest bf16
3. Measure ULP distance between bf16 approximation and bf16 reference

Ground truth uses float64 with `erfc()` for negative x to avoid catastrophic cancellation:

```cpp
double gelu_reference_f64(double x) {
    double z = x * INV_SQRT_2;
    double phi;
    if (x >= 0) {
        phi = 0.5 * (1.0 + std::erf(z));
    } else {
        // Use erfc(-z) to avoid cancellation when erf(z) ≈ -1
        phi = 0.5 * std::erfc(-z);
    }
    return x * phi;
}
```

**Critical insight**: The naive formula `0.5 * (1 + erf(z))` suffers catastrophic cancellation for large negative z because `erf(z) → -1`, making `1 + erf(z) → 0`. Using `erfc(-z) = 1 - erf(-z) = 1 + erf(z)` avoids this since erfc is computed directly without subtraction.

## Strategy Taxonomy

Based on FinalLists.md, the project follows a phased implementation approach:

| Phase | Focus | Status |
|-------|-------|--------|
| 0 | Infrastructure (F1, G1, G3) | Complete |
| 1 | Core baselines (A1, A2, B1, B3, C4) | Complete |
| 2 | ULP control (C1, C3, B2) | Complete |
| 3 | BF16 optimizations (E1-E7) | Complete (E2, E6, E7) |
| 4 | Validation (G2, G4, G5, G7/G8) | Complete |
| 5 | Advanced (D2-D4, H1-H3) | Complete |

### Implementation Coverage

**Overall: 40/40 methods (100%)**

| Category | Methods | Implemented | Details |
|----------|---------|-------------|---------|
| A: Direct | 4 | 4/4 ✓ | A1 (poly-7,9), A2 ([4/4]), A3, A4 |
| B: Sub-function | 4 | 4/4 ✓ | B1, B1v2, B2/R4, B3, B4 |
| C: Piecewise | 5 | 5/5 ✓ | C1, C2, C3/R3, C4/R1, C5 (EPSS analysis) |
| D: Hybrid/LUT | 4 | 4/4 ✓ | D1/R5, D2, D3, D4 |
| E: BF16 Knobs | 8 | 8/8 ✓ | E1-E8 (E3 range-scale, E5/E8 denormal/FTZ) |
| F: Reference | 4 | 4/4 ✓ | F1, F2, F3, F4 (in B3) |
| G: Methodology | 8 | 8/8 ✓ | G1-G8 complete |
| H: Advanced | 3 | 3/3 ✓ | H1, H2 (GELU-Softmax), H3 |

### All Methods Fixed

All methods now achieve Max ULP ≤ 1547 (A1 Poly-7). **The top 16 methods achieve Max ULP ≤ 88**, with six Pure methods tied at Max ULP = 33.

**Key fixes applied:**
- **Reference function**: Use `erfc(-z)` for negative x to avoid catastrophic cancellation in `1 + erf(z)`
- **Deep tail**: Asymptotic expansion `GELU(x) ≈ -φ(x)·(1 - 1/x² + 3/x⁴)` for x < -8.3125
- **D2**: Replaced polynomial core with B3-style piecewise erf
- **D4**: Added Taylor approximation for |x| < 0.125
- **F2, F3**: Added B3-style erf fallback for x < -2

## Key Insights

1. **erfc() avoids catastrophic cancellation**: The naive `1 + erf(z)` formula fails for large negative z because erf(z) → -1. Use `erfc(-z)` for negative inputs.

2. **Asymptotic expansion beats LUT for deep tail**: For x < -8.3125, the formula `GELU(x) ≈ -φ(x)·(1 - 1/x² + 3/x⁴ - 15/x⁶)` achieves better accuracy than LUT interpolation (33 vs 87 ULP).

3. **Track bf16 underflow boundary**: At x ≈ -8.35, bf16 underflows to -0 (0x8000). The tail handler must transition before this point.

4. **core_neg is the new bottleneck**: After fixing the deep tail, most methods now fail at x ≈ -3.5 (the TAIL_START boundary) with Max ULP 500-1500.

5. **Monotonicity constraints**: Fritsch-Carlson condition (α² + β² ≤ 9) prevents unphysical artifacts in cubic spline interpolation.

6. **Asymmetric thresholds**: GELU approaches 0 slowly for negative x, requiring different handling than positive saturation.

7. **Entire range testing**: Methods optimized for [-8, 8] may fail catastrophically outside this range.

8. **Sixteen methods achieve Max ULP ≤ 88**: R5 Pure/B3 Pure/D2 Pure/F3 Pure/R4 Pure (33), C1 Pure (35), E4 Hermite (58), E4v3 (61), E4v2 (81), R5/C1/B3/D2/F2/F3 (87), D4 (88).

9. **B3 erf is the universal fallback**: When arithmetic-only exp() fails (|x| > 2), the B3 piecewise erf (Taylor + A-S rational) provides reliable fallback.

10. **Pure arithmetic can beat LUT**: B3 Pure (no LUT, Max ULP 33) outperforms all LUT-based methods (Max ULP 87) by using asymptotic expansion.

11. **100% taxonomy coverage**: All 40 methods from FinalLists.md are now implemented, including analysis functions for C5 (EPSS), E5/E8 (denormal/FTZ), and implementations for E3 (range-scaling) and H2 (GELU-Softmax).

## References

- Hendrycks, D., & Gimpel, K. (2016). Gaussian Error Linear Units (GELUs). arXiv:1606.08415
- Abramowitz, M., & Stegun, I. A. (1964). Handbook of Mathematical Functions, Section 7.1.26 (erf/erfc)
- Schraudolph, N. N. (1999). A Fast, Compact Approximation of the Exponential Function. Neural Computation, 11(4), 853-862
- Fritsch, F. N., & Carlson, R. E. (1980). Monotone Piecewise Cubic Interpolation. SIAM J. Numerical Analysis, 17(2), 238-246
- NIST Digital Library of Mathematical Functions, Chapter 7: Error Functions and Mill's Ratio. https://dlmf.nist.gov/7
- Higham, N. J. (2019). The Rise of bfloat16. SIAM News
- PyTorch Documentation: torch.nn.functional.gelu

## License

This project is provided for research and educational purposes.
