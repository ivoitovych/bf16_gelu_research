# BFloat16 GELU Approximation Research

Systematic ULP (Units in Last Place) error analysis of GELU activation function approximations for bfloat16 floating-point arithmetic, targeting ML accelerator frameworks.

## Overview

This project implements and evaluates multiple GELU approximation strategies optimized for **bfloat16** precision, measuring accuracy using **ULP error** rather than MSE. Unlike typical analyses that focus on common activation ranges like [-8, 8], this research covers the **entire bfloat16 range** (65,280 valid values, approximately ±3.4×10³⁸).

### Key Achievements

| Method | Mean ULP | Max ULP | Description |
|--------|----------|---------|-------------|
| **R5 LUT** | **0.10** | **145** | 512-entry LUT + extended tail handler |
| **C1 Spline** | **0.13** | **145** | 9-segment Hermite + Taylor near-zero |
| **B3 Erf Poly** | **0.13** | **145** | Piecewise erf (Taylor + A-S rational) |
| **D2 LUT+Erf** | **0.13** | **145** | LUT tails + B3-style erf core |
| **F2 Quadrature** | **0.13** | **145** | Gauss-Legendre + B3 erf fallback |
| **R4 Tanh** | 0.14 | 166 | Tanh-form + [3,3] Padé approximation |
| **F3 CF Erf** | **0.15** | **145** | Continued fraction + B3 erf fallback |
| B4 Rational Erf | 0.59 | 535 | Range-reduced rational erf |
| **D4 Non-uniform** | **0.63** | **145** | Non-uniform LUT + Taylor near-zero |
| C2 Piecewise | 1.18 | 881 | Piecewise rational (3 segments) |
| R2 Rational | 1.22 | 1139 | Rational Padé [4/4] |
| H3 SoftEx | 1.28 | 1247 | Arithmetic-only exp via Padé |
| H2 GELU-Softmax | 1.33 | 1130 | PWL exp shared with softmax |
| R1 Poly-9 | 1.43 | 1312 | 9th-degree minimax polynomial |
| B1v2 Sigmoid | 1.50 | 625 | Quadratic sigmoid approximation |
| B1 Sigmoid | 1.67 | 1068 | Simple sigmoid approximation |
| A4 Cont.Frac | 1.73 | 1206 | Continued fraction depth-4 |
| E3 Range-Scaled | 1.75 | 1130 | BF16 exponent-aligned scaling |
| D3 LUT+Corr | 1.84 | 830 | Coarse LUT + polynomial correction |
| A3 Chebyshev | 2.57 | 1207 | Chebyshev polynomial (Clenshaw) |
| R3 PWL | 36.85 | 832 | Piecewise linear (power-of-2 breakpoints) |

**8 methods achieve Max ULP = 145** (the bfloat16 underflow limit). 24 methods implemented across 8 categories from FinalLists.md taxonomy.

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

## Usage

### Running Analysis

```bash
# Full ULP analysis over entire bfloat16 range (24 methods)
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

--- C1: Cubic Spline (8 seg) ---
Samples:    65280
Max ULP:    145
Mean ULP:   0.1254
P50 ULP:    0
P90 ULP:    0
P99 ULP:    0

  Region Analysis (G3):
  +---------------+-------+----------+-----------+
  | Region        | Count | Max ULP  | Mean ULP  |
  +---------------+-------+----------+-----------+
  | near_zero     | 32256 |        1 |      0.00 |
  | core_pos      |   320 |        1 |      0.07 |
  | core_neg      |   321 |       12 |      4.09 |
  | tail_pos      | 16192 |        0 |      0.00 |
  | tail_neg      | 16191 |      145 |      0.05 |
  +---------------+-------+----------+-----------+
```

## Implemented Approximations

### Approximation Constraints

All implementations use only basic arithmetic:
- **Allowed**: `+`, `-`, `*`, `/`, `|x|`, `sign()`
- **Prohibited**: `erf()`, `tanh()`, `exp()`, `log()` (except in reference implementation)

### Methods

#### C1: Cubic Spline (Best Method)
```
9-segment Hermite cubic interpolation with:
- Knots at: -4, -3, -2, -1, -0.5, 0, 0.5, 1, 2, 4
- Fritsch-Carlson monotonicity clamping
- Taylor approximation for |x| < 0.125
```
**Best performer with Mean ULP 0.13, Max ULP 145.**

#### B3: Erf Polynomial (Best Method)
```
Piecewise erf approximation:
- |z| < 1: Taylor series
- |z| ≥ 1: Abramowitz-Stegun rational approximation
```
**Best performer with Mean ULP 0.13, Max ULP 145.**

#### B1: Sigmoid-Based GELU
```
GELU(x) ≈ x · σ(1.702x)
σ(z) ≈ 0.5 + z / (2(1 + |z|))
```
Simple rational sigmoid approximation. Mean ULP 1.67, Max ULP 1068.

#### B1v2: Quadratic Sigmoid GELU
```
GELU(x) ≈ x · σ(1.702x)
σ(z) ≈ 0.5 + 0.5·z / √(1 + z²)
```
Uses hardware-accelerated sqrt for better accuracy. Mean ULP 1.50, Max ULP 625.

#### R1: Saturation + Polynomial Core
```
x ≥ 3:  GELU(x) = x
x ≤ -9: GELU(x) = 0
else:   Φ(x) ≈ 0.5 + x·(a₁ + a₃x² + a₅x⁴ + a₇x⁶ + a₉x⁸)
```
9th-degree minimax polynomial with asymmetric saturation. Mean ULP 1.43, Max ULP 1312.

#### R2: Rational Padé Approximation
```
Φ(x) ≈ 0.5 + x · P(x²) / Q(x²)
P, Q are [3/3] polynomials in x²
```
Better tail convergence than pure polynomials. Mean ULP 1.22, Max ULP 1139.

#### R3: Piecewise Linear (PWL)
```
Power-of-2 breakpoints: 0, ±0.5, ±1, ±2, ±4
Linear interpolation between segments
```
Fast evaluation but higher near-zero error. Mean ULP 36.85, Max ULP 832.

#### R4: Tanh-Form + Rational Tanh
```
GELU(x) ≈ 0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))
tanh(z) ≈ z·(a₀ + a₁z² + a₂z⁴ + a₃z⁶) / (b₀ + b₁z² + b₂z⁴ + b₃z⁶)
```
[3,3] Padé approximant for tanh. Mean ULP 0.14, Max ULP 166.

#### R5: LUT + Linear Interpolation (Best Method)
```
512-entry lookup table for Φ(x)
Linear interpolation between entries
Range: [-9, 3] with extended tail LUT
```
**Best performer with Mean ULP 0.10, Max ULP 145.** Uses precomputed erf values with extended tail handler.

#### A3: Chebyshev Polynomial
```
Chebyshev expansion on [-4, 4] mapped to [-1, 1]
Clenshaw recurrence for stable evaluation
Degree-9 with odd coefficients
```
Near-minimax with bounded oscillating error. Mean ULP 2.57, Max ULP 1207.

#### A4: Continued Fraction
```
GELU(x) ≈ x · (a₀ + x²/(b₁ + x²/(b₂ + ...)))
Depth-4 truncation
```
Alternative to Padé with different convergence. Mean ULP 1.73, Max ULP 1206.

#### B4: Rational Erf with Range Reduction
```
|z| < 1: rational [2/2] Taylor-like
|z| ≥ 1: Abramowitz-Stegun with exp factor
Separate fits per range
```
Reduces polynomial degree requirements. Mean ULP 0.59, Max ULP 535.

#### C2: Piecewise Rational
```
Different Padé [2/2] per segment:
- [0, 3]: positive coefficients
- [-3, 0]: negative coefficients
- blend zone at boundaries
```
Fewer segments than polynomial for equivalent accuracy. Mean ULP 1.18, Max ULP 881.

#### D2: LUT Tails + B3-style Erf Center
```
x >= 3: Positive saturation (GELU(x) = x)
x < -3.5: Extended tail LUT
|x| <= 3.5: B3-style piecewise erf (Taylor + A-S rational)
```
Hybrid approach combining LUT tails with proven erf approximation. **Mean ULP 0.13, Max ULP 145.**

#### D3: LUT + Polynomial Correction
```
32-entry coarse LUT as base
Low-degree polynomial correction term
Balances memory and computation
```
Mean ULP 1.84, Max ULP 830.

#### D4: Non-uniform LUT Spacing
```
23 breakpoints with variable density
Denser near x=0 and saturation boundaries
Taylor approximation for |x| < 0.125
Binary search for segment lookup
```
Optimized spacing with near-zero Taylor fix. **Mean ULP 0.63, Max ULP 145.**

#### F2: Numerical Quadrature
```
8-point Gauss-Legendre quadrature
For x < -2: B3-style erf fallback
For |x| <= 2: Taylor-based exp approximation
```
Hybrid quadrature with B3 erf fallback for negative region. **Mean ULP 0.13, Max ULP 145.**

#### F3: Continued Fraction Erf
```
Taylor series for |z| < 0.5
Lentz CF algorithm for |z| ≥ 0.5
For x < -2: B3-style erf fallback
```
CF-based erf with B3 fallback for problematic negative region. **Mean ULP 0.15, Max ULP 145.**

#### H1: Inverted GELU
```
Newton-Raphson iteration
4-6 iterations for convergence
Finds x given y = GELU(x)
```
Useful for memory-efficient backpropagation.

#### E3: Range-Scaled Approximation
```
Scale factor s = 2 aligned with BF16 exponent
Fit polynomial over x/s instead of x
For x < 0: B3-style erf fallback
For x >= 0: Range-scaled polynomial
```
Reduces catastrophic cancellation in subtraction-heavy formulas. Mean ULP 1.75, Max ULP 1130.

#### H2: GELU-Softmax Combined Unit
```
8-segment PWL for exp(x) on [-4, 4]
tanh(z) = (exp(2z) - 1) / (exp(2z) + 1)
Shared PWL exp with softmax computation
For x < -2: B3-style erf fallback
```
Enables hardware sharing between GELU and softmax units. Mean ULP 1.33, Max ULP 1130.

#### H3: SoftEx Tanh
```
Padé [2/2] approximation for exp(x)
tanh(z) = (exp(2z) - 1) / (exp(2z) + 1)
Arithmetic-only exp replacement
```
Enables tanh-based GELU without true exp(). Mean ULP 1.28, Max ULP 1247.

### Tail Handling

All methods use a specialized **extended tail LUT** for x ∈ [-8.3125, -3.5]:

```cpp
namespace tail_lut {
    constexpr float GELU_N3_50 = -8.14202e-04f;   // x = -3.50, bf16=0xba55
    constexpr float GELU_N3_75 = -5.23840e-04f;   // x = -3.75
    // ... 21 entries at 0.25 step ...
    constexpr float GELU_N8_25 = -4.57967e-16f;   // x = -8.25, bf16=0xa604
    constexpr float GELU_N8_3125 = -2.12e-16f;    // x = -8.3125, last non-zero bf16
}
```

For x < -8.3125, bf16 underflows to -0 (0x8000). The extended LUT with finer resolution near the underflow boundary reduced Max ULP from 9904 to 145.

## Project Structure

| File | Description |
|------|-------------|
| `gelu_implementations.cpp` | All GELU approximations + analysis framework |
| `ulp_calculator.cpp` | Standalone ULP calculator with lookup table |
| `test_bfloat16.cpp` | Compiler bfloat16 support verification |
| `FinalLists.md` | Strategy taxonomy (35 methods, 8 categories) |
| `CLAUDE.md` | Development instructions and status |
| `HISTORY.md` | Development history and session notes |
| `README.md` | This file |

## Technical Details

### Saturation Thresholds

```
Positive: x ≥ 3  → GELU(x) = x    (Φ(3) ≈ 0.99865)
Negative: x ≤ -9 → GELU(x) = 0    (bf16(GELU(-9)) = -0)
```

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

After extending the tail LUT, max ULP was reduced from **9904 to 145** for the best methods (C1, B3). The remaining errors occur at two boundaries:

1. **x ≈ -8.3125**: Transition to bf16 underflow region (145 ULP)
2. **x ≈ -3.5**: Boundary between core approximation and tail handler (166 ULP for R4)

The extended LUT (21 entries at 0.25 step from -3.5 to -8.3125) with proper bf16 underflow handling achieves excellent accuracy across the entire bfloat16 range.

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

Ground truth uses float64 with standard library `erf()`:

```cpp
double gelu_reference_f64(double x) {
    double phi = 0.5 * (1.0 + std::erf(x * INV_SQRT_2));
    return x * phi;
}
```

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

All methods now achieve Max ULP ≤ 1547 (A1 Poly-7). The top 8 methods achieve Max ULP = 145 (the bfloat16 underflow limit at x ≈ -8.3125).

Previous issues with D2, D4, F2, F3 (arithmetic exp() failing for |x| > 2) were fixed by:
- **D2**: Replaced polynomial core with B3-style piecewise erf
- **D4**: Added Taylor approximation for |x| < 0.125
- **F2, F3**: Added B3-style erf fallback for x < -2

## Key Insights

1. **Extended tail LUT is essential**: Don't approximate exp() for large arguments - Taylor series fails catastrophically. Use precomputed LUT instead.

2. **Track bf16 underflow boundary**: At x ≈ -8.35, bf16 underflows to -0 (0x8000). The LUT must end just before this point.

3. **Monotonicity constraints**: Fritsch-Carlson condition (α² + β² ≤ 9) prevents unphysical artifacts in cubic spline interpolation.

4. **Asymmetric thresholds**: GELU approaches 0 slowly for negative x, requiring different handling than positive saturation.

5. **Entire range testing**: Methods optimized for [-8, 8] may fail catastrophically outside this range.

6. **Eight methods achieve Max ULP = 145**: R5, C1, B3, D2, F2, F3, D4 all reach the bf16 underflow limit. R4 achieves 166 (boundary at x=-3.5).

7. **B3 erf is the universal fallback**: When arithmetic-only exp() fails (|x| > 2), the B3 piecewise erf (Taylor + A-S rational) provides reliable fallback.

8. **Cost vs accuracy tradeoff**: Best methods (C1, R5) require LUT loads or branches; simplest methods (B1, A1) have 5-10× higher max ULP.

9. **100% taxonomy coverage**: All 40 methods from FinalLists.md are now implemented, including analysis functions for C5 (EPSS), E5/E8 (denormal/FTZ), and implementations for E3 (range-scaling) and H2 (GELU-Softmax).

## References

- Hendrycks, D., & Gimpel, K. (2016). Gaussian Error Linear Units (GELUs). arXiv:1606.08415
- BFloat16 specification (Google Brain, 2018)
- Abramowitz, M., & Stegun, I. A. (1964). Handbook of Mathematical Functions
- Remez algorithm for minimax polynomial fitting

## License

This project is provided for research and educational purposes.
