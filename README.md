# BFloat16 GELU Approximation Research

Systematic ULP (Units in Last Place) error analysis of GELU activation function approximations for bfloat16 floating-point arithmetic, targeting ML accelerator frameworks.

## Overview

This project implements and evaluates multiple GELU approximation strategies optimized for **bfloat16** precision, measuring accuracy using **ULP error** rather than MSE. Unlike typical analyses that focus on common activation ranges like [-8, 8], this research covers the **entire bfloat16 range** (65,280 valid values, approximately ±3.4×10³⁸).

### Key Achievements

| Method | Mean ULP | Max ULP | Description |
|--------|----------|---------|-------------|
| **C1 Spline** | **0.60** | 9904 | 9-segment Hermite + Taylor near-zero |
| **B3 Erf Poly** | **0.61** | 9904 | Piecewise erf (Taylor + A-S rational) |
| **R4 Tanh** | **0.61** | 9904 | Tanh-form + [3,3] Padé approximation |
| R2 Rational | 1.69 | 9904 | Rational Padé [4/4] |
| R1 Poly-9 | 1.90 | 9904 | 9th-degree minimax polynomial |
| B1v2 Sigmoid | 1.97 | 9904 | Quadratic sigmoid approximation |
| B1 Sigmoid | 2.14 | 9904 | Simple sigmoid approximation |

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
# Full ULP analysis over entire bfloat16 range
./gelu_analysis --analyze

# Diagnostic mode with specific test points
./gelu_analysis --diagnose

# Show reference GELU values at key points
./gelu_analysis --reference

# Analyze saturation boundary behavior
./gelu_analysis --saturation

# Compute correct tail LUT calibration values
./gelu_analysis --calibrate

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

--- R4: B2 Tanh-form + Rational ---
Samples:    65280
Max ULP:    9904
Mean ULP:   0.6126
P50 ULP:    0
P90 ULP:    0
P99 ULP:    0

  Region Analysis (G3):
  +---------------+-------+----------+-----------+
  | Region        | Count | Max ULP  | Mean ULP  |
  +---------------+-------+----------+-----------+
  | near_zero     | 32256 |        1 |      0.00 |
  | core_pos      |   320 |        1 |      0.03 |
  | core_neg      |   321 |       29 |      1.75 |
  | tail_pos      | 16192 |        0 |      0.00 |
  | tail_neg      | 16191 |     9904 |      2.43 |
  +---------------+-------+----------+-----------+
```

## Implemented Approximations

### Approximation Constraints

All implementations use only basic arithmetic:
- **Allowed**: `+`, `-`, `*`, `/`, `|x|`, `sign()`
- **Prohibited**: `erf()`, `tanh()`, `exp()`, `log()` (except in reference implementation)

### Methods

#### B1: Sigmoid-Based GELU
```
GELU(x) ≈ x · σ(1.702x)
σ(z) ≈ 0.5 + z / (2(1 + |z|))
```
Simple rational sigmoid approximation. Good baseline with mean ULP ~2.14.

#### B1v2: Quadratic Sigmoid GELU
```
GELU(x) ≈ x · σ(1.702x)
σ(z) ≈ 0.5 + 0.5·z / √(1 + z²)
```
Uses hardware-accelerated sqrt for better accuracy. Mean ULP ~1.97.

#### R1: Saturation + Polynomial Core
```
x ≥ 3:  GELU(x) = x
x ≤ -9: GELU(x) = 0
else:   Φ(x) ≈ 0.5 + x·(a₁ + a₃x² + a₅x⁴ + a₇x⁶ + a₉x⁸)
```
9th-degree minimax polynomial with asymmetric saturation. Mean ULP ~1.90.

#### R2: Rational Padé Approximation
```
Φ(x) ≈ 0.5 + x · P(x²) / Q(x²)
P, Q are [3/3] polynomials in x²
```
Better tail convergence than pure polynomials. Mean ULP ~1.69.

#### R3: Piecewise Linear (PWL)
```
Power-of-2 breakpoints: 0, ±0.5, ±1, ±2, ±4
Linear interpolation between segments
```
Fast evaluation but higher near-zero error. Mean ULP ~37.32.

#### R4: Tanh-Form + Rational Tanh (Best Method)
```
GELU(x) ≈ 0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))
tanh(z) ≈ z·(a₀ + a₁z² + a₂z⁴ + a₃z⁶) / (b₀ + b₁z² + b₂z⁴ + b₃z⁶)
```
[3,3] Padé approximant for tanh. **Best performer with mean ULP 0.61**.

#### R5: LUT + Linear Interpolation
```
512-entry lookup table for Φ(x)
Linear interpolation between entries
Range: [-9, 3]
```
Uses precomputed erf values. Excellent core accuracy but needs tail handler update.

### Tail Handling

All methods use a specialized **tail LUT** for x ∈ [-8.5, -3.5]:

```cpp
namespace tail_lut {
    constexpr float GELU_N3_5 = -8.14202e-04f;   // x = -3.5
    constexpr float GELU_N4_0 = -1.26685e-04f;   // x = -4.0
    constexpr float GELU_N4_5 = -1.52895e-05f;   // x = -4.5
    // ... continues to x = -8.0
}
```

This handles the exponential decay region where polynomial/rational approximations fail.

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

The remaining max ULP (~9904) occurs at x ≈ -8.375 due to:
1. GELU values are ~10⁻¹⁵ in this region
2. Linear interpolation in tail LUT doesn't perfectly match exponential decay
3. BFloat16 representation limits at these tiny magnitudes

Further improvement requires finer LUT resolution or is fundamentally limited by bf16 precision.

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
| 3 | BF16 optimizations (E1-E7) | Not started |
| 4 | Validation (G2, G4, G5, G7/G8) | Partial (G4, G7/G8 done) |
| 5 | Advanced (D2-D4, H1-H3) | Not started |

### Future Work

1. **E2 Coefficient Quantization**: Validate hardware behavior with bf16 coefficients
2. **C2 Piecewise Rational**: 3-5 segment rational functions
3. **G2 FMA Comparison**: Measure impact of fused multiply-add

## Key Insights

1. **Tail saturation is critical**: BF16's 7-bit mantissa makes |x| > 3 extremely ULP-sensitive

2. **Power-of-2 breakpoints**: Reduces quantization error in piecewise methods

3. **Monotonicity constraints**: Prevents unphysical artifacts from coefficient rounding

4. **Asymmetric thresholds**: GELU approaches 0 slowly for negative x, requiring different handling than positive saturation

5. **Entire range testing**: Methods optimized for [-8, 8] may fail catastrophically outside this range

6. **Three methods tie for best**: C1 Spline (0.60), B3 Erf (0.61), and R4 Tanh (0.61) all achieve excellent accuracy

## References

- Hendrycks, D., & Gimpel, K. (2016). Gaussian Error Linear Units (GELUs). arXiv:1606.08415
- BFloat16 specification (Google Brain, 2018)
- Abramowitz, M., & Stegun, I. A. (1964). Handbook of Mathematical Functions
- Remez algorithm for minimax polynomial fitting

## License

This project is provided for research and educational purposes.
