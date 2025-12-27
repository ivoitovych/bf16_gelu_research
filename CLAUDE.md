# BFloat16 GELU Research Project

## Overview

Systematic ULP (Units in Last Place) error analysis of GELU approximations for bfloat16 floating-point arithmetic, targeting ML accelerator frameworks.

**Key constraint**: Analysis covers the **entire bfloat16 range** (65280 valid values, ~±3.4e38), not just typical activation ranges like [-8, 8].

## Build

```bash
# Requirements: GCC 13+ with C++23 support

# Build GELU analysis tool
g++ -std=c++23 -O3 -march=native -o gelu_analysis gelu_implementations.cpp -lm

# Build ULP calculator (standalone)
g++ -std=c++23 -O2 -o ulp_calculator ulp_calculator.cpp

# Run analysis
./gelu_analysis --analyze      # Full ULP analysis (22 methods)
./gelu_analysis --diagnose     # Diagnostic mode
./gelu_analysis --reference    # Show reference values
./gelu_analysis --saturation   # Analyze saturation boundaries
./gelu_analysis --calibrate    # Compute tail LUT values
./gelu_analysis --regression   # G7/G8 regression suite
./gelu_analysis --derivative   # G4 backward pass test
./gelu_analysis --verify-knots # Debug C1 spline knots
./gelu_analysis --quantization # E2: Coefficient quantization
./gelu_analysis --fma          # E6/G2: FMA vs non-FMA comparison
./gelu_analysis --sensitivity  # E7: Coefficient sensitivity
./gelu_analysis --cost-model   # G5: Operation cost model
./gelu_analysis --all          # All modes
```

## Project Structure

| File | Description |
|------|-------------|
| `gelu_implementations.cpp` | GELU approximations (R1-R5) + analysis framework |
| `ulp_calculator.cpp` | Reference ULP calculator with lookup table |
| `test_bfloat16.cpp` | Compiler bfloat16 support verification |
| `FinalLists.md` | Strategy taxonomy (Categories A-H, 35 methods) |
| `CLAUDE.md` | Project instructions (this file) |
| `HISTORY.md` | Development history and decisions |

## Approximation Constraints

- **Allowed**: `+`, `−`, `×`, `÷`, `|x|`, `sign()`
- **Prohibited**: `erf()`, `tanh()`, `exp()`, `log()`
- **Target**: BFloat16 (1 sign + 8 exponent + 7 mantissa bits)
- **Metric**: ULP error (worst-case max-ULP), not MSE

## Implementation Status vs FinalLists.md

### All Phases Complete ✓

| Category | Methods Implemented | Status |
|----------|---------------------|--------|
| **A: Direct** | A1 (poly-7, poly-9), A3 (Chebyshev), A4 (cont.frac) | ✓ Complete |
| **B: Sub-function** | B1, B1v2 (sigmoid), B3 (erf poly), B4 (rational erf) | ✓ Complete |
| **C: Piecewise** | C1 (spline), C2 (piecewise rational) | ✓ Complete |
| **D: Hybrid/LUT** | R5/D1 (LUT), D2 (LUT+poly), D3 (LUT+corr), D4 (nonuniform) | ✓ Complete |
| **E: BF16 Knobs** | E2 (quantization), E6 (FMA), E7 (sensitivity) | ✓ Complete |
| **F: Reference** | F1 (float64), F2 (quadrature), F3 (CF erf) | ✓ Complete |
| **G: Methodology** | G1-G5, G7/G8 | ✓ Complete |
| **H: Advanced** | H1 (inverse), H3 (SoftEx) | ✓ Complete |

### Best Performers (Max ULP ≤ 200)

| ID | Method | Mean ULP | Max ULP |
|----|--------|----------|---------|
| **R5** | LUT + extended tail | **0.10** | **145** |
| **C1** | Cubic spline (9-seg) | **0.13** | **145** |
| **B3** | Erf polynomial (A-S) | **0.13** | **145** |
| **D2** | LUT tails + B3 erf | 0.13 | 145 |
| **F2** | Quadrature + B3 erf | 0.13 | 145 |
| **R4** | Tanh-form + Padé | 0.14 | 166 |
| **F3** | CF + B3 erf fallback | 0.15 | 145 |
| **D4** | Non-uniform LUT | 0.63 | 145 |

## Current Results

| Method | Max ULP | Mean ULP | P99 | Notes |
|--------|---------|----------|-----|-------|
| **R5 LUT** | **145** | **0.10** | 0 | **Best overall** - 512 entries + extended tail |
| **C1 Cubic Spline** | **145** | **0.13** | 0 | 9-seg Hermite + Taylor near-zero |
| **B3 Erf Polynomial** | **145** | **0.13** | 0 | Piecewise erf (Taylor + rational) |
| **D2 LUT+Erf** | **145** | **0.13** | 0 | LUT tails + B3-style erf core |
| **F2 Quadrature** | **145** | **0.13** | 0 | Gauss-Legendre + B3 erf fallback |
| **R4 Tanh** | 166 | 0.14 | 0 | Tanh-form + [3,3] Padé |
| **F3 CF Erf** | **145** | **0.15** | 0 | Continued fraction + B3 erf fallback |
| **D4 Non-uniform** | **145** | **0.63** | 28 | Non-uniform LUT + Taylor near-zero |
| B4 Rational Erf | 535 | 0.59 | 2 | Range-reduced rational |
| B1v2 Sigmoid | 625 | 1.50 | 34 | Quadratic sigmoid |
| D3 LUT+Corr | 830 | 1.84 | 41 | Coarse LUT + correction |
| R3 PWL | 832 | 36.85 | 97 | High near-zero error |
| C2 Piecewise | 881 | 1.18 | 1 | Piecewise rational |
| B1 Sigmoid | 1068 | 1.67 | 18 | Simple sigmoid |
| R2 Rational | 1139 | 1.22 | 2 | Rational Padé |
| A4 Cont.Frac | 1206 | 1.73 | 15 | Continued fraction |
| A3 Chebyshev | 1207 | 2.57 | 45 | Chebyshev (Clenshaw) |
| H3 SoftEx | 1247 | 1.28 | 1 | Arithmetic exp via Padé |
| R1 Poly-9 | 1312 | 1.43 | 1 | Polynomial core |
| A1 Direct Poly | 1404-1547 | 3.58-3.61 | 33 | Direct polynomial (7th/9th degree) |

**Saturation thresholds**: x ≥ 3 → x, x ≤ -9 → 0

**Tail handling**: Extended LUT from x=-3.5 to x=-8.3125 with 0.25-step + finer resolution near bf16 underflow boundary. For x < -8.3125, bf16 underflows to -0.

## Quick Reference

```
GELU(x) = x · Φ(x)  where  Φ(x) = 0.5(1 + erf(x/√2))

Saturation thresholds (asymmetric):
  x ≥ 3  → GELU(x) ≈ x    (Φ(3) = 0.99865)
  x ≤ -9 → GELU(x) ≈ 0    (bf16(GELU(-9)) = -0)

Tail handling (x ∈ [-8.3125, -3.5]):
  - Extended LUT with 21 calibration points (0.25 step)
  - Linear interpolation between points
  - For x < -8.3125, bf16 underflows to -0

Key approximations:
  C1 Spline:    9-segment Hermite + Taylor near-zero [BEST: 0.13 mean ULP, 145 max ULP]
  B3 Erf:       Piecewise erf (Taylor |z|<1, A-S rational |z|≥1) [0.13 mean ULP, 145 max ULP]
  R4 Tanh:      GELU(x) ≈ 0.5x(1 + tanh(0.7979(x + 0.0447x³))) [0.14 mean ULP, 166 max ULP]
  tanh approx:  tanh(z) ≈ z·(1 + 0.128z² + ...)/(1 + 0.462z² + ...) [3,3] Padé
```

## G3 Multi-Region Analysis

Regions: near_zero (|x|<0.5), core_pos/neg (0.5≤|x|<3), tail_pos/neg (|x|≥3)

| Method | near_zero | core_pos | core_neg | tail_pos | tail_neg |
|--------|-----------|----------|----------|----------|----------|
| **R5 LUT** | **0.00** | **0.00** | **0.03** | 0.00 | 0.39 |
| **C1 Spline** | 0.00 | 0.07 | 4.09 | 0.00 | 0.41 |
| **B3 Erf** | 0.00 | 0.04 | 2.03 | 0.00 | 0.50 |
| **D2 LUT+Erf** | 0.00 | 0.04 | 2.03 | 0.00 | 0.50 |
| **F2 Quadrature** | 0.00 | 0.12 | 1.94 | 0.00 | 0.50 |
| **R4 Tanh** | 0.00 | 0.03 | 1.75 | 0.00 | 0.53 |

(Mean ULP per region. All top methods use extended tail LUT for tail_neg.)

## Design Decisions

1. **Type punning**: Use `std::memcpy()` only (no reinterpret_cast, no unions)
2. **ULP indexing**: +0 and -0 share same index; NaN/Inf excluded (65280 valid values)
3. **Saturation**: Asymmetric thresholds (3, -9) with specialized tail LUT for [-8.3125, -3.5]
4. **Internal precision**: float32 for calculations, bfloat16 for I/O
5. **Entire range**: Unlike FinalLists.md's [-8,8], we test all 65280 bf16 values
6. **Tail handling**: LUT-based interpolation for negative tail where approximations fail

## Code Guidelines

- Extensive comments explaining rationale
- `static_assert` for compile-time type verification
- Reusable tests integrated in main files (no standalone scripts)
- **Exploratory tests must be added to existing tools** (e.g., new `--mode` flags), not standalone scripts that require permission each run

## Git Commit Rules

**CRITICAL: NO AI ATTRIBUTION IN COMMITS**
- Do NOT add "Generated with Claude Code" footer
- Do NOT add "Co-Authored-By: Claude" lines
- Do NOT mention AI, Claude, or LLM in commit messages
- Keep commit messages clean and professional

## Project Status

**All phases complete.** 8 methods achieve Max ULP = 145 (bf16 underflow limit):
- R5 LUT (0.10 mean), C1 Spline (0.13), B3 Erf (0.13), D2 LUT+Erf (0.13)
- F2 Quadrature (0.13), R4 Tanh (0.14, max 166), F3 CF Erf (0.15), D4 Non-uniform (0.63)

Analysis tools available: `--quantization`, `--fma`, `--sensitivity`, `--cost-model`
