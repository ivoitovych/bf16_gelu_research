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
./gelu_analysis --analyze      # Full ULP analysis
./gelu_analysis --diagnose     # Diagnostic mode
./gelu_analysis --reference    # Show reference values
./gelu_analysis --saturation   # Analyze saturation boundaries
./gelu_analysis --calibrate    # Compute tail LUT values
./gelu_analysis --regression   # G7/G8 regression suite
./gelu_analysis --derivative   # G4 backward pass test
./gelu_analysis --verify-knots # Debug C1 spline knots
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

### Implemented ✓

| Phase | ID | Method | Status |
|-------|-----|--------|--------|
| 0 | F1 | High-precision GELU (float64) | ✓ Complete |
| 0 | G1 | ULP measurement framework | ✓ Complete |
| 0 | G3 | Multi-region error analysis | ✓ Complete |
| 1 | A1 | Direct polynomial (7th/9th degree) | ✓ Complete |
| 1 | B1 | Sigmoid-based: x·σ(1.702x) | ✓ Complete |
| 1 | B1v2 | Quadratic sigmoid variant | ✓ Complete |
| 1 | B3 | Erf polynomial (piecewise A-S) | ✓ **Best performer** (0.13 mean, 145 max ULP) |
| 2 | C1 | Cubic spline (9 segments + Fritsch-Carlson) | ✓ **Best performer** (0.13 mean, 145 max ULP) |
| 1 | R1/C4 | Saturation + poly-9 core | ✓ Complete |
| 1 | R2/A2 | Rational Padé | ✓ Complete |
| 1 | R3/C3 | Piecewise linear (ISPA) | ✓ Complete |
| 1 | R4/B2 | Tanh-form + [3,3] Padé tanh | ✓ Excellent (0.14 mean, 166 max ULP) |
| 1 | R5/D1 | LUT (512 entries) + interpolation | ✓ Complete |
| 4 | G4 | Backward pass (GELU' derivative) | ✓ Complete |
| 4 | G7/G8 | Regression suite (adversarial points) | ✓ Complete |

### Not Yet Implemented

**Phase 2 - Better ULP Control:**
| ID | Method | Priority |
|----|--------|----------|
| C2 | Piecewise rational (3-5 segments) | ★★★★☆ |

**Phase 3 - BF16 Optimizations (E1-E7):**
| ID | Optimization | Priority |
|----|--------------|----------|
| E1 | Monotonicity/bounds-constrained fitting | ★★★★☆ |
| E2 | Coefficient quantization to BF16 + refit | ★★★★★ |
| E6 | FMA-aware coefficient sets (Horner vs Estrin) | ★★★★☆ |

**Phase 4 - Validation & Training:**
| ID | Method | Priority |
|----|--------|----------|
| G2 | FMA vs non-FMA comparison | ★★★★☆ |
| G5 | Cost model (mul/add/div count) | ★★★☆☆ |

## Current Results

| Method | Max ULP | Mean ULP | P99 | Notes |
|--------|---------|----------|-----|-------|
| **C1 Cubic Spline** | **145** | **0.13** | 0 | **Best overall** - 9-seg Hermite + Taylor near-zero |
| **B3 Erf Polynomial** | **145** | **0.13** | 0 | Piecewise erf (Taylor + rational) |
| **R4 Tanh** | 166 | 0.14 | 0 | Tanh-form + [3,3] Padé |
| B1v2 Sigmoid | 625 | 1.50 | 34 | Quadratic sigmoid |
| R3 PWL | 832 | 36.85 | 98 | High near-zero error |
| B1 Sigmoid | 1068 | 1.67 | 18 | Simple sigmoid |
| R2 Rational | 1139 | 1.22 | 3 | Rational Padé |
| R1 Poly-9 | 1312 | 1.43 | 33 | Polynomial core |
| A1 Direct Poly | 1404-1547 | 3.58-3.61 | 33 | Direct polynomial (7th/9th degree) |
| R5 LUT | 13215 | 15.02 | 0 | LUT (needs separate tail handler) |

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
| **C1 Spline** | **0.00** | 0.07 | 4.09 | 0.00 | **0.05** |
| **B3 Erf** | **0.00** | 0.04 | 2.03 | 0.00 | 0.05 |
| **R4 Tanh** | **0.00** | **0.03** | **1.75** | 0.00 | 0.53 |
| R2 Rational | 0.00 | 5.02 | 126.52 | 0.00 | 1.08 |
| R1 Poly-9 | 0.00 | 7.38 | 150.87 | 0.00 | 3.12 |

(Mean ULP per region. Extended tail LUT dramatically reduced tail_neg errors.)

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

## Priority Next Steps

1. **E2 Coefficient quantization** to validate BF16 hardware behavior
2. **C2 Piecewise rational** (3-5 segments) for potential improvement
3. **G2 FMA comparison** to measure impact of fused multiply-add
