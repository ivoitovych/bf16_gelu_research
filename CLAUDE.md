# BFloat16 GELU Research Project

## Overview

Systematic ULP (Units in Last Place) error analysis of GELU approximations for bfloat16 floating-point arithmetic, targeting ML accelerator frameworks.

**Key constraint**: Analysis covers the **entire bfloat16 range** (65280 valid values, ~±3.4e38), not just typical activation ranges like [-8, 8].

## Build

```bash
# Requirements: GCC 13+ with C++23 support

# Build GELU analysis tool
g++ -std=c++23 -O2 -o gelu_analysis gelu_implementations.cpp

# Build ULP calculator (standalone)
g++ -std=c++23 -O2 -o ulp_calculator ulp_calculator.cpp

# Run analysis
./gelu_analysis --analyze    # Full ULP analysis
./gelu_analysis --diagnose   # Diagnostic mode
./gelu_analysis --reference  # Show reference values
./gelu_analysis --saturation # Analyze saturation boundaries
./gelu_analysis --calibrate  # Compute tail LUT values
./gelu_analysis --all        # All modes
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
| 1 | B1 | Sigmoid-based: x·σ(1.702x) | ✓ Complete |
| 1 | B1v2 | Quadratic sigmoid variant | ✓ Complete |
| 1 | R1/C4 | Saturation + poly-9 core | ✓ Improved coefficients |
| 1 | R2/A2 | Rational Padé | ✓ Improved coefficients |
| 1 | R3/C3 | Piecewise linear (ISPA) | ✓ Optimized segments |
| 1 | R4/B2 | Tanh-form + [3,3] Padé tanh | ✓ **Best performer** (0.61 mean ULP) |
| 1 | R5/D1 | LUT (512 entries) + interpolation | ✓ Complete |

### Not Yet Implemented

**Phase 2 - Better ULP Control:**
| ID | Method | Priority |
|----|--------|----------|
| C1 | Cubic spline (8-16 segments, C¹/C² continuity) | ★★★★★ |
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
| G4 | Backward pass (GELU') | ★★★★☆ |
| G5 | Cost model (mul/add/div count) | ★★★☆☆ |

## Current Results

| Method | Max ULP | Mean ULP | P99 | Notes |
|--------|---------|----------|-----|-------|
| **R4 Tanh** | 9904 | **0.61** | 0 | **Best mean ULP** - tanh-form + [3,3] Padé |
| R2 Rational | 9904 | 1.69 | **2** | **Best P99** - rational Padé |
| R1 Poly-9 | 9904 | 1.90 | 1 | Polynomial core |
| B1v2 Sigmoid | 9904 | 1.97 | 38 | Quadratic sigmoid |
| B1 Sigmoid | 9904 | 2.14 | 19 | Simple sigmoid |
| R5 LUT | 13215 | 15.02 | 0 | LUT (needs tail handler update) |
| R3 PWL | 9904 | 37.32 | 98 | High near-zero error |

**Saturation thresholds**: x ≥ 3 → x, x ≤ -9 → 0 (with specialized tail LUT from -8.5 to -3.5)

**Max-ULP analysis**: The 9904 max-ULP comes from the extreme negative tail (x=-8.375). At this point, GELU values are ~1e-15, and linear interpolation in our tail LUT doesn't perfectly match the exponential decay. Further improvement requires finer LUT resolution or accepting this as a hardware limitation of bf16.

## Quick Reference

```
GELU(x) = x · Φ(x)  where  Φ(x) = 0.5(1 + erf(x/√2))

Saturation thresholds (asymmetric):
  x ≥ 3  → GELU(x) ≈ x    (Φ(3) = 0.99865)
  x ≤ -9 → GELU(x) ≈ 0    (bf16(GELU(-9)) = -0)

Tail handling (x ∈ [-8.5, -3.5]):
  - LUT with 10 calibration points (0.5 step)
  - Linear interpolation between points
  - Handles exponential decay that polynomial/rational approximations miss

Key approximations:
  R4 Tanh:      GELU(x) ≈ 0.5x(1 + tanh(0.7979(x + 0.0447x³))) [BEST: 0.61 mean ULP]
  tanh approx:  tanh(z) ≈ z·(1 + 0.128z² + ...)/(1 + 0.462z² + ...) [3,3] Padé
  B1 sigmoid:   GELU(x) ≈ x · σ(1.702x), σ(z) ≈ 0.5 + z/(2(1+|z|))
  B1v2 sqrt:    GELU(x) ≈ x · [0.5 + 0.5·z/√(1+z²)], z = 1.702x
```

## G3 Multi-Region Analysis

Regions: near_zero (|x|<0.5), core_pos/neg (0.5≤|x|<3), tail_pos/neg (|x|≥3)

| Method | near_zero | core_pos | core_neg | tail_pos | tail_neg |
|--------|-----------|----------|----------|----------|----------|
| **R4 Tanh** | 0.00 | 0.03 | **1.75** | 0.00 | 2.43 |
| R2 Rational | 0.00 | 5.02 | 126.52 | 0.00 | 4.19 |
| R1 Poly-9 | 0.00 | 7.38 | 150.87 | 0.00 | 4.52 |
| B1v2 | 0.93 | 13.91 | 129.44 | 0.00 | 3.27 |
| R5 LUT | 0.00 | 0.00 | 0.03 | 0.00 | 60.56 |

(Mean ULP per region. R4 excels in core_neg due to accurate tanh approximation.)

## Design Decisions

1. **Type punning**: Use `std::memcpy()` only (no reinterpret_cast, no unions)
2. **ULP indexing**: +0 and -0 share same index; NaN/Inf excluded (65280 valid values)
3. **Saturation**: Asymmetric thresholds (3, -9) with specialized tail LUT for [-8.5, -3.5]
4. **Internal precision**: float32 for calculations, bfloat16 for I/O
5. **Entire range**: Unlike FinalLists.md's [-8,8], we test all 65280 bf16 values
6. **Tail handling**: LUT-based interpolation for negative tail where approximations fail

## Code Guidelines

- Extensive comments explaining rationale
- `static_assert` for compile-time type verification
- Reusable tests integrated in main files (no standalone scripts)
- No AI attribution in commits
- **Exploratory tests must be added to existing tools** (e.g., new `--mode` flags), not standalone scripts that require permission each run

## Priority Next Steps

1. **C1 Cubic spline** (expected best ULP control)
2. **E2 Coefficient quantization** to validate BF16 hardware behavior
3. **G4 Backward pass** (GELU') for training support
