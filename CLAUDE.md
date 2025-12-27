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
| 1 | R1/C4 | Saturation + polynomial core | ⚠ Placeholder coefficients |
| 1 | R2/A2 | Rational Padé [5/5] | ⚠ Placeholder coefficients |
| 1 | R3/C3 | Piecewise linear (ISPA) | ⚠ Needs segment optimization |
| 1 | R4/B2 | Tanh-form + rational tanh | ⚠ Placeholder coefficients |
| 1 | R5/D1 | LUT + linear interpolation | ✓ Best performer |

### Not Yet Implemented

**Phase 1 - Core Baselines:**
| ID | Method | Priority |
|----|--------|----------|
| A1 | Minimax polynomial (Remez-fitted, degrees 5/7/9) | ★★★★★ |
| B1 | Sigmoid-based: GELU(x) ≈ x·σ(1.702x) | ★★★★☆ |

**Phase 2 - Better ULP Control:**
| ID | Method | Priority |
|----|--------|----------|
| C1 | Cubic spline (8-16 segments, C¹/C² continuity) | ★★★★★ |
| C2 | Piecewise rational (3-5 segments) | ★★★★☆ |

**Phase 3 - BF16 Optimizations (E1-E7):**
| ID | Optimization | Priority |
|----|--------------|----------|
| E1 | Monotonicity/bounds-constrained fitting | ★★★★★ |
| E2 | Coefficient quantization to BF16 + refit | ★★★★★ |
| E3 | Range-scaled approximation | ★★★☆☆ |
| E6 | FMA-aware coefficient sets (Horner vs Estrin) | ★★★★☆ |
| E7 | Coefficient sensitivity/robustness testing | ★★★☆☆ |

**Phase 4 - Validation & Training:**
| ID | Method | Priority |
|----|--------|----------|
| G2 | FMA vs non-FMA comparison | ★★★★☆ |
| G3 | Multi-region error analysis (near-zero/core/tails) | ★★★★☆ |
| G4 | Backward pass (GELU') | ★★★★☆ |
| G5 | Cost model (mul/add/div count) | ★★★☆☆ |

**Phase 5 - Advanced:**
| ID | Method | Priority |
|----|--------|----------|
| D2 | LUT tails + polynomial center | ★★★★☆ |
| H1 | Inverted GELU (for training memory) | ★★★☆☆ |

## Current Results

| Method | Max ULP | Mean ULP | P99 | Issue |
|--------|---------|----------|-----|-------|
| R1 Poly | 15760 | 36.0 | 3 | Coefficients diverge for x<-1 |
| R2 Rational | 13760 | 21.8 | 5 | Saturation boundary error |
| R3 PWL | 13760 | 55.7 | 98 | High baseline error |
| R4 Tanh | 15351 | 44.1 | 1 | tanh approx fails for x<-2 |
| R5 LUT | 13760 | 18.7 | 0 | Best, limited by saturation |

**Root cause of high max-ULP**: At x=-5, GELU(-5)≈-1.4e-6 but approximations return 0. This saturation boundary contributes ~13760 ULP. Extending the negative threshold would reduce max-ULP but add complexity.

## Quick Reference

```
GELU(x) = x · Φ(x)  where  Φ(x) = 0.5(1 + erf(x/√2))

Saturation thresholds (asymmetric):
  x ≥ 3  → GELU(x) ≈ x    (Φ(3) = 0.9987)
  x ≤ -5 → GELU(x) ≈ 0    (Φ(-5) = 2.87e-7)

Key approximations from FinalLists.md:
  tanh(z) ≈ z(27 + z²) / (27 + 9z²)
  σ(z) ≈ 0.5 + z / (2(1 + |z|))
  GELU via sigmoid: GELU(x) ≈ x · σ(1.702x)
  GELU via tanh: GELU(x) ≈ 0.5x(1 + tanh(0.7979(x + 0.0447x³)))
```

## Design Decisions

1. **Type punning**: Use `std::memcpy()` only (no reinterpret_cast, no unions)
2. **ULP indexing**: +0 and -0 share same index; NaN/Inf excluded (65280 valid values)
3. **Saturation**: Asymmetric thresholds (3, -5) based on ULP analysis
4. **Internal precision**: float32 for calculations, bfloat16 for I/O
5. **Entire range**: Unlike FinalLists.md's [-8,8], we test all 65280 bf16 values

## Code Guidelines

- Extensive comments explaining rationale
- `static_assert` for compile-time type verification
- Reusable tests integrated in main files (no standalone scripts)
- No AI attribution in commits

## Priority Next Steps

1. **Remez fitting** for R1, R2, R4 coefficients (replace placeholders)
2. **B1 Sigmoid-based** GELU (simple, good baseline)
3. **C1 Cubic spline** (expected best ULP control)
4. **G3 Multi-region analysis** (separate metrics for near-zero/core/tails)
5. **E2 Coefficient quantization** to validate BF16 hardware behavior
