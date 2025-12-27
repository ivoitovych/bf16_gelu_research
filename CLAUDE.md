# BFloat16 GELU Research Project

## Overview

Systematic ULP (Units in Last Place) error analysis of GELU approximations for bfloat16 floating-point arithmetic, targeting ML accelerator frameworks.

**Key constraint**: Analysis covers the **entire bfloat16 range** (65280 valid values), not just typical activation ranges.

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
| `FinalLists.md` | Strategy taxonomy (Categories A-H) |
| `CLAUDE.md` | Project instructions (this file) |
| `HISTORY.md` | Development history and decisions |

## Approximation Constraints

- **Allowed**: `+`, `−`, `×`, `÷`, `|x|`, `sign()`
- **Prohibited**: `erf()`, `tanh()`, `exp()`, `log()`
- **Target**: BFloat16 (1 sign + 8 exponent + 7 mantissa bits)
- **Metric**: ULP error (worst-case), not MSE

## Implemented Methods (R1-R5)

| ID | Strategy | Method |
|----|----------|--------|
| R1 | C4 | Saturation + polynomial core |
| R2 | A2 | Rational Padé approximation |
| R3 | C3 | Piecewise linear (ISPA) |
| R4 | B2 | Tanh-form + rational tanh |
| R5 | D1 | LUT + linear interpolation |

## Current Results

| Method | Max ULP | Mean ULP | P99 | Status |
|--------|---------|----------|-----|--------|
| R1 Poly | 15760 | 36.0 | 3 | Needs coefficient fitting |
| R2 Rational | 13760 | 21.8 | 5 | Needs coefficient fitting |
| R3 PWL | 13760 | 55.7 | 98 | Needs segment optimization |
| R4 Tanh | 15351 | 44.1 | 1 | Needs coefficient fitting |
| R5 LUT | 13760 | 18.7 | 0 | Best performer |

**Note**: High max-ULP values are due to saturation boundary at x=-5 where GELU(-5)≈-1.4e-6 but approximations return 0.

## Quick Reference

```
GELU(x) = x · Φ(x)  where  Φ(x) = 0.5(1 + erf(x/√2))

Saturation thresholds (asymmetric):
  x ≥ 3  → GELU(x) ≈ x
  x ≤ -5 → GELU(x) ≈ 0

Tanh approximation:  tanh(z) ≈ z(27 + z²) / (27 + 9z²)
Sigmoid approximation: σ(z) ≈ 0.5 + z / (2(1 + |z|))
```

## Design Decisions

1. **Type punning**: Use `std::memcpy()` only (no reinterpret_cast, no unions)
2. **ULP indexing**: +0 and -0 share same index; NaN/Inf excluded
3. **Saturation**: Asymmetric thresholds (3, -5) minimize boundary errors
4. **Internal precision**: float32 for calculations, bfloat16 for I/O

## Code Guidelines

- Extensive comments explaining rationale
- `static_assert` for compile-time type verification
- Reusable tests integrated in main files (no standalone scripts)
- No AI attribution in commits

## Next Steps

1. Proper Remez/least-squares coefficient fitting for R1, R2, R4
2. Segment optimization for R3
3. E1-E7 BF16-specific optimizations
4. FMA vs non-FMA comparison
