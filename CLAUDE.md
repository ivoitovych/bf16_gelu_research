# BFloat16 GELU Research Project

## Overview

Systematic ULP error analysis of GELU approximations for bfloat16 arithmetic. See [README.md](README.md) for full documentation.

**Key constraint**: Analysis covers the **entire bfloat16 range** (65280 valid values, ~±3.4e38), not just [-8, 8].

## Build & Run

```bash
# Requirements: GCC 13+ with C++23 support

# Build and run
g++ -std=c++23 -O3 -march=native -o gelu_analysis gelu_implementations.cpp -lm
./gelu_analysis --analyze      # Full ULP analysis (38 methods)
./gelu_analysis --all          # All analysis modes

# Other modes: --diagnose, --reference, --saturation, --calibrate,
# --regression, --derivative, --verify-knots, --quantization, --fma,
# --sensitivity, --cost-model, --epss, --range-scale, --denormal,
# --softmax-unit, --tail-debug, --sanity
```

## Project Structure

| File | Description |
|------|-------------|
| `gelu_implementations.cpp` | All implementations + analysis framework |
| `ulp_calculator.cpp` | Standalone ULP calculator |
| `debug_tools.cpp` | Exploratory debugging tools |
| `FinalLists.md` | Strategy taxonomy (40 methods, 8 categories) |
| `README.md` | Full documentation and results |
| `HISTORY.md` | Development history (18 sessions) |
| `CLAUDE.md` | This file - project instructions |

## Constraints

**Hardware Target**: Tenstorrent Wormhole/Blackhole ML accelerators. The tensor core (Tensix) supports only basic arithmetic; transcendental functions require the slower SFPU or must be approximated.

- **Allowed**: `+`, `−`, `×`, `÷`, `|x|`, `sign()`, comparison, bit manipulation, polynomial evaluation
- **Prohibited**: `std::erf()`, `std::tanh()`, `std::exp()`, `std::log()` (except reference)
- **Note**: Pure methods use `fast_exp_neg()` via IEEE754 bit manipulation (Schraudolph 1999), not std::exp()
- **Target**: BFloat16 (1 sign + 8 exponent + 7 mantissa bits)
- **Metric**: ULP error (worst-case max-ULP), not MSE

## Current Status

**38 methods implemented. R5 Pure achieves Max ULP = 2 (best overall) after subnormal exp fix.**

See [README.md](README.md) for complete results table with per-region analysis.

### Top Methods

| Method | Max ULP | Mean ULP | Notes |
|--------|---------|----------|-------|
| **R5 Pure** | **2** | **0.002** | LUT core + asymptotic tail (best overall) |
| **B3 Pure** | **23** | 0.01 | Pure arithmetic, asymptotic expansion |
| **D2 Pure** | **23** | 0.01 | Hybrid LUT+erf + asymptotic tail |
| **F3 Pure** | **28** | 0.02 | Continued fraction + asymptotic tail |
| **R4 Pure** | **29** | 0.01 | Tanh-form + asymptotic tail |
| **C1 Pure** | **35** | 0.03 | Cubic spline + asymptotic tail |
| E4 Hermite | 58 | 0.05 | Hermite blending at transition |
| R5/C1/B3/D2/F2/F3 | 87 | — | Limited by shared tail LUT interpolation |

### Tenstorrent Hardware Reference Benchmarks

Two reference implementations from Tenstorrent Wormhole/Blackhole hardware (DO NOT MODIFY):
- **TT Accurate**: 15th-degree Chebyshev polynomial (default in tt-train)
- **TT Fast**: 6-piece piecewise linear LUT (fast mode)

These are NOT optimized for full bf16 range - included for precision comparison only.

### Key Technical Achievements

1. **erfc-based reference**: Avoids catastrophic cancellation in `1 + erf(z)` for large negative z
2. **Asymptotic expansion**: `GELU(x) ≈ -φ(x)·(1 - 1/x² + 3/x⁴ - 15/x⁶)` beats LUT for deep tail
3. **Two-tier tail LUT**: 0.25-step main + 0.0625-step fine near underflow boundary
4. **Subnormal exp handling**: `fast_exp2_neg` handles x down to -149 for deep tail accuracy near bf16 saturation (-13.5625)

## Quick Reference

```
GELU(x) = x · Φ(x)  where  Φ(x) = 0.5(1 + erf(x/√2))

Reference (avoids cancellation for negative x):
  if (x >= 0): Φ(x) = 0.5 * (1 + erf(x/√2))
  if (x < 0):  Φ(x) = 0.5 * erfc(-x/√2)

Saturation thresholds:
  x ≥ 3  → GELU(x) = x
  x ≤ -9 → GELU(x) = 0

Deep tail (x < -8.3125):
  GELU(x) ≈ -exp(-x²/2)/√(2π) · (1 - 1/x² + 3/x⁴ - 15/x⁶)
```

## Code Guidelines

- Use `std::memcpy()` for type punning (no reinterpret_cast, no unions)
- `static_assert` for compile-time type verification
- Add new tests as `--mode` flags, not standalone scripts
- Extensive comments explaining rationale

## Git Commit Rules

**Author**: All commits must use:
```
Iaroslav Voitovych <yaroslav.voytovych@gmail.com>
```

**CRITICAL: NO AI ATTRIBUTION IN COMMITS**
- Do NOT add "Generated with Claude Code" footer
- Do NOT add "Co-Authored-By: Claude" lines
- Do NOT mention AI, Claude, or LLM in commit messages

## Development History

See [HISTORY.md](HISTORY.md) for detailed session-by-session development notes covering:
- Sessions 1-3: Initial implementation, gap analysis, improvements
- Sessions 4-6: Tail handler, bug fixes, extended tail LUT (Max ULP 9904→145)
- Sessions 7-10: Complete taxonomy coverage (40/40 methods)
- Session 11: Two-tier LUT fix (Max ULP 145→87)
- Session 12: erfc + asymptotic expansion (Max ULP 87→33 for B3 Pure)
- Session 13: Tenstorrent hardware reference benchmarks (TT Accurate, TT Fast)
- Session 14: ULP measurement sanity check framework (--sanity flag)
- Session 15: R5 Pure - LUT with asymptotic tail (ties B3 Pure at Max ULP 33, best Mean ULP 0.003)
- Session 16: C1/D2/F3 Pure variants (5 Pure methods total, all Max ULP ≤ 35)
- Session 17: A1 Pure, R4 Pure, E4 Hermite blend, E9 Remez BF16 (6 Pure methods, 34 total)
- Session 18: Subnormal exp fix (R5 Pure: 33→2, other Pure methods improved)
