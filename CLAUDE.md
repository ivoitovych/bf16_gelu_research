# BFloat16 GELU Research Project

## Project Overview

This project implements reference GELU (Gaussian Error Linear Unit) approximations and ULP (Unit in the Last Place) error analysis tools for bfloat16 floating-point arithmetic research.

**Critical**: ULP analysis is performed over the **entire bfloat16 range**, not just typical activation ranges like [-8, 8]. The real ULP measurements will validate theoretical expectations.

## Build Requirements

- **Compiler**: GCC 13+ with C++23 support
- **Required flags**: `-std=c++23`
- **Optional flags**: `-O2` for optimized builds, `-g` for debugging

## Build Commands

```bash
# Compile the ULP calculator
g++ -std=c++23 -O2 -o ulp_calculator ulp_calculator.cpp

# Compile with debug symbols
g++ -std=c++23 -g -o ulp_calculator_debug ulp_calculator.cpp
```

## Project Structure

- `ulp_calculator.cpp` - Reference ULP calculator implementation
- `gelu_implementations.cpp` - All GELU approximation implementations
- `test_bfloat16.cpp` - Compiler bfloat16 support verification
- `FinalLists.md` - Consolidated approximation strategy lists
- `CLAUDE.md` - This file (project instructions)

---

## GELU Approximation Strategy

### Constraints
- **Allowed operations**: `+`, `−`, `×`, `÷`, `|·|`, `sign()` only
- **Prohibited functions**: `erf()`, `tanh()`, `exp()`, `log()`
- **Target format**: BFloat16 (1 sign + 8 exponent + 7 mantissa bits)
- **Primary metric**: ULP (Units in Last Place), not MSE
- **Analysis range**: **Entire bfloat16 range** (all valid finite values)

### Implementation Categories

#### Category A: Direct GELU Approximations
- **A1**: Minimax Polynomial (Remez) - degrees 5, 7, 9
- **A2**: Rational Function (Padé/Minimax) - orders [3/3], [4/4], [5/5]
- **A3**: Chebyshev Polynomial
- **A4**: Continued Fraction

#### Category B: Sub-Function Approximations
- **B1**: Sigmoid-based (no exp): `GELU(x) ≈ x · σ(1.702x)`
- **B2**: Tanh-form + rational tanh (K-TanH style)
- **B3**: Erf polynomial → Φ → GELU
- **B4**: Rational erf/Φ with range reduction

#### Category C: Piecewise Methods
- **C1**: Piecewise polynomial splines (4-16 segments)
- **C2**: Piecewise rational functions
- **C3**: Piecewise linear (ISPA/NLI) with power-of-2 breakpoints
- **C4**: Asymptotic saturation + core approximation

#### Category D: Hybrid & LUT-Based
- **D1**: LUT + linear/quadratic interpolation
- **D2**: LUT tails + polynomial center
- **D3**: Non-uniform linear interpolation
- **D4**: Lookup + polynomial correction

#### Category E: BF16-Specific Optimizations
- **E1**: ULP/monotonicity-constrained fitting
- **E2**: Coefficient quantization to BF16
- **E3**: Range-scaled approximation
- **E4**: Breakpoint/knot constraints (power-of-2)
- **E5**: Denormal/subnormal policy
- **E6**: FMA-aware coefficient sets
- **E7**: Coefficient sensitivity/robustness testing

#### Category F: Reference & Ground-Truth
- **F1**: High-precision exact GELU (float64) ★ PRIMARY REFERENCE
- **F2**: Numerical quadrature for Φ(x)
- **F3**: Continued fraction erf/Φ
- **F4**: Abramowitz-Stegun erf polynomial

#### Category G: Methodology & Evaluation
- **G1**: ULP measurement framework
- **G2**: FMA vs non-FMA comparison
- **G3**: Backward pass (GELU derivative)
- **G4**: Breakpoint/knot optimization
- **G5**: Multi-region error analysis
- **G6**: Coefficient robustness analysis
- **G7**: Cost model
- **G8**: Regression suite

#### Category H: Advanced & Research
- **H1**: Inverted GELU for training memory reduction
- **H2**: Combined GELU-Softmax arithmetic unit
- **H3**: SoftEx-inspired exp approximation

---

## Recommended Minimal Baseline Set (R1-R5)

| ID | Method | Description |
|----|--------|-------------|
| R1 | C4 | Saturation + minimax core (poly-7) |
| R2 | A2 | Rational Padé [4/4] direct GELU |
| R3 | C3 | PWL ISPA with power-of-2 breakpoints (16 segments) |
| R4 | B2 | Tanh-form + odd rational tanh |
| R5 | D1 | LUT + linear interpolation |

---

## Quick Reference Formulas

```
Sigmoid (simple):     σ(z) = 0.5 + z / (2(1 + |z|))
Tanh (simple):        tanh(z) ≈ z(27 + z²) / (27 + 9z²)
GELU via sigmoid:     GELU(x) ≈ x · σ(1.702x)
GELU via tanh:        GELU(x) ≈ 0.5x(1 + tanh(0.7979(x + 0.0447x³)))
Saturation:           x > T: GELU ≈ x;  x < -T: GELU ≈ 0
```

---

## Implementation Phases

### Phase 0: Infrastructure (Complete)
- [x] Verify bfloat16 compiler support (GCC 13.3 with C++23)
- [x] Implement ULP calculator with lookup table
- [x] Implement simple ULP calculator for verification
- [x] Compile and test ULP infrastructure
- [x] Implement F1: High-precision GELU reference (float64)

### Phase 1: Core Baselines (Complete - Placeholder Coefficients)
- [x] R1: C4 (saturation + polynomial core) - NEEDS COEFFICIENT FITTING
- [x] R2: A2 (rational Padé) - NEEDS COEFFICIENT FITTING
- [x] R3: C3 (PWL ISPA) - NEEDS SEGMENT OPTIMIZATION
- [x] R4: B2 (tanh-form + odd rational) - NEEDS COEFFICIENT FITTING
- [x] R5: D1 (LUT + linear interpolation) - WORKS WELL

### Phase 2: ULP Analysis (Complete - Initial Results)
- [x] Run all implementations over entire bfloat16 range
- [x] Compute max-ULP, mean-ULP, percentile-ULP
- [x] Identify worst-case inputs (saturation boundaries)
- [x] Initial multi-region analysis

**Current ULP Results (with placeholder coefficients):**
| Method | Max ULP | Mean ULP | P99 ULP | Issue |
|--------|---------|----------|---------|-------|
| R1 Poly | 15760 | 36.0 | 3 | Bad coefficients for x<-1 |
| R2 Rational | 13760 | 21.8 | 5 | Saturation boundary |
| R3 PWL | 13760 | 55.7 | 98 | High baseline error |
| R4 Tanh | 15351 | 44.1 | 1 | tanh approx fails x<-2 |
| R5 LUT | 13760 | 18.7 | 0 | Best, limited by saturation |

### Phase 3: BF16 Optimization (Next Steps)
- [ ] Proper Remez/least-squares coefficient fitting
- [ ] Apply E1-E7 optimizations to best performers
- [ ] Coefficient quantization refinement
- [ ] FMA vs non-FMA comparison

---

## Key Design Decisions

### Data Types
- **Reference implementation**: Uses `double` (float64) for maximum precision
- **BFloat16 implementations**: Use `std::bfloat16_t` for input/output, `float` (float32) internally
- **Type punning**: Always use `std::memcpy()` to convert between bfloat16 and uint16 to avoid undefined behavior

### ULP Calculation Strategy
1. Enumerate all 65536 possible 16-bit patterns
2. Convert each to bfloat16 and filter out non-numeric values (NaN, Inf)
3. Sort valid values in ascending order
4. Assign ULP indices where equal values (e.g., +0 and -0) share the same index
5. Create a lookup table mapping each bit pattern to its ULP index
6. ULP error = |index(computed) - index(reference)|

### Key Insights
- **Tail saturation is critical**: BF16's 7-bit mantissa makes large |x| highly ULP-sensitive
- **Power-of-2 breakpoints**: Reduces quantization error in piecewise methods
- **Monotonicity constraints**: Prevents artifacts from coefficient rounding
- **FMA matters**: Can reduce ULP by 30-50% vs non-FMA evaluation
- **Real ULP will show the truth**: Theoretical expectations must be validated empirically

---

## Code Style Guidelines
- Use extensive comments explaining the rationale
- Prefer `std::memcpy()` over reinterpret_cast for type punning
- Use `static_assert` to verify type sizes at compile time
- Include verification/test functions for all implementations
- All GELU implementations take `std::bfloat16_t` and return `std::bfloat16_t`
- Reference GELU uses `double` for both input and output
- **Create reusable tests, not standalone scripts** - diagnostics and tests should be integrated into main source files with command-line options or conditional compilation
- **Avoid AI attribution in commits** - do not include AI tool names, co-author tags, or generated-by notices in commit messages to maintain neutrality
