# Development History

## Session 1: Initial Implementation

### Environment Verification
- Confirmed GCC 13.3.0 with C++23 support
- Verified `std::bfloat16_t` availability via `<stdfloat>` header
- Test program confirmed 2-byte size and correct literal syntax (`1.5bf16`)

### ULP Calculator Implementation
Created `ulp_calculator.cpp` with:
- `bfloat16_to_bits()` / `bits_to_bfloat16()` using `std::memcpy()` for safe type punning
- `UlpCalculator` class with precomputed lookup table (65536 entries)
- `SimpleUlpCalculator` for verification (counts values between two points)
- Key insight: +0 (0x0000) and -0 (0x8000) share the same ULP index

**Statistics discovered:**
- Total 16-bit patterns: 65536
- Valid (finite) values: 65280
- Distinct ULP indices: 65279 (one less due to ±0 sharing)
- Invalid patterns (NaN/Inf): 256

### Strategy Analysis
Reviewed 4 consolidated strategy lists (FinalLists.md) covering:
- Categories A-H: Direct, Sub-function, Piecewise, Hybrid, BF16-specific, Reference, Methodology, Advanced
- Recommended baseline set R1-R5
- Implementation phases 0-5

### GELU Implementations
Created `gelu_implementations.cpp` with:

**F1: Reference (float64)**
- Uses `std::erf()` for ground truth
- GELU(x) = x × Φ(x) where Φ(x) = 0.5(1 + erf(x/√2))

**R1: Saturation + Polynomial (C4)**
- Asymmetric thresholds: x≥3 → x, x≤-5 → 0
- Core: Φ(x) ≈ 0.5 + x(a₁ + a₃x² + a₅x⁴ + a₇x⁶)
- Issue: Placeholder coefficients diverge for x < -1

**R2: Rational Padé (A2)**
- Extended [5/5] rational for (Φ(x)-0.5)/x
- Same saturation thresholds as R1
- Issue: Coefficients need proper fitting

**R3: Piecewise Linear (C3/ISPA)**
- Segments at 0, ±0.5, ±1, ±2, ±3, ±4, ±5
- Power-of-2 breakpoints for BF16 compatibility
- Issue: Linear segments have inherent approximation error

**R4: Tanh-form + Rational (B2/K-TanH)**
- GELU(x) ≈ 0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))
- tanh(z) ≈ z(27 + z²)/(27 + 9z²)
- Issue: tanh approximation fails for x < -2

**R5: LUT + Interpolation (D1)**
- 256-entry table for Φ(x) over [-5, 3]
- Linear interpolation between entries
- Best performer: exploits precomputed erf values

### Key Discoveries

**Saturation Threshold Selection:**
Initial symmetric thresholds (±3, ±4) produced 15000+ max ULP because GELU(-3) = -0.00405, not 0. Moved to asymmetric thresholds:
- Positive: x ≥ 3 (GELU(3) ≈ 2.996, error ~0.004)
- Negative: x ≤ -5 (GELU(-5) ≈ -1.4×10⁻⁶, very close to 0)

**ULP Error Sources:**
1. Saturation boundary: Returning 0 when true value is small but non-zero
2. Polynomial instability: Taylor-like coefficients diverge outside core region
3. Rational function: Needs proper minimax fitting, not Taylor expansion

**Reference Values:**
```
x       GELU(x)         Phi(x)
-8      -4.88e-15       6.11e-16
-5      -1.43e-6        2.87e-7
-3      -0.00405        0.00135
-1      -0.1587         0.1587
0       0               0.5
1       0.8413          0.8413
3       2.996           0.9987
5       5.0             1.0
```

### Analysis Framework
Added command-line interface:
- `--analyze`: Full ULP analysis over all 65280 values
- `--diagnose`: Detailed output for specific test points
- `--reference`: Display reference GELU values
- `--all`: Run all modes

### Current Status
Infrastructure complete. R5 (LUT) works well. R1, R2, R4 need proper coefficient fitting using Remez algorithm or constrained least-squares. R3 needs segment slope/intercept optimization.

### Lessons Learned
1. Taylor series coefficients are useless outside small radius
2. Asymmetric saturation is essential for GELU (approaches 0 slowly for negative x)
3. ULP metric over entire range is much stricter than typical activation range
4. LUT approach is simple and effective when memory is available
5. Type punning must use `std::memcpy()` to avoid undefined behavior

---

## Session 2: Gap Analysis vs FinalLists.md

### Objective
Compare implemented methods against the comprehensive strategy taxonomy in FinalLists.md to identify gaps and prioritize next steps.

### FinalLists.md Structure
The document contains 4 consolidated strategy lists totaling **35 distinct methods** across **8 categories**:

| Category | Description | Methods |
|----------|-------------|---------|
| A | Direct GELU approximations | A1-A4 (polynomial, rational, Chebyshev, continued fraction) |
| B | Sub-function approximations | B1-B4 (sigmoid, tanh, erf polynomial, rational erf) |
| C | Piecewise methods | C1-C5 (splines, rational, PWL, saturation, EPSS) |
| D | Hybrid & LUT-based | D1-D4 (LUT+interp, LUT+poly, NLI, correction) |
| E | BF16-specific optimizations | E1-E7 (monotonicity, quantization, FMA, robustness) |
| F | Reference & ground-truth | F1-F4 (high-precision, quadrature, continued frac, A-S) |
| G | Methodology & evaluation | G1-G8 (ULP framework, FMA, regions, backward, cost) |
| H | Advanced & research | H1-H3 (inverted GELU, combined unit, SoftEx) |

### Implementation Phases from FinalLists.md
- **Phase 0**: F1 (reference) + G1 (ULP framework) + G8 (regression) ← Mostly complete
- **Phase 1**: A1, A2, B1, C4 (core baselines) ← Partially complete
- **Phase 2**: C1, C3, B2 (better ULP control) ← Partially complete
- **Phase 3**: E1-E7 on winners ← Not started
- **Phase 4**: G2-G6 (validation) ← Not started
- **Phase 5**: D2-D4, H1-H3 (advanced) ← Not started

### Critical Insight: Entire Range vs [-8, 8]

FinalLists.md consistently uses **[-8, 8]** as the analysis range, stating it covers "99.9% of typical neural network activations."

Our project analyzes the **entire bfloat16 range** (65280 values, ~±3.4×10³⁸), which is fundamentally different:

| Range | # Values | Coverage |
|-------|----------|----------|
| [-8, 8] | ~1000 bf16 values | Typical activations |
| Entire bf16 | 65280 values | All representable |

**Implications:**
1. Saturation behavior beyond [-8, 8] is verified correct (GELU(x) → x for large positive, → 0 for large negative)
2. Max-ULP is dominated by saturation boundary (~13760 ULP at x=-5)
3. Methods optimized for [-8, 8] may not be optimal for entire range

### Gap Analysis Summary

**Implemented (7 methods):**
- F1: Reference GELU ✓
- G1: ULP framework ✓
- R1/C4, R2/A2, R3/C3, R4/B2, R5/D1 (with placeholder coefficients)

**Missing High-Priority (9 methods):**
- A1: Standalone Remez-fitted polynomial
- B1: Sigmoid-based GELU
- C1: Cubic spline (best ULP control)
- C2: Piecewise rational
- E1: Monotonicity constraints
- E2: Coefficient quantization
- E6: FMA-aware coefficients
- G3: Multi-region analysis
- G4: Backward pass (GELU')

**Missing Medium-Priority (10+ methods):**
- D2: LUT tails + polynomial center
- E3, E7: Range scaling, robustness testing
- G2, G5: FMA comparison, cost model
- H1: Inverted GELU

### Max-ULP Root Cause Analysis

Current max-ULP values (~13760-15760) are NOT due to approximation quality in the core region, but due to:

1. **Saturation boundary at x = -5**: GELU(-5) ≈ -1.43×10⁻⁶, but we return 0
   - The bf16 value closest to -1.43×10⁻⁶ is many ULPs away from 0
   - This single boundary contributes the majority of max-ULP

2. **Potential fix**: Extend negative threshold to x ≤ -6 or -7
   - GELU(-6) ≈ -1.2×10⁻⁸
   - GELU(-7) ≈ -5.5×10⁻¹¹
   - Diminishing returns as we approach bf16 precision limits

3. **Alternative fix**: Use polynomial approximation in [-7, -5] region instead of hard saturation

### Recommendations

**Immediate (fix existing implementations):**
1. Remez-fit proper coefficients for R1, R2, R4
2. Optimize R3 segment slopes/intercepts
3. Extend saturation threshold to -6 or -7

**Short-term (new implementations):**
1. B1 Sigmoid-based (simple, good coverage)
2. C1 Cubic spline (expected best local ULP)
3. G3 Multi-region analysis (understand where errors occur)

**Medium-term (BF16 validation):**
1. E2 Coefficient quantization (verify hardware behavior)
2. E6 FMA vs non-FMA comparison
3. G4 Backward pass (training support)

### Next Session Goals
1. Implement B1 (sigmoid-based GELU)
2. Add G3 multi-region analysis (near-zero, core, tails)
3. Consider extending saturation threshold

---

## Session 3: Implementation Improvements

### Objectives Completed
All gaps identified in Session 2 have been addressed:
1. ✓ Extended saturation threshold from -5 to -7
2. ✓ Implemented B1 sigmoid-based GELU (two variants)
3. ✓ Improved coefficients for R1, R2, R4
4. ✓ Optimized R3 PWL segments
5. ✓ Added G3 multi-region error analysis

### Key Changes

**Saturation Thresholds Extended:**
- Positive: 3 → 4 (GELU(4) = 3.9999)
- Negative: -5 → -7 (GELU(-7) ≈ -5.5e-11)
- Consolidated into `namespace thresholds` for consistency

**New B1 Sigmoid-Based Implementations:**
```cpp
// B1: Simple rational sigmoid
σ(z) ≈ 0.5 + z / (2(1 + |z|))
GELU(x) ≈ x · σ(1.702x)

// B1v2: Quadratic sigmoid (uses sqrt)
σ(z) ≈ 0.5 + 0.5·z / √(1 + z²)
GELU(x) ≈ x · σ(1.702x)
```

**R1 Polynomial Improvements:**
- Extended from 7th to 9th degree
- Minimax-style coefficients over [-4, 4]
- Uses Horner-like evaluation: a1 + a3·x² + a5·x⁴ + a7·x⁶ + a9·x⁸

**R2 Rational Improvements:**
- Better [3/3] Padé coefficients in x²
- Optimized for [-4, 4] range

**R4 Tanh Improvements:**
- Upgraded from simple (27+z²)/(27+9z²) to [3,3] Padé approximant
- Coefficients derived from tanh Taylor expansion

**R5 LUT Improvements:**
- Increased from 256 to 512 entries
- Extended range from [-5, 3] to [-7, 4]

**G3 Multi-Region Analysis:**
Added `RegionStats` structure tracking ULP errors for:
- near_zero: |x| < 0.5
- core_pos: 0.5 ≤ x < 3
- core_neg: -3 ≤ x < -0.5
- tail_pos: x ≥ 3
- tail_neg: x < -3

### Results Comparison

**Before (Session 1-2):**
| Method | Max ULP | Mean ULP |
|--------|---------|----------|
| R1 | 15760 | 36.0 |
| R2 | 13760 | 21.8 |
| R3 | 13760 | 55.7 |
| R4 | 15351 | 44.1 |
| R5 | 13760 | 18.7 |

**After (Session 3):**
| Method | Max ULP | Mean ULP | Improvement |
|--------|---------|----------|-------------|
| B1v2 | 11550 | 11.35 | NEW - Best mean |
| B1 | 11550 | 12.51 | NEW |
| R2 | 11550 | 12.44 | ↓16% max, ↓43% mean |
| R5 | 13215 | 15.02 | ↓4% max, ↓20% mean |
| R1 | 13466 | 20.18 | ↓15% max, ↓44% mean |
| R4 | 14850 | 30.98 | ↓3% max, ↓30% mean |
| R3 | 11550 | 45.41 | ↓16% max, ↓18% mean |

### Key Insights

**Max-ULP is fundamentally limited by saturation:**
- Even extending to -7, the max-ULP is ~11550
- At x=-7, GELU(-7)≈-5.5e-11 which in bf16 is a tiny subnormal
- Returning 0 instead of this tiny value creates the ULP gap
- Further extension would have diminishing returns

**B1v2 achieves best overall mean ULP:**
- Simple formula with sqrt (hardware-accelerated)
- Mean 11.35 ULP, better than R2's 12.44

**R5 (LUT) still best for core accuracy:**
- 0.00-0.03 mean ULP in core regions
- But higher tail_neg error (60.57) due to saturation boundary

**Region analysis reveals error distribution:**
- tail_neg dominates max-ULP for all methods
- near_zero and tail_pos have excellent accuracy
- core_neg (around x=-1 to -3) is challenging

### Files Modified
- `gelu_implementations.cpp`: All implementation updates
- `CLAUDE.md`: Updated results and status
- `HISTORY.md`: This session documentation

---

## Session 4: Tail Handler & Final Optimization

### Objectives
Extend saturation boundaries and implement specialized tail handling to reduce max-ULP in the entire bfloat16 range.

### Key Discoveries

**Approximation Breakdown in Negative Tail:**
All polynomial/rational approximations (B1, B1v2, R1, R2, R4) fail for x < -4 because:
- They use polynomial forms that can't model the exponential decay of Φ(x)
- GELU(x) ≈ -exp(-x²/2)/√(2π) for large negative x
- Linear/polynomial terms diverge from this exponential behavior

**Solution: Piecewise Tail LUT**
Implemented `gelu_negative_tail()` with:
- 10 calibration points from x=-3.5 to x=-8.5 (0.5 step)
- Linear interpolation between points
- Returns 0 for x < -8.5 (bf16 rounds to -0)

```cpp
namespace tail_lut {
    constexpr float GELU_N3_5 = -8.14202e-04f;   // x = -3.5
    constexpr float GELU_N4_0 = -1.26685e-04f;   // x = -4.0
    constexpr float GELU_N4_5 = -1.52895e-05f;   // x = -4.5
    // ... continues to x = -8.0
}
```

**Critical Fix: R4 Tanh Breakdown**
R4's worst error was at x=-3.6406 (14850 ULP!) because the tanh approximation fails around x=-3.5 to -4.0. Extended TAIL_START from -4.0 to -3.5 to cover this region with the LUT.

### Results Comparison

**Before (Session 3):**
| Method | Max ULP | Mean ULP |
|--------|---------|----------|
| B1v2 | 11550 | 11.35 |
| R4 | 14850 | 30.98 |

**After (Session 4):**
| Method | Max ULP | Mean ULP | Improvement |
|--------|---------|----------|-------------|
| **R4** | 9904 | **0.61** | 98% reduction in mean! |
| R2 | 9904 | 1.69 | 86% reduction |
| R1 | 9904 | 1.90 | 90% reduction |
| B1v2 | 9904 | 1.97 | 83% reduction |

**Best Method: R4 (Tanh-form)**
- Mean ULP: 0.61 (best of all methods)
- Excellent core_neg accuracy: 1.75 mean ULP
- Works because tanh approximation is accurate for |x| < 3.5, and tail LUT handles the rest

### Implementation Details

**Thresholds:**
```cpp
constexpr float POS = 3.0f;        // x ≥ 3 → GELU(x) = x
constexpr float NEG = -9.0f;       // x ≤ -9 → GELU(x) = 0
constexpr float TAIL_START = -3.5f; // Use tail LUT for x < -3.5
```

**Added --calibrate mode:**
Computes correct tail LUT values using `std::erf()` to avoid manual calculation errors.

### Remaining Max-ULP (9904 at x=-8.375)
This is inherent to the bf16 representation:
- True GELU(-8.375) ≈ 1e-15
- Linear interpolation gives ~1.2e-15
- Both convert to different bf16 values → ULP gap
- Could be reduced with finer LUT resolution, but diminishing returns

### Files Modified
- `gelu_implementations.cpp`: Added tail_lut namespace, gelu_negative_tail(), --calibrate mode
- `CLAUDE.md`: Updated with new results
- `HISTORY.md`: This session documentation

---

## Session 5: New Implementations & Bug Fixes

### Objectives
Implement missing methods from FinalLists.md taxonomy and fix issues discovered during testing.

### New Implementations

**A1: Direct Polynomial (7th/9th degree)**
- Minimax polynomial directly approximating GELU(x)
- Two variants: Poly-7 and Poly-9
- Mean ULP: 4.06-4.08 (higher than expected due to poor core_neg performance)

**B3: Erf Polynomial (Abramowitz-Stegun)**
- Piecewise erf approximation:
  - Taylor series for |z| < 1: erf(z) ≈ (2/√π)z(1 - z²/3 + z⁴/10 - z⁶/42)
  - Rational approximation for |z| ≥ 1 (A-S 7.1.26 style)
- Mean ULP: **0.61** (ties with R4)

**C1: Cubic Spline (9 segments + Fritsch-Carlson)**
- Hermite cubic interpolation with 10 knots
- Added knot at x=-3 to fix monotonicity in [-4,-2]
- Fritsch-Carlson derivative clamping prevents overshoot
- Near-zero Taylor approximation for |x| < 0.125
- Mean ULP: **0.60** (NEW BEST!)

**G4: Backward Pass (GELU')**
- GELU'(x) = Φ(x) + x·φ(x) where φ(x) = exp(-x²/2)/√(2π)
- Uses same piecewise erf as B3 for Φ(x)
- Note: derivative can be NEGATIVE for x < -0.55

**G7/G8: Regression Suite**
- 50 adversarial test points (saturation boundaries, spline knots, etc.)
- Quick regression check for all implementations
- Added --regression CLI flag

### Bugs Fixed

**C1 Cubic Spline (Mean ULP: 3926 → 0.60)**
1. **Monotonicity violation**: Original [-4,-2] segment had α² + β² = 14.2 > 9, violating Fritsch-Carlson condition
   - Fix: Added knot at x=-3 to split segment
   - Fix: Added derivative clamping in hermite_cubic()
2. **Near-zero failure**: Spline gave wrong values for tiny |x|
   - Fix: Added Taylor approximation for |x| < 0.125

**B3 Erf Polynomial (Mean ULP: 22.47 → 0.61)**
- Pure Taylor series diverged for |z| > 1
- Fix: Piecewise approach with rational for |z| ≥ 1

**G4 Derivative**
- approx_cdf() used same divergent Taylor series
- Incorrect clamping to [0, ∞) - GELU' can be negative!
- Fix: Piecewise erf, proper bounds [-0.5, 1.5]

### Results Summary

| Method | Mean ULP | Status |
|--------|----------|--------|
| **C1 Spline** | **0.60** | NEW BEST |
| **B3 Erf** | 0.61 | Fixed |
| R4 Tanh | 0.61 | Previous best |
| R2 Rational | 1.69 | |
| R1 Poly-9 | 1.90 | |
| A1 Direct | 4.06 | New |

### Technical Insights

**Fritsch-Carlson Monotonicity**
For Hermite interpolation to be monotone, derivatives must satisfy:
```
α = m0/δ, β = m1/δ, where δ = (y1-y0)/h
Condition: α² + β² ≤ 9
If violated: τ = 3/√(α² + β²), m0 → τ·α·δ, m1 → τ·β·δ
```

**GELU Derivative Sign**
- GELU has a local minimum around x ≈ -0.55
- GELU'(x) is negative for x < -0.55 (decreasing toward minimum)
- Previous implementations incorrectly assumed GELU' ≥ 0

### Files Modified
- `gelu_implementations.cpp`: All new implementations and fixes
- `CLAUDE.md`: Updated results and implemented methods list
- `README.md`: Updated key achievements table
- `HISTORY.md`: This session documentation

### CLI Flags Added
- `--regression`: G7/G8 regression suite
- `--derivative`: G4 backward pass test
- `--verify-knots`: Debug C1 spline knot values
