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
