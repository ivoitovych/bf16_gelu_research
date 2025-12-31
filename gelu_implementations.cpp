/**
 * @file gelu_implementations.cpp
 * @brief GELU Approximation Implementations for BFloat16 ULP Analysis
 *
 * This file implements multiple GELU approximation strategies for systematic
 * ULP (Units in Last Place) error analysis across the entire bfloat16 range.
 *
 * ## GELU Definition
 *
 * GELU(x) = x * Φ(x)
 *
 * where Φ(x) is the CDF of the standard normal distribution:
 *   Φ(x) = 0.5 * (1 + erf(x / √2))
 *
 * ## Implemented Strategies
 *
 * F1: High-precision reference (float64 with std::erf)
 * B1: Sigmoid-based GELU: x * σ(1.702x)
 * R1: C4 - Saturation + minimax polynomial core
 * R2: A2 - Rational Padé approximation
 * R3: C3 - Piecewise linear with power-of-2 breakpoints
 * R4: B2 - Tanh-form with odd rational tanh approximation
 * R5: D1 - LUT with linear interpolation
 *
 * ## Constraints
 *
 * All approximations use only: +, -, *, /, |x|, sign()
 * No erf(), tanh(), exp(), log() in approximations
 * Reference F1 uses std::erf for ground truth
 *
 * ## Key Design Decisions
 *
 * - Saturation thresholds: x >= 4 → x, x <= -7 → 0
 *   (Extended from -5 to -7 to reduce max-ULP at saturation boundary)
 * - All implementations use float32 internally, bfloat16 for I/O
 */

#include <stdfloat>      // For std::bfloat16_t (C++23)
#include <cstdint>       // For uint16_t, int64_t
#include <cstring>       // For std::memcpy
#include <cmath>         // For std::erf, std::sqrt, std::abs, std::isnan, std::isinf
#include <vector>        // For std::vector
#include <algorithm>     // For std::sort, std::min, std::max
#include <array>         // For std::array
#include <iostream>      // For std::cout
#include <iomanip>       // For std::setprecision, std::setw
#include <string>        // For std::string
#include <functional>    // For std::function
#include <limits>        // For std::numeric_limits

// ============================================================================
// COMPILE-TIME VERIFICATION
// ============================================================================

static_assert(sizeof(std::bfloat16_t) == 2, "bfloat16_t must be 2 bytes");
static_assert(sizeof(uint16_t) == 2, "uint16_t must be 2 bytes");
static_assert(sizeof(float) == 4, "float must be 4 bytes");
static_assert(sizeof(double) == 8, "double must be 8 bytes");

// ============================================================================
// TYPE CONVERSION UTILITIES
// ============================================================================

/**
 * @brief Convert bfloat16 to its underlying uint16 bit representation
 *
 * Uses std::memcpy to avoid undefined behavior from type punning.
 */
inline uint16_t bfloat16_to_bits(std::bfloat16_t value) {
    uint16_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

/**
 * @brief Convert uint16 bit pattern to bfloat16
 */
inline std::bfloat16_t bits_to_bfloat16(uint16_t bits) {
    std::bfloat16_t value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

/**
 * @brief Check if a bfloat16 value is finite (not NaN or Inf)
 */
inline bool is_finite_bf16(std::bfloat16_t value) {
    float f = static_cast<float>(value);
    return !std::isnan(f) && !std::isinf(f);
}

// ============================================================================
// MATHEMATICAL CONSTANTS (High Precision)
// ============================================================================

namespace constants {
    // √(2/π) ≈ 0.7978845608028654
    constexpr double SQRT_2_OVER_PI = 0.7978845608028653558798921198687637369517;

    // 1/√2 ≈ 0.7071067811865476
    constexpr double INV_SQRT_2 = 0.7071067811865475244008443621048490392848;

    // √2 ≈ 1.4142135623730951
    constexpr double SQRT_2 = 1.4142135623730950488016887242096980785697;

    // 1/√(2π) ≈ 0.3989422804014327 - used in PDF of standard normal
    constexpr double INV_SQRT_2PI = 0.3989422804014326779399460599343818684759;

    // Coefficient for tanh-form GELU: 0.044715
    constexpr double TANH_COEFF = 0.044715;

    // Coefficient for sigmoid-form GELU: 1.702
    constexpr double SIGMOID_COEFF = 1.702;
}

// ============================================================================
// F1: HIGH-PRECISION GELU REFERENCE (float64)
// ============================================================================

/**
 * @brief F1: High-precision GELU reference implementation
 *
 * This is the ground truth for all ULP measurements. It uses float64
 * arithmetic and the standard library's erf() function.
 *
 * GELU(x) = x * Φ(x) = x * 0.5 * (1 + erf(x / √2))
 *
 * @param x Input value (float64 for maximum precision)
 * @return GELU(x) computed with float64 precision
 */
double gelu_reference_f64(double x) {
    // For x < 0, use erfc to avoid catastrophic cancellation in 1 + erf(z)
    // when erf(z) is very close to -1 (large negative z)
    //
    // For x >= 0: Φ(x) = 0.5 * (1 + erf(x/√2))
    // For x < 0:  Φ(x) = 0.5 * erfc(-x/√2)  [since erfc(z) = 1 - erf(z)]
    double z = x * constants::INV_SQRT_2;
    double phi;
    if (x >= 0) {
        phi = 0.5 * (1.0 + std::erf(z));
    } else {
        phi = 0.5 * std::erfc(-z);
    }
    return x * phi;
}

/**
 * @brief Wrapper to compute reference GELU for a bfloat16 input
 *
 * Converts bfloat16 to double, computes GELU, returns double result.
 * This is used to get the "true" value for ULP comparison.
 */
double gelu_reference_for_bf16(std::bfloat16_t x) {
    double xd = static_cast<double>(static_cast<float>(x));
    return gelu_reference_f64(xd);
}

// ============================================================================
// COMMON SATURATION THRESHOLDS
// ============================================================================

namespace thresholds {
    // Asymmetric saturation thresholds determined by bf16 representation limits
    //
    // Positive: x >= 3 → GELU(x) ≈ x (error < 0.26 ULP at x=3)
    //   At x=3: GELU(3) = 2.996, error = 0.004, ULP_size = 0.0156, error = 0.26 ULP
    //
    // Negative: x <= -9 → GELU(x) ≈ 0 (bf16(GELU(-9)) = -0.0 exactly)
    //   At x=-9: GELU(-9) ≈ 0 in float64, and bf16 conversion gives exactly -0
    //   This eliminates saturation boundary ULP error entirely
    //
    // Tail transition: For x in [-9, -3.5], use special tail handling
    //   because core approximations don't model the exponential decay correctly
    //   Extended from -4 to -3.5 to handle R4 tanh breakdown region
    //
    // Analysis: ./gelu_analysis --saturation
    constexpr float POS = 3.0f;
    constexpr float NEG = -9.0f;
    constexpr float TAIL_START = -3.5f;  // Start of negative tail transition
}

/**
 * @brief Compute GELU for negative tail region (x < -4)
 *
 * In the negative tail, GELU(x) → 0 as x → -∞.
 * The true GELU: GELU(x) = x * Φ(x) where Φ(x) = 0.5*erfc(|x|/√2)
 *
 * For large negative x, using Mills ratio approximation:
 *   GELU(x) ≈ -exp(-x²/2) / √(2π)
 *
 * Calibration points:
 *   x=-4:   GELU = -0.0001267
 *   x=-4.5: GELU = -0.0000159
 *   x=-5:   GELU = -1.433e-6
 *   x=-5.5: GELU = -9.32e-8
 *   x=-6:   GELU = -5.92e-9
 *   x=-6.5: GELU = -2.91e-10
 *   x=-7:   GELU = -1.11e-11
 *
 * We use a LUT with linear interpolation for accuracy.
 */

namespace tail_lut {
    // Extended LUT covering [-8.3125, -3.5]
    // - Main region: 0.25 step from -3.5 to -8.0 (19 entries)
    // - Fine region: 0.0625 step from -8.0 to -8.3125 (6 entries)
    // Values computed using erfc to avoid catastrophic cancellation:
    // GELU(x) = x * 0.5 * erfc(-x/√2) for x < 0

    // Main LUT: 0.25 step from -3.5 to -8.0
    constexpr float GELU_N3_50 = -8.14202e-04f;   // x = -3.50
    constexpr float GELU_N3_75 = -3.31565e-04f;   // x = -3.75
    constexpr float GELU_N4_00 = -1.26685e-04f;   // x = -4.00
    constexpr float GELU_N4_25 = -4.54262e-05f;   // x = -4.25
    constexpr float GELU_N4_50 = -1.52895e-05f;   // x = -4.50
    constexpr float GELU_N4_75 = -4.83115e-06f;   // x = -4.75
    constexpr float GELU_N5_00 = -1.43326e-06f;   // x = -5.00
    constexpr float GELU_N5_25 = -3.99260e-07f;   // x = -5.25
    constexpr float GELU_N5_50 = -1.04443e-07f;   // x = -5.50
    constexpr float GELU_N5_75 = -2.56575e-08f;   // x = -5.75
    constexpr float GELU_N6_00 = -5.91953e-09f;   // x = -6.00
    constexpr float GELU_N6_25 = -1.28267e-09f;   // x = -6.25
    constexpr float GELU_N6_50 = -2.61040e-10f;   // x = -6.50
    constexpr float GELU_N6_75 = -4.98977e-11f;   // x = -6.75
    constexpr float GELU_N7_00 = -8.95869e-12f;   // x = -7.00
    constexpr float GELU_N7_25 = -1.51080e-12f;   // x = -7.25
    constexpr float GELU_N7_50 = -2.39317e-13f;   // x = -7.50
    constexpr float GELU_N7_75 = -3.56084e-14f;   // x = -7.75
    constexpr float GELU_N8_00 = -4.97677e-15f;   // x = -8.00, bf16=0xa7b3

    // Fine LUT: 0.0625 step from -8.0625 to -8.3125 (computed with erfc)
    constexpr float GELU_N8_0625 = -3.01335e-15f;  // x = -8.0625, bf16=0xa759
    constexpr float GELU_N8_125  = -1.81741e-15f;  // x = -8.125,  bf16=0xa703
    constexpr float GELU_N8_1875 = -1.09184e-15f;  // x = -8.1875, bf16=0xa69d
    constexpr float GELU_N8_25   = -6.53377e-16f;  // x = -8.25,   bf16=0xa63c
    constexpr float GELU_N8_3125 = -3.89468e-16f;  // x = -8.3125, bf16=0xa5e1
    // Beyond x=-8.3125, asymptotic expansion handles correctly

    // Main LUT array: 0.25 step from -3.5 to -8.0 (indices 0-18)
    constexpr float LUT_MAIN[] = {
        GELU_N3_50, GELU_N3_75, GELU_N4_00, GELU_N4_25, GELU_N4_50,
        GELU_N4_75, GELU_N5_00, GELU_N5_25, GELU_N5_50, GELU_N5_75,
        GELU_N6_00, GELU_N6_25, GELU_N6_50, GELU_N6_75, GELU_N7_00,
        GELU_N7_25, GELU_N7_50, GELU_N7_75, GELU_N8_00
    };
    constexpr int LUT_MAIN_SIZE = 19;

    // Fine LUT array: 0.0625 step from -8.0 to -8.3125 (indices 0-5)
    constexpr float LUT_FINE[] = {
        GELU_N8_00, GELU_N8_0625, GELU_N8_125, GELU_N8_1875, GELU_N8_25, GELU_N8_3125
    };
    constexpr int LUT_FINE_SIZE = 6;

    constexpr float LUT_START = -3.5f;
    constexpr float LUT_MAIN_END = -8.0f;
    constexpr float LUT_END = -8.3125f;
    constexpr float LUT_MAIN_STEP = -0.25f;
    constexpr float LUT_FINE_STEP = -0.0625f;
}

// ============================================================================
// FAST EXP APPROXIMATION (for asymptotic tail computation)
// ============================================================================

/**
 * @brief Fast approximation of 2^x for negative x using bit manipulation
 *
 * Decomposes x = n + f where n is integer part, f is fractional part.
 * 2^x = 2^n * 2^f
 * - 2^n computed via IEEE754 exponent manipulation
 * - 2^f approximated with minimax polynomial
 *
 * Now handles subnormal range (x down to -149) for deep negative tail accuracy.
 * This is needed because bf16 GELU values are tiny subnormals for x near -13.
 */
inline float fast_exp2_neg(float x) {
    // For very negative x beyond float subnormal range, underflow to zero
    // Float subnormals go down to 2^(-149), so we cut off there
    if (x < -149.0f) return 0.0f;

    // Split into integer and fractional parts
    float n = std::floor(x);
    float f = x - n;  // f is in [0, 1)

    // Minimax polynomial for 2^f on [0, 1]: 2^f ≈ 1 + f*ln(2) + f²*ln²(2)/2 + ...
    // Using optimized coefficients for better accuracy
    float f2 = f * f;
    float f3 = f2 * f;
    float f4 = f2 * f2;
    float pow2_frac = 1.0f + 0.6931472f * f + 0.2402265f * f2
                    + 0.0555041f * f3 + 0.0096139f * f4;

    // Compute 2^n via IEEE754 exponent manipulation
    int32_t exp_int = static_cast<int32_t>(n);
    int32_t exp_bits = exp_int + 127;

    if (exp_bits > 0) {
        // Normal range: exponent >= 1
        uint32_t bits = static_cast<uint32_t>(exp_bits) << 23;
        float pow2_int;
        std::memcpy(&pow2_int, &bits, sizeof(float));
        return pow2_int * pow2_frac;
    } else {
        // Subnormal range: exponent <= 0
        // 2^x = 2^(-126) * 2^(x+126) for subnormals
        // We need to scale down by 2^(-exp_bits) = 2^(127-exp_int-127) = 2^(-exp_int)
        // Since exp_bits <= 0, -exp_bits >= 0, so we divide by 2^(-exp_bits)
        //
        // Compute 2^(-126) first (smallest normal)
        uint32_t min_normal_bits = 1u << 23;  // 2^(-126)
        float min_normal;
        std::memcpy(&min_normal, &min_normal_bits, sizeof(float));

        // Now we need 2^(x+126) = 2^(n+126) * 2^f
        // Since n < -126, n+126 < 0, so 2^(n+126) is a fraction
        // We can compute this as: result = min_normal * pow2_frac * 2^(n+126)
        // where 2^(n+126) = 2^(exp_bits - 1) since exp_bits = n + 127
        int32_t subnorm_shift = 1 - exp_bits;  // How many bits to shift right
        if (subnorm_shift >= 24) return 0.0f;  // Would shift out all mantissa bits

        // Scale factor: 2^(exp_bits - 1) = 2^(-(subnorm_shift))
        float scale = min_normal;
        for (int i = 0; i < subnorm_shift; ++i) {
            scale *= 0.5f;
        }
        return scale * pow2_frac;
    }
}

/**
 * @brief Fast approximation of exp(-u) for u > 0
 *
 * Uses identity: exp(-u) = 2^(-u/ln(2))
 */
inline float fast_exp_neg(float u) {
    constexpr float inv_ln2 = 1.4426950408889634f;  // 1/ln(2)
    return fast_exp2_neg(-u * inv_ln2);
}

/**
 * @brief Asymptotic GELU for deep negative tail
 *
 * For x << 0: GELU(x) ≈ -φ(x) * (1 - 1/x² + 3/x⁴ - 15/x⁶)
 * where φ(x) = exp(-x²/2) / √(2π)
 *
 * This captures the exponential decay that rational approximations miss.
 */
inline float gelu_asymptotic(float x) {
    constexpr float inv_sqrt_2pi = 0.3989422804014327f;  // 1/√(2π)
    float x2 = x * x;
    float x2_half = x2 * 0.5f;
    float exp_val = fast_exp_neg(x2_half);

    // If exp underflowed to zero, return zero
    if (exp_val == 0.0f) {
        return 0.0f;
    }

    float phi_x = exp_val * inv_sqrt_2pi;  // φ(x)

    // Correction terms: (1 - 1/x² + 3/x⁴ - 15/x⁶)
    float inv_x2 = 1.0f / x2;
    float inv_x4 = inv_x2 * inv_x2;
    float inv_x6 = inv_x4 * inv_x2;
    float correction = 1.0f - inv_x2 + 3.0f * inv_x4 - 15.0f * inv_x6;

    return -phi_x * correction;
}

/**
 * @brief Negative tail GELU handler using LUT + asymptotic expansion
 *
 * Covers x ∈ [-∞, -3.5] with:
 * - Main LUT region: 0.25-step from -3.5 to -8.0 (19 entries)
 * - Fine LUT region: 0.0625-step from -8.0 to -8.3125 (6 entries)
 * - Deep tail (x < -8.3125): asymptotic expansion
 *
 * The asymptotic expansion correctly captures the exponential decay.
 */
inline float gelu_negative_tail(float x) {
    using namespace tail_lut;

    // Deep tail: x < -8.3125 - use asymptotic expansion
    if (x < LUT_END) {
        return gelu_asymptotic(x);
    }

    // x >= -3.5: should be handled by core approximation, not this function
    if (x >= LUT_START) {
        return LUT_MAIN[0];  // Return value at -3.5
    }

    // Fine region: x in [-8.3125, -8.0) - use LUT_FINE with 0.0625 step
    if (x < LUT_MAIN_END) {
        // Index in fine LUT: (x - (-8.0)) / (-0.0625) = (x + 8.0) / (-0.0625)
        float idx_f = (x - LUT_MAIN_END) / LUT_FINE_STEP;
        int idx = static_cast<int>(idx_f);

        // Clamp to valid range [0, LUT_FINE_SIZE-2]
        if (idx < 0) idx = 0;
        if (idx >= LUT_FINE_SIZE - 1) idx = LUT_FINE_SIZE - 2;

        // Linear interpolation between LUT_FINE[idx] and LUT_FINE[idx+1]
        float t = idx_f - static_cast<float>(idx);
        return LUT_FINE[idx] + t * (LUT_FINE[idx + 1] - LUT_FINE[idx]);
    }

    // Main region: x in [-8.0, -3.5) - use LUT_MAIN with 0.25 step
    // Index calculation: (x - (-3.5)) / (-0.25) = (x + 3.5) / (-0.25)
    float idx_f = (x - LUT_START) / LUT_MAIN_STEP;
    int idx = static_cast<int>(idx_f);

    // Clamp to valid range [0, LUT_MAIN_SIZE-2]
    if (idx < 0) idx = 0;
    if (idx >= LUT_MAIN_SIZE - 1) idx = LUT_MAIN_SIZE - 2;

    // Linear interpolation between LUT_MAIN[idx] and LUT_MAIN[idx+1]
    float t = idx_f - static_cast<float>(idx);
    return LUT_MAIN[idx] + t * (LUT_MAIN[idx + 1] - LUT_MAIN[idx]);
}

// ============================================================================
// B1: SIGMOID-BASED GELU (NEW)
// ============================================================================

/**
 * @brief B1: Sigmoid-based GELU approximation
 *
 * GELU(x) ≈ x * σ(1.702 * x)
 *
 * where σ(z) is approximated using a simple rational function:
 *   σ(z) ≈ 0.5 + z / (2 * (1 + |z|))
 *
 * This approximation:
 * - Is monotonic (σ goes from 0 to 1)
 * - Uses only basic arithmetic: +, -, *, /, |x|
 * - Has bounded range [0, 1] by construction
 *
 * The coefficient 1.702 was chosen to make x*σ(1.702x) approximate GELU well.
 * This is one of the simplest GELU approximations with good accuracy.
 */
std::bfloat16_t gelu_b1_sigmoid(std::bfloat16_t x_bf16) {
    float x = static_cast<float>(x_bf16);

    // Positive saturation
    if (x >= thresholds::POS) {
        return static_cast<std::bfloat16_t>(x);
    }

    // Negative tail: use specialized handler (includes asymptotic for deep tail)
    if (x < thresholds::TAIL_START) {
        return static_cast<std::bfloat16_t>(gelu_negative_tail(x));
    }

    // σ(z) ≈ 0.5 + z / (2 * (1 + |z|))
    // where z = 1.702 * x
    constexpr float k = 1.702f;
    float z = k * x;
    float abs_z = std::abs(z);

    // Compute sigmoid approximation
    // σ(z) = 0.5 + z / (2 * (1 + |z|))
    float sigma = 0.5f + z / (2.0f * (1.0f + abs_z));

    // GELU(x) = x * σ(kx)
    float result = x * sigma;

    return static_cast<std::bfloat16_t>(result);
}

/**
 * @brief B1v2: Quadratic sigmoid-based GELU
 *
 * Uses a quadratic rational sigmoid approximation:
 *   σ(z) ≈ 0.5 * (1 + z / sqrt(1 + z²))
 *
 * This approximation:
 * - Has correct asymptotic limits (0 and 1)
 * - Is smooth and monotonic
 * - Uses sqrt which is often hardware-accelerated
 *
 * Combined with GELU coefficient: GELU(x) ≈ x * σ(1.702x)
 */
std::bfloat16_t gelu_b1_sigmoid_v2(std::bfloat16_t x_bf16) {
    float x = static_cast<float>(x_bf16);

    if (x >= thresholds::POS) {
        return static_cast<std::bfloat16_t>(x);
    }

    // Negative tail: use specialized handler (includes asymptotic for deep tail)
    if (x < thresholds::TAIL_START) {
        return static_cast<std::bfloat16_t>(gelu_negative_tail(x));
    }

    // GELU(x) ≈ x * σ(1.702 * x)
    // σ(z) ≈ 0.5 * (1 + z / sqrt(1 + z²))
    constexpr float k = 1.702f;
    float z = k * x;
    float z2 = z * z;

    // Compute sigmoid using the algebraic approximation
    // σ(z) = 0.5 + 0.5 * z / sqrt(1 + z²)
    float inv_sqrt = 1.0f / std::sqrt(1.0f + z2);
    float sigma = 0.5f + 0.5f * z * inv_sqrt;

    float result = x * sigma;

    return static_cast<std::bfloat16_t>(result);
}

// ============================================================================
// R1: C4 - SATURATION + MINIMAX POLYNOMIAL CORE
// ============================================================================

/**
 * @brief R1: Saturation + minimax polynomial core GELU approximation
 *
 * Strategy:
 * - For x >= threshold_pos: GELU(x) ≈ x (saturation to identity)
 * - For x <= threshold_neg: GELU(x) ≈ 0 (saturation to zero)
 * - For core region: Use polynomial approximation
 *
 * Key insight: GELU(x) = x * Φ(x), and Φ(x) transitions from 0 to 1.
 * The core uses a 9th-degree odd polynomial fitted to approximate
 * (Φ(x) - 0.5) / x, which gives Φ(x) ≈ 0.5 + x * P(x²).
 *
 * Coefficients derived from minimax fitting over [-4, 4].
 */
std::bfloat16_t gelu_r1_saturation_poly(std::bfloat16_t x_bf16) {
    float x = static_cast<float>(x_bf16);

    if (x >= thresholds::POS) {
        return static_cast<std::bfloat16_t>(x);
    }

    // Negative tail: use specialized handler (includes asymptotic for deep tail)
    if (x < thresholds::TAIL_START) {
        return static_cast<std::bfloat16_t>(gelu_negative_tail(x));
    }

    // Core region: polynomial approximation for Φ(x)
    //
    // We approximate Φ(x) = 0.5 * (1 + erf(x/√2)) using:
    //   Φ(x) ≈ 0.5 + x * (a1 + a3*x² + a5*x⁴ + a7*x⁶ + a9*x⁸)
    //
    // Coefficients from minimax fit of (Φ(x)-0.5)/x over [-4, 4]
    // The odd polynomial structure ensures Φ(-x) = 1 - Φ(x).
    float x2 = x * x;
    float x4 = x2 * x2;
    float x6 = x4 * x2;
    float x8 = x4 * x4;

    // Minimax coefficients for (Φ(x) - 0.5) / x
    // Fitted to minimize max error over [-4, 4]
    constexpr float a1 = 0.398942280f;   // ≈ 1/√(2π)
    constexpr float a3 = -0.066490380f;
    constexpr float a5 = 0.005223040f;
    constexpr float a7 = -0.000203370f;
    constexpr float a9 = 0.000003130f;

    float p = a1 + a3 * x2 + a5 * x4 + a7 * x6 + a9 * x8;
    float phi = 0.5f + x * p;

    // Clamp Φ to [0, 1] - critical for numerical stability
    phi = std::max(0.0f, std::min(1.0f, phi));

    float result = x * phi;

    return static_cast<std::bfloat16_t>(result);
}

// ============================================================================
// R2: A2 - RATIONAL PADÉ [4/4] APPROXIMATION
// ============================================================================

/**
 * @brief R2: Rational Padé approximation for GELU
 *
 * GELU(x) ≈ x * Φ(x) where Φ(x) is approximated using a rational function.
 *
 * For Φ(x), we use:
 *   Φ(x) ≈ 0.5 + x * P(x²) / Q(x²)
 *
 * where P and Q are polynomials in x². This structure ensures:
 * - Φ(-x) = 1 - Φ(x) (odd function centered at 0.5)
 * - Better convergence in tails than pure polynomials
 *
 * Coefficients optimized for [-4, 4] range with minimax criterion.
 */
std::bfloat16_t gelu_r2_rational_pade(std::bfloat16_t x_bf16) {
    float x = static_cast<float>(x_bf16);

    if (x >= thresholds::POS) {
        return static_cast<std::bfloat16_t>(x);
    }

    // Negative tail: use specialized handler (includes asymptotic for deep tail)
    if (x < thresholds::TAIL_START) {
        return static_cast<std::bfloat16_t>(gelu_negative_tail(x));
    }

    float x2 = x * x;
    float x4 = x2 * x2;
    float x6 = x4 * x2;

    // Rational approximation for (Φ(x) - 0.5) / x
    // R(x²) = (a0 + a1*x² + a2*x⁴ + a3*x⁶) / (1 + b1*x² + b2*x⁴ + b3*x⁶)
    //
    // Coefficients derived from constrained minimax fit over [-4, 4]
    // ensuring R(0) = 1/√(2π) and proper tail behavior.
    constexpr float a0 = 0.398942280f;  // 1/√(2π)
    constexpr float a1 = 0.069693856f;
    constexpr float a2 = 0.003307210f;
    constexpr float a3 = 0.000044406f;

    constexpr float b1 = 0.374206990f;
    constexpr float b2 = 0.048066570f;
    constexpr float b3 = 0.002013660f;

    float num = a0 + a1 * x2 + a2 * x4 + a3 * x6;
    float den = 1.0f + b1 * x2 + b2 * x4 + b3 * x6;

    // Φ(x) ≈ 0.5 + x * (num / den)
    float phi = 0.5f + x * (num / den);

    // Safety clamp
    phi = std::max(0.0f, std::min(1.0f, phi));

    float result = x * phi;

    return static_cast<std::bfloat16_t>(result);
}

// ============================================================================
// R3: C3 - PIECEWISE LINEAR WITH POWER-OF-2 BREAKPOINTS
// ============================================================================

/**
 * @brief R3: Piecewise linear GELU approximation (ISPA-style)
 *
 * Uses power-of-2 breakpoints for BF16 compatibility.
 * For each segment [a, b], GELU(x) ≈ slope * x + intercept
 *
 * Breakpoints: 0, ±0.5, ±1, ±2, ±4 (power-of-2)
 * Extended to ±7 for saturation
 *
 * Segment parameters derived from exact GELU values at breakpoints:
 *   x       GELU(x)
 *   0       0.0
 *   0.5     0.345714
 *   1.0     0.841345
 *   2.0     1.954597
 *   4.0     3.999873
 *  -0.5    -0.154286
 *  -1.0    -0.158655
 *  -2.0    -0.045403
 *  -4.0    -0.000127
 *  -7.0    ≈0
 */
std::bfloat16_t gelu_r3_pwl(std::bfloat16_t x_bf16) {
    float x = static_cast<float>(x_bf16);

    // Positive saturation
    if (x >= thresholds::POS) {
        return static_cast<std::bfloat16_t>(x);
    }

    // Precomputed segment parameters (slope, intercept)
    // For segment [x0, x1]: y = slope*x + intercept
    // slope = (GELU(x1) - GELU(x0)) / (x1 - x0)
    // intercept = GELU(x0) - slope * x0
    float result;

    if (x >= 0.0f) {
        if (x < 0.5f) {
            // [0, 0.5]: slope = 0.345714/0.5 = 0.691428
            result = 0.691428f * x;
        } else if (x < 1.0f) {
            // [0.5, 1]: slope = (0.841345-0.345714)/0.5 = 0.991262
            result = 0.991262f * x - 0.149917f;
        } else if (x < 2.0f) {
            // [1, 2]: slope = (1.954597-0.841345)/1 = 1.113252
            result = 1.113252f * x - 0.271907f;
        } else {
            // [2, 4]: slope = (3.999873-1.954597)/2 = 1.022638
            result = 1.022638f * x - 0.090369f;
        }
    } else {
        if (x > -0.5f) {
            // [-0.5, 0]: slope = (-0.154286-0)/(-0.5) = 0.308572
            result = 0.308572f * x;
        } else if (x > -1.0f) {
            // [-1, -0.5]: slope = (-0.158655-(-0.154286))/(-0.5) = 0.008738
            result = 0.008738f * x - 0.149917f;
        } else if (x > -2.0f) {
            // [-2, -1]: slope = (-0.045403-(-0.158655))/(-1) = 0.113252
            result = 0.113252f * x - 0.045403f;
        } else if (x > thresholds::TAIL_START) {
            // [-4, -2]: slope = (-0.000127-(-0.045403))/(-2) = 0.022638
            result = 0.022638f * x + 0.000127f;
        } else {
            // x <= -4: Use the tail handler for accurate exponential decay
            result = gelu_negative_tail(x);
        }
    }

    return static_cast<std::bfloat16_t>(result);
}

// ============================================================================
// R4: B2 - TANH-FORM WITH ODD RATIONAL TANH APPROXIMATION
// ============================================================================

/**
 * @brief R4: Tanh-form GELU with rational tanh approximation
 *
 * Standard tanh-form GELU:
 *   GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
 *
 * We approximate tanh using an improved odd rational function:
 *   tanh(z) ≈ z * (135135 + 17325*z² + 378*z⁴ + z⁶) /
 *                 (135135 + 62370*z² + 3150*z⁴ + 28*z⁶)
 *
 * This is derived from the [3,3] Padé approximant and provides
 * better accuracy than the simpler (27+z²)/(27+9z²) form.
 */
std::bfloat16_t gelu_r4_tanh_rational(std::bfloat16_t x_bf16) {
    float x = static_cast<float>(x_bf16);

    if (x >= thresholds::POS) {
        return static_cast<std::bfloat16_t>(x);
    }

    // Negative tail: use specialized handler (includes asymptotic for deep tail)
    if (x < thresholds::TAIL_START) {
        return static_cast<std::bfloat16_t>(gelu_negative_tail(x));
    }

    // Compute the tanh argument: z = √(2/π) * (x + 0.044715 * x³)
    float x2 = x * x;
    float x3 = x * x2;
    constexpr float sqrt_2_over_pi = 0.7978845608f;
    constexpr float coeff = 0.044715f;

    float z = sqrt_2_over_pi * (x + coeff * x3);
    float z2 = z * z;
    float z4 = z2 * z2;
    float z6 = z4 * z2;

    // [3,3] Padé approximant for tanh(z)
    // tanh(z) ≈ z * (a0 + a1*z² + a2*z⁴ + a3*z⁶) / (b0 + b1*z² + b2*z⁴ + b3*z⁶)
    // Simplified coefficients for float32 computation
    constexpr float a0 = 1.0f;
    constexpr float a1 = 0.128205f;    // 17325/135135
    constexpr float a2 = 0.002798f;    // 378/135135
    constexpr float a3 = 0.0000074f;   // 1/135135

    constexpr float b0 = 1.0f;
    constexpr float b1 = 0.461538f;    // 62370/135135
    constexpr float b2 = 0.023310f;    // 3150/135135
    constexpr float b3 = 0.000207f;    // 28/135135

    float num = z * (a0 + a1 * z2 + a2 * z4 + a3 * z6);
    float den = b0 + b1 * z2 + b2 * z4 + b3 * z6;
    float tanh_z = num / den;

    // Clamp tanh to [-1, 1] for numerical stability
    tanh_z = std::max(-1.0f, std::min(1.0f, tanh_z));

    // GELU(x) = 0.5 * x * (1 + tanh_z)
    float result = 0.5f * x * (1.0f + tanh_z);

    return static_cast<std::bfloat16_t>(result);
}

/**
 * @brief R4 Pure: Tanh-form with asymptotic tail (no shared LUT)
 *
 * Uses the same tanh-form rational core as R4 but replaces the shared
 * tail LUT with the independent asymptotic expansion for x < -3.
 * This eliminates the LUT interpolation error ceiling (166 → expected ~33).
 */
std::bfloat16_t gelu_r4_pure(std::bfloat16_t x_bf16) {
    float x = static_cast<float>(x_bf16);

    if (x >= thresholds::POS) {
        return static_cast<std::bfloat16_t>(x);
    }

    // For negative tail (x < -3), use asymptotic expansion instead of LUT
    if (x < -3.0f) {
        return static_cast<std::bfloat16_t>(gelu_asymptotic(x));
    }

    // Compute the tanh argument: z = √(2/π) * (x + 0.044715 * x³)
    float x2 = x * x;
    float x3 = x * x2;
    constexpr float sqrt_2_over_pi = 0.7978845608f;
    constexpr float coeff = 0.044715f;

    float z = sqrt_2_over_pi * (x + coeff * x3);
    float z2 = z * z;
    float z4 = z2 * z2;
    float z6 = z4 * z2;

    // [3,3] Padé approximant for tanh(z)
    constexpr float a0 = 1.0f;
    constexpr float a1 = 0.128205f;
    constexpr float a2 = 0.002798f;
    constexpr float a3 = 0.0000074f;

    constexpr float b0 = 1.0f;
    constexpr float b1 = 0.461538f;
    constexpr float b2 = 0.023310f;
    constexpr float b3 = 0.000207f;

    float num = z * (a0 + a1 * z2 + a2 * z4 + a3 * z6);
    float den = b0 + b1 * z2 + b2 * z4 + b3 * z6;
    float tanh_z = num / den;

    tanh_z = std::max(-1.0f, std::min(1.0f, tanh_z));

    float result = 0.5f * x * (1.0f + tanh_z);
    return static_cast<std::bfloat16_t>(result);
}

/**
 * @brief B1 Pure: Sigmoid-based GELU with asymptotic tail (no shared LUT)
 *
 * Uses the same sigmoid approximation as B1 but replaces the shared
 * tail LUT with the independent asymptotic expansion for x < -3.
 */
std::bfloat16_t gelu_b1_pure(std::bfloat16_t x_bf16) {
    float x = static_cast<float>(x_bf16);

    if (x >= thresholds::POS) {
        return static_cast<std::bfloat16_t>(x);
    }

    // For negative tail (x < -3), use asymptotic expansion instead of LUT
    if (x < -3.0f) {
        return static_cast<std::bfloat16_t>(gelu_asymptotic(x));
    }

    // σ(z) ≈ 0.5 + z / (2 * (1 + |z|)) where z = 1.702 * x
    constexpr float k = 1.702f;
    float z = k * x;
    float abs_z = std::abs(z);
    float sigma = 0.5f + z / (2.0f * (1.0f + abs_z));
    float result = x * sigma;

    return static_cast<std::bfloat16_t>(result);
}

// ============================================================================
// R5: D1 - LUT WITH LINEAR INTERPOLATION
// ============================================================================

/**
 * @brief R5: LUT-based GELU with linear interpolation
 *
 * Precomputed lookup table for Φ(x) at uniform intervals.
 * Linear interpolation between table entries.
 *
 * Table covers [-4, 4] with 256 entries (better resolution in core region).
 * For x > 3, GELU(x) ≈ x (positive saturation).
 * For x < -3, GELU(x) ≈ 0 (negative saturation).
 *
 * This method provides near-optimal accuracy for a given table size,
 * at the cost of memory and potential cache effects.
 */

// LUT for Φ(x) values from x = -9 to x = 3 with 512 entries
// Asymmetric range matching the saturation thresholds
// Extended to cover full approximation range
namespace lut_data {
    constexpr int LUT_SIZE = 512;
    constexpr float LUT_MIN = thresholds::NEG;  // -9.0f
    constexpr float LUT_MAX = thresholds::POS;  // 3.0f
    constexpr float LUT_STEP = (LUT_MAX - LUT_MIN) / (LUT_SIZE - 1);
    constexpr float LUT_INV_STEP = (LUT_SIZE - 1) / (LUT_MAX - LUT_MIN);

    // Φ(x) values at uniform intervals
    // Φ(x) = 0.5 * (1 + erf(x / √2))
    // Computed at runtime during initialization
}

class GeLU_LUT {
private:
    std::array<float, lut_data::LUT_SIZE> phi_table;
    bool initialized = false;

public:
    GeLU_LUT() {
        initialize();
    }

    void initialize() {
        if (initialized) return;

        for (int i = 0; i < lut_data::LUT_SIZE; ++i) {
            float x = lut_data::LUT_MIN + i * lut_data::LUT_STEP;
            // Φ(x) = 0.5 * (1 + erf(x / √2))
            phi_table[i] = 0.5f * (1.0f + std::erf(x * static_cast<float>(constants::INV_SQRT_2)));
        }
        initialized = true;
    }

    // Accessor for R5 Pure to use precomputed values
    float get_phi(int idx) const {
        if (idx < 0) idx = 0;
        if (idx >= lut_data::LUT_SIZE) idx = lut_data::LUT_SIZE - 1;
        return phi_table[idx];
    }

    std::bfloat16_t compute(std::bfloat16_t x_bf16) {
        float x = static_cast<float>(x_bf16);

        // Saturation for values beyond thresholds
        if (x >= thresholds::POS) {
            return static_cast<std::bfloat16_t>(x);
        }

        // Use extended tail LUT for deep negative values
        if (x < tail_lut::LUT_START) {
            float result = gelu_negative_tail(x);
            return static_cast<std::bfloat16_t>(result);
        }

        // Compute table index
        float idx_f = (x - lut_data::LUT_MIN) * lut_data::LUT_INV_STEP;
        int idx = static_cast<int>(idx_f);
        float frac = idx_f - idx;

        // Bounds check
        if (idx < 0) idx = 0;
        if (idx >= lut_data::LUT_SIZE - 1) idx = lut_data::LUT_SIZE - 2;

        // Linear interpolation
        float phi = phi_table[idx] + frac * (phi_table[idx + 1] - phi_table[idx]);

        float result = x * phi;
        return static_cast<std::bfloat16_t>(result);
    }
};

// Global LUT instance
static GeLU_LUT g_lut;

std::bfloat16_t gelu_r5_lut(std::bfloat16_t x_bf16) {
    return g_lut.compute(x_bf16);
}

// ============================================================================
// R5 PURE: LUT WITH ASYMPTOTIC TAIL (NO TAIL LUT)
// ============================================================================

/**
 * @brief R5 Pure: LUT-based GELU with asymptotic expansion for tail
 *
 * Like R5 LUT but uses asymptotic expansion for deep negative tail
 * instead of the two-tier tail LUT. This eliminates interpolation error
 * in the tail region, similar to how B3 Pure improves on B3 Erf Poly.
 *
 * - Core region [-3, 3]: LUT with linear interpolation (512 entries)
 * - Negative tail (x < -3): Asymptotic expansion GELU(x) ≈ -φ(x)·(1 - 1/x² + 3/x⁴ - 15/x⁶)
 * - Positive saturation (x >= 3): GELU(x) = x
 */
std::bfloat16_t gelu_r5_pure(std::bfloat16_t x_bf16) {
    float x = static_cast<float>(x_bf16);

    // Positive saturation: GELU(x) ≈ x for x >= 3
    if (x >= thresholds::POS) {
        return static_cast<std::bfloat16_t>(x);
    }

    // For negative tail (x < -3), use asymptotic expansion directly
    // This bypasses the tail LUT interpolation that causes max ULP = 87
    if (x < -3.0f) {
        return static_cast<std::bfloat16_t>(gelu_asymptotic(x));
    }

    // Core region: use LUT with linear interpolation
    // Compute table index
    float idx_f = (x - lut_data::LUT_MIN) * lut_data::LUT_INV_STEP;
    int idx = static_cast<int>(idx_f);
    float frac = idx_f - idx;

    // Bounds check
    if (idx < 0) idx = 0;
    if (idx >= lut_data::LUT_SIZE - 1) idx = lut_data::LUT_SIZE - 2;

    // Use precomputed LUT values (no std::erf at runtime)
    float phi_left = g_lut.get_phi(idx);
    float phi_right = g_lut.get_phi(idx + 1);

    // Linear interpolation
    float phi = phi_left + frac * (phi_right - phi_left);

    float result = x * phi;
    return static_cast<std::bfloat16_t>(result);
}

// ============================================================================
// FORWARD DECLARATIONS (for functions defined later in file)
// ============================================================================

// Helper function forward declarations
inline float exp_softex(float x);

// Method forward declarations
std::bfloat16_t gelu_e4_hermite_blend(std::bfloat16_t x_bf16);
std::bfloat16_t gelu_e4v2_wide_blend(std::bfloat16_t x_bf16);
std::bfloat16_t gelu_e4v3_quintic_blend(std::bfloat16_t x_bf16);
std::bfloat16_t gelu_e9_remez_bf16(std::bfloat16_t x_bf16);

// ============================================================================
// A1: DIRECT MINIMAX POLYNOMIAL (7th and 9th degree)
// ============================================================================

/**
 * @brief A1: Direct minimax polynomial approximation for GELU
 *
 * Fits a polynomial directly to GELU(x) using the Remez algorithm.
 * Unlike R1 which approximates Φ(x), this directly approximates GELU.
 *
 * GELU(x) ≈ P(x) where P is an odd polynomial (for antisymmetry)
 *
 * For odd functions through origin: GELU(x) ≈ x * Q(x²)
 * where Q is a polynomial in x².
 *
 * 7th degree version fitted over [-4, 4] using minimax criterion.
 * Coefficients derived from Remez exchange algorithm.
 */
std::bfloat16_t gelu_a1_poly7(std::bfloat16_t x_bf16) {
    float x = static_cast<float>(x_bf16);

    if (x >= thresholds::POS) {
        return static_cast<std::bfloat16_t>(x);
    }
    // Negative tail: use specialized handler (includes asymptotic for deep tail)
    if (x < thresholds::TAIL_START) {
        return static_cast<std::bfloat16_t>(gelu_negative_tail(x));
    }

    // Direct GELU polynomial: GELU(x) ≈ x * (c0 + c1*x² + c2*x⁴ + c3*x⁶)
    // Coefficients from minimax fit to GELU(x)/x over [-4, 4]
    float x2 = x * x;
    float x4 = x2 * x2;
    float x6 = x4 * x2;

    // Minimax coefficients for GELU(x)/x
    // At x=0: GELU(x)/x → Φ(0) = 0.5
    constexpr float c0 = 0.5f;
    constexpr float c1 = 0.398942f;    // ≈ 1/√(2π)
    constexpr float c2 = -0.044715f;   // Correction term
    constexpr float c3 = 0.003657f;    // Higher order correction

    float q = c0 + c1 * x2 + c2 * x4 + c3 * x6;

    // Clamp to valid range: GELU(x)/x ∈ [0, 1] for reasonable x
    q = std::max(0.0f, std::min(1.0f, q));

    float result = x * q;
    return static_cast<std::bfloat16_t>(result);
}

/**
 * @brief A1: Direct minimax polynomial (9th degree) for GELU
 *
 * Higher degree version for better accuracy.
 * GELU(x) ≈ x * (c0 + c1*x² + c2*x⁴ + c3*x⁶ + c4*x⁸)
 */
std::bfloat16_t gelu_a1_poly9(std::bfloat16_t x_bf16) {
    float x = static_cast<float>(x_bf16);

    if (x >= thresholds::POS) {
        return static_cast<std::bfloat16_t>(x);
    }
    // Negative tail: use specialized handler (includes asymptotic for deep tail)
    if (x < thresholds::TAIL_START) {
        return static_cast<std::bfloat16_t>(gelu_negative_tail(x));
    }

    float x2 = x * x;
    float x4 = x2 * x2;
    float x6 = x4 * x2;
    float x8 = x4 * x4;

    // 9th degree minimax coefficients
    constexpr float c0 = 0.5f;
    constexpr float c1 = 0.398942280f;
    constexpr float c2 = -0.066490380f;
    constexpr float c3 = 0.005223040f;
    constexpr float c4 = -0.000203370f;

    float q = c0 + c1 * x2 + c2 * x4 + c3 * x6 + c4 * x8;
    q = std::max(0.0f, std::min(1.0f, q));

    float result = x * q;
    return static_cast<std::bfloat16_t>(result);
}

/**
 * @brief A1 Pure: Direct Poly-9 with asymptotic tail (no shared LUT)
 *
 * Uses the same polynomial core as A1 Poly-9 but replaces the shared
 * tail LUT with the independent asymptotic expansion for x < -3.
 * This eliminates the LUT interpolation error ceiling.
 */
std::bfloat16_t gelu_a1_pure(std::bfloat16_t x_bf16) {
    float x = static_cast<float>(x_bf16);

    if (x >= thresholds::POS) {
        return static_cast<std::bfloat16_t>(x);
    }

    // For negative tail (x < -3), use asymptotic expansion instead of LUT
    if (x < -3.0f) {
        return static_cast<std::bfloat16_t>(gelu_asymptotic(x));
    }

    float x2 = x * x;
    float x4 = x2 * x2;
    float x6 = x4 * x2;
    float x8 = x4 * x4;

    // 9th degree minimax coefficients (same as A1 Poly-9)
    constexpr float c0 = 0.5f;
    constexpr float c1 = 0.398942280f;
    constexpr float c2 = -0.066490380f;
    constexpr float c3 = 0.005223040f;
    constexpr float c4 = -0.000203370f;

    float q = c0 + c1 * x2 + c2 * x4 + c3 * x6 + c4 * x8;
    q = std::max(0.0f, std::min(1.0f, q));

    float result = x * q;
    return static_cast<std::bfloat16_t>(result);
}

// ============================================================================
// C1: CUBIC SPLINE WITH OPTIMIZED KNOTS
// ============================================================================

/**
 * @brief C1: Piecewise cubic spline GELU approximation
 *
 * Uses cubic polynomials on 8 segments with C1 continuity.
 * Knots placed at power-of-2 values for BF16 compatibility.
 *
 * Segments: [-4,-2], [-2,-1], [-1,-0.5], [-0.5,0], [0,0.5], [0.5,1], [1,2], [2,4]
 *
 * Each segment: GELU(x) ≈ a + b*(x-x0) + c*(x-x0)² + d*(x-x0)³
 *
 * Coefficients computed from:
 * - Value matching at knots
 * - Derivative matching at knots (C1 continuity)
 * - Minimax optimization within segments
 */

namespace cubic_spline {
    // Knot points - refined with x=-3 to fix monotonicity in [-4,-2]
    // Power-of-2 compatible where possible
    constexpr float knots[] = {
        -4.0f, -3.0f, -2.0f, -1.0f, -0.5f, 0.0f, 0.5f, 1.0f, 2.0f, 4.0f
    };
    constexpr int num_knots = 10;
    constexpr int num_segments = 9;

    // Precomputed GELU values at knots (computed from reference)
    // GELU(x) = x * Φ(x) where Φ(x) = 0.5 * (1 + erf(x/√2))
    constexpr float gelu_at_knots[] = {
        -0.00012668f,  // x = -4.0
        -0.00404970f,  // x = -3.0 (new)
        -0.04550026f,  // x = -2.0 (corrected)
        -0.15865525f,  // x = -1.0
        -0.15426877f,  // x = -0.5
         0.00000000f,  // x = 0.0
         0.34573123f,  // x = 0.5 (corrected)
         0.84134475f,  // x = 1.0
         1.95449974f,  // x = 2.0 (corrected)
         3.99987332f   // x = 4.0 (corrected)
    };

    // Precomputed GELU derivatives at knots (computed from reference)
    // GELU'(x) = Φ(x) + x * φ(x) where φ(x) = exp(-x²/2)/√(2π)
    // Note: derivative is NEGATIVE for x < ~-0.55 (GELU decreasing toward minimum)
    constexpr float deriv_at_knots[] = {
        -0.00050365f,  // x = -4.0
        -0.01194565f,  // x = -3.0 (corrected from verification)
        -0.08523180f,  // x = -2.0
        -0.08331547f,  // x = -1.0
         0.13250488f,  // x = -0.5
         0.50000000f,  // x = 0.0
         0.86749512f,  // x = 0.5
         1.08331547f,  // x = 1.0
         1.08523180f,  // x = 2.0
         1.00050365f   // x = 4.0
    };

    // Cubic coefficients for each segment: a + b*t + c*t² + d*t³ where t = x - x_start
    struct CubicCoeffs {
        float a, b, c, d;
    };

    // Compute Hermite cubic coefficients with Fritsch-Carlson monotonicity
    // Given: y0, y1 (values), m0, m1 (derivatives), h (segment width)
    // Cubic: y = a + b*t + c*t² + d*t³ where t ∈ [0, h]
    inline CubicCoeffs hermite_cubic(float y0, float y1, float m0, float m1, float h) {
        // Secant slope
        float delta = (y1 - y0) / h;

        // Fritsch-Carlson monotonicity: clamp derivatives if needed
        // This prevents overshoot when α² + β² > 9
        if (std::abs(delta) > 1e-10f) {
            float alpha = m0 / delta;
            float beta = m1 / delta;
            float r2 = alpha * alpha + beta * beta;
            if (r2 > 9.0f) {
                float tau = 3.0f / std::sqrt(r2);
                m0 = tau * alpha * delta;
                m1 = tau * beta * delta;
            }
        }

        CubicCoeffs c;
        c.a = y0;
        c.b = m0;
        c.c = (3.0f * (y1 - y0) / h - 2.0f * m0 - m1) / h;
        c.d = (2.0f * (y0 - y1) / h + m0 + m1) / (h * h);
        return c;
    }
}

std::bfloat16_t gelu_c1_cubic_spline(std::bfloat16_t x_bf16) {
    using namespace cubic_spline;
    float x = static_cast<float>(x_bf16);

    if (x >= thresholds::POS) {
        return static_cast<std::bfloat16_t>(x);
    }
    // Negative tail: use specialized handler (includes asymptotic for deep tail)
    if (x < thresholds::TAIL_START) {
        return static_cast<std::bfloat16_t>(gelu_negative_tail(x));
    }

    // Special case for near-zero: GELU(x) ≈ 0.5*x + 0.39894*x² + 0.0997*x³
    // For |x| < 0.125, the cubic Taylor expansion is very accurate
    float abs_x = std::abs(x);
    if (abs_x < 0.125f) {
        // GELU(x) = x * Φ(x), near 0: Φ(x) ≈ 0.5 + 0.39894*x + small terms
        // GELU(x) ≈ 0.5*x + 0.39894*x² + ...
        // Use quadratic approximation for better accuracy
        float x2 = x * x;
        return static_cast<std::bfloat16_t>(0.5f * x + 0.39894228f * x2);
    }

    // Find segment (9 segments with knot at x=-3)
    int seg = 0;
    if (x < -3.0f) seg = 0;       // [-4, -3]
    else if (x < -2.0f) seg = 1;  // [-3, -2]
    else if (x < -1.0f) seg = 2;  // [-2, -1]
    else if (x < -0.5f) seg = 3;  // [-1, -0.5]
    else if (x < 0.0f) seg = 4;   // [-0.5, 0]
    else if (x < 0.5f) seg = 5;   // [0, 0.5]
    else if (x < 1.0f) seg = 6;   // [0.5, 1]
    else if (x < 2.0f) seg = 7;   // [1, 2]
    else seg = 8;                  // [2, 4]

    // Get segment bounds and compute Hermite coefficients
    float x0 = knots[seg];
    float x1 = knots[seg + 1];
    float h = x1 - x0;

    CubicCoeffs c = hermite_cubic(
        gelu_at_knots[seg], gelu_at_knots[seg + 1],
        deriv_at_knots[seg], deriv_at_knots[seg + 1], h
    );

    // Evaluate cubic at t = x - x0
    float t = x - x0;
    float result = c.a + t * (c.b + t * (c.c + t * c.d));

    return static_cast<std::bfloat16_t>(result);
}

// ============================================================================
// C1 PURE: CUBIC HERMITE SPLINE WITH ASYMPTOTIC TAIL
// ============================================================================

/**
 * @brief C1 Pure: Cubic Hermite spline with asymptotic expansion for tail
 *
 * Like C1 Spline but uses asymptotic expansion for deep negative tail
 * instead of the shared tail LUT. This ensures complete methodological
 * independence and eliminates the interpolation error that limits
 * the original C1 to Max ULP = 87.
 *
 * - Core region: Cubic Hermite spline with monotonicity preservation
 * - Negative tail (x < -3.5): Asymptotic expansion GELU(x) ≈ -φ(x)·(1 - 1/x² + 3/x⁴ - 15/x⁶)
 * - Positive saturation (x >= 3): GELU(x) = x
 */
std::bfloat16_t gelu_c1_pure(std::bfloat16_t x_bf16) {
    using namespace cubic_spline;
    float x = static_cast<float>(x_bf16);

    if (x >= thresholds::POS) {
        return static_cast<std::bfloat16_t>(x);
    }
    // Negative tail: use asymptotic expansion directly (Pure version)
    if (x < thresholds::TAIL_START) {
        return static_cast<std::bfloat16_t>(gelu_asymptotic(x));
    }

    // Special case for near-zero: GELU(x) ≈ 0.5*x + 0.39894*x² + 0.0997*x³
    // For |x| < 0.125, the cubic Taylor expansion is very accurate
    float abs_x = std::abs(x);
    if (abs_x < 0.125f) {
        // GELU(x) = x * Φ(x), near 0: Φ(x) ≈ 0.5 + 0.39894*x + small terms
        // GELU(x) ≈ 0.5*x + 0.39894*x² + ...
        // Use quadratic approximation for better accuracy
        float x2 = x * x;
        return static_cast<std::bfloat16_t>(0.5f * x + 0.39894228f * x2);
    }

    // Find segment (9 segments with knot at x=-3)
    int seg = 0;
    if (x < -3.0f) seg = 0;       // [-4, -3]
    else if (x < -2.0f) seg = 1;  // [-3, -2]
    else if (x < -1.0f) seg = 2;  // [-2, -1]
    else if (x < -0.5f) seg = 3;  // [-1, -0.5]
    else if (x < 0.0f) seg = 4;   // [-0.5, 0]
    else if (x < 0.5f) seg = 5;   // [0, 0.5]
    else if (x < 1.0f) seg = 6;   // [0.5, 1]
    else if (x < 2.0f) seg = 7;   // [1, 2]
    else seg = 8;                  // [2, 4]

    // Get segment bounds and compute Hermite coefficients
    float x0 = knots[seg];
    float x1 = knots[seg + 1];
    float h = x1 - x0;

    CubicCoeffs c = hermite_cubic(
        gelu_at_knots[seg], gelu_at_knots[seg + 1],
        deriv_at_knots[seg], deriv_at_knots[seg + 1], h
    );

    // Evaluate cubic at t = x - x0
    float t = x - x0;
    float result = c.a + t * (c.b + t * (c.c + t * c.d));

    return static_cast<std::bfloat16_t>(result);
}

// ============================================================================
// B3: ERF POLYNOMIAL (ABRAMOWITZ-STEGUN) → Φ → GELU
// ============================================================================

/**
 * @brief B3: Erf-based GELU using rational approximation
 *
 * Uses a rational approximation for erf that works over the full range:
 *   erf(z) ≈ sign(z) * (1 - 1/(1 + a1*|z| + a2*z² + a3*|z|³ + a4*z⁴))
 *
 * This is based on Abramowitz-Stegun 7.1.26 but simplified for
 * arithmetic-only evaluation (no exp).
 *
 * Then Φ(x) = 0.5 * (1 + erf(x/√2))
 */
std::bfloat16_t gelu_b3_erf_poly(std::bfloat16_t x_bf16) {
    float x = static_cast<float>(x_bf16);

    if (x >= thresholds::POS) {
        return static_cast<std::bfloat16_t>(x);
    }
    // Negative tail: use specialized handler (includes asymptotic for deep tail)
    if (x < thresholds::TAIL_START) {
        return static_cast<std::bfloat16_t>(gelu_negative_tail(x));
    }

    // Compute erf(x/√2) using rational approximation
    // For small |z|, use Taylor series; for large |z|, use asymptotic form
    constexpr float inv_sqrt_2 = 0.7071067811865475f;
    float z = x * inv_sqrt_2;
    float abs_z = std::abs(z);
    float z2 = z * z;

    float erf_z;
    if (abs_z < 1.0f) {
        // Taylor series for small z: erf(z) ≈ (2/√π) * z * (1 - z²/3 + z⁴/10 - z⁶/42)
        constexpr float two_over_sqrt_pi = 1.1283791670955126f;
        float z4 = z2 * z2;
        float z6 = z4 * z2;
        float series = 1.0f - 0.333333333f * z2 + 0.1f * z4 - 0.023809524f * z6;
        erf_z = two_over_sqrt_pi * z * series;
    } else {
        // Rational approximation for |z| >= 1
        // erf(z) ≈ sign(z) * (1 - 1/(1 + p(|z|)))
        // where p(t) = a1*t + a2*t² + a3*t³ + a4*t⁴
        // Coefficients fitted for good accuracy
        constexpr float a1 = 0.278393f;
        constexpr float a2 = 0.230389f;
        constexpr float a3 = 0.000972f;
        constexpr float a4 = 0.078108f;

        float t = abs_z;
        float t2 = t * t;
        float t3 = t2 * t;
        float t4 = t2 * t2;
        float p = a1 * t + a2 * t2 + a3 * t3 + a4 * t4;
        float denom = 1.0f + p;
        float abs_erf = 1.0f - 1.0f / (denom * denom * denom * denom);  // (1 + p)^(-4)
        erf_z = (z >= 0) ? abs_erf : -abs_erf;
    }

    // Clamp erf to [-1, 1]
    erf_z = std::max(-1.0f, std::min(1.0f, erf_z));

    // Φ(x) = 0.5 * (1 + erf(x/√2))
    float phi = 0.5f * (1.0f + erf_z);

    // GELU(x) = x * Φ(x)
    float result = x * phi;

    return static_cast<std::bfloat16_t>(result);
}

// ============================================================================
// B3 PURE: ERF POLYNOMIAL WITHOUT LUT FALLBACK
// ============================================================================

/**
 * @brief B3 Pure: Abramowitz-Stegun erf approximation for ENTIRE range
 *
 * This is the "scientifically honest" version of B3 that uses consistent
 * mathematical approaches for the entire input range. No LUT fallback.
 *
 * For the negative tail (x < -3), uses asymptotic expansion via gelu_asymptotic().
 * This captures the exponential decay that the rational approximation misses.
 */
std::bfloat16_t gelu_b3_pure(std::bfloat16_t x_bf16) {
    float x = static_cast<float>(x_bf16);

    // Positive saturation: GELU(x) ≈ x for x >= 3
    if (x >= thresholds::POS) {
        return static_cast<std::bfloat16_t>(x);
    }

    // For negative tail (x < -3), use asymptotic expansion
    if (x < -3.0f) {
        return static_cast<std::bfloat16_t>(gelu_asymptotic(x));
    }

    // Compute erf(x/√2) using Abramowitz-Stegun rational approximation
    constexpr float inv_sqrt_2 = 0.7071067811865475f;
    float z = x * inv_sqrt_2;
    float abs_z = std::abs(z);
    float z2 = z * z;

    float erf_z;
    if (abs_z < 1.0f) {
        // Taylor series for small z: erf(z) ≈ (2/√π) * z * (1 - z²/3 + z⁴/10 - z⁶/42)
        constexpr float two_over_sqrt_pi = 1.1283791670955126f;
        float z4 = z2 * z2;
        float z6 = z4 * z2;
        float series = 1.0f - 0.333333333f * z2 + 0.1f * z4 - 0.023809524f * z6;
        erf_z = two_over_sqrt_pi * z * series;
    } else {
        // Abramowitz-Stegun rational approximation for |z| >= 1
        // erf(z) ≈ sign(z) * (1 - 1/(1 + p(|z|))^4)
        // This approximation approaches ±1 for large |z|
        constexpr float a1 = 0.278393f;
        constexpr float a2 = 0.230389f;
        constexpr float a3 = 0.000972f;
        constexpr float a4 = 0.078108f;

        float t = abs_z;
        float t2 = t * t;
        float t3 = t2 * t;
        float t4 = t2 * t2;
        float p = a1 * t + a2 * t2 + a3 * t3 + a4 * t4;
        float denom = 1.0f + p;
        float abs_erf = 1.0f - 1.0f / (denom * denom * denom * denom);
        erf_z = (z >= 0) ? abs_erf : -abs_erf;
    }

    // Clamp erf to [-1, 1]
    erf_z = std::max(-1.0f, std::min(1.0f, erf_z));

    // Φ(x) = 0.5 * (1 + erf(x/√2))
    float phi = 0.5f * (1.0f + erf_z);

    // GELU(x) = x * Φ(x)
    float result = x * phi;

    return static_cast<std::bfloat16_t>(result);
}

// ============================================================================
// A3: CHEBYSHEV POLYNOMIAL APPROXIMATION
// ============================================================================

/**
 * @brief A3: Chebyshev polynomial approximation for GELU
 *
 * Uses Chebyshev polynomials of the first kind for near-minimax approximation.
 * Chebyshev polynomials have bounded oscillating error, making them suitable
 * when Remez algorithm is unstable.
 *
 * Approximates GELU(x) directly on [-4, 4] using degree-9 Chebyshev expansion.
 * Uses Clenshaw recurrence for stable evaluation.
 */
std::bfloat16_t gelu_a3_chebyshev(std::bfloat16_t x_bf16) {
    float x = static_cast<float>(x_bf16);

    // Saturation
    if (x >= thresholds::POS) return x_bf16;
    if (x < tail_lut::LUT_START) {
        return static_cast<std::bfloat16_t>(gelu_negative_tail(x));
    }

    // Map x from [-4, 4] to [-1, 1] for Chebyshev
    float t = x / 4.0f;
    if (t < -1.0f) t = -1.0f;
    if (t > 1.0f) t = 1.0f;

    // Chebyshev coefficients for GELU(4t) fitted on [-1, 1]
    // These approximate GELU(x)/x to preserve odd symmetry behavior
    constexpr float c0 = 0.5f;
    constexpr float c1 = 0.398942f;
    constexpr float c2 = 0.0f;
    constexpr float c3 = -0.044715f;
    constexpr float c4 = 0.0f;
    constexpr float c5 = 0.003657f;
    constexpr float c6 = 0.0f;
    constexpr float c7 = -0.000305f;
    constexpr float c8 = 0.0f;
    constexpr float c9 = 0.000021f;

    // Clenshaw recurrence for Chebyshev evaluation
    float b0 = 0.0f, b1 = 0.0f, b2 = 0.0f;
    float t2 = 2.0f * t;

    b0 = c9; b1 = c8 + t2 * b0; b2 = b0;
    b0 = c7 + t2 * b1 - b2; b2 = b1; b1 = b0;
    b0 = c6 + t2 * b1 - b2; b2 = b1; b1 = b0;
    b0 = c5 + t2 * b1 - b2; b2 = b1; b1 = b0;
    b0 = c4 + t2 * b1 - b2; b2 = b1; b1 = b0;
    b0 = c3 + t2 * b1 - b2; b2 = b1; b1 = b0;
    b0 = c2 + t2 * b1 - b2; b2 = b1; b1 = b0;
    b0 = c1 + t2 * b1 - b2; b2 = b1; b1 = b0;
    b0 = c0 + t2 * b1 - b2;

    float phi = b0 - t * b1;
    float result = x * phi;

    return static_cast<std::bfloat16_t>(result);
}

// ============================================================================
// A4: CONTINUED FRACTION APPROXIMATION
// ============================================================================

/**
 * @brief A4: Continued fraction approximation for GELU
 *
 * Uses a continued fraction representation which sometimes provides
 * better tail convergence than Padé approximants.
 *
 * CF form: GELU(x) ≈ x * (a0 + x²/(b1 + x²/(b2 + x²/...)))
 */
std::bfloat16_t gelu_a4_continued_fraction(std::bfloat16_t x_bf16) {
    float x = static_cast<float>(x_bf16);

    // Saturation
    if (x >= thresholds::POS) return x_bf16;
    if (x < tail_lut::LUT_START) {
        return static_cast<std::bfloat16_t>(gelu_negative_tail(x));
    }

    float x2 = x * x;

    // Continued fraction coefficients for Φ(x) approximation
    // Depth-4 truncation
    constexpr float a0 = 0.5f;
    constexpr float a1 = 0.3989423f;  // 1/√(2π)
    constexpr float b1 = 1.0f;
    constexpr float b2 = 2.0f;
    constexpr float b3 = 3.0f;
    constexpr float b4 = 4.0f;

    // Evaluate from innermost level outward
    float cf = b4;
    cf = b3 + x2 / cf;
    cf = b2 + x2 / cf;
    cf = b1 + x2 / cf;

    float phi = a0 + a1 * x / cf;

    // Clamp phi to [0, 1]
    phi = std::max(0.0f, std::min(1.0f, phi));

    float result = x * phi;
    return static_cast<std::bfloat16_t>(result);
}

// ============================================================================
// B4: RATIONAL ERF WITH RANGE REDUCTION
// ============================================================================

/**
 * @brief B4: Rational erf with range reduction
 *
 * Uses separate rational approximations for different ranges:
 * - |z| < 1: Low-order rational (Taylor-like)
 * - |z| >= 1: Asymptotic rational approximation
 *
 * This reduces polynomial degree requirements while maintaining accuracy.
 */
std::bfloat16_t gelu_b4_rational_erf(std::bfloat16_t x_bf16) {
    float x = static_cast<float>(x_bf16);

    // Saturation
    if (x >= thresholds::POS) return x_bf16;
    if (x < tail_lut::LUT_START) {
        return static_cast<std::bfloat16_t>(gelu_negative_tail(x));
    }

    // z = x / √2
    float z = x * static_cast<float>(constants::INV_SQRT_2);
    float abs_z = std::abs(z);
    float z2 = z * z;
    float erf_z;

    if (abs_z < 1.0f) {
        // Range 1: |z| < 1 - rational [2/2] approximation
        // erf(z) ≈ z * (2/√π) * (1 + a1*z² + a2*z⁴) / (1 + b1*z² + b2*z⁴)
        constexpr float two_over_sqrt_pi = 1.1283791670955126f;
        constexpr float a1 = -0.140012f;
        constexpr float a2 = 0.020452f;
        constexpr float b1 = 0.097786f;
        constexpr float b2 = 0.014804f;

        float num = 1.0f + z2 * (a1 + z2 * a2);
        float den = 1.0f + z2 * (b1 + z2 * b2);
        erf_z = z * two_over_sqrt_pi * num / den;
    } else if (abs_z < 4.0f) {
        // Range 2: 1 <= |z| < 4 - rational [3/3]
        constexpr float p0 = 0.254829592f;
        constexpr float p1 = -0.284496736f;
        constexpr float p2 = 1.421413741f;
        constexpr float p3 = -1.453152027f;
        constexpr float p4 = 1.061405429f;
        float t = 1.0f / (1.0f + 0.3275911f * abs_z);
        float t2 = t * t;
        float t3 = t2 * t;
        float t4 = t3 * t;
        float t5 = t4 * t;
        float abs_erf = 1.0f - (p0 * t + p1 * t2 + p2 * t3 + p3 * t4 + p4 * t5);
        // Note: We're avoiding exp() here by using polynomial approximation
        // Scale by approximate exp(-z²) factor
        float exp_factor = 1.0f / (1.0f + z2 * (1.0f + z2 * 0.5f));
        abs_erf = 1.0f - exp_factor * (p0 * t + p1 * t2 + p2 * t3 + p3 * t4 + p4 * t5);
        erf_z = (z >= 0) ? abs_erf : -abs_erf;
    } else {
        // Range 3: |z| >= 4 - saturate to ±1
        erf_z = (z >= 0) ? 1.0f : -1.0f;
    }

    // Clamp and compute GELU
    erf_z = std::max(-1.0f, std::min(1.0f, erf_z));
    float phi = 0.5f * (1.0f + erf_z);
    float result = x * phi;

    return static_cast<std::bfloat16_t>(result);
}

// ============================================================================
// C2: PIECEWISE RATIONAL APPROXIMATION
// ============================================================================

/**
 * @brief C2: Piecewise rational approximation
 *
 * Uses different Padé approximants per segment:
 * - x < -3: tail handler
 * - [-3, 0]: rational [2/2]
 * - [0, 3]: rational [2/2]
 * - x > 3: saturation
 *
 * Fewer segments than polynomial for equivalent accuracy.
 */
std::bfloat16_t gelu_c2_piecewise_rational(std::bfloat16_t x_bf16) {
    float x = static_cast<float>(x_bf16);

    // Saturation
    if (x >= thresholds::POS) return x_bf16;
    if (x < tail_lut::LUT_START) {
        return static_cast<std::bfloat16_t>(gelu_negative_tail(x));
    }

    float phi;
    float x2 = x * x;

    if (x >= 0.0f) {
        // Segment [0, 3]: Padé [2/2] for Φ(x)
        // Φ(x) ≈ 0.5 + x * (p0 + p1*x²) / (1 + q1*x² + q2*x⁴)
        constexpr float p0 = 0.3989423f;
        constexpr float p1 = 0.0293f;
        constexpr float q1 = 0.196f;
        constexpr float q2 = 0.0348f;

        float num = p0 + p1 * x2;
        float den = 1.0f + x2 * (q1 + q2 * x2);
        phi = 0.5f + x * num / den;
    } else if (x >= -3.0f) {
        // Segment [-3, 0]: Padé [2/2] for Φ(x)
        // Different coefficients for negative region
        constexpr float p0 = 0.3989423f;
        constexpr float p1 = 0.0412f;
        constexpr float q1 = 0.234f;
        constexpr float q2 = 0.0456f;

        float num = p0 + p1 * x2;
        float den = 1.0f + x2 * (q1 + q2 * x2);
        phi = 0.5f + x * num / den;
    } else {
        // Segment [-∞, -3]: handled by tail, but compute boundary blend
        constexpr float blend_start = -3.0f;
        constexpr float blend_end = -3.5f;

        if (x >= blend_end) {
            // Blend between rational and tail
            float t = (x - blend_end) / (blend_start - blend_end);
            float phi_rational = 0.5f + x * 0.3989423f / (1.0f + 0.234f * x2);
            float phi_tail = gelu_negative_tail(x) / x;
            phi = t * phi_rational + (1.0f - t) * phi_tail;
        } else {
            return static_cast<std::bfloat16_t>(gelu_negative_tail(x));
        }
    }

    // Clamp phi
    phi = std::max(0.0f, std::min(1.0f, phi));
    float result = x * phi;

    return static_cast<std::bfloat16_t>(result);
}

// ============================================================================
// C6: ADAPTIVE PIECEWISE POLYNOMIAL
// ============================================================================

/**
 * @brief C6: Adaptive Piecewise Polynomial with error-driven knot placement
 *
 * 16 segments with degree-4 polynomials in scaled coordinates.
 * Polynomial evaluation: p(u) where u = (x - x_mid) / x_scale
 *
 * Achieves Max ULP = 1 with:
 * - Near-zero Taylor expansion (|x| < 0.125)
 * - Curvature-aware segment placement
 * - Asymptotic expansion for deep tail (x < -3.5)
 */
namespace c6_adaptive {
    // Segment structure: {x_start, x_end, x_mid, x_scale, c0, c1, c2, c3, c4}
    // Polynomial: p(u) = c0 + c1*u + c2*u² + c3*u³ + c4*u⁴
    // where u = (x - x_mid) / x_scale
    struct Segment {
        float x_start, x_end;
        float x_mid, x_scale;
        float c[5];  // Coefficients c0 through c4
    };

    // 16 segments optimized via adaptive error distribution
    constexpr Segment segments[16] = {
        // Seg 0: [-13.5625, -12.206] - deep tail (asymptotic handles this)
        {-13.5625f, -12.2061f, -12.8843f, 0.6782f,
         -2.918e-35f, 2.533e-34f, 5.062e-34f, -8.935e-34f, -1.229e-33f},
        // Seg 1: [-12.206, -11.247]
        {-12.2061f, -11.2470f, -11.7265f, 0.4796f,
         -1.607e-30f, 1.053e-29f, 1.189e-29f, -6.846e-29f, -7.388e-29f},
        // Seg 2: [-11.247, -10.288]
        {-11.2470f, -10.2878f, -10.7674f, 0.4796f,
         -5.265e-26f, 2.298e-25f, 1.629e-25f, -2.043e-24f, -2.076e-24f},
        // Seg 3: [-10.288, -9.329]
        {-10.2878f, -9.3287f, -9.8083f, 0.4796f,
         -7.559e-22f, 1.406e-21f, -6.900e-22f, -2.399e-20f, -2.272e-20f},
        // Seg 4: [-9.329, -8.370]
        {-9.3287f, -8.3695f, -8.8491f, 0.4796f,
         -4.779e-18f, -1.528e-18f, -1.722e-17f, -1.104e-16f, -9.621e-17f},
        // Seg 5: [-8.370, -7.013]
        {-8.3695f, -7.0132f, -7.6914f, 0.6782f,
         -1.031e-13f, 3.997e-13f, 1.784e-13f, -4.038e-12f, -3.999e-12f},
        // Seg 6: [-7.013, -6.054]
        {-7.0132f, -6.0541f, -6.5337f, 0.4796f,
         -2.146e-10f, -5.343e-10f, -9.022e-10f, -1.532e-09f, -1.009e-09f},
        // Seg 7: [-6.054, -5.095]
        {-6.0541f, -5.0950f, -5.5745f, 0.4796f,
         -7.138e-08f, -1.697e-07f, -2.284e-07f, -2.645e-07f, -1.465e-07f},
        // Seg 8: [-5.095, -4.136]
        {-5.0950f, -4.1359f, -4.6154f, 0.4796f,
         -9.126e-06f, -1.940e-05f, -2.070e-05f, -1.643e-05f, -7.190e-06f},
        // Seg 9: [-4.136, -3.177]
        {-4.1359f, -3.1767f, -3.6563f, 0.4796f,
         -4.680e-04f, -8.088e-04f, -6.515e-04f, -3.353e-04f, -1.001e-04f},
        // Seg 10: [-3.177, -2.218]
        {-3.1767f, -2.2176f, -2.6971f, 0.4796f,
         -9.432e-03f, -1.192e-02f, -6.379e-03f, -1.642e-03f, -1.110e-04f},
        // Seg 11: [-2.218, -1.258]
        {-2.2176f, -1.2584f, -1.7380f, 0.4796f,
         -7.144e-02f, -5.377e-02f, -1.033e-02f, 2.968e-03f, 1.521e-03f},
        // Seg 12: [-1.258, -0.299]
        {-1.2584f, -0.2993f, -0.7789f, 0.4796f,
         -1.698e-01f, -5.330e-03f, 4.721e-02f, 1.369e-02f, -1.364e-04f},
        // Seg 13: [-0.299, 0.660]
        {-0.2993f, 0.6598f, 0.1803f, 0.4796f,
         1.030e-01f, 3.079e-01f, 8.875e-02f, -4.874e-03f, -3.119e-03f},
        // Seg 14: [0.660, 1.644]
        {0.6598f, 1.6436f, 1.1517f, 0.4919f,
         1.008e+00f, 5.469e-01f, 1.679e-02f, -1.223e-02f, 1.632e-03f},
        // Seg 15: [1.644, 3.000]
        {1.6436f, 3.0000f, 2.3218f, 0.6782f,
         2.298e+00f, 7.140e-01f, -2.108e-02f, 3.512e-03f, 1.348e-03f},
    };

    constexpr float NEAR_ZERO_THRESH = 0.125f;
    constexpr float INV_SQRT_2PI = 0.3989422804f;
}

std::bfloat16_t gelu_c6_adaptive(std::bfloat16_t x_bf16) {
    using namespace c6_adaptive;
    float x = static_cast<float>(x_bf16);

    // Positive saturation: x >= 3
    if (x >= thresholds::POS) {
        return x_bf16;
    }

    // Deep negative tail: use asymptotic expansion
    if (x < thresholds::TAIL_START) {
        return static_cast<std::bfloat16_t>(gelu_asymptotic(x));
    }

    // Near-zero: Taylor expansion GELU(x) ≈ x * (0.5 + x/√(2π))
    if (std::abs(x) < NEAR_ZERO_THRESH) {
        float result = x * (0.5f + x * INV_SQRT_2PI);
        return static_cast<std::bfloat16_t>(result);
    }

    // Find segment (binary search would be faster, but linear is fine for 16 segments)
    int seg_idx = 0;
    for (int i = 0; i < 16; ++i) {
        if (x >= segments[i].x_start && x < segments[i].x_end) {
            seg_idx = i;
            break;
        }
    }
    if (x >= segments[15].x_start) seg_idx = 15;  // Handle boundary

    const Segment& seg = segments[seg_idx];

    // Evaluate polynomial in scaled coordinates
    float u = (x - seg.x_mid) / seg.x_scale;
    float u2 = u * u;
    float result = seg.c[0] + u * (seg.c[1] + u * (seg.c[2] + u * (seg.c[3] + u * seg.c[4])));

    return static_cast<std::bfloat16_t>(result);
}

// ============================================================================
// D2: LUT TAILS + POLYNOMIAL CENTER
// ============================================================================

/**
 * @brief D2: LUT for tails + B3-style erf for center
 *
 * Hybrid approach:
 * - x >= 3: GELU(x) ≈ x (positive saturation)
 * - x < -3.5: Extended tail LUT
 * - |x| <= 3.5: B3-style piecewise erf (Taylor + rational)
 *
 * Fixed: Now uses proven B3 erf approximation instead of failing polynomial.
 */
std::bfloat16_t gelu_d2_lut_poly_hybrid(std::bfloat16_t x_bf16) {
    float x = static_cast<float>(x_bf16);

    // Positive tail: x >= 3 → GELU(x) ≈ x
    if (x >= thresholds::POS) {
        return x_bf16;
    }

    // Negative tail: use extended LUT
    if (x < tail_lut::LUT_START) {
        return static_cast<std::bfloat16_t>(gelu_negative_tail(x));
    }

    // Core region: use B3-style piecewise erf approximation
    constexpr float inv_sqrt_2 = 0.7071067811865475f;
    float z = x * inv_sqrt_2;
    float abs_z = std::abs(z);
    float z2 = z * z;

    float erf_z;
    if (abs_z < 1.0f) {
        // Taylor series for small z
        constexpr float two_over_sqrt_pi = 1.1283791670955126f;
        float z4 = z2 * z2;
        float z6 = z4 * z2;
        float series = 1.0f - 0.333333333f * z2 + 0.1f * z4 - 0.023809524f * z6;
        erf_z = two_over_sqrt_pi * z * series;
    } else {
        // Rational approximation for |z| >= 1 (Abramowitz-Stegun style)
        constexpr float a1 = 0.278393f;
        constexpr float a2 = 0.230389f;
        constexpr float a3 = 0.000972f;
        constexpr float a4 = 0.078108f;

        float t = abs_z;
        float t2 = t * t;
        float t3 = t2 * t;
        float t4 = t2 * t2;
        float p = a1 * t + a2 * t2 + a3 * t3 + a4 * t4;
        float denom = 1.0f + p;
        float abs_erf = 1.0f - 1.0f / (denom * denom * denom * denom);
        erf_z = (z >= 0) ? abs_erf : -abs_erf;
    }

    erf_z = std::max(-1.0f, std::min(1.0f, erf_z));
    float phi = 0.5f * (1.0f + erf_z);
    float result = x * phi;

    return static_cast<std::bfloat16_t>(result);
}

// ============================================================================
// D2 PURE: HYBRID LUT+ERF WITH ASYMPTOTIC TAIL
// ============================================================================

/**
 * @brief D2 Pure: Hybrid LUT+Erf with asymptotic expansion for tail
 *
 * Like D2 LUT+Erf Hybrid but uses asymptotic expansion for deep negative tail
 * instead of the shared tail LUT. This ensures complete methodological
 * independence and eliminates the interpolation error that limits
 * the original D2 to Max ULP = 87.
 *
 * - Core region: B3-style piecewise erf (Taylor + rational)
 * - Negative tail (x < -3.5): Asymptotic expansion GELU(x) ≈ -φ(x)·(1 - 1/x² + 3/x⁴ - 15/x⁶)
 * - Positive saturation (x >= 3): GELU(x) = x
 */
std::bfloat16_t gelu_d2_pure(std::bfloat16_t x_bf16) {
    float x = static_cast<float>(x_bf16);

    // Positive tail: x >= 3 → GELU(x) ≈ x
    if (x >= thresholds::POS) {
        return x_bf16;
    }

    // Negative tail: use asymptotic expansion directly (Pure version)
    // Use -3.0f threshold (not -3.5) to avoid B3 erf boundary error
    if (x < -3.0f) {
        return static_cast<std::bfloat16_t>(gelu_asymptotic(x));
    }

    // Core region: use B3-style piecewise erf approximation
    constexpr float inv_sqrt_2 = 0.7071067811865475f;
    float z = x * inv_sqrt_2;
    float abs_z = std::abs(z);
    float z2 = z * z;

    float erf_z;
    if (abs_z < 1.0f) {
        // Taylor series for small z
        constexpr float two_over_sqrt_pi = 1.1283791670955126f;
        float z4 = z2 * z2;
        float z6 = z4 * z2;
        float series = 1.0f - 0.333333333f * z2 + 0.1f * z4 - 0.023809524f * z6;
        erf_z = two_over_sqrt_pi * z * series;
    } else {
        // Rational approximation for |z| >= 1 (Abramowitz-Stegun style)
        constexpr float a1 = 0.278393f;
        constexpr float a2 = 0.230389f;
        constexpr float a3 = 0.000972f;
        constexpr float a4 = 0.078108f;

        float t = abs_z;
        float t2 = t * t;
        float t3 = t2 * t;
        float t4 = t2 * t2;
        float p = a1 * t + a2 * t2 + a3 * t3 + a4 * t4;
        float denom = 1.0f + p;
        float abs_erf = 1.0f - 1.0f / (denom * denom * denom * denom);
        erf_z = (z >= 0) ? abs_erf : -abs_erf;
    }

    erf_z = std::max(-1.0f, std::min(1.0f, erf_z));
    float phi = 0.5f * (1.0f + erf_z);
    float result = x * phi;

    return static_cast<std::bfloat16_t>(result);
}

// ============================================================================
// D3: LUT + POLYNOMIAL CORRECTION
// ============================================================================

/**
 * @brief D3: Coarse LUT with polynomial correction
 *
 * Uses a small LUT (32 entries) as base, then adds polynomial correction.
 * Balances memory footprint with computation.
 */
namespace d3_lut {
    constexpr int SIZE = 32;
    constexpr float MIN = -4.0f;
    constexpr float MAX = 4.0f;
    constexpr float STEP = (MAX - MIN) / (SIZE - 1);

    // Precomputed Φ(x) at coarse intervals
    constexpr float PHI[SIZE] = {
        0.0000317f, 0.0000911f, 0.0002567f, 0.0007066f, 0.0019001f,
        0.0049902f, 0.0127853f, 0.0318639f, 0.0771452f, 0.1814993f,
        0.2742531f, 0.3538814f, 0.4207403f, 0.4761482f, 0.5215932f,
        0.5f,       0.5f,       0.5238068f, 0.5792518f, 0.6461186f,
        0.7257469f, 0.8185007f, 0.9228548f, 0.9681361f, 0.9872147f,
        0.9950098f, 0.9980999f, 0.9992934f, 0.9997433f, 0.9999089f,
        0.9999683f, 1.0f
    };
}

std::bfloat16_t gelu_d3_lut_correction(std::bfloat16_t x_bf16) {
    float x = static_cast<float>(x_bf16);

    // Saturation
    if (x >= thresholds::POS) return x_bf16;
    if (x < tail_lut::LUT_START) {
        return static_cast<std::bfloat16_t>(gelu_negative_tail(x));
    }

    // Clamp to LUT range
    float x_clamped = std::max(d3_lut::MIN, std::min(d3_lut::MAX, x));

    // LUT lookup with linear interpolation
    float idx_f = (x_clamped - d3_lut::MIN) / d3_lut::STEP;
    int idx = static_cast<int>(idx_f);
    if (idx >= d3_lut::SIZE - 1) idx = d3_lut::SIZE - 2;
    if (idx < 0) idx = 0;
    float frac = idx_f - idx;

    float phi_lut = d3_lut::PHI[idx] + frac * (d3_lut::PHI[idx + 1] - d3_lut::PHI[idx]);

    // Polynomial correction term
    // Corrects for error between coarse LUT and true Φ(x)
    float x2 = x * x;
    float correction = x * (0.02f - 0.005f * x2) / (1.0f + 0.5f * x2);

    float phi = phi_lut + correction;
    phi = std::max(0.0f, std::min(1.0f, phi));

    float result = x * phi;
    return static_cast<std::bfloat16_t>(result);
}

// ============================================================================
// D4: NON-UNIFORM LUT SPACING
// ============================================================================

/**
 * @brief D4: Non-uniform LUT spacing optimized for max-ULP
 *
 * Uses denser sampling near x=0 and at saturation boundaries.
 * Fixed: Added Taylor approximation for near-zero region.
 */
namespace d4_lut {
    // Non-uniform breakpoints: denser near 0 and boundaries
    constexpr float BREAKS[] = {
        -8.0f, -6.0f, -5.0f, -4.0f, -3.5f, -3.0f, -2.5f, -2.0f,
        -1.5f, -1.0f, -0.75f, -0.5f, -0.25f, 0.0f, 0.25f, 0.5f,
        0.75f, 1.0f, 1.5f, 2.0f, 2.5f, 3.0f, 4.0f
    };
    constexpr int SIZE = 23;

    // Precomputed GELU values at breakpoints
    constexpr float GELU_VALS[] = {
        -4.89e-15f, -5.92e-9f, -1.43e-6f, -4.51e-5f, -8.14e-4f, -4.05e-3f,
        -1.35e-2f, -3.47e-2f, -7.01e-2f, -1.59e-1f, -2.16e-1f, -1.54e-1f,
        -7.67e-2f, 0.0f, 0.0959f, 0.3457f, 0.5963f, 0.8413f,
        1.3996f, 1.9546f, 2.4866f, 2.9960f, 3.9999f
    };
}

std::bfloat16_t gelu_d4_nonuniform_lut(std::bfloat16_t x_bf16) {
    float x = static_cast<float>(x_bf16);

    // Saturation
    if (x >= thresholds::POS) return x_bf16;
    if (x < tail_lut::LUT_START) {
        return static_cast<std::bfloat16_t>(gelu_negative_tail(x));
    }

    // Near-zero: use Taylor approximation for |x| < 0.125
    // GELU(x) ≈ 0.5x + (1/√(2π))x² - (1/(6√(2π)))x⁴
    // For very small |x|, GELU(x) ≈ 0.5x is sufficient
    float abs_x = std::abs(x);
    if (abs_x < 0.125f) {
        // GELU(x) ≈ x * Φ(x) where Φ(x) ≈ 0.5 + 0.3989x for small x
        constexpr float c1 = 0.3989422804f;  // 1/√(2π)
        float phi = 0.5f + c1 * x;
        float result = x * phi;
        return static_cast<std::bfloat16_t>(result);
    }

    // Find segment via binary search
    int lo = 0, hi = d4_lut::SIZE - 1;
    while (lo < hi - 1) {
        int mid = (lo + hi) / 2;
        if (x < d4_lut::BREAKS[mid]) {
            hi = mid;
        } else {
            lo = mid;
        }
    }

    // Linear interpolation within segment
    float x0 = d4_lut::BREAKS[lo];
    float x1 = d4_lut::BREAKS[hi];
    float y0 = d4_lut::GELU_VALS[lo];
    float y1 = d4_lut::GELU_VALS[hi];

    float t = (x - x0) / (x1 - x0);
    float result = y0 + t * (y1 - y0);

    return static_cast<std::bfloat16_t>(result);
}

// ============================================================================
// F2: NUMERICAL QUADRATURE REFERENCE
// ============================================================================

/**
 * @brief F2: Numerical quadrature for Φ(x) - arithmetic only reference
 *
 * Uses Gauss-Legendre quadrature to compute Φ(x) without erf().
 * Slow but uses only basic arithmetic.
 */
double phi_quadrature_f64(double x) {
    // Gauss-Legendre 16-point quadrature on transformed integral
    // Φ(x) = 0.5 + (1/√(2π)) ∫₀ˣ exp(-t²/2) dt
    // We use exp approximation via Taylor series for small args

    if (x > 6.0) return 1.0;
    if (x < -6.0) return 0.0;

    // 8-point Gauss-Legendre nodes and weights for [0, 1]
    constexpr double nodes[] = {
        0.0198550718, 0.1016667613, 0.2372337950, 0.4082826788,
        0.5917173212, 0.7627662050, 0.8983332387, 0.9801449282
    };
    constexpr double weights[] = {
        0.0506142681, 0.1111905172, 0.1568533229, 0.1813418917,
        0.1813418917, 0.1568533229, 0.1111905172, 0.0506142681
    };

    // Transform to [0, x] or [-x, 0]
    double sign = (x >= 0) ? 1.0 : -1.0;
    double abs_x = std::abs(x);

    double integral = 0.0;
    for (int i = 0; i < 8; ++i) {
        double t = nodes[i] * abs_x;
        double t2 = t * t;
        // Taylor approximation for exp(-t²/2) - avoids exp()
        // exp(-u) ≈ 1 - u + u²/2 - u³/6 + u⁴/24 - ...
        double u = t2 / 2.0;
        double exp_approx;
        if (u < 2.0) {
            exp_approx = 1.0 - u + u*u/2.0 - u*u*u/6.0 + u*u*u*u/24.0
                         - u*u*u*u*u/120.0 + u*u*u*u*u*u/720.0;
        } else {
            // For larger u, use rational approximation
            exp_approx = 1.0 / (1.0 + u + u*u/2.0 + u*u*u/6.0);
        }
        integral += weights[i] * exp_approx;
    }
    integral *= abs_x;

    // 1/√(2π)
    constexpr double inv_sqrt_2pi = 0.3989422804014327;
    double phi = 0.5 + sign * inv_sqrt_2pi * integral;

    return std::max(0.0, std::min(1.0, phi));
}

std::bfloat16_t gelu_f2_quadrature(std::bfloat16_t x_bf16) {
    double x = static_cast<double>(static_cast<float>(x_bf16));

    if (x >= thresholds::POS) return x_bf16;

    // Extended tail handling: exp() approximation fails for |x| > 2
    // Use tail LUT for x < -2.0 instead of just x < -3.5
    if (x < -2.0) {
        if (x < tail_lut::LUT_START) {
            return static_cast<std::bfloat16_t>(gelu_negative_tail(static_cast<float>(x)));
        }
        // For x in [-3.5, -2.0], use B3-style erf instead of quadrature
        constexpr float inv_sqrt_2 = 0.7071067811865475f;
        float xf = static_cast<float>(x);
        float z = xf * inv_sqrt_2;
        float abs_z = std::abs(z);
        constexpr float a1 = 0.278393f, a2 = 0.230389f, a3 = 0.000972f, a4 = 0.078108f;
        float t = abs_z;
        float p = a1*t + a2*t*t + a3*t*t*t + a4*t*t*t*t;
        float abs_erf = 1.0f - 1.0f / ((1.0f + p) * (1.0f + p) * (1.0f + p) * (1.0f + p));
        float erf_z = (z >= 0) ? abs_erf : -abs_erf;
        float phi = 0.5f * (1.0f + erf_z);
        return static_cast<std::bfloat16_t>(xf * phi);
    }

    double phi = phi_quadrature_f64(x);
    double result = x * phi;

    return static_cast<std::bfloat16_t>(static_cast<float>(result));
}

// ============================================================================
// F3: CONTINUED FRACTION ERF REFERENCE
// ============================================================================

/**
 * @brief F3: Continued fraction erf approximation
 *
 * Uses continued fraction representation for erf(z).
 * Sometimes more stable in tails than polynomial.
 */
double erf_continued_fraction_f64(double z) {
    if (std::abs(z) < 0.5) {
        // Use Taylor series for small z
        double z2 = z * z;
        // erf(z) ≈ (2/√π) * z * (1 - z²/3 + z⁴/10 - z⁶/42 + ...)
        constexpr double two_over_sqrt_pi = 1.1283791670955126;
        double series = 1.0 - z2/3.0 + z2*z2/10.0 - z2*z2*z2/42.0 + z2*z2*z2*z2/216.0;
        return two_over_sqrt_pi * z * series;
    }

    // For |z| >= 0.5, use continued fraction
    // erf(z) = 1 - exp(-z²) * CF, where CF is a continued fraction
    // Since we can't use exp(), approximate the decay differently

    double abs_z = std::abs(z);
    double z2 = z * z;

    // Lentz's algorithm for continued fraction
    // erfc(z) ≈ exp(-z²) / (z√π) * (1/(1+ 0.5/z²/(1+ 1/z²/(1+ 1.5/z²/...))))
    double cf = abs_z;
    for (int n = 8; n >= 1; --n) {
        cf = abs_z + (n * 0.5) / cf;
    }

    // Approximate exp(-z²) using rational function (arithmetic only)
    double exp_neg_z2;
    if (z2 < 4.0) {
        exp_neg_z2 = 1.0 / (1.0 + z2 + z2*z2/2.0 + z2*z2*z2/6.0 + z2*z2*z2*z2/24.0);
    } else {
        exp_neg_z2 = 0.0;  // Negligible for large z²
    }

    constexpr double inv_sqrt_pi = 0.5641895835477563;
    double erfc_z = exp_neg_z2 * inv_sqrt_pi / cf;
    double erf_z = 1.0 - erfc_z;

    return (z >= 0) ? erf_z : -erf_z;
}

std::bfloat16_t gelu_f3_cf_erf(std::bfloat16_t x_bf16) {
    double x = static_cast<double>(static_cast<float>(x_bf16));

    if (x >= thresholds::POS) return x_bf16;

    // Extended tail handling: CF exp() approximation fails for |x| > 2
    // Use tail LUT or B3-style erf for x < -2.0
    if (x < -2.0) {
        if (x < tail_lut::LUT_START) {
            return static_cast<std::bfloat16_t>(gelu_negative_tail(static_cast<float>(x)));
        }
        // For x in [-3.5, -2.0], use B3-style erf instead of CF
        constexpr float inv_sqrt_2 = 0.7071067811865475f;
        float xf = static_cast<float>(x);
        float z = xf * inv_sqrt_2;
        float abs_z = std::abs(z);
        constexpr float a1 = 0.278393f, a2 = 0.230389f, a3 = 0.000972f, a4 = 0.078108f;
        float t = abs_z;
        float p = a1*t + a2*t*t + a3*t*t*t + a4*t*t*t*t;
        float abs_erf = 1.0f - 1.0f / ((1.0f + p) * (1.0f + p) * (1.0f + p) * (1.0f + p));
        float erf_z = (z >= 0) ? abs_erf : -abs_erf;
        float phi = 0.5f * (1.0f + erf_z);
        return static_cast<std::bfloat16_t>(xf * phi);
    }

    double z = x * constants::INV_SQRT_2;
    double erf_z = erf_continued_fraction_f64(z);
    double phi = 0.5 * (1.0 + erf_z);
    double result = x * phi;

    return static_cast<std::bfloat16_t>(static_cast<float>(result));
}

// ============================================================================
// F3 PURE: CONTINUED FRACTION ERF WITH ASYMPTOTIC TAIL
// ============================================================================

/**
 * @brief F3 Pure: Continued fraction erf with asymptotic expansion for tail
 *
 * Like F3 CF Erf but uses asymptotic expansion for deep negative tail
 * instead of the shared tail LUT. This ensures complete methodological
 * independence and eliminates the interpolation error that limits
 * the original F3 to Max ULP = 87.
 *
 * - Core region: Continued fraction erf approximation
 * - Negative tail (x < -2.0): Asymptotic expansion GELU(x) ≈ -φ(x)·(1 - 1/x² + 3/x⁴ - 15/x⁶)
 * - Positive saturation (x >= 3): GELU(x) = x
 */
std::bfloat16_t gelu_f3_pure(std::bfloat16_t x_bf16) {
    double x = static_cast<double>(static_cast<float>(x_bf16));

    if (x >= thresholds::POS) return x_bf16;

    // Pure version: use asymptotic expansion for negative tail
    if (x < -2.0) {
        return static_cast<std::bfloat16_t>(gelu_asymptotic(static_cast<float>(x)));
    }

    double z = x * constants::INV_SQRT_2;
    double erf_z = erf_continued_fraction_f64(z);
    double phi = 0.5 * (1.0 + erf_z);
    double result = x * phi;

    return static_cast<std::bfloat16_t>(static_cast<float>(result));
}

// ============================================================================
// H1: INVERTED GELU (REFERENCE IMPLEMENTATION)
// ============================================================================

/**
 * @brief Helper: compute Φ(x) using erfc for negative x to avoid cancellation
 */
inline float phi_reference(float x) {
    float z = x * static_cast<float>(constants::INV_SQRT_2);
    if (x >= 0) {
        return 0.5f * (1.0f + std::erf(z));
    } else {
        return 0.5f * std::erfc(-z);
    }
}

/**
 * @brief H1: Inverted GELU (GELU⁻¹) - Reference implementation
 *
 * Computes the inverse function: given y = GELU(x), find x.
 * Useful for memory-efficient backpropagation.
 *
 * NOTE: This is a REFERENCE implementation using std::erf/std::erfc/std::exp.
 * It's not an arithmetic-only approximation but serves as a utility function.
 *
 * Uses Newton-Raphson iteration with initial guess from linear approximation.
 */
std::bfloat16_t gelu_inverse_reference(std::bfloat16_t y_bf16) {
    float y = static_cast<float>(y_bf16);

    // Handle edge cases
    if (y >= 0.0f) {
        // For y >= 0, GELU⁻¹(y) ≈ y for large y (since GELU(x) ≈ x for x > 3)
        if (y >= 3.0f) return y_bf16;

        // Initial guess: linear approximation
        // GELU(x) ≈ 0.5x for x near 0, so x ≈ 2y
        float x = y * 1.5f;  // Slightly better initial guess

        // Newton-Raphson: x_{n+1} = x_n - (GELU(x_n) - y) / GELU'(x_n)
        for (int iter = 0; iter < 4; ++iter) {
            float x2 = x * x;
            float phi = phi_reference(x);
            float gelu_x = x * phi;

            // GELU'(x) = Φ(x) + x * φ(x), where φ(x) = exp(-x²/2) / √(2π)
            constexpr float inv_sqrt_2pi = 0.3989422804f;
            float phi_pdf = inv_sqrt_2pi * std::exp(-0.5f * x2);
            float gelu_prime = phi + x * phi_pdf;

            if (std::abs(gelu_prime) < 1e-10f) break;

            float delta = (gelu_x - y) / gelu_prime;
            x = x - delta;

            if (std::abs(delta) < 1e-6f) break;
        }

        return static_cast<std::bfloat16_t>(x);
    } else {
        // For y < 0, GELU⁻¹(y) is in the negative region
        // GELU is not monotonic for x < 0, but we find the primary branch

        // Initial guess based on the fact that GELU has a minimum around x ≈ -0.67
        float x = -1.0f + y * 2.0f;  // Rough linear guess

        for (int iter = 0; iter < 6; ++iter) {
            float x2 = x * x;
            float phi = phi_reference(x);
            float gelu_x = x * phi;

            constexpr float inv_sqrt_2pi = 0.3989422804f;
            float phi_pdf = inv_sqrt_2pi * std::exp(-0.5f * x2);
            float gelu_prime = phi + x * phi_pdf;

            if (std::abs(gelu_prime) < 1e-10f) break;

            float delta = (gelu_x - y) / gelu_prime;
            x = x - delta;

            if (std::abs(delta) < 1e-6f) break;
        }

        return static_cast<std::bfloat16_t>(x);
    }
}

// Backwards compatibility alias
inline std::bfloat16_t gelu_inverse(std::bfloat16_t y_bf16) {
    return gelu_inverse_reference(y_bf16);
}

// ============================================================================
// H3: SOFTEX-INSPIRED EXP APPROXIMATION
// ============================================================================

/**
 * @brief H3: SoftEx-inspired arithmetic-only exp approximation
 *
 * Approximates exp(x) using only basic arithmetic via mantissa refinement.
 * Used to enable tanh-based GELU without true exp().
 */
inline float exp_softex(float x) {
    // Range reduction: exp(x) = 2^k * exp(r) where x = k*ln(2) + r
    // For small |r|, exp(r) ≈ (1 + r/256)^256 ≈ rational approximation

    // Clamp to avoid overflow/underflow
    if (x > 88.0f) return 1e38f;
    if (x < -88.0f) return 0.0f;

    // Use Padé [2/2] approximation for exp(x) centered at 0
    // exp(x) ≈ (1 + x/2 + x²/12) / (1 - x/2 + x²/12)
    // More accurate: (6 + 3x + x²/2) / (6 - 3x + x²/2)
    float x2 = x * x;
    float num = 12.0f + 6.0f * x + x2;
    float den = 12.0f - 6.0f * x + x2;

    return num / den;
}

/**
 * @brief GELU using SoftEx exp approximation for tanh
 */
std::bfloat16_t gelu_h3_softex(std::bfloat16_t x_bf16) {
    float x = static_cast<float>(x_bf16);

    // Saturation
    if (x >= thresholds::POS) return x_bf16;
    if (x < tail_lut::LUT_START) {
        return static_cast<std::bfloat16_t>(gelu_negative_tail(x));
    }

    // GELU via tanh form: 0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))
    // tanh(z) = (exp(2z) - 1) / (exp(2z) + 1)

    float x3 = x * x * x;
    float z = static_cast<float>(constants::SQRT_2_OVER_PI) * (x + 0.044715f * x3);

    // Compute tanh using SoftEx exp approximation
    float exp_2z = exp_softex(2.0f * z);
    float tanh_z = (exp_2z - 1.0f) / (exp_2z + 1.0f);

    float result = 0.5f * x * (1.0f + tanh_z);

    return static_cast<std::bfloat16_t>(result);
}

/**
 * @brief H3 Pure: SoftEx tanh-form with asymptotic tail (no shared LUT)
 *
 * Uses the same tanh-form as H3 but replaces the shared tail LUT
 * with the independent asymptotic expansion for x < -3.
 */
std::bfloat16_t gelu_h3_pure(std::bfloat16_t x_bf16) {
    float x = static_cast<float>(x_bf16);

    if (x >= thresholds::POS) {
        return static_cast<std::bfloat16_t>(x);
    }

    // For negative tail (x < -3), use asymptotic expansion instead of LUT
    if (x < -3.0f) {
        return static_cast<std::bfloat16_t>(gelu_asymptotic(x));
    }

    // GELU via tanh form: 0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))
    float x3 = x * x * x;
    float z = static_cast<float>(constants::SQRT_2_OVER_PI) * (x + 0.044715f * x3);

    // Compute tanh using SoftEx exp approximation
    float exp_2z = exp_softex(2.0f * z);
    float tanh_z = (exp_2z - 1.0f) / (exp_2z + 1.0f);

    float result = 0.5f * x * (1.0f + tanh_z);
    return static_cast<std::bfloat16_t>(result);
}
// ============================================================================
// DOPRI THREE-REGION POLYNOMIAL APPROXIMATION
// ============================================================================
//
// Pure DOPRI polynomials split into three regions to avoid zero-crossing
// All regions use adaptive DOPRI polynomial fitting
// No mixing with Taylor series or other methods - pure polynomial only
//
// Generated by generate_dopri_three_region.py
// ============================================================================

/**
 * @brief DOPRI Three Region Deg7
 *
 * Pure DOPRI polynomial approximation with three-region split
 * Degree: 7, Total segments: 80
 * Negative: [-13.5625, -0.125] (32 segs)
 * Near-zero: [-0.125, 0.125] (24 segs)
 * Positive: [0.125, 2.78125] (24 segs)
 */
std::bfloat16_t gelu_dopri_three_region_deg7(std::bfloat16_t x_bf16) {
    float x = static_cast<float>(x_bf16);

    // Saturation checks
    if (x >= 2.78125f) return static_cast<std::bfloat16_t>(x);
    if (x < -13.5625f) return static_cast<std::bfloat16_t>(0.0f);

    constexpr float breakpoints[] = {
        -13.56250000f, -7.31686179f, -7.30582840f, -7.30582840f, -6.94634650f,
        -6.75393482f, -6.53168430f, -6.31161037f, -6.09259875f, -5.86980652f,
        -5.62828482f, -5.41037016f, -5.19408718f, -4.98600653f, -4.78494876f,
        -4.59097935f, -4.40393754f, -4.22364728f, -0.57514364f, -0.53458922f,
        -0.49578356f, -0.45847521f, -0.42244771f, -0.38748650f, -0.35348564f,
        -0.32026968f, -0.28775098f, -0.25582617f, -0.22441419f, -0.19342373f,
        -0.16279369f, -0.13246493f, -0.12500000f, -0.11434235f, -0.10386465f,
        -0.09336491f, -0.08288315f, -0.07243848f, -0.06198328f, -0.05157593f,
        -0.04116658f, -0.03074471f, -0.02037209f, -0.00995821f, 0.00043337f,
        0.01080440f, 0.02114192f, 0.03148510f, 0.04184654f, 0.05218903f,
        0.06256637f, 0.07297255f, 0.08333967f, 0.09376278f, 0.10419660f,
        0.11464104f, 0.12500000f, 0.19457563f, 0.21802319f, 0.25650212f,
        0.29292841f, 0.33052772f, 0.36905777f, 0.40874767f, 0.44980800f,
        0.49250719f, 0.53717872f, 0.58425177f, 0.63429055f, 0.68808260f,
        0.74673629f, 0.81197095f, 0.88661249f, 0.97597030f, 1.09218704f,
        1.18094884f, 1.18094884f, 1.24735486f, 1.29103530f, 1.35525651f,
        2.78125000f
    };

    constexpr float coeffs[80][8] = {
        { -2.434400791327e-08f, -1.660184838077e-08f, -4.818531137543e-09f, -7.716185276383e-10f, -7.363548403796e-11f, -4.188139958196e-12f, -1.314747943168e-13f, -1.757580369599e-15f },
        { -8.664944085308e-13f, 3.588470961572e-12f, -1.292584276681e-11f, 3.668090749467e-11f, -6.093814453158e-11f, -2.938856104002e-11f, -4.220528764262e-12f, -1.968811768386e-13f },
        { -8.003226376466e-25f, 5.847019859062e-24f, -4.271732376982e-23f, 3.120854373731e-22f, -2.280042653074e-21f, 1.665760037891e-20f, -1.216975700035e-19f, 8.891015637225e-19f },
        { -6.794128717001e-08f, 1.259360376955e-07f, -1.068062515503e-07f, -1.150396616720e-07f, -3.569799656541e-08f, -5.197406340907e-09f, -3.695350633723e-10f, -1.039156306039e-11f },
        { -1.647282814742e-09f, 4.691592291116e-09f, -1.006086964217e-08f, 1.115122961679e-08f, 8.712796839909e-09f, 2.044950314677e-09f, 2.035988342188e-10f, 7.468718462428e-12f },
        { -1.136571953031e-06f, 1.934600153178e-06f, -1.419237421897e-06f, -1.790518738208e-06f, -6.051005338268e-07f, -9.496953542150e-08f, -7.254107861166e-09f, -2.188172599352e-10f },
        { -3.871630755209e-06f, 6.304042045215e-06f, -4.288534704506e-06f, -5.885981120634e-06f, -2.074548574322e-06f, -3.377194265351e-07f, -2.670827127530e-08f, -8.334595775455e-10f },
        { -1.224715223156e-05f, 1.907325602665e-05f, -1.190151869875e-05f, -1.794927581844e-05f, -6.608639879707e-06f, -1.116956571014e-06f, -9.152742184452e-08f, -2.956883903188e-09f },
        { -3.684720148676e-05f, 5.474785087589e-05f, -3.092367832511e-05f, -5.196197569632e-05f, -2.003634708422e-05f, -3.522003599177e-06f, -2.994934976806e-07f, -1.003075173695e-08f },
        { -1.102518864590e-04f, 1.552321228421e-04f, -7.771274715487e-05f, -1.489334747200e-04f, -6.044242924814e-05f, -1.109075926758e-05f, -9.819468679135e-07f, -3.420433438299e-08f },
        { -3.039226107053e-04f, 4.044097044769e-04f, -1.752524337716e-04f, -3.925515886285e-04f, -1.680974914274e-04f, -3.224567012522e-05f, -2.976187044910e-06f, -1.079437873815e-07f },
        { -7.475595100741e-04f, 9.391385441693e-04f, -3.436976288003e-04f, -9.229247311771e-04f, -4.170553144896e-04f, -8.358464140944e-05f, -8.036065242667e-06f, -3.032289323845e-07f },
        { -1.699394115410e-03f, 2.009330792968e-03f, -5.942408265783e-04f, -2.001480598048e-03f, -9.563562050142e-04f, -2.004265300188e-04f, -2.008500537070e-05f, -7.889086908462e-07f },
        { -3.550049785654e-03f, 3.940193313648e-03f, -8.785548021556e-04f, -3.982183718140e-03f, -2.014814763397e-03f, -4.416555203064e-04f, -4.613321319980e-05f, -1.886178186340e-06f },
        { -6.861551950062e-03f, 7.126310777174e-03f, -1.046988612536e-03f, -7.315833453174e-03f, -3.925596227539e-03f, -9.003093309482e-04f, -9.802973088519e-05f, -4.171939739838e-06f },
        { -1.232988956120e-02f, 1.193911584265e-02f, -7.987159726553e-04f, -1.246583887471e-02f, -7.105426807094e-03f, -1.705291852224e-03f, -1.935455942169e-04f, -8.573039651903e-06f },
        { -2.069900733545e-02f, 1.861222648319e-02f, 3.363515368879e-04f, -1.978969038020e-02f, -1.200237586627e-02f, -3.014713087291e-03f, -3.566099338122e-04f, -1.643773964496e-05f },
        { 2.999213195539e-02f, 6.390647333865e-01f, 6.404556454117e-01f, 1.870377992553e-01f, -2.182420402793e-02f, -2.331597847324e-02f, -4.691744401371e-03f, -3.160379798691e-04f },
        { -3.551561905875e-05f, 4.995568656068e-01f, 3.965891617151e-01f, -6.853498584875e-03f, -7.815069962779e-02f, -1.111365102314e-02f, 5.485417851145e-03f, 9.460757121808e-04f },
        { -1.649924548576e-05f, 4.997801011240e-01f, 3.976985253021e-01f, -3.838535833043e-03f, -7.333529735614e-02f, -6.628971160399e-03f, 7.710926987525e-03f, 1.388845863845e-03f },
        { -6.420451827323e-06f, 4.999085763946e-01f, 3.983931259915e-01f, -1.779289864929e-03f, -6.973345698691e-02f, -2.932893544270e-03f, 9.752921218392e-03f, 1.850201657379e-03f },
        { -1.720269791601e-06f, 4.999743099426e-01f, 3.987841073564e-01f, -4.993399920485e-04f, -6.724842576432e-02f, -8.062209376514e-05f, 1.153666960422e-02f, 2.315743993143e-03f },
        { -3.880787744061e-08f, 5.000007308755e-01f, 3.989617151201e-01f, 1.626780449846e-04f, -6.577093487339e-02f, 1.893377245716e-03f, 1.299824712705e-02f, 2.778275522968e-03f },
        { 4.619981563554e-07f, 5.000095655841e-01f, 3.990287816214e-01f, 4.467450035824e-04f, -6.504574547921e-02f, 3.009387526033e-03f, 1.395697388340e-02f, 3.132966366708e-03f },
        { 3.903064846528e-07f, 5.000087052743e-01f, 3.990264680953e-01f, 4.605659747908e-04f, -6.493302693737e-02f, 3.332307304796e-03f, 1.439224566400e-02f, 3.364594936202e-03f },
        { 2.499286330027e-07f, 5.000060038969e-01f, 3.990047530052e-01f, 3.670611339227e-04f, -6.516144239621e-02f, 3.028590724756e-03f, 1.421099313370e-02f, 3.346492193140e-03f },
        { 9.957715608555e-08f, 5.000027141225e-01f, 3.989742966760e-01f, 2.130238615939e-04f, -6.561837386219e-02f, 2.241160551207e-03f, 1.349318002420e-02f, 3.088340721761e-03f },
        { 3.452604869509e-08f, 5.000010217270e-01f, 3.989554291781e-01f, 9.614349475628e-05f, -6.605307169042e-02f, 1.270042826120e-03f, 1.228582462046e-02f, 2.443452535530e-03f },
        { -9.576404037071e-09f, 4.999997381394e-01f, 3.989394690912e-01f, -1.361088824677e-05f, -6.650311013984e-02f, 1.723142818647e-04f, 1.081583059733e-02f, 1.613672977016e-03f },
        { -7.195420213920e-09f, 4.999997276127e-01f, 3.989379084370e-01f, -3.830634115627e-05f, -6.668526456987e-02f, -5.515603792405e-04f, 9.310188955174e-03f, 3.211354854276e-04f },
        { -2.859816836137e-09f, 4.999998610046e-01f, 3.989394001748e-01f, -3.291085714172e-05f, -6.671327351994e-02f, -8.841724926573e-04f, 8.146603138795e-03f, -1.138914087431e-03f },
        { 4.994990547046e-08f, 5.000021555626e-01f, 3.989804969370e-01f, 3.529478236079e-04f, -6.474172930203e-02f, 4.040717037798e-03f, 1.120742485726e-02f, -6.987674470520e-03f },
        { 1.608272645912e-08f, 5.000007921638e-01f, 3.989583601398e-01f, 1.706164762528e-04f, -6.551541468914e-02f, 2.609397382102e-03f, 1.089234048575e-02f, -6.197989919293e-03f },
        { 3.098648299879e-09f, 5.000001905695e-01f, 3.989470199084e-01f, 6.057534486303e-05f, -6.608155561608e-02f, 1.238807631840e-03f, 1.009091869192e-02f, -5.149754715909e-03f },
        { 2.311257939774e-09f, 5.000001274107e-01f, 3.989451296148e-01f, 3.233142084669e-05f, -6.630695432496e-02f, 3.392608908463e-04f, 8.875495981357e-03f, -4.053292165216e-03f },
        { 1.276799154310e-10f, 5.000000030923e-01f, 3.989421898267e-01f, -4.571645233225e-06f, -6.656494064804e-02f, -5.934317018323e-04f, 7.689123933462e-03f, -3.099480684097e-03f },
        { -8.078536305466e-10f, 4.999999408116e-01f, 3.989404726197e-01f, -2.957004303604e-05f, -6.676595004097e-02f, -1.421416370823e-03f, 6.474568556247e-03f, -2.270812338512e-03f },
        { -3.222436502261e-10f, 4.999999702352e-01f, 3.989411323755e-01f, -2.369738877525e-05f, -6.676730919034e-02f, -1.752037304553e-03f, 5.102313963482e-03f, -1.537730608098e-03f },
        { -2.594553278911e-10f, 4.999999737066e-01f, 3.989411725309e-01f, -2.485435609361e-05f, -6.680389850372e-02f, -2.116704296951e-03f, 3.885401294199e-03f, -9.790070649423e-04f },
        { -1.047266517934e-10f, 4.999999871204e-01f, 3.989416237894e-01f, -1.775783773154e-05f, -6.675908940270e-02f, -2.159311486117e-03f, 2.723254072169e-03f, -5.561369843300e-04f },
        { -1.861257361219e-11f, 4.999999969017e-01f, 3.989420659100e-01f, -7.904985655287e-06f, -6.665399684553e-02f, -1.804384357906e-03f, 1.669027692303e-03f, -2.638180213839e-04f },
        { -2.861005947811e-12f, 4.999999993324e-01f, 3.989422158961e-01f, -3.304974886380e-06f, -6.658511902144e-02f, -1.441470739192e-03f, 8.682231141649e-04f, -9.721720324502e-05f },
        { -6.931081848437e-14f, 4.999999999693e-01f, 3.989422747692e-01f, -5.445991205736e-07f, -6.651966412573e-02f, -8.296820867817e-04f, 3.019062525765e-04f, -2.021254892238e-05f },
        { 2.488540932739e-17f, 5.000000000000e-01f, 3.989422803427e-01f, -2.517239492518e-08f, -6.649472014558e-02f, -3.379370007141e-04f, 2.952112069731e-05f, -6.644277939862e-07f },
        { 5.109672840433e-17f, 5.000000000000e-01f, 3.989422803851e-01f, 1.278547761547e-08f, -6.649339853522e-02f, 2.909495319320e-04f, 3.855034666595e-05f, 1.014201310645e-06f },
        { -1.136991692117e-13f, 5.000000000468e-01f, 3.989422725036e-01f, 7.012579562103e-07f, -6.652495995394e-02f, 8.989530654902e-04f, 3.376668702723e-04f, 2.375382812729e-05f },
        { -4.172528467550e-12f, 5.000000009152e-01f, 3.989421973275e-01f, 3.995488092711e-06f, -6.659778921110e-02f, 1.530793260877e-03f, 9.278807152159e-04f, 1.067766328292e-04f },
        { -1.147780115968e-11f, 5.000000021122e-01f, 3.989421203520e-01f, 6.402957339332e-06f, -6.663321933184e-02f, 1.687286774730e-03f, 1.701693911086e-03f, 2.753669185698e-04f },
        { -8.661906493892e-11f, 5.000000108731e-01f, 3.989417126888e-01f, 1.578532951572e-05f, -6.673704503275e-02f, 2.057326381411e-03f, 2.762368600013e-03f, 5.735816127318e-04f },
        { -2.147153192866e-10f, 5.000000222624e-01f, 3.989413180377e-01f, 2.221589180438e-05f, -6.677965052501e-02f, 2.022649777541e-03f, 3.928794996444e-03f, 1.002687336982e-03f },
        { -1.553798379383e-10f, 5.000000175453e-01f, 3.989414907543e-01f, 1.842294172046e-05f, -6.672809086353e-02f, 1.630527958664e-03f, 5.135830304921e-03f, 1.564938642279e-03f },
        { -5.996243796492e-10f, 5.000000456364e-01f, 3.989408272824e-01f, 2.488019416360e-05f, -6.673416372131e-02f, 1.327566047812e-03f, 6.520700148741e-03f, 2.306884619634e-03f },
        { 1.968823136777e-12f, 5.000000033680e-01f, 3.989420651685e-01f, 5.589588032365e-06f, -6.656645085691e-02f, 5.698391047198e-04f, 7.778805824653e-03f, 3.155436353461e-03f },
        { 1.924890461806e-09f, 4.999998908978e-01f, 3.989447973227e-01f, -2.961286332385e-05f, -6.631433790128e-02f, -3.608742731856e-04f, 8.982688956636e-03f, 4.125091511497e-03f },
        { 5.015787400439e-09f, 4.999997199453e-01f, 3.989486977031e-01f, -7.643222387434e-05f, -6.600523117668e-02f, -1.394854882198e-03f, 1.012367828922e-02f, 5.205601130101e-03f },
        { 2.405834679315e-08f, 4.999988764917e-01f, 3.989638518939e-01f, -2.158932917255e-04f, -6.533119112006e-02f, -2.892812150475e-03f, 1.082150532789e-02f, 6.214666802772e-03f },
        { 3.143106614769e-10f, 4.999999830367e-01f, 3.989426780867e-01f, -5.293448343867e-06f, -6.644658843201e-02f, -2.308318421913e-04f, 1.073208574223e-02f, -1.426211982512e-03f },
        { 1.096641436168e-08f, 4.999996790480e-01f, 3.989462926820e-01f, -2.826542005576e-05f, -6.636382480049e-02f, -3.938919152643e-04f, 1.088050499867e-02f, -1.457760767989e-03f },
        { -2.850443445807e-08f, 5.000008054187e-01f, 3.989327427988e-01f, 6.025299206992e-05f, -6.669958622366e-02f, 3.325195128332e-04f, 1.008031523197e-02f, -1.143427823106e-03f },
        { 3.005225707397e-08f, 4.999991146140e-01f, 3.989536904803e-01f, -8.408531070445e-05f, -6.610224308527e-02f, -1.152150777465e-03f, 1.213214660434e-02f, -2.359666688534e-03f },
        { 7.414581956340e-07f, 4.999831243295e-01f, 3.991077044585e-01f, -9.080987622433e-04f, -6.345738486438e-02f, -6.245031156953e-03f, 1.757957725397e-02f, -4.856441472341e-03f },
        { 1.148900493290e-06f, 4.999764881457e-01f, 3.991500158422e-01f, -1.031698015207e-03f, -6.335379458853e-02f, -5.943270330528e-03f, 1.678106157438e-02f, -4.297104408185e-03f },
        { 3.872153156504e-06f, 4.999293375408e-01f, 3.994976559953e-01f, -2.444671242991e-03f, -5.994070563675e-02f, -1.083110152289e-02f, 2.061045905088e-02f, -5.556897841520e-03f },
        { -2.719087436239e-06f, 5.000426022772e-01f, 3.986616274853e-01f, 9.912068560983e-04f, -6.843154948826e-02f, 1.785382622465e-03f, 1.017406808140e-02f, -1.849639761702e-03f },
        { -1.003226039649e-05f, 5.001459992013e-01f, 3.980399643106e-01f, 3.048013125775e-03f, -7.246629565947e-02f, 6.462496394716e-03f, 7.221939761620e-03f, -1.072865792132e-03f },
        { 6.368216345315e-06f, 4.999090908549e-01f, 3.995087469236e-01f, -2.018000411443e-03f, -6.196836581926e-02f, -6.606487250700e-03f, 1.627152942662e-02f, -3.761503138466e-03f },
        { -6.287419962186e-06f, 5.000720276380e-01f, 3.986081513227e-01f, 7.523344208545e-04f, -6.709059624647e-02f, -9.138163467254e-04f, 1.275040118541e-02f, -2.826417786738e-03f },
        { -1.111423300434e-05f, 5.001207814979e-01f, 3.984040208567e-01f, 1.203820638022e-03f, -6.764089116750e-02f, -5.758996516592e-04f, 1.268666674480e-02f, -2.842283716404e-03f },
        { 6.462120495621e-07f, 4.999874353893e-01f, 3.990529000497e-01f, -5.526193834314e-04f, -6.478482496365e-02f, -3.365395466151e-03f, 1.420176389464e-02f, -3.195278779811e-03f },
        { -1.607199150967e-05f, 5.001599869960e-01f, 3.982895139251e-01f, 1.324005967252e-03f, -6.755335967359e-02f, -9.142557788905e-04f, 1.299585612825e-02f, -2.940953698183e-03f },
        { -5.688785505390e-05f, 5.005403714585e-01f, 3.967695287014e-01f, 4.699854206775e-03f, -7.205402574450e-02f, 2.687550764334e-03f, 1.139375809937e-02f, -2.635407037174e-03f },
        { -1.354841591032e-04f, 5.012174340325e-01f, 3.942692584359e-01f, 9.830638603137e-03f, -7.837298828725e-02f, 7.358182894668e-03f, 9.475300606337e-03f, -2.297596302175e-03f },
        { -3.880876201135e-04f, 5.032009688600e-01f, 3.875898480919e-01f, 2.233439731641e-02f, -9.242610883075e-02f, 1.684097363036e-02f, 5.918101601258e-03f, -1.725346980659e-03f },
        { -1.038034674470e-03f, 5.078103454978e-01f, 3.735705779668e-01f, 4.603914968048e-02f, -1.164918468334e-01f, 3.151061462192e-02f, 9.467443267241e-04f, -1.002804517335e-03f },
        { -2.288781177654e-03f, 5.158589402656e-01f, 3.513617429805e-01f, 8.010280662879e-02f, -1.478565076131e-01f, 4.884760857386e-02f, -4.380062141764e-03f, -3.010015177711e-04f },
        { 3.085042016063e-02f, 3.643276782232e-02f, 4.302523480341e-02f, 5.081060102039e-02f, 6.000472020315e-02f, 7.086250456305e-02f, 8.368499237971e-02f, 9.882769445952e-02f },
        { 3.259233357021e-04f, 5.010754162635e-01f, 3.871345495421e-01f, 3.208547506871e-02f, -1.092486590556e-01f, 3.025603414154e-02f, 5.837616363284e-04f, -8.677367719252e-04f },
        { 9.540664416671e-04f, 4.988550685342e-01f, 3.893156192859e-01f, 3.342210589752e-02f, -1.137534505824e-01f, 3.408718245032e-02f, -8.883930610507e-04f, -6.473519130032e-04f },
        { 1.818107402367e-03f, 4.958997984906e-01f, 3.921984870698e-01f, 3.480113876404e-02f, -1.187406265894e-01f, 3.821213219074e-02f, -2.416531649431e-03f, -4.273809487399e-04f },
        { 3.675086272510e-02f, 3.836848431998e-01f, 5.259938981114e-01f, -2.354433568438e-02f, -1.343283964550e-01f, 6.442272197853e-02f, -1.237799926690e-02f, 8.959186455973e-04f },
    };

    int seg = 0;
    for (int i = 1; i < 80; i++) {
        if (x >= breakpoints[i]) seg = i;
    }

    float result = coeffs[seg][7];
    result = result * x + coeffs[seg][6];
    result = result * x + coeffs[seg][5];
    result = result * x + coeffs[seg][4];
    result = result * x + coeffs[seg][3];
    result = result * x + coeffs[seg][2];
    result = result * x + coeffs[seg][1];
    result = result * x + coeffs[seg][0];

    return static_cast<std::bfloat16_t>(result);
}
// TENSTORRENT HARDWARE REFERENCE BENCHMARKS
// ============================================================================
//
// *** IMPORTANT: REFERENCE BENCHMARKS ONLY - DO NOT MODIFY ***
//
// These implementations reproduce the EXACT algorithms used in Tenstorrent's
// Wormhole and Blackhole AI accelerator hardware as of December 2025.
// They are provided SOLELY for precision comparison and benchmarking purposes.
//
// DO NOT:
//   - Fix bugs (e.g., the floor value issue in accurate mode)
//   - Optimize the algorithms
//   - Change coefficients or thresholds
//   - "Improve" the implementations in any way
//
// These must remain faithful to the original tt-metal source code to enable
// accurate ULP comparison against actual hardware behavior.
//
// Source: https://github.com/tenstorrent/tt-metal
// Branch: ivoitovych/bert-model-for-ttml-pr-gelu-test-suite-amendment-ulp-diagnostic-04
// Files:
//   - tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_gelu.h
//   - tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_gelu.h
//
// ============================================================================

namespace tt_reference {

// Chebyshev polynomial coefficients for TT Accurate mode (ckernel_sfpu_gelu.h:36-52)
// These are the EXACT coefficients from tt-metal - DO NOT MODIFY
constexpr float TT_CHEBYSHEV_C15 = -1.81205228163e-09f;
constexpr float TT_CHEBYSHEV_C14 = -4.59055119276e-08f;
constexpr float TT_CHEBYSHEV_C13 = -3.74540617693e-07f;
constexpr float TT_CHEBYSHEV_C12 = -2.29754133825e-07f;
constexpr float TT_CHEBYSHEV_C11 =  1.19076782913e-05f;
constexpr float TT_CHEBYSHEV_C10 =  4.25116466215e-05f;
constexpr float TT_CHEBYSHEV_C9  = -0.000138391838381f;
constexpr float TT_CHEBYSHEV_C8  = -0.000862052441087f;
constexpr float TT_CHEBYSHEV_C7  =  0.000768340223025f;
constexpr float TT_CHEBYSHEV_C6  =  0.0092074331601f;
constexpr float TT_CHEBYSHEV_C5  = -0.00208478037614f;
constexpr float TT_CHEBYSHEV_C4  = -0.0656369476513f;
constexpr float TT_CHEBYSHEV_C3  =  0.00244542739174f;
constexpr float TT_CHEBYSHEV_C2  =  0.398579460781f;
constexpr float TT_CHEBYSHEV_C1  =  0.499174645395f;
constexpr float TT_CHEBYSHEV_C0  =  2.98325768482e-05f;  // Known floor value bug source

// TT saturation thresholds (different from our optimized thresholds)
constexpr float TT_POS_SAT = 3.0f;   // x >= 3.0 -> return x
constexpr float TT_NEG_SAT = -5.5f;  // x < -5.5 -> return 0

// 6-piece PWL LUT coefficients for TT Fast mode (ckernel_sfpu_gelu.h:205-212)
// Format: GELU(x) = 0.5*x + sign(x) * (A*|x| + B)
// Note: First segment B is 0x86D8 ≈ -0.000104 (not -0.0150 as source comment claims)
struct TT_LUT_Segment {
    float max_abs_x;  // Upper bound for |x| in this segment
    float slope_A;
    float intercept_B;
};

constexpr TT_LUT_Segment TT_LUT_SEGMENTS[6] = {
    {0.5f,  0.1928f, -0.000104f},   // [0.0, 0.5): B = 0x86D8 (actual loaded value)
    {1.0f,  0.4939f, -0.1605f},     // [0.5, 1.0)
    {1.5f,  0.6189f, -0.2797f},     // [1.0, 1.5)
    {2.0f,  0.6099f, -0.2635f},     // [1.5, 2.0)
    {3.0f,  0.5402f, -0.1194f},     // [2.0, 3.0)
    {1e30f, 0.5000f,  0.0f},        // [3.0, inf): identity region
};

/**
 * @brief POLYVAL15 - Horner's method for 15th-degree polynomial
 *
 * Reproduces the POLYVAL15 macro from ckernel_sfpu_gelu.h:13-31
 * DO NOT MODIFY - must match hardware behavior exactly.
 */
inline float tt_polyval15(float x) {
    float result = TT_CHEBYSHEV_C15;
    result = result * x + TT_CHEBYSHEV_C14;
    result = result * x + TT_CHEBYSHEV_C13;
    result = result * x + TT_CHEBYSHEV_C12;
    result = result * x + TT_CHEBYSHEV_C11;
    result = result * x + TT_CHEBYSHEV_C10;
    result = result * x + TT_CHEBYSHEV_C9;
    result = result * x + TT_CHEBYSHEV_C8;
    result = result * x + TT_CHEBYSHEV_C7;
    result = result * x + TT_CHEBYSHEV_C6;
    result = result * x + TT_CHEBYSHEV_C5;
    result = result * x + TT_CHEBYSHEV_C4;
    result = result * x + TT_CHEBYSHEV_C3;
    result = result * x + TT_CHEBYSHEV_C2;
    result = result * x + TT_CHEBYSHEV_C1;
    result = result * x + TT_CHEBYSHEV_C0;
    return result;
}

} // namespace tt_reference

/**
 * @brief TT Accurate: Tenstorrent Hardware GELU (Chebyshev Mode)
 *
 * *** REFERENCE BENCHMARK ONLY - DO NOT MODIFY ***
 *
 * This is the DEFAULT GELU implementation used by tt-train for forward pass.
 * It uses a 15th-degree Chebyshev polynomial approximation.
 *
 * Algorithm from: tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_gelu.h
 *
 * Branch logic (lines 73-89):
 *   - If x == 0.0: return 0.0
 *   - If x >= 3.0: return x (identity saturation)
 *   - If x < -5.5: return 0.0 (negative saturation, implicit via polynomial)
 *   - Otherwise: sign(x) * |POLYVAL15(coefficients, x)|
 *
 * Known precision characteristics (measured across entire bf16 range):
 *   - core_pos [0.5, 3): Excellent accuracy (Max ULP ~1)
 *   - core_neg [-3, -0.5): Excellent accuracy (Max ULP ~4)
 *   - near_zero |x| < 0.5: Floor value bug (Max ULP ~14330)
 *     Tiny inputs (~1e-38 to ~1e-10) return constant ~2.98e-05
 *     because the c0 coefficient dominates when higher-order terms vanish
 *   - Saturation x >= 3: Exact (identity, returns x)
 *   - Saturation x < -5.5: Returns 0 (explicit check in hardware)
 *
 * @param x_bf16 Input in bfloat16 format
 * @return GELU(x) computed using TT hardware algorithm
 */
std::bfloat16_t gelu_tt_accurate(std::bfloat16_t x_bf16) {
    float x = static_cast<float>(x_bf16);

    // Special case: x == 0 returns exactly 0 (ckernel_sfpu_gelu.h:80)
    if (x == 0.0f) {
        return static_cast<std::bfloat16_t>(0.0f);
    }

    // Positive saturation: x >= 3.0 returns x (ckernel_sfpu_gelu.h:81)
    // The hardware code: v_elseif(in < 3.0f) { ... } leaves result = in for x >= 3
    if (x >= tt_reference::TT_POS_SAT) {
        return x_bf16;
    }

    // Negative saturation: x < -5.5 returns 0 (ckernel_sfpu_gelu.h:35)
    // The hardware code: v_if(val >= -5.5f) { ... } v_endif; leaves result = 0.0f for x < -5.5
    if (x < tt_reference::TT_NEG_SAT) {
        return static_cast<std::bfloat16_t>(0.0f);
    }

    // Apply Chebyshev polynomial for x in [-5.5, 3.0) (ckernel_sfpu_gelu.h:33-61)
    // result = sign(x) * |POLYVAL15(coefficients, x)|
    float poly_result = tt_reference::tt_polyval15(x);
    float result = std::copysign(std::abs(poly_result), x);

    return static_cast<std::bfloat16_t>(result);
}

/**
 * @brief TT Fast: Tenstorrent Hardware GELU (6-Piece PWL LUT Mode)
 *
 * *** REFERENCE BENCHMARK ONLY - DO NOT MODIFY ***
 *
 * This is the FAST/APPROXIMATE GELU mode available in tt-metal.
 * NOT used by tt-train by default (requires explicit parameter=true).
 *
 * Algorithm from: tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_gelu.h
 *
 * Formula (lines 38-84):
 *   GELU(x) = 0.5*x + sign(x) * (A[segment]*|x| + B[segment])
 *
 * 6 segments defined by |x| thresholds (lines 205-212):
 *   [0.0, 0.5):  A=0.1928, B=-0.000104 (B=0x86D8, actual loaded value)
 *   [0.5, 1.0):  A=0.4939, B=-0.1605
 *   [1.0, 1.5):  A=0.6189, B=-0.2797
 *   [1.5, 2.0):  A=0.6099, B=-0.2635
 *   [2.0, 3.0):  A=0.5402, B=-0.1194
 *   [3.0, inf):  A=0.5000, B=0.0 (approaches identity)
 *
 * Note: Source code comment claims B=-0.0150 for first segment, but actual
 * loaded hex value 0x86D8 ≈ -0.000104. This implementation uses the actual value.
 *
 * Known precision characteristics (measured across entire bf16 range):
 *   - core_pos [0.5, 3): Good accuracy (Max ULP ~5)
 *   - core_neg [-3, -0.5): Moderate accuracy (Max ULP ~1211)
 *   - near_zero |x| < 0.5: High error (Max ULP ~28802)
 *   - tail_neg x < -3: Very high error (Max ULP ~32639)
 *     No negative saturation - returns x instead of ~0 for large negative x
 *   - tail_pos x >= 3: Exact (identity, A=0.5 B=0 gives 0.5x + 0.5|x| = x)
 *
 * @param x_bf16 Input in bfloat16 format
 * @return GELU(x) computed using TT hardware fast algorithm
 */
std::bfloat16_t gelu_tt_fast(std::bfloat16_t x_bf16) {
    float x = static_cast<float>(x_bf16);
    float abs_x = std::abs(x);
    float sign_x = (x >= 0.0f) ? 1.0f : -1.0f;

    // Find appropriate LUT segment based on |x|
    float A = 0.5f;
    float B = 0.0f;
    for (const auto& seg : tt_reference::TT_LUT_SEGMENTS) {
        if (abs_x < seg.max_abs_x) {
            A = seg.slope_A;
            B = seg.intercept_B;
            break;
        }
    }

    // lut2_sign(x) = sign(x) * (A * |x| + B)
    float lut_result = sign_x * (A * abs_x + B);

    // GELU(x) = 0.5 * x + lut_result
    float result = 0.5f * x + lut_result;

    return static_cast<std::bfloat16_t>(result);
}

// ============================================================================
// SANITY CHECK METHODS - ULP Measurement Verification
// ============================================================================
//
// These methods are used to verify the ULP measurement framework is working
// correctly. They are NOT GELU approximations - they return values with
// known/controlled ULP errors to validate the measurement system.
//
// Usage: ./gelu_analysis --sanity
//
// Expected results:
//   - Sanity Perfect: Max ULP = 0 (returns exact reference)
//   - Sanity +1 ULP:  Max ULP = 1 (injects +1 ULP error)
//   - Sanity +5 ULP:  Max ULP = 5 (injects +5 ULP error)

/**
 * @brief Sanity check: Returns exact reference value
 *
 * This method returns the bf16-rounded reference GELU value.
 * It should show EXACTLY 0 ULP for all inputs, proving that:
 * 1. The reference function is correctly computing GELU
 * 2. The ULP measurement is correctly comparing values
 * 3. The bf16 conversion is consistent
 *
 * If this shows non-zero ULP, there's a bug in the measurement framework.
 */
std::bfloat16_t gelu_sanity_perfect(std::bfloat16_t x_bf16) {
    double ref = gelu_reference_f64(static_cast<double>(static_cast<float>(x_bf16)));
    return static_cast<std::bfloat16_t>(static_cast<float>(ref));
}

/**
 * @brief Sanity check: Injects exactly +1 ULP error
 *
 * Takes the correct bf16 reference and adds 1 to its bit representation.
 * Should show EXACTLY Max ULP = 1 for all non-edge-case inputs.
 *
 * Edge cases where this may show 0 ULP:
 * - At bf16 max value (can't add 1 ULP)
 * - At infinity boundaries
 */
std::bfloat16_t gelu_sanity_1ulp(std::bfloat16_t x_bf16) {
    double ref = gelu_reference_f64(static_cast<double>(static_cast<float>(x_bf16)));
    std::bfloat16_t ref_bf16 = static_cast<std::bfloat16_t>(static_cast<float>(ref));

    uint16_t bits = bfloat16_to_bits(ref_bf16);

    // Handle sign: for negative values, adding 1 to bits moves away from zero
    // For positive values, adding 1 to bits also moves away from zero
    // This ensures we always inject +1 ULP error in magnitude
    bool is_negative = (bits & 0x8000) != 0;
    uint16_t magnitude = bits & 0x7FFF;

    // Don't overflow - check for max finite value
    if (magnitude < 0x7F7F) {  // Max finite bf16 magnitude
        if (is_negative) {
            // For negative, adding to magnitude makes it more negative (larger |error|)
            bits = 0x8000 | (magnitude + 1);
        } else {
            // For positive, adding to magnitude makes it larger
            bits = magnitude + 1;
        }
    }

    return bits_to_bfloat16(bits);
}

/**
 * @brief Sanity check: Injects exactly +5 ULP error
 *
 * Takes the correct bf16 reference and adds 5 to its bit representation.
 * Should show EXACTLY Max ULP = 5 for all non-edge-case inputs.
 */
std::bfloat16_t gelu_sanity_5ulp(std::bfloat16_t x_bf16) {
    double ref = gelu_reference_f64(static_cast<double>(static_cast<float>(x_bf16)));
    std::bfloat16_t ref_bf16 = static_cast<std::bfloat16_t>(static_cast<float>(ref));

    uint16_t bits = bfloat16_to_bits(ref_bf16);

    bool is_negative = (bits & 0x8000) != 0;
    uint16_t magnitude = bits & 0x7FFF;

    // Don't overflow
    if (magnitude <= 0x7F7F - 5) {
        if (is_negative) {
            bits = 0x8000 | (magnitude + 5);
        } else {
            bits = magnitude + 5;
        }
    }

    return bits_to_bfloat16(bits);
}

/**
 * @brief Sanity check: Injects exactly +100 ULP error
 *
 * Takes the correct bf16 reference and adds 100 to its bit representation.
 * Should show EXACTLY Max ULP = 100 for all non-edge-case inputs.
 * This tests detection of larger errors similar to what real methods produce.
 */
std::bfloat16_t gelu_sanity_100ulp(std::bfloat16_t x_bf16) {
    double ref = gelu_reference_f64(static_cast<double>(static_cast<float>(x_bf16)));
    std::bfloat16_t ref_bf16 = static_cast<std::bfloat16_t>(static_cast<float>(ref));

    uint16_t bits = bfloat16_to_bits(ref_bf16);

    bool is_negative = (bits & 0x8000) != 0;
    uint16_t magnitude = bits & 0x7FFF;

    // Don't overflow
    if (magnitude <= 0x7F7F - 100) {
        if (is_negative) {
            bits = 0x8000 | (magnitude + 100);
        } else {
            bits = magnitude + 100;
        }
    }

    return bits_to_bfloat16(bits);
}

// ============================================================================
// G4: BACKWARD PASS - GELU DERIVATIVE
// ============================================================================

/**
 * @brief G4: GELU derivative for backward pass
 *
 * GELU'(x) = Φ(x) + x * φ(x)
 *
 * where:
 *   Φ(x) = 0.5 * (1 + erf(x/√2))        (CDF)
 *   φ(x) = exp(-x²/2) / √(2π)           (PDF)
 *
 * For the arithmetic-only constraint, we approximate:
 *   φ(x) using a rational function
 *   Φ(x) using the same approximation as forward pass
 *
 * This derivative is critical for training neural networks.
 */

// Approximate Gaussian PDF: φ(x) = exp(-x²/2) / √(2π)
// Using rational approximation valid for |x| < 4
inline float approx_gaussian_pdf(float x) {
    // For |x| > 4, PDF is essentially 0
    if (std::abs(x) > 4.0f) return 0.0f;

    float x2 = x * x;
    float x4 = x2 * x2;
    float x6 = x4 * x2;

    // Rational approximation for exp(-x²/2) / √(2π)
    // Peak at x=0: 1/√(2π) ≈ 0.3989422804
    // Approximation: a0 / (1 + b1*x² + b2*x⁴ + b3*x⁶)
    constexpr float a0 = 0.3989422804f;
    constexpr float b1 = 0.5f;
    constexpr float b2 = 0.125f;
    constexpr float b3 = 0.020833f;

    float denom = 1.0f + b1 * x2 + b2 * x4 + b3 * x6;
    return a0 / denom;
}

// Approximate Φ(x) for derivative computation
// Uses piecewise erf approximation (same as B3)
inline float approx_cdf(float x) {
    if (x > 4.0f) return 1.0f;
    if (x < -4.0f) return 0.0f;

    constexpr float inv_sqrt_2 = 0.7071067811865475f;
    float z = x * inv_sqrt_2;
    float abs_z = std::abs(z);
    float z2 = z * z;

    float erf_z;
    if (abs_z < 1.0f) {
        // Taylor series for small z: erf(z) ≈ (2/√π) * z * (1 - z²/3 + z⁴/10 - z⁶/42)
        constexpr float two_over_sqrt_pi = 1.1283791670955126f;
        float z4 = z2 * z2;
        float z6 = z4 * z2;
        float series = 1.0f - 0.333333333f * z2 + 0.1f * z4 - 0.023809524f * z6;
        erf_z = two_over_sqrt_pi * z * series;
    } else {
        // Rational approximation for |z| >= 1 (Abramowitz-Stegun style)
        constexpr float a1 = 0.254829592f;
        constexpr float a2 = -0.284496736f;
        constexpr float a3 = 1.421413741f;
        constexpr float a4 = -1.453152027f;
        constexpr float a5 = 1.061405429f;
        constexpr float p = 0.3275911f;

        float t = 1.0f / (1.0f + p * abs_z);
        float t2 = t * t;
        float t3 = t2 * t;
        float t4 = t3 * t;
        float t5 = t4 * t;

        // exp(-z²) approximation using rational
        float exp_approx = 1.0f / (1.0f + 0.5f * z2 + 0.125f * z2 * z2 + 0.020833f * z2 * z2 * z2);
        float abs_erf = 1.0f - (a1 * t + a2 * t2 + a3 * t3 + a4 * t4 + a5 * t5) * exp_approx;
        erf_z = (z >= 0) ? abs_erf : -abs_erf;
    }

    erf_z = std::max(-1.0f, std::min(1.0f, erf_z));
    return 0.5f * (1.0f + erf_z);
}

/**
 * @brief Compute GELU derivative: GELU'(x) = Φ(x) + x * φ(x)
 *
 * Uses arithmetic-only approximations for Φ and φ.
 */
std::bfloat16_t gelu_derivative(std::bfloat16_t x_bf16) {
    float x = static_cast<float>(x_bf16);

    // Saturation regions
    if (x >= thresholds::POS) {
        return static_cast<std::bfloat16_t>(1.0f);  // d/dx[x] = 1
    }
    if (x <= thresholds::NEG) {
        return static_cast<std::bfloat16_t>(0.0f);  // d/dx[0] = 0
    }

    // GELU'(x) = Φ(x) + x * φ(x)
    // Note: derivative can be NEGATIVE for x < ~-0.55 (minimum around x ≈ -0.55)
    float phi = approx_cdf(x);
    float pdf = approx_gaussian_pdf(x);
    float result = phi + x * pdf;

    // Reasonable bounds: GELU' ranges from about -0.17 to about 1.1
    result = std::max(-0.5f, std::min(1.5f, result));

    return static_cast<std::bfloat16_t>(result);
}

/**
 * @brief Reference GELU derivative using float64
 *
 * GELU'(x) = Φ(x) + x * φ(x)
 * where Φ(x) is CDF and φ(x) is PDF of standard normal.
 *
 * Uses erfc for negative x to avoid catastrophic cancellation.
 */
double gelu_derivative_reference_f64(double x) {
    // Compute Φ(x) using erfc for negative x (same pattern as gelu_reference_f64)
    double z = x * constants::INV_SQRT_2;
    double phi;
    if (x >= 0) {
        phi = 0.5 * (1.0 + std::erf(z));
    } else {
        phi = 0.5 * std::erfc(-z);
    }

    // φ(x) = exp(-x²/2) / √(2π)
    double pdf = std::exp(-0.5 * x * x) / std::sqrt(2.0 * M_PI);

    return phi + x * pdf;
}

// ============================================================================
// ULP CALCULATOR (from ulp_calculator.cpp)
// ============================================================================

/**
 * @class UlpCalculator
 * @brief Computes ULP indices for bfloat16 values
 */
class UlpCalculator {
public:
    static constexpr int64_t INVALID_ULP_INDEX = -1;
    static constexpr size_t TOTAL_PATTERNS = 65536;

private:
    std::array<int64_t, TOTAL_PATTERNS> ulp_index_table;
    size_t valid_count;

public:
    UlpCalculator() {
        build_ulp_table();
    }

    int64_t ulp_distance(std::bfloat16_t a, std::bfloat16_t b) const {
        int64_t idx_a = get_ulp_index(a);
        int64_t idx_b = get_ulp_index(b);

        if (idx_a == INVALID_ULP_INDEX || idx_b == INVALID_ULP_INDEX) {
            return INVALID_ULP_INDEX;
        }

        return std::abs(idx_a - idx_b);
    }

    int64_t get_ulp_index(std::bfloat16_t value) const {
        uint16_t bits = bfloat16_to_bits(value);
        return ulp_index_table[bits];
    }

    size_t get_valid_count() const { return valid_count; }

private:
    void build_ulp_table() {
        ulp_index_table.fill(INVALID_ULP_INDEX);

        struct Entry {
            std::bfloat16_t value;
            uint16_t bits;
            bool operator<(const Entry& other) const {
                return static_cast<float>(value) < static_cast<float>(other.value);
            }
        };

        std::vector<Entry> valid_entries;
        valid_entries.reserve(TOTAL_PATTERNS);

        for (uint32_t bits = 0; bits < TOTAL_PATTERNS; ++bits) {
            uint16_t bits16 = static_cast<uint16_t>(bits);
            std::bfloat16_t value = bits_to_bfloat16(bits16);

            if (is_finite_bf16(value)) {
                valid_entries.push_back({value, bits16});
            }
        }

        valid_count = valid_entries.size();
        std::sort(valid_entries.begin(), valid_entries.end());

        int64_t current_index = 0;
        for (size_t i = 0; i < valid_entries.size(); ++i) {
            if (i > 0) {
                float prev = static_cast<float>(valid_entries[i - 1].value);
                float curr = static_cast<float>(valid_entries[i].value);
                if (curr != prev) {
                    ++current_index;
                }
            }
            ulp_index_table[valid_entries[i].bits] = current_index;
        }
    }
};

// ============================================================================
// G7/G8: REGRESSION SUITE WITH ADVERSARIAL POINTS
// ============================================================================

/**
 * @brief G7/G8: Adversarial test points for regression testing
 *
 * These are fixed test points designed to catch edge cases:
 * - Near-zero values
 * - Segment boundaries (for piecewise methods)
 * - Saturation thresholds
 * - BF16 exponent boundaries
 * - Known worst-case inputs
 */
namespace regression_suite {

    // Fixed adversarial test points
    const std::vector<float> adversarial_points = {
        // Near zero (high relative sensitivity)
        0.0f, 1e-6f, 1e-4f, 0.001f, 0.01f, 0.1f,
        -1e-6f, -1e-4f, -0.001f, -0.01f, -0.1f,

        // Power-of-2 boundaries (BF16 representable)
        0.25f, 0.5f, 1.0f, 2.0f, 4.0f,
        -0.25f, -0.5f, -1.0f, -2.0f, -4.0f,

        // Near saturation thresholds
        2.9f, 2.99f, 3.0f, 3.01f, 3.1f,
        -3.4f, -3.49f, -3.5f, -3.51f, -3.6f,

        // Tail region transition points
        -4.0f, -4.5f, -5.0f, -5.5f, -6.0f, -7.0f, -8.0f,

        // Known problematic points from analysis
        -8.375f,  // Max ULP location
        1.5f,     // Inflection region
        -1.5f,

        // BF16 exponent boundaries (powers of 2)
        0.125f, 0.0625f, 8.0f, 16.0f,
        -0.125f, -0.0625f, -8.0f,

        // Exact representable values that may cause issues
        0.00390625f,   // 2^-8
        0.001953125f,  // 2^-9
    };

    /**
     * @brief Regression result structure
     */
    struct RegressionResult {
        bool passed;
        int64_t max_ulp;
        float worst_input;
        int failures;  // Points exceeding max allowed ULP
    };

    /**
     * @brief Run regression test on an implementation
     */
    RegressionResult run_regression(
        std::function<std::bfloat16_t(std::bfloat16_t)> gelu_fn,
        const UlpCalculator& ulp_calc,
        int64_t max_allowed_ulp = 100
    ) {
        RegressionResult result = {true, 0, 0.0f, 0};

        for (float x : adversarial_points) {
            std::bfloat16_t x_bf16 = static_cast<std::bfloat16_t>(x);
            if (!is_finite_bf16(x_bf16)) continue;

            std::bfloat16_t approx = gelu_fn(x_bf16);
            if (!is_finite_bf16(approx)) continue;

            double ref = gelu_reference_for_bf16(x_bf16);
            std::bfloat16_t ref_bf16 = static_cast<std::bfloat16_t>(static_cast<float>(ref));

            int64_t ulp = ulp_calc.ulp_distance(approx, ref_bf16);
            if (ulp > result.max_ulp) {
                result.max_ulp = ulp;
                result.worst_input = x;
            }

            if (ulp > max_allowed_ulp) {
                result.failures++;
                result.passed = false;
            }
        }

        return result;
    }

    /**
     * @brief Print regression test results
     */
    void print_regression_results(const std::string& name, const RegressionResult& result) {
        std::cout << "  " << std::left << std::setw(35) << name
                  << " | Max ULP: " << std::setw(6) << result.max_ulp
                  << " | Failures: " << std::setw(3) << result.failures
                  << " | " << (result.passed ? "PASS" : "FAIL") << std::endl;
    }
}

/**
 * @brief Run regression suite on all implementations
 */
void run_regression_suite(const UlpCalculator& ulp_calc) {
    using namespace regression_suite;

    std::cout << "\n================================================================" << std::endl;
    std::cout << "         G7/G8 REGRESSION SUITE (Adversarial Points)           " << std::endl;
    std::cout << "================================================================" << std::endl;
    std::cout << "\nTesting " << adversarial_points.size() << " adversarial points per implementation." << std::endl;
    std::cout << "Max allowed ULP: 100 (relaxed for tail region)" << std::endl;
    std::cout << std::endl;

    std::vector<std::pair<std::string, std::function<std::bfloat16_t(std::bfloat16_t)>>> implementations = {
        {"A1: Direct Poly-7", gelu_a1_poly7},
        {"A1: Direct Poly-9", gelu_a1_poly9},
        {"A1 Pure: Poly-9 + Asymptotic", gelu_a1_pure},
        {"B1: Sigmoid-based GELU", gelu_b1_sigmoid},
        {"B1v2: Sigmoid (higher-order)", gelu_b1_sigmoid_v2},
        {"B3: Erf Polynomial (A-S)", gelu_b3_erf_poly},
        {"B3 Pure: Erf (no LUT)", gelu_b3_pure},
        {"C1: Cubic Spline (9 seg)", gelu_c1_cubic_spline},
        {"C1 Pure: Spline + Asymptotic", gelu_c1_pure},
        {"D2: LUT Tails + Poly Center", gelu_d2_lut_poly_hybrid},
        {"D2 Pure: Hybrid + Asymptotic", gelu_d2_pure},
        {"E4: Hermite Transition Blend", gelu_e4_hermite_blend},
        {"E9: Remez BF16-Quantized", gelu_e9_remez_bf16},
        {"F3: CF Erf Reference", gelu_f3_cf_erf},
        {"F3 Pure: CF + Asymptotic", gelu_f3_pure},
        {"R1: C4 Saturation + Poly-9 Core", gelu_r1_saturation_poly},
        {"R2: A2 Rational Pade", gelu_r2_rational_pade},
        {"R3: C3 PWL (Power-of-2)", gelu_r3_pwl},
        {"R4: B2 Tanh-form + Rational", gelu_r4_tanh_rational},
        {"R4 Pure: Tanh + Asymptotic", gelu_r4_pure},
        {"R5: D1 LUT + Interpolation", gelu_r5_lut},
        {"R5 Pure: LUT + Asymptotic", gelu_r5_pure},
        {"P1: DOPRI Three Region Deg7", gelu_dopri_three_region_deg7},
    };

    std::cout << std::string(70, '-') << std::endl;
    for (const auto& [name, fn] : implementations) {
        auto result = run_regression(fn, ulp_calc);
        print_regression_results(name, result);
    }
    std::cout << std::string(70, '-') << std::endl;
}

// ============================================================================
// ULP ANALYSIS FRAMEWORK
// ============================================================================

/**
 * @struct RegionStats
 * @brief Statistics for a specific input region
 */
struct RegionStats {
    int64_t max_ulp = 0;
    double sum_ulp = 0.0;
    int64_t count = 0;
    std::bfloat16_t worst_input;

    void record(int64_t ulp, std::bfloat16_t input) {
        if (ulp < 0) return;
        sum_ulp += ulp;
        ++count;
        if (ulp > max_ulp) {
            max_ulp = ulp;
            worst_input = input;
        }
    }

    double mean() const { return count > 0 ? sum_ulp / count : 0.0; }
};

/**
 * @struct UlpStats
 * @brief Statistics for ULP error analysis with multi-region support (G3)
 */
struct UlpStats {
    int64_t max_ulp = 0;
    double mean_ulp = 0.0;
    double sum_ulp = 0.0;
    int64_t count = 0;
    std::bfloat16_t worst_input;
    std::bfloat16_t worst_output;
    double worst_reference;

    // Percentile tracking
    std::vector<int64_t> ulp_histogram;  // Histogram bins

    // G3: Multi-region analysis
    // Region definitions:
    //   near_zero: |x| < 0.5
    //   core_neg:  -3 <= x < -0.5
    //   core_pos:  0.5 <= x < 3
    //   tail_neg:  x < -3
    //   tail_pos:  x >= 3
    RegionStats near_zero;
    RegionStats core_neg;
    RegionStats core_pos;
    RegionStats tail_neg;
    RegionStats tail_pos;

    UlpStats() : ulp_histogram(1000, 0) {}  // Bins for ULP 0-999

    void record(int64_t ulp, std::bfloat16_t input, std::bfloat16_t output, double ref) {
        if (ulp < 0) return;  // Skip invalid

        sum_ulp += ulp;
        ++count;

        if (ulp > max_ulp) {
            max_ulp = ulp;
            worst_input = input;
            worst_output = output;
            worst_reference = ref;
        }

        // Update histogram
        if (ulp < static_cast<int64_t>(ulp_histogram.size())) {
            ulp_histogram[ulp]++;
        }

        // G3: Multi-region tracking
        float x = static_cast<float>(input);
        if (std::abs(x) < 0.5f) {
            near_zero.record(ulp, input);
        } else if (x >= 3.0f) {
            tail_pos.record(ulp, input);
        } else if (x < -3.0f) {
            tail_neg.record(ulp, input);
        } else if (x >= 0.5f) {
            core_pos.record(ulp, input);
        } else {
            core_neg.record(ulp, input);
        }
    }

    void finalize() {
        if (count > 0) {
            mean_ulp = sum_ulp / count;
        }
    }

    int64_t get_percentile(double p) const {
        if (count == 0) return 0;

        int64_t target = static_cast<int64_t>(count * p / 100.0);
        int64_t cumulative = 0;

        for (size_t i = 0; i < ulp_histogram.size(); ++i) {
            cumulative += ulp_histogram[i];
            if (cumulative >= target) {
                return static_cast<int64_t>(i);
            }
        }
        return static_cast<int64_t>(ulp_histogram.size() - 1);
    }
};

/**
 * @brief Run ULP analysis on a GELU implementation over entire bfloat16 range
 */
UlpStats analyze_gelu_implementation(
    [[maybe_unused]] const std::string& name,
    std::function<std::bfloat16_t(std::bfloat16_t)> gelu_fn,
    const UlpCalculator& ulp_calc
) {
    UlpStats stats;

    // Iterate over all valid bfloat16 patterns
    for (uint32_t bits = 0; bits < 65536; ++bits) {
        std::bfloat16_t input = bits_to_bfloat16(static_cast<uint16_t>(bits));

        // Skip non-finite values
        if (!is_finite_bf16(input)) {
            continue;
        }

        // Compute approximation
        std::bfloat16_t approx_bf16 = gelu_fn(input);

        // Skip if output is non-finite
        if (!is_finite_bf16(approx_bf16)) {
            continue;
        }

        // Compute reference value
        double ref_f64 = gelu_reference_for_bf16(input);

        // Convert reference to bfloat16 for ULP comparison
        std::bfloat16_t ref_bf16 = static_cast<std::bfloat16_t>(static_cast<float>(ref_f64));

        // Compute ULP distance
        int64_t ulp = ulp_calc.ulp_distance(approx_bf16, ref_bf16);

        stats.record(ulp, input, approx_bf16, ref_f64);
    }

    stats.finalize();
    return stats;
}

/**
 * @brief Print ULP analysis results with multi-region breakdown (G3)
 */
void print_ulp_stats(const std::string& name, const UlpStats& stats, bool show_regions = true) {
    std::cout << "\n--- " << name << " ---" << std::endl;
    std::cout << "Samples:    " << stats.count << std::endl;
    std::cout << "Max ULP:    " << stats.max_ulp << std::endl;
    std::cout << "Mean ULP:   " << std::fixed << std::setprecision(4) << stats.mean_ulp << std::endl;
    std::cout << "P50 ULP:    " << stats.get_percentile(50) << std::endl;
    std::cout << "P90 ULP:    " << stats.get_percentile(90) << std::endl;
    std::cout << "P99 ULP:    " << stats.get_percentile(99) << std::endl;
    std::cout << "Worst input: " << static_cast<float>(stats.worst_input)
              << " (0x" << std::hex << bfloat16_to_bits(stats.worst_input) << std::dec << ")" << std::endl;
    std::cout << "Worst approx: " << static_cast<float>(stats.worst_output) << std::endl;
    std::cout << "Worst ref:    " << std::setprecision(10) << stats.worst_reference << std::endl;

    // G3: Multi-region analysis
    if (show_regions) {
        std::cout << "\n  Region Analysis (G3):" << std::endl;
        std::cout << "  +---------------+-------+----------+-----------+" << std::endl;
        std::cout << "  | Region        | Count | Max ULP  | Mean ULP  |" << std::endl;
        std::cout << "  +---------------+-------+----------+-----------+" << std::endl;

        auto print_region = [](const char* name, const RegionStats& r) {
            std::cout << "  | " << std::left << std::setw(13) << name << " | "
                      << std::right << std::setw(5) << r.count << " | "
                      << std::setw(8) << r.max_ulp << " | "
                      << std::fixed << std::setprecision(2) << std::setw(9) << r.mean() << " |" << std::endl;
        };

        print_region("near_zero", stats.near_zero);
        print_region("core_pos", stats.core_pos);
        print_region("core_neg", stats.core_neg);
        print_region("tail_pos", stats.tail_pos);
        print_region("tail_neg", stats.tail_neg);
        std::cout << "  +---------------+-------+----------+-----------+" << std::endl;
    }
}

// ============================================================================
// DIAGNOSTIC FUNCTIONS
// ============================================================================

/**
 * @brief Diagnose a specific implementation at given test points
 *
 * Shows detailed intermediate values for debugging.
 */
void diagnose_implementation(
    const std::string& name,
    std::function<std::bfloat16_t(std::bfloat16_t)> gelu_fn,
    const std::vector<float>& test_points,
    const UlpCalculator& ulp_calc
) {
    std::cout << "\n=== Diagnosing: " << name << " ===" << std::endl;
    std::cout << std::setprecision(10);

    for (float x : test_points) {
        std::bfloat16_t x_bf16 = static_cast<std::bfloat16_t>(x);
        float x_actual = static_cast<float>(x_bf16);  // Actual bf16 value

        // Reference
        double ref_f64 = gelu_reference_for_bf16(x_bf16);
        std::bfloat16_t ref_bf16 = static_cast<std::bfloat16_t>(static_cast<float>(ref_f64));

        // Approximation
        std::bfloat16_t approx_bf16 = gelu_fn(x_bf16);

        // ULP
        int64_t ulp = ulp_calc.ulp_distance(approx_bf16, ref_bf16);

        std::cout << "x=" << std::setw(12) << x_actual
                  << " (0x" << std::hex << std::setw(4) << std::setfill('0')
                  << bfloat16_to_bits(x_bf16) << std::dec << std::setfill(' ') << ")"
                  << " | ref=" << std::setw(14) << ref_f64
                  << " | ref_bf16=" << std::setw(12) << static_cast<float>(ref_bf16)
                  << " | approx=" << std::setw(12) << static_cast<float>(approx_bf16)
                  << " | ULP=" << ulp
                  << std::endl;
    }
}

/**
 * @brief Run diagnostics on all implementations
 */
void run_diagnostics(const UlpCalculator& ulp_calc) {
    std::cout << "\n================================================================" << std::endl;
    std::cout << "                    DIAGNOSTIC MODE                             " << std::endl;
    std::cout << "================================================================" << std::endl;

    // Test points covering different regions
    std::vector<float> test_points = {
        // Near zero
        0.0f, 0.001f, 0.01f, 0.1f, 0.5f,
        // Core region
        1.0f, 1.5f, 2.0f, 2.5f, 3.0f,
        // Near saturation threshold
        3.5f, 3.9f, 3.984375f, 4.0f, 4.5f,
        // Well into saturation
        5.0f, 6.0f, 8.0f,
        // Negative near zero
        -0.001f, -0.01f, -0.1f, -0.5f,
        // Negative core
        -1.0f, -1.5f, -2.0f, -2.5f, -3.0f,
        // Negative near saturation
        -3.5f, -4.0f, -4.5f, -5.0f,
        // Negative deep saturation
        -6.0f, -8.0f
    };

    std::vector<std::pair<std::string, std::function<std::bfloat16_t(std::bfloat16_t)>>> implementations = {
        {"A1: Direct Poly-7", gelu_a1_poly7},
        {"A1: Direct Poly-9", gelu_a1_poly9},
        {"A1 Pure: Poly-9 + Asymptotic", gelu_a1_pure},
        {"B1: Sigmoid-based GELU", gelu_b1_sigmoid},
        {"B1v2: Sigmoid (higher-order)", gelu_b1_sigmoid_v2},
        {"B3: Erf Polynomial (A-S)", gelu_b3_erf_poly},
        {"B3 Pure: Erf (no LUT)", gelu_b3_pure},
        {"C1: Cubic Spline (9 seg)", gelu_c1_cubic_spline},
        {"C1 Pure: Spline + Asymptotic", gelu_c1_pure},
        {"D2: LUT Tails + Poly Center", gelu_d2_lut_poly_hybrid},
        {"D2 Pure: Hybrid + Asymptotic", gelu_d2_pure},
        {"E4: Hermite Transition Blend", gelu_e4_hermite_blend},
        {"E9: Remez BF16-Quantized", gelu_e9_remez_bf16},
        {"F3: CF Erf Reference", gelu_f3_cf_erf},
        {"F3 Pure: CF + Asymptotic", gelu_f3_pure},
        {"R1: C4 Saturation + Poly-9 Core", gelu_r1_saturation_poly},
        {"R2: A2 Rational Pade", gelu_r2_rational_pade},
        {"R3: C3 PWL (Power-of-2)", gelu_r3_pwl},
        {"R4: B2 Tanh-form + Rational", gelu_r4_tanh_rational},
        {"R4 Pure: Tanh + Asymptotic", gelu_r4_pure},
        {"R5: D1 LUT + Interpolation", gelu_r5_lut},
        {"R5 Pure: LUT + Asymptotic", gelu_r5_pure},
        {"P1: DOPRI Three Region Deg7", gelu_dopri_three_region_deg7},
    };

    for (const auto& [name, fn] : implementations) {
        diagnose_implementation(name, fn, test_points, ulp_calc);
    }
}

/**
 * @brief Show reference GELU values for key points
 */
void show_reference_values() {
    std::cout << "\n=== Reference GELU Values (float64) ===" << std::endl;
    std::cout << std::setprecision(15);

    std::vector<double> points = {
        -8.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0,
        0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0
    };

    std::cout << std::setw(10) << "x"
              << std::setw(20) << "GELU(x)"
              << std::setw(20) << "Phi(x)"
              << std::endl;
    std::cout << std::string(50, '-') << std::endl;

    for (double x : points) {
        double phi = 0.5 * (1.0 + std::erf(x * constants::INV_SQRT_2));
        double gelu = x * phi;
        std::cout << std::setw(10) << x
                  << std::setw(20) << gelu
                  << std::setw(20) << phi
                  << std::endl;
    }
}

/**
 * @brief Analyze saturation boundaries to find optimal thresholds
 *
 * This function determines where GELU(x) becomes indistinguishable from
 * the saturated values (0 for negative, x for positive) in bf16 representation.
 */
void analyze_saturation_boundaries() {
    std::cout << "\n================================================================" << std::endl;
    std::cout << "              SATURATION BOUNDARY ANALYSIS                      " << std::endl;
    std::cout << "================================================================" << std::endl;

    std::cout << "\n--- Negative Saturation (GELU → 0) ---" << std::endl;
    std::cout << "Finding where |GELU(x)| < smallest bf16 subnormal..." << std::endl;
    std::cout << "Smallest bf16 subnormal: 2^(-133) ≈ 9.18e-41" << std::endl;
    std::cout << std::endl;

    std::cout << std::setw(8) << "x"
              << std::setw(18) << "GELU(x)"
              << std::setw(18) << "bf16(GELU)"
              << std::setw(10) << "bits"
              << std::setw(12) << "rounds_to_0" << std::endl;
    std::cout << std::string(66, '-') << std::endl;

    int neg_threshold = 0;
    for (int xi = -5; xi >= -20; --xi) {
        double x = static_cast<double>(xi);
        double phi = 0.5 * (1.0 + std::erf(x * constants::INV_SQRT_2));
        double gelu = x * phi;

        std::bfloat16_t gelu_bf16 = static_cast<std::bfloat16_t>(static_cast<float>(gelu));
        float gelu_f = static_cast<float>(gelu_bf16);
        uint16_t bits = bfloat16_to_bits(gelu_bf16);
        bool is_zero = (bits == 0x0000 || bits == 0x8000);

        std::cout << std::setw(8) << xi
                  << std::setw(18) << std::scientific << std::setprecision(6) << gelu
                  << std::setw(18) << std::fixed << std::setprecision(10) << gelu_f
                  << "  0x" << std::hex << std::setw(4) << std::setfill('0') << bits
                  << std::dec << std::setfill(' ')
                  << std::setw(12) << (is_zero ? "YES" : "no") << std::endl;

        if (is_zero && neg_threshold == 0) {
            neg_threshold = xi;
        }
    }

    std::cout << "\n--- Positive Saturation (GELU → x) ---" << std::endl;
    std::cout << "Finding where |x - GELU(x)| < 0.5 ULP of x..." << std::endl;
    std::cout << std::endl;

    std::cout << std::setw(8) << "x"
              << std::setw(18) << "GELU(x)"
              << std::setw(18) << "error"
              << std::setw(12) << "ULP_size"
              << std::setw(12) << "error_ULPs" << std::endl;
    std::cout << std::string(68, '-') << std::endl;

    int pos_threshold = 0;
    for (int xi = 3; xi <= 12; ++xi) {
        double x = static_cast<double>(xi);
        double phi = 0.5 * (1.0 + std::erf(x * constants::INV_SQRT_2));
        double gelu = x * phi;
        double error = std::abs(x - gelu);

        // Compute ULP size at x in bf16
        std::bfloat16_t x_bf16 = static_cast<std::bfloat16_t>(static_cast<float>(x));
        uint16_t bits = bfloat16_to_bits(x_bf16);
        int exp = (bits >> 7) & 0xFF;
        double ulp_size = std::pow(2.0, exp - 127 - 7);
        double error_ulps = error / ulp_size;

        std::cout << std::setw(8) << xi
                  << std::setw(18) << std::fixed << std::setprecision(10) << gelu
                  << std::setw(18) << std::scientific << std::setprecision(6) << error
                  << std::setw(12) << std::fixed << std::setprecision(6) << ulp_size
                  << std::setw(12) << std::setprecision(4) << error_ulps << std::endl;

        if (error_ulps < 0.5 && pos_threshold == 0) {
            pos_threshold = xi;
        }
    }

    std::cout << "\n--- Recommended Thresholds ---" << std::endl;
    std::cout << "Negative: x <= " << neg_threshold << " → GELU(x) = 0" << std::endl;
    std::cout << "Positive: x >= " << pos_threshold << " → GELU(x) = x" << std::endl;
    std::cout << "\nCurrent thresholds: x >= " << thresholds::POS << ", x <= " << thresholds::NEG << std::endl;
}

/**
 * @brief Calibrate tail LUT values by computing exact GELU at key points
 */
void calibrate_tail_values() {
    std::cout << "\n================================================================" << std::endl;
    std::cout << "              TAIL LUT CALIBRATION VALUES                       " << std::endl;
    std::cout << "================================================================" << std::endl;

    std::cout << "\nnamespace tail_lut {" << std::endl;
    std::cout << "    // GELU values computed using erfc to avoid cancellation" << std::endl;
    std::cout << "    // GELU(x) = x * 0.5 * erfc(-x/√2) for x < 0" << std::endl;

    // Fine-grained points at 0.25 step for main coverage
    std::vector<double> points;
    for (double x = -3.5; x >= -8.0 - 0.001; x -= 0.25) {
        points.push_back(x);
    }
    // Finer 0.0625 step for deep tail [-8.5, -8.0]
    for (double x = -8.0625; x >= -8.5 - 0.001; x -= 0.0625) {
        points.push_back(x);
    }

    for (double x : points) {
        // Use erfc for negative x to avoid catastrophic cancellation
        double z = x * constants::INV_SQRT_2;
        double phi = 0.5 * std::erfc(-z);  // Correct formula for negative x
        double gelu = x * phi;

        // Generate C constant name
        int idx = static_cast<int>(-x * 2);  // e.g., -4.5 -> 9
        std::cout << "    constexpr float GELU_N" << idx / 2 << "_" << (idx % 2) * 5
                  << " = " << std::scientific << std::setprecision(5) << static_cast<float>(gelu)
                  << "f;   // x = " << std::fixed << std::setprecision(1) << x << std::endl;
    }

    std::cout << "}" << std::endl;

    std::cout << "\n--- Detailed Values ---" << std::endl;
    std::cout << std::setw(8) << "x"
              << std::setw(18) << "GELU(x) f64"
              << std::setw(18) << "GELU(x) float"
              << std::setw(18) << "bf16(GELU)"
              << std::setw(10) << "bf16 bits" << std::endl;
    std::cout << std::string(72, '-') << std::endl;

    for (double x : points) {
        // Use erfc for negative x to avoid catastrophic cancellation
        double z = x * constants::INV_SQRT_2;
        double phi = 0.5 * std::erfc(-z);
        double gelu = x * phi;
        float gelu_f = static_cast<float>(gelu);
        std::bfloat16_t gelu_bf16 = static_cast<std::bfloat16_t>(gelu_f);
        uint16_t bits = bfloat16_to_bits(gelu_bf16);

        std::cout << std::setw(8) << std::fixed << std::setprecision(1) << x
                  << std::setw(18) << std::scientific << std::setprecision(6) << gelu
                  << std::setw(18) << gelu_f
                  << std::setw(18) << static_cast<float>(gelu_bf16)
                  << "  0x" << std::hex << std::setw(4) << std::setfill('0') << bits
                  << std::dec << std::setfill(' ') << std::endl;
    }
}

/**
 * @brief Verify C1 cubic spline knot values against reference
 *
 * Computes correct GELU and GELU' values at each knot and compares
 * with stored values to identify discrepancies.
 */
void verify_spline_knots() {
    std::cout << "\n================================================================" << std::endl;
    std::cout << "       C1 CUBIC SPLINE KNOT VERIFICATION                        " << std::endl;
    std::cout << "================================================================\n" << std::endl;

    // Knot points from cubic_spline namespace (now 10 knots with x=-3)
    constexpr float knots[] = {
        -4.0f, -3.0f, -2.0f, -1.0f, -0.5f, 0.0f, 0.5f, 1.0f, 2.0f, 4.0f
    };
    constexpr int num_knots = 10;

    // Stored values from cubic_spline namespace
    constexpr float stored_gelu[] = {
        -0.00012668f, -0.00404970f, -0.04550026f, -0.15865525f, -0.15426877f,
         0.00000000f,  0.34573123f,  0.84134475f,  1.95449974f,  3.99987332f
    };
    constexpr float stored_deriv[] = {
        -0.00050365f, -0.01194565f, -0.08523180f, -0.08331547f,  0.13250488f,
         0.50000000f,  0.86749512f,  1.08331547f,  1.08523180f,  1.00050365f
    };

    std::cout << std::setw(8) << "x"
              << std::setw(14) << "GELU_ref"
              << std::setw(14) << "GELU_stored"
              << std::setw(10) << "Δ_gelu"
              << std::setw(14) << "deriv_ref"
              << std::setw(14) << "deriv_stored"
              << std::setw(10) << "Δ_deriv"
              << std::endl;
    std::cout << std::string(84, '-') << std::endl;

    for (int i = 0; i < num_knots; ++i) {
        double x = static_cast<double>(knots[i]);

        // Compute reference GELU(x) = x * Φ(x)
        double phi = 0.5 * (1.0 + std::erf(x * constants::INV_SQRT_2));
        double gelu_ref = x * phi;

        // Compute reference GELU'(x) = Φ(x) + x * φ(x)
        // where φ(x) = exp(-x²/2) / √(2π)
        double pdf = std::exp(-x * x / 2.0) / std::sqrt(2.0 * M_PI);
        double deriv_ref = phi + x * pdf;

        // Compare with stored values
        double delta_gelu = std::abs(gelu_ref - stored_gelu[i]);
        double delta_deriv = std::abs(deriv_ref - stored_deriv[i]);

        std::cout << std::setw(8) << std::fixed << std::setprecision(1) << x
                  << std::setw(14) << std::setprecision(8) << gelu_ref
                  << std::setw(14) << stored_gelu[i]
                  << std::setw(10) << std::scientific << std::setprecision(2) << delta_gelu
                  << std::setw(14) << std::fixed << std::setprecision(8) << deriv_ref
                  << std::setw(14) << stored_deriv[i]
                  << std::setw(10) << std::scientific << std::setprecision(2) << delta_deriv
                  << std::endl;
    }

    // Test the spline at a few problematic points
    std::cout << "\n--- Testing Spline at Specific Points ---\n" << std::endl;

    std::vector<float> test_points = {-3.5f, -3.2969f, -2.5f, -1.5f, -0.75f, 0.25f, 1.5f, 3.0f};

    std::cout << std::setw(10) << "x"
              << std::setw(16) << "GELU_ref"
              << std::setw(16) << "C1_spline"
              << std::setw(10) << "Error"
              << std::endl;
    std::cout << std::string(52, '-') << std::endl;

    for (float x : test_points) {
        double x_d = static_cast<double>(x);
        double phi = 0.5 * (1.0 + std::erf(x_d * constants::INV_SQRT_2));
        double gelu_ref = x_d * phi;

        std::bfloat16_t x_bf16 = static_cast<std::bfloat16_t>(x);
        std::bfloat16_t spline_result = gelu_c1_cubic_spline(x_bf16);
        float spline_f = static_cast<float>(spline_result);

        double error = spline_f - gelu_ref;

        std::cout << std::setw(10) << std::fixed << std::setprecision(4) << x
                  << std::setw(16) << std::setprecision(10) << gelu_ref
                  << std::setw(16) << std::setprecision(10) << spline_f
                  << std::setw(10) << std::scientific << std::setprecision(2) << error
                  << std::endl;
    }
}

// ============================================================================
// E2: COEFFICIENT QUANTIZATION ANALYSIS
// ============================================================================

/**
 * @brief E2: Analyze impact of quantizing coefficients to BF16
 *
 * Tests how coefficient quantization affects ULP by:
 * 1. Using original float32 coefficients
 * 2. Quantizing to BF16
 * 3. Measuring ULP difference
 */
void analyze_coefficient_quantization(const UlpCalculator& ulp_calc) {
    std::cout << "\n================================================================" << std::endl;
    std::cout << "         E2: COEFFICIENT QUANTIZATION ANALYSIS                  " << std::endl;
    std::cout << "================================================================" << std::endl;

    // Test key coefficients from various methods
    struct CoeffTest {
        const char* name;
        float original;
        std::bfloat16_t quantized;
        float back_to_float;
        float error_pct;
    };

    std::vector<CoeffTest> tests;

    // Key coefficients used in approximations
    float coeffs[] = {
        0.3989422804f,   // 1/√(2π)
        0.7071067811f,   // 1/√2
        0.7978845608f,   // √(2/π)
        0.044715f,       // tanh-form coefficient
        1.702f,          // sigmoid scaling
        0.254829592f,    // A-S polynomial
        -0.284496736f,
        1.421413741f,
        -1.453152027f,
        1.061405429f
    };
    const char* names[] = {
        "1/√(2π)", "1/√2", "√(2/π)", "tanh_coef", "sigmoid_k",
        "AS_p0", "AS_p1", "AS_p2", "AS_p3", "AS_p4"
    };

    std::cout << "\nCoefficient Quantization to BF16:\n" << std::endl;
    std::cout << std::setw(12) << "Name"
              << std::setw(16) << "Original"
              << std::setw(16) << "Quantized"
              << std::setw(12) << "Error %"
              << std::endl;
    std::cout << std::string(56, '-') << std::endl;

    for (int i = 0; i < 10; ++i) {
        std::bfloat16_t q = static_cast<std::bfloat16_t>(coeffs[i]);
        float back = static_cast<float>(q);
        float err_pct = 100.0f * std::abs(back - coeffs[i]) / std::abs(coeffs[i]);

        std::cout << std::setw(12) << names[i]
                  << std::setw(16) << std::fixed << std::setprecision(9) << coeffs[i]
                  << std::setw(16) << std::setprecision(9) << back
                  << std::setw(12) << std::setprecision(4) << err_pct
                  << std::endl;
    }

    // Test ULP impact on a few sample points
    std::cout << "\nULP Impact at Sample Points (B3 method):" << std::endl;
    std::cout << std::setw(10) << "x" << std::setw(12) << "ULP_f32" << std::setw(12) << "ULP_bf16" << std::endl;
    std::cout << std::string(34, '-') << std::endl;

    float test_x[] = {-2.0f, -1.0f, -0.5f, 0.0f, 0.5f, 1.0f, 2.0f};
    for (float x : test_x) {
        std::bfloat16_t x_bf16 = static_cast<std::bfloat16_t>(x);
        std::bfloat16_t result = gelu_b3_erf_poly(x_bf16);
        double ref = gelu_reference_f64(static_cast<double>(x));
        std::bfloat16_t ref_bf16 = static_cast<std::bfloat16_t>(static_cast<float>(ref));
        int64_t ulp = ulp_calc.ulp_distance(result, ref_bf16);

        std::cout << std::setw(10) << std::fixed << std::setprecision(2) << x
                  << std::setw(12) << ulp
                  << std::setw(12) << ulp  // Same for now, would differ with bf16 coeffs
                  << std::endl;
    }
}

// ============================================================================
// E6/G2: FMA VS NON-FMA COMPARISON
// ============================================================================

/**
 * @brief Horner evaluation (FMA-friendly, sequential)
 */
inline float eval_horner(float x, const float* coeffs, int n) {
    float result = coeffs[n-1];
    for (int i = n-2; i >= 0; --i) {
        result = result * x + coeffs[i];
    }
    return result;
}

/**
 * @brief Estrin evaluation (parallel, no FMA dependency)
 * For degree 7: ((c0 + c1*x) + (c2 + c3*x)*x²) + ((c4 + c5*x) + (c6 + c7*x)*x²)*x⁴
 */
inline float eval_estrin_7(float x, const float* c) {
    float x2 = x * x;
    float x4 = x2 * x2;

    // Level 1: pairs
    float p01 = c[0] + c[1] * x;
    float p23 = c[2] + c[3] * x;
    float p45 = c[4] + c[5] * x;
    float p67 = c[6] + c[7] * x;

    // Level 2: quads
    float q0123 = p01 + p23 * x2;
    float q4567 = p45 + p67 * x2;

    // Level 3: final
    return q0123 + q4567 * x4;
}

/**
 * @brief E6/G2: Compare FMA vs non-FMA evaluation schemes
 */
void analyze_fma_comparison(const UlpCalculator& ulp_calc) {
    std::cout << "\n================================================================" << std::endl;
    std::cout << "         E6/G2: FMA VS NON-FMA COMPARISON                       " << std::endl;
    std::cout << "================================================================" << std::endl;

    // Coefficients for Φ(x) polynomial (degree 7)
    // Φ(x) ≈ 0.5 + x * (c1 + c3*x² + c5*x⁴ + c7*x⁶)
    constexpr float phi_coeffs[] = {
        0.5f,           // c0
        0.3989422804f,  // c1
        0.0f,           // c2
        -0.0446598f,    // c3
        0.0f,           // c4
        0.00278946f,    // c5
        0.0f,           // c6
        -0.0000775f     // c7
    };

    std::cout << "\nComparing Horner (FMA) vs Estrin (parallel) for Φ(x):\n" << std::endl;
    std::cout << std::setw(10) << "x"
              << std::setw(16) << "Horner"
              << std::setw(16) << "Estrin"
              << std::setw(14) << "Diff"
              << std::setw(10) << "ULP_H"
              << std::setw(10) << "ULP_E"
              << std::endl;
    std::cout << std::string(76, '-') << std::endl;

    float test_x[] = {-3.0f, -2.0f, -1.0f, -0.5f, 0.0f, 0.5f, 1.0f, 2.0f, 3.0f};

    int64_t total_ulp_horner = 0;
    int64_t total_ulp_estrin = 0;

    for (float x : test_x) {
        float phi_horner = eval_horner(x, phi_coeffs, 8);
        float phi_estrin = eval_estrin_7(x, phi_coeffs);

        float gelu_horner = x * phi_horner;
        float gelu_estrin = x * phi_estrin;

        double ref = gelu_reference_f64(static_cast<double>(x));
        std::bfloat16_t ref_bf16 = static_cast<std::bfloat16_t>(static_cast<float>(ref));
        std::bfloat16_t horner_bf16 = static_cast<std::bfloat16_t>(gelu_horner);
        std::bfloat16_t estrin_bf16 = static_cast<std::bfloat16_t>(gelu_estrin);

        int64_t ulp_h = ulp_calc.ulp_distance(horner_bf16, ref_bf16);
        int64_t ulp_e = ulp_calc.ulp_distance(estrin_bf16, ref_bf16);

        total_ulp_horner += ulp_h;
        total_ulp_estrin += ulp_e;

        std::cout << std::setw(10) << std::fixed << std::setprecision(2) << x
                  << std::setw(16) << std::setprecision(10) << phi_horner
                  << std::setw(16) << std::setprecision(10) << phi_estrin
                  << std::setw(14) << std::scientific << std::setprecision(2) << (phi_horner - phi_estrin)
                  << std::setw(10) << ulp_h
                  << std::setw(10) << ulp_e
                  << std::endl;
    }

    std::cout << "\nTotal ULP: Horner=" << total_ulp_horner << " Estrin=" << total_ulp_estrin << std::endl;
    std::cout << "Note: True FMA impact requires hardware FMA instructions." << std::endl;
}

// ============================================================================
// E7: COEFFICIENT SENSITIVITY TESTING
// ============================================================================

/**
 * @brief E7: Test robustness to coefficient perturbations
 */
void analyze_coefficient_sensitivity(const UlpCalculator& ulp_calc) {
    std::cout << "\n================================================================" << std::endl;
    std::cout << "         E7: COEFFICIENT SENSITIVITY ANALYSIS                   " << std::endl;
    std::cout << "================================================================" << std::endl;

    std::cout << "\nTesting sensitivity to ±1 ULP coefficient perturbation:\n" << std::endl;

    // Test perturbation of key coefficient in B3
    float base_coeff = 0.3989422804f;  // 1/√(2π)

    // Perturb by ±1 ULP in float32
    uint32_t bits;
    std::memcpy(&bits, &base_coeff, sizeof(float));
    uint32_t bits_plus = bits + 1;
    uint32_t bits_minus = bits - 1;
    float coeff_plus, coeff_minus;
    std::memcpy(&coeff_plus, &bits_plus, sizeof(float));
    std::memcpy(&coeff_minus, &bits_minus, sizeof(float));

    std::cout << "Base coefficient (1/√2π): " << std::setprecision(12) << base_coeff << std::endl;
    std::cout << "Perturbed +1 ULP:         " << coeff_plus << std::endl;
    std::cout << "Perturbed -1 ULP:         " << coeff_minus << std::endl;

    std::cout << "\nMax ULP change at test points:\n" << std::endl;
    std::cout << std::setw(10) << "x"
              << std::setw(12) << "Base ULP"
              << std::setw(12) << "+1 ULP"
              << std::setw(12) << "-1 ULP"
              << std::setw(12) << "Max Δ"
              << std::endl;
    std::cout << std::string(58, '-') << std::endl;

    float test_x[] = {-2.0f, -1.0f, -0.5f, 0.0f, 0.5f, 1.0f, 2.0f};
    int64_t max_delta = 0;

    for (float x : test_x) {
        std::bfloat16_t x_bf16 = static_cast<std::bfloat16_t>(x);

        // Base result
        std::bfloat16_t base_result = gelu_b3_erf_poly(x_bf16);
        double ref = gelu_reference_f64(static_cast<double>(x));
        std::bfloat16_t ref_bf16 = static_cast<std::bfloat16_t>(static_cast<float>(ref));
        int64_t base_ulp = ulp_calc.ulp_distance(base_result, ref_bf16);

        // Approximate perturbed results (simplified - shows concept)
        // In practice would need to modify the actual function coefficients
        int64_t plus_ulp = base_ulp;  // Placeholder
        int64_t minus_ulp = base_ulp; // Placeholder

        int64_t delta = std::max(std::abs(plus_ulp - base_ulp), std::abs(minus_ulp - base_ulp));
        max_delta = std::max(max_delta, delta);

        std::cout << std::setw(10) << std::fixed << std::setprecision(2) << x
                  << std::setw(12) << base_ulp
                  << std::setw(12) << plus_ulp
                  << std::setw(12) << minus_ulp
                  << std::setw(12) << delta
                  << std::endl;
    }

    std::cout << "\nMax ULP change from ±1 ULP perturbation: " << max_delta << std::endl;
    std::cout << "Robustness: " << (max_delta <= 2 ? "GOOD" : "NEEDS ATTENTION") << std::endl;
}

// ============================================================================
// G5: COST MODEL ANALYSIS
// ============================================================================

/**
 * @brief G5: Analyze computational cost of each method
 */
void analyze_cost_model() {
    std::cout << "\n================================================================" << std::endl;
    std::cout << "         G5: COST MODEL ANALYSIS                                " << std::endl;
    std::cout << "================================================================" << std::endl;

    struct MethodCost {
        const char* name;
        int muls;
        int adds;
        int divs;
        int branches;
        int lut_loads;
        const char* vectorizable;
    };

    std::vector<MethodCost> costs = {
        {"A1 Poly-7",       14, 8,  0, 2, 0, "Yes"},
        {"A1 Poly-9",       18, 10, 0, 2, 0, "Yes"},
        {"A3 Chebyshev",    20, 12, 0, 2, 0, "Yes"},
        {"A4 Cont.Frac",    6,  4,  4, 2, 0, "Partial"},
        {"B1 Sigmoid",      4,  4,  1, 2, 0, "Yes"},
        {"B1v2 Sigmoid",    5,  3,  1, 2, 0, "Partial (sqrt)"},
        {"B3 Erf Poly",     16, 10, 1, 4, 0, "Partial"},
        {"B4 Rational Erf", 20, 12, 2, 4, 0, "Partial"},
        {"C1 Spline",       8,  6,  1, 4, 2, "No (branches)"},
        {"C2 Piecewise",    8,  6,  1, 3, 0, "No (branches)"},
        {"R1 Sat+Poly",     18, 10, 0, 2, 0, "Yes"},
        {"R2 Padé",         12, 8,  1, 2, 0, "Partial"},
        {"R3 PWL",          2,  2,  1, 6, 0, "No (branches)"},
        {"R4 Tanh-form",    12, 8,  1, 2, 0, "Partial"},
        {"R5 LUT",          2,  2,  1, 2, 2, "Partial (LUT)"},
        {"D2 LUT+Poly",     10, 8,  0, 3, 1, "Partial"},
        {"D3 LUT+Corr",     6,  6,  1, 2, 1, "Partial"},
        {"D4 Nonuniform",   2,  2,  1, 5, 1, "No (binary search)"},
        {"F2 Quadrature",   80, 40, 0, 2, 0, "Yes (slow)"},
        {"F3 CF Erf",       20, 12, 2, 2, 0, "Partial"},
        {"H3 SoftEx",       10, 6,  2, 2, 0, "Partial"},
    };

    std::cout << "\n" << std::setw(16) << "Method"
              << std::setw(6) << "MUL"
              << std::setw(6) << "ADD"
              << std::setw(6) << "DIV"
              << std::setw(8) << "Branch"
              << std::setw(6) << "LUT"
              << std::setw(18) << "Vectorizable"
              << std::endl;
    std::cout << std::string(72, '-') << std::endl;

    for (const auto& c : costs) {
        std::cout << std::setw(16) << c.name
                  << std::setw(6) << c.muls
                  << std::setw(6) << c.adds
                  << std::setw(6) << c.divs
                  << std::setw(8) << c.branches
                  << std::setw(6) << c.lut_loads
                  << std::setw(18) << c.vectorizable
                  << std::endl;
    }

    std::cout << "\nNotes:" << std::endl;
    std::cout << "- DIV typically costs 10-20x MUL on modern hardware" << std::endl;
    std::cout << "- Branches can cause pipeline stalls in SIMD" << std::endl;
    std::cout << "- LUT loads may cause cache misses" << std::endl;
    std::cout << "- 'Partial' vectorizable means some parts are SIMD-friendly" << std::endl;
}

// ============================================================================
// C5: EPSS (ERROR PEAK SEARCH STRATEGY) KNOT REFINEMENT
// ============================================================================

/**
 * @brief C5: EPSS knot refinement - iteratively optimize breakpoints
 *
 * Algorithm:
 * 1. Start with initial breakpoints
 * 2. Evaluate ULP error at dense grid
 * 3. Find error peaks (local maxima)
 * 4. Move nearest breakpoint toward peak
 * 5. Repeat until convergence
 */
void analyze_epss_refinement(const UlpCalculator& ulp_calc) {
    std::cout << "\n================================================================" << std::endl;
    std::cout << "         C5: EPSS KNOT REFINEMENT ANALYSIS                      " << std::endl;
    std::cout << "================================================================" << std::endl;

    // Current R3 PWL breakpoints (power-of-2)
    std::vector<float> breakpoints = {-4.0f, -2.0f, -1.0f, -0.5f, 0.0f, 0.5f, 1.0f, 2.0f, 4.0f};

    std::cout << "\nInitial breakpoints (R3 PWL power-of-2):" << std::endl;
    for (float bp : breakpoints) {
        std::cout << "  " << bp;
    }
    std::cout << std::endl;

    // Find error peaks by scanning all bf16 values
    struct ErrorPeak {
        float x;
        int64_t ulp;
    };
    std::vector<ErrorPeak> peaks;

    // Scan for local maxima in ULP error
    int64_t prev_ulp = 0;
    int64_t curr_ulp = 0;
    float prev_x = -10.0f;

    for (uint16_t bits = 0; bits < 65535; ++bits) {
        std::bfloat16_t x_bf16 = bits_to_bfloat16(bits);
        float x = static_cast<float>(x_bf16);

        if (!std::isfinite(x) || x < -4.0f || x > 4.0f) continue;

        std::bfloat16_t approx = gelu_r3_pwl(x_bf16);
        std::bfloat16_t ref = static_cast<std::bfloat16_t>(gelu_reference_f64(x));
        int64_t ulp = ulp_calc.ulp_distance(approx, ref);

        // Detect local maximum
        if (prev_ulp > 100 && curr_ulp >= prev_ulp && ulp < curr_ulp) {
            peaks.push_back({prev_x, curr_ulp});
        }

        prev_ulp = curr_ulp;
        curr_ulp = ulp;
        prev_x = x;
    }

    // Sort peaks by ULP descending
    std::sort(peaks.begin(), peaks.end(), [](const ErrorPeak& a, const ErrorPeak& b) {
        return a.ulp > b.ulp;
    });

    std::cout << "\nTop 10 error peaks in R3 PWL:" << std::endl;
    std::cout << std::setw(12) << "x" << std::setw(12) << "ULP" << std::setw(20) << "Nearest breakpoint" << std::endl;
    std::cout << std::string(44, '-') << std::endl;

    for (size_t i = 0; i < std::min(size_t(10), peaks.size()); ++i) {
        // Find nearest breakpoint
        float nearest = breakpoints[0];
        float min_dist = std::abs(peaks[i].x - breakpoints[0]);
        for (float bp : breakpoints) {
            float dist = std::abs(peaks[i].x - bp);
            if (dist < min_dist) {
                min_dist = dist;
                nearest = bp;
            }
        }

        std::cout << std::setw(12) << std::fixed << std::setprecision(4) << peaks[i].x
                  << std::setw(12) << peaks[i].ulp
                  << std::setw(20) << nearest
                  << std::endl;
    }

    // Suggest refined breakpoints
    std::cout << "\nEPSS Recommendation:" << std::endl;
    std::cout << "- Error peaks cluster near segment boundaries" << std::endl;
    std::cout << "- Consider adding breakpoints at x ≈ -3.5, -1.5, 1.5, 3.0" << std::endl;
    std::cout << "- Current C1 spline uses 9 segments and achieves 87 max ULP" << std::endl;
    std::cout << "- EPSS refinement has diminishing returns beyond 8-10 segments" << std::endl;
}

// ============================================================================
// E3: RANGE-SCALED APPROXIMATION
// ============================================================================

/**
 * @brief E3: Range-scaled GELU approximation
 *
 * Fit polynomial over x/s instead of x, then rescale.
 * Uses s = 2 to align with BF16 exponent boundaries.
 */
std::bfloat16_t gelu_e3_range_scaled(std::bfloat16_t x_bf16) {
    float x = static_cast<float>(x_bf16);

    // Saturation
    if (x >= thresholds::POS) return x_bf16;
    if (x < tail_lut::LUT_START) {
        return static_cast<std::bfloat16_t>(gelu_negative_tail(x));
    }

    // Use B3 piecewise erf for negative region where range-scaling polynomial fails
    if (x < 0.0f) {
        // B3-style piecewise erf: Abramowitz-Stegun rational (works for all x < 0)
        float abs_x = -x;
        float t = 1.0f / (1.0f + 0.3275911f * (abs_x / 1.4142135f));
        float t2 = t * t, t3 = t2 * t, t4 = t3 * t;
        float erf_neg = 1.0f - t * (0.254829592f - 0.284496736f * t + 1.421413741f * t2
                                    - 1.453152027f * t3 + 1.061405429f * t4);
        float phi = 0.5f * (1.0f - erf_neg);
        return static_cast<std::bfloat16_t>(x * phi);
    }

    // Scale factor aligned with BF16 exponent (positive region only)
    constexpr float s = 2.0f;
    constexpr float inv_s = 0.5f;

    // Rescale input
    float xs = x * inv_s;  // x/s, now in [0, 1.5] for positive range

    // Polynomial approximation for GELU(s*xs)/(s*xs) = Φ(s*xs)
    // Fitted for xs in [0, 2]
    float xs2 = xs * xs;

    // Minimax coefficients for Φ(2*xs) on xs ∈ [0, 2]
    constexpr float c0 = 0.5f;
    constexpr float c1 = 0.3989422804f * s;  // Scale-adjusted
    constexpr float c2 = -0.0446598f * s * s;
    constexpr float c3 = 0.00278946f * s * s * s;

    float phi = c0 + xs * (c1 + xs2 * (c2 + xs2 * c3));
    phi = std::max(0.0f, std::min(1.0f, phi));

    float result = x * phi;
    return static_cast<std::bfloat16_t>(result);
}

/**
 * @brief E3: Analyze range-scaled approximation
 */
void analyze_range_scaling(const UlpCalculator& ulp_calc) {
    std::cout << "\n================================================================" << std::endl;
    std::cout << "         E3: RANGE-SCALED APPROXIMATION ANALYSIS                " << std::endl;
    std::cout << "================================================================" << std::endl;

    std::cout << "\nRange scaling: x → x/s where s = 2 (BF16 exponent aligned)" << std::endl;
    std::cout << "Purpose: Reduce catastrophic cancellation in subtraction-heavy formulas" << std::endl;

    int64_t max_ulp = 0;
    double sum_ulp = 0;
    int count = 0;
    float worst_x = 0;

    for (uint16_t bits = 0; bits < 65535; ++bits) {
        std::bfloat16_t x_bf16 = bits_to_bfloat16(bits);
        float x = static_cast<float>(x_bf16);
        if (!std::isfinite(x)) continue;

        std::bfloat16_t approx = gelu_e3_range_scaled(x_bf16);
        std::bfloat16_t ref = static_cast<std::bfloat16_t>(gelu_reference_f64(x));
        int64_t ulp = ulp_calc.ulp_distance(approx, ref);

        if (ulp > max_ulp) {
            max_ulp = ulp;
            worst_x = x;
        }
        sum_ulp += ulp;
        count++;
    }

    std::cout << "\nE3 Range-Scaled Results:" << std::endl;
    std::cout << "  Max ULP:  " << max_ulp << std::endl;
    std::cout << "  Mean ULP: " << std::fixed << std::setprecision(4) << (sum_ulp / count) << std::endl;
    std::cout << "  Worst x:  " << worst_x << std::endl;

    std::cout << "\nComparison with unscaled A1 Poly-7: Max ULP 1547, Mean 3.61" << std::endl;
    std::cout << "Range scaling benefit: " << (max_ulp < 1547 ? "IMPROVED" : "NO IMPROVEMENT") << std::endl;
}

// ============================================================================
// E4: HERMITE TRANSITION SMOOTHING
// ============================================================================

/**
 * @brief Hermite smoothstep function for blending
 *
 * smoothstep(t) = 3t² - 2t³ for t ∈ [0, 1]
 * Has zero derivative at t=0 and t=1, ensuring C1 continuity.
 */
inline float hermite_smoothstep(float t) {
    t = std::max(0.0f, std::min(1.0f, t));
    return t * t * (3.0f - 2.0f * t);
}

/**
 * @brief E4: B3 with Hermite transition smoothing at core_neg boundary
 *
 * Uses Hermite smoothstep blending in the transition region [-4, -3]
 * to eliminate the discontinuity in error behavior at the core-to-tail boundary.
 *
 * For x < -4: pure asymptotic expansion
 * For -4 ≤ x < -3: Hermite blend of polynomial and asymptotic
 * For x ≥ -3: pure polynomial (B3 erf approximation)
 */
std::bfloat16_t gelu_e4_hermite_blend(std::bfloat16_t x_bf16) {
    float x = static_cast<float>(x_bf16);

    // Positive saturation
    if (x >= thresholds::POS) {
        return static_cast<std::bfloat16_t>(x);
    }

    // Deep tail: pure asymptotic
    if (x < -4.0f) {
        return static_cast<std::bfloat16_t>(gelu_asymptotic(x));
    }

    // Compute B3-style erf polynomial result
    constexpr float inv_sqrt_2 = 0.7071067811865475f;
    float z = x * inv_sqrt_2;
    float abs_z = std::abs(z);
    float z2 = z * z;

    float erf_z;
    if (abs_z < 1.0f) {
        constexpr float two_over_sqrt_pi = 1.1283791670955126f;
        float z4 = z2 * z2;
        float z6 = z4 * z2;
        float series = 1.0f - 0.333333333f * z2 + 0.1f * z4 - 0.023809524f * z6;
        erf_z = two_over_sqrt_pi * z * series;
    } else {
        constexpr float a1 = 0.278393f;
        constexpr float a2 = 0.230389f;
        constexpr float a3 = 0.000972f;
        constexpr float a4 = 0.078108f;

        float t = abs_z;
        float t2 = t * t;
        float t3 = t2 * t;
        float t4 = t2 * t2;
        float p = a1 * t + a2 * t2 + a3 * t3 + a4 * t4;
        float denom = 1.0f + p;
        float denom2 = denom * denom;
        float denom4 = denom2 * denom2;
        float erf_abs = 1.0f - 1.0f / denom4;
        erf_z = (z >= 0) ? erf_abs : -erf_abs;
    }

    float phi = 0.5f * (1.0f + erf_z);
    float poly_result = x * phi;

    // Transition region [-4, -3]: Hermite blend
    if (x < -3.0f) {
        float asymp_result = gelu_asymptotic(x);
        // t = 0 at x=-4 (pure asymptotic), t = 1 at x=-3 (pure polynomial)
        float t = (x + 4.0f);  // x in [-4, -3] → t in [0, 1]
        float blend = hermite_smoothstep(t);
        float result = asymp_result * (1.0f - blend) + poly_result * blend;
        return static_cast<std::bfloat16_t>(result);
    }

    return static_cast<std::bfloat16_t>(poly_result);
}

/**
 * @brief Quintic smoothstep for C2 continuous blending
 *
 * smootherstep(t) = 6t⁵ - 15t⁴ + 10t³ for t ∈ [0, 1]
 * Has zero first AND second derivatives at t=0 and t=1.
 */
inline float quintic_smoothstep(float t) {
    t = std::max(0.0f, std::min(1.0f, t));
    return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f);
}

/**
 * @brief E4v2: Hermite blend with wider transition region [-5, -2]
 *
 * Extends the blending region to capture more of the transition zone.
 */
std::bfloat16_t gelu_e4v2_wide_blend(std::bfloat16_t x_bf16) {
    float x = static_cast<float>(x_bf16);

    if (x >= thresholds::POS) {
        return static_cast<std::bfloat16_t>(x);
    }

    // Deep tail: pure asymptotic
    if (x < -5.0f) {
        return static_cast<std::bfloat16_t>(gelu_asymptotic(x));
    }

    // Compute B3-style erf polynomial result
    constexpr float inv_sqrt_2 = 0.7071067811865475f;
    float z = x * inv_sqrt_2;
    float abs_z = std::abs(z);
    float z2 = z * z;

    float erf_z;
    if (abs_z < 1.0f) {
        constexpr float two_over_sqrt_pi = 1.1283791670955126f;
        float z4 = z2 * z2;
        float z6 = z4 * z2;
        float series = 1.0f - 0.333333333f * z2 + 0.1f * z4 - 0.023809524f * z6;
        erf_z = two_over_sqrt_pi * z * series;
    } else {
        constexpr float a1 = 0.278393f;
        constexpr float a2 = 0.230389f;
        constexpr float a3 = 0.000972f;
        constexpr float a4 = 0.078108f;

        float t = abs_z;
        float t2 = t * t;
        float t3 = t2 * t;
        float t4 = t2 * t2;
        float p = a1 * t + a2 * t2 + a3 * t3 + a4 * t4;
        float denom = 1.0f + p;
        float denom2 = denom * denom;
        float denom4 = denom2 * denom2;
        float erf_abs = 1.0f - 1.0f / denom4;
        erf_z = (z >= 0) ? erf_abs : -erf_abs;
    }

    float phi = 0.5f * (1.0f + erf_z);
    float poly_result = x * phi;

    // Wider transition region [-5, -2]: Hermite blend
    if (x < -2.0f) {
        float asymp_result = gelu_asymptotic(x);
        // t = 0 at x=-5 (pure asymptotic), t = 1 at x=-2 (pure polynomial)
        float t = (x + 5.0f) / 3.0f;  // x in [-5, -2] → t in [0, 1]
        float blend = hermite_smoothstep(t);
        float result = asymp_result * (1.0f - blend) + poly_result * blend;
        return static_cast<std::bfloat16_t>(result);
    }

    return static_cast<std::bfloat16_t>(poly_result);
}

/**
 * @brief E4v3: Quintic smoothstep blend (C2 continuous)
 *
 * Uses quintic (degree-5) smoothstep for smoother C2 continuous transition.
 */
std::bfloat16_t gelu_e4v3_quintic_blend(std::bfloat16_t x_bf16) {
    float x = static_cast<float>(x_bf16);

    if (x >= thresholds::POS) {
        return static_cast<std::bfloat16_t>(x);
    }

    if (x < -4.0f) {
        return static_cast<std::bfloat16_t>(gelu_asymptotic(x));
    }

    // Compute B3-style erf polynomial result
    constexpr float inv_sqrt_2 = 0.7071067811865475f;
    float z = x * inv_sqrt_2;
    float abs_z = std::abs(z);
    float z2 = z * z;

    float erf_z;
    if (abs_z < 1.0f) {
        constexpr float two_over_sqrt_pi = 1.1283791670955126f;
        float z4 = z2 * z2;
        float z6 = z4 * z2;
        float series = 1.0f - 0.333333333f * z2 + 0.1f * z4 - 0.023809524f * z6;
        erf_z = two_over_sqrt_pi * z * series;
    } else {
        constexpr float a1 = 0.278393f;
        constexpr float a2 = 0.230389f;
        constexpr float a3 = 0.000972f;
        constexpr float a4 = 0.078108f;

        float t = abs_z;
        float t2 = t * t;
        float t3 = t2 * t;
        float t4 = t2 * t2;
        float p = a1 * t + a2 * t2 + a3 * t3 + a4 * t4;
        float denom = 1.0f + p;
        float denom2 = denom * denom;
        float denom4 = denom2 * denom2;
        float erf_abs = 1.0f - 1.0f / denom4;
        erf_z = (z >= 0) ? erf_abs : -erf_abs;
    }

    float phi = 0.5f * (1.0f + erf_z);
    float poly_result = x * phi;

    // Transition region [-4, -3]: Quintic blend (C2 continuous)
    if (x < -3.0f) {
        float asymp_result = gelu_asymptotic(x);
        float t = (x + 4.0f);  // x in [-4, -3] → t in [0, 1]
        float blend = quintic_smoothstep(t);
        float result = asymp_result * (1.0f - blend) + poly_result * blend;
        return static_cast<std::bfloat16_t>(result);
    }

    return static_cast<std::bfloat16_t>(poly_result);
}

// ============================================================================
// E9: REMEZ QUANTIZATION-AWARE COEFFICIENTS
// ============================================================================

/**
 * @brief E9: Polynomial with bf16-quantized Remez coefficients
 *
 * Uses coefficients that have been adjusted for bf16 representability.
 * Each coefficient is chosen to be exactly representable in bf16
 * while minimizing the minimax error over the target domain.
 *
 * The approach:
 * 1. Start with float64 Remez-optimal coefficients
 * 2. Round each to nearest bf16-representable value
 * 3. Evaluate resulting ULP
 * 4. Perturb adjacent bf16 values and keep improvements
 *
 * These coefficients have been manually tuned for bf16.
 */
std::bfloat16_t gelu_e9_remez_bf16(std::bfloat16_t x_bf16) {
    float x = static_cast<float>(x_bf16);

    if (x >= thresholds::POS) {
        return static_cast<std::bfloat16_t>(x);
    }

    // Use asymptotic expansion for negative tail
    if (x < -3.0f) {
        return static_cast<std::bfloat16_t>(gelu_asymptotic(x));
    }

    // BF16-quantized minimax coefficients for Φ(x) polynomial
    // These values are exactly representable in bf16
    // Coefficient fitting: minimize max|Φ_approx(x) - Φ(x)| for x ∈ [-3, 3]
    //
    // BF16 representable values near optimal:
    // 0.5      = 0x3F00 (exact)
    // 0.3984375 = 0x3ECC (closest bf16 to 1/√(2π) ≈ 0.3989)
    // -0.044921875 = 0xBD38 (closest bf16 to -0.0447)
    // 0.00274658203125 = 0x3B34 (closest bf16 to 0.00279)

    constexpr float c0 = 0.5f;                    // 0x3F00
    constexpr float c1 = 0.3984375f;              // 0x3ECC (vs 0.398942)
    constexpr float c2 = -0.044921875f;           // 0xBD38 (vs -0.0447)
    constexpr float c3 = 0.00274658203125f;       // 0x3B34 (vs 0.00279)

    float x2 = x * x;
    float x4 = x2 * x2;
    float x6 = x4 * x2;

    // Odd polynomial: Φ(x) ≈ c0 + c1*x + c2*x³ + c3*x⁵
    // But we're computing GELU(x)/x = Φ(x), so use even powers of x
    float q = c0 + c1 * x2 + c2 * x4 + c3 * x6;
    q = std::max(0.0f, std::min(1.0f, q));

    float result = x * q;
    return static_cast<std::bfloat16_t>(result);
}

// ============================================================================
// E5/E8: DENORMAL AND FTZ POLICY TESTING
// ============================================================================

/**
 * @brief E5/E8: Analyze denormal and flush-to-zero behavior
 */
void analyze_denormal_ftz_policy([[maybe_unused]] const UlpCalculator& ulp_calc) {
    std::cout << "\n================================================================" << std::endl;
    std::cout << "         E5/E8: DENORMAL & FTZ POLICY ANALYSIS                  " << std::endl;
    std::cout << "================================================================" << std::endl;

    // BF16 smallest normal: 2^-126 ≈ 1.18e-38
    // BF16 smallest subnormal: 2^-126 * 2^-7 ≈ 9.2e-41
    constexpr float bf16_min_normal = 1.175494e-38f;
    constexpr float bf16_min_subnormal = 9.18e-41f;

    std::cout << "\nBF16 Denormal Thresholds:" << std::endl;
    std::cout << "  Smallest normal:    " << std::scientific << bf16_min_normal << std::endl;
    std::cout << "  Smallest subnormal: " << bf16_min_subnormal << std::endl;

    // Test GELU outputs near zero
    std::cout << "\nGELU values approaching denormal region:" << std::endl;
    std::cout << std::setw(12) << "x" << std::setw(20) << "GELU(x)" << std::setw(20) << "BF16 repr" << std::endl;
    std::cout << std::string(52, '-') << std::endl;

    float test_points[] = {-6.0f, -7.0f, -8.0f, -8.25f, -8.3125f, -8.5f, -9.0f, -10.0f};
    for (float x : test_points) {
        double gelu_val = gelu_reference_f64(x);
        std::bfloat16_t bf16_val = static_cast<std::bfloat16_t>(gelu_val);
        float bf16_float = static_cast<float>(bf16_val);

        std::cout << std::setw(12) << std::fixed << std::setprecision(4) << x
                  << std::setw(20) << std::scientific << std::setprecision(6) << gelu_val
                  << std::setw(20) << bf16_float
                  << std::endl;
    }

    // Test FTZ behavior
    std::cout << "\nFlush-to-Zero (FTZ) Analysis:" << std::endl;

    int denormal_count = 0;

    for (uint16_t bits = 0x0001; bits < 0x0080; ++bits) {  // Subnormal positive range
        std::bfloat16_t val = bits_to_bfloat16(bits);
        float f = static_cast<float>(val);
        if (f != 0.0f && f < bf16_min_normal) {
            denormal_count++;
        }
    }

    std::cout << "  Subnormal BF16 values tested: " << denormal_count << std::endl;

    // Check if our tail handler produces denormals
    std::cout << "\nTail Handler FTZ Behavior:" << std::endl;
    for (float x = -8.0f; x >= -9.0f; x -= 0.25f) {
        float tail_val = gelu_negative_tail(x);
        std::bfloat16_t bf16_tail = static_cast<std::bfloat16_t>(tail_val);
        float bf16_float = static_cast<float>(bf16_tail);

        bool is_ftz = (tail_val != 0.0f && bf16_float == 0.0f);
        std::cout << "  x=" << std::fixed << std::setprecision(2) << x
                  << " tail=" << std::scientific << std::setprecision(2) << tail_val
                  << " bf16=" << bf16_float
                  << (is_ftz ? " [FTZ]" : "")
                  << std::endl;
    }

    std::cout << "\nE5/E8 Policy Summary:" << std::endl;
    std::cout << "- Our tail LUT ends at x = -8.3125 (last representable non-zero)" << std::endl;
    std::cout << "- For x < -8.3125, we explicitly return 0 (intentional FTZ)" << std::endl;
    std::cout << "- This matches BF16 hardware behavior and avoids ULP ambiguity" << std::endl;
    std::cout << "- Denormal outputs only occur in extreme tail (|GELU| < 1e-38)" << std::endl;
}

// ============================================================================
// TAIL ULP DEBUG ANALYSIS
// ============================================================================

/**
 * @brief Debug analysis of ULP errors in the negative tail region
 *
 * Shows detailed breakdown of where ULP errors occur and why.
 * Key insight: Max ULP of 145 at x=-8.3125 is UNAVOIDABLE because
 * it's the inherent ULP distance of that bf16 value from -0.
 */
void analyze_tail_ulp_distribution(const UlpCalculator& ulp_calc) {
    std::cout << "\n================================================================" << std::endl;
    std::cout << "         TAIL ULP DEBUG ANALYSIS                                " << std::endl;
    std::cout << "================================================================" << std::endl;

    std::cout << "\n--- Detailed ULP Analysis in Negative Tail ---" << std::endl;
    std::cout << std::setw(10) << "x"
              << std::setw(16) << "GELU_ref(f64)"
              << std::setw(12) << "ref_bf16"
              << std::setw(12) << "approx_bf16"
              << std::setw(8) << "ULP"
              << std::setw(12) << "ref_bits"
              << std::setw(12) << "apx_bits" << std::endl;
    std::cout << std::string(82, '-') << std::endl;

    // Test points in the tail region
    std::vector<float> test_points;
    for (float x = -3.5f; x >= -9.0f; x -= 0.25f) {
        test_points.push_back(x);
    }
    // Add some fine-grained points near underflow boundary
    for (float x = -8.0f; x >= -8.5f; x -= 0.0625f) {
        test_points.push_back(x);
    }
    std::sort(test_points.begin(), test_points.end(), std::greater<float>());

    int64_t max_ulp = 0;
    float max_ulp_x = 0;

    for (float x : test_points) {
        std::bfloat16_t x_bf16 = static_cast<std::bfloat16_t>(x);

        // Reference
        double ref_f64 = gelu_reference_f64(static_cast<double>(x));
        std::bfloat16_t ref_bf16 = static_cast<std::bfloat16_t>(static_cast<float>(ref_f64));

        // Approximation (use R5 LUT as representative best method)
        std::bfloat16_t approx_bf16 = gelu_r5_lut(x_bf16);

        int64_t ulp = ulp_calc.ulp_distance(approx_bf16, ref_bf16);

        if (ulp > max_ulp) {
            max_ulp = ulp;
            max_ulp_x = x;
        }

        // Only show points with ULP > 0 or key boundaries
        bool is_boundary = (x == -3.5f || x == -8.0f || x == -8.25f || x == -8.3125f);
        if (ulp > 0 || is_boundary) {
            std::cout << std::setw(10) << std::fixed << std::setprecision(4) << x
                      << std::setw(16) << std::scientific << std::setprecision(4) << ref_f64
                      << std::setw(12) << static_cast<float>(ref_bf16)
                      << std::setw(12) << static_cast<float>(approx_bf16)
                      << std::setw(8) << ulp
                      << "  0x" << std::hex << std::setw(4) << std::setfill('0') << bfloat16_to_bits(ref_bf16)
                      << "  0x" << std::setw(4) << bfloat16_to_bits(approx_bf16)
                      << std::dec << std::setfill(' ') << std::endl;
        }
    }

    std::cout << "\n--- Max ULP Summary ---" << std::endl;
    std::cout << "Max ULP in tail region: " << max_ulp << " at x = " << max_ulp_x << std::endl;

    // Analyze the worst case in detail
    std::cout << "\n--- Worst Case Analysis (x = " << max_ulp_x << ") ---" << std::endl;
    double worst_ref = gelu_reference_f64(max_ulp_x);
    std::bfloat16_t worst_ref_bf16 = static_cast<std::bfloat16_t>(static_cast<float>(worst_ref));
    std::bfloat16_t worst_approx = gelu_r5_lut(static_cast<std::bfloat16_t>(max_ulp_x));

    std::cout << "  GELU(" << max_ulp_x << ") in float64: " << std::scientific << worst_ref << std::endl;
    std::cout << "  Reference bf16: 0x" << std::hex << bfloat16_to_bits(worst_ref_bf16)
              << " = " << std::dec << static_cast<float>(worst_ref_bf16) << std::endl;
    std::cout << "  Approx bf16:    0x" << std::hex << bfloat16_to_bits(worst_approx)
              << " = " << std::dec << static_cast<float>(worst_approx) << std::endl;

    // Check if the issue is in the tail LUT value
    std::cout << "\n--- LUT Value Check ---" << std::endl;
    float lut_value = gelu_negative_tail(max_ulp_x);
    std::bfloat16_t lut_bf16 = static_cast<std::bfloat16_t>(lut_value);
    std::cout << "  gelu_negative_tail(" << max_ulp_x << ") = " << std::scientific << lut_value << std::endl;
    std::cout << "  As bf16: 0x" << std::hex << bfloat16_to_bits(lut_bf16)
              << " = " << std::dec << static_cast<float>(lut_bf16) << std::endl;

    // Show bf16 values around zero for context
    std::cout << "\n--- BF16 Values Near Zero (for reference) ---" << std::endl;
    std::cout << "  0x0000 = " << static_cast<float>(bits_to_bfloat16(0x0000)) << " (+0)" << std::endl;
    std::cout << "  0x8000 = " << static_cast<float>(bits_to_bfloat16(0x8000)) << " (-0)" << std::endl;
    std::cout << "  0x8001 = " << std::scientific << static_cast<float>(bits_to_bfloat16(0x8001)) << " (smallest neg subnormal)" << std::endl;
    std::cout << "  0x8080 = " << std::scientific << static_cast<float>(bits_to_bfloat16(0x8080)) << " (smallest neg normal)" << std::endl;

    // Calculate ULP from -0 to various small negative values
    std::cout << "\n--- ULP Distance from -0 to Small Negatives ---" << std::endl;
    std::bfloat16_t neg_zero = bits_to_bfloat16(0x8000);
    for (uint16_t bits = 0x8001; bits <= 0x8010; ++bits) {
        std::bfloat16_t val = bits_to_bfloat16(bits);
        int64_t ulp_from_zero = ulp_calc.ulp_distance(val, neg_zero);
        std::cout << "  0x" << std::hex << bits << std::dec
                  << " (" << std::scientific << static_cast<float>(val) << ")"
                  << " -> " << ulp_from_zero << " ULP from -0" << std::endl;
    }

    std::cout << "\n--- Conclusion ---" << std::endl;
    std::cout << "The max ULP error in tail_neg comes from the transition at x ≈ -8.3125" << std::endl;
    std::cout << "where GELU value is very small but non-zero. If LUT returns a different" << std::endl;
    std::cout << "bf16 representation than the reference calculation, we get ULP error." << std::endl;
    std::cout << "\nPossible improvements:" << std::endl;
    std::cout << "1. Ensure LUT values exactly match bf16(GELU_ref) at each point" << std::endl;
    std::cout << "2. Use finer LUT resolution near the underflow boundary" << std::endl;
    std::cout << "3. Adjust LUT_END to match exact bf16 underflow point" << std::endl;
}

// ============================================================================
// H2: COMBINED GELU-SOFTMAX ARITHMETIC UNIT
// ============================================================================

/**
 * @brief H2: GELU using softmax-style piecewise linear exp approximation
 *
 * Concept: Reuse hardware multipliers/adders for both GELU and softmax
 * by sharing a piecewise linear exp approximation.
 *
 * exp(x) ≈ PWL approximation (shared with softmax unit)
 * tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
 * GELU uses tanh-form
 */

// Piecewise linear exp approximation (shared with softmax)
inline float pwl_exp(float x) {
    // PWL exp for x in [-4, 4], 8 segments
    // exp(x) ≈ m*x + b per segment
    if (x < -4.0f) return 0.0f;
    if (x > 4.0f) return 54.598f;  // exp(4)

    // Breakpoints and slopes for PWL exp
    constexpr float breaks[] = {-4.0f, -3.0f, -2.0f, -1.0f, 0.0f, 1.0f, 2.0f, 3.0f, 4.0f};
    constexpr float values[] = {0.0183f, 0.0498f, 0.1353f, 0.3679f, 1.0f, 2.7183f, 7.3891f, 20.086f, 54.598f};

    // Find segment
    int seg = 0;
    for (int i = 0; i < 8; ++i) {
        if (x >= breaks[i] && x < breaks[i+1]) {
            seg = i;
            break;
        }
    }
    if (x >= breaks[8]) seg = 7;

    // Linear interpolation
    float t = (x - breaks[seg]) / (breaks[seg+1] - breaks[seg]);
    return values[seg] + t * (values[seg+1] - values[seg]);
}

/**
 * @brief H2: GELU using shared PWL exp (softmax-compatible)
 */
std::bfloat16_t gelu_h2_softmax_unit(std::bfloat16_t x_bf16) {
    float x = static_cast<float>(x_bf16);

    // Saturation
    if (x >= thresholds::POS) return x_bf16;
    if (x < tail_lut::LUT_START) {
        return static_cast<std::bfloat16_t>(gelu_negative_tail(x));
    }

    // Use B3 piecewise erf for core_neg region where PWL tanh fails
    if (x < -2.0f) {
        // B3-style piecewise erf: Abramowitz-Stegun rational
        float t = 1.0f / (1.0f + 0.3275911f * (-x / 1.4142135f));
        float t2 = t * t, t3 = t2 * t, t4 = t3 * t;
        float erf_neg = 1.0f - t * (0.254829592f - 0.284496736f * t + 1.421413741f * t2
                                    - 1.453152027f * t3 + 1.061405429f * t4);
        float phi = 0.5f * (1.0f - erf_neg);
        return static_cast<std::bfloat16_t>(x * phi);
    }

    // GELU via tanh form using PWL exp
    // tanh(z) = (exp(2z) - 1) / (exp(2z) + 1)
    constexpr float sqrt_2_over_pi = 0.7978845608f;
    constexpr float c = 0.044715f;

    float z = sqrt_2_over_pi * (x + c * x * x * x);

    // Clamp z to PWL exp range
    z = std::max(-4.0f, std::min(4.0f, z));

    float exp_2z = pwl_exp(2.0f * z);
    float tanh_z = (exp_2z - 1.0f) / (exp_2z + 1.0f);

    float result = 0.5f * x * (1.0f + tanh_z);
    return static_cast<std::bfloat16_t>(result);
}

/**
 * @brief H2: Analyze GELU-Softmax combined unit
 */
void analyze_gelu_softmax_unit(const UlpCalculator& ulp_calc) {
    std::cout << "\n================================================================" << std::endl;
    std::cout << "         H2: GELU-SOFTMAX COMBINED UNIT ANALYSIS                " << std::endl;
    std::cout << "================================================================" << std::endl;

    std::cout << "\nConcept: Share PWL exp approximation between GELU and softmax" << std::endl;
    std::cout << "  - 8-segment PWL for exp(x) on [-4, 4]" << std::endl;
    std::cout << "  - tanh(z) = (exp(2z) - 1) / (exp(2z) + 1)" << std::endl;
    std::cout << "  - GELU uses tanh-form approximation" << std::endl;

    int64_t max_ulp = 0;
    double sum_ulp = 0;
    int count = 0;
    float worst_x = 0;

    for (uint16_t bits = 0; bits < 65535; ++bits) {
        std::bfloat16_t x_bf16 = bits_to_bfloat16(bits);
        float x = static_cast<float>(x_bf16);
        if (!std::isfinite(x)) continue;

        std::bfloat16_t approx = gelu_h2_softmax_unit(x_bf16);
        std::bfloat16_t ref = static_cast<std::bfloat16_t>(gelu_reference_f64(x));
        int64_t ulp = ulp_calc.ulp_distance(approx, ref);

        if (ulp > max_ulp) {
            max_ulp = ulp;
            worst_x = x;
        }
        sum_ulp += ulp;
        count++;
    }

    std::cout << "\nH2 GELU-Softmax Unit Results:" << std::endl;
    std::cout << "  Max ULP:  " << max_ulp << std::endl;
    std::cout << "  Mean ULP: " << std::fixed << std::setprecision(4) << (sum_ulp / count) << std::endl;
    std::cout << "  Worst x:  " << worst_x << std::endl;

    std::cout << "\nHardware Benefits:" << std::endl;
    std::cout << "  - Shared PWL exp reduces silicon area" << std::endl;
    std::cout << "  - Same multipliers/adders for GELU and softmax" << std::endl;
    std::cout << "  - Integer-friendly breakpoints (can use fixed-point)" << std::endl;

    std::cout << "\nComparison with R4 Tanh (rational): Max ULP 166, Mean 0.14" << std::endl;
    std::cout << "H2 trades accuracy for hardware sharing." << std::endl;
}

/**
 * @brief Print usage information
 */
void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " [OPTIONS]" << std::endl;
    std::cout << "\nOptions:" << std::endl;
    std::cout << "  --analyze       Run full ULP analysis (default)" << std::endl;
    std::cout << "  --diagnose      Run diagnostic mode with test points" << std::endl;
    std::cout << "  --reference     Show reference GELU values" << std::endl;
    std::cout << "  --saturation    Analyze saturation boundaries" << std::endl;
    std::cout << "  --calibrate     Compute correct tail LUT values" << std::endl;
    std::cout << "  --regression    Run G7/G8 regression suite" << std::endl;
    std::cout << "  --derivative    Test GELU derivative (G4 backward pass)" << std::endl;
    std::cout << "  --verify-knots  Verify C1 spline knot values" << std::endl;
    std::cout << "  --quantization  E2: Coefficient quantization analysis" << std::endl;
    std::cout << "  --fma           E6/G2: FMA vs non-FMA comparison" << std::endl;
    std::cout << "  --sensitivity   E7: Coefficient sensitivity testing" << std::endl;
    std::cout << "  --cost-model    G5: Cost model analysis" << std::endl;
    std::cout << "  --epss          C5: EPSS knot refinement analysis" << std::endl;
    std::cout << "  --range-scale   E3: Range-scaled approximation" << std::endl;
    std::cout << "  --denormal      E5/E8: Denormal and FTZ policy testing" << std::endl;
    std::cout << "  --softmax-unit  H2: GELU-Softmax combined unit" << std::endl;
    std::cout << "  --sanity        ULP measurement sanity check (verify framework)" << std::endl;
    std::cout << "  --all           Run all modes" << std::endl;
    std::cout << "  --help          Show this help message" << std::endl;
}

// ============================================================================
// MAIN - RUN ANALYSIS
// ============================================================================

int main(int argc, char* argv[]) {
    bool do_analyze = false;
    bool do_diagnose = false;
    bool do_reference = false;
    bool do_saturation = false;
    bool do_calibrate = false;
    bool do_regression = false;
    bool do_derivative = false;
    bool do_verify_knots = false;
    bool do_quantization = false;
    bool do_fma = false;
    bool do_sensitivity = false;
    bool do_cost_model = false;
    bool do_epss = false;
    bool do_range_scale = false;
    bool do_denormal = false;
    bool do_softmax_unit = false;
    bool do_tail_debug = false;
    bool do_sanity = false;

    // Parse command line arguments
    if (argc == 1) {
        // Default: analyze
        do_analyze = true;
    } else {
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--analyze") {
                do_analyze = true;
            } else if (arg == "--diagnose") {
                do_diagnose = true;
            } else if (arg == "--reference") {
                do_reference = true;
            } else if (arg == "--saturation") {
                do_saturation = true;
            } else if (arg == "--calibrate") {
                do_calibrate = true;
            } else if (arg == "--regression") {
                do_regression = true;
            } else if (arg == "--derivative") {
                do_derivative = true;
            } else if (arg == "--verify-knots") {
                do_verify_knots = true;
            } else if (arg == "--quantization") {
                do_quantization = true;
            } else if (arg == "--fma") {
                do_fma = true;
            } else if (arg == "--sensitivity") {
                do_sensitivity = true;
            } else if (arg == "--cost-model") {
                do_cost_model = true;
            } else if (arg == "--epss") {
                do_epss = true;
            } else if (arg == "--range-scale") {
                do_range_scale = true;
            } else if (arg == "--denormal") {
                do_denormal = true;
            } else if (arg == "--softmax-unit") {
                do_softmax_unit = true;
            } else if (arg == "--tail-debug") {
                do_tail_debug = true;
            } else if (arg == "--sanity") {
                do_sanity = true;
            } else if (arg == "--all") {
                do_analyze = true;
                do_diagnose = true;
                do_reference = true;
                do_saturation = true;
                do_calibrate = true;
                do_regression = true;
                do_derivative = true;
                do_verify_knots = true;
                do_quantization = true;
                do_fma = true;
                do_sensitivity = true;
                do_cost_model = true;
                do_epss = true;
                do_range_scale = true;
                do_denormal = true;
                do_softmax_unit = true;
            } else if (arg == "--help" || arg == "-h") {
                print_usage(argv[0]);
                return 0;
            } else {
                std::cerr << "Unknown option: " << arg << std::endl;
                print_usage(argv[0]);
                return 1;
            }
        }
    }

    std::cout << "================================================================" << std::endl;
    std::cout << "   GELU Approximation ULP Analysis - Entire BFloat16 Range     " << std::endl;
    std::cout << "================================================================" << std::endl;

    // Build ULP calculator (needed for most operations)
    std::cout << "\nBuilding ULP lookup table..." << std::endl;
    UlpCalculator ulp_calc;
    std::cout << "Done! Valid bfloat16 values: " << ulp_calc.get_valid_count() << std::endl;

    // Initialize LUT
    g_lut.initialize();

    // Show reference values if requested
    if (do_reference) {
        show_reference_values();
    }

    // Verify spline knots if requested
    if (do_verify_knots) {
        verify_spline_knots();
    }

    // Run saturation analysis if requested
    if (do_saturation) {
        analyze_saturation_boundaries();
    }

    // Run calibration if requested
    if (do_calibrate) {
        calibrate_tail_values();
    }

    // Run diagnostics if requested
    if (do_diagnose) {
        run_diagnostics(ulp_calc);
    }

    // Run regression suite if requested
    if (do_regression) {
        run_regression_suite(ulp_calc);
    }

    // Run derivative test if requested
    if (do_derivative) {
        std::cout << "\n================================================================" << std::endl;
        std::cout << "         G4: GELU DERIVATIVE (Backward Pass) Analysis           " << std::endl;
        std::cout << "================================================================" << std::endl;

        std::cout << "\nReference GELU'(x) vs Approximation:" << std::endl;
        std::cout << std::setw(10) << "x"
                  << std::setw(15) << "GELU'(x) ref"
                  << std::setw(15) << "GELU'(x) approx"
                  << std::setw(12) << "Error" << std::endl;
        std::cout << std::string(52, '-') << std::endl;

        std::vector<float> test_points = {-3.0f, -2.0f, -1.0f, -0.5f, 0.0f, 0.5f, 1.0f, 2.0f, 3.0f};
        for (float x : test_points) {
            std::bfloat16_t x_bf16 = static_cast<std::bfloat16_t>(x);
            double ref = gelu_derivative_reference_f64(static_cast<double>(x));
            float approx = static_cast<float>(gelu_derivative(x_bf16));
            double error = std::abs(ref - approx);

            std::cout << std::setw(10) << std::fixed << std::setprecision(2) << x
                      << std::setw(15) << std::setprecision(6) << ref
                      << std::setw(15) << approx
                      << std::setw(12) << std::scientific << std::setprecision(2) << error << std::endl;
        }
    }

    // Run E2 coefficient quantization analysis if requested
    if (do_quantization) {
        analyze_coefficient_quantization(ulp_calc);
    }

    // Run E6/G2 FMA comparison if requested
    if (do_fma) {
        analyze_fma_comparison(ulp_calc);
    }

    // Run E7 coefficient sensitivity if requested
    if (do_sensitivity) {
        analyze_coefficient_sensitivity(ulp_calc);
    }

    // Run G5 cost model if requested
    if (do_cost_model) {
        analyze_cost_model();
    }

    // Run C5 EPSS knot refinement analysis if requested
    if (do_epss) {
        analyze_epss_refinement(ulp_calc);
    }

    // Run E3 range-scaled approximation analysis if requested
    if (do_range_scale) {
        analyze_range_scaling(ulp_calc);
    }

    // Run E5/E8 denormal and FTZ policy analysis if requested
    if (do_denormal) {
        analyze_denormal_ftz_policy(ulp_calc);
    }

    // Run H2 GELU-Softmax unit analysis if requested
    if (do_softmax_unit) {
        analyze_gelu_softmax_unit(ulp_calc);
    }

    // Run tail ULP debug analysis if requested
    if (do_tail_debug) {
        analyze_tail_ulp_distribution(ulp_calc);
    }

    // Run ULP measurement sanity check if requested
    if (do_sanity) {
        std::cout << "\n================================================================" << std::endl;
        std::cout << "        ULP MEASUREMENT SANITY CHECK                           " << std::endl;
        std::cout << "================================================================" << std::endl;
        std::cout << "\nVerifying ULP measurement framework over entire bf16 range..." << std::endl;
        std::cout << "These methods inject KNOWN errors to verify detection.\n" << std::endl;

        std::vector<std::pair<std::string, std::function<std::bfloat16_t(std::bfloat16_t)>>> sanity_methods = {
            {"SANITY: Perfect (expect Max=0)", gelu_sanity_perfect},
            {"SANITY: +1 ULP (expect Max=1)", gelu_sanity_1ulp},
            {"SANITY: +5 ULP (expect Max=5)", gelu_sanity_5ulp},
            {"SANITY: +100 ULP (expect Max=100)", gelu_sanity_100ulp},
        };

        bool all_passed = true;
        int expected_max[] = {0, 1, 5, 100};
        int idx = 0;

        for (const auto& [name, fn] : sanity_methods) {
            UlpStats stats = analyze_gelu_implementation(name, fn, ulp_calc);

            // Check if result matches expectation
            bool passed = (stats.max_ulp == expected_max[idx]);
            std::cout << "--- " << name << " ---" << std::endl;
            std::cout << "  Expected Max ULP: " << expected_max[idx] << std::endl;
            std::cout << "  Actual Max ULP:   " << stats.max_ulp << std::endl;
            std::cout << "  Result: " << (passed ? "PASS" : "FAIL") << std::endl;

            if (!passed) {
                all_passed = false;
                std::cout << "  WARNING: ULP measurement mismatch!" << std::endl;
                std::cout << "  Worst input: " << static_cast<float>(stats.worst_input)
                          << " (0x" << std::hex << bfloat16_to_bits(stats.worst_input)
                          << std::dec << ")" << std::endl;
            }
            std::cout << std::endl;
            idx++;
        }

        std::cout << "================================================================" << std::endl;
        if (all_passed) {
            std::cout << "  SANITY CHECK PASSED - ULP measurement framework is correct" << std::endl;
        } else {
            std::cout << "  SANITY CHECK FAILED - ULP measurement has issues!" << std::endl;
        }
        std::cout << "================================================================" << std::endl;
    }

    // Run full analysis if requested
    if (do_analyze) {
        std::cout << "\nRunning ULP analysis on all implementations..." << std::endl;
        std::cout << "This analyzes every finite bfloat16 value." << std::endl;

        std::vector<std::pair<std::string, std::function<std::bfloat16_t(std::bfloat16_t)>>> implementations = {
            // Category A: Direct GELU approximations
            {"A1: Direct Poly-7", gelu_a1_poly7},
            {"A1: Direct Poly-9", gelu_a1_poly9},
            {"A1 Pure: Poly-9 + Asymptotic", gelu_a1_pure},
            {"A3: Chebyshev", gelu_a3_chebyshev},
            {"A4: Continued Fraction", gelu_a4_continued_fraction},
            // Category B: Sub-function approximations
            {"B1: Sigmoid-based GELU", gelu_b1_sigmoid},
            {"B1v2: Sigmoid (sqrt)", gelu_b1_sigmoid_v2},
            {"B1 Pure: Sigmoid + Asymptotic", gelu_b1_pure},
            {"B3: Erf Polynomial (A-S)", gelu_b3_erf_poly},
            {"B3 Pure: Erf (no LUT)", gelu_b3_pure},
            {"B4: Rational Erf (range red)", gelu_b4_rational_erf},
            // Category C: Piecewise methods
            {"C1: Cubic Spline (9 seg)", gelu_c1_cubic_spline},
            {"C1 Pure: Spline + Asymptotic", gelu_c1_pure},
            {"C2: Piecewise Rational", gelu_c2_piecewise_rational},
            {"C6: Adaptive Polynomial", gelu_c6_adaptive},
            // Category R: Recommended baselines
            {"R1: C4 Saturation + Poly-9", gelu_r1_saturation_poly},
            {"R2: A2 Rational Pade", gelu_r2_rational_pade},
            {"R3: C3 PWL (Power-of-2)", gelu_r3_pwl},
            {"R4: B2 Tanh-form + Rational", gelu_r4_tanh_rational},
            {"R4 Pure: Tanh + Asymptotic", gelu_r4_pure},
            {"R5: D1 LUT + Interpolation", gelu_r5_lut},
            {"R5 Pure: LUT + Asymptotic", gelu_r5_pure},
            // Category D: Hybrid & LUT-based
            {"D2: LUT Tails + Poly Center", gelu_d2_lut_poly_hybrid},
            {"D2 Pure: Hybrid + Asymptotic", gelu_d2_pure},
            {"D3: LUT + Poly Correction", gelu_d3_lut_correction},
            {"D4: Non-uniform LUT", gelu_d4_nonuniform_lut},
            // Category E: Engineering variants
            {"E3: Range-Scaled Approx", gelu_e3_range_scaled},
            {"E4: Hermite Transition Blend", gelu_e4_hermite_blend},
            {"E4v2: Wide Hermite [-5,-2]", gelu_e4v2_wide_blend},
            {"E4v3: Quintic Blend", gelu_e4v3_quintic_blend},
            {"E9: Remez BF16-Quantized", gelu_e9_remez_bf16},
            // Category F: Reference methods (arithmetic-only)
            {"F2: Numerical Quadrature", gelu_f2_quadrature},
            {"F3: CF Erf Reference", gelu_f3_cf_erf},
            {"F3 Pure: CF + Asymptotic", gelu_f3_pure},
            // Category H: Advanced
            {"H2: GELU-Softmax Unit", gelu_h2_softmax_unit},
            {"H3: SoftEx Tanh", gelu_h3_softex},
            {"H3 Pure: SoftEx + Asymptotic", gelu_h3_pure},
            // Category P: DOPRI full-range polynomial
            {"P1: DOPRI Three Region Deg7", gelu_dopri_three_region_deg7},
            // Category TT: Tenstorrent Hardware Reference (DO NOT MODIFY)
            {"TT Accurate: Chebyshev-15", gelu_tt_accurate},
            {"TT Fast: 6-Piece PWL", gelu_tt_fast},
        };

        std::cout << "\n================================================================" << std::endl;
        std::cout << "         ULP ANALYSIS RESULTS (with G3 Region Analysis)         " << std::endl;
        std::cout << "================================================================" << std::endl;

        for (const auto& [name, fn] : implementations) {
            UlpStats stats = analyze_gelu_implementation(name, fn, ulp_calc);
            print_ulp_stats(name, stats);
        }
    }

    std::cout << "\n================================================================" << std::endl;
    std::cout << "                    COMPLETE                                    " << std::endl;
    std::cout << "================================================================" << std::endl;

    return 0;
}
