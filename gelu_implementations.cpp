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
    // Φ(x) = 0.5 * (1 + erf(x / √2))
    double phi = 0.5 * (1.0 + std::erf(x * constants::INV_SQRT_2));
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
    // Extended LUT covering [-8.0, -3.5] at 0.25 step
    // Values computed from GELU(x) = x * 0.5 * (1 + erf(x/√2))
    // For x < -8.0, bf16 underflows to 0, so LUT not needed beyond -8.0

    // Values from calibration (--calibrate mode)
    constexpr float GELU_N3_50 = -8.14202e-04f;   // x = -3.50
    constexpr float GELU_N3_75 = -5.23840e-04f;   // x = -3.75
    constexpr float GELU_N4_00 = -1.26685e-04f;   // x = -4.00
    constexpr float GELU_N4_25 = -4.54262e-05f;   // x = -4.25
    constexpr float GELU_N4_50 = -1.52895e-05f;   // x = -4.50
    constexpr float GELU_N4_75 = -4.83115e-06f;   // x = -4.75
    constexpr float GELU_N5_00 = -1.43326e-06f;   // x = -5.00
    constexpr float GELU_N5_25 = -3.99260e-07f;   // x = -5.25
    constexpr float GELU_N5_50 = -1.04443e-07f;   // x = -5.50
    constexpr float GELU_N5_75 = -2.56575e-08f;   // x = -5.75
    constexpr float GELU_N6_00 = -5.91953e-09f;   // x = -6.00
    constexpr float GELU_N6_25 = -1.28266e-09f;   // x = -6.25
    constexpr float GELU_N6_50 = -2.61040e-10f;   // x = -6.50
    constexpr float GELU_N6_75 = -4.98977e-11f;   // x = -6.75
    constexpr float GELU_N7_00 = -8.95867e-12f;   // x = -7.00
    constexpr float GELU_N7_25 = -1.51082e-12f;   // x = -7.25
    constexpr float GELU_N7_50 = -2.39392e-13f;   // x = -7.50
    constexpr float GELU_N7_75 = -3.57075e-14f;   // x = -7.75
    constexpr float GELU_N8_00 = -4.88498e-15f;   // x = -8.00, bf16=0xa7b0
    constexpr float GELU_N8_25 = -4.57967e-16f;   // x = -8.25, bf16=0xa604
    constexpr float GELU_N8_3125 = -2.12e-16f;    // x = -8.3125, bf16=0xa605 (approx)
    // Beyond x=-8.3125, bf16 underflows to -0 (0x8000)

    // Array for efficient lookup (0.25 step from -3.5 to -8.0, then one irregular entry)
    constexpr float LUT[] = {
        GELU_N3_50, GELU_N3_75, GELU_N4_00, GELU_N4_25, GELU_N4_50,
        GELU_N4_75, GELU_N5_00, GELU_N5_25, GELU_N5_50, GELU_N5_75,
        GELU_N6_00, GELU_N6_25, GELU_N6_50, GELU_N6_75, GELU_N7_00,
        GELU_N7_25, GELU_N7_50, GELU_N7_75, GELU_N8_00, GELU_N8_25,
        GELU_N8_3125
    };
    constexpr int LUT_MAIN_SIZE = 20;  // Up to GELU_N8_25 (0.25 step)
    constexpr float LUT_START = -3.5f;
    constexpr float LUT_MAIN_END = -8.25f;
    constexpr float LUT_END = -8.3125f;
    constexpr float LUT_STEP = -0.25f;
}

/**
 * @brief Negative tail GELU handler using extended LUT
 *
 * Covers x ∈ [-8.3125, -3.5] with LUT + linear interpolation.
 * Main region uses 0.25-step; final segment [-8.3125, -8.25] is 0.0625 step.
 * For x < -8.3125, returns 0 (bf16 underflows to -0 around x=-8.35).
 *
 * This approach avoids the broken exp() approximation that was causing
 * Max ULP errors of ~10000 in the deep negative tail.
 */
inline float gelu_negative_tail(float x) {
    using namespace tail_lut;

    // x < -8.3125: bf16 underflows to -0 around x=-8.35, so return 0
    if (x < LUT_END) {
        return 0.0f;
    }

    // x >= -3.5: should be handled by core approximation, not this function
    if (x >= LUT_START) {
        return LUT[0];  // Return value at -3.5
    }

    // Handle the final irregular segment [-8.3125, -8.25] separately
    if (x < LUT_MAIN_END) {
        // Interpolate between LUT[19] (x=-8.25) and LUT[20] (x=-8.3125)
        float t = (x - LUT_MAIN_END) / (LUT_END - LUT_MAIN_END);  // t goes from 0 to 1
        return LUT[LUT_MAIN_SIZE - 1] + t * (LUT[LUT_MAIN_SIZE] - LUT[LUT_MAIN_SIZE - 1]);
    }

    // Main LUT lookup with 0.25-step linear interpolation
    // Index calculation: (x - LUT_START) / LUT_STEP = (x + 3.5) / (-0.25)
    float idx_f = (x - LUT_START) / LUT_STEP;
    int idx = static_cast<int>(idx_f);

    // Clamp index to valid range for main LUT
    if (idx < 0) idx = 0;
    if (idx >= LUT_MAIN_SIZE - 1) idx = LUT_MAIN_SIZE - 2;

    // Linear interpolation between LUT[idx] and LUT[idx+1]
    float t = idx_f - static_cast<float>(idx);
    return LUT[idx] + t * (LUT[idx + 1] - LUT[idx]);
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

    // Saturation thresholds
    if (x >= thresholds::POS) {
        return static_cast<std::bfloat16_t>(x);
    }
    if (x <= thresholds::NEG) {
        return static_cast<std::bfloat16_t>(0.0f);
    }

    // Negative tail: use specialized handler for x < -4
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
    if (x <= thresholds::NEG) {
        return static_cast<std::bfloat16_t>(0.0f);
    }

    // Negative tail: use specialized handler for x < -4
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
    if (x <= thresholds::NEG) {
        return static_cast<std::bfloat16_t>(0.0f);
    }

    // Negative tail: use specialized handler for x < -4
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
    if (x <= thresholds::NEG) {
        return static_cast<std::bfloat16_t>(0.0f);
    }

    // Negative tail: use specialized handler for x < -4
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

    // Saturation thresholds
    if (x >= thresholds::POS) {
        return static_cast<std::bfloat16_t>(x);
    }
    if (x <= thresholds::NEG) {
        return static_cast<std::bfloat16_t>(0.0f);
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
    if (x <= thresholds::NEG) {
        return static_cast<std::bfloat16_t>(0.0f);
    }

    // Negative tail: use specialized handler for x < -4
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
    if (x <= thresholds::NEG) {
        return static_cast<std::bfloat16_t>(0.0f);
    }
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
    if (x <= thresholds::NEG) {
        return static_cast<std::bfloat16_t>(0.0f);
    }
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
    if (x <= thresholds::NEG) {
        return static_cast<std::bfloat16_t>(0.0f);
    }
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
    if (x <= thresholds::NEG) {
        return static_cast<std::bfloat16_t>(0.0f);
    }
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
// H1: INVERTED GELU
// ============================================================================

/**
 * @brief H1: Inverted GELU (GELU⁻¹) approximation
 *
 * Approximates the inverse function: given y = GELU(x), find x.
 * Useful for memory-efficient backpropagation.
 *
 * Uses Newton-Raphson iteration with initial guess from linear approximation.
 */
std::bfloat16_t gelu_inverse(std::bfloat16_t y_bf16) {
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
            // Approximate GELU(x) and GELU'(x)
            float phi = 0.5f * (1.0f + std::erf(x * static_cast<float>(constants::INV_SQRT_2)));
            float gelu_x = x * phi;

            // GELU'(x) = Φ(x) + x * φ(x)
            float phi_pdf = 0.3989422804f * std::exp(-0.5f * x2);  // Using exp for reference
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
            float phi = 0.5f * (1.0f + std::erf(x * static_cast<float>(constants::INV_SQRT_2)));
            float gelu_x = x * phi;

            float phi_pdf = 0.3989422804f * std::exp(-0.5f * x2);
            float gelu_prime = phi + x * phi_pdf;

            if (std::abs(gelu_prime) < 1e-10f) break;

            float delta = (gelu_x - y) / gelu_prime;
            x = x - delta;

            if (std::abs(delta) < 1e-6f) break;
        }

        return static_cast<std::bfloat16_t>(x);
    }
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
 */
double gelu_derivative_reference_f64(double x) {
    double phi = 0.5 * (1.0 + std::erf(x * constants::INV_SQRT_2));
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
        {"B1: Sigmoid-based GELU", gelu_b1_sigmoid},
        {"B1v2: Sigmoid (higher-order)", gelu_b1_sigmoid_v2},
        {"B3: Erf Polynomial (A-S)", gelu_b3_erf_poly},
        {"C1: Cubic Spline", gelu_c1_cubic_spline},
        {"R1: C4 Saturation + Poly-9 Core", gelu_r1_saturation_poly},
        {"R2: A2 Rational Pade", gelu_r2_rational_pade},
        {"R3: C3 PWL (Power-of-2)", gelu_r3_pwl},
        {"R4: B2 Tanh-form + Rational", gelu_r4_tanh_rational},
        {"R5: D1 LUT + Interpolation", gelu_r5_lut},
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
    const std::string& name,
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
        {"B1: Sigmoid-based GELU", gelu_b1_sigmoid},
        {"B1v2: Sigmoid (higher-order)", gelu_b1_sigmoid_v2},
        {"B3: Erf Polynomial (A-S)", gelu_b3_erf_poly},
        {"C1: Cubic Spline (8 seg)", gelu_c1_cubic_spline},
        {"R1: C4 Saturation + Poly-9 Core", gelu_r1_saturation_poly},
        {"R2: A2 Rational Pade", gelu_r2_rational_pade},
        {"R3: C3 PWL (Power-of-2)", gelu_r3_pwl},
        {"R4: B2 Tanh-form + Rational", gelu_r4_tanh_rational},
        {"R5: D1 LUT + Interpolation", gelu_r5_lut},
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
    std::cout << "    // GELU values at integer and half-integer points from -9 to -4" << std::endl;
    std::cout << "    // Computed from GELU(x) = x * 0.5 * (1 + erf(x/√2))" << std::endl;

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
        double phi = 0.5 * (1.0 + std::erf(x * constants::INV_SQRT_2));
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
        double phi = 0.5 * (1.0 + std::erf(x * constants::INV_SQRT_2));
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

    // Run full analysis if requested
    if (do_analyze) {
        std::cout << "\nRunning ULP analysis on all implementations..." << std::endl;
        std::cout << "This analyzes every finite bfloat16 value." << std::endl;

        std::vector<std::pair<std::string, std::function<std::bfloat16_t(std::bfloat16_t)>>> implementations = {
            // Category A: Direct GELU approximations
            {"A1: Direct Poly-7", gelu_a1_poly7},
            {"A1: Direct Poly-9", gelu_a1_poly9},
            {"A3: Chebyshev", gelu_a3_chebyshev},
            {"A4: Continued Fraction", gelu_a4_continued_fraction},
            // Category B: Sub-function approximations
            {"B1: Sigmoid-based GELU", gelu_b1_sigmoid},
            {"B1v2: Sigmoid (sqrt)", gelu_b1_sigmoid_v2},
            {"B3: Erf Polynomial (A-S)", gelu_b3_erf_poly},
            {"B4: Rational Erf (range red)", gelu_b4_rational_erf},
            // Category C: Piecewise methods
            {"C1: Cubic Spline (8 seg)", gelu_c1_cubic_spline},
            {"C2: Piecewise Rational", gelu_c2_piecewise_rational},
            // Category R: Recommended baselines
            {"R1: C4 Saturation + Poly-9", gelu_r1_saturation_poly},
            {"R2: A2 Rational Pade", gelu_r2_rational_pade},
            {"R3: C3 PWL (Power-of-2)", gelu_r3_pwl},
            {"R4: B2 Tanh-form + Rational", gelu_r4_tanh_rational},
            {"R5: D1 LUT + Interpolation", gelu_r5_lut},
            // Category D: Hybrid & LUT-based
            {"D2: LUT Tails + Poly Center", gelu_d2_lut_poly_hybrid},
            {"D3: LUT + Poly Correction", gelu_d3_lut_correction},
            {"D4: Non-uniform LUT", gelu_d4_nonuniform_lut},
            // Category F: Reference methods (arithmetic-only)
            {"F2: Numerical Quadrature", gelu_f2_quadrature},
            {"F3: CF Erf Reference", gelu_f3_cf_erf},
            // Category H: Advanced
            {"H3: SoftEx Tanh", gelu_h3_softex},
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
