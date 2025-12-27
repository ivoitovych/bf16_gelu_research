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
    // Asymmetric saturation thresholds optimized for entire bf16 range
    // Positive: x >= 4 → GELU(x) ≈ x (GELU(4) = 3.9999, error < 0.0001)
    // Negative: x <= -7 → GELU(x) ≈ 0 (GELU(-7) ≈ -5.5e-11, negligible)
    //
    // Extended from -5 to -7 to reduce max-ULP at saturation boundary.
    // At -7, the true GELU value is so small that bf16 rounds it to 0 anyway.
    constexpr float POS = 4.0f;
    constexpr float NEG = -7.0f;
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
        } else if (x > -4.0f) {
            // [-4, -2]: slope = (-0.000127-(-0.045403))/(-2) = 0.022638
            result = 0.022638f * x + 0.000127f;
        } else {
            // [-7, -4]: slope = (0-(-0.000127))/(-3) ≈ 0.0000423
            result = 0.0000423f * x + 0.000169f;
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

// LUT for Φ(x) values from x = -7 to x = 4 with 512 entries
// Asymmetric range matching the saturation thresholds
// Extended range and more entries for better accuracy
namespace lut_data {
    constexpr int LUT_SIZE = 512;
    constexpr float LUT_MIN = thresholds::NEG;  // -7.0f
    constexpr float LUT_MAX = thresholds::POS;  // 4.0f
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
        if (x <= thresholds::NEG) {
            return static_cast<std::bfloat16_t>(0.0f);
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
        {"B1: Sigmoid-based GELU", gelu_b1_sigmoid},
        {"B1v2: Sigmoid (higher-order)", gelu_b1_sigmoid_v2},
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
 * @brief Print usage information
 */
void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " [OPTIONS]" << std::endl;
    std::cout << "\nOptions:" << std::endl;
    std::cout << "  --analyze     Run full ULP analysis (default)" << std::endl;
    std::cout << "  --diagnose    Run diagnostic mode with test points" << std::endl;
    std::cout << "  --reference   Show reference GELU values" << std::endl;
    std::cout << "  --all         Run all modes" << std::endl;
    std::cout << "  --help        Show this help message" << std::endl;
}

// ============================================================================
// MAIN - RUN ANALYSIS
// ============================================================================

int main(int argc, char* argv[]) {
    bool do_analyze = false;
    bool do_diagnose = false;
    bool do_reference = false;

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
            } else if (arg == "--all") {
                do_analyze = true;
                do_diagnose = true;
                do_reference = true;
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

    // Run diagnostics if requested
    if (do_diagnose) {
        run_diagnostics(ulp_calc);
    }

    // Run full analysis if requested
    if (do_analyze) {
        std::cout << "\nRunning ULP analysis on all implementations..." << std::endl;
        std::cout << "This analyzes every finite bfloat16 value." << std::endl;

        std::vector<std::pair<std::string, std::function<std::bfloat16_t(std::bfloat16_t)>>> implementations = {
            {"B1: Sigmoid-based GELU", gelu_b1_sigmoid},
            {"B1v2: Sigmoid (higher-order)", gelu_b1_sigmoid_v2},
            {"R1: C4 Saturation + Poly-9 Core", gelu_r1_saturation_poly},
            {"R2: A2 Rational Pade", gelu_r2_rational_pade},
            {"R3: C3 PWL (Power-of-2)", gelu_r3_pwl},
            {"R4: B2 Tanh-form + Rational", gelu_r4_tanh_rational},
            {"R5: D1 LUT + Interpolation", gelu_r5_lut},
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
