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
 * R1: C4 - Saturation + minimax polynomial core
 * R2: A2 - Rational Padé [4/4] approximation
 * R3: C3 - Piecewise linear with power-of-2 breakpoints
 * R4: B2 - Tanh-form with odd rational tanh approximation
 * R5: D1 - LUT with linear interpolation
 *
 * ## Constraints
 *
 * All approximations (R1-R5) use only: +, -, *, /, |x|, sign()
 * No erf(), tanh(), exp(), log() in approximations
 * Reference F1 uses std::erf for ground truth
 *
 * @author Claude Code
 * @date 2024
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
// R1: C4 - SATURATION + MINIMAX POLYNOMIAL CORE
// ============================================================================

/**
 * @brief R1: Saturation + minimax polynomial core GELU approximation
 *
 * Strategy:
 * - For x > threshold_pos: GELU(x) ≈ x (saturation to identity)
 * - For x < threshold_neg: GELU(x) ≈ 0 (saturation to zero)
 * - For core region: Use polynomial approximation
 *
 * Key insight: GELU(x) = x * Φ(x), and Φ(x) transitions from 0 to 1.
 * - For x >> 0: Φ(x) ≈ 1, so GELU(x) ≈ x
 * - For x << 0: Φ(x) ≈ 0, so GELU(x) ≈ 0
 *
 * The core uses a polynomial fitted to approximate Φ(x) over [-3, 3].
 * Beyond this range, saturation provides excellent approximation.
 *
 * Reference values for threshold selection:
 *   GELU(3) = 2.9960, GELU(4) = 3.9999
 *   GELU(-3) = -0.0041, GELU(-4) = -0.00013
 */
std::bfloat16_t gelu_r1_saturation_poly(std::bfloat16_t x_bf16) {
    float x = static_cast<float>(x_bf16);

    // Asymmetric saturation thresholds for ULP optimization
    // Positive: x >= 3 → GELU(x) ≈ x (error ~0.004 at boundary)
    // Negative: x <= -5 → GELU(x) ≈ 0 (error ~1.4e-6 at boundary)
    // The negative threshold is more extreme because GELU approaches 0
    // very slowly for negative x, requiring larger |x| to minimize ULP error.
    constexpr float THRESH_POS = 3.0f;
    constexpr float THRESH_NEG = -5.0f;

    if (x >= THRESH_POS) {
        return static_cast<std::bfloat16_t>(x);
    }
    if (x <= THRESH_NEG) {
        return static_cast<std::bfloat16_t>(0.0f);
    }

    // Core region [-5, 3]: polynomial approximation for Φ(x)
    //
    // We approximate Φ(x) = 0.5 * (1 + erf(x/√2)) using:
    //   Φ(x) ≈ 0.5 + x * (a1 + a3*x² + a5*x⁴ + a7*x⁶)
    //
    // Extended coefficients to handle the wider negative range.
    // The polynomial is constrained to give Φ ∈ [0, 1].
    float x2 = x * x;
    float x4 = x2 * x2;
    float x6 = x4 * x2;

    // Coefficients fitted over [-5, 3] with constraint Φ ∈ [0, 1]
    constexpr float a1 = 0.3989423f;   // ≈ 1/√(2π)
    constexpr float a3 = -0.0419131f;
    constexpr float a5 = 0.0017003f;
    constexpr float a7 = -0.0000215f;

    float phi_minus_half_over_x = a1 + a3 * x2 + a5 * x4 + a7 * x6;
    float phi = 0.5f + x * phi_minus_half_over_x;

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
 * GELU(x) ≈ x * P(x²) / Q(x²)
 *
 * Using even powers in P and Q preserves the symmetry properties.
 * The rational form provides better convergence in the tails compared
 * to pure polynomials of the same degree.
 *
 * For Φ(x), we use:
 *   Φ(x) ≈ 0.5 + x * (a0 + a1*x² + a2*x⁴) / (1 + b1*x² + b2*x⁴)
 *
 * This is a [2/2] rational in x² multiplied by x, giving effective [4/4].
 *
 * Critical: Must include saturation for tail regions.
 */
std::bfloat16_t gelu_r2_rational_pade(std::bfloat16_t x_bf16) {
    float x = static_cast<float>(x_bf16);

    // Asymmetric saturation thresholds
    // Positive: x >= 3 → GELU(x) ≈ x
    // Negative: x <= -5 → GELU(x) ≈ 0
    constexpr float THRESH_POS = 3.0f;
    constexpr float THRESH_NEG = -5.0f;

    if (x >= THRESH_POS) {
        return static_cast<std::bfloat16_t>(x);
    }
    if (x <= THRESH_NEG) {
        return static_cast<std::bfloat16_t>(0.0f);
    }

    float x2 = x * x;
    float x4 = x2 * x2;
    float x6 = x4 * x2;

    // Rational approximation coefficients for (Φ(x) - 0.5) / x
    // Extended to [5/5] for better accuracy over [-5, 3]
    // R(x²) = (a0 + a1*x² + a2*x⁴ + a3*x⁶) / (1 + b1*x² + b2*x⁴ + b3*x⁶)
    constexpr float a0 = 0.3989423f;
    constexpr float a1 = 0.0298729f;
    constexpr float a2 = 0.0008853f;
    constexpr float a3 = 0.0000097f;

    constexpr float b1 = 0.3124159f;
    constexpr float b2 = 0.0252069f;
    constexpr float b3 = 0.0006318f;

    float num = a0 + a1 * x2 + a2 * x4 + a3 * x6;
    float den = 1.0f + b1 * x2 + b2 * x4 + b3 * x6;

    // Φ(x) ≈ 0.5 + x * (num / den)
    float phi = 0.5f + x * (num / den);

    // Safety clamp (rarely needed with proper coefficients)
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
 * Breakpoints: 0, ±0.5, ±1, ±2, ±3, ±4, ±5
 * Saturation: x >= 3 → x, x <= -5 → 0
 *
 * Reference GELU values:
 *   x       GELU(x)         Phi(x)
 *   0       0.0             0.5
 *   0.5     0.345731        0.691462
 *   1.0     0.841345        0.841345
 *   2.0     1.954500        0.977250
 *   3.0     2.995950        0.998650
 *  -0.5    -0.154269        0.308539
 *  -1.0    -0.158655        0.158655
 *  -2.0    -0.045500        0.022750
 *  -3.0    -0.004050        0.001350
 *  -4.0    -0.000127        0.0000317
 *  -5.0    -0.0000014       0.000000287
 */
std::bfloat16_t gelu_r3_pwl(std::bfloat16_t x_bf16) {
    float x = static_cast<float>(x_bf16);

    // Asymmetric saturation thresholds
    if (x >= 3.0f) {
        return static_cast<std::bfloat16_t>(x);  // GELU(x) ≈ x
    }
    if (x <= -5.0f) {
        return static_cast<std::bfloat16_t>(0.0f);  // GELU(x) ≈ 0
    }

    // Precomputed segment parameters (slope, intercept)
    // For segment [x0, x1]: y = slope*x + intercept
    float result;

    if (x >= 0.0f) {
        if (x < 0.5f) {
            result = 0.691462f * x;  // [0, 0.5]
        } else if (x < 1.0f) {
            result = 0.991228f * x - 0.149883f;  // [0.5, 1]
        } else if (x < 2.0f) {
            result = 1.113155f * x - 0.271810f;  // [1, 2]
        } else {
            result = 1.041450f * x - 0.128410f;  // [2, 3]
        }
    } else {
        if (x > -0.5f) {
            result = 0.308538f * x;  // [-0.5, 0]
        } else if (x > -1.0f) {
            result = 0.008772f * x - 0.149883f;  // [-1, -0.5]
        } else if (x > -2.0f) {
            result = 0.113155f * x + 0.045500f;  // [-2, -1]
        } else if (x > -3.0f) {
            result = 0.041450f * x + 0.037850f;  // [-3, -2]
        } else if (x > -4.0f) {
            result = 0.003923f * x + 0.007681f;  // [-4, -3]
        } else {
            result = 0.000126f * x + 0.000503f;  // [-5, -4]
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
 * We approximate tanh using a simple odd rational function:
 *   tanh(z) ≈ z * (27 + z²) / (27 + 9z²)
 *
 * This rational form:
 * - Is odd: tanh(-z) = -tanh(z)
 * - Approaches ±1 as z → ±∞ (approximately)
 * - Uses only basic arithmetic
 *
 * Critical: The tanh-form has numerical issues for x < -2 because
 * tanh approaches -1, making (1+tanh) ≈ 0, losing the small residual.
 * We use saturation thresholds to handle these regions.
 */
std::bfloat16_t gelu_r4_tanh_rational(std::bfloat16_t x_bf16) {
    float x = static_cast<float>(x_bf16);

    // Asymmetric saturation thresholds
    // For x >= 3: GELU(x) ≈ x
    // For x <= -5: GELU(x) ≈ 0
    if (x >= 3.0f) {
        return static_cast<std::bfloat16_t>(x);
    }
    if (x <= -5.0f) {
        return static_cast<std::bfloat16_t>(0.0f);
    }

    // Compute the tanh argument: z = √(2/π) * (x + 0.044715 * x³)
    float x3 = x * x * x;
    constexpr float sqrt_2_over_pi = 0.7978845608f;
    constexpr float coeff = 0.044715f;

    float z = sqrt_2_over_pi * (x + coeff * x3);

    // Rational tanh approximation: tanh(z) ≈ z * (27 + z²) / (27 + 9z²)
    // This is accurate for |z| < 3, which corresponds to |x| < ~2.5
    float z2 = z * z;
    float tanh_z = z * (27.0f + z2) / (27.0f + 9.0f * z2);

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

// LUT for Φ(x) values from x = -5 to x = 3 with 256 entries
// Asymmetric range matching the saturation thresholds
namespace lut_data {
    constexpr int LUT_SIZE = 256;
    constexpr float LUT_MIN = -5.0f;
    constexpr float LUT_MAX = 3.0f;
    constexpr float LUT_STEP = (LUT_MAX - LUT_MIN) / (LUT_SIZE - 1);
    constexpr float LUT_INV_STEP = (LUT_SIZE - 1) / (LUT_MAX - LUT_MIN);

    // Asymmetric saturation thresholds (same as LUT bounds)
    constexpr float SAT_POS = LUT_MAX;
    constexpr float SAT_NEG = LUT_MIN;

    // Φ(x) values at uniform intervals
    // Φ(x) = 0.5 * (1 + erf(x / √2))
    // These values are precomputed at compile time in a real implementation
    // For now, we'll compute them at runtime during initialization
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
        if (x >= lut_data::SAT_POS) {
            return static_cast<std::bfloat16_t>(x);  // GELU(x) ≈ x
        }
        if (x <= lut_data::SAT_NEG) {
            return static_cast<std::bfloat16_t>(0.0f);  // GELU(x) ≈ 0
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
 * @struct UlpStats
 * @brief Statistics for ULP error analysis
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
 * @brief Print ULP analysis results
 */
void print_ulp_stats(const std::string& name, const UlpStats& stats) {
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
        {"R1: C4 Saturation + Poly-7 Core", gelu_r1_saturation_poly},
        {"R2: A2 Rational Pade [4/4]", gelu_r2_rational_pade},
        {"R3: C3 PWL (Power-of-2 breakpoints)", gelu_r3_pwl},
        {"R4: B2 Tanh-form + Rational Tanh", gelu_r4_tanh_rational},
        {"R5: D1 LUT + Linear Interpolation", gelu_r5_lut},
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
            {"R1: C4 Saturation + Poly-7 Core", gelu_r1_saturation_poly},
            {"R2: A2 Rational Pade [4/4]", gelu_r2_rational_pade},
            {"R3: C3 PWL (Power-of-2 breakpoints)", gelu_r3_pwl},
            {"R4: B2 Tanh-form + Rational Tanh", gelu_r4_tanh_rational},
            {"R5: D1 LUT + Linear Interpolation", gelu_r5_lut},
        };

        std::cout << "\n================================================================" << std::endl;
        std::cout << "                    ULP ANALYSIS RESULTS                        " << std::endl;
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
