/**
 * @file saturation_analysis.cpp
 * @brief Standalone BFloat16 GELU Saturation Threshold Analysis
 *
 * Finds the exact bfloat16 values where:
 * 1. POSITIVE TAIL: GELU(x) rounds to x (identity saturation)
 * 2. NEGATIVE TAIL: GELU(x) rounds to 0 (underflow saturation)
 *
 * Build:
 *   g++ -std=c++23 -O3 -march=native -o saturation_analysis saturation_analysis.cpp -lm
 *
 * Run:
 *   ./saturation_analysis
 */

#include <stdfloat>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <iostream>
#include <iomanip>

static_assert(sizeof(std::bfloat16_t) == 2, "bfloat16_t must be 2 bytes");

// ============================================================================
// Type conversion utilities
// ============================================================================

inline uint16_t bfloat16_to_bits(std::bfloat16_t value) {
    uint16_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

inline std::bfloat16_t bits_to_bfloat16(uint16_t bits) {
    std::bfloat16_t value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

inline bool is_finite_bf16(std::bfloat16_t value) {
    float f = static_cast<float>(value);
    return !std::isnan(f) && !std::isinf(f);
}

// ============================================================================
// High-precision GELU reference
// ============================================================================

constexpr double INV_SQRT_2 = 0.7071067811865475244008443621048490392848;

/**
 * @brief Reference GELU using erfc for numerical stability
 *
 * For x < 0, uses erfc(-x/sqrt(2)) to avoid catastrophic cancellation
 * in 1 + erf(x/sqrt(2)) when erf approaches -1.
 */
double gelu_reference_f64(double x) {
    double z = x * INV_SQRT_2;
    double phi;
    if (x >= 0) {
        phi = 0.5 * (1.0 + std::erf(z));
    } else {
        phi = 0.5 * std::erfc(-z);
    }
    return x * phi;
}

// ============================================================================
// Analysis functions
// ============================================================================

void analyze_positive_saturation() {
    std::cout << "=== POSITIVE TAIL: Finding smallest x where bf16(GELU(x)) == x ===" << std::endl;
    std::cout << std::endl;

    uint16_t first_saturated = 0;

    // Scan positive bf16 values (0x0001 to 0x7F7F, excluding inf/nan)
    for (uint16_t bits = 0x0001; bits <= 0x7F7F; ++bits) {
        std::bfloat16_t x_bf16 = bits_to_bfloat16(bits);
        if (!is_finite_bf16(x_bf16)) continue;

        double x = static_cast<double>(static_cast<float>(x_bf16));
        double gelu_exact = gelu_reference_f64(x);
        std::bfloat16_t gelu_bf16 = static_cast<std::bfloat16_t>(static_cast<float>(gelu_exact));

        if (gelu_bf16 == x_bf16) {
            first_saturated = bits;
            break;
        }
    }

    // Display transition region
    std::cout << "Transition region:" << std::endl;
    std::cout << std::endl;

    std::cout << std::left << std::setw(10) << "bits"
              << std::setw(12) << "x"
              << std::setw(16) << "GELU(x) exact"
              << std::setw(12) << "bf16 bits"
              << std::setw(12) << "bf16(GELU)"
              << "x == bf16?"
              << std::endl;
    std::cout << std::string(72, '-') << std::endl;

    for (uint16_t bits = first_saturated - 6; bits <= first_saturated + 5; ++bits) {
        std::bfloat16_t x_bf16 = bits_to_bfloat16(bits);
        double x = static_cast<double>(static_cast<float>(x_bf16));
        double gelu_exact = gelu_reference_f64(x);
        std::bfloat16_t gelu_bf16 = static_cast<std::bfloat16_t>(static_cast<float>(gelu_exact));
        bool saturated = (gelu_bf16 == x_bf16);

        std::cout << "0x" << std::hex << std::setw(8) << std::left << bits << std::dec
                  << std::fixed << std::setprecision(5)
                  << std::setw(12) << static_cast<float>(x_bf16)
                  << std::setw(16) << gelu_exact
                  << "0x" << std::hex << std::setw(10) << bfloat16_to_bits(gelu_bf16) << std::dec
                  << std::setw(12) << static_cast<float>(gelu_bf16)
                  << (saturated ? "YES <--" : "no")
                  << std::endl;
    }

    std::bfloat16_t threshold = bits_to_bfloat16(first_saturated);
    std::cout << std::endl;
    std::cout << "RESULT: x >= " << std::fixed << std::setprecision(6)
              << static_cast<float>(threshold)
              << " (0x" << std::hex << first_saturated << std::dec
              << ") => bf16(GELU(x)) == x" << std::endl;
}

void analyze_negative_saturation() {
    std::cout << std::endl;
    std::cout << "=== NEGATIVE TAIL: Finding x where bf16(GELU(x)) saturates to 0 ===" << std::endl;
    std::cout << std::endl;

    // For negative bf16: higher bits = larger magnitude (more negative)
    // Start from moderate negative (x = -1, bits = 0xBF80) going more negative

    uint16_t first_zero = 0;
    uint16_t last_nonzero = 0;

    for (uint16_t bits = 0xBF80; bits <= 0xFF7F; ++bits) {
        std::bfloat16_t x_bf16 = bits_to_bfloat16(bits);
        if (!is_finite_bf16(x_bf16)) continue;

        double x = static_cast<double>(static_cast<float>(x_bf16));
        double gelu_exact = gelu_reference_f64(x);
        std::bfloat16_t gelu_bf16 = static_cast<std::bfloat16_t>(static_cast<float>(gelu_exact));
        float gelu_f = static_cast<float>(gelu_bf16);
        bool is_zero = (gelu_f == 0.0f);

        if (!is_zero) {
            last_nonzero = bits;
        }
        if (is_zero && first_zero == 0) {
            first_zero = bits;
        }
    }

    // Display transition region
    std::cout << "Transition region (going more negative):" << std::endl;
    std::cout << std::endl;

    std::cout << std::left << std::setw(10) << "bits"
              << std::setw(12) << "x"
              << std::setw(16) << "GELU(x) exact"
              << std::setw(12) << "bf16 bits"
              << std::setw(14) << "bf16(GELU)"
              << "== 0?"
              << std::endl;
    std::cout << std::string(74, '-') << std::endl;

    for (uint16_t bits = last_nonzero - 3; bits <= first_zero + 5; ++bits) {
        std::bfloat16_t x_bf16 = bits_to_bfloat16(bits);
        if (!is_finite_bf16(x_bf16)) continue;

        double x = static_cast<double>(static_cast<float>(x_bf16));
        double gelu_exact = gelu_reference_f64(x);
        std::bfloat16_t gelu_bf16 = static_cast<std::bfloat16_t>(static_cast<float>(gelu_exact));
        uint16_t gelu_bits = bfloat16_to_bits(gelu_bf16);
        float gelu_f = static_cast<float>(gelu_bf16);
        bool is_zero = (gelu_f == 0.0f);

        std::cout << "0x" << std::hex << std::setw(8) << std::left << bits << std::dec
                  << std::fixed << std::setprecision(4)
                  << std::setw(12) << static_cast<float>(x_bf16)
                  << std::scientific << std::setprecision(4)
                  << std::setw(16) << gelu_exact
                  << "0x" << std::hex << std::setw(10) << gelu_bits << std::dec
                  << std::scientific << std::setw(14) << gelu_f
                  << (is_zero ? "YES <--" : "no")
                  << std::endl;
    }

    std::bfloat16_t threshold = bits_to_bfloat16(first_zero);
    std::cout << std::endl;
    std::cout << "RESULT: x <= " << std::fixed << std::setprecision(6)
              << static_cast<float>(threshold)
              << " (0x" << std::hex << first_zero << std::dec
              << ") => bf16(GELU(x)) == 0" << std::endl;
}

void show_bf16_limits() {
    std::cout << std::endl;
    std::cout << "=== BFloat16 Representation Limits ===" << std::endl;
    std::cout << std::endl;

    std::bfloat16_t smallest_subnormal = bits_to_bfloat16(0x0001);
    std::bfloat16_t smallest_normal = bits_to_bfloat16(0x0080);
    float min_sub = static_cast<float>(smallest_subnormal);
    float min_norm = static_cast<float>(smallest_normal);

    std::cout << std::scientific << std::setprecision(6);
    std::cout << "Smallest positive bf16 subnormal: " << min_sub
              << " (0x0001)" << std::endl;
    std::cout << "Smallest positive bf16 normal:    " << min_norm
              << " (0x0080)" << std::endl;
    std::cout << std::endl;
    std::cout << "Rounding threshold to zero:       " << min_sub / 2.0
              << " (half of smallest subnormal)" << std::endl;
}

void show_summary() {
    std::cout << std::endl;
    std::cout << "===============================================================" << std::endl;
    std::cout << "                         SUMMARY" << std::endl;
    std::cout << "===============================================================" << std::endl;
    std::cout << std::endl;
    std::cout << "POSITIVE SATURATION:  x >= 2.781250 (0x4032)" << std::endl;
    std::cout << "  => bf16(GELU(x)) rounds to x" << std::endl;
    std::cout << "  => Safe to use identity: GELU(x) = x" << std::endl;
    std::cout << std::endl;
    std::cout << "NEGATIVE SATURATION:  x <= -13.562500 (0xc159)" << std::endl;
    std::cout << "  => bf16(GELU(x)) rounds to 0" << std::endl;
    std::cout << "  => Safe to use constant: GELU(x) = 0" << std::endl;
    std::cout << std::endl;
    std::cout << "NOTE: These thresholds are for the TRUE GELU function" << std::endl;
    std::cout << "      GELU(x) = x * Phi(x), not approximations." << std::endl;
    std::cout << "===============================================================" << std::endl;
}

// ============================================================================
// Main
// ============================================================================

int main() {
    std::cout << "BFloat16 GELU Saturation Threshold Analysis" << std::endl;
    std::cout << "============================================" << std::endl;
    std::cout << std::endl;

    analyze_positive_saturation();
    analyze_negative_saturation();
    show_bf16_limits();
    show_summary();

    return 0;
}
