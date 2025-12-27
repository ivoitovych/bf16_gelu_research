/**
 * @file debug_tools.cpp
 * @brief Exploratory and debugging tools for BFloat16 GELU research
 *
 * This file contains standalone test code for investigating specific
 * behaviors, debugging implementations, and validating mathematical formulas.
 *
 * Compile: g++ -std=c++17 -O2 -o debug_tools debug_tools.cpp
 * Run: ./debug_tools
 */

#include <iostream>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>

// ============================================================================
// FAST EXP APPROXIMATION (from B3 Pure)
// ============================================================================

inline float fast_exp2_neg(float x) {
    if (x < -126.0f) return 0.0f;
    float n = std::floor(x);
    float f = x - n;
    float f2 = f * f;
    float f3 = f2 * f;
    float f4 = f2 * f2;
    float pow2_frac = 1.0f + 0.6931472f * f + 0.2402265f * f2
                    + 0.0555041f * f3 + 0.0096139f * f4;
    int32_t exp_bits = static_cast<int32_t>(n) + 127;
    if (exp_bits <= 0) return 0.0f;
    uint32_t bits = static_cast<uint32_t>(exp_bits) << 23;
    float pow2_int;
    std::memcpy(&pow2_int, &bits, sizeof(float));
    return pow2_int * pow2_frac;
}

inline float fast_exp_neg(float u) {
    constexpr float inv_ln2 = 1.4426950408889634f;
    return fast_exp2_neg(-u * inv_ln2);
}

// ============================================================================
// DEBUG FUNCTIONS
// ============================================================================

void debug_exp_at_point(float x) {
    float x2_half = x * x * 0.5f;

    std::cout << "\n=== Debug exp at x = " << x << " ===" << std::endl;
    std::cout << "x^2/2 = " << x2_half << std::endl;

    // Reference exp
    float ref_exp = std::exp(-x2_half);
    std::cout << "exp(-x^2/2) reference = " << std::scientific << ref_exp << std::endl;

    // Our approximation
    float our_exp = fast_exp_neg(x2_half);
    std::cout << "exp(-x^2/2) approx    = " << our_exp << std::endl;

    // GELU reference
    float phi = 0.5 * (1.0 + std::erf(x / std::sqrt(2.0)));
    float ref_gelu = x * phi;
    std::cout << "GELU reference        = " << ref_gelu << std::endl;

    // Our asymptotic GELU
    constexpr float inv_sqrt_2pi = 0.3989422804014327f;
    float our_gelu = -our_exp * inv_sqrt_2pi;
    std::cout << "GELU asymptotic       = " << our_gelu << std::endl;

    // Debug exp2 internals
    float arg = -x2_half * 1.4426950408889634f;
    std::cout << "\nexp2_neg internals:" << std::endl;
    std::cout << "  argument to exp2    = " << std::fixed << arg << std::endl;
    std::cout << "  floor(arg)          = " << std::floor(arg) << std::endl;
}

void test_exp_range() {
    std::cout << "\n=== Testing exp approximation across range ===" << std::endl;
    std::cout << std::setw(10) << "x"
              << std::setw(15) << "x^2/2"
              << std::setw(18) << "ref_exp"
              << std::setw(18) << "our_exp"
              << std::setw(12) << "rel_error" << std::endl;

    for (float x = -3.0f; x >= -9.0f; x -= 0.5f) {
        float x2_half = x * x * 0.5f;
        float ref_exp = std::exp(-x2_half);
        float our_exp = fast_exp_neg(x2_half);
        float rel_err = (ref_exp != 0) ? std::abs((our_exp - ref_exp) / ref_exp) : 0;

        std::cout << std::fixed << std::setprecision(2) << std::setw(10) << x
                  << std::setprecision(2) << std::setw(15) << x2_half
                  << std::scientific << std::setprecision(4) << std::setw(18) << ref_exp
                  << std::setw(18) << our_exp
                  << std::fixed << std::setprecision(4) << std::setw(12) << rel_err << std::endl;
    }
}

void test_gelu_asymptotic() {
    std::cout << "\n=== Testing GELU asymptotic expansion ===" << std::endl;
    std::cout << std::setw(8) << "x"
              << std::setw(16) << "ref_gelu"
              << std::setw(16) << "asymp_v1"
              << std::setw(10) << "err_v1"
              << std::setw(16) << "asymp_v2"
              << std::setw(10) << "err_v2" << std::endl;

    constexpr float inv_sqrt_2pi = 0.3989422804014327f;

    for (float x = -3.0f; x >= -9.0f; x -= 0.5f) {
        float x2 = x * x;
        float x2_half = x2 * 0.5f;

        // Reference GELU
        float phi = 0.5 * (1.0 + std::erf(x / std::sqrt(2.0)));
        float ref_gelu = x * phi;

        // Asymptotic v1: GELU(x) ≈ -exp(-x²/2) / √(2π) (leading term only)
        float exp_val = fast_exp_neg(x2_half);
        float phi_x = exp_val * inv_sqrt_2pi;  // φ(x) = exp(-x²/2) / √(2π)
        float asymp_v1 = -phi_x;

        // Asymptotic v2: GELU(x) ≈ -φ(x) * (1 - 1/x² + 3/x⁴ - 15/x⁶)
        // This is the full asymptotic with correction terms
        float inv_x2 = 1.0f / x2;
        float correction = 1.0f - inv_x2 + 3.0f * inv_x2 * inv_x2 - 15.0f * inv_x2 * inv_x2 * inv_x2;
        float asymp_v2 = -phi_x * correction;

        float err_v1 = (ref_gelu != 0) ? std::abs((asymp_v1 - ref_gelu) / ref_gelu) : 0;
        float err_v2 = (ref_gelu != 0) ? std::abs((asymp_v2 - ref_gelu) / ref_gelu) : 0;

        std::cout << std::fixed << std::setprecision(2) << std::setw(8) << x
                  << std::scientific << std::setprecision(3) << std::setw(16) << ref_gelu
                  << std::setw(16) << asymp_v1
                  << std::fixed << std::setprecision(4) << std::setw(10) << err_v1
                  << std::scientific << std::setprecision(3) << std::setw(16) << asymp_v2
                  << std::fixed << std::setprecision(4) << std::setw(10) << err_v2 << std::endl;
    }
}

// Test BFloat16 representation
void test_bf16_representation() {
    std::cout << "\n=== Testing BFloat16 representation at tail ===" << std::endl;
    std::cout << "Comparing naive erf vs erfc-based reference" << std::endl;

    constexpr float inv_sqrt_2pi = 0.3989422804014327f;

    for (float x = -7.0f; x >= -9.0f; x -= 0.125f) {
        float x2 = x * x;
        float x2_half = x2 * 0.5f;

        // Reference GELU using naive formula (has cancellation error!)
        double phi_naive = 0.5 * (1.0 + std::erf(static_cast<double>(x) / std::sqrt(2.0)));
        double ref_gelu_naive = x * phi_naive;

        // Better reference using erfc (avoids cancellation)
        // For x < 0: Φ(x) = 0.5 * erfc(-x/√2)
        double phi_erfc = 0.5 * std::erfc(-static_cast<double>(x) / std::sqrt(2.0));
        double ref_gelu_d = x * phi_erfc;

        // Our asymptotic (float)
        float exp_val = fast_exp_neg(x2_half);
        float phi_x = exp_val * inv_sqrt_2pi;
        float inv_x2 = 1.0f / x2;
        float correction = 1.0f - inv_x2 + 3.0f * inv_x2 * inv_x2 - 15.0f * inv_x2 * inv_x2 * inv_x2;
        float our_gelu = -phi_x * correction;

        // Convert to BFloat16 (simulated as truncation of float32)
        // BFloat16 = sign(1) + exp(8) + mantissa(7)
        // We can simulate by just casting to float and back through truncation
        uint32_t ref_bits, our_bits;
        float ref_f = static_cast<float>(ref_gelu_d);
        std::memcpy(&ref_bits, &ref_f, sizeof(float));
        std::memcpy(&our_bits, &our_gelu, sizeof(float));

        // BFloat16 keeps top 16 bits
        uint16_t ref_bf16 = ref_bits >> 16;
        uint16_t our_bf16 = our_bits >> 16;

        // Also convert naive reference
        uint32_t naive_bits;
        float naive_f = static_cast<float>(ref_gelu_naive);
        std::memcpy(&naive_bits, &naive_f, sizeof(float));
        uint16_t naive_bf16 = naive_bits >> 16;

        // Compute ULP difference vs proper reference
        int ulp_diff = std::abs(static_cast<int>(ref_bf16) - static_cast<int>(our_bf16));
        int ulp_naive = std::abs(static_cast<int>(naive_bf16) - static_cast<int>(our_bf16));

        std::cout << std::fixed << std::setprecision(4) << "x=" << x
                  << std::scientific << std::setprecision(2)
                  << "  erfc=" << ref_gelu_d
                  << "  naive=" << ref_gelu_naive
                  << "  ours=" << our_gelu
                  << std::dec
                  << "  ULP(erfc)=" << ulp_diff
                  << "  ULP(naive)=" << ulp_naive << std::endl;
    }
}

int main(int argc, char* argv[]) {
    std::cout << "BFloat16 GELU Research - Debug Tools" << std::endl;
    std::cout << "=====================================" << std::endl;

    // Default: debug worst point
    debug_exp_at_point(-8.375f);

    // Test range
    test_exp_range();

    // Test GELU asymptotic
    test_gelu_asymptotic();

    // Test BFloat16 representation
    test_bf16_representation();

    return 0;
}
