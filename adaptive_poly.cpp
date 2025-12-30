// Adaptive Piecewise Polynomial GELU Approximation - Research Tool
//
// Implements error-driven knot placement for bf16 GELU approximation:
// 1. Divide [-13.6, 3] into N segments with polynomial degree M
// 2. Fit coefficients via least-squares or minimax
// 3. Measure max ULP error per segment
// 4. Iteratively adjust segment boundaries to equalize error
//
// Build: g++ -std=c++23 -O3 -march=native -o adaptive_poly adaptive_poly.cpp -lm
// Usage: ./adaptive_poly [--segments N] [--degree M] [--iterations I]

#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <array>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <iomanip>
#include <limits>
#include <cassert>
#include <stdfloat>

// ============================================================================
// Configuration
// ============================================================================

struct Config {
    int num_segments = 32;       // Number of polynomial segments
    int poly_degree = 5;         // Polynomial degree (0 to poly_degree, so poly_degree+1 coeffs)
    int max_iterations = 50;     // Max optimization iterations
    double convergence_threshold = 0.1;  // Stop when max/min ULP ratio < 1 + threshold
    bool verbose = false;

    // Region bounds (bf16 non-saturated GELU region)
    double x_min = -13.5625;     // Last bf16 value before GELU saturates to 0
    double x_max = 3.0;          // Conservative positive saturation
};

// ============================================================================
// BFloat16 Utilities
// ============================================================================

using bfloat16_t = std::bfloat16_t;

inline uint16_t to_bits(bfloat16_t x) {
    uint16_t bits;
    std::memcpy(&bits, &x, sizeof(bits));
    return bits;
}

inline bfloat16_t from_bits(uint16_t bits) {
    bfloat16_t x;
    std::memcpy(&x, &bits, sizeof(x));
    return x;
}

inline bfloat16_t float_to_bf16(float x) {
    return static_cast<bfloat16_t>(x);
}

inline float bf16_to_float(bfloat16_t x) {
    return static_cast<float>(x);
}

// Build sorted list of all valid bf16 values for ULP calculation
std::vector<uint16_t> build_bf16_index() {
    std::vector<uint16_t> index;
    index.reserve(65536);

    // Positive values (including +0)
    for (uint32_t i = 0; i < 0x8000; ++i) {
        bfloat16_t val = from_bits(static_cast<uint16_t>(i));
        if (!std::isnan(bf16_to_float(val)) && !std::isinf(bf16_to_float(val))) {
            index.push_back(static_cast<uint16_t>(i));
        }
    }

    // Negative values (excluding -0, which maps to +0)
    for (uint32_t i = 0x8001; i < 0x10000; ++i) {
        bfloat16_t val = from_bits(static_cast<uint16_t>(i));
        if (!std::isnan(bf16_to_float(val)) && !std::isinf(bf16_to_float(val))) {
            index.push_back(static_cast<uint16_t>(i));
        }
    }

    // Sort by actual float value
    std::sort(index.begin(), index.end(), [](uint16_t a, uint16_t b) {
        return bf16_to_float(from_bits(a)) < bf16_to_float(from_bits(b));
    });

    return index;
}

// Global bf16 index for ULP calculations
static std::vector<uint16_t> g_bf16_index;
static std::vector<int64_t> g_bits_to_index(65536, -1);

void init_bf16_tables() {
    if (!g_bf16_index.empty()) return;

    g_bf16_index = build_bf16_index();
    for (size_t i = 0; i < g_bf16_index.size(); ++i) {
        g_bits_to_index[g_bf16_index[i]] = static_cast<int64_t>(i);
    }
    // Map -0 to same index as +0
    g_bits_to_index[0x8000] = g_bits_to_index[0x0000];
}

int64_t ulp_distance(bfloat16_t a, bfloat16_t b) {
    uint16_t bits_a = to_bits(a);
    uint16_t bits_b = to_bits(b);

    int64_t idx_a = g_bits_to_index[bits_a];
    int64_t idx_b = g_bits_to_index[bits_b];

    if (idx_a < 0 || idx_b < 0) return INT64_MAX;  // NaN or Inf

    return std::abs(idx_a - idx_b);
}

// ============================================================================
// Reference GELU (fp64, using erfc to avoid cancellation)
// ============================================================================

constexpr double INV_SQRT_2 = 0.7071067811865475244;

double gelu_reference_f64(double x) {
    double z = x * INV_SQRT_2;
    double phi;
    if (x >= 0) {
        phi = 0.5 * (1.0 + std::erf(z));
    } else {
        // Use erfc(-z) to avoid catastrophic cancellation when erf(z) ≈ -1
        phi = 0.5 * std::erfc(-z);
    }
    return x * phi;
}

bfloat16_t gelu_reference_bf16(float x) {
    double result = gelu_reference_f64(static_cast<double>(x));
    return float_to_bf16(static_cast<float>(result));
}

// ============================================================================
// Polynomial Representation
// ============================================================================

struct Polynomial {
    std::vector<double> coeffs;  // coeffs[i] is coefficient of x^i

    Polynomial(int degree = 0) : coeffs(degree + 1, 0.0) {}

    // Evaluate at x using Horner's method
    double eval(double x) const {
        double result = 0.0;
        for (int i = static_cast<int>(coeffs.size()) - 1; i >= 0; --i) {
            result = result * x + coeffs[i];
        }
        return result;
    }

    // Evaluate in float32 (for implementation)
    float eval_f32(float x) const {
        float result = 0.0f;
        for (int i = static_cast<int>(coeffs.size()) - 1; i >= 0; --i) {
            result = result * x + static_cast<float>(coeffs[i]);
        }
        return result;
    }
};

// ============================================================================
// Segment with Polynomial Fit
// ============================================================================

struct Segment {
    double x_start;
    double x_end;
    Polynomial poly;

    // Coordinate transformation: u = (x - x_mid) / x_scale maps [x_start, x_end] to [-1, 1]
    double x_mid = 0.0;
    double x_scale = 1.0;

    // Error statistics (computed after fitting)
    int64_t max_ulp = 0;
    double mean_ulp = 0.0;
    int sample_count = 0;
    float worst_x = 0.0f;

    // Evaluate polynomial using stored scaling
    float eval_f32(float x) const {
        float u = (x - static_cast<float>(x_mid)) / static_cast<float>(x_scale);
        return poly.eval_f32(u);
    }
};

// ============================================================================
// Least-Squares Polynomial Fitting
// ============================================================================

// Simple Gaussian elimination for solving linear systems
bool solve_linear_system(std::vector<std::vector<double>>& A, std::vector<double>& b) {
    int n = static_cast<int>(b.size());

    // Forward elimination with partial pivoting
    for (int i = 0; i < n; ++i) {
        // Find pivot
        int max_row = i;
        for (int k = i + 1; k < n; ++k) {
            if (std::abs(A[k][i]) > std::abs(A[max_row][i])) {
                max_row = k;
            }
        }
        std::swap(A[i], A[max_row]);
        std::swap(b[i], b[max_row]);

        if (std::abs(A[i][i]) < 1e-12) {
            return false;  // Singular matrix
        }

        // Eliminate column
        for (int k = i + 1; k < n; ++k) {
            double factor = A[k][i] / A[i][i];
            for (int j = i; j < n; ++j) {
                A[k][j] -= factor * A[i][j];
            }
            b[k] -= factor * b[i];
        }
    }

    // Back substitution
    for (int i = n - 1; i >= 0; --i) {
        for (int j = i + 1; j < n; ++j) {
            b[i] -= A[i][j] * b[j];
        }
        b[i] /= A[i][i];
    }

    return true;
}

// Fit polynomial to GELU using least squares on dense samples
// Returns polynomial in scaled coordinates u = (x - x_mid) / x_scale
// Also outputs scaling parameters
Polynomial fit_polynomial_lsq(double x_start, double x_end, int degree,
                               double& out_x_mid, double& out_x_scale,
                               int num_samples = 200) {
    int n = degree + 1;  // Number of coefficients

    // Shift and scale to [-1, 1] for numerical stability
    out_x_mid = (x_start + x_end) / 2.0;
    out_x_scale = (x_end - x_start) / 2.0;
    if (out_x_scale < 1e-10) out_x_scale = 1.0;  // Prevent division by zero

    // Generate sample points
    std::vector<double> us(num_samples);  // Scaled coordinates
    std::vector<double> ys(num_samples);

    for (int i = 0; i < num_samples; ++i) {
        double t = static_cast<double>(i) / (num_samples - 1);
        double x = x_start + t * (x_end - x_start);
        us[i] = (x - out_x_mid) / out_x_scale;  // Map to [-1, 1]
        ys[i] = gelu_reference_f64(x);
    }

    // Build normal equations in scaled coordinates
    std::vector<std::vector<double>> ATA(n, std::vector<double>(n, 0.0));
    std::vector<double> ATy(n, 0.0);

    for (int i = 0; i < num_samples; ++i) {
        std::vector<double> powers(n);
        powers[0] = 1.0;
        for (int j = 1; j < n; ++j) {
            powers[j] = powers[j-1] * us[i];
        }

        for (int j = 0; j < n; ++j) {
            for (int k = 0; k < n; ++k) {
                ATA[j][k] += powers[j] * powers[k];
            }
            ATy[j] += powers[j] * ys[i];
        }
    }

    // Solve the system
    if (!solve_linear_system(ATA, ATy)) {
        std::cerr << "Warning: Singular matrix in polynomial fitting\n";
    }

    // Return polynomial in scaled coordinates (no conversion needed)
    Polynomial poly(degree);
    poly.coeffs = ATy;
    return poly;
}

// ============================================================================
// Minimax Refinement (Remez-like iteration)
// ============================================================================

// Simple minimax refinement: perturb coefficients to reduce max error
// Works in scaled coordinates u = (x - x_mid) / x_scale
Polynomial refine_minimax(const Polynomial& initial, double x_start, double x_end,
                          double x_mid, double x_scale,
                          int iterations = 10, int num_samples = 500) {
    Polynomial poly = initial;
    int n = static_cast<int>(poly.coeffs.size());

    // Sample points for error evaluation (in scaled coordinates)
    std::vector<double> us(num_samples);
    std::vector<double> ys(num_samples);

    for (int i = 0; i < num_samples; ++i) {
        double t = static_cast<double>(i) / (num_samples - 1);
        double x = x_start + t * (x_end - x_start);
        us[i] = (x - x_mid) / x_scale;
        ys[i] = gelu_reference_f64(x);
    }

    for (int iter = 0; iter < iterations; ++iter) {
        // Find max error point
        double max_err = 0.0;
        int max_idx = 0;
        double max_err_signed = 0.0;

        for (int i = 0; i < num_samples; ++i) {
            double err = poly.eval(us[i]) - ys[i];
            if (std::abs(err) > max_err) {
                max_err = std::abs(err);
                max_idx = i;
                max_err_signed = err;
            }
        }

        if (max_err < 1e-10) break;

        // Adjust coefficients to reduce error at max point
        // Gradient descent step on coefficient that has most effect
        double u_max = us[max_idx];
        double power = 1.0;
        int best_coeff = 0;
        double best_effect = 0.0;

        for (int j = 0; j < n; ++j) {
            double effect = std::abs(power);
            if (effect > best_effect) {
                best_effect = effect;
                best_coeff = j;
            }
            power *= u_max;
        }

        // Small step to reduce error
        double step = 0.1 * max_err_signed;
        power = 1.0;
        for (int j = 0; j < best_coeff; ++j) power *= u_max;

        if (std::abs(power) > 1e-10) {
            poly.coeffs[best_coeff] -= step / power * 0.5;
        }
    }

    return poly;
}

// ============================================================================
// ULP Error Measurement for a Segment
// ============================================================================

// Near-zero threshold: use Taylor expansion for |x| < this value
// GELU(x) ≈ x * (0.5 + x/√(2π)) for small x
constexpr float NEAR_ZERO_THRESHOLD = 0.125f;
constexpr float INV_SQRT_2PI_F = 0.3989422804f;

// Deep tail threshold: use asymptotic expansion for x < this value
constexpr float DEEP_TAIL_THRESHOLD = -3.5f;

// Fast 2^x for negative x, with proper subnormal handling
// Matches the implementation in gelu_implementations.cpp
float fast_exp2_neg(float x) {
    // For very negative x beyond float subnormal range, underflow to zero
    // Float subnormals go down to 2^(-149), so we cut off there
    if (x < -149.0f) return 0.0f;

    // Split into integer and fractional parts
    float n = std::floor(x);
    float f = x - n;  // f is in [0, 1)

    // Minimax polynomial for 2^f on [0, 1]
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
        // Compute 2^(-126) first (smallest normal)
        uint32_t min_normal_bits = 1u << 23;  // 2^(-126)
        float min_normal;
        std::memcpy(&min_normal, &min_normal_bits, sizeof(float));

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

// Fast exp(-u) for u > 0: exp(-u) = 2^(-u/ln(2))
float fast_exp_neg(float u) {
    constexpr float INV_LN2 = 1.4426950408889634f;  // 1/ln(2)
    return fast_exp2_neg(-u * INV_LN2);
}

// Asymptotic expansion for deep negative tail
// GELU(x) ≈ -φ(x) * (1 - 1/x² + 3/x⁴ - 15/x⁶)
// where φ(x) = exp(-x²/2) / √(2π)
float gelu_asymptotic(float x) {
    float x2 = x * x;
    float exp_val = fast_exp_neg(x2 * 0.5f);
    float phi_x = exp_val * INV_SQRT_2PI_F;

    float inv_x2 = 1.0f / x2;
    float correction = 1.0f - inv_x2 * (1.0f - 3.0f * inv_x2 * (1.0f - 5.0f * inv_x2));

    return -phi_x * correction;
}

// Evaluate GELU approximation with special handling for near-zero and deep tail
float eval_with_special_regions(const Segment& seg, float x) {
    // Near-zero: use Taylor expansion
    if (std::abs(x) < NEAR_ZERO_THRESHOLD) {
        return x * (0.5f + x * INV_SQRT_2PI_F);
    }

    // Deep negative tail: use asymptotic expansion
    if (x < DEEP_TAIL_THRESHOLD) {
        return gelu_asymptotic(x);
    }

    return seg.eval_f32(x);
}

void measure_segment_error(Segment& seg) {
    seg.max_ulp = 0;
    seg.mean_ulp = 0.0;
    seg.sample_count = 0;
    seg.worst_x = static_cast<float>(seg.x_start);

    double ulp_sum = 0.0;

    // Iterate through all bf16 values in this segment
    for (uint16_t bits : g_bf16_index) {
        float x = bf16_to_float(from_bits(bits));

        if (x < seg.x_start || x >= seg.x_end) continue;

        // Compute approximation in fp32 with special region handling
        float approx = eval_with_special_regions(seg, x);
        bfloat16_t approx_bf16 = float_to_bf16(approx);

        // Reference in bf16
        bfloat16_t ref_bf16 = gelu_reference_bf16(x);

        // ULP distance
        int64_t ulp = ulp_distance(approx_bf16, ref_bf16);

        if (ulp > seg.max_ulp) {
            seg.max_ulp = ulp;
            seg.worst_x = x;
        }

        ulp_sum += static_cast<double>(ulp);
        seg.sample_count++;
    }

    if (seg.sample_count > 0) {
        seg.mean_ulp = ulp_sum / seg.sample_count;
    }
}

// ============================================================================
// Adaptive Segmentation
// ============================================================================

class AdaptiveGELU {
public:
    Config config;
    std::vector<Segment> segments;

    AdaptiveGELU(const Config& cfg) : config(cfg) {}

    // Initialize with uniform segments
    void init_uniform() {
        segments.clear();
        double step = (config.x_max - config.x_min) / config.num_segments;

        for (int i = 0; i < config.num_segments; ++i) {
            Segment seg;
            seg.x_start = config.x_min + i * step;
            seg.x_end = config.x_min + (i + 1) * step;
            segments.push_back(seg);
        }
    }

    // Initialize with curvature-aware spacing
    void init_curvature_aware() {
        segments.clear();

        // Estimate curvature at sample points
        std::vector<double> xs;
        std::vector<double> curvatures;

        int num_samples = 1000;
        double h = 0.001;  // Step for numerical derivative

        for (int i = 0; i < num_samples; ++i) {
            double t = static_cast<double>(i) / (num_samples - 1);
            double x = config.x_min + t * (config.x_max - config.x_min);

            // Second derivative via finite differences
            double y_minus = gelu_reference_f64(x - h);
            double y = gelu_reference_f64(x);
            double y_plus = gelu_reference_f64(x + h);
            double d2y = (y_plus - 2*y + y_minus) / (h * h);

            xs.push_back(x);
            curvatures.push_back(std::abs(d2y) + 0.1);  // Add small constant to avoid zero
        }

        // Integrate curvature to get arc-length-like parameter
        std::vector<double> cumulative(num_samples);
        cumulative[0] = 0.0;
        for (int i = 1; i < num_samples; ++i) {
            double dx = xs[i] - xs[i-1];
            cumulative[i] = cumulative[i-1] + curvatures[i] * dx;
        }

        // Normalize
        double total = cumulative.back();
        for (auto& c : cumulative) c /= total;

        // Find segment boundaries at uniform intervals of cumulative curvature
        std::vector<double> boundaries;
        boundaries.push_back(config.x_min);

        for (int i = 1; i < config.num_segments; ++i) {
            double target = static_cast<double>(i) / config.num_segments;

            // Binary search for x where cumulative(x) = target
            int lo = 0, hi = num_samples - 1;
            while (lo < hi) {
                int mid = (lo + hi) / 2;
                if (cumulative[mid] < target) lo = mid + 1;
                else hi = mid;
            }

            boundaries.push_back(xs[lo]);
        }
        boundaries.push_back(config.x_max);

        // Create segments
        for (int i = 0; i < config.num_segments; ++i) {
            Segment seg;
            seg.x_start = boundaries[i];
            seg.x_end = boundaries[i + 1];
            segments.push_back(seg);
        }
    }

    // Fit polynomials to all segments
    void fit_all() {
        for (auto& seg : segments) {
            seg.poly = fit_polynomial_lsq(seg.x_start, seg.x_end, config.poly_degree,
                                          seg.x_mid, seg.x_scale);

            // Optional minimax refinement
            seg.poly = refine_minimax(seg.poly, seg.x_start, seg.x_end,
                                      seg.x_mid, seg.x_scale, 5);
        }
    }

    // Measure error for all segments
    void measure_all() {
        for (auto& seg : segments) {
            measure_segment_error(seg);
        }
    }

    // Get overall statistics
    struct Stats {
        int64_t max_ulp = 0;
        double mean_ulp = 0.0;
        int64_t min_segment_ulp = INT64_MAX;
        int64_t max_segment_ulp = 0;
        float worst_x = 0.0f;
        int worst_segment = 0;
    };

    Stats get_stats() const {
        Stats stats;
        double total_ulp = 0.0;
        int total_samples = 0;

        for (int i = 0; i < static_cast<int>(segments.size()); ++i) {
            const auto& seg = segments[i];

            if (seg.max_ulp > stats.max_ulp) {
                stats.max_ulp = seg.max_ulp;
                stats.worst_x = seg.worst_x;
                stats.worst_segment = i;
            }

            stats.min_segment_ulp = std::min(stats.min_segment_ulp, seg.max_ulp);
            stats.max_segment_ulp = std::max(stats.max_segment_ulp, seg.max_ulp);

            total_ulp += seg.mean_ulp * seg.sample_count;
            total_samples += seg.sample_count;
        }

        if (total_samples > 0) {
            stats.mean_ulp = total_ulp / total_samples;
        }

        return stats;
    }

    // Adaptive refinement: redistribute segments based on error
    // Uses smooth blending to prevent segment collapse
    void refine_segments() {
        // Minimum segment width to prevent collapse
        double total_range = config.x_max - config.x_min;
        double min_width = total_range / (config.num_segments * 10);

        // Compute desired segment widths based on error
        // High-error segments should be narrower (more resolution)
        std::vector<double> desired_widths(segments.size());
        double total_desired = 0.0;

        for (size_t i = 0; i < segments.size(); ++i) {
            // Inverse relationship: higher error -> narrower segment
            // Use sqrt to soften the effect and prevent extreme narrowing
            double error_factor = std::sqrt(static_cast<double>(segments[i].max_ulp) + 1.0);
            desired_widths[i] = 1.0 / error_factor;
            total_desired += desired_widths[i];
        }

        // Normalize to sum to total range
        for (auto& w : desired_widths) {
            w = w * total_range / total_desired;
            // Enforce minimum width
            w = std::max(w, min_width);
        }

        // Re-normalize after enforcing minimums
        double sum = std::accumulate(desired_widths.begin(), desired_widths.end(), 0.0);
        for (auto& w : desired_widths) {
            w = w * total_range / sum;
        }

        // Blend with current widths for stability (70% new, 30% old)
        constexpr double blend = 0.7;
        for (size_t i = 0; i < segments.size(); ++i) {
            double current_width = segments[i].x_end - segments[i].x_start;
            desired_widths[i] = blend * desired_widths[i] + (1.0 - blend) * current_width;
        }

        // Final normalization
        sum = std::accumulate(desired_widths.begin(), desired_widths.end(), 0.0);
        for (auto& w : desired_widths) {
            w = w * total_range / sum;
        }

        // Apply new boundaries
        double pos = config.x_min;
        for (size_t i = 0; i < segments.size(); ++i) {
            segments[i].x_start = pos;
            pos += desired_widths[i];
            segments[i].x_end = pos;
        }
        // Fix last boundary exactly
        segments.back().x_end = config.x_max;
    }

    // Full optimization loop
    void optimize() {
        std::cout << "\n=== Adaptive Piecewise Polynomial GELU Optimization ===\n";
        std::cout << "Segments: " << config.num_segments
                  << ", Degree: " << config.poly_degree
                  << ", Range: [" << config.x_min << ", " << config.x_max << "]\n\n";

        // Start with curvature-aware initialization
        init_curvature_aware();

        for (int iter = 0; iter < config.max_iterations; ++iter) {
            fit_all();
            measure_all();

            Stats stats = get_stats();

            std::cout << "Iteration " << std::setw(2) << iter + 1 << ": "
                      << "Max ULP = " << std::setw(6) << stats.max_ulp
                      << ", Mean = " << std::fixed << std::setprecision(4) << stats.mean_ulp
                      << ", Min/Max segment = " << stats.min_segment_ulp
                      << "/" << stats.max_segment_ulp
                      << ", Worst x = " << std::setprecision(4) << stats.worst_x
                      << "\n";

            // Check convergence
            if (stats.min_segment_ulp > 0) {
                double ratio = static_cast<double>(stats.max_segment_ulp) / stats.min_segment_ulp;
                if (ratio < 1.0 + config.convergence_threshold) {
                    std::cout << "\nConverged! Error ratio = " << std::setprecision(2) << ratio << "\n";
                    break;
                }
            }

            // Don't refine on last iteration
            if (iter < config.max_iterations - 1) {
                refine_segments();
            }
        }
    }

    // Print detailed segment info
    void print_segments() const {
        std::cout << "\n=== Segment Details ===\n";
        std::cout << std::setw(4) << "Seg"
                  << std::setw(12) << "Start"
                  << std::setw(12) << "End"
                  << std::setw(10) << "Width"
                  << std::setw(10) << "MaxULP"
                  << std::setw(10) << "MeanULP"
                  << std::setw(8) << "Count"
                  << std::setw(12) << "WorstX"
                  << "\n";
        std::cout << std::string(88, '-') << "\n";

        for (size_t i = 0; i < segments.size(); ++i) {
            const auto& seg = segments[i];
            std::cout << std::setw(4) << i
                      << std::fixed << std::setprecision(4)
                      << std::setw(12) << seg.x_start
                      << std::setw(12) << seg.x_end
                      << std::setw(10) << (seg.x_end - seg.x_start)
                      << std::setw(10) << seg.max_ulp
                      << std::setw(10) << std::setprecision(3) << seg.mean_ulp
                      << std::setw(8) << seg.sample_count
                      << std::setw(12) << std::setprecision(4) << seg.worst_x
                      << "\n";
        }
    }

    // Print polynomial coefficients
    void print_coefficients() const {
        std::cout << "\n=== Polynomial Coefficients ===\n";
        std::cout << "// Polynomial in scaled coordinates: p(u) where u = (x - x_mid) / x_scale\n";
        std::cout << "// Format: { x_start, x_end, x_mid, x_scale, c0, c1, c2, ... }\n\n";

        std::cout << std::scientific << std::setprecision(8);

        for (size_t i = 0; i < segments.size(); ++i) {
            const auto& seg = segments[i];
            std::cout << "// Segment " << i << ": [" << seg.x_start << ", " << seg.x_end
                      << "], MaxULP=" << seg.max_ulp << "\n";
            std::cout << "{ " << seg.x_start << ", " << seg.x_end << ", "
                      << seg.x_mid << ", " << seg.x_scale << ", ";
            for (size_t j = 0; j < seg.poly.coeffs.size(); ++j) {
                if (j > 0) std::cout << ", ";
                std::cout << seg.poly.coeffs[j];
            }
            std::cout << " },\n";
        }
    }

    // Evaluate the piecewise polynomial
    float eval(float x) const {
        // Handle saturation
        if (x >= config.x_max) return x;
        if (x <= config.x_min) return 0.0f;

        // Near-zero: use Taylor expansion
        if (std::abs(x) < NEAR_ZERO_THRESHOLD) {
            return x * (0.5f + x * INV_SQRT_2PI_F);
        }

        // Deep negative tail: use asymptotic expansion
        if (x < DEEP_TAIL_THRESHOLD) {
            return gelu_asymptotic(x);
        }

        // Find segment (linear search for now)
        for (const auto& seg : segments) {
            if (x >= seg.x_start && x < seg.x_end) {
                return seg.eval_f32(x);
            }
        }

        // Fallback (shouldn't reach here)
        return segments.back().eval_f32(x);
    }
};

// ============================================================================
// Parameter Sweep
// ============================================================================

void parameter_sweep() {
    std::cout << "\n=== Parameter Sweep ===\n";
    std::cout << std::setw(10) << "Segments"
              << std::setw(8) << "Degree"
              << std::setw(10) << "MaxULP"
              << std::setw(12) << "MeanULP"
              << std::setw(10) << "Coeffs"
              << "\n";
    std::cout << std::string(50, '-') << "\n";

    for (int segs : {8, 16, 32, 64}) {
        for (int deg : {2, 3, 4, 5}) {
            Config cfg;
            cfg.num_segments = segs;
            cfg.poly_degree = deg;
            cfg.max_iterations = 20;
            cfg.verbose = false;

            AdaptiveGELU gelu(cfg);
            gelu.init_curvature_aware();

            // Run optimization silently
            for (int iter = 0; iter < cfg.max_iterations; ++iter) {
                gelu.fit_all();
                gelu.measure_all();
                if (iter < cfg.max_iterations - 1) {
                    gelu.refine_segments();
                }
            }

            auto stats = gelu.get_stats();
            int total_coeffs = segs * (deg + 1);

            std::cout << std::setw(10) << segs
                      << std::setw(8) << deg
                      << std::setw(10) << stats.max_ulp
                      << std::fixed << std::setprecision(4)
                      << std::setw(12) << stats.mean_ulp
                      << std::setw(10) << total_coeffs
                      << "\n";
        }
    }
}

// ============================================================================
// Comparison with Existing Methods
// ============================================================================

void compare_with_reference() {
    std::cout << "\n=== Comparison with Existing Methods ===\n";
    std::cout << "Reference targets from README.md:\n";
    std::cout << "  R5 Pure:  Max ULP = 2,  Mean = 0.002\n";
    std::cout << "  B3 Pure:  Max ULP = 23, Mean = 0.01\n";
    std::cout << "  C1 Pure:  Max ULP = 35, Mean = 0.03\n";
    std::cout << "  LUT-87:   Max ULP = 87, Mean = 0.07-0.12\n";
    std::cout << "\n";

    // Test a good configuration
    Config cfg;
    cfg.num_segments = 64;
    cfg.poly_degree = 4;
    cfg.max_iterations = 30;

    AdaptiveGELU gelu(cfg);
    gelu.optimize();
    gelu.print_segments();

    auto stats = gelu.get_stats();
    std::cout << "\n=== Final Result ===\n";
    std::cout << "Max ULP:  " << stats.max_ulp << "\n";
    std::cout << "Mean ULP: " << std::fixed << std::setprecision(4) << stats.mean_ulp << "\n";
    std::cout << "Worst x:  " << stats.worst_x << " (segment " << stats.worst_segment << ")\n";
    std::cout << "Storage:  " << cfg.num_segments * (cfg.poly_degree + 1) << " coefficients + "
              << cfg.num_segments + 1 << " boundaries\n";
}

// ============================================================================
// Main
// ============================================================================

void print_usage() {
    std::cout << "Adaptive Piecewise Polynomial GELU Approximation\n";
    std::cout << "\nUsage: ./adaptive_poly [options]\n";
    std::cout << "\nOptions:\n";
    std::cout << "  --segments N    Number of polynomial segments (default: 32)\n";
    std::cout << "  --degree M      Polynomial degree (default: 5)\n";
    std::cout << "  --iterations I  Max optimization iterations (default: 50)\n";
    std::cout << "  --sweep         Run parameter sweep\n";
    std::cout << "  --compare       Compare with existing methods\n";
    std::cout << "  --coeffs        Print polynomial coefficients\n";
    std::cout << "  --verbose       Verbose output\n";
    std::cout << "  --help          Show this help\n";
}

int main(int argc, char* argv[]) {
    Config cfg;
    bool do_sweep = false;
    bool do_compare = false;
    bool print_coeffs = false;

    // Parse arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--segments" && i + 1 < argc) {
            cfg.num_segments = std::stoi(argv[++i]);
        } else if (arg == "--degree" && i + 1 < argc) {
            cfg.poly_degree = std::stoi(argv[++i]);
        } else if (arg == "--iterations" && i + 1 < argc) {
            cfg.max_iterations = std::stoi(argv[++i]);
        } else if (arg == "--sweep") {
            do_sweep = true;
        } else if (arg == "--compare") {
            do_compare = true;
        } else if (arg == "--coeffs") {
            print_coeffs = true;
        } else if (arg == "--verbose") {
            cfg.verbose = true;
        } else if (arg == "--help") {
            print_usage();
            return 0;
        }
    }

    // Initialize bf16 tables
    std::cout << "Initializing BFloat16 tables...\n";
    init_bf16_tables();
    std::cout << "Valid bf16 values: " << g_bf16_index.size() << "\n";

    if (do_sweep) {
        parameter_sweep();
        return 0;
    }

    if (do_compare) {
        compare_with_reference();
        return 0;
    }

    // Default: run optimization with given parameters
    AdaptiveGELU gelu(cfg);
    gelu.optimize();
    gelu.print_segments();

    if (print_coeffs) {
        gelu.print_coefficients();
    }

    auto stats = gelu.get_stats();
    std::cout << "\n=== Summary ===\n";
    std::cout << "Configuration: " << cfg.num_segments << " segments, degree " << cfg.poly_degree << "\n";
    std::cout << "Max ULP:  " << stats.max_ulp << "\n";
    std::cout << "Mean ULP: " << std::fixed << std::setprecision(4) << stats.mean_ulp << "\n";
    std::cout << "Worst x:  " << stats.worst_x << "\n";

    return 0;
}
