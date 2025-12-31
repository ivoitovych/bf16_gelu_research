#!/usr/bin/env python3
"""
DOPRI-64 GELU Approximation with Correct Reference

Uses the EXACT same erfc-based reference as gelu_implementations.cpp:
  For x >= 0: Φ(x) = 0.5 * (1 + erf(x/√2))
  For x < 0:  Φ(x) = 0.5 * erfc(-x/√2)  [avoids catastrophic cancellation]
  GELU(x) = x * Φ(x)

This matches the C++ reference implementation exactly, ensuring ULP measurements
are accurate.
"""

import numpy as np
from scipy.special import erf, erfc
from typing import Tuple
from dataclasses import dataclass


def gelu_reference_f64(x: np.ndarray) -> np.ndarray:
    """
    Reference GELU using erfc for negative x (matches C++ implementation).

    This is the EXACT reference used in gelu_implementations.cpp for ULP analysis.
    """
    x = np.asarray(x, dtype=np.float64)
    INV_SQRT_2 = 1.0 / np.sqrt(2.0)
    z = x * INV_SQRT_2

    # Use erfc for negative x to avoid catastrophic cancellation
    phi = np.where(x >= 0,
                   0.5 * (1.0 + erf(z)),
                   0.5 * erfc(-z))

    return x * phi


def horner_eval(coeffs: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Horner's method polynomial evaluation."""
    result = np.zeros_like(x, dtype=np.float64)
    for c in reversed(coeffs):
        result = result * x + c
    return result


def fit_minimax_polynomial(func, x_min: float, x_max: float, degree: int,
                           n_iterations: int = 50) -> Tuple[np.ndarray, float]:
    """Minimax polynomial fitting using Remez algorithm."""
    n_coeffs = degree + 1
    n_nodes = degree + 2

    # Initialize with Chebyshev nodes
    k = np.arange(n_nodes)
    nodes = 0.5 * (x_max + x_min) + 0.5 * (x_max - x_min) * np.cos(np.pi * k / (n_nodes - 1))
    nodes = np.sort(nodes)

    best_coeffs = None
    best_error = float('inf')

    for iteration in range(n_iterations):
        # Build linear system with alternating error
        A = np.zeros((n_nodes, n_coeffs + 1))
        for i in range(n_nodes):
            for j in range(n_coeffs):
                A[i, j] = nodes[i] ** j
            A[i, n_coeffs] = (-1) ** i

        b = func(nodes)

        try:
            solution, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        except:
            break

        coeffs = solution[:n_coeffs]

        # Evaluate error on dense grid
        x_dense = np.linspace(x_min, x_max, 2000)
        error = func(x_dense) - horner_eval(coeffs, x_dense)
        max_err = np.max(np.abs(error))

        if max_err < best_error:
            best_error = max_err
            best_coeffs = coeffs.copy()

        # Find error extrema for next iteration
        extrema_x = [x_min]
        for i in range(1, len(error) - 1):
            if (error[i] > error[i-1] and error[i] > error[i+1]) or \
               (error[i] < error[i-1] and error[i] < error[i+1]):
                extrema_x.append(x_dense[i])
        extrema_x.append(x_max)

        if len(extrema_x) >= n_nodes:
            extrema_x = np.array(extrema_x)
            extrema_err = np.abs(func(extrema_x) - horner_eval(coeffs, extrema_x))
            sorted_idx = np.argsort(-extrema_err)[:n_nodes]
            nodes = np.sort(extrema_x[sorted_idx])

    return best_coeffs, best_error


def fit_constrained_zero_polynomial(func, x_min: float, x_max: float, degree: int) -> Tuple[np.ndarray, float]:
    """
    Fit polynomial with p(0) = 0 constraint.
    Uses form p(x) = x * q(x) where q is degree-1 polynomial.
    """
    def func_over_x(x):
        result = np.zeros_like(x, dtype=np.float64)
        nonzero = x != 0
        result[nonzero] = func(x[nonzero]) / x[nonzero]
        # limit as x->0: GELU(x)/x -> 0.5
        result[~nonzero] = 0.5
        return result

    # Fit q(x) = GELU(x)/x as degree-1 polynomial
    q_coeffs, max_err = fit_minimax_polynomial(func_over_x, x_min, x_max, degree - 1, n_iterations=50)

    # Convert: p(x) = x * q(x) = 0 + q0*x + q1*x² + ...
    p_coeffs = np.zeros(degree + 1)
    p_coeffs[1:len(q_coeffs)+1] = q_coeffs

    # Compute actual error of p(x) vs GELU(x)
    x_test = np.linspace(x_min, x_max, 2000)
    p_vals = horner_eval(p_coeffs, x_test)
    actual_error = np.max(np.abs(func(x_test) - p_vals))

    return p_coeffs, actual_error


@dataclass
class DOPRI64Config:
    """Configuration matching the documented DOPRI-64 structure."""
    degree: int = 5

    # Segment distribution (from documentation)
    deep_neg_segments: int = 24    # [-13.5625, -9.0]
    mod_neg_segments: int = 8      # [-9.0, -5.0]
    mild_neg_segments: int = 8     # [-5.0, -0.5]
    near_zero_neg_segments: int = 4  # [-0.5, 0.0]
    near_zero_pos_segments: int = 4  # [0.0, 0.5]
    pos_segments: int = 16         # [0.5, 3.5]

    # Total: 64 segments

    # Boundaries
    sat_low: float = -13.5625
    deep_neg_end: float = -9.0
    mod_neg_end: float = -5.0
    mild_neg_end: float = -0.5
    zero_point: float = 0.0
    near_zero_pos_end: float = 0.5
    sat_high: float = 3.5


def generate_dopri64_coefficients(config: DOPRI64Config):
    """Generate DOPRI-64 coefficients with proper reference."""
    print("=" * 70)
    print("DOPRI-64 GELU Approximation (Correct Reference)")
    print("=" * 70)
    print(f"\nUsing erfc-based reference (matches C++ test framework)")
    print(f"Degree: {config.degree}")
    print(f"Total segments: 64")
    print()

    all_breakpoints = []
    all_coefficients = []
    all_errors = []

    # Region 1: Deep negative [-13.5625, -9.0] - 24 segments (subnormal GELU values)
    print("Region 1: Deep negative [-13.5625, -9.0] (24 segments)")
    bp = np.linspace(config.sat_low, config.deep_neg_end, config.deep_neg_segments + 1)
    for i in range(config.deep_neg_segments):
        coeffs, err = fit_minimax_polynomial(gelu_reference_f64, bp[i], bp[i+1], config.degree, n_iterations=50)
        all_coefficients.append(coeffs)
        all_errors.append(err)
    all_breakpoints.extend(bp[:-1])
    print(f"  Max error: {np.max(all_errors[-config.deep_neg_segments:]):.6e}")

    # Region 2: Moderate negative [-9.0, -5.0] - 8 segments
    print("Region 2: Moderate negative [-9.0, -5.0] (8 segments)")
    bp = np.linspace(config.deep_neg_end, config.mod_neg_end, config.mod_neg_segments + 1)
    for i in range(config.mod_neg_segments):
        coeffs, err = fit_minimax_polynomial(gelu_reference_f64, bp[i], bp[i+1], config.degree, n_iterations=50)
        all_coefficients.append(coeffs)
        all_errors.append(err)
    all_breakpoints.extend(bp[:-1])
    print(f"  Max error: {np.max(all_errors[-config.mod_neg_segments:]):.6e}")

    # Region 3: Mild negative [-5.0, -0.5] - 8 segments
    print("Region 3: Mild negative [-5.0, -0.5] (8 segments)")
    bp = np.linspace(config.mod_neg_end, config.mild_neg_end, config.mild_neg_segments + 1)
    for i in range(config.mild_neg_segments):
        coeffs, err = fit_minimax_polynomial(gelu_reference_f64, bp[i], bp[i+1], config.degree, n_iterations=50)
        all_coefficients.append(coeffs)
        all_errors.append(err)
    all_breakpoints.extend(bp[:-1])
    print(f"  Max error: {np.max(all_errors[-config.mild_neg_segments:]):.6e}")

    # Region 4: Near-zero negative [-0.5, 0.0] - 4 segments with p(0)=0 constraint
    print("Region 4: Near-zero negative [-0.5, 0.0] (4 segments, constrained p(0)=0)")
    bp = np.linspace(config.mild_neg_end, config.zero_point, config.near_zero_neg_segments + 1)
    for i in range(config.near_zero_neg_segments):
        coeffs, err = fit_constrained_zero_polynomial(gelu_reference_f64, bp[i], bp[i+1], config.degree)
        all_coefficients.append(coeffs)
        all_errors.append(err)
    all_breakpoints.extend(bp[:-1])
    print(f"  Max error: {np.max(all_errors[-config.near_zero_neg_segments:]):.6e}")

    # Region 5: Near-zero positive [0.0, 0.5] - 4 segments with p(0)=0 constraint
    print("Region 5: Near-zero positive [0.0, 0.5] (4 segments, constrained p(0)=0)")
    bp = np.linspace(config.zero_point, config.near_zero_pos_end, config.near_zero_pos_segments + 1)
    for i in range(config.near_zero_pos_segments):
        coeffs, err = fit_constrained_zero_polynomial(gelu_reference_f64, bp[i], bp[i+1], config.degree)
        all_coefficients.append(coeffs)
        all_errors.append(err)
    all_breakpoints.extend(bp[:-1])
    print(f"  Max error: {np.max(all_errors[-config.near_zero_pos_segments:]):.6e}")

    # Region 6: Positive [0.5, 3.5] - 16 segments
    print("Region 6: Positive [0.5, 3.5] (16 segments)")
    bp = np.linspace(config.near_zero_pos_end, config.sat_high, config.pos_segments + 1)
    for i in range(config.pos_segments):
        coeffs, err = fit_minimax_polynomial(gelu_reference_f64, bp[i], bp[i+1], config.degree, n_iterations=50)
        all_coefficients.append(coeffs)
        all_errors.append(err)
    all_breakpoints.extend(bp[:-1])
    print(f"  Max error: {np.max(all_errors[-config.pos_segments:]):.6e}")

    # Final breakpoint
    all_breakpoints.append(config.sat_high)

    print()
    print(f"Overall maximum error: {np.max(all_errors):.6e}")
    print(f"Overall mean error: {np.mean(all_errors):.6e}")

    return {
        'breakpoints': np.array(all_breakpoints),
        'coefficients': all_coefficients,
        'errors': all_errors,
        'config': config
    }


def generate_cpp_code(data: dict) -> str:
    """Generate C++ implementation."""
    config = data['config']
    bp = data['breakpoints']
    coeffs = data['coefficients']

    lines = [
        "// ============================================================================",
        "// DOPRI-64 PIECEWISE POLYNOMIAL APPROXIMATION (CORRECT REFERENCE)",
        "// ============================================================================",
        "//",
        "// Generated using erfc-based reference matching gelu_implementations.cpp",
        "// This ensures accurate ULP measurements.",
        "//",
        "// 64 segments, degree 5 polynomials",
        "// Polynomial region: [-13.5625, 3.5]",
        "// Identity region: [3.5, ∞)",
        "//",
        "// Region distribution:",
        "//   [-13.5625, -9.0]: 24 segments (deep negative, subnormal outputs)",
        "//   [-9.0, -5.0]:      8 segments (moderate negative)",
        "//   [-5.0, -0.5]:      8 segments (mild negative)",
        "//   [-0.5, 0.0]:       4 segments (near-zero, p(0)=0 constraint)",
        "//   [0.0, 0.5]:        4 segments (near-zero, p(0)=0 constraint)",
        "//   [0.5, 3.5]:       16 segments (positive)",
        "//",
        "// ============================================================================",
        "",
        "#define DOPRI64_N_SEGMENTS 64",
        "#define DOPRI64_DEGREE 5",
        "#define DOPRI64_SAT_LOW   -13.5625f",
        "#define DOPRI64_SAT_HIGH   3.5f",
        "",
        f"static const float dopri64_breakpoints[65] = {{",
    ]

    # Format breakpoints (8 per line)
    for i in range(0, len(bp), 8):
        chunk = bp[i:i+8]
        line = "    " + ", ".join(f"{b:.10f}f" for b in chunk) + ","
        lines.append(line)
    lines[-1] = lines[-1].rstrip(',')
    lines.append("};")
    lines.append("")

    # Format coefficients
    lines.append(f"static const float dopri64_coeffs[DOPRI64_N_SEGMENTS][DOPRI64_DEGREE + 1] = {{")
    for i, c in enumerate(coeffs):
        coeff_str = ", ".join(f"{v:.12e}f" for v in c)
        lines.append(f"    /* seg {i:2d} */ {{ {coeff_str} }},")
    lines.append("};")
    lines.append("")

    # Helper functions and main function
    lines.extend([
        "// Binary search for segment index",
        "static inline int dopri64_find_segment(float x) {",
        "    int lo = 0, hi = DOPRI64_N_SEGMENTS - 1;",
        "    while (lo < hi) {",
        "        int mid = (lo + hi + 1) / 2;",
        "        if (x >= dopri64_breakpoints[mid]) {",
        "            lo = mid;",
        "        } else {",
        "            hi = mid - 1;",
        "        }",
        "    }",
        "    return lo;",
        "}",
        "",
        "// Horner's method for polynomial evaluation",
        "static inline float dopri64_horner(int seg, float x) {",
        "    float result = dopri64_coeffs[seg][DOPRI64_DEGREE];",
        "    for (int i = DOPRI64_DEGREE - 1; i >= 0; i--) {",
        "        result = result * x + dopri64_coeffs[seg][i];",
        "    }",
        "    return result;",
        "}",
        "",
        "/**",
        " * @brief DOPRI-64 Piecewise Polynomial GELU Approximation",
        " *",
        " * Generated with erfc-based reference matching C++ test framework.",
        " * Target: Max ULP <= 1 across polynomial region.",
        " *",
        " * - 64 segments, degree 5 polynomials",
        " * - Constrained p(0)=0 for near-zero segments",
        " * - Binary search segment lookup (6 comparisons)",
        " * - 5 FMA operations per evaluation (Horner's method)",
        " */",
        "std::bfloat16_t gelu_dopri64(std::bfloat16_t x_bf16) {",
        "    float x = static_cast<float>(x_bf16);",
        "",
        "    // Saturation: x < -13.5625 -> 0",
        "    if (x < DOPRI64_SAT_LOW) return static_cast<std::bfloat16_t>(0.0f);",
        "",
        "    // Identity: x >= 3.5 -> x",
        "    if (x >= DOPRI64_SAT_HIGH) return static_cast<std::bfloat16_t>(x);",
        "",
        "    // Polynomial region: [-13.5625, 3.5)",
        "    int seg = dopri64_find_segment(x);",
        "    float result = dopri64_horner(seg, x);",
        "    return static_cast<std::bfloat16_t>(result);",
        "}",
    ])

    return "\n".join(lines)


def main():
    config = DOPRI64Config()
    data = generate_dopri64_coefficients(config)

    print("\n" + "=" * 70)
    print("Generating C++ code...")
    print("=" * 70)

    cpp_code = generate_cpp_code(data)

    output_file = "/home/user/bf16_gelu_research/dopri_64_implementation.cpp"
    with open(output_file, 'w') as f:
        f.write(cpp_code)

    print(f"\nC++ code written to: {output_file}")
    print(f"Total segments: 64")
    print(f"Maximum approximation error: {np.max(data['errors']):.6e}")
    print()
    print("Ready to integrate into gelu_implementations.cpp")


if __name__ == "__main__":
    main()
