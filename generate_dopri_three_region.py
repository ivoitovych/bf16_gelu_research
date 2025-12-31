#!/usr/bin/env python3
"""
DOPRI-based GELU Approximation with Three-Region Split

Pure DOPRI polynomial segments, split into three regions:
1. Negative region: [-13.5625, -0.125]
2. Near-zero region: [-0.125, +0.125] - fine polynomial segments
3. Positive region: [+0.125, 2.78125]

All regions use DOPRI polynomial fitting - no mixing with other methods.
The split prevents segments from crossing zero and causing large errors.
"""

import numpy as np
from scipy.special import erf
from typing import Tuple
from dataclasses import dataclass


def gelu_exact(x: np.ndarray) -> np.ndarray:
    """Exact GELU."""
    return x * 0.5 * (1.0 + erf(x / np.sqrt(2.0)))


def horner_eval(coeffs: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Horner's method polynomial evaluation."""
    result = np.zeros_like(x, dtype=np.float64)
    for c in reversed(coeffs):
        result = result * x + c
    return result


def fit_minimax_polynomial(func, x_min: float, x_max: float, degree: int,
                           n_iterations: int = 30) -> Tuple[np.ndarray, float]:
    """Minimax polynomial fitting using Remez algorithm."""
    n_coeffs = degree + 1
    n_nodes = degree + 2

    k = np.arange(n_nodes)
    nodes = 0.5 * (x_max + x_min) + 0.5 * (x_max - x_min) * np.cos(np.pi * k / (n_nodes - 1))
    nodes = np.sort(nodes)

    best_coeffs = None
    best_error = float('inf')

    for _ in range(n_iterations):
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

        x_dense = np.linspace(x_min, x_max, 2000)
        error = func(x_dense) - horner_eval(coeffs, x_dense)
        max_err = np.max(np.abs(error))

        if max_err < best_error:
            best_error = max_err
            best_coeffs = coeffs.copy()

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


def select_breakpoints_adaptive(func, x_min: float, x_max: float,
                                n_segments: int, degree: int) -> np.ndarray:
    """Adaptive breakpoint selection with error equalization."""
    x = np.linspace(x_min, x_max, 2000)
    y = func(x)
    h = x[1] - x[0]

    d2y = np.zeros_like(y)
    d2y[1:-1] = np.abs(y[2:] - 2*y[1:-1] + y[:-2]) / (h**2)
    d2y[0] = d2y[1]
    d2y[-1] = d2y[-2]
    d2y = d2y + 0.01 * np.max(d2y)

    cumsum = np.cumsum(d2y)
    cumsum = cumsum / cumsum[-1]
    targets = np.linspace(0, 1, n_segments + 1)
    breakpoints = np.interp(targets, cumsum, x)
    breakpoints[0] = x_min
    breakpoints[-1] = x_max

    for iteration in range(15):
        segment_errors = []
        for i in range(n_segments):
            _, max_err = fit_minimax_polynomial(
                func, breakpoints[i], breakpoints[i+1], degree, n_iterations=10)
            segment_errors.append(max_err)

        segment_errors = np.array(segment_errors)
        mean_error = np.mean(segment_errors)

        new_breakpoints = [breakpoints[0]]
        for i in range(n_segments - 1):
            width = breakpoints[i+1] - breakpoints[i]
            ratio = segment_errors[i] / (mean_error + 1e-15)
            adjustment = 0.05 * (ratio - 1) * width
            new_bp = breakpoints[i+1] - adjustment

            lower = breakpoints[i] + 0.05 * width
            upper = breakpoints[i+2] - 0.05 * (breakpoints[i+2] - breakpoints[i+1]) \
                    if i+2 < len(breakpoints) else x_max
            new_bp = np.clip(new_bp, lower, upper)
            new_breakpoints.append(new_bp)
        new_breakpoints.append(breakpoints[-1])
        breakpoints = np.array(new_breakpoints)

        if np.max(segment_errors) / (np.min(segment_errors) + 1e-15) < 2.0:
            break

    return breakpoints


@dataclass
class ThreeRegionConfig:
    """Configuration for three-region DOPRI approximation."""
    name: str
    degree: int
    neg_segments: int
    zero_segments: int
    pos_segments: int
    neg_min: float = -13.5625
    zero_bound: float = 0.125  # Boundary between regions
    pos_max: float = 2.78125


def generate_three_region_dopri(config: ThreeRegionConfig) -> dict:
    """Generate three-region DOPRI approximation."""
    print(f"\nGenerating {config.name}...")
    print(f"  Degree: {config.degree}")
    print(f"  Negative region [{config.neg_min}, -{config.zero_bound}]: {config.neg_segments} segments")
    print(f"  Near-zero region [{-config.zero_bound}, {config.zero_bound}]: {config.zero_segments} segments")
    print(f"  Positive region [{config.zero_bound}, {config.pos_max}]: {config.pos_segments} segments")

    # Negative region
    neg_breakpoints = select_breakpoints_adaptive(
        gelu_exact, config.neg_min, -config.zero_bound, config.neg_segments, config.degree)
    neg_coeffs = []
    neg_errors = []
    for i in range(config.neg_segments):
        coeffs, err = fit_minimax_polynomial(
            gelu_exact, neg_breakpoints[i], neg_breakpoints[i+1], config.degree, n_iterations=30)
        neg_coeffs.append(coeffs)
        neg_errors.append(err)

    # Near-zero region
    zero_breakpoints = select_breakpoints_adaptive(
        gelu_exact, -config.zero_bound, config.zero_bound, config.zero_segments, config.degree)
    zero_coeffs = []
    zero_errors = []
    for i in range(config.zero_segments):
        coeffs, err = fit_minimax_polynomial(
            gelu_exact, zero_breakpoints[i], zero_breakpoints[i+1], config.degree, n_iterations=30)
        zero_coeffs.append(coeffs)
        zero_errors.append(err)

    # Positive region
    pos_breakpoints = select_breakpoints_adaptive(
        gelu_exact, config.zero_bound, config.pos_max, config.pos_segments, config.degree)
    pos_coeffs = []
    pos_errors = []
    for i in range(config.pos_segments):
        coeffs, err = fit_minimax_polynomial(
            gelu_exact, pos_breakpoints[i], pos_breakpoints[i+1], config.degree, n_iterations=30)
        pos_coeffs.append(coeffs)
        pos_errors.append(err)

    print(f"  Negative max error: {np.max(neg_errors):.6e}")
    print(f"  Near-zero max error: {np.max(zero_errors):.6e}")
    print(f"  Positive max error: {np.max(pos_errors):.6e}")
    print(f"  Overall max error: {np.max([np.max(neg_errors), np.max(zero_errors), np.max(pos_errors)]):.6e}")

    # Combine all regions
    all_breakpoints = np.concatenate([neg_breakpoints[:-1], zero_breakpoints[:-1], pos_breakpoints])
    all_coeffs = neg_coeffs + zero_coeffs + pos_coeffs

    return {
        'config': config,
        'breakpoints': all_breakpoints,
        'coefficients': all_coeffs,
        'n_total_segments': config.neg_segments + config.zero_segments + config.pos_segments,
    }


def generate_cpp_code(data: dict) -> str:
    """Generate C++ code."""
    config = data['config']
    func_name = config.name.replace(" ", "_").replace("-", "_").lower()
    n_segs = data['n_total_segments']

    lines = [
        "/**",
        f" * @brief {config.name}",
        f" *",
        f" * Pure DOPRI polynomial approximation with three-region split",
        f" * Degree: {config.degree}, Total segments: {n_segs}",
        f" * Negative: [{config.neg_min}, -{config.zero_bound}] ({config.neg_segments} segs)",
        f" * Near-zero: [-{config.zero_bound}, {config.zero_bound}] ({config.zero_segments} segs)",
        f" * Positive: [{config.zero_bound}, {config.pos_max}] ({config.pos_segments} segs)",
        f" */",
        f"std::bfloat16_t gelu_{func_name}(std::bfloat16_t x_bf16) {{",
        f"    float x = static_cast<float>(x_bf16);",
        f"",
        f"    // Saturation checks",
        f"    if (x >= {config.pos_max}f) return static_cast<std::bfloat16_t>(x);",
        f"    if (x < {config.neg_min}f) return static_cast<std::bfloat16_t>(0.0f);",
        f"",
        f"    constexpr float breakpoints[] = {{",
    ]

    bp = data['breakpoints']
    for i in range(0, len(bp), 5):
        chunk = bp[i:i+5]
        lines.append("        " + ", ".join(f"{b:.8f}f" for b in chunk) + ",")
    lines[-1] = lines[-1].rstrip(',')
    lines.append("    };")
    lines.append("")

    coeffs = data['coefficients']
    lines.append(f"    constexpr float coeffs[{n_segs}][{config.degree + 1}] = {{")
    for c in coeffs:
        coeff_str = ", ".join(f"{v:.12e}f" for v in c)
        lines.append(f"        {{ {coeff_str} }},")
    lines.append("    };")
    lines.append("")

    lines.extend([
        f"    int seg = 0;",
        f"    for (int i = 1; i < {n_segs}; i++) {{",
        f"        if (x >= breakpoints[i]) seg = i;",
        f"    }}",
        f"",
        f"    float result = coeffs[seg][{config.degree}];",
    ])

    for d in range(config.degree - 1, -1, -1):
        lines.append(f"    result = result * x + coeffs[seg][{d}];")

    lines.extend([
        f"",
        f"    return static_cast<std::bfloat16_t>(result);",
        f"}}",
        f"",
    ])

    return "\n".join(lines)


def main():
    print("=" * 70)
    print("DOPRI Three-Region GELU Approximation")
    print("=" * 70)

    config = ThreeRegionConfig(
        "DOPRI Three Region Deg7",
        degree=7,
        neg_segments=32,    # Deep negative tail to -0.125
        zero_segments=24,   # Fine segmentation near zero
        pos_segments=24,    # Positive region from 0.125 to 2.78
    )

    data = generate_three_region_dopri(config)

    cpp_code = [
        "// ============================================================================",
        "// DOPRI THREE-REGION POLYNOMIAL APPROXIMATION",
        "// ============================================================================",
        "//",
        "// Pure DOPRI polynomials split into three regions to avoid zero-crossing",
        "// All regions use adaptive DOPRI polynomial fitting",
        "// No mixing with Taylor series or other methods - pure polynomial only",
        "//",
        "// Generated by generate_dopri_three_region.py",
        "// ============================================================================",
        "",
        generate_cpp_code(data)
    ]

    output_file = "/home/user/bf16_gelu_research/dopri_three_region.cpp"
    with open(output_file, 'w') as f:
        f.write('\n'.join(cpp_code))

    print(f"\nC++ code written to: {output_file}")
    print(f"Total segments: {data['n_total_segments']}")


if __name__ == "__main__":
    main()
