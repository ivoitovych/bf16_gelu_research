#!/usr/bin/env python3
"""
GELU Piecewise Polynomial Approximation Research Tool (DOPRI-Inspired)

CORRECTED bounds for full bfloat16 range:
- Core approximation region: [-3.5, 3.0]
- Negative tail (x < -3.5): Handled by gelu_negative_tail() in C++
- Positive saturation (x >= 3.0): GELU(x) = x

Target: bfloat16 with <1 ULP max error in core region
Operations: +, -, * only (HW accelerated)
"""

import numpy as np
from scipy.special import erf
from typing import List, Tuple
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

    # Chebyshev nodes for initial approximation
    k = np.arange(n_nodes)
    nodes = 0.5 * (x_max + x_min) + 0.5 * (x_max - x_min) * np.cos(np.pi * k / (n_nodes - 1))
    nodes = np.sort(nodes)

    best_coeffs = None
    best_error = float('inf')

    for _ in range(n_iterations):
        # Construct the linear system for Remez exchange
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

        # Evaluate on dense grid to find extrema
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


def select_breakpoints_adaptive(func, x_min: float, x_max: float,
                                n_segments: int, degree: int) -> np.ndarray:
    """Adaptive breakpoint selection with error equalization (DOPRI-inspired)."""
    # Initialize with curvature-based placement
    x = np.linspace(x_min, x_max, 1000)
    y = func(x)
    h = x[1] - x[0]

    # Estimate second derivative (curvature)
    d2y = np.zeros_like(y)
    d2y[1:-1] = np.abs(y[2:] - 2*y[1:-1] + y[:-2]) / (h**2)
    d2y[0] = d2y[1]
    d2y[-1] = d2y[-2]
    d2y = d2y + 0.1 * np.max(d2y)  # Add baseline to avoid zero regions

    # Initial breakpoints weighted by curvature
    cumsum = np.cumsum(d2y)
    cumsum = cumsum / cumsum[-1]
    targets = np.linspace(0, 1, n_segments + 1)
    breakpoints = np.interp(targets, cumsum, x)
    breakpoints[0] = x_min
    breakpoints[-1] = x_max

    # Refine to equalize errors (DOPRI-inspired adaptive step control)
    for iteration in range(10):
        segment_errors = []
        for i in range(n_segments):
            _, max_err = fit_minimax_polynomial(
                func, breakpoints[i], breakpoints[i+1], degree, n_iterations=10)
            segment_errors.append(max_err)

        segment_errors = np.array(segment_errors)
        mean_error = np.mean(segment_errors)

        # Adjust breakpoints to equalize errors
        new_breakpoints = [breakpoints[0]]
        for i in range(n_segments - 1):
            width = breakpoints[i+1] - breakpoints[i]
            ratio = segment_errors[i] / (mean_error + 1e-15)
            # Shrink high-error segments, expand low-error segments
            adjustment = 0.05 * (ratio - 1) * width
            new_bp = breakpoints[i+1] - adjustment

            # Constrain to valid range
            lower = breakpoints[i] + 0.05 * width
            upper = breakpoints[i+2] - 0.05 * (breakpoints[i+2] - breakpoints[i+1]) \
                    if i+2 < len(breakpoints) else x_max
            new_bp = np.clip(new_bp, lower, upper)
            new_breakpoints.append(new_bp)
        new_breakpoints.append(breakpoints[-1])
        breakpoints = np.array(new_breakpoints)

        # Check convergence: max_error / min_error < 2.0
        if np.max(segment_errors) / (np.min(segment_errors) + 1e-15) < 2.0:
            break

    return breakpoints


@dataclass
class DOPRIConfig:
    """Configuration for a DOPRI-based approximation."""
    name: str
    degree: int
    n_segments: int
    x_min: float = -3.5
    x_max: float = 3.0


def generate_dopri_approximation(config: DOPRIConfig) -> dict:
    """Generate DOPRI-based approximation for given configuration."""
    print(f"\nGenerating {config.name}...")
    print(f"  Degree: {config.degree}, Segments: {config.n_segments}")
    print(f"  Range: [{config.x_min}, {config.x_max}]")

    # Generate adaptive breakpoints
    breakpoints = select_breakpoints_adaptive(
        gelu_exact, config.x_min, config.x_max, config.n_segments, config.degree)

    # Fit minimax polynomial for each segment
    coefficients = []
    max_errors = []
    for i in range(config.n_segments):
        coeffs, max_err = fit_minimax_polynomial(
            gelu_exact, breakpoints[i], breakpoints[i+1], config.degree)
        coefficients.append(coeffs)
        max_errors.append(max_err)

    print(f"  Max segment error: {np.max(max_errors):.6e}")
    print(f"  Mean segment error: {np.mean(max_errors):.6e}")
    print(f"  Error ratio: {np.max(max_errors) / (np.min(max_errors) + 1e-15):.3f}")

    return {
        'config': config,
        'breakpoints': breakpoints,
        'coefficients': coefficients,
        'max_errors': max_errors,
    }


def generate_cpp_code(data: dict) -> str:
    """Generate C++ code for DOPRI approximation."""
    config = data['config']
    breakpoints = data['breakpoints']
    coefficients = data['coefficients']

    func_name = config.name.replace(" ", "_").replace("-", "_").lower()

    lines = [
        "/**",
        f" * @brief {config.name}",
        f" *",
        f" * DOPRI-inspired adaptive piecewise polynomial approximation",
        f" * Degree: {config.degree}, Segments: {config.n_segments}",
        f" * Core region: [{config.x_min}, {config.x_max}]",
        f" * Negative tail (x < {config.x_min}): Uses gelu_negative_tail()",
        f" * Positive tail (x >= {config.x_max}): GELU(x) = x",
        f" *",
        f" * Operations: {config.degree} MUL + {config.degree} ADD + log2({config.n_segments}) CMP",
        f" */",
        f"std::bfloat16_t gelu_{func_name}(std::bfloat16_t x_bf16) {{",
        f"    float x = static_cast<float>(x_bf16);",
        f"",
        f"    // Positive saturation",
        f"    if (x >= thresholds::POS) {{",
        f"        return static_cast<std::bfloat16_t>(x);",
        f"    }}",
        f"",
        f"    // Negative tail: use specialized handler",
        f"    if (x < thresholds::TAIL_START) {{",
        f"        return static_cast<std::bfloat16_t>(gelu_negative_tail(x));",
        f"    }}",
        f"",
        f"    // Breakpoints for segment lookup",
        f"    constexpr float breakpoints[] = {{",
    ]

    # Add breakpoints
    for i in range(0, len(breakpoints), 5):
        chunk = breakpoints[i:i+5]
        lines.append("        " + ", ".join(f"{bp:.8f}f" for bp in chunk) + ",")
    lines[-1] = lines[-1].rstrip(',')  # Remove trailing comma
    lines.append("    };")
    lines.append("")

    # Add coefficients
    lines.append(f"    // Coefficients [segment][coefficient]")
    lines.append(f"    constexpr float coeffs[{config.n_segments}][{config.degree + 1}] = {{")
    for i, coeffs in enumerate(coefficients):
        coeff_str = ", ".join(f"{c:.12e}f" for c in coeffs)
        lines.append(f"        {{ {coeff_str} }},")
    lines.append("    };")
    lines.append("")

    # Binary search for segment (if n_segments >= 8)
    if config.n_segments >= 8:
        lines.extend([
            f"    // Binary search for segment (log2({config.n_segments}) comparisons)",
            f"    int seg = 0;",
            f"    int low = 0, high = {config.n_segments - 1};",
            f"    while (low < high) {{",
            f"        int mid = (low + high + 1) / 2;",
            f"        if (x >= breakpoints[mid]) {{",
            f"            low = mid;",
            f"        }} else {{",
            f"            high = mid - 1;",
            f"        }}",
            f"    }}",
            f"    seg = low;",
        ])
    else:
        # Linear search for small number of segments
        lines.extend([
            f"    // Linear search for segment",
            f"    int seg = 0;",
        ])
        for i in range(1, config.n_segments):
            lines.append(f"    if (x >= breakpoints[{i}]) seg = {i};")

    lines.extend([
        "",
        "    // Horner's method evaluation",
        f"    float result = coeffs[seg][{config.degree}];",
    ])

    for d in range(config.degree - 1, -1, -1):
        lines.append(f"    result = result * x + coeffs[seg][{d}];")

    lines.extend([
        "",
        "    return static_cast<std::bfloat16_t>(result);",
        "}",
        "",
    ])

    return "\n".join(lines)


def main():
    """Generate DOPRI coefficients for multiple configurations."""
    print("=" * 70)
    print("DOPRI-Inspired GELU Approximation Coefficient Generator")
    print("=" * 70)
    print(f"\nCore approximation range: [-3.5, 3.0]")
    print(f"Negative tail (x < -3.5): Handled by existing gelu_negative_tail()")
    print(f"Positive saturation (x >= 3.0): GELU(x) = x")

    # Configurations to generate
    configs = [
        DOPRIConfig("DOPRI Deg4 Seg8", degree=4, n_segments=8),
        DOPRIConfig("DOPRI Deg4 Seg16", degree=4, n_segments=16),
        DOPRIConfig("DOPRI Deg5 Seg6", degree=5, n_segments=6),
        DOPRIConfig("DOPRI Deg5 Seg10", degree=5, n_segments=10),
        DOPRIConfig("DOPRI Deg5 Seg12", degree=5, n_segments=12),
        DOPRIConfig("DOPRI Deg5 Seg16", degree=5, n_segments=16),
    ]

    print("\n" + "=" * 70)
    print("Generating approximations...")
    print("=" * 70)

    all_data = []
    for config in configs:
        data = generate_dopri_approximation(config)
        all_data.append(data)

    print("\n" + "=" * 70)
    print("Generating C++ code...")
    print("=" * 70)

    # Generate C++ code for all configurations
    cpp_code = []
    cpp_code.append("// ============================================================================")
    cpp_code.append("// DOPRI-INSPIRED ADAPTIVE PIECEWISE POLYNOMIAL APPROXIMATIONS")
    cpp_code.append("// ============================================================================")
    cpp_code.append("//")
    cpp_code.append("// These methods use DOPRI-inspired adaptive step control to equalize errors")
    cpp_code.append("// across segments. Breakpoints are placed based on curvature and iteratively")
    cpp_code.append("// refined to minimize max error.")
    cpp_code.append("//")
    cpp_code.append("// Core approximation: [-3.5, 3.0]")
    cpp_code.append("// Negative tail: Uses gelu_negative_tail() for x < -3.5")
    cpp_code.append("// Positive tail: GELU(x) = x for x >= 3.0")
    cpp_code.append("//")
    cpp_code.append("// Generated by generate_dopri_coeffs.py")
    cpp_code.append("// ============================================================================")
    cpp_code.append("")

    for data in all_data:
        cpp_code.append(generate_cpp_code(data))

    # Write to file
    output_file = "/home/user/bf16_gelu_research/dopri_implementations.cpp"
    with open(output_file, 'w') as f:
        f.write('\n'.join(cpp_code))

    print(f"\nC++ code written to: {output_file}")
    print(f"Generated {len(configs)} implementations")

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"{'Name':<25} {'Degree':>6} {'Segments':>8} {'Max Error':>12}")
    print("-" * 70)
    for data in all_data:
        config = data['config']
        max_err = np.max(data['max_errors'])
        print(f"{config.name:<25} {config.degree:>6} {config.n_segments:>8} {max_err:>12.6e}")


if __name__ == "__main__":
    main()
