#!/usr/bin/env python3
"""
DOPRI-based GELU Approximation for FULL BFloat16 Range

Pure DOPRI polynomial segments covering the entire range:
- Full range: [-13.5625, 2.78125] (entire bfloat16 non-saturating range)
- No mixing with other methods (no Taylor series, no asymptotic expansion)
- Adaptive segment placement based on error equalization
- Minimax polynomial fitting per segment

Target: Sub-1 ULP error over the FULL bfloat16 range using ONLY DOPRI polynomials.
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
    """Adaptive breakpoint selection with error equalization and zero-handling."""
    x = np.linspace(x_min, x_max, 2000)
    y = func(x)
    h = x[1] - x[0]

    # Estimate second derivative (curvature)
    d2y = np.zeros_like(y)
    d2y[1:-1] = np.abs(y[2:] - 2*y[1:-1] + y[:-2]) / (h**2)
    d2y[0] = d2y[1]
    d2y[-1] = d2y[-2]

    # Boost curvature near zero to ensure fine segmentation
    zero_idx = np.argmin(np.abs(x))
    zero_boost = np.exp(-((x - x[zero_idx])**2) / 0.1)  # Gaussian boost centered at zero
    d2y = d2y + 10.0 * np.max(d2y) * zero_boost  # Strong boost near zero

    # Add baseline to avoid zero regions
    d2y = d2y + 0.01 * np.max(d2y)

    # Initial breakpoints weighted by curvature
    cumsum = np.cumsum(d2y)
    cumsum = cumsum / cumsum[-1]
    targets = np.linspace(0, 1, n_segments + 1)
    breakpoints = np.interp(targets, cumsum, x)
    breakpoints[0] = x_min
    breakpoints[-1] = x_max

    # Force the breakpoint closest to zero to BE exactly zero
    zero_idx = np.argmin(np.abs(breakpoints))
    breakpoints[zero_idx] = 0.0

    # Refine to equalize errors (DOPRI-inspired)
    for iteration in range(15):
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
class DOPRIConfig:
    """Configuration for full-range DOPRI approximation."""
    name: str
    degree: int
    n_segments: int
    x_min: float = -13.5625  # Full bfloat16 range (GELU doesn't underflow)
    x_max: float = 2.78125   # Positive saturation threshold


def generate_dopri_full_range(config: DOPRIConfig) -> dict:
    """Generate DOPRI approximation for full range using single polynomial approach."""
    print(f"\nGenerating {config.name}...")
    print(f"  Degree: {config.degree}, Segments: {config.n_segments}")
    print(f"  Full range: [{config.x_min}, {config.x_max}]")

    # Generate breakpoints for entire range
    breakpoints = select_breakpoints_adaptive(
        gelu_exact, config.x_min, config.x_max, config.n_segments, config.degree)

    # Fit minimax polynomial for each segment
    coefficients = []
    errors = []
    for i in range(config.n_segments):
        coeffs, max_err = fit_minimax_polynomial(
            gelu_exact, breakpoints[i], breakpoints[i+1], config.degree, n_iterations=30)
        coefficients.append(coeffs)
        errors.append(max_err)
        if i < 10 or max_err > 1e-7:  # Show first 10 and any problematic segments
            print(f"    Segment {i:2d} [{breakpoints[i]:9.5f}, {breakpoints[i+1]:9.5f}]: error = {max_err:.6e}")

    print(f"  Maximum error: {np.max(errors):.6e}")
    print(f"  Mean error: {np.mean(errors):.6e}")

    return {
        'config': config,
        'breakpoints': breakpoints,
        'coefficients': coefficients,
        'errors': errors,
    }


def generate_cpp_code(data: dict) -> str:
    """Generate C++ code for full-range DOPRI approximation."""
    config = data['config']
    func_name = config.name.replace(" ", "_").replace("-", "_").lower()

    lines = [
        "/**",
        f" * @brief {config.name}",
        f" *",
        f" * Pure DOPRI polynomial approximation for FULL bfloat16 range",
        f" * Degree: {config.degree}, Segments: {config.n_segments}",
        f" * Range: [{config.x_min}, {config.x_max}]",
        f" * No mixing with other methods - pure polynomial only",
        f" */",
        f"std::bfloat16_t gelu_{func_name}(std::bfloat16_t x_bf16) {{",
        f"    float x = static_cast<float>(x_bf16);",
        f"",
        f"    // Saturation check",
        f"    if (x >= {config.x_max}f) {{",
        f"        return static_cast<std::bfloat16_t>(x);",
        f"    }}",
        f"",
        f"    constexpr float breakpoints[] = {{",
    ]

    # Add breakpoints
    bp = data['breakpoints']
    for i in range(0, len(bp), 5):
        chunk = bp[i:i+5]
        lines.append("        " + ", ".join(f"{b:.8f}f" for b in chunk) + ",")
    lines[-1] = lines[-1].rstrip(',')
    lines.append("    };")
    lines.append("")

    # Add coefficients
    coeffs = data['coefficients']
    lines.append(f"    constexpr float coeffs[{config.n_segments}][{config.degree + 1}] = {{")
    for c in coeffs:
        coeff_str = ", ".join(f"{v:.12e}f" for v in c)
        lines.append(f"        {{ {coeff_str} }},")
    lines.append("    };")
    lines.append("")

    # Segment lookup
    lines.extend([
        f"    // Find segment",
        f"    int seg = 0;",
        f"    for (int i = 1; i < {config.n_segments}; i++) {{",
        f"        if (x >= breakpoints[i]) seg = i;",
        f"    }}",
        f"",
        f"    // Horner's method evaluation",
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
    """Generate full-range DOPRI coefficients."""
    print("=" * 70)
    print("DOPRI Full-Range GELU Approximation")
    print("=" * 70)
    print(f"\nCovering ENTIRE bfloat16 range with DOPRI polynomials only")
    print(f"Range: [-13.5625, 2.78125] (no saturation/underflow)")
    print(f"Pure polynomial - no mixing with other methods")

    # Use very many segments to handle challenging regions, with forced zero boundary
    config = DOPRIConfig(
        "DOPRI Full Range Deg7 Seg96",
        degree=7,  # Even higher degree for better accuracy
        n_segments=96,  # Very many segments for full range
        x_min=-13.5625,
        x_max=2.78125
    )

    print("\n" + "=" * 70)
    print("Generating approximation...")
    print("=" * 70)

    data = generate_dopri_full_range(config)

    print("\n" + "=" * 70)
    print("Generating C++ code...")
    print("=" * 70)

    cpp_code = [
        "// ============================================================================",
        "// DOPRI FULL-RANGE POLYNOMIAL APPROXIMATION",
        "// ============================================================================",
        "//",
        "// Pure DOPRI polynomial segments covering entire bfloat16 range",
        "// Range: [-13.5625, 2.78125] (no saturation/underflow)",
        "// No mixing with Taylor series or asymptotic expansion",
        "//",
        "// Generated by generate_dopri_full_range.py",
        "// ============================================================================",
        "",
        generate_cpp_code(data)
    ]

    output_file = "/home/user/bf16_gelu_research/dopri_full_range.cpp"
    with open(output_file, 'w') as f:
        f.write('\n'.join(cpp_code))

    print(f"\nC++ code written to: {output_file}")
    print(f"Maximum approximation error: {np.max(data['errors']):.6e}")


if __name__ == "__main__":
    main()
