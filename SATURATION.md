# BFloat16 GELU Saturation Thresholds

## Overview

This document presents the **exact bfloat16 saturation thresholds** for the true GELU function, determined by exhaustive enumeration of all 65,280 valid bf16 values.

## Key Results

| Tail | Threshold | Hex | Condition |
|------|-----------|-----|-----------|
| **Positive** | x ≥ **2.78125** | 0x4032 | bf16(GELU(x)) == x |
| **Negative** | x ≤ **-13.5625** | 0xc159 | bf16(GELU(x)) == 0 |

## Positive Tail: GELU(x) = x

For large positive x, GELU(x) → x asymptotically. The saturation threshold is where GELU(x) is close enough to x that rounding to bf16 produces exactly x.

### Transition Region

| bits | x | GELU(x) exact | bf16(GELU) | Match? |
|------|---------|---------------|------------|--------|
| 0x402f | 2.734375 | 2.72583 | 2.71875 | no |
| 0x4030 | 2.75 | 2.74181 | 2.734375 | no |
| 0x4031 | 2.765625 | 2.75777 | 2.75 | no |
| **0x4032** | **2.78125** | 2.77372 | **2.78125** | **YES** |
| 0x4033 | 2.796875 | 2.78966 | 2.796875 | YES |
| 0x4034 | 2.8125 | 2.80559 | 2.8125 | YES |

### Analysis

At x = 2.78125:
- GELU(2.78125) = 2.773719763...
- Error = 2.78125 - 2.77372 = 0.00753
- bf16 ULP at 2.78125 = 0.015625
- Error < 0.5 ULP, so rounds to x

## Negative Tail: GELU(x) = 0

For large negative x, GELU(x) → 0 exponentially. The saturation threshold is where |GELU(x)| becomes smaller than half the smallest bf16 subnormal.

### Transition Region

| bits | x | GELU(x) exact | bf16 bits | bf16(GELU) | Zero? |
|------|----------|---------------|-----------|------------|-------|
| 0xc156 | -13.375 | -5.66e-40 | 0x8006 | -5.5e-40 | no |
| 0xc157 | -13.4375 | -2.45e-40 | 0x8003 | -2.8e-40 | no |
| 0xc158 | -13.5 | -1.06e-40 | 0x8001 | -9.2e-41 | no |
| **0xc159** | **-13.5625** | -4.53e-41 | **0x8000** | **-0** | **YES** |
| 0xc15a | -13.625 | -1.94e-41 | 0x8000 | -0 | YES |
| 0xc15b | -13.6875 | -8.25e-42 | 0x8000 | -0 | YES |

### Analysis

At x = -13.5625:
- GELU(-13.5625) = -4.531e-41
- Smallest bf16 subnormal = 9.184e-41 (0x0001)
- Rounding threshold = 4.592e-41 (half of smallest)
- |GELU| < threshold, so rounds to 0

## BFloat16 Representation Limits

| Value | Magnitude | Hex |
|-------|-----------|-----|
| Smallest subnormal | 9.1835e-41 | 0x0001 |
| Smallest normal | 1.1755e-38 | 0x0080 |
| Rounding threshold to 0 | 4.5918e-41 | — |

## Comparison with Implementation Thresholds

The CLAUDE.md mentions conservative thresholds used in approximations:

| | True Saturation | Implementation |
|---|-----------------|----------------|
| Positive | x ≥ 2.78125 | x ≥ 3.0 |
| Negative | x ≤ -13.5625 | x ≤ -9.0 |

The implementation thresholds are more conservative because:
1. **Positive**: Using x ≥ 3 ensures GELU(x) ≈ x with comfortable margin
2. **Negative**: Using x ≤ -9 allows tail LUT/approximation to handle [-9, -13.5625] region where GELU is tiny but non-zero in bf16

## Standalone Analysis Tool

```bash
# Build
g++ -std=c++23 -O3 -march=native -o saturation_analysis saturation_analysis.cpp -lm

# Run
./saturation_analysis
```

## Mathematical Background

GELU(x) = x · Φ(x) where Φ(x) = 0.5 · (1 + erf(x/√2))

For numerical stability with negative x:
- Φ(x) = 0.5 · erfc(-x/√2)

This avoids catastrophic cancellation in `1 + erf(z)` when erf(z) ≈ -1.
