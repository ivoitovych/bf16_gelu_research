# GELU Approximation Strategy Taxonomy
This document contains four independently developed strategy lists for GELU approximation using basic arithmetic only. Each list approaches the problem from a slightly different angle, providing complementary perspectives on the design space.
**Common Constraints Across All Lists:**
- **Allowed operations**: `+`, `-`, `×`, `÷`, `|·|`, `sign()`
- **Prohibited functions**: `erf()`, `tanh()`, `exp()`, `log()`
- **Target format**: BFloat16 (7-bit mantissa, 8-bit exponent)
- **Primary metric**: ULP (Units in Last Place)
---

# Final List 1:


Ось **Final List** (синтез найкращого з Consolidated Lists 1–4 + рев’ю), у вигляді чіткої таксономії **A–G + R**, з “Quick formulas” і фазами впровадження. 

## A. Прямі апроксимації GELU (без підфункцій)

* **A1. Minimax-поліном (Remez)**: степені 5/7/9 на обраному діапазоні; обчислення Horner (FMA) або Estrin (паралельно).
* **A2. Раціональна функція (Padé/minimax)**:
  (\mathrm{GELU}(x)\approx x\cdot \dfrac{P_n(x^2)}{Q_m(x^2)}) (парні степені для симетрії); порядки [3/3], [4/4], [5/5].
* **A3. Chebyshev-поліном**: альтернатива minimax, коли потрібно “контрольоване” коливання похибки й простіша побудова.
* **A4. Continued fraction (ланцюговий дріб)**: як альтернативний “раціональний” базис (інколи зручніший у хвостах).

## B. Через підфункції (але теж тільки арифметика)

* **B1. Sigmoid-based без exp**: (\mathrm{GELU}(x)\approx x\cdot\sigma(kx)), де (\sigma) — раціонал / PWL, з (|z|).
* **B2. Tanh-форма + апроксимація tanh**:
  (0.5x(1+\tanh(\alpha(x+\beta x^3)))), де (\tanh) замінити непарним раціоналом або minimax-поліномом (K-TanH style).
* **B3. Поліном erf → (\Phi) → GELU**: (\Phi(x)=0.5(1+\mathrm{erf}(x/\sqrt2))), де erf — непарний поліном + clamp до ±1.
* **B4. Раціональний erf/(\Phi) з range reduction**: окремі фіти для “ядра” й “хвостів”, віддзеркалення по знаку + сатурація.

## C. Piecewise-методи (найсильніший важіль для max-ULP)

* **C1. Piecewise-polynomial / сплайни**: квадр./кубічні, 4–16 сегментів; C¹/C² узгодження на стиках; оптимізація вузлів.
* **C2. Piecewise-rational**: різні Padé на сегментах; менше сегментів для тієї ж точності.
* **C3. Piecewise-linear (PWL)**:

  * ISPA-style (симетрія, 8–16 сегментів),
  * NLI (неоднорідні вузли через dynamic programming),
  * ReLU-network → екстракція breakpoints/slopes (автоматичний PWL-фіт).
* **C4. Asymptotic saturation + core**: хвости (\approx 0) / (\approx x); “ядро” (наприклад ([-3,3])) — poly/rational; гладке зшивання на межах.
* **C5. EPSS (error-peak driven knot refinement)**: ітеративно підтискати вузли/сегменти у точках піків похибки.

## D. Гібриди та LUT (reference-friendly, також good baselines)

* **D1. LUT + інтерполяція**: таблиця для (\Phi(x)) або (\mathrm{GELU}(x)/x), лінійна/квадратична інтерполяція.
* **D2. LUT-хвости + поліном-центр**: хвости таблично (стабільність), центр — minimax poly/rational.
* **D3. LUT + “polynomial correction”**: грубий LUT + низький степінь для поправки.
* **D4. Non-uniform LUT spacing**: вузли не рівномірні (оптимізуються під max-ULP / worst-case).

## E. BF16-орієнтовані “knobs” (накладаються на будь-який метод)

* **E1. Обмеження монотонності/меж**: (0\le \Phi \le 1), монотонність, контроль похідної, уникнення “петель” через округлення.
* **E2. Ціль оптимізації — max-ULP** (не L2/MSE): fit під worst-case.
* **E3. Цикл квантування коефіцієнтів**: fit → quantize (bf16/fp16) → repair/refit → повторити.
* **E4. Range scaling / reduction**: фіт на (x/s), підігнати пороги під “експонентні зони” bf16.
* **E5. Обмеження вузлів/порогів**: breakpoints лише representable (часто pow2) + симетрія.
* **E6. FMA vs non-FMA варіанти**: окремі набори коефіцієнтів/форми (Horner vs Estrin).
* **E7. Robustness / sensitivity**: тест стійкості до ±1 ULP у коефіцієнтах і до дрібних змін порогів.
* **E8. Denormals/FTZ policy**: явний clamp/flush-to-zero та перевірка впливу на хвости й динамічний діапазон.

## F. Еталони (ground truth) для ULP-вимірювання

* **F1. High-precision GELU** (float64/MPFR) як “істина”.
* **F2. Чисельна квадратура для (\Phi(x))** (Gauss/Simpson) як “арифметика-only” reference (повільно, але корисно).
* **F3. Continued fraction erf/(\Phi)** як альтернативний стабільний reference-block.
* **F4. “Документований” поліном erf (baseline)** з відомими межами похибки (для стартового порівняння).

## G. Методологія оцінювання (ULP-first)

* **G1. Метрики**: max-ULP / mean-ULP / p99-ULP + “worst-case input” лог.
* **G2. Пошук worst-case**: щільна сітка + локальне уточнення; фокус: near 0, стики сегментів, пороги сатурації.
* **G3. Регіональний аналіз**: near-zero / core / tails, окремо для (x<0) і (x>0).
* **G4. Симуляція округлення bf16 per-op** (і окремо сценарій fp32-accum vs bf16-accum).
* **G5. Порівняння FMA/non-FMA** як окремий експеримент.
* **G6. Backward pass**: похідна (\mathrm{GELU}') (аналітична від апроксимації або окремий fit) + узгодження гладкості.
* **G7. Cost model**: mul/add/div, гілки, LUT loads, векторизаційна дружність.
* **G8. Regression suite**: фіксовані діапазони/сиди + “adversarial points” (стики/пороги/near-zero).

## R. Рекомендований мінімальний baseline-набір (щоб швидко стартувати порівняння)

* **R1. C4 saturation + core minimax polynomial (deg 7)**
* **R2. A2 rational Padé/minimax [4/4] (direct GELU)**
* **R3. C3 PWL (ISPA-style) з breakpoints, обмеженими до pow2/representable**
* **R4. B2 tanh-form + odd rational tanh**
* **R5. D1 LUT + linear interpolation** (як сильний reference серед “only arithmetic”)

## Quick reference formulas (коротко)

* **Tanh-form GELU**: (\mathrm{GELU}(x)\approx 0.5x\left(1+\tanh(\alpha(x+\beta x^3))\right))
* **Rational tanh (приклад простого odd rational)**: (\tanh(z)\approx \dfrac{z(27+z^2)}{27+9z^2})
* **Sigmoid simple (без exp)**: (\sigma(z)\approx 0.5+\dfrac{z}{2(1+|z|)})
* **GELU via sigmoid**: (\mathrm{GELU}(x)\approx x\cdot\sigma(kx))
* **Saturation idea**: (x>T: \mathrm{GELU}\approx x;; x<-T: \mathrm{GELU}\approx 0) (T підбирається експериментально під ULP)

## Implementation priority (фази)

* **Phase 0**: F1 + G1 + G8 (істина, ULP-фреймворк, регресія)
* **Phase 1**: A1 + A2 + B1 + C4 (перші baselines)
* **Phase 2**: C1 + C3 + B2 (кращий контроль max-ULP)
* **Phase 3**: E1–E8 на “переможцях” (BF16-оптимізація)
* **Phase 4**: G5 + G6 + G3 (FMA, backward, регіони)
* **Phase 5 (optional/advanced)**: розширення гібридів LUT (D2–D4) і додаткові “research” ідеї лише за потреби





---
# Final List 2:


## Final GELU Approximation Strategy List

*Constraints: Basic arithmetic only (+, −, ×, ÷, |·|, sign). No erf(), tanh(), exp(). Target: BF16 ULP error characterization.*

---

### Category A: Direct GELU Approximations

**A1. Minimax Polynomial (Remez)**
- Fit directly to GELU(x) over [−5, 5] or [−8, 8]
- Degrees: 5, 7, 9 (odd for antisymmetry)
- Evaluation: Horner's (FMA-friendly) or Estrin's (parallel)
- Equiripple error distribution via Remez algorithm

**A2. Rational Function (Padé/Minimax)**
- `GELU(x) ≈ x · P_n(x²) / Q_m(x²)` — even powers preserve symmetry
- Orders: [3/3], [4/4], [5/5]
- Generally superior tail convergence vs polynomials

**A3. Chebyshev Polynomial**
- Near-optimal with bounded oscillating error
- Alternative when Remez algorithm unstable
- Same degree range and evaluation as A1

**A4. Continued Fraction**
- Alternative to Padé with different convergence properties
- Sometimes more stable in tail regions
- Truncate at various depths for accuracy/cost tradeoff

---

### Category B: Sub-Function Approximations

**B1. Sigmoid-Based (no exp)**
- `GELU(x) ≈ x · σ(1.702x)`
- Simple: `σ(z) ≈ 0.5 + z / (2(1 + |z|))`
- Higher-order: `σ(z) ≈ 0.5 + z(c₁ + c₂z²) / (1 + c₃|z| + c₄z²)`
- PWL σ variant as ultra-fast baseline

**B2. Tanh-Form + Rational Tanh (K-TanH)**
- Standard: `0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))`
- Replace tanh with odd rational:
  - Simple: `tanh(z) ≈ z(27 + z²) / (27 + 9z²)`
  - Better: odd minimax rational fitted to tanh

**B3. Erf Polynomial → Φ → GELU**
- `erf(x) ≈ Σ aᵢx^(2i+1)` with clamp to ±1 outside core
- `Φ(x) = 0.5(1 + erf(x/√2))`, then `GELU(x) = x·Φ(x)`
- Abramowitz-Stegun coefficients as documented baseline

**B4. Rational Erf/Φ with Range Reduction**
- Separate P/Q coefficients for [0, a] vs [a, ∞)
- Mirror for x < 0, saturate tails to ±1
- Reduces polynomial degree requirements

---

### Category C: Piecewise Methods

**C1. Piecewise Polynomial (Splines)**
- Cubic or quadratic over 4–16 segments
- Knot optimization: EPSS (error-peak search) or dynamic programming
- Ensure C¹ or C² continuity at boundaries
- Exploit symmetry: fit x ≥ 0 only, mirror

**C2. Piecewise Rational**
- Different Padé approximant per segment
- Typically 3–5 segments: (−∞, −3], [−3, 0], [0, 3], [3, ∞)
- Fewer segments than polynomial for equivalent accuracy

**C3. Piecewise Linear (ISPA/NLI)**
- **ISPA**: Symmetry-exploiting, 8–16 segments on positive axis, mirror
- **Power-of-2 breakpoints**: BF16-representable values reduce rounding errors
- **NLI**: Non-uniform breakpoints via dynamic programming
- **ReLU-derived**: Train 1-hidden-layer network, extract breakpoints/slopes

**C4. Asymptotic Saturation + Core**
- Tails: `GELU(x) ≈ x` for x ≫ 0, `GELU(x) ≈ 0` for x ≪ 0
- Core [−3, 3]: minimax polynomial or rational
- Smooth transition functions at boundaries (e.g., quintic blend)
- **Critical for BF16 dynamic range preservation**

---

### Category D: Hybrid & LUT-Based

**D1. LUT + Interpolation** (reference baseline)
- 64–256 entries for Φ(x) or GELU(x)/x
- Linear or quadratic interpolation between entries

**D2. LUT Tails + Polynomial Center**
- Table handles |x| > 3 (BF16 dynamic range critical)
- Dense polynomial/rational approximation in [−3, 3]

**D3. Lookup + Polynomial Correction**
- Coarse LUT provides base value
- Low-degree polynomial refines locally

---

### Category E: BF16-Specific Optimizations

*Apply these as "knobs" to any winning method from A–D.*

**E1. Monotonicity/Bounds-Constrained Fitting**
- Enforce: monotonic Φ, 0 ≤ Φ ≤ 1, bounded derivatives
- Objective: minimize **max-ULP** (not L²)
- Prevents "catastrophic" ULP spikes from sign/derivative violations

**E2. Coefficient Quantization**
- Fit (float64) → quantize (BF16) → re-evaluate ULP → iterate
- Find quantization-robust coefficients

**E3. Range-Scaled Approximation**
- Fit over x/s instead of x, restore scale after
- Align s with BF16 exponent boundaries
- Reduces catastrophic cancellation

**E4. Breakpoint/Knot Constraints**
- Restrict to BF16-representable values (powers of 2)
- Exploit symmetry to halve storage/computation

**E5. Denormal/Subnormal Policy**
- Explicit clamp or flush-to-zero below BF16 denormal threshold
- Verify impact on ULP and usable dynamic range

**E6. FMA-Aware Coefficient Sets**
- Horner's scheme: FMA-friendly, sequential
- Estrin's scheme: parallel, no FMA dependency
- Generate separate optimized coefficients for each

**E7. Coefficient Sensitivity Testing**
- Perturb/round coefficients, re-measure max-ULP
- Ensure robustness to ±1 ULP coefficient perturbation

---

### Category F: Reference & Ground-Truth

**F1. High-Precision Exact GELU**
- float64 or MPFR arbitrary-precision implementation
- Ground truth for all ULP measurements

**F2. Numerical Quadrature Φ(x)**
- Gauss-Legendre or Simpson's rule on fixed grid
- Uses only arithmetic — slow but configurable precision
- Alternative ground truth avoiding special functions

**F3. Continued Fraction erf/Φ**
- Alternative stable reference block
- Sometimes more accurate in tails than polynomial

**F4. Abramowitz-Stegun Erf Polynomial**
- Documented coefficients with known error bounds (~1.5×10⁻⁷)
- Historical baseline for comparison

---

### Category G: Methodology & Evaluation

**G1. ULP Measurement Framework**
- Dense grid sampling + local max search
- Focus regions: near x=0, segment boundaries, tail transitions
- Metrics: **max-ULP**, mean-ULP, 99th percentile ULP
- Range [−8, 8] covers 99.9% of typical activations
- **Simulate BF16 rounding at each intermediate operation**

**G2. FMA vs Non-FMA Comparison**
- Generate separate coefficient sets for Horner vs Estrin
- Measure ULP difference (can be 30–50%)

**G3. Multi-Region Error Analysis**
- Near-zero (|x| < 0.5): high relative sensitivity
- Core (0.5 ≤ |x| ≤ 3): main operational range
- Tails (|x| > 3): saturation, dynamic range critical
- Report separate metrics per region

**G4. Backward Pass (GELU′)**
- Analytic derivative of chosen forward approximation
- Or separate fitted approximation for GELU′(x)
- Match smoothness/monotonicity with forward pass
- **Critical for training accuracy**

**G5. Breakpoint/Knot Optimization**
- EPSS: iteratively refine based on max error locations
- Dynamic programming: optimal non-uniform spacing
- Constrain to BF16-representable values

**G6. Coefficient Robustness Analysis**
- Perturbation testing (add noise, round, re-evaluate)
- Threshold: tolerate ±1 ULP coefficient perturbation

**G7. Cost Model**
- Mul/add count, branches, LUT loads
- Vectorization friendliness (SIMD width utilization)

**G8. Regression Suite**
- Fixed seeds, fixed ranges [−8, 8]
- Targeted adversarial points: segment stitches, near-zero, tail transitions

---

### Category H: Advanced & Research-Inspired (Optional)

*Lower priority — implement after core methods validated.*

**H1. Inverted GELU (GELU⁻¹)**
- Approximate inverse for backward pass memory savings
- Recompute activations from gradients
- Measure gradient ULP

**H2. Combined GELU-Softmax Unit**
- Reuse hardware via shared PWL exp approximation
- Integer-based for efficiency
- Evaluate in transformer inference context

---

### Quick Reference: Key Formulas

| Method | Formula |
|--------|---------|
| Sigmoid (simple) | `σ(z) = 0.5 + z / (2(1 + |z|))` |
| Sigmoid (higher-order) | `σ(z) = 0.5 + z(c₁ + c₂z²) / (1 + c₃|z| + c₄z²)` |
| Tanh (simple) | `tanh(z) ≈ z(27 + z²) / (27 + 9z²)` |
| GELU via sigmoid | `GELU(x) ≈ x · σ(1.702x)` |
| GELU via tanh | `GELU(x) ≈ 0.5x(1 + tanh(0.7979(x + 0.0447x³)))` |
| Saturation thresholds | `x > 4: GELU ≈ x`, `x < −4: GELU ≈ 0` |

---

### Implementation Priority

| Phase | Methods | Rationale |
|-------|---------|-----------|
| **0** | F1, G1, G8 | Establish ground truth + measurement infrastructure |
| **1** | A1 (poly-7), A2 ([4/4]), B1 (sigmoid), C4 (sat+core) | Core baselines, simple to implement |
| **2** | C1 (cubic 8-seg), C3 (ISPA 16-seg), B2 (K-TanH) | Best ULP control candidates |
| **3** | E1–E7 applied to Phase 1–2 winners | Hardware-realistic BF16 variants |
| **4** | G2 (FMA), G3 (multi-region), G4 (backward) | Validation & training support |
| **5** | H1, H2 (if needed) | Advanced/research extensions |

---

### Recommended Minimal Baseline Set

| ID | Method | Purpose |
|----|--------|---------|
| **R1** | C4 (saturation + minimax poly-7 core) | Best tail handling baseline |
| **R2** | A2 (rational [4/4]) | Direct rational baseline |
| **R3** | C3 (PWL, power-of-2 breakpoints) | Fast baseline |
| **R4** | B2 (tanh-form + odd rational) | Classic form baseline |
| **R5** | D1 (LUT + linear interpolation) | Strong reference among arithmetic-only |

---

### Key Insights

1. **Tail saturation is critical** — BF16's 7-bit mantissa makes |x| > 3 highly ULP-sensitive
2. **Power-of-2 breakpoints** — Reduces quantization error in piecewise methods
3. **Monotonicity constraints** — Prevents unphysical artifacts from coefficient rounding
4. **FMA matters** — Can reduce ULP by 30–50% vs non-FMA evaluation
5. **Backward pass is essential** — Training accuracy depends on GELU′(x) quality
6. **Multi-region analysis** — Methods excel in different ranges; report separately
7. **Coefficient quantization** — Post-fit quantization can double ULP; iterative refit compensates

---

**Total: 30 core strategies** across 8 categories, with 7 BF16 optimization knobs, 8-item methodology framework, 5-phase implementation plan, and 5-item minimal baseline set.

---




---
# Final List 3:

# **FINAL GELU APPROXIMATION STRATEGY LIST**
## *Comprehensive Reference for BF16 ULP Analysis Using Basic Arithmetic Only*

---

## **CONSTRAINTS & SCOPE**

**Arithmetic Operations:** `+`, `−`, `×`, `÷`, `|·|`, `sign()` only  
**Prohibited Functions:** `erf()`, `tanh()`, `exp()`, `log()`  
**Target Format:** BFloat16 (7-bit mantissa, 8-bit exponent)  
**Primary Metric:** ULP (Units in Last Place), not MSE  
**Input Range:** `[−8, 8]` covering 99.9% of typical neural network activations  
**Goal:** Reference implementations for systematic ULP characterization

---

## **CATEGORY A: DIRECT GELU APPROXIMATIONS**

### **A1. Minimax Polynomial (Remez Algorithm)**
- **Method:** Fit polynomial directly to `GELU(x)` with equiripple error distribution
- **Degrees:** 5, 7, 9 (odd for anti-symmetry exploitation)
- **Range:** `[−5, 5]` or `[−8, 8]`
- **Evaluation:** Horner's scheme (FMA-friendly) or Estrin's scheme (parallel)
- **Expected ULP:** ~1–5

### **A2. Rational Function (Padé/Minimax)**
- **Formula:** `GELU(x) ≈ x · P_n(x²) / Q_m(x²)` (even powers preserve symmetry)
- **Orders:** [3/3], [4/4], [5/5]
- **Fitting:** Remez for minimax, least-squares for Padé
- **Advantage:** Superior convergence in tails vs polynomials
- **Expected ULP:** ~0.5–2

### **A3. Chebyshev Polynomial**
- **Method:** Near-optimal with bounded oscillating error
- **Use Case:** Alternative when Remez algorithm is unstable
- **Degrees:** Same as A1 (5, 7, 9)
- **Evaluation:** Horner's scheme

### **A4. Continued Fraction**
- **Application:** Alternative to Padé for GELU or erf/Φ
- **Advantage:** Sometimes more stable in tail regions
- **Implementation:** Truncate at various depths for accuracy/cost tradeoff

---

## **CATEGORY B: SUB-FUNCTION APPROXIMATIONS**

### **B1. Sigmoid-Based (Without Exp)**
- **Formula:** `GELU(x) ≈ x · σ(kx)` where `k ≈ 1.702`
- **Rational Variants:**
  - Simple: `σ(z) ≈ 0.5 + z / (2(1 + |z|))`
  - Higher-order: `σ(z) ≈ 0.5 + z(c₁ + c₂z²) / (1 + c₃|z| + c₄z²)`
- **Constraints:** Enforce monotonicity and `0 ≤ σ ≤ 1`
- **Fast Variant:** Piecewise linear σ as ultra-fast baseline
- **Expected ULP:** ~2–4

### **B2. Tanh-Form with Rational Tanh (K-TanH Inspired)**
- **Standard Form:** `0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))`
- **Tanh Replacements:**
  - Simple: `tanh(z) ≈ z(27 + z²) / (27 + 9z²)`
  - Better: Odd minimax rational fitted to tanh
- **Optimization:** Tune coefficients specifically for BF16 ULP
- **Expected ULP:** ~1–2

### **B3. Erf Polynomial → Φ → GELU**
- **Chain:** `erf(x) → Φ(x) = 0.5(1 + erf(x/√2)) → GELU(x) = x·Φ(x)`
- **Erf Approximation:** `erf(x) ≈ Σ aᵢx^(2i+1)` with clamp to `±1`
- **Baseline:** Abramowitz-Stegun coefficients (documented error ~1.5×10⁻⁷)
- **Range:** Core polynomial in `[−a, a]`, saturation outside

### **B4. Rational Erf/Φ with Range Reduction**
- **Method:** Separate `P_n/Q_m` for `[0, a]` vs `[a, ∞)`
- **Symmetry:** Mirror for `x < 0`
- **Tails:** Saturate to `±1` (erf) or `0/1` (Φ)
- **Orders:** [3/3] to [5/5]

---

## **CATEGORY C: PIECEWISE METHODS**

### **C1. Piecewise Polynomial Splines**
- **Type:** Cubic or quadratic splines
- **Segments:** 4–16 intervals
- **Knot Optimization:**
  - **EPSS** (Error Peak Search Strategy): Iterative refinement at error peaks
  - **Dynamic Programming:** Optimal non-uniform spacing for fixed segment count
  - Minimize MSE or max-error per segment
- **Continuity:** Ensure C¹ or C² at boundaries
- **Symmetry:** Fit `x ≥ 0` only, mirror for negatives
- **Evaluation:** Horner's scheme within each segment
- **Expected ULP:** ~0.3–1

### **C2. Piecewise Rational Functions**
- **Method:** Different Padé approximant per segment
- **Typical Layout:** 3–5 segments: `(−∞, −3]`, `[−3, 0]`, `[0, 3]`, `[3, ∞)`
- **Orders:** [2/2] or [3/3] per segment
- **Advantage:** Fewer segments than polynomial for same accuracy

### **C3. Symmetry-Exploiting Piecewise Linear**
- **ISPA Variant** (Internal Symmetry Piecewise Approximation):
  - Approximate `x ≥ 0` only, mirror for negatives
  - 8–16 linear segments
  - **Power-of-2 breakpoints:** Use BF16-representable values (e.g., 0.5, 1, 2, 4)
  - Target: Sub-10⁻⁹ MSE with negligible DNN accuracy loss
- **NLI Variant** (Non-uniform Linear Interpolation):
  - Breakpoints via dynamic programming optimization
  - Hardware-friendly, FP16/BF16 inference optimized
- **ReLU Network-Derived:**
  - Train 1-hidden-layer ReLU network on GELU
  - Extract breakpoints and slopes as conditional segments
  - No LUT required, automatic segment discovery
- **Expected ULP:** ~0.2–0.8

### **C4. Asymptotic Saturation + Core Approximation**
- **Tail Behavior:**
  - `x ≫ 0` (e.g., `x > 4`): `GELU(x) ≈ x`
  - `x ≪ 0` (e.g., `x < −4`): `GELU(x) ≈ 0`
- **Core Region:** `[−3, 3]` with minimax polynomial or rational
- **Transition:** Smooth blending functions (e.g., quintic polynomials) at boundaries
- **Critical:** Preserves BF16 dynamic range in extreme values
- **Expected ULP:** ~1–3

---

## **CATEGORY D: HYBRID & LUT-BASED METHODS**

### **D1. LUT + Linear/Quadratic Interpolation**
- **Structure:** Coarse table (64–256 entries) for `Φ(x)` or `GELU(x)/x`
- **Interpolation:** Linear or quadratic between entries
- **Purpose:** Reference baseline for ULP comparison (near-optimal among arithmetic methods)

### **D2. LUT for Tails + Polynomial Center**
- **Tails:** Table handles `|x| > 3` (critical for BF16 dynamic range)
- **Center:** Dense polynomial/rational approximation in `[−3, 3]`
- **Advantage:** Minimizes relative errors where BF16 struggles most

### **D3. Lookup + Polynomial Correction**
- **Structure:** Coarse LUT provides base value
- **Refinement:** Low-degree polynomial corrects locally
- **Balance:** Speed (LUT) vs accuracy (polynomial)

---

## **CATEGORY E: BF16-SPECIFIC OPTIMIZATION KNOBS**
*Apply to any method from Categories A-D*

### **E1. ULP/Monotonicity-Constrained Fitting**
- **Constraints During Optimization:**
  - Monotonicity of `Φ(x)` and `GELU(x)`
  - Bounds: `0 ≤ Φ(x) ≤ 1`
  - Smoothness: bounded first/second derivatives
  - Sign consistency at segment boundaries
- **Objective:** Minimize **max-ULP** (not L² MSE)
- **Result:** Prevents "catastrophic" ULP spikes from rounding artifacts

### **E2. Coefficient Quantization to BF16**
- **Process:**
  1. Fit coefficients in float64
  2. Quantize to BF16-representable values
  3. Re-evaluate ULP after quantization
  4. Iterate: refit compensating for quantization
- **Goal:** Find quantization-robust coefficient sets
- **Impact:** Post-fit quantization can double ULP—refitting compensates

### **E3. Range-Scaled Approximation**
- **Method:** Approximate over `x/s` instead of `x`, then rescale
- **Scale Selection:** Align `s` with BF16 exponent boundaries
- **Example:** `s = 2` or `s = √2` for symmetric exponent coverage
- **Advantage:** Reduces catastrophic cancellation in subtraction-heavy formulas

### **E4. Breakpoint Constraints**
- **Restriction:** Limit breakpoints to BF16-representable values
- **Power-of-2 Strategy:** Use `2^k` breakpoints (0.25, 0.5, 1, 2, 4, 8...)
- **Benefit:** Reduces quantization ULP in piecewise methods
- **Symmetry:** Exploit anti-symmetry to halve computation

### **E5. Denormal/Subnormal Policy**
- **Policy:** Explicit clamp or flush-to-zero for `|x|` below BF16 denormal threshold
- **Threshold:** ~5.88 × 10⁻³⁹ for BF16
- **Impact:** Verify effect on ULP and dynamic range preservation
- **Variants:** Test gradual vs hard clamp

### **E6. FMA-Aware Coefficient Design**
- **Evaluation Schemes:**
  - **Horner's:** Sequential, FMA-friendly (fused multiply-add)
  - **Estrin's:** Parallel, no FMA dependency
- **Strategy:** Generate separate optimized coefficient sets for each scheme
- **Impact:** FMA can reduce ULP by 30–50% vs non-FMA evaluation

### **E7. Coefficient Sensitivity & Robustness Testing**
- **Perturbation Test:** Add small noise (±1 ULP) to fitted coefficients
- **Re-evaluate ULP:** Measure stability under perturbation
- **Threshold:** Coefficients should tolerate ±1 ULP perturbation without catastrophic failure
- **Purpose:** Assess brittleness of approximation to coefficient errors

---

## **CATEGORY F: REFERENCE & GROUND-TRUTH METHODS**

### **F1. High-Precision Exact GELU**
- **Implementation:** Float64, float128, or MPFR arbitrary precision
- **Formula:** Exact `GELU(x) = x · Φ(x)` using high-precision erf
- **Purpose:** Ground truth for all ULP measurements

### **F2. Numerical Quadrature for Φ(x)**
- **Methods:** Gauss-Legendre or Simpson's rule on fixed grid
- **Integral:** `Φ(x) = ∫_{-∞}^x (1/√(2π)) exp(-t²/2) dt`
- **Arithmetic Only:** Uses only `+, −, ×, ÷` (slow but pure arithmetic)
- **Use Case:** Independent validation of F1

### **F3. Continued Fraction erf/Φ**
- **Application:** Alternative to Padé for erf/Φ approximation
- **Advantage:** Sometimes more stable in tails
- **Convergence:** Test truncation at various depths

### **F4. Abramowitz-Stegun Erf Polynomial**
- **Source:** Well-documented coefficients with known error bounds
- **Formula:** Standard erf polynomial approximation
- **Accuracy:** Known max error ~1.5 × 10⁻⁷
- **Purpose:** Historical baseline for comparison

---

## **CATEGORY G: METHODOLOGY & EVALUATION FRAMEWORK**

### **G1. ULP Measurement Infrastructure**
- **Search Strategy:**
  - Dense grid sampling over `[−8, 8]`
  - Local maximum search near `x = 0` and segment boundaries
  - Segment-specific analysis (center vs tails)
- **Metrics:**
  - **Max-ULP:** Worst-case error
  - **Mean-ULP:** Average error
  - **99th Percentile ULP:** Tail behavior characterization
- **Simulation:** BF16 rounding at **each operation** (not just final result)
- **Range Justification:** `[−8, 8]` covers 99.9% of activations in transformers

### **G2. FMA vs Non-FMA Comparison**
- **Generate:** Separate coefficient sets for Horner (FMA) vs Estrin (parallel)
- **Measure:** ULP difference with/without FMA
- **Typical Impact:** 30–50% ULP reduction with FMA
- **Hardware Profiles:** Test both scenarios for portability

### **G3. Multi-Region Error Analysis**
- **Regions:**
  - **Near-zero:** `|x| < 0.5` (high relative error sensitivity)
  - **Core:** `0.5 ≤ |x| ≤ 3` (main operational range)
  - **Tails:** `|x| > 3` (saturation, dynamic range critical)
- **Metrics:** Report separate max-ULP, mean-ULP per region
- **Purpose:** Identify method strengths/weaknesses by input range

### **G4. Backward Pass (GELU Derivative) Approximation**
- **Options:**
  1. **Analytic derivative** of chosen forward approximation
  2. **Separate approximation** for `GELU'(x)` with matched smoothness
- **Formula:** `GELU'(x) = Φ(x) + x·φ(x)` where `φ(x) = (1/√(2π)) exp(-x²/2)`
- **Constraints:** Ensure derivative monotonicity and sign consistency
- **Critical:** Training accuracy depends on gradient approximation quality
- **Advanced:** Inverted GELU for memory-efficient backpropagation

### **G5. Cost Model Analysis**
- **Metrics:**
  - Multiply/add count per evaluation
  - Branch count (for piecewise methods)
  - LUT loads and memory footprint
  - Vectorization friendliness (SIMD potential)
- **Purpose:** Complexity vs accuracy tradeoffs
- **Application:** Compare operational cost across methods

### **G6. Breakpoint/Knot Optimization Algorithms**
- **EPSS** (Error Peak Search Strategy):
  - Iteratively refine breakpoints by locating error peaks
  - Move breakpoints toward high-error regions
  - Converge when max-error minimized
- **Dynamic Programming:**
  - Optimal non-uniform spacing for fixed segment count
  - Minimize cumulative error over entire range
- **BF16 Constraint:** Restrict breakpoints to BF16-representable values
- **Power-of-2 Strategy:** Preferentially use `2^k` breakpoints

### **G7. Regression Suite & Adversarial Testing**
- **Fixed Test Cases:**
  - Fixed random seeds for reproducibility
  - Dense sampling over `[−8, 8]`
- **Adversarial Points:**
  - Segment stitch boundaries
  - Near `x = 0` (inflection region)
  - Saturation thresholds
  - BF16 exponent boundaries
- **Purpose:** Catch edge-case ULP failures not found in random sampling

---

## **CATEGORY H: ADVANCED & RESEARCH-INSPIRED METHODS**
*Optional extensions beyond core reference implementations*

### **H1. Inverted GELU for Training Memory Reduction**
- **Concept:** Approximate `GELU⁻¹(y)` for backward pass storage savings
- **Advantage:** Recompute activations from gradients, reduces memory footprint
- **BF16 Adaptation:** Superior approximation quality vs low-bit quantization methods
- **Metric:** Combined forward + backward ULP for training stability

### **H2. Combined GELU-Softmax Arithmetic Unit**
- **Concept:** Reuse hardware for GELU via softmax's piecewise linear exp approximation
- **Implementation:** Shared multipliers/adders for both functions
- **Application:** Efficiency gains in transformer inference
- **Constraint:** Maintain accuracy for both GELU and softmax operations

### **H3. SoftEx-Inspired Exp Approximation**
- **Application:** Replace exp in tanh/sigmoid with arithmetic-only approximation
- **Method:** Mantissa refinement techniques for exp without transcendental functions
- **Use:** Enables tanh-based GELU (B2) without true `exp()`
- **ULP Trade-off:** Measure compounded error from nested approximations

---

## **IMPLEMENTATION PRIORITY FRAMEWORK**

### **Phase 0: Infrastructure (Required First)**
| ID | Method | Purpose | Priority |
|----|--------|---------|----------|
| F1 | High-precision GELU | Ground truth | ★★★★★ |
| G1 | ULP measurement | Framework | ★★★★★ |
| G7 | Regression suite | Validation | ★★★★☆ |

### **Phase 1: Core Baselines**
| ID | Method | Expected ULP | Priority |
|----|--------|--------------|----------|
| A1 | Minimax poly (7th, 9th) | ~1–5 | ★★★★★ |
| A2 | Rational [4/4], [5/5] | ~0.5–2 | ★★★★★ |
| B1 | Sigmoid rational | ~2–4 | ★★★★☆ |
| C4 | Saturation + core | ~1–3 | ★★★★★ |

### **Phase 2: Best ULP Control**
| ID | Method | Segments/Order | Expected ULP | Priority |
|----|--------|----------------|--------------|----------|
| C1 | Cubic spline | 8 segments | ~0.3–1 | ★★★★★ |
| C3 | ISPA PWL | 16 segments, pow2 | ~0.2–0.8 | ★★★★★ |
| B2 | K-TanH rational | — | ~1–2 | ★★★★☆ |

### **Phase 3: BF16-Optimized Variants**
| ID | Method | Optimization | Priority |
|----|--------|--------------|----------|
| E2 | Quantization | Apply to Phase 1-2 winners | ★★★★★ |
| E1 | Monotonicity | Constrained refit | ★★★★☆ |
| E6 | FMA variants | Horner vs Estrin | ★★★★☆ |
| D2 | Hybrid LUT+poly | Tail handling | ★★★★☆ |

### **Phase 4: Validation & Training Support**
| ID | Method | Purpose | Priority |
|----|--------|---------|----------|
| G2 | FMA comparison | Hardware profiles | ★★★★☆ |
| G3 | Region analysis | Range-specific ULP | ★★★★☆ |
| G4 | Backward pass | Training gradients | ★★★★☆ |
| G5 | Cost model | Complexity analysis | ★★★☆☆ |

### **Phase 5: Advanced & Research**
| ID | Method | Application | Priority |
|----|--------|-------------|----------|
| H1 | Inverted GELU | Memory optimization | ★★★☆☆ |
| F2 | Numerical quadrature | Alternative truth | ★★☆☆☆ |
| F3 | Continued fraction | Tail stability test | ★★☆☆☆ |
| E7 | Coefficient sensitivity | Robustness | ★★★☆☆ |

---

## **RECOMMENDED MINIMAL BASELINE SET**

**For comprehensive ULP characterization with minimal implementation effort:**

| # | ID | Method | Rationale |
|---|----|--------|-----------|
| **R1** | C4 | Saturation + minimax core (7th deg) | Best tail handling + simple |
| **R2** | A2 | Rational [4/4] direct GELU | Baseline rational performance |
| **R3** | C3 | ISPA PWL (power-of-2, 16 seg) | Fast, hardware-friendly |
| **R4** | B2 | Tanh-form + odd rational | Classical GELU formula |
| **R5** | D1 | LUT + linear interpolation | Reference upper bound |

**Minimal implementation: 5 core methods covering polynomial, rational, piecewise, classical, and LUT approaches.**

**For full validation, add:**
- **R6:** E2 (quantization) applied to R1-R4
- **R7:** G1 (full ULP framework)
- **R8:** G4 (backward pass for top 3 methods)

**Total: 8 implementations for comprehensive BF16 ULP assessment.**

---

## **QUICK REFERENCE: KEY FORMULAS**

| Method | Formula |
|--------|---------|
| **Sigmoid (simple)** | `σ(z) = 0.5 + z / (2(1 + |z|))` |
| **Tanh (simple)** | `tanh(z) ≈ z(27 + z²) / (27 + 9z²)` |
| **GELU via sigmoid** | `GELU(x) ≈ x · σ(1.702x)` |
| **GELU via tanh** | `GELU(x) ≈ 0.5x(1 + tanh(0.7979(x + 0.0447x³)))` |
| **Saturation thresholds** | `x > 4: GELU ≈ x`, `x < −4: GELU ≈ 0` |
| **GELU derivative** | `GELU'(x) = Φ(x) + x·φ(x)` |

---

## **KEY INSIGHTS & DESIGN PRINCIPLES**

✓ **Tail Saturation is Critical:** BF16's 7-bit mantissa makes `|x| > 3` regions extremely ULP-sensitive  
✓ **Power-of-2 Breakpoints:** Reduces quantization error in piecewise methods by ~30%  
✓ **Monotonicity Constraints:** Prevents unphysical artifacts from coefficient rounding  
✓ **FMA Matters:** Can reduce ULP by 30–50% vs non-FMA evaluation (hardware-dependent)  
✓ **Backward Pass is Essential:** Training accuracy depends critically on `GELU'(x)` quality  
✓ **Multi-Region Analysis:** Methods excel in different ranges—no single winner across all inputs  
✓ **Coefficient Quantization:** Post-fit quantization can double ULP—iterative refitting compensates  
✓ **Max-ULP vs MSE:** Optimize for max-ULP, not MSE—MSE hides catastrophic tail errors  

---

**TOTAL STRATEGIES:** 35 distinct methods organized into 8 categories  
**Implementation Phases:** 0–5 with explicit priorities  
**Minimal Set:** 5 baseline methods (R1-R5), 8 with validation (R1-R8)  

**Document Version:** Final Consolidated (January 2025)  
**Scope:** Reference implementations for BF16 ULP analysis using basic arithmetic only

---



---
# Final List 4: 

### Final GELU Approximation Strategies List

This final list synthesizes the best elements from all four Consolidated Lists, prioritizing BF16 ULP minimization with basic arithmetic only (+, −, ×, ÷, |·|, sign()). It adopts List 2's clean categorical structure (A-G) for navigability, List 1's BF16 knobs and methodology as dedicated sections (E and G), List 3's phased implementation with priorities and insights, and List 4's dense, implementation-ready descriptions. Redundancies are eliminated (e.g., minimax appears only in direct methods), and advanced methods (Category H) are included selectively for completeness without overemphasizing hardware-specific ideas. Total: 28 distinct strategies across 8 categories.

Constraints: No erf(), tanh(), exp(); focus on reference implementations for ULP assessment over [-8,8] (99.9% activation coverage). Use Remez/least-squares for fitting; simulate BF16 rounding per operation.

#### Category A: Direct GELU Approximations
A1. Minimax Polynomial (Remez)
- Fit directly to GELU(x) over [-8,8] for equiripple error; degrees 5, 7, 9 (odd for antisymmetry).
- Evaluate via Horner's (FMA-friendly) or Estrin's (parallel) scheme to reduce accumulation errors.
- Expected ULP: ~1-5; baseline for polynomial tradeoffs.

A2. Rational Function (Padé/Minimax)
- GELU(x) ≈ x · P_n(x²) / Q_m(x²) leveraging even powers for symmetry; orders [3/3], [4/4], [5/5].
- Fitting with ULP objectives; superior tail convergence vs. polynomials.
- Expected ULP: ~0.5-2; often outperforms polynomials in dynamic range.

A3. Chebyshev Polynomial
- Near-optimal with bounded oscillating error; alternative to minimax for stability.
- Same degrees and evaluation as A1; apply to GELU or sub-functions.

A4. Continued Fraction
- Alternative to Padé for GELU/erf with tail stability; truncate at depths for accuracy/complexity balance.

#### Category B: Sub-Function Approximations
B1. Sigmoid-Based (No Exp)
- GELU(x) ≈ x · σ(1.702x); simple rational σ(z) ≈ 0.5 + z / (2(1 + |z|)); higher-order σ(z) ≈ 0.5 + z(c₁ + c₂z²) / (1 + c₃|z| + c₄z²).
- Enforce monotonicity (0 ≤ σ ≤ 1) and bounded derivatives; piecewise linear σ as ultra-fast baseline.
- Expected ULP: ~2-4; simple yet effective for BF16.

B2. Tanh-Form with Rational Tanh (K-TanH)
- GELU(x) ≈ 0.5x (1 + tanh(√(2/π)(x + 0.044715 x³))); replace tanh with odd minimax rational, e.g., tanh(z) ≈ z(27 + z²) / (27 + 9z²).
- Optimize for BF16 ULP and monotonicity; SoftEx-inspired mantissa refinement for nested approximations if needed.
- Expected ULP: ~1-2; classic form with arithmetic-only tanh.

B3. Erf Polynomial → Φ → GELU
- erf(x) ≈ ∑ a_i x^{2i+1} (odd series) with clamp to ±1 outside core [-a,a]; derive Φ(x) = 0.5(1 + erf(x/√2)).
- Abramowitz-Stegun coefficients as baseline (~1.5×10^{-7} error); quantize for BF16.

B4. Rational Erf/Φ with Range Reduction
- Separate P/Q for [0, a] vs [a, ∞); mirror for negatives, saturate tails.
- Orders [3/3]-[5/5]; reduces degree needs vs. global fits.

#### Category C: Piecewise Methods
C1. Piecewise Polynomial Splines with Optimized Knots
- Cubic/quadratic over 4-16 segments; optimize knots via EPSS (error peak search) or dynamic programming (NLI-style) for MSE/max error minimization.
- Ensure C¹/C² continuity; exploit symmetry (fit x ≥ 0, mirror); Horner's per segment.
- Expected ULP: ~0.3-1; best local ULP control.

C2. Piecewise Rational
- Different Padé per 3-5 segments (e.g., (-∞, -3], [-3, 0], [0, 3], [3, ∞)).
- Fewer segments than polynomials for equivalent accuracy; smooth transitions.

C3. Piecewise Linear (ISPA/NLI)
- ISPA: Symmetry-exploiting positive axis with 8-16 segments, power-of-2 breakpoints (BF16-representable) for reduced rounding errors; sub-10^{-9} MSE target.
- NLI: Non-uniform via dynamic programming for FP16/BF16 optimization.
- ReLU-derived: Train 1-hidden-layer network, extract conditional segments without LUTs.
- Expected ULP: ~0.2-0.8; efficient baseline.

C4. Asymptotic Saturation + Core Approximation
- Tails (|x| > 3-5): GELU(x) ≈ x (positive) or 0 (negative); core [-3,3]: minimax polynomial/rational.
- Smooth blending at boundaries; critical for BF16 dynamic range.
- Expected ULP: ~1-3; simple tail handling.

#### Category D: Hybrid & LUT-Based
D1. LUT + Interpolation (Reference Baseline)
- 64-256 entries for Φ(x) or GELU(x)/x; linear/quadratic interpolation.
- Near-optimal ULP benchmark among arithmetic methods.

D2. LUT Tails + Polynomial Center
- Table for |x| > 3 to preserve BF16 range; dense polynomial/rational in [-3,3].
- Minimizes relative errors in tails.

D3. Non-Uniform Linear Interpolation (NLI)
- Piecewise linear with dynamic programming-optimized breakpoints; hardware-friendly for low-latency inference.

D4. Lookup + Polynomial Correction
- Coarse LUT base + low-degree polynomial refinement; balances table size and computation.

#### Category E: BF16-Specific Optimizations (Apply to Any Method)
E1. ULP/Monotonicity-Constrained Fitting
- During Remez/least-squares: enforce monotonicity of Φ/GELU, bounds (0 ≤ Φ ≤ 1), bounded derivatives/smoothness.
- Objective: Minimize max-ULP over MSE; prevents artifacts from rounding.

E2. Coefficient Quantization
- Fit in float64 → quantize to BF16-representable → re-evaluate ULP → iterate/refit for robustness.
- Simulate hardware constraints; tolerate ±1 ULP perturbations.

E3. Range-Scaled Approximation
- Fit over x/s, rescale output; choose s to align with BF16 exponent boundaries (e.g., s=√2).
- Reduces catastrophic cancellation/subtraction errors.

E4. Breakpoint/Knot Constraints
- Restrict to BF16-representable values (e.g., powers of 2); exploit symmetry for efficiency.

E5. Denormal/Subnormal Policy
- Explicit clamp or flush-to-zero for |x| < BF16 threshold (~5.88×10^{-39}); verify dynamic range/ULP impact.

E6. FMA-Aware Coefficient Sets
- Separate fits for Horner's (FMA-friendly, sequential) vs. Estrin's (parallel, no FMA); measure ULP differences (up to 30-50% reduction with FMA).

E7. Coefficient Sensitivity/Robustness Testing
- Perturb/round coefficients (±1 ULP), re-measure max-ULP; ensure no catastrophic failure.

#### Category F: Reference & Ground-Truth Methods
F1. High-Precision Exact GELU
- Float64/float128/MPFR with erf; ground truth for all ULP measurements.

F2. Numerical Quadrature for Φ(x)
- Gauss-Legendre/Simpson on fixed grid; arithmetic-only slow baseline.

F3. Continued Fraction Erf/Φ
- Tail-stable alternative to Padé; truncate for configurable precision.

F4. Abramowitz-Stegun Erf Polynomial
- Documented coefficients with known bounds (~1.5×10^{-7}); historical baseline.

#### Category G: Methodology & Evaluation Framework
G1. ULP Measurement Framework
- Dense grid sampling + local max search (near 0, boundaries, tails); metrics: max/mean/99th percentile ULP.
- Range [-8,8]; simulate BF16 rounding at each op (not just final); separate fp32-accum vs. bf16-accum scenarios.

G2. FMA vs. Non-FMA Comparison
- Generate separate coefficient sets; quantify ULP impact.

G3. Backward Pass (GELU')
- Analytic derivative of forward approx or separate fit; match smoothness/monotonicity.
- Inverted GELU variant for memory-efficient backprop; measure gradient ULP for training stability.

G4. Breakpoint/Knot Optimization
- EPSS: Iterative refinement at error peaks; dynamic programming: Optimal non-uniform spacing.
- Constrain to BF16-representable; validate continuity.

G5. Multi-Region Error Analysis
- Near-zero (|x|<0.5: relative sensitivity), core (0.5≤|x|≤3), tails (|x|>3: dynamic range); report metrics per region, positive/negative separately.

G6. Coefficient Robustness Analysis
- Perturbation testing; threshold for ±1 ULP tolerance without failure.

G7. Cost Model
- Mul/add count, branches, LUT loads, vectorization friendliness; compare complexity vs. ULP tradeoff.

G8. Regression Suite
- Fixed seeds/ranges + adversarial points (stitches, near-zero); ensure reproducibility.

#### Category H: Advanced & Research-Inspired Methods
H1. Inverted GELU for Training Memory Reduction
- Approximate GELU^{-1}(y) for recomputing activations from gradients; adapt to BF16; superior vs. low-bit methods for gradient quality.

H2. Combined GELU-Softmax Arithmetic Unit
- Reuse shared multipliers/adders for GELU via piecewise linear exp from softmax; integer-based; evaluate ULP in transformers (arithmetic-only compliance).

H3. SoftEx-Inspired Exp Approximation
- Arithmetic mantissa refinement for exp in tanh/sigmoid; enables nested approx without true exp; measure compounded ULP (5x speedup potential).

#### Implementation Priority
| Phase | Methods | Rationale | Expected ULP |
|-------|---------|-----------|--------------|
| 0: Infrastructure | F1 (ground truth), G1 (ULP framework), G8 (regression suite) | Establish validation baseline | N/A |
| 1: Core Baselines | A1 (poly-7), A2 ([4/4]), B1 (sigmoid rational), C4 (saturation + core) | Simple, well-understood; quick prototyping | ~1-5 |
| 2: Best ULP Control | C1 (cubic spline 8-seg), C3 (ISPA 16-seg pow2), B2 (K-TanH rational) | Localized optimization for tails/center | ~0.2-1 |
| 3: BF16-Optimized | E1-E7 applied to Phase 1-2 winners | Hardware-realistic variants; quantization robustness | ~0.5-2 (post-quant) |
| 4: Validation & Training | G2 (FMA comparison), G3 (backward pass), G5 (multi-region), H1 (inverted GELU) | Full error profiling + training support | Varies |
| 5: Advanced | D1 (LUT interp), H2 (combined unit), H3 (SoftEx exp) | Benchmarks + research extensions | ~0.1-1 |

#### Recommended Minimal Implementation Set
1. F1: High-precision ground truth.
2. A1: Minimax poly-7 (polynomial baseline).
3. A2: Padé [4/4] (rational baseline).
4. C1: Cubic spline 8-seg (piecewise winner).
5. C3: ISPA PWL 16-seg (fast baseline).
6. E2: Quantization on #2-5 (BF16 variants).
7. G1 + G3: ULP framework + backward pass.
8. G7: Regression suite for testing.

#### Quick Reference: Key Formulas
| Method | Formula |
|--------|---------|
| Sigmoid (simple) | σ(z) = 0.5 + z / (2(1 + \|z\|)) |
| Sigmoid (higher-order) | σ(z) = 0.5 + z(c₁ + c₂z²) / (1 + c₃\|z\| + c₄z²) |
| Tanh (simple) | tanh(z) ≈ z(27 + z²) / (27 + 9z²) |
| GELU via sigmoid | GELU(x) ≈ x · σ(1.702x) |
| GELU via tanh | GELU(x) ≈ 0.5x (1 + tanh(0.7979(x + 0.0447x³))) |
| Saturation thresholds | x > 4: GELU ≈ x; x < -4: GELU ≈ 0 |

#### Key Insights & Design Principles
- Tail saturation is critical: BF16's 7-bit mantissa makes |x| > 3 ULP-sensitive; always clamp to preserve dynamic range.
- Power-of-2 breakpoints: Reduces quantization error in piecewise methods; ideal for BF16 representability.
- Monotonicity constraints: Prevents unphysical artifacts from coefficient rounding; enforce 0 ≤ Φ ≤ 1 and positive derivatives.
- FMA matters: Can reduce ULP by 30-50% vs. non-FMA; use Horner's for sequential ops.
- Backward pass is essential: Training accuracy depends on GELU' quality; inverted GELU saves memory in low-precision.
- Multi-region analysis: Methods excel differently (e.g., rational in tails, piecewise near zero); report separately.
- Coefficient quantization: Post-fit can double ULP—refit compensates; test robustness to perturbations.