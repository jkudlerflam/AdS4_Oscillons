# Large-ℓ Asymptotics for Oscillons in AdS₄

## 1. Setup

Global AdS₄ metric (L = 1, so Λ = -3):

$$ds^2 = -(1+r^2)\,dt^2 + \frac{dr^2}{1+r^2} + r^2\,(d\theta^2 + \sin^2\theta\,d\varphi^2)$$

Scalar field mass–dimension relation in AdS₄ (boundary dimension d = 3):

$$m^2 = \Delta(\Delta - 3)$$

| Δ  | m²  |
|----|-----|
| 4  | 4   |
| 6  | 18  |
| 10 | 70  |

## 2. Exact Linear Modes

The linearized Klein–Gordon equation □φ = m²φ on this background, with
φ = e^{-iωt} R(r) Y_ℓ^0(θ), admits normalizable solutions:

$$R_{n,\ell}(r) \propto r^\ell\,(1+r^2)^{-(\Delta+\ell)/2}\;{}_2F_1\!\bigl(-n,\,\Delta+\ell+n;\,\ell+\tfrac{3}{2};\,\tfrac{r^2}{1+r^2}\bigr)$$

with eigenfrequencies

$$\boxed{\omega_{n,\ell} = \Delta + 2n + \ell}$$

The fundamental (n = 0) has ₂F₁ = 1, so:

$$R_{0,\ell}(r) = C\,r^\ell\,(1+r^2)^{-(\Delta+\ell)/2}$$

### Check against numerics

The code starts the continuation at small amplitude, so the initial ω should approach ω₀ = Δ + ℓ:

| ℓ | Δ  | ω₀ = Δ + ℓ | ω (first CSV row) |
|---|-----|------------|-----|
| 2 | 10  | 12         | 11.87 (at w ≈ 0.47) |
| 4 | 6   | 10         | 9.90 (at w ≈ 0.98)  |

The ~1% deviations are because the first branch point is at finite amplitude, not ε → 0.

## 3. Radial Profile at Large ℓ

### Peak location

Setting d(ln R₀)/dr = 0:

$$\frac{\ell}{r} - \frac{(\Delta+\ell)\,r}{1+r^2} = 0 \quad\Longrightarrow\quad r_*^2 = \frac{\ell}{\Delta}$$

$$\boxed{r_* = \sqrt{\ell/\Delta}}$$

### Width (WKB envelope)

Substitute r = r* e^s near the peak. For ℓ ≫ Δ (so r* ≫ 1):

$$\ln R_{0,\ell} = \ell\,\ln(r_* e^s) - \tfrac{\Delta+\ell}{2}\,\ln(1 + r_*^2 e^{2s})$$

With 1 + r*²e^{2s} ≈ r*²e^{2s} for r* ≫ 1:

$$\approx \ell(\ln r_* + s) - \tfrac{\Delta+\ell}{2}(2\ln r_* + 2s) = -\Delta\,\ln r_* - \Delta\,s$$

This gives only the power-law envelope R ~ r^{-Δ}. The Gaussian localization comes from the next-order correction. Keeping 1/r*² terms:

$$\ln(1 + r_*^2 e^{2s}) = 2\ln r_* + 2s + \frac{e^{-2s}}{r_*^2} + O(r_*^{-4})$$

$$\ln R \approx \text{const} - \Delta\,s - \frac{\Delta}{2}\,e^{-2s}$$

Setting d/ds = 0: −Δ + Δ e^{-2s} = 0  ⟹  s = 0. ✓ (peak at r = r*)

Second derivative at s = 0:

$$\frac{d^2\!\ln R}{ds^2}\bigg|_{s=0} = -2\Delta$$

So near the peak:

$$R_{0,\ell}(r) \propto \exp\!\Bigl(-\Delta\,\bigl(\ln(r/r_*)\bigr)^2\Bigr)$$

This is a **log-normal profile** with width in ln(r/r*):

$$\boxed{\sigma_{\ln r} = \frac{1}{\sqrt{2\Delta}}}$$

Width in physical radius: σ_r = r*/√(2Δ) = √(ℓ/(2Δ²)).

**Key point:** The fractional width σ_r/r* = 1/√(2Δ) is **independent of ℓ**. In rescaled coordinates r/r*, the mode shape is universal (depends only on Δ).

## 4. Energy Scaling at Small Amplitude

At small amplitude, φ = ε R_{0,ℓ}(r) Y_ℓ^0(θ) cos(ωt) + O(ε²).

### Normalization convention

From the code: w = φ̂(τ=0, reference point) ≈ φ_max at small amplitude.

$$\phi_{\max} = A \cdot R_{0,\ell}(r_*) \cdot Y_\ell^0(0) = A \cdot \sqrt{\frac{2\ell+1}{4\pi}}$$

where A is the amplitude coefficient (with R normalized so R(r*) = 1). Thus:

$$A = w \cdot \sqrt{\frac{4\pi}{2\ell+1}} \approx \frac{w\sqrt{4\pi}}{\sqrt{2\ell}} \quad (\ell \gg 1)$$

### Radial norm integral

With R(r*) = 1, R ∝ exp(−Δ(ln r/r*)²):

$$I_{\rm rad} \equiv \int_0^\infty R^2\,r^2\,dr$$

Substituting r = r* e^s, dr = r* e^s ds:

$$I_{\rm rad} = r_*^3 \int_{-\infty}^{\infty} e^{-2\Delta s^2}\,e^{3s}\,ds = r_*^3\,\exp\!\Bigl(\frac{9}{8\Delta}\Bigr)\,\sqrt{\frac{\pi}{2\Delta}}$$

For large Δ the exponential correction is O(1). The key scaling is:

$$I_{\rm rad} \sim r_*^3\,\Delta^{-1/2} = \bigl(\ell/\Delta\bigr)^{3/2}\,\Delta^{-1/2} = \frac{\ell^{3/2}}{\Delta^2}$$

### Energy

$$E = \frac{1}{2}\int\!\Bigl[\frac{\omega_0^2}{1+r^2}\,\phi^2 + (1+r^2)(\partial_r\phi)^2 + \frac{\ell(\ell+1)}{r^2}\,\phi^2 + m^2\phi^2\Bigr]\,\frac{r^2}{\sqrt{1+r^2}}\,\sin\theta\;dr\,d\theta\,d\varphi$$

At leading order the ω₀² and m² terms dominate. With the angular normalization ∫|Y_ℓ^0|²dΩ = 1:

$$E \approx \frac{1}{2}A^2(\omega_0^2 + m^2)\int_0^\infty R^2\,\frac{r^2}{\sqrt{1+r^2}}\,dr$$

For r* ≫ 1: 1/√(1+r²) ≈ 1/r near the mode peak, so the integral becomes:

$$\int R^2\,\frac{r^2}{\sqrt{1+r^2}}\,dr \approx r_*^2 \sqrt{\frac{\pi}{2\Delta}}\;\exp\!\Bigl(\frac{1}{\Delta}\Bigr)$$

Combining A² ~ w²/ℓ and ω₀² + m² = (Δ+ℓ)² + Δ(Δ-3) ~ ℓ² for ℓ ≫ Δ:

$$\boxed{E_{\rm scalar} \sim \frac{w^2\,\ell^2\,(\ell/\Delta)}{\ell\,\sqrt{\Delta}} = \frac{w^2\,\ell^2}{\Delta^{3/2}} \quad (\ell \gg \Delta)}$$

So at fixed amplitude w, the energy grows as **ℓ²** at large ℓ.

### Consistency check

Defining the ratio R_E ≡ E_scalar / w² at the first (small-amplitude) branch point, we predict:

$$R_E(\ell, \Delta) \propto \frac{\ell^2}{\Delta^{3/2}}$$

This can be checked from the initial row of each CSV.

## 5. Maximum Energy (Chandrasekhar-Type Bound)

The oscillon reaches its maximum energy when the gravitational self-interaction becomes O(1). The metric perturbation at the mode location scales as:

$$h \sim \frac{8\pi G_N\,E}{r_*}$$

(Newtonian potential of mass E at distance r*, valid since r* ≫ L for ℓ ≫ Δ).

Setting h ~ 1 gives the critical energy:

$$E_{\max} \sim \frac{r_*}{8\pi G_N} = \frac{1}{8\pi G_N}\sqrt{\frac{\ell}{\Delta}}$$

$$\boxed{E_{\max} \propto \sqrt{\ell/\Delta}}$$

The critical amplitude at E_max:

$$w_{\rm crit}^2 \sim \frac{E_{\max}\,\Delta^{3/2}}{\ell^2} \sim \frac{\Delta}{\ell^{3/2}}$$

$$\boxed{w_{\rm crit} \propto \Delta^{1/2}\,\ell^{-3/4}}$$

So at large ℓ: E_max **increases** as √ℓ, but the maximum field amplitude **decreases** as ℓ^{−3/4}. The energy is spread over the larger volume of the mode.

## 6. Frequency at E_max

The nonlinear frequency shift from gravitational self-interaction is:

$$\delta\omega \equiv \omega_0 - \omega \sim \frac{G_N\,E\,\Delta}{\sqrt{\ell}}$$

(The Δ/√ℓ factor comes from the local energy density × curvature radius.)

At E = E_max:

$$\delta\omega_{\max} \sim \frac{G_N}{\sqrt{\ell}} \cdot \frac{\sqrt{\ell}}{G_N\sqrt{\Delta}} \cdot \Delta = \sqrt{\Delta}$$

So:

$$\omega(E_{\max}) \approx \Delta + \ell - c\sqrt{\Delta}$$

for some O(1) constant c. The fractional shift δω/ω₀ → 0 as ℓ → ∞: the oscillon frequency stays close to the linear value even at maximum energy.

## 7. Summary of Scaling Predictions

| Quantity | Scaling (ℓ ≫ Δ) |
|----------|-----------------|
| Linear frequency ω₀ | Δ + ℓ |
| Mode peak radius r* | √(ℓ/Δ) |
| Mode width σ_r | √(ℓ/(2Δ²)) |
| Fractional width σ_r/r* | 1/√(2Δ) (ℓ-independent) |
| Energy at fixed w | w² ℓ² / Δ^{3/2} |
| E_max | √(ℓ/Δ) / G_N |
| w at E_max | Δ^{1/2} ℓ^{−3/4} |
| Frequency at E_max | Δ + ℓ − O(√Δ) |

## 8. What the Numerics Can Check Now

Even without reaching E_max, the existing data at small amplitude can verify:

1. **ω₀ = Δ + ℓ**: extrapolate ω to w → 0 for each branch
2. **E/w² ∝ ℓ² / Δ^{3/2}**: compare the initial E_scalar/w² across branches
3. **dω/dw² ∝ ℓ²/Δ^{1/2}**: the initial slope of ω vs w² should scale predictably
4. **φ_max/w ratio**: should depend on ℓ through the Y_ℓ^0 normalization

If E_max ~ √ℓ is confirmed, it has an important physical interpretation: higher angular momentum oscillons can store more energy before becoming relativistic, analogous to how the Kerr bound increases with spin for rotating black holes.
