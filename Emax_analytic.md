# Analytic Computation of E_max for AdS₄ Oscillons at Large ℓ

## 1. Setup and Conventions

### Background geometry

Global AdS₄ with L = 1 (Λ = −3):

$$ds^2 = -(1+r^2)\,dt^2 + \frac{dr^2}{1+r^2} + r^2\,d\Omega_2^2$$

### Gravitational coupling

The code uses 8πG_N = 1, i.e.:

$$G_{\mu\nu} + \Lambda\,g_{\mu\nu} = T_{\mu\nu}$$

So **G_N = 1/(8π) ≈ 0.0398**.

### Scalar field

Mass–dimension relation: m² = Δ(Δ − 3).

Klein–Gordon equation: □φ = m²φ (no self-interaction potential — the nonlinearity is purely gravitational).

### Linear normal mode

The fundamental (n = 0) mode:

$$\phi_1 = R_{0,\ell}(r)\,Y_\ell^0(\theta)\,\cos(\omega_0 t)$$

with frequency ω₀ = Δ + ℓ and radial profile:

$$R_{0,\ell}(r) = C_\ell\,r^\ell\,(1+r^2)^{-(\Delta+\ell)/2}$$

Peak at r* = √(ℓ/Δ). We normalize so R₀(r*) = 1:

$$C_\ell = r_*^{-\ell}\,(1+r_*^2)^{(\Delta+\ell)/2}$$

---

## 2. Perturbative Expansion

The oscillon is a one-parameter family of solutions labeled by amplitude ε:

$$\phi = \varepsilon\,\phi_1 + \varepsilon^3\,\phi_3 + O(\varepsilon^5)$$
$$g_{\mu\nu} = \mathring{g}_{\mu\nu} + \varepsilon^2\,h_{\mu\nu} + O(\varepsilon^4)$$
$$\omega = \omega_0 + \varepsilon^2\,\omega_2 + O(\varepsilon^4)$$

Here ε is defined so that at the mode peak:

$$\phi_{\rm max} = \varepsilon\,Y_\ell^0(0) = \varepsilon\,\sqrt{\frac{2\ell+1}{4\pi}}$$

The ADM energy (mass) expands as:

$$\boxed{M(\varepsilon) = E_2\,\varepsilon^2 + E_4\,\varepsilon^4 + O(\varepsilon^6)}$$

where E₂ > 0 (matter energy) and E₄ < 0 (gravitational binding energy). The maximum mass occurs at dM/dε = 0.

---

## 3. Computing E₂ (Matter Energy at Leading Order)

### ADM mass integral

$$M = \int_\Sigma \frac{T_{00}}{1+r^2} \cdot \frac{r^2\sin\theta}{\sqrt{1+r^2}}\;dr\,d\theta\,d\varphi$$

For φ₁ = ε F cos(ω₀t) with F(r,θ) = R₀(r) Y_ℓ⁰(θ), time-averaging gives:

$$\langle T_{00}\rangle = \frac{\varepsilon^2}{4}\Bigl[\frac{\omega_0^2\,F^2}{1+r^2} + (1+r^2)(\partial_r F)^2 + \frac{(\partial_\theta F)^2}{r^2} + m^2 F^2\Bigr]$$

### Virial relation

The linear mode satisfies the eigenvalue equation (from □φ₁ = m²φ₁):

$$\frac{1}{r^2}\partial_r\bigl[r^2(1+r^2)\partial_r F\bigr] + \frac{1}{r^2\sin\theta}\partial_\theta(\sin\theta\,\partial_\theta F) + \Bigl[\frac{\omega_0^2}{1+r^2} - m^2\Bigr]F = 0$$

Multiplying by F and integrating against √g₃ d³x, integration by parts gives the virial identity:

$$\int\Bigl[(1+r^2)(\partial_r F)^2 + \frac{(\partial_\theta F)^2}{r^2}\Bigr]\frac{r^2\sin\theta}{\sqrt{1+r^2}}\,dr\,d\theta\,d\varphi = \int\Bigl[\frac{\omega_0^2}{1+r^2} - m^2\Bigr]F^2\,\frac{r^2\sin\theta}{\sqrt{1+r^2}}\,dr\,d\theta\,d\varphi$$

This means the gradient energy equals the (ω₀² − m²) potential energy. Using this in the time-averaged T₀₀:

$$E_2 = \frac{1}{2}\,\omega_0^2\,\mathcal{I}$$

where the **norm integral** is:

$$\mathcal{I} \equiv \int_0^\infty R_0^2(r)\,\frac{r^2}{(1+r^2)^{3/2}}\,dr$$

(The angular integral ∫|Y_ℓ⁰|² dΩ = 1 has been used.)

### Evaluating I at large ℓ

Near the peak r*, substitute s = ln(r/r*) so that dr = r ds. With R₀ ≈ exp(−Δs²) and r²/(1+r²)^{3/2} ≈ 1/r for r* ≫ 1:

$$\mathcal{I} = \int_{-\infty}^{\infty} e^{-2\Delta s^2}\,\frac{r_*^2 e^{2s}}{(r_*^2 e^{2s})^{3/2}}\,r_* e^s\,ds = \int_{-\infty}^{\infty} e^{-2\Delta s^2}\,ds$$

$$\boxed{\mathcal{I} = \sqrt{\frac{\pi}{2\Delta}}}$$

**This is independent of ℓ.** All the ℓ-dependence in E₂ comes from ω₀²:

$$\boxed{E_2 = \frac{\omega_0^2}{2}\sqrt{\frac{\pi}{2\Delta}} = \frac{(\Delta+\ell)^2}{2}\sqrt{\frac{\pi}{2\Delta}}}$$

### Consistency check

For ℓ = 0 and Δ = 4: ω₀ = 4, E₂ = 8√(π/8) ≈ 5.01. This should be checked against the numerical mode energy.

---

## 4. Computing E₄ (Gravitational Self-Energy)

### Metric perturbation at O(ε²)

The time-averaged stress tensor sources a static metric perturbation. In the Newtonian limit (valid when the mode is extended, r* ≫ 1):

$$\nabla^2 \Phi_N = 4\pi G_N\,\rho_{\rm eff}$$

where ρ_eff = ⟨T⁰₀⟩ is the time-averaged energy density as measured by the ADM decomposition:

$$\rho_{\rm eff} = \frac{\langle T_{00}\rangle}{(1+r^2)} \approx \frac{\varepsilon^2\,\omega_0^2\,F^2}{4(1+r^2)^2}$$

### Gravitational self-energy (monopole approximation)

The binding energy at O(ε⁴) is:

$$E_4 = -\frac{1}{2}\int\rho_{\rm eff}\,\Phi_N\;\sqrt{g_3}\,d^3x$$

In the monopole approximation (which dominates since the source is localized at r*), the gravitational potential at the mode is:

$$\Phi_N(r_*) \approx -\frac{G_N M}{r_*} = -\frac{G_N\,E_2\,\varepsilon^2}{r_*}$$

and the self-energy of the mass distribution is:

$$E_{\rm grav} = -\frac{G_N\,M^2}{2\,r_*} = -\frac{G_N\,(E_2\,\varepsilon^2)^2}{2\,r_*}$$

Therefore:

$$\boxed{E_4 = -\frac{G_N\,E_2^2}{2\,r_*}}$$

### Multipole corrections

The energy density has angular structure ∝ |Y_ℓ⁰(θ)|², which couples to even multipoles L = 0, 2, 4, ..., 2ℓ through the Clebsch–Gordan integral:

$$\int (Y_\ell^0)^2\,Y_L^0\,d\Omega = \sqrt{\frac{(2\ell+1)^2(2L+1)}{(4\pi)^3}}\,\begin{pmatrix}\ell & \ell & L \\ 0 & 0 & 0\end{pmatrix}^2$$

For L ≥ 2, the radial part contributes r*^L/r*^{L+1} ~ 1/r*, same as the monopole. However, the angular Clebsch–Gordan coefficients scale as O(1/ℓ) for L ≥ 2. So the monopole term dominates at large ℓ, and the corrections are O(1/ℓ) relative.

---

## 5. E_max from the Turning Point

### Energy as a function of amplitude

$$M(\varepsilon) = E_2\,\varepsilon^2 - \frac{G_N\,E_2^2}{2\,r_*}\,\varepsilon^4$$

$$= E_2\,\varepsilon^2\Bigl(1 - \frac{\varepsilon^2}{\varepsilon_c^2}\Bigr)$$

where the **critical amplitude** is:

$$\varepsilon_c^2 \equiv \frac{2\,r_*}{G_N\,E_2}$$

### Turning point

dM/dε = 0 gives ε²_max = ε²_c / 2, and:

$$\boxed{M_{\rm max} = \frac{E_2\,\varepsilon_c^2}{4} = \frac{r_*}{2\,G_N}}$$

### Remarkable cancellation

**E_max depends only on r* and G_N — not on ω₀, Δ, or the mode profile.**

Substituting r* = √(ℓ/Δ) and G_N = 1/(8π):

$$\boxed{E_{\rm max} = 4\pi\,\sqrt{\frac{\ell}{\Delta}}}$$

### Explicit values

| ℓ | Δ | r* | E_max (analytic) |
|---|---|-----|-----------------|
| 2 | 4 | 0.707 | 8.89 |
| 2 | 6 | 0.577 | 7.26 |
| 2 | 10 | 0.447 | 5.62 |
| 4 | 4 | 1.000 | 12.57 |
| 4 | 6 | 0.816 | 10.27 |
| 4 | 10 | 0.632 | 7.95 |
| 6 | 4 | 1.225 | 15.40 |
| 6 | 6 | 1.000 | 12.57 |
| 6 | 10 | 0.775 | 9.74 |
| 8 | 4 | 1.414 | 17.77 |
| 10 | 4 | 1.581 | 19.87 |
| 10 | 6 | 1.291 | 16.23 |
| 10 | 10 | 1.000 | 12.57 |
| 15 | 4 | 1.936 | 24.33 |
| 20 | 4 | 2.236 | 28.10 |
| 40 | 10 | 2.000 | 25.13 |

---

## 6. Critical Amplitude and Field Value

### ε at E_max

$$\varepsilon_{\rm max}^2 = \frac{r_*}{G_N\,E_2} = \frac{\sqrt{\ell/\Delta}}{G_N\,\frac{(\Delta+\ell)^2}{2}\sqrt{\frac{\pi}{2\Delta}}} = \frac{2\sqrt{2}\,\ell^{1/2}\,\Delta^{1/2}}{G_N\,(\Delta+\ell)^2\,\sqrt{\pi/\Delta}} = \frac{2\sqrt{2}\,\Delta\,\ell^{1/2}}{G_N\,(\Delta+\ell)^2\,\sqrt{\pi}}$$

For ℓ ≫ Δ:

$$\varepsilon_{\rm max}^2 \approx \frac{2\sqrt{2}\,\Delta}{\sqrt{\pi}\,G_N\,\ell^{3/2}} = \frac{16\sqrt{2}\,\pi^{1/2}\,\Delta}{\ell^{3/2}}$$

(using G_N = 1/(8π))

### Peak scalar field at E_max

$$\phi_{\rm max} = \varepsilon_{\rm max}\,Y_\ell^0(0) = \varepsilon_{\rm max}\,\sqrt{\frac{2\ell+1}{4\pi}}$$

For large ℓ:

$$\phi_{\rm max}^{\rm crit} \approx \sqrt{\frac{2\ell+1}{4\pi}} \times \frac{4\sqrt[4]{2}\,\pi^{1/4}\,\Delta^{1/2}}{\ell^{3/4}} \sim \frac{c\,\Delta^{1/2}}{\ell^{1/4}}$$

So the **peak field value at E_max decreases slowly as ℓ⁻¹/⁴**: high-ℓ oscillons can store more energy despite having weaker fields, because the energy is distributed over a larger volume.

---

## 7. Frequency at E_max

### Nonlinear frequency shift ω₂

The gravitational potential modifies the scalar equation. The effective frequency equation becomes:

$$\frac{\omega^2}{(1+r^2)(1 + 2\Phi_N)} \approx \frac{\omega_0^2}{1+r^2}\Bigl(1 + \frac{2\varepsilon^2\omega_2}{\omega_0} - \frac{2\Phi_N}{1+r^2}\Bigr)$$

The solvability condition at O(ε³) (Fredholm alternative) gives:

$$\omega_2 = \omega_0\,\frac{\langle F^2\,\Phi_N/(1+r^2)^2\rangle}{\langle F^2/(1+r^2)\rangle} \sim \frac{\omega_0\,G_N\,E_2}{r_*^3}$$

**Step by step:**

The gravitational potential at the mode:

$$\Phi_N(r_*) = -G_N\,E_2\,\varepsilon^2 / r_*$$

The effective potential seen by the scalar includes a factor 1/(1+r²) from the redshift:

$$\frac{\Phi_N}{1+r_*^2} \approx \frac{-G_N\,E_2\,\varepsilon^2}{r_*^3}$$

The solvability condition matches this to the frequency shift:

$$\omega_2 \approx \frac{\omega_0\,G_N\,E_2}{r_*^3}$$

### Frequency shift at ε_max

$$\delta\omega = \omega_2\,\varepsilon_{\rm max}^2 = \frac{\omega_0\,G_N\,E_2}{r_*^3} \times \frac{r_*}{G_N\,E_2} = \frac{\omega_0}{r_*^2} = \frac{(\Delta+\ell)\,\Delta}{\ell}$$

For ℓ ≫ Δ:

$$\boxed{\delta\omega_{\rm max} \approx \Delta}$$

$$\boxed{\omega(E_{\rm max}) \approx \ell + \Delta - c\,\Delta \approx \ell + (1-c)\,\Delta}$$

where c is an O(1) constant. The **fractional frequency shift δω/ω₀ ~ Δ/ℓ → 0** as ℓ → ∞: the frequency barely changes from the linear value, even at maximum energy.

---

## 8. Validity and Corrections

### When does the large-ℓ approximation hold?

The key assumption is r* ≫ 1, i.e., ℓ ≫ Δ. For the numerical branches:

| (ℓ, Δ) | r* | r* ≫ 1? |
|---------|-----|---------|
| (2, 10) | 0.45 | No — corrections large |
| (4, 4) | 1.00 | Marginal |
| (6, 6) | 1.00 | Marginal |
| (10, 4) | 1.58 | Starting to work |
| (15, 4) | 1.94 | OK |
| (20, 4) | 2.24 | Good |

So the low-ℓ branches (2, 4) are **not in the large-ℓ regime**. The analytic E_max will overestimate the true value there because:

1. The 1/(1+r²) factors cannot be replaced by 1/r²
2. The mode shape is not well-approximated by the log-normal
3. AdS curvature effects modify the gravitational self-energy

### Beyond the monopole

The quadrupole (L=2) and higher multipole corrections to E₄ increase the binding energy by a factor:

$$1 + \sum_{L=2,4,...}^{2\ell} \frac{(2L+1)}{r_*^{2L}} \times \bigl|\text{3j symbol}\bigr|^2 \times (\text{radial integral})$$

For r* ≫ 1, the L ≥ 2 terms are suppressed by 1/r*² relative to the monopole.

### Beyond quartic order

The ε⁶ and higher corrections to M(ε) shift E_max by relative amounts of order ε²_max / ε²_c ~ 1. So the quartic approximation gives the right scaling but the numerical prefactor has O(1) uncertainty. A more precise computation would require the full numerical solution (which is what the code does).

---

## 9. Comparison with Flat-Space Boson Stars

For comparison, the Kaup limit for a free mini boson star in flat space:

$$M_{\rm Kaup} = 0.633\,\frac{M_{\rm Pl}^2}{m}$$

In our units with 8πG = 1: M_Pl² = 1/(8πG) = 1, and m = √(Δ(Δ−3)):

$$M_{\rm Kaup} = \frac{0.633}{\sqrt{\Delta(\Delta-3)}}$$

The AdS oscillon E_max at large ℓ:

$$E_{\rm max} = 4\pi\sqrt{\ell/\Delta}$$

These scale completely differently:
- Flat boson star: E_max ~ 1/m (independent of ℓ)  
- AdS oscillon at large ℓ: E_max ~ √ℓ (growing with ℓ)

The reason: the flat-space boson star is self-bound (gravity determines its size). The AdS oscillon is bound by the centrifugal + AdS potential (gravity is a perturbation). Higher ℓ spreads the mode over a larger region, allowing more mass before gravitational collapse.

---

## 10. Summary

$$\boxed{E_{\rm max} = \frac{r_*}{2\,G_N} = \frac{\sqrt{\ell/\Delta}}{2\,G_N} = 4\pi\sqrt{\frac{\ell}{\Delta}}}$$

$$\boxed{\omega(E_{\rm max}) \approx \Delta + \ell - O(\Delta)}$$

$$\boxed{\phi_{\rm max}^{\rm crit} \sim \frac{\Delta^{1/2}}{\ell^{1/4}}\;\text{(decreasing slowly)}}$$

The result E_max = r*/(2G_N) has an elegant interpretation: **the maximum energy of the oscillon equals the gravitational mass that would produce an O(1) Newtonian potential at the mode's peak radius**. This is the AdS analogue of the Chandrasekhar/Kaup limit, adapted to a mode trapped by the centrifugal barrier rather than self-bound.
