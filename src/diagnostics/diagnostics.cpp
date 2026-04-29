// diagnostics.cpp — Solution diagnostics implementation
//
// Constraint monitoring: Re-evaluates the PDE residuals at interior points
//   WITHOUT the ρ²·Ω³ regularization, to get physical constraint violations.
//
// Spectral tails: Forward-transforms field data in each direction and
//   reports coefficient magnitude decay.
//
// Physical observables: Volume integrals using Gauss quadrature on the
//   spectral collocation grid.

#include "diagnostics/diagnostics.h"
#include "geometry/ads4.h"
#include "spectral/chebyshev.h"
#include "spectral/legendre.h"
#include "spectral/fourier.h"
#include <cmath>
#include <cstring>
#include <algorithm>
#include <numeric>

namespace diagnostics {

using namespace geometry;
using namespace solver;

static const double PI = 3.14159265358979323846;

// ============================================================================
// Constraint monitoring
// ============================================================================
// We reuse the OscillonSystem's derivative machinery indirectly:
// The residual vector R already contains equation residuals at each point.
// But those residuals have ρ²·Ω³ regularization. For physical constraint
// monitoring, we want the UN-regularized values.
//
// Strategy: compute the full residual vector, then divide out the
// regularization factors to recover physical residuals.

ConstraintData computeConstraints(
    const OscillonSystem& sys,
    const double* u, double omega)
{
    ConstraintData cd;
    std::memset(&cd, 0, sizeof(cd));

    int n = sys.stateSize() - 1;
    std::vector<double> R(n, 0.0);
    sys.computeResidual(u, omega, R.data());

    int nT = sys.nTau();
    int nR = sys.nRadial();
    int nA = sys.nAngular();
    int nS = sys.nSpatial();

    double sum_trK2 = 0, sum_evol2 = 0;
    int n_interior = 0;

    const auto& params = sys.params();
    int n_origin = params.N_nuc;
    int n_boundary = params.N_nuc + 1;

    for (int m = 0; m < nT; m++) {
        for (int nn = 0; nn < nR; nn++) {
            // Skip boundary points
            if (nn == n_origin || nn == n_boundary) continue;

            double rho = sys.rhoGrid(nn);
            double Om = (1.0 - rho*rho) / (1.0 + rho*rho);
            double rho2 = rho * rho;
            double Om3 = Om * Om * Om;
            double reg = rho2 * Om3;

            for (int j = 0; j < nA; j++) {
                int si = nn * nA + j;

                // Skip odd fields at time endpoints
                bool at_endpoint = (m == 0 || m == params.N_t);

                // trK (stored in FLD_LAPSE slot) — regularized by nothing
                // Actually trK has no ρ²·Ω³ factor in the residual
                double trK_val = R[sys.stateIdx(FLD_LAPSE, m, si)];
                if (!at_endpoint) {
                    cd.max_trK = std::max(cd.max_trK, std::abs(trK_val));
                    sum_trK2 += trK_val * trK_val;
                }

                // V^rho (stored in FLD_SHIFT_R slot) — regularized by ρ²
                double Vr_reg = R[sys.stateIdx(FLD_SHIFT_R, m, si)];
                double Vr_phys = (rho2 > 1e-20) ? Vr_reg / rho2 : Vr_reg;
                if (!at_endpoint || isTimeParity_even(FLD_SHIFT_R)) {
                    // shifts are odd, so at endpoints R = u (BC), not PDE residual
                    if (!at_endpoint)
                        cd.max_V_rho = std::max(cd.max_V_rho, std::abs(Vr_phys));
                }

                // V^theta similarly
                double Vt_reg = R[sys.stateIdx(FLD_SHIFT_T, m, si)];
                double Vt_phys = (rho2 > 1e-20) ? Vt_reg / rho2 : Vt_reg;
                if (!at_endpoint)
                    cd.max_V_theta = std::max(cd.max_V_theta, std::abs(Vt_phys));

                // Evolution equations (FLD_GAMMA_* slots) — regularized by ρ²·Ω³
                for (int c = 0; c < 4; c++) {
                    int fld = FLD_GAMMA_RR + c;
                    double evol_reg = R[sys.stateIdx(fld, m, si)];
                    double evol_phys = (reg > 1e-20) ? evol_reg / reg : evol_reg;
                    if (!at_endpoint || isTimeParity_even(fld)) {
                        cd.max_evol_res = std::max(cd.max_evol_res, std::abs(evol_phys));
                        sum_evol2 += evol_phys * evol_phys;
                    }
                }

                // Scalar wave equation (no extra regularization)
                double sw_val = R[sys.stateIdx(FLD_SCALAR, m, si)];
                if (!at_endpoint)
                    cd.max_scalar_res = std::max(cd.max_scalar_res, std::abs(sw_val));

                if (!at_endpoint) n_interior++;
            }
        }
    }

    cd.n_interior = n_interior;
    cd.rms_trK = (n_interior > 0) ? std::sqrt(sum_trK2 / n_interior) : 0.0;
    cd.rms_evol = (n_interior > 0) ? std::sqrt(sum_evol2 / n_interior) : 0.0;

    return cd;
}

// ============================================================================
// Spectral tail decay
// ============================================================================

SpectralTails computeSpectralTails(
    const OscillonSystem& sys,
    const double* u)
{
    SpectralTails st;

    int nT = sys.nTau();
    int nR = sys.nRadial();
    int nA = sys.nAngular();
    int nS = sys.nSpatial();
    const auto& params = sys.params();

    // Focus on the scalar field for tail analysis (field 7)
    int fld = FLD_SCALAR;

    // --- Fourier tails: at each spatial point, forward transform in time ---
    st.fourier_tail.resize(nT, 0.0);
    {
        spectral::Fourier four(params.N_t);
        for (int si = 0; si < nS; si++) {
            std::vector<double> v(nT);
            for (int m = 0; m < nT; m++)
                v[m] = u[sys.stateIdx(fld, m, si)];

            // Scalar is even-parity: cosine transform
            std::vector<double> a(nT);
            four.forwardCos(v.data(), a.data());
            for (int k = 0; k < nT; k++)
                st.fourier_tail[k] = std::max(st.fourier_tail[k], std::abs(a[k]));
        }
    }

    // --- Chebyshev tails: at each (time, angular) point, forward in radial ---
    // Use the nucleus domain for Chebyshev analysis
    int N_nuc = params.N_nuc;
    st.chebyshev_tail.resize(N_nuc + 1, 0.0);
    {
        spectral::Chebyshev cheb(N_nuc);
        for (int m = 0; m < nT; m++) {
            for (int j = 0; j < nA; j++) {
                // Extract radial values in the nucleus domain
                std::vector<double> f_nuc(N_nuc + 1);
                for (int i = 0; i <= N_nuc; i++)
                    f_nuc[i] = u[sys.stateIdx(fld, m, i * nA + j)];

                // Forward Chebyshev transform
                std::vector<double> coeffs(N_nuc + 1);
                cheb.forward(f_nuc.data(), coeffs.data());
                for (int k = 0; k <= N_nuc; k++)
                    st.chebyshev_tail[k] = std::max(st.chebyshev_tail[k], std::abs(coeffs[k]));
            }
        }
    }

    // --- Legendre tails: at each (time, radial) point, forward in angular ---
    st.legendre_tail.resize(nA, 0.0);
    {
        spectral::Legendre leg(params.N_theta);
        for (int m = 0; m < nT; m++) {
            for (int nn = 0; nn < nR; nn++) {
                std::vector<double> f_ang(nA);
                for (int j = 0; j < nA; j++)
                    f_ang[j] = u[sys.stateIdx(fld, m, nn * nA + j)];

                std::vector<double> coeffs(nA);
                leg.forward(f_ang.data(), coeffs.data(), nA - 1);
                for (int l = 0; l < nA; l++)
                    st.legendre_tail[l] = std::max(st.legendre_tail[l], std::abs(coeffs[l]));
            }
        }
    }

    // Compute decay rates
    auto decayRate = [](const std::vector<double>& tail) -> double {
        if (tail.size() < 3) return 0.0;
        // Find first and last significant coefficients
        double first = 0, last = 0;
        int i_first = -1, i_last = -1;
        for (int i = 0; i < (int)tail.size(); i++) {
            if (tail[i] > 1e-16) {
                if (i_first < 0) { i_first = i; first = tail[i]; }
                i_last = i; last = tail[i];
            }
        }
        if (i_first < 0 || i_first == i_last) return 0.0;
        return -std::log10(last / first) / (i_last - i_first);
    };

    st.cheb_decay_rate = decayRate(st.chebyshev_tail);
    st.leg_decay_rate = decayRate(st.legendre_tail);
    st.four_decay_rate = decayRate(st.fourier_tail);

    // Digits: -log10(last_tail / max_tail)
    auto digits = [](const std::vector<double>& tail) -> double {
        double mx = *std::max_element(tail.begin(), tail.end());
        double last = tail.back();
        if (mx < 1e-30 || last < 1e-30) return 15.0; // effectively zero
        return -std::log10(last / mx);
    };

    st.cheb_digits = digits(st.chebyshev_tail);
    st.leg_digits = digits(st.legendre_tail);
    st.four_digits = digits(st.fourier_tail);

    return st;
}

// ============================================================================
// Physical observables
// ============================================================================

PhysicalObservables computeObservables(
    const OscillonSystem& sys,
    const double* u, double omega)
{
    PhysicalObservables obs;
    obs.omega = omega;
    obs.w = sys.normalization(u);
    obs.phi_max = 0.0;
    obs.E_scalar = 0.0;
    obs.dNhat_dr_boundary = 0.0;

    int nT = sys.nTau();
    int nR = sys.nRadial();
    int nA = sys.nAngular();
    const auto& params = sys.params();

    // --- Max |phi_hat| (proxy for max |phi| since phi = phi_hat * Omega^{Delta/2}) ---
    for (int m = 0; m < nT; m++)
        for (int si = 0; si < sys.nSpatial(); si++) {
            double val = std::abs(u[sys.stateIdx(FLD_SCALAR, m, si)]);
            obs.phi_max = std::max(obs.phi_max, val);
        }

    // --- Time-averaged scalar energy ---
    // On the AdS₄ background, the time-averaged energy of a scalar field
    // φ(t,r,θ) = Φ(r,θ)cos(ωt) is (using the virial relation for the
    // linear mode, valid to leading order):
    //
    //   E = ω² ∫ φ² / (1+r²) · r²sinθ/√(1+r²) dr dθ dφ   (time-averaged)
    //     = (ω²/2) ∫ Φ² / (1+r²) · r²sinθ/√(1+r²) dr dθ dφ
    //
    // This is the ω²·I term from the analytic calculation. It equals the
    // full energy E₂ε² at leading order because the virial relation gives
    // gradient energy = kinetic energy - mass term, so total = 2·kinetic - mass
    // = 2·(ω²/2)·I - m²·I = (ω² - m²)·I + ω²·I... no, let's be precise.
    //
    // Actually, for a general (possibly nonlinear) time-periodic solution,
    // compute the full T₀₀ integral:
    //   E = ∫ T₀₀/N · √γ d³x, time-averaged
    //     = (1/2) ∫ [(∂ₜφ)²/N² + (1+r²)(∂ᵣφ)² + (∂θφ)²/r² + m²φ²]
    //              · r²sinθ/√(1+r²) dr dθ dφ, time-averaged
    //
    // For φ periodic in t, <(∂ₜφ)²> = ω²<φ²> at leading order.
    // Including ω²<φ²>/N² gives the full kinetic contribution, and at
    // leading order the virial relation ensures:
    //   E = ω² ∫ φ²/(1+r²) · r²sinθ/√(1+r²) dr dθ dφ
    //
    // Beyond leading order, this underestimates the energy because it
    // misses the nonlinear metric corrections. But it's correct to O(ε²)
    // and gives the right scaling.
    //
    // Converting to compactified coordinates: r = 2ρ/(1-ρ²), dr = 2(1+ρ²)/(1-ρ²)² dρ
    //   1+r² = (1+ρ²)²/(1-ρ²)²
    //   r² = 4ρ²/(1-ρ²)²
    //   √(1+r²) = (1+ρ²)/(1-ρ²)
    //   φ = φ̂ · Ω^{Δ/2} where Ω = (1-ρ²)/(1+ρ²)
    //
    // The integrand in compactified coords:
    //   φ²/(1+r²) · r²sinθ/√(1+r²) · dr
    //   = φ̂²·Ω^Δ · (1-ρ²)²/(1+ρ²)² · 4ρ²/(1-ρ²)² · sinθ · (1-ρ²)/(1+ρ²) · 2(1+ρ²)/(1-ρ²)² dρ
    //   = φ̂²·Ω^Δ · 8ρ²sinθ/(1+ρ²)² · 1/(1-ρ²) dρ
    //
    // With Ω = (1-ρ²)/(1+ρ²):
    //   Ω^Δ/(1+ρ²)² · 1/(1-ρ²) = (1-ρ²)^{Δ-1} / (1+ρ²)^{Δ+2}

    // Trapezoid weights in time
    std::vector<double> t_weights(nT);
    double dt_val = PI / params.N_t;
    for (int m = 0; m < nT; m++) {
        t_weights[m] = dt_val;
        if (m == 0 || m == params.N_t) t_weights[m] *= 0.5;
    }
    double T_period = PI; // half-period (symmetry)
    for (auto& tw : t_weights) tw /= T_period;

    for (int m = 0; m < nT; m++) {
        for (int nn = 0; nn < nR; nn++) {
            double rho = sys.rhoGrid(nn);
            double rho2 = rho * rho;
            double opr2 = 1.0 + rho2;  // 1 + ρ²
            double omr2 = 1.0 - rho2;  // 1 - ρ²

            // Skip boundary (ρ=1 where Ω=0, φ=0)
            if (std::abs(omr2) < 1e-14) continue;

            // Radial part of integrand (without φ̂² and sinθ):
            // 8ρ² · (1-ρ²)^{Δ-1} / (1+ρ²)^{Δ+2}
            double radial_factor = 8.0 * rho2
                * std::pow(omr2, params.Delta - 1.0)
                / std::pow(opr2, params.Delta + 2.0);

            // Radial quadrature weight (midpoint rule on Chebyshev nodes)
            double dr_weight = 0.0;
            if (nn > 0 && nn < nR - 1) {
                dr_weight = 0.5 * std::abs(sys.rhoGrid(nn+1) - sys.rhoGrid(nn-1));
            } else if (nn == 0) {
                dr_weight = 0.5 * std::abs(sys.rhoGrid(1) - sys.rhoGrid(0));
            } else {
                dr_weight = 0.5 * std::abs(sys.rhoGrid(nR-1) - sys.rhoGrid(nR-2));
            }

            for (int j = 0; j < nA; j++) {
                double theta = sys.thetaGrid(j);
                double sth = std::sin(theta);

                double dtheta_weight = PI / nA; // rough angular weight

                int si = nn * nA + j;
                double phi_hat = u[sys.stateIdx(FLD_SCALAR, m, si)];

                // Integrand: ω² · φ̂² · radial_factor · sinθ · dρ · dθ · 2π (azimuthal)
                double dV = radial_factor * sth * dr_weight * dtheta_weight * 2.0 * PI;
                obs.E_scalar += t_weights[m] * omega * omega * phi_hat * phi_hat * dV;
            }
        }
    }

    // --- Boundary derivative of N_hat (AMD mass proxy) ---
    // N_hat is stored in field slot FLD_LAPSE. At the boundary (ρ=1),
    // N_hat = 1 (Dirichlet BC). The radial derivative at the boundary
    // encodes the mass information.
    // We extract d_ρ(N_hat) at ρ=1 using spectral differentiation of the
    // shell domain. For simplicity, use finite difference from the last
    // two shell points.
    int n_boundary = params.N_nuc + 1; // global index of ρ=1
    if (n_boundary + 1 < nR) {
        // Average over angular points at m=0 (time-symmetric slice)
        double sum_dN = 0;
        for (int j = 0; j < nA; j++) {
            int si_bdy = n_boundary * nA + j;
            int si_near = (n_boundary + 1) * nA + j;
            double N_bdy = u[sys.stateIdx(FLD_LAPSE, 0, si_bdy)];
            double N_near = u[sys.stateIdx(FLD_LAPSE, 0, si_near)];
            double rho_bdy = sys.rhoGrid(n_boundary);
            double rho_near = sys.rhoGrid(n_boundary + 1);
            double dr = rho_near - rho_bdy;
            if (std::abs(dr) > 1e-15)
                sum_dN += (N_near - N_bdy) / dr;
        }
        obs.dNhat_dr_boundary = sum_dN / nA;
    }

    return obs;
}

// ============================================================================
// Full diagnostic report
// ============================================================================

DiagnosticReport fullDiagnostics(
    const OscillonSystem& sys,
    const double* u, double omega)
{
    DiagnosticReport report;
    report.constraints = computeConstraints(sys, u, omega);
    report.tails = computeSpectralTails(sys, u);
    report.observables = computeObservables(sys, u, omega);
    return report;
}

} // namespace diagnostics
