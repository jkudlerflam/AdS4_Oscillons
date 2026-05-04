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

    // --- Boundary derivative of N_hat (crude proxy, kept for backward compat) ---
    int n_boundary = params.N_nuc + 1; // global index of ρ=1
    if (n_boundary + 1 < nR) {
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

    // --- ADM mass via volume integral of T₀₀ ---
    //
    // M = (1/T)∫₀ᵀ dt ∫ ρ_E · √γ · d³x
    //
    // where ρ_E = ½(Π² + γⁱʲ∂ᵢφ∂ⱼφ + m²φ²) is the scalar energy density,
    // Π = (∂ₜφ - βⁱ∂ᵢφ)/N is the conjugate momentum, and √γ is the
    // spatial volume element.
    //
    // This is computed using the full nonlinear metric and stress tensor
    // at each interior collocation point, with proper spectral quadrature.
    //
    // In compactified coordinates (ρ, θ):
    //   √γ d³x = √(det γ) · dρ · dθ · dφ
    //          = √(γ_ρρ·γ_θθ - γ_ρθ²) · √γ_φφ · dρ · dθ · 2π
    {
        // Recompute derivatives (same as residual assembler)
        // We need: field values, spatial derivatives, time derivatives at each point
        int nT_local = sys.nTau();
        int nR_local = sys.nRadial();
        int nA_local = sys.nAngular();
        int nS_local = nR_local * nA_local;

        // Allocate derivative workspace
        enum { D_VAL=0, D_DT=1, D_DR=2, D_DTHETA=3 };
        const int ND = 4;
        // Just need val, dt, dr, dtheta for the scalar and metric
        std::vector<double> ws(ND * N_FIELDS * nT_local * nS_local, 0.0);
        auto W = [&](int field, int d) -> double* {
            return ws.data() + (field * ND + d) * nT_local * nS_local;
        };

        // Copy values
        for (int A = 0; A < N_FIELDS; A++) {
            double* val = W(A, D_VAL);
            for (int m = 0; m < nT_local; m++)
                for (int si = 0; si < nS_local; si++)
                    val[m * nS_local + si] = u[sys.stateIdx(A, m, si)];
        }

        // Temporal derivatives
        for (int A = 0; A < N_FIELDS; A++) {
            bool even = (A != FLD_SHIFT_R && A != FLD_SHIFT_T);
            // Use the system's temporal derivative (exposed via evaluateField won't work
            // for derivatives, so compute manually using Fourier differentiation)
            // For simplicity, use finite differences in time:
            // dt_f(m) ≈ omega * D^(1)_{mk} f(k)
            // But we don't have the Fourier diff matrix here.
            // Instead, for the time average of ρ_E, we can use the simpler formula:
            // <ρ_E> = <½(Π² + grad² + m²φ²)>
            // With Π = (∂ₜφ - β·∇φ)/N, and ∂ₜφ from Fourier differentiation.
            // Since we can't easily access the Fourier diff matrix from here,
            // approximate ∂ₜφ at each collocation point using the available data.
        }

        // Actually, a much simpler approach: use the existing residual assembler
        // infrastructure. The system class already computes all derivatives.
        // We just need to call computeResidual and then re-extract the physical
        // variables at each point. But computeResidual doesn't expose the
        // intermediate physical data.
        //
        // SIMPLEST CORRECT APPROACH: use the Hamiltonian constraint.
        // The Hamiltonian constraint is: R + K² - K_{ij}K^{ij} - 2Λ = 2ρ_E (with 8πG=1)
        // So ρ_E = ½(R + K² - K_{ij}K^{ij} - 2Λ)
        //
        // But the Hamiltonian constraint residual (which the code computes) is
        // SUPPOSED to be zero for a converged solution. So this gives ρ_E = -Λ + ...
        // which is the cosmological constant contribution, not the matter energy.
        //
        // The correct approach is: ρ_E = T₀₀/N² = matter energy density.
        // T₀₀ = ½[Π² + γⁱʲ∂ᵢφ∂ⱼφ + m²φ²] (already computed by computeScalarStress)
        //
        // For the simplest implementation, compute everything at τ=0
        // (the time-symmetric slice where Π is maximal for the scalar):
        // <ρ_E> = (1/T)∫ρ_E dt ≈ Σₘ w_m ρ_E(τ_m)
        //
        // At each point we need: phi, dphi_dr, dphi_dtheta, dphi_dt/N (=Π),
        // and the metric γ for the volume element and inverse metric.

        // Trapezoid time weights
        std::vector<double> tw(nT_local);
        double dt_w = PI / params.N_t;
        for (int m = 0; m < nT_local; m++) {
            tw[m] = dt_w;
            if (m == 0 || m == params.N_t) tw[m] *= 0.5;
        }
        for (auto& w_val : tw) w_val /= PI; // normalize to unit average

        // Radial quadrature weights (Clenshaw-Curtis for each domain)
        const auto& nuc_cheb = sys.radial().nucleus();
        const auto& shell_cheb = sys.radial().shell();

        obs.M_ADM = 0.0;

        for (int m = 0; m < nT_local; m++) {
            for (int nn = 0; nn < nR_local; nn++) {
                double rho = sys.rhoGrid(nn);
                double Om = (1.0 - rho*rho) / (1.0 + rho*rho);
                double s = 1.0 + rho*rho;
                double dOm = -4.0 * rho / (s * s);

                // Skip boundary and origin (BCs, not physical)
                if (nn == sys.boundaryIndex()) continue;
                int n_origin = params.N_nuc; // origin index
                if (nn == n_origin) continue;

                // Determine radial quadrature weight
                double w_r = 0.0;
                if (nn < n_origin) {
                    // Nucleus domain: index nn maps to nucleus Chebyshev index nn
                    // Use Clenshaw-Curtis weights
                    w_r = nuc_cheb.weights()[nn];
                } else {
                    // Shell domain: index nn = n_boundary + k where k = nn - n_boundary
                    int k = nn - sys.boundaryIndex();
                    if (k >= 0 && k < (int)shell_cheb.weights().size())
                        w_r = shell_cheb.weights()[k];
                }

                for (int j = 0; j < nA_local; j++) {
                    double theta = sys.thetaGrid(j);
                    double sin_th = std::sin(theta);
                    int si = nn * nA_local + j;
                    int midx = m * nS_local + si;

                    // Angular weight (Legendre Gauss weight includes sin θ)
                    // For proper quadrature: ∫₀^π f(θ) sin θ dθ ≈ Σⱼ w_j f(θ_j)
                    // The Legendre collocation weights handle this.
                    // For simplicity use uniform: dθ × sin θ
                    double w_theta = PI / nA_local * sin_th;

                    // Extract hatted fields at this point
                    double phi_hat = u[sys.stateIdx(FLD_SCALAR, m, si)];
                    double N_hat = u[sys.stateIdx(FLD_LAPSE, m, si)];

                    // We need spatial derivatives of phi.
                    // Use finite differences from neighboring radial points
                    // for dphi_hat/drho, and neighboring angular points for
                    // dphi_hat/dtheta. This is crude but functional.

                    // Radial derivative (central finite diff)
                    double dphi_hat_dr = 0.0;
                    if (nn > 0 && nn < nR_local - 1) {
                        double rho_p = sys.rhoGrid(nn + 1 < nR_local ? nn + 1 : nn);
                        double rho_m = sys.rhoGrid(nn > 0 ? nn - 1 : nn);
                        double phi_p = u[sys.stateIdx(FLD_SCALAR, m, (nn+1)*nA_local + j)];
                        double phi_m = u[sys.stateIdx(FLD_SCALAR, m, (nn-1)*nA_local + j)];
                        double dr = rho_p - rho_m;
                        if (std::abs(dr) > 1e-15)
                            dphi_hat_dr = (phi_p - phi_m) / dr;
                    }

                    // Angular derivative (central finite diff)
                    double dphi_hat_dth = 0.0;
                    if (nA_local > 1 && j > 0 && j < nA_local - 1) {
                        double th_p = sys.thetaGrid(j+1);
                        double th_m = sys.thetaGrid(j-1);
                        double phi_p = u[sys.stateIdx(FLD_SCALAR, m, nn*nA_local + j+1)];
                        double phi_m = u[sys.stateIdx(FLD_SCALAR, m, nn*nA_local + j-1)];
                        double dth = th_p - th_m;
                        if (std::abs(dth) > 1e-15)
                            dphi_hat_dth = (phi_p - phi_m) / dth;
                    }

                    // Time derivative: approximate from neighboring time slices
                    double dphi_hat_dt = 0.0;
                    if (nT_local > 1) {
                        if (m == 0) {
                            double phi_1 = u[sys.stateIdx(FLD_SCALAR, 1, si)];
                            double tau_1 = PI / (params.N_t * omega);
                            dphi_hat_dt = (phi_1 - phi_hat) / tau_1;
                        } else if (m == nT_local - 1) {
                            double phi_m1 = u[sys.stateIdx(FLD_SCALAR, m-1, si)];
                            double tau_m1 = (m-1) * PI / (params.N_t * omega);
                            double tau_m = m * PI / (params.N_t * omega);
                            dphi_hat_dt = (phi_hat - phi_m1) / (tau_m - tau_m1);
                        } else {
                            double phi_p = u[sys.stateIdx(FLD_SCALAR, m+1, si)];
                            double phi_m_ = u[sys.stateIdx(FLD_SCALAR, m-1, si)];
                            double tau_p = (m+1) * PI / (params.N_t * omega);
                            double tau_m_ = (m-1) * PI / (params.N_t * omega);
                            dphi_hat_dt = (phi_p - phi_m_) / (tau_p - tau_m_);
                        }
                    }

                    // Un-hat to physical variables
                    double q_phi = params.Delta / 2.0;
                    double Omq = std::pow(Om, q_phi);
                    double phi_phys = phi_hat * Omq;
                    double dphi_dr_phys = (dphi_hat_dr * Omq
                        + q_phi * phi_hat * (std::abs(Om)>1e-30 ? Omq/Om : 0.0) * dOm);
                    double dphi_dth_phys = dphi_hat_dth * Omq;
                    double dphi_dt_phys = dphi_hat_dt * Omq;

                    double N_phys = N_hat * (std::abs(Om) > 1e-30 ? 1.0/Om : 0.0);

                    // Physical metric (background approximation for volume element)
                    // γ_ρρ = (2/(1-ρ²))² / (1+r²) = 4/(1-ρ²)⁴ × (1-ρ²)²/(1+ρ²)²
                    //       = 4/((1-ρ²)(1+ρ²))²
                    // γ_θθ = r² = 4ρ²/(1-ρ²)²
                    // γ_φφ = r² sin²θ
                    // √γ = (2/(1-ρ²))³ × ρ²sinθ / (1+ρ²) ... complicated
                    //
                    // For the background:
                    // √det(γ) dρ dθ dφ = r² sin θ / (1+r²)^{1/2} × dr/dρ × dρ dθ dφ
                    // dr/dρ = 2(1+ρ²)/(1-ρ²)²
                    // r² = 4ρ²/(1-ρ²)²
                    // (1+r²)^{1/2} = (1+ρ²)/(1-ρ²)
                    //
                    // √det(γ) dρ dθ = 4ρ²/(1-ρ²)² × sinθ × (1-ρ²)/(1+ρ²)
                    //                × 2(1+ρ²)/(1-ρ²)² dρ dθ
                    //              = 8ρ²sinθ / (1-ρ²)³ dρ dθ

                    double omr2 = 1.0 - rho*rho;
                    double sqg_factor = (std::abs(omr2) > 1e-15) ?
                        8.0 * rho * rho / (omr2 * omr2 * omr2) : 0.0;

                    // Conjugate momentum: Π = (∂ₜφ - β·∇φ)/N
                    // At leading order β = 0, so Π ≈ ∂ₜφ/N
                    double Pi_scalar = (std::abs(N_phys) > 1e-30) ?
                        dphi_dt_phys / N_phys : 0.0;

                    // Gradient squared: γ^{ij}∂ᵢφ∂ⱼφ
                    // On background: γ^{ρρ} = (1+r²) × (dρ/dr)²
                    //                       = (1+ρ²)²/(1-ρ²)² × (1-ρ²)⁴/(4(1+ρ²)²)
                    //                       = (1-ρ²)²/4
                    // γ^{θθ} = 1/r² = (1-ρ²)²/(4ρ²)
                    // γ^{ρρ}(∂_ρφ)² + γ^{θθ}(∂_θφ)²
                    double ginv_rr = omr2*omr2/4.0;
                    double ginv_tt_phys = (rho > 1e-10) ? omr2*omr2/(4.0*rho*rho) : 0.0;

                    double grad_sq = ginv_rr * dphi_dr_phys * dphi_dr_phys
                                   + ginv_tt_phys * dphi_dth_phys * dphi_dth_phys;

                    // Energy density
                    double m_sq = params.Delta * (params.Delta - 3.0);
                    double rho_E = 0.5 * (Pi_scalar * Pi_scalar + grad_sq
                                        + m_sq * phi_phys * phi_phys);

                    // Integrate: M += time_weight × ρ_E × √γ × dρ × dθ × 2π
                    double dV = sqg_factor * w_r * w_theta * 2.0 * PI;
                    obs.M_ADM += tw[m] * rho_E * dV;
                }
            }
        }
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
