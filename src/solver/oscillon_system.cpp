// oscillon_system.cpp — Residual assembler implementation
//
// TIME-COLLOCATION LAYOUT:
// State vector u[A * nTau * nSp + m * nSp + si] stores hatted field values
// at collocation points (tau_m, rho_n, theta_j), where si = n * nAng + j.
//
// Residual assembly:
//   1. Copy values from u
//   2. Compute temporal derivatives (Fourier transform → analytic differentiation)
//   3. Compute spatial derivatives (Chebyshev radial + Legendre angular)
//   4. Compute mixed time-space derivatives
//   5. Build FullPointData at each (m, n, j), call computeFullResiduals
//   6. Store residuals with BC enforcement at special points
//
// Radial grid follows TwoDomainChebyshev:
//   [rho_mid, ..., 0, 1, ..., near_rho_mid]
//   Nucleus indices: 0..N_nuc; Shell indices: N_nuc+1..N_nuc+N_shell
//   Interface (rho_mid) at index 0; Origin (rho=0) at index N_nuc;
//   Boundary (rho=1) at index N_nuc+1.

#include "solver/oscillon_system.h"
#include "geometry/ads4.h"
#include <cstring>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <algorithm>

namespace solver {

using namespace geometry;

static const double PI = 3.14159265358979323846;

// ============================================================================
// Construction
// ============================================================================

OscillonSystem::OscillonSystem(const OscillonParams& params)
    : params_(params),
      N_t_(params.N_t),
      four_(params.N_t),
      radial_(params.N_nuc, params.N_shell, params.rho_mid),
      angular_(params.N_theta)
{
    n_nuc_pts_ = radial_.nucleus().size();    // N_nuc + 1
    n_shell_pts_ = radial_.shell().size();    // N_shell + 1

    // Build radial grid matching TwoDomainChebyshev:
    // Nucleus: grid(0)=rho_mid, ..., grid(N_nuc)=0
    // Shell:   grid(0)=1, ..., grid(N_shell-1)≈rho_mid+eps
    // (shell's last point grid(N_shell)=rho_mid is already at nucleus grid(0))
    for (int i = 0; i < n_nuc_pts_; i++)
        rho_grid_.push_back(radial_.nucleus().grid(i));
    for (int i = 0; i < n_shell_pts_ - 1; i++)
        rho_grid_.push_back(radial_.shell().grid(i));

    // Special indices
    n_origin_ = radial_.nucleus().order();   // N_nuc, where grid = 0
    n_boundary_ = n_nuc_pts_;                // N_nuc+1, where grid = 1

    // Angular grid
    for (int j = 0; j < angular_.size(); j++)
        theta_grid_.push_back(angular_.theta(j));

    // Find normalization reference point: spatial index where the eigenmode
    // phi_hat is largest at tau=0. Use the midpoint of the nucleus domain
    // and the angular point closest to theta=pi/4.
    ScalarEigenmode mode(params.ell, params.Delta);
    double max_val = 0;
    norm_ref_si_ = 0;
    int nR = nRadial(), nA = nAngular();
    for (int n = 0; n < nR; n++) {
        double rho = rhoGrid(n);
        for (int j = 0; j < nA; j++) {
            double theta = thetaGrid(j);
            double val = std::abs(mode.phi_hat(0.0, rho, theta));
            if (val > max_val) {
                max_val = val;
                norm_ref_si_ = n * nA + j;
            }
        }
    }

    // Precompute background residual for subtraction.
    // The pure-AdS background satisfies the continuous equations exactly, but
    // spectral truncation of the un-hatted second derivatives creates O(1) errors
    // The ρ²·Ω³ regularization in equations.cpp reduces the background residual
    // from O(10) to O(1e-4) at N=8. We additionally subtract this small background
    // residual to reach the O(eps²) perturbation level. Because the subtracted
    // vector is now tiny, the Bianchi identity inconsistency it introduces is
    // proportionally small, and GMRES can handle it.
    {
        int n_state = stateSize() - 1;
        std::vector<double> u_bg(n_state);
        double omega_bg;
        setLinearSeed(u_bg.data(), omega_bg, 0.0); // pure background

        // bg_residual_ must be empty during this call so computeResidual
        // doesn't subtract (the check inside uses .empty()).
        bg_residual_.clear();

        std::vector<double> raw_bg(n_state, 0.0);
        computeResidual(u_bg.data(), omega_bg, raw_bg.data());
        bg_residual_ = std::move(raw_bg);
    }
}

// ============================================================================
// Dimensions
// ============================================================================

int OscillonSystem::stateSize() const {
    return N_FIELDS * nTau() * nSpatial() + 1; // +1 for omega
}

double OscillonSystem::tauGrid(int m) const {
    return m * PI / N_t_;
}

// ============================================================================
// Background hatted field values
// ============================================================================

double OscillonSystem::backgroundValue(int fieldIdx, double rho, double theta) const {
    double op2 = (1.0 + rho * rho);
    double sth = sin(theta);
    switch (fieldIdx) {
        case FLD_LAPSE:    return 1.0;
        case FLD_SHIFT_R:  return 0.0;
        case FLD_SHIFT_T:  return 0.0;
        case FLD_GAMMA_RR: return 4.0 / (op2 * op2);
        case FLD_GAMMA_RT: return 0.0;
        case FLD_GAMMA_TT: return 4.0 * rho * rho / (op2 * op2);
        case FLD_GAMMA_PP: return 4.0 * rho * rho * sth * sth / (op2 * op2);
        case FLD_SCALAR:   return 0.0;
        default:           return 0.0;
    }
}

// ============================================================================
// Radial derivative helper
// ============================================================================
// Differentiates f[n * nAng + j] in the radial direction for all angular j.
// f has nSpatial() entries. df_dr and d2f_drr (if not null) have same size.

void OscillonSystem::radialDeriv(
    const double* f, double* df_dr, double* d2f_drr) const
{
    int nA = nAngular();
    int nR = nRadial();

    for (int j = 0; j < nA; j++) {
        // --- Nucleus domain ---
        std::vector<double> f_nuc(n_nuc_pts_), df_nuc(n_nuc_pts_);
        for (int i = 0; i < n_nuc_pts_; i++)
            f_nuc[i] = f[i * nA + j];

        radial_.nucleus().differentiate(f_nuc.data(), df_nuc.data());
        for (int i = 0; i < n_nuc_pts_; i++)
            df_dr[i * nA + j] = df_nuc[i];

        if (d2f_drr) {
            std::vector<double> d2f_nuc(n_nuc_pts_);
            radial_.nucleus().differentiate2(f_nuc.data(), d2f_nuc.data());
            for (int i = 0; i < n_nuc_pts_; i++)
                d2f_drr[i * nA + j] = d2f_nuc[i];
        }

        // --- Shell domain ---
        // Shell has n_shell_pts_ points. Stored globally: indices n_nuc_pts_ to
        // n_nuc_pts_ + (n_shell_pts_-2). The missing point shell.grid(N_shell) = rho_mid
        // is at global index 0 (= nucleus.grid(0)).
        std::vector<double> f_shell(n_shell_pts_), df_shell(n_shell_pts_);
        int n_shell_stored = n_shell_pts_ - 1; // shell points in global grid
        for (int i = 0; i < n_shell_stored; i++)
            f_shell[i] = f[(n_nuc_pts_ + i) * nA + j];
        f_shell[n_shell_pts_ - 1] = f[0 * nA + j]; // interface point = rho_mid

        radial_.shell().differentiate(f_shell.data(), df_shell.data());
        for (int i = 0; i < n_shell_stored; i++)
            df_dr[(n_nuc_pts_ + i) * nA + j] = df_shell[i];

        if (d2f_drr) {
            std::vector<double> d2f_shell(n_shell_pts_);
            radial_.shell().differentiate2(f_shell.data(), d2f_shell.data());
            for (int i = 0; i < n_shell_stored; i++)
                d2f_drr[(n_nuc_pts_ + i) * nA + j] = d2f_shell[i];
        }
    }
}

// ============================================================================
// Angular derivative helper
// ============================================================================
// Differentiates f[n * nAng + j] in the theta direction for all radial n.
// Uses forward Legendre transform + analytic derivative of P_l.

void OscillonSystem::angularDeriv(
    const double* f, double* df_dtheta, double* d2f_dtt) const
{
    int nA = nAngular();
    int nR = nRadial();
    int L_max = nA - 1;

    for (int n = 0; n < nR; n++) {
        // Extract angular values at this radial point
        std::vector<double> f_ang(nA);
        for (int j = 0; j < nA; j++)
            f_ang[j] = f[n * nA + j];

        // Forward Legendre transform
        std::vector<double> coeffs(L_max + 1);
        angular_.forward(f_ang.data(), coeffs.data(), L_max);

        // Evaluate derivatives at each angular grid point
        for (int j = 0; j < nA; j++) {
            double theta = theta_grid_[j];
            double ct = cos(theta), st = sin(theta);
            double s2 = 1.0 - ct * ct; // sin^2(theta)

            std::vector<double> P_vals(L_max + 2);
            spectral::Legendre::P_all(L_max + 1, ct, P_vals.data());

            double df1 = 0.0, df2 = 0.0;
            for (int l = 1; l <= L_max; l++) {
                // dP_l/dx where x = cos(theta)
                double dPdx;
                if (s2 > 1e-14) {
                    dPdx = l * (P_vals[l-1] - ct * P_vals[l]) / s2;
                } else {
                    dPdx = l * (l + 1) / 2.0;
                    if (ct < 0) dPdx *= ((l % 2 == 0) ? 1.0 : -1.0);
                }
                // d/dtheta P_l(cos theta) = -sin(theta) * dP_l/dx
                double dP_dtheta = -st * dPdx;

                // d2P/dx2 for second derivative
                double d2Pdx2;
                if (s2 > 1e-14) {
                    d2Pdx2 = (2.0 * ct * dPdx - l * (l + 1) * P_vals[l]) / s2;
                } else {
                    // At poles: use limiting form
                    // d2P_l/dx2(1) = (l-1)*l*(l+1)*(l+2)/8
                    d2Pdx2 = (l - 1.0) * l * (l + 1.0) * (l + 2.0) / 8.0;
                    if (ct < 0 && l % 2 == 1) d2Pdx2 = -d2Pdx2;
                }
                // d2/dtheta2 P_l = -cos(theta)*dPdx + sin^2(theta)*d2Pdx2
                double d2P_dtheta2 = -ct * dPdx + s2 * d2Pdx2;

                df1 += coeffs[l] * dP_dtheta;
                df2 += coeffs[l] * d2P_dtheta2;
            }

            df_dtheta[n * nA + j] = df1;
            d2f_dtt[n * nA + j] = df2;
        }
    }
}

// ============================================================================
// Temporal derivative helpers
// ============================================================================
// Input arr has layout arr[m * nSp + si] for m=0..N_t, si=0..nSp-1.
// Output dt_arr has the same layout.

void OscillonSystem::temporalDeriv1(
    const double* arr, bool even, double omega, double* dt_arr) const
{
    int nT = nTau();
    int nS = nSpatial();

    for (int si = 0; si < nS; si++) {
        // Extract temporal values at this spatial point
        std::vector<double> v(nT);
        for (int m = 0; m < nT; m++)
            v[m] = arr[m * nS + si];

        if (even) {
            // Forward cosine transform → a[k], k=0..N_t
            std::vector<double> a(nT);
            four_.forwardCos(v.data(), a.data());

            for (int m = 0; m < nT; m++) {
                double tau = tauGrid(m);
                double dtv = 0.0;
                for (int k = 1; k <= N_t_; k++)
                    dtv += a[k] * (-k * omega) * sin(k * tau);
                dt_arr[m * nS + si] = dtv;
            }
        } else {
            // Forward sine transform → b[k-1] for k=1..N_t
            std::vector<double> b(N_t_);
            four_.forwardSin(v.data(), b.data());

            for (int m = 0; m < nT; m++) {
                double tau = tauGrid(m);
                double dtv = 0.0;
                for (int k = 1; k <= N_t_; k++)
                    dtv += b[k-1] * (k * omega) * cos(k * tau);
                dt_arr[m * nS + si] = dtv;
            }
        }
    }
}

void OscillonSystem::temporalDeriv12(
    const double* arr, bool even, double omega,
    double* dt_arr, double* d2t_arr) const
{
    int nT = nTau();
    int nS = nSpatial();

    for (int si = 0; si < nS; si++) {
        std::vector<double> v(nT);
        for (int m = 0; m < nT; m++)
            v[m] = arr[m * nS + si];

        if (even) {
            std::vector<double> a(nT);
            four_.forwardCos(v.data(), a.data());

            for (int m = 0; m < nT; m++) {
                double tau = tauGrid(m);
                double dtv = 0.0, d2tv = 0.0;
                for (int k = 0; k <= N_t_; k++) {
                    double sk = sin(k * tau);
                    double ck = cos(k * tau);
                    dtv  += a[k] * (-k * omega) * sk;
                    d2tv += a[k] * (-k * k * omega * omega) * ck;
                }
                dt_arr[m * nS + si] = dtv;
                d2t_arr[m * nS + si] = d2tv;
            }
        } else {
            std::vector<double> b(N_t_);
            four_.forwardSin(v.data(), b.data());

            for (int m = 0; m < nT; m++) {
                double tau = tauGrid(m);
                double dtv = 0.0, d2tv = 0.0;
                for (int k = 1; k <= N_t_; k++) {
                    double ck = cos(k * tau);
                    double sk = sin(k * tau);
                    dtv  += b[k-1] * (k * omega) * ck;
                    d2tv += b[k-1] * (-k * k * omega * omega) * sk;
                }
                dt_arr[m * nS + si] = dtv;
                d2t_arr[m * nS + si] = d2tv;
            }
        }
    }
}

// ============================================================================
// Main residual computation
// ============================================================================

void OscillonSystem::computeResidual(
    const double* u, double omega, double* R) const
{
    const int nT = nTau();
    const int nS = nSpatial();
    const int nR_val = nRadial();
    const int nA = nAngular();
    const int n_state = stateSize() - 1;

    // Clear output
    std::memset(R, 0, n_state * sizeof(double));

    // ================================================================
    // Step 1: Workspace allocation
    // ================================================================
    // Derivative arrays: 10 per field, each nT * nS
    enum { D_VAL=0, D_DT=1, D_D2T=2, D_DR=3, D_DTHETA=4,
           D_D2RR=5, D_D2TT=6, D_D2RT=7, D_DT_DR=8, D_DT_DTHETA=9 };
    const int N_DERIV = 10;
    std::vector<double> workspace(N_DERIV * N_FIELDS * nT * nS, 0.0);
    auto W = [&](int field, int deriv) -> double* {
        return workspace.data() + (field * N_DERIV + deriv) * nT * nS;
    };

    // ================================================================
    // Step 2: Copy field values from state vector
    // ================================================================
    for (int A = 0; A < N_FIELDS; A++) {
        double* val = W(A, D_VAL);
        for (int m = 0; m < nT; m++)
            for (int si = 0; si < nS; si++)
                val[m * nS + si] = u[stateIdx(A, m, si)];
    }

    // ================================================================
    // Step 3: Temporal derivatives (dt, d2t) via Fourier transform
    // ================================================================
    for (int A = 0; A < N_FIELDS; A++) {
        bool even = isTimeParity_even(A);
        temporalDeriv12(W(A, D_VAL), even, omega, W(A, D_DT), W(A, D_D2T));
    }

    // ================================================================
    // Step 4: Spatial derivatives for each time slice
    // ================================================================
    for (int A = 0; A < N_FIELDS; A++) {
        for (int m = 0; m < nT; m++) {
            double* f_m     = W(A, D_VAL) + m * nS;
            double* dr_m    = W(A, D_DR) + m * nS;
            double* d2rr_m  = W(A, D_D2RR) + m * nS;
            double* dth_m   = W(A, D_DTHETA) + m * nS;
            double* d2tt_m  = W(A, D_D2TT) + m * nS;
            double* d2rt_m  = W(A, D_D2RT) + m * nS;

            // 4a: Radial derivatives
            radialDeriv(f_m, dr_m, d2rr_m);

            // 4b: Angular derivatives
            angularDeriv(f_m, dth_m, d2tt_m);

            // 4c: Mixed d_rho(d_theta f)
            // Differentiate the angular derivative in rho
            radialDeriv(dth_m, d2rt_m, nullptr);
        }
    }

    // ================================================================
    // Step 5: Mixed time-space derivatives
    // ================================================================
    // dt_dr[A] = temporal derivative of dr[A]
    // dt_dtheta[A] = temporal derivative of dtheta[A]
    for (int A = 0; A < N_FIELDS; A++) {
        bool even = isTimeParity_even(A);
        temporalDeriv1(W(A, D_DR), even, omega, W(A, D_DT_DR));
        temporalDeriv1(W(A, D_DTHETA), even, omega, W(A, D_DT_DTHETA));
    }

    // ================================================================
    // Step 6: Assemble residuals
    // ================================================================
    // Equation → field slot mapping
    static const int eq_to_field[8] = {
        FLD_LAPSE, FLD_SHIFT_R, FLD_SHIFT_T,
        FLD_GAMMA_RR, FLD_GAMMA_RT, FLD_GAMMA_TT,
        FLD_GAMMA_PP, FLD_SCALAR
    };

    for (int m = 0; m < nT; m++) {
        for (int n = 0; n < nR_val; n++) {
            for (int j = 0; j < nA; j++) {
                int si = spatialIdx(n, j);
                int midx = m * nS + si;  // workspace index

                // --- Check for special points ---

                // (a) Odd fields at tau endpoints: enforce u = 0
                for (int A = 0; A < N_FIELDS; A++) {
                    if (!isTimeParity_even(A) && (m == 0 || m == N_t_)) {
                        R[stateIdx(A, m, si)] = u[stateIdx(A, m, si)];
                    }
                }

                // (b) Origin (rho=0): BC enforcement
                // Metric fields: Dirichlet to background (coordinate regularity)
                // Scalar field:
                //   ell >= 1: Dirichlet to background (phi_hat ~ rho^ell -> 0)
                //   ell == 0: Neumann (d_rho phi_hat = 0, regularity of spherical mode)
                if (n == n_origin_) {
                    double rho = rhoGrid(n);
                    double theta = thetaGrid(j);
                    for (int A = 0; A < N_FIELDS; A++) {
                        // Skip if already set by (a)
                        if (!isTimeParity_even(A) && (m == 0 || m == N_t_))
                            continue;
                        if (A == FLD_SCALAR && params_.ell == 0) {
                            // Neumann BC: d_rho(phi_hat) = 0 at origin
                            R[stateIdx(A, m, si)] = W(A, D_DR)[m * nS + si];
                        } else {
                            // Dirichlet BC: phi_hat = background
                            R[stateIdx(A, m, si)] =
                                u[stateIdx(A, m, si)] - backgroundValue(A, rho, theta);
                        }
                    }
                    continue; // skip PDE residual at origin
                }

                // (c) Boundary (rho=1): Dirichlet BC to background
                if (n == n_boundary_) {
                    double rho = rhoGrid(n);
                    double theta = thetaGrid(j);
                    for (int A = 0; A < N_FIELDS; A++) {
                        if (!isTimeParity_even(A) && (m == 0 || m == N_t_))
                            continue;
                        R[stateIdx(A, m, si)] =
                            u[stateIdx(A, m, si)] - backgroundValue(A, rho, theta);
                    }
                    continue; // skip PDE residual at boundary
                }

                // --- Interior points: evaluate PDE residuals ---

                // Build FullPointData
                FullPointData data;
                std::memset(&data, 0, sizeof(data));
                data.rho = rhoGrid(n);
                data.theta = thetaGrid(j);
                data.Delta = params_.Delta;
                data.Lambda = params_.Lambda;

                for (int A = 0; A < N_FIELDS; A++) {
                    data.fields[A]     = W(A, D_VAL)[midx];
                    data.dt[A]         = W(A, D_DT)[midx];
                    data.d2t[A]        = W(A, D_D2T)[midx];
                    data.dr[A]         = W(A, D_DR)[midx];
                    data.dtheta[A]     = W(A, D_DTHETA)[midx];
                    data.d2rr[A]       = W(A, D_D2RR)[midx];
                    data.d2tt[A]       = W(A, D_D2TT)[midx];
                    data.d2rt[A]       = W(A, D_D2RT)[midx];
                    data.dt_dr[A]      = W(A, D_DT_DR)[midx];
                    data.dt_dtheta[A]  = W(A, D_DT_DTHETA)[midx];
                }

                // Compute equation residuals
                EquationResiduals res;
                computeFullResiduals(data, res);

                double eq_vals[8] = {
                    res.hamiltonian, res.momentum_r, res.momentum_t,
                    res.evolution_rr, res.evolution_rt, res.evolution_tt,
                    res.evolution_pp, res.scalar_wave
                };

                // Store in R using equation → field mapping
                for (int eq = 0; eq < 8; eq++) {
                    int fld = eq_to_field[eq];
                    bool is_odd = !isTimeParity_even(fld);

                    // Skip slots already set by odd-field endpoint enforcement
                    if (is_odd && (m == 0 || m == N_t_))
                        continue;

                    R[stateIdx(fld, m, si)] = eq_vals[eq];
                }
            }
        }
    }

    // Subtract precomputed background residual.
    // With ρ²·Ω³ regularization this is O(ε_N) ≈ 1e-4 at N=8, so the Bianchi
    // inconsistency it introduces is small enough for GMRES to handle.
    if (!bg_residual_.empty()) {
        for (int i = 0; i < n_state; i++)
            R[i] -= bg_residual_[i];
    }
}

// ============================================================================
// Normalization
// ============================================================================

double OscillonSystem::normalization(const double* u) const {
    // phi_hat at tau=0 at the reference spatial point
    return u[stateIdx(FLD_SCALAR, 0, norm_ref_si_)];
}

// ============================================================================
// Initial guess: background + epsilon * scalar eigenmode
// ============================================================================

void OscillonSystem::setLinearSeed(
    double* u, double& omega, double epsilon) const
{
    int nT = nTau(), nS = nSpatial();
    int nR_val = nRadial(), nA = nAngular();
    int n = stateSize() - 1;
    std::memset(u, 0, n * sizeof(double));

    ScalarEigenmode mode(params_.ell, params_.Delta);
    omega = mode.omega0();

    for (int m = 0; m < nT; m++) {
        double tau = tauGrid(m);
        double t = tau / omega;  // physical time

        for (int nn = 0; nn < nR_val; nn++) {
            double rho = rhoGrid(nn);
            for (int j = 0; j < nA; j++) {
                double theta = thetaGrid(j);
                int si = spatialIdx(nn, j);

                // Background hatted values
                for (int A = 0; A < N_FIELDS; A++)
                    u[stateIdx(A, m, si)] = backgroundValue(A, rho, theta);

                // Add scalar eigenmode perturbation
                u[stateIdx(FLD_SCALAR, m, si)] +=
                    mode.phi_hat(t, rho, theta, epsilon);
            }
        }
    }
}

// ============================================================================
// Field evaluation (diagnostic)
// ============================================================================

double OscillonSystem::evaluateField(
    const double* u, int fieldIdx,
    double tau, double rho, double theta) const
{
    // TODO: full interpolation from collocation points
    // For now, return 0 (this is only used for diagnostics, not for the solver)
    (void)u; (void)fieldIdx; (void)tau; (void)rho; (void)theta;
    return 0.0;
}

void OscillonSystem::getFieldOnGrid(
    const double* u, int fieldIdx,
    std::vector<double>& values) const
{
    int nT = nTau(), nS = nSpatial();
    values.resize(nT * nS);
    for (int m = 0; m < nT; m++)
        for (int si = 0; si < nS; si++)
            values[m * nS + si] = u[stateIdx(fieldIdx, m, si)];
}

} // namespace solver
