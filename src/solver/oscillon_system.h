// oscillon_system.h — Residual assembler for the oscillon Newton system
//
// TIME-COLLOCATION LAYOUT:
// State vector u stores hatted field values at all (time, space) collocation points:
//   u[A * nTau * nSp + m * nSp + n * nAng + j]
// where A = field index (0..7), m = temporal (0..N_t), n = radial, j = angular.
//
// Residuals map equations to field slots:
//   R[FLD_LAPSE slot]   = trK = 0     (maximal slicing)
//   R[FLD_SHIFT_* slot] = V^i = 0     (harmonic gauge)
//   R[FLD_GAMMA_* slot] = K_{ij} evol (Einstein evolution)
//   R[FLD_SCALAR slot]  = scalar wave  (Klein-Gordon)
//
// Special handling:
//   - Odd fields at m=0,N_t: R = u (enforce zero by time-reversal symmetry)
//   - Origin (rho=0): Dirichlet to background (coordinate regularity)
//   - Boundary (rho=1): Dirichlet to background + phi_hat=0
#pragma once

#include "spectral/fourier.h"
#include "spectral/chebyshev.h"
#include "spectral/legendre.h"
#include "geometry/adm.h"
#include "geometry/equations.h"
#include <vector>
#include <functional>

namespace solver {

// Parameters for the oscillon system
struct OscillonParams {
    int N_t = 4;         // number of Fourier modes
    int N_nuc = 16;      // Chebyshev order in nucleus [0, rho_mid]
    int N_shell = 16;    // Chebyshev order in shell [rho_mid, 1]
    double rho_mid = 0.5;
    int N_theta = 8;     // Legendre order

    double Delta = 6.0;
    double Lambda = -3.0;

    int ell = 2;         // angular mode of the seed
};

class OscillonSystem {
public:
    OscillonSystem(const OscillonParams& params);

    // ---- Dimensions ----
    int nTau() const { return N_t_ + 1; }     // temporal collocation points
    int nRadial() const { return (int)rho_grid_.size(); }
    int nAngular() const { return angular_.size(); }
    int nSpatial() const { return nRadial() * nAngular(); }

    // Size of the full state: 8 * nTau * nSpatial + 1 (the +1 is for omega)
    // u has size stateSize()-1; omega is passed separately
    int stateSize() const;

    // ---- Grid access ----
    double rhoGrid(int n) const { return rho_grid_[n]; }
    double thetaGrid(int j) const { return theta_grid_[j]; }
    double tauGrid(int m) const;

    // ---- State vector indexing ----
    // u[stateIdx(A, m, si)] = f_hat_A at (tau_m, spatial point si)
    int stateIdx(int fieldIdx, int m, int si) const {
        return fieldIdx * nTau() * nSpatial() + m * nSpatial() + si;
    }
    int spatialIdx(int n, int j) const { return n * nAngular() + j; }

    // ---- Main residual function ----
    // F(u, omega) -> R, where u and R have size stateSize()-1
    void computeResidual(const double* u, double omega, double* R) const;

    // ---- Normalization ----
    // Returns the scalar field value at the normalization reference point
    double normalization(const double* u) const;

    // ---- Initial guess ----
    // Set u to background + epsilon * scalar eigenmode
    void setLinearSeed(double* u, double& omega, double epsilon) const;

    // ---- Diagnostics ----
    double evaluateField(const double* u, int fieldIdx,
                         double tau, double rho, double theta) const;
    void getFieldOnGrid(const double* u, int fieldIdx,
                        std::vector<double>& values) const;

    // ---- Access to params ----
    const OscillonParams& params() const { return params_; }

private:
    OscillonParams params_;
    int N_t_;

    // Spectral bases
    spectral::Fourier four_;
    spectral::TwoDomainChebyshev radial_;
    spectral::Legendre angular_;

    // Grids
    std::vector<double> rho_grid_;    // nRadial() points
    std::vector<double> theta_grid_;  // nAngular() points

    // Domain sizes
    int n_nuc_pts_;   // nucleus().size() = N_nuc + 1
    int n_shell_pts_; // shell().size() = N_shell + 1

    // Special grid indices
    int n_origin_;    // global radial index where rho = 0
    int n_boundary_;  // global radial index where rho = 1

    // Normalization reference point
    int norm_ref_si_; // spatial index for normalization

    // Background residual subtraction:
    // The pure-AdS background should have zero residual, but spectral truncation
    // of d^2/drho^2(gamma_hat/Omega^2) introduces O(1) errors in the evolution
    // equation residuals at low resolution. We precompute and subtract this.
    std::vector<double> bg_residual_; // size stateSize()-1

    // ---- Internal helpers ----

    // Background hatted field value at (rho, theta)
    double backgroundValue(int fieldIdx, double rho, double theta) const;

    // Compute first and second radial derivatives of f[si] for all spatial points.
    // f, df_dr, d2f_drr each have nSpatial() entries.
    // If d2f_drr == nullptr, skip second derivative.
    void radialDeriv(const double* f, double* df_dr, double* d2f_drr) const;

    // Compute first and second angular derivatives at all spatial points.
    void angularDeriv(const double* f, double* df_dtheta, double* d2f_dtt) const;

    // Compute first temporal derivative of arr[m * nSp + si] for each si.
    // Input: arr with nTau * nSpatial entries. Output: dt_arr with same size.
    void temporalDeriv1(const double* arr, bool even_parity,
                        double omega, double* dt_arr) const;

    // Compute first and second temporal derivatives.
    void temporalDeriv12(const double* arr, bool even_parity, double omega,
                         double* dt_arr, double* d2t_arr) const;
};

} // namespace solver
