// equations.h — Regularized Einstein-scalar-AM system
// Martinon et al. (2017) eqs. (33a)-(33c) with scalar stress tensor
#pragma once

#include <vector>
#include "geometry/adm.h"

namespace geometry {

// Forward declarations
class AdS4Background;

// ============================================================================
// Equation residuals: 8 equations for 8 unknowns
// ============================================================================
struct EquationResiduals {
    double hamiltonian;      // Hamiltonian constraint (= trK = 0 for maximal slicing)
    double momentum_r;       // Gauge condition V^rho = 0
    double momentum_t;       // Gauge condition V^theta = 0
    double evolution_rr;     // K evolution for gamma_{rho rho}
    double evolution_rt;     // K evolution for gamma_{rho theta}
    double evolution_tt;     // K evolution for gamma_{theta theta}
    double evolution_pp;     // K evolution for gamma_{phi phi}
    double scalar_wave;      // Scalar wave equation
};

// ============================================================================
// Full point data: everything needed to evaluate residuals at one point
// ============================================================================
// All quantities are in HATTED (regularized) variables.
// The residual function un-hats, computes physical residuals, and re-regularizes.
struct FullPointData {
    double rho, theta;

    // --- Hatted field values ---
    // fields[FLD_*] with indexing from FieldIdx
    double fields[N_FIELDS];
    // Convenience accessors
    double Nhat()      const { return fields[FLD_LAPSE]; }
    double beta_r()    const { return fields[FLD_SHIFT_R]; }
    double beta_t()    const { return fields[FLD_SHIFT_T]; }
    double ghat_rr()   const { return fields[FLD_GAMMA_RR]; }
    double ghat_rt()   const { return fields[FLD_GAMMA_RT]; }
    double ghat_tt()   const { return fields[FLD_GAMMA_TT]; }
    double ghat_pp()   const { return fields[FLD_GAMMA_PP]; }
    double phihat()    const { return fields[FLD_SCALAR]; }

    // --- Time derivatives (of hatted fields) ---
    double dt[N_FIELDS];       // ∂_t f̂_A
    double d2t[N_FIELDS];      // ∂²_t f̂_A

    // --- Spatial derivatives (of hatted fields) ---
    double dr[N_FIELDS];       // ∂_ρ f̂_A
    double dtheta[N_FIELDS];   // ∂_θ f̂_A
    double d2rr[N_FIELDS];     // ∂²_ρρ f̂_A
    double d2rt[N_FIELDS];     // ∂_ρ∂_θ f̂_A
    double d2tt[N_FIELDS];     // ∂²_θθ f̂_A

    // --- Mixed time-space derivatives (of hatted fields) ---
    double dt_dr[N_FIELDS];    // ∂_t ∂_ρ f̂_A
    double dt_dtheta[N_FIELDS]; // ∂_t ∂_θ f̂_A

    // --- Physical parameters ---
    double Delta;       // conformal dimension
    double Lambda;      // cosmological constant (-3 for AdS4 with L=1)
};

// ============================================================================
// Full nonlinear residual computation
// ============================================================================
// Evaluates all 8 equation residuals at one collocation point.
// Input: hatted field values and all their derivatives.
// Output: regularized residuals (multiplied by appropriate Ω powers).
//
// Equations:
//   0: trK = 0                          (maximal slicing → lapse equation)
//   1: V^ρ = 0                          (harmonic gauge → shift^ρ equation)
//   2: V^θ = 0                          (harmonic gauge → shift^θ equation)
//   3-6: ∂_tK_{ij} - RHS_{ij} = 0      (Einstein evolution → metric equations)
//   7: □φ - m²φ = 0                     (scalar wave equation)
void computeFullResiduals(const FullPointData& data, EquationResiduals& res);

// ============================================================================
// Scalar stress tensor (unchanged from before)
// ============================================================================
struct ScalarStressTensor {
    double rho_E;     // energy density
    double j_r, j_t;  // momentum density
    double S_rr, S_rt, S_tt, S_pp; // spatial stress
};

ScalarStressTensor computeScalarStress(
    double phi, double Pi, double dphi_dr, double dphi_dt,
    const double gamma[4], double m_sq
);

// ============================================================================
// Background-only residuals (retained for checkpoint tests)
// ============================================================================
double scalarResidualBackground(
    double rho, double theta,
    double phi_val, double dphi_drho, double dphi_dtheta,
    double d2phi_drr, double d2phi_drt, double d2phi_dtt,
    double omega, int fourier_k,
    double m_sq);

// Old interface (placeholder, kept for backward compatibility)
EquationResiduals computeResiduals(
    double rho, double theta,
    double Nhat, double beta_r, double beta_t,
    double gammahat_rr, double gammahat_rt, double gammahat_tt, double gammahat_pp,
    double phihat,
    double dNhat_dt, double dbeta_r_dt, double dbeta_t_dt,
    double dgammahat_rr_dt, double dgammahat_rt_dt,
    double dgammahat_tt_dt, double dgammahat_pp_dt,
    double dphihat_dt,
    const double* d_rho, const double* d_theta,
    const double* d2_rr, const double* d2_rt, const double* d2_tt,
    double Delta, double Lambda = -3.0
);

} // namespace geometry
