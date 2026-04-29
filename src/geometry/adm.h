// adm.h — ADM decomposition and 3+1 variables
// Following Martinon et al. (2017) eq. (22)-(24)
#pragma once

#include <array>

namespace geometry {

// Indices for the 3-metric components in (rho, theta, phi)
// With axisymmetry: gamma_{rho phi} = gamma_{theta phi} = 0
enum MetricIdx {
    IDX_RR = 0,    // gamma_{rho rho}
    IDX_RT = 1,    // gamma_{rho theta}
    IDX_TT = 2,    // gamma_{theta theta}
    IDX_PP = 3,    // gamma_{phi phi}
    N_METRIC = 4
};

// All field components
enum FieldIdx {
    FLD_LAPSE = 0,       // N (lapse)
    FLD_SHIFT_R = 1,     // beta^rho
    FLD_SHIFT_T = 2,     // beta^theta
    FLD_GAMMA_RR = 3,    // gamma_{rho rho}
    FLD_GAMMA_RT = 4,    // gamma_{rho theta}
    FLD_GAMMA_TT = 5,    // gamma_{theta theta}
    FLD_GAMMA_PP = 6,    // gamma_{phi phi}
    FLD_SCALAR = 7,      // scalar field phi
    N_FIELDS = 8
};

// Time parity under t -> -t:
// Even (cosine): lapse, metric, scalar
// Odd (sine): shifts, extrinsic curvature
inline bool isTimeParity_even(int field_idx) {
    switch (field_idx) {
        case FLD_LAPSE:
        case FLD_GAMMA_RR:
        case FLD_GAMMA_RT:
        case FLD_GAMMA_TT:
        case FLD_GAMMA_PP:
        case FLD_SCALAR:
            return true;
        case FLD_SHIFT_R:
        case FLD_SHIFT_T:
            return false;
        default:
            return true;
    }
}

// Compute extrinsic curvature K_{ij} from metric time derivative and shift
// K_{ij} = -(1/(2N))(d_t gamma_{ij} - D_i beta_j - D_j beta_i)
struct ADMPoint {
    double N;           // lapse
    double beta_r, beta_t; // shift components (contravariant)
    double gamma[4];    // metric components (IDX_RR, IDX_RT, IDX_TT, IDX_PP)
    double dgamma_dt[4]; // time derivatives
    double phi;         // scalar field
    double dphi_dt;     // scalar time derivative

    // Spatial position
    double rho, theta;

    // Spatial derivatives of metric
    double dgamma_drho[4], dgamma_dtheta[4];
    double dN_drho, dN_dtheta;
    double dbeta_r_drho, dbeta_r_dtheta;
    double dbeta_t_drho, dbeta_t_dtheta;
    double dphi_drho, dphi_dtheta;

    // Second derivatives (needed for Ricci tensor)
    double d2gamma_drr[4], d2gamma_drt[4], d2gamma_dtt[4];
    double d2N_drr, d2N_drt, d2N_dtt;
    double d2phi_drr, d2phi_drt, d2phi_dtt;
};

// Compute 3D Christoffel symbols from metric
struct Christoffel3D {
    // Gamma^i_{jk} with i,j,k in {rho, theta, phi}
    // Stored as: G[i][j*(j+1)/2 + k] for j >= k (symmetric in lower)
    // Or more simply as a flat array
    double G[3][3][3]; // G[upper][lower1][lower2], symmetric in lower indices
};

Christoffel3D computeChristoffel3D(const double gamma[4],
                                     const double dgamma_drho[4],
                                     const double dgamma_dtheta[4],
                                     double rho, double theta);

// Compute Ricci tensor R_{ij} of the 3-metric from ADMPoint
void computeRicci3D(const ADMPoint& pt, double R[4]);

// Compute Ricci tensor R_{ij} from raw metric data at a point
// g[4] = {γ_rr, γ_rt, γ_tt, γ_pp}
// dg_r[4], dg_t[4] = first spatial derivatives
// d2g_rr[4], d2g_rt[4], d2g_tt[4] = second spatial derivatives
// R[4] = output: {R_rr, R_rt, R_tt, R_pp}
void computeRicci3D_raw(
    const double g[4],
    const double dg_r[4], const double dg_t[4],
    const double d2g_rr[4], const double d2g_rt[4], const double d2g_tt[4],
    double R[4]);

// Compute extrinsic curvature
void computeExtrinsicCurvature(const ADMPoint& pt, double K[4]);

// Trace of extrinsic curvature
double traceK(const ADMPoint& pt, const double K[4]);

} // namespace geometry
