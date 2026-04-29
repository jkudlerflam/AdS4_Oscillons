// adm.cpp — ADM decomposition implementation (stub for compilation)
#include "geometry/adm.h"
#include <cmath>
#include <cstring>

namespace geometry {

Christoffel3D computeChristoffel3D(const double gamma[4],
                                     const double dgamma_drho[4],
                                     const double dgamma_dtheta[4],
                                     double rho, double theta) {
    Christoffel3D G;
    std::memset(&G, 0, sizeof(G));

    // gamma_{ij}: rr, rt, tt, pp
    // Inverse 2x2 block (rho, theta) plus 1/gamma_pp for phi
    double grr = gamma[IDX_RR], grt = gamma[IDX_RT];
    double gtt = gamma[IDX_TT], gpp = gamma[IDX_PP];

    double det2 = grr * gtt - grt * grt;
    if (std::abs(det2) < 1e-30) return G;

    double inv_rr = gtt / det2;
    double inv_rt = -grt / det2;
    double inv_tt = grr / det2;
    double inv_pp = (std::abs(gpp) > 1e-30) ? 1.0 / gpp : 0.0;

    // d_rho gamma_{ij} and d_theta gamma_{ij}
    // Gamma^i_{jk} = 0.5 * g^{il} (d_j g_{lk} + d_k g_{lj} - d_l g_{jk})

    // Coordinates: 0=rho, 1=theta, 2=phi
    // gamma has nonzero: (0,0), (0,1)=(1,0), (1,1), (2,2)
    // Derivatives: d_0 = d_rho, d_1 = d_theta (d_2 = 0 by axisymmetry)

    double dg[2][4]; // dg[coord][metric_component]
    for (int c = 0; c < 4; c++) {
        dg[0][c] = dgamma_drho[c];
        dg[1][c] = dgamma_dtheta[c];
    }

    // Build full Christoffel - this is the detailed implementation
    // For now, set the key components

    // Gamma^rho_{rho rho} = 0.5 * g^{rho l} (d_rho g_{l rho} + d_rho g_{l rho} - d_l g_{rho rho})
    // = 0.5 * (g^{rr}(2 d_rho g_{rr} - d_rho g_{rr}) + g^{rt}(2 d_rho g_{rt} - d_theta g_{rr}))
    // = 0.5 * (g^{rr} d_rho g_{rr} + g^{rt}(2 d_rho g_{rt} - d_theta g_{rr}))

    // This is getting complex - implement it properly with a loop
    // g^{ab} for a,b in {rho, theta}: inv_rr, inv_rt, inv_tt
    // g^{phi phi} = inv_pp

    // Gamma^a_{bc} = 0.5 * g^{ad} (d_b g_{dc} + d_c g_{db} - d_d g_{bc})
    // where a,b,c,d in {rho, theta, phi}

    // Helper: get metric component
    auto gcomp = [&](int i, int j) -> double {
        if (i > j) std::swap(i, j);
        if (i == 0 && j == 0) return gamma[IDX_RR];
        if (i == 0 && j == 1) return gamma[IDX_RT];
        if (i == 1 && j == 1) return gamma[IDX_TT];
        if (i == 2 && j == 2) return gamma[IDX_PP];
        return 0.0; // cross terms with phi
    };

    // Helper: get derivative of metric component
    // d_coord g_{ij}: coord in {0=rho, 1=theta}, phi-derivative = 0
    auto dgcomp = [&](int coord, int i, int j) -> double {
        if (coord >= 2) return 0.0; // no phi-derivative
        if (i > j) std::swap(i, j);
        if (i == 0 && j == 0) return dg[coord][IDX_RR];
        if (i == 0 && j == 1) return dg[coord][IDX_RT];
        if (i == 1 && j == 1) return dg[coord][IDX_TT];
        if (i == 2 && j == 2) return dg[coord][IDX_PP];
        return 0.0;
    };

    // Inverse metric: g^{ab}
    auto ginv = [&](int i, int j) -> double {
        if (i > j) std::swap(i, j);
        if (i == 0 && j == 0) return inv_rr;
        if (i == 0 && j == 1) return inv_rt;
        if (i == 1 && j == 1) return inv_tt;
        if (i == 2 && j == 2) return inv_pp;
        return 0.0;
    };

    // Compute all Christoffel symbols
    for (int a = 0; a < 3; a++) {
        for (int b = 0; b < 3; b++) {
            for (int c = b; c < 3; c++) { // symmetric in b,c
                double val = 0.0;
                for (int d = 0; d < 3; d++) {
                    val += ginv(a, d) * (dgcomp(b, d, c) + dgcomp(c, d, b) - dgcomp(d, b, c));
                }
                val *= 0.5;
                G.G[a][b][c] = val;
                G.G[a][c][b] = val;
            }
        }
    }

    return G;
}

void computeExtrinsicCurvature(const ADMPoint& pt, double K[4]) {
    // K_{ij} = -(1/(2N))(d_t gamma_{ij} - D_i beta_j - D_j beta_i)
    // D_i beta_j = d_i beta_j - Gamma^k_{ij} beta_k
    // beta_j = gamma_{jk} beta^k (lower index)

    // For now, simplified computation
    double N = pt.N;
    if (std::abs(N) < 1e-30) {
        K[0] = K[1] = K[2] = K[3] = 0.0;
        return;
    }

    // Lower the shift: beta_i = gamma_{ij} beta^j
    double beta_low_r = pt.gamma[IDX_RR] * pt.beta_r + pt.gamma[IDX_RT] * pt.beta_t;
    double beta_low_t = pt.gamma[IDX_RT] * pt.beta_r + pt.gamma[IDX_TT] * pt.beta_t;

    // Lie derivative of gamma along beta (partial terms)
    // L_beta gamma_{ij} = beta^k d_k gamma_{ij} + gamma_{kj} d_i beta^k + gamma_{ik} d_j beta^k
    double Lb[4];
    Lb[IDX_RR] = pt.beta_r * pt.dgamma_drho[IDX_RR] + pt.beta_t * pt.dgamma_dtheta[IDX_RR]
                 + 2.0 * (pt.gamma[IDX_RR] * pt.dbeta_r_drho + pt.gamma[IDX_RT] * pt.dbeta_t_drho);
    Lb[IDX_RT] = pt.beta_r * pt.dgamma_drho[IDX_RT] + pt.beta_t * pt.dgamma_dtheta[IDX_RT]
                 + pt.gamma[IDX_RR] * pt.dbeta_r_dtheta + pt.gamma[IDX_RT] * pt.dbeta_t_dtheta
                 + pt.gamma[IDX_RT] * pt.dbeta_r_drho + pt.gamma[IDX_TT] * pt.dbeta_t_drho;
    Lb[IDX_TT] = pt.beta_r * pt.dgamma_drho[IDX_TT] + pt.beta_t * pt.dgamma_dtheta[IDX_TT]
                 + 2.0 * (pt.gamma[IDX_RT] * pt.dbeta_r_dtheta + pt.gamma[IDX_TT] * pt.dbeta_t_dtheta);
    Lb[IDX_PP] = pt.beta_r * pt.dgamma_drho[IDX_PP] + pt.beta_t * pt.dgamma_dtheta[IDX_PP];

    for (int c = 0; c < 4; c++) {
        K[c] = -(pt.dgamma_dt[c] - Lb[c]) / (2.0 * N);
    }
}

double traceK(const ADMPoint& pt, const double K[4]) {
    // tr K = gamma^{ij} K_{ij}
    double grr = pt.gamma[IDX_RR], grt = pt.gamma[IDX_RT];
    double gtt = pt.gamma[IDX_TT], gpp = pt.gamma[IDX_PP];
    double det2 = grr * gtt - grt * grt;
    if (std::abs(det2) < 1e-30) return 0.0;

    double inv_rr = gtt / det2;
    double inv_rt = -grt / det2;
    double inv_tt = grr / det2;
    double inv_pp = (std::abs(gpp) > 1e-30) ? 1.0 / gpp : 0.0;

    return inv_rr * K[IDX_RR] + 2.0 * inv_rt * K[IDX_RT]
         + inv_tt * K[IDX_TT] + inv_pp * K[IDX_PP];
}

// ============================================================================
// Full 3-Ricci tensor computation
// ============================================================================
// Uses R_{ij} = ∂_k Γ^k_{ij} - ∂_j Γ^k_{ik} + Γ^k_{km}Γ^m_{ij} - Γ^k_{jm}Γ^m_{ik}
// Coordinates: 0=ρ, 1=θ, 2=φ (axisymmetric: ∂_2 = 0)
// Non-zero metric: γ_{00}, γ_{01}, γ_{11}, γ_{22}

void computeRicci3D_raw(
    const double g[4],
    const double dg_r[4], const double dg_t[4],
    const double d2g_rr[4], const double d2g_rt[4], const double d2g_tt[4],
    double R[4])
{
    // Inverse metric
    double det2 = g[IDX_RR] * g[IDX_TT] - g[IDX_RT] * g[IDX_RT];
    if (std::abs(det2) < 1e-30) { R[0]=R[1]=R[2]=R[3]=0; return; }
    double gi00 = g[IDX_TT] / det2;
    double gi01 = -g[IDX_RT] / det2;
    double gi11 = g[IDX_RR] / det2;
    double gi22 = (std::abs(g[IDX_PP]) > 1e-30) ? 1.0 / g[IDX_PP] : 0.0;

    // ---------- Helpers ----------
    // Map (i,j) with i<=j to MetricIdx; return -1 for cross-terms with φ
    auto cidx = [](int i, int j) -> int {
        if (i > j) std::swap(i, j);
        if (i==0 && j==0) return IDX_RR;
        if (i==0 && j==1) return IDX_RT;
        if (i==1 && j==1) return IDX_TT;
        if (i==2 && j==2) return IDX_PP;
        return -1;
    };

    // γ_{ij}
    auto gmet = [&](int i, int j) -> double {
        int c = cidx(i, j);
        return (c >= 0) ? g[c] : 0.0;
    };

    // γ^{ij}
    auto ginv = [&](int i, int j) -> double {
        if (i > j) std::swap(i, j);
        if (i==0 && j==0) return gi00;
        if (i==0 && j==1) return gi01;
        if (i==1 && j==1) return gi11;
        if (i==2 && j==2) return gi22;
        return 0.0;
    };

    // ∂_a γ_{ij}, a ∈ {0,1,2}
    auto dg1 = [&](int a, int i, int j) -> double {
        if (a == 2) return 0.0;
        int c = cidx(i, j);
        if (c < 0) return 0.0;
        return (a == 0) ? dg_r[c] : dg_t[c];
    };

    // ∂²_{ab} γ_{ij}, a,b ∈ {0,1,2}
    auto dg2 = [&](int a, int b, int i, int j) -> double {
        if (a == 2 || b == 2) return 0.0;
        int c = cidx(i, j);
        if (c < 0) return 0.0;
        if (a == 0 && b == 0) return d2g_rr[c];
        if ((a==0 && b==1) || (a==1 && b==0)) return d2g_rt[c];
        if (a == 1 && b == 1) return d2g_tt[c];
        return 0.0;
    };

    // ---------- Step 1: Christoffel symbols Γ^a_{bc} ----------
    double Gamma[3][3][3];
    for (int a = 0; a < 3; a++)
        for (int b = 0; b < 3; b++)
            for (int c = b; c < 3; c++) {
                double v = 0.0;
                for (int m = 0; m < 3; m++)
                    v += ginv(a, m) * (dg1(b, c, m) + dg1(c, b, m) - dg1(m, b, c));
                v *= 0.5;
                Gamma[a][b][c] = v;
                Gamma[a][c][b] = v;
            }

    // ---------- Step 2: ∂_e γ^{am} = -γ^{ap}γ^{mq} ∂_e γ_{pq} ----------
    double dginv[2][3][3]; // dginv[e][a][m]
    for (int e = 0; e < 2; e++)
        for (int a = 0; a < 3; a++)
            for (int m = 0; m < 3; m++) {
                double v = 0.0;
                for (int p = 0; p < 3; p++)
                    for (int q = 0; q < 3; q++)
                        v -= ginv(a, p) * ginv(m, q) * dg1(e, p, q);
                dginv[e][a][m] = v;
            }

    // ---------- Step 3: ∂_e Γ^a_{bc} ----------
    // ∂_e Γ^a_{bc} = ½(∂_e γ^{am})(∂_b γ_{cm} + ∂_c γ_{bm} - ∂_m γ_{bc})
    //              + ½ γ^{am}(∂_e∂_b γ_{cm} + ∂_e∂_c γ_{bm} - ∂_e∂_m γ_{bc})
    double dGamma[2][3][3][3]; // dGamma[e][a][b][c]
    for (int e = 0; e < 2; e++)
        for (int a = 0; a < 3; a++)
            for (int b = 0; b < 3; b++)
                for (int c = b; c < 3; c++) {
                    double v = 0.0;
                    for (int m = 0; m < 3; m++) {
                        double S1 = dg1(b, c, m) + dg1(c, b, m) - dg1(m, b, c);
                        double S2 = dg2(e, b, c, m) + dg2(e, c, b, m) - dg2(e, m, b, c);
                        v += dginv[e][a][m] * S1 + ginv(a, m) * S2;
                    }
                    v *= 0.5;
                    dGamma[e][a][b][c] = v;
                    dGamma[e][a][c][b] = v;
                }

    // ---------- Step 4: Assemble R_{ij} ----------
    // R_{ij} = ∂_k Γ^k_{ij} - ∂_j Γ^k_{ik} + Γ^k_{km}Γ^m_{ij} - Γ^k_{jm}Γ^m_{ik}
    int pairs[4][2] = {{0,0},{0,1},{1,1},{2,2}};
    for (int p = 0; p < 4; p++) {
        int i = pairs[p][0], j = pairs[p][1];

        // Term 1: ∂_k Γ^k_{ij} (sum k=0,1; ∂_2 = 0)
        double t1 = 0.0;
        for (int k = 0; k < 2; k++)
            t1 += dGamma[k][k][i][j];

        // Term 2: -∂_j Γ^k_{ik} (sum over k; j-derivative only if j<2)
        double t2 = 0.0;
        if (j < 2) {
            for (int k = 0; k < 3; k++)
                t2 -= dGamma[j][k][i][k];
        }
        // If j=2, ∂_j = ∂_φ = 0, so t2 = 0.

        // Term 3: Γ^k_{km}Γ^m_{ij}
        double t3 = 0.0;
        for (int k = 0; k < 3; k++)
            for (int m = 0; m < 3; m++)
                t3 += Gamma[k][k][m] * Gamma[m][i][j];

        // Term 4: -Γ^k_{jm}Γ^m_{ik}
        double t4 = 0.0;
        for (int k = 0; k < 3; k++)
            for (int m = 0; m < 3; m++)
                t4 -= Gamma[k][j][m] * Gamma[m][i][k];

        R[p] = t1 + t2 + t3 + t4;
    }
}

void computeRicci3D(const ADMPoint& pt, double R[4]) {
    computeRicci3D_raw(
        pt.gamma,
        pt.dgamma_drho, pt.dgamma_dtheta,
        pt.d2gamma_drr, pt.d2gamma_drt, pt.d2gamma_dtt,
        R);
}

} // namespace geometry
