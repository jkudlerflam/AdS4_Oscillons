// test_equations.cpp — Tests for the Einstein-scalar equation residuals
// Checkpoint 4: Test A (pure AdS) and Test B (linear scalar)
// Checkpoint 6: Test C (Ricci tensor), Test D (full residuals on background),
//               Test E (full residuals with eigenmode)
#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cstring>
#include "spectral/chebyshev.h"
#include "spectral/legendre.h"
#include "geometry/ads4.h"
#include "geometry/adm.h"
#include "geometry/equations.h"

static const double PI = 3.14159265358979323846264338327950288;
static int tests_passed = 0;
static int tests_total = 0;

void check(bool condition, const char* name, double err = 0.0) {
    tests_total++;
    if (condition) {
        tests_passed++;
        printf("  [PASS] %s (err = %.2e)\n", name, err);
    } else {
        printf("  [FAIL] %s (err = %.2e)\n", name, err);
    }
}

// Forward declaration of scalarResidualBackground from equations.cpp
namespace geometry {
double scalarResidualBackground(
    double rho, double theta,
    double phi_val, double dphi_drho, double dphi_dtheta,
    double d2phi_drr, double d2phi_drt, double d2phi_dtt,
    double omega, int fourier_k,
    double m_sq);
}

// ============================================================================
// Test A: Pure AdS background, phi = 0  (from checkpoint 4)
// ============================================================================
void test_A_pure_ads() {
    printf("\n--- Test A: Pure AdS, all residuals = 0 ---\n");

    for (double rho : {0.1, 0.3, 0.5, 0.7}) {
        for (double theta : {0.5, 1.0, 2.0}) {
            double res = geometry::scalarResidualBackground(
                rho, theta,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                8.0, 1, 18.0);

            char name[64];
            snprintf(name, sizeof(name), "Test A: rho=%.1f, theta=%.1f", rho, theta);
            check(std::abs(res) < 1e-15, name, std::abs(res));
        }
    }
}

// ============================================================================
// Test B: Linear scalar on background  (from checkpoint 4)
// ============================================================================
void test_B_linear_scalar() {
    printf("\n--- Test B: Linear scalar eigenmode on background ---\n");

    int ell = 2;
    double Delta = 6.0;
    double m_sq = Delta * (Delta - 3.0);
    geometry::ScalarEigenmode mode(ell, Delta);
    double omega0 = mode.omega0();

    int N_rho = 30;
    spectral::Chebyshev cheb(N_rho, 0.05, 0.95);

    double theta_test = 0.8;
    double P_ell = spectral::Legendre::P(ell, cos(theta_test));

    std::vector<double> f_vals(N_rho + 1);
    for (int j = 0; j <= N_rho; j++) {
        f_vals[j] = mode.f_rho(cheb.grid(j));
    }

    std::vector<double> df(N_rho + 1), d2f(N_rho + 1);
    cheb.differentiate(f_vals.data(), df.data());
    cheb.differentiate2(f_vals.data(), d2f.data());

    double ct = cos(theta_test), st = sin(theta_test);
    double P_ell_m1 = spectral::Legendre::P(ell - 1, ct);
    double dP_dx = ell * (P_ell_m1 - ct * P_ell) / (1.0 - ct * ct);
    double dP_dtheta = -st * dP_dx;
    double d2P_dx2 = (2.0 * ct * dP_dx - ell * (ell + 1) * P_ell) / (1.0 - ct * ct);
    double d2P_dtheta2 = -ct * dP_dx + st * st * d2P_dx2;

    printf("  Scalar wave equation residual scaling:\n");

    std::vector<double> eps_vals = {1e-3, 1e-4, 1e-5, 1e-6};
    std::vector<double> max_residuals(eps_vals.size());

    for (size_t ie = 0; ie < eps_vals.size(); ie++) {
        double eps = eps_vals[ie];
        double max_res = 0.0;

        for (int j = 3; j <= N_rho - 3; j++) {
            double rho = cheb.grid(j);

            double phi_val = eps * f_vals[j] * P_ell;
            double dphi_drho = eps * df[j] * P_ell;
            double dphi_dtheta = eps * f_vals[j] * dP_dtheta;
            double d2phi_drr = eps * d2f[j] * P_ell;
            double d2phi_drt = eps * df[j] * dP_dtheta;
            double d2phi_dtt = eps * f_vals[j] * d2P_dtheta2;

            double res = geometry::scalarResidualBackground(
                rho, theta_test,
                phi_val, dphi_drho, dphi_dtheta,
                d2phi_drr, d2phi_drt, d2phi_dtt,
                omega0, 1, m_sq);

            max_res = std::max(max_res, std::abs(res));
        }

        max_residuals[ie] = max_res;
        printf("    eps = %.0e: max |residual| = %.6e\n", eps, max_res);
    }

    check(max_residuals[0] < 1e-4, "Test B: scalar residual small at eps=1e-3", max_residuals[0]);

    for (size_t ie = 1; ie < eps_vals.size(); ie++) {
        double ratio = max_residuals[ie] / max_residuals[ie-1];
        double expected_ratio = eps_vals[ie] / eps_vals[ie-1];
        double deviation = std::abs(ratio - expected_ratio) / expected_ratio;

        char name[80];
        snprintf(name, sizeof(name), "Test B: linear scaling eps=%.0e->%.0e (ratio=%.3f, expect=%.3f)",
                 eps_vals[ie-1], eps_vals[ie], ratio, expected_ratio);
        check(deviation < 0.1, name, deviation);
    }
}

// ============================================================================
// Test: Stress tensor components  (from checkpoint 4)
// ============================================================================
void test_stress_tensor() {
    printf("\n--- Scalar stress tensor ---\n");

    double phi = 1.0;
    double m_sq = 18.0;
    double gamma[4] = {4.0, 0.0, 1.0, 0.5};

    auto T = geometry::computeScalarStress(phi, 0.0, 0.0, 0.0, gamma, m_sq);

    check(std::abs(T.rho_E - 0.5 * m_sq) < 1e-14, "Stress: rho_E = m^2/2 for const phi",
          std::abs(T.rho_E - 0.5 * m_sq));
    check(std::abs(T.j_r) < 1e-15, "Stress: j_r = 0 for const phi", std::abs(T.j_r));
    check(std::abs(T.j_t) < 1e-15, "Stress: j_t = 0 for const phi", std::abs(T.j_t));

    double expected_Srr = -0.5 * gamma[0] * (0.0 - 0.0 - m_sq);
    check(std::abs(T.S_rr - expected_Srr) < 1e-14, "Stress: S_rr for const phi",
          std::abs(T.S_rr - expected_Srr));
}

// ============================================================================
// Test: Christoffel symbols on AdS4 background  (from checkpoint 4)
// ============================================================================
void test_background_christoffel() {
    printf("\n--- Background Christoffel symbols ---\n");
    using namespace geometry;

    double rho = 0.5, theta = PI / 3.0;
    auto G = AdS4Background::christoffel_bar(rho, theta);

    double r2 = rho * rho;
    double onemr2 = 1.0 - r2;

    double expected_Grr = 2.0 * rho / onemr2;
    check(std::abs(G.G_rho_rhorho - expected_Grr) < 1e-14,
          "Christoffel: G^rho_{rho rho}", std::abs(G.G_rho_rhorho - expected_Grr));

    double expected_Grt = (1.0 + r2) / (rho * onemr2);
    check(std::abs(G.G_theta_rhotheta - expected_Grt) < 1e-14,
          "Christoffel: G^theta_{rho theta}", std::abs(G.G_theta_rhotheta - expected_Grt));
}

// ============================================================================
// Test C: 3-Ricci tensor on AdS4 background
// On the H³ spatial slices: R_{ij} = -2 gamma_{ij}
// Equivalently: R_{ij}/gamma_{ij} = -2 for each diagonal component
// ============================================================================
void test_C_ricci_background() {
    printf("\n--- Test C: Ricci tensor on AdS4 background ---\n");
    using namespace geometry;

    // Test at several points
    for (double rho : {0.15, 0.3, 0.5, 0.7}) {
        for (double theta : {0.5, 1.0, 1.5}) {
            double r2 = rho*rho;
            double onemr2 = 1.0 - r2;
            double st = sin(theta), ct = cos(theta);

            // Physical background metric
            double g[4];
            g[IDX_RR] = 4.0 / (onemr2*onemr2);
            g[IDX_RT] = 0.0;
            g[IDX_TT] = 4.0*r2 / (onemr2*onemr2);
            g[IDX_PP] = 4.0*r2*st*st / (onemr2*onemr2);

            // d_rho gamma_{ij}
            // gamma_rr = 4/(1-r2)^2 => d_rho = 16*rho/(1-r2)^3
            // gamma_tt = 4*r2/(1-r2)^2 => d_rho = 8*rho*(1+r2)/(1-r2)^3
            // gamma_pp = gamma_tt * sin^2(theta)
            double dg_r[4];
            dg_r[IDX_RR] = 16.0*rho / (onemr2*onemr2*onemr2);
            dg_r[IDX_RT] = 0.0;
            dg_r[IDX_TT] = 8.0*rho*(1.0+r2) / (onemr2*onemr2*onemr2);
            dg_r[IDX_PP] = dg_r[IDX_TT] * st*st;

            // d_theta gamma_{ij}
            double dg_t[4];
            dg_t[IDX_RR] = 0.0;
            dg_t[IDX_RT] = 0.0;
            dg_t[IDX_TT] = 0.0;
            dg_t[IDX_PP] = 4.0*r2*2.0*st*ct / (onemr2*onemr2);

            // d2_rhorho gamma_{ij}
            // d_rho[16*rho/(1-r2)^3] = 16/(1-r2)^3 + 16*rho*6*rho/(1-r2)^4
            //    = 16(1+5*r2)/(1-r2)^4   (check: at rho=0 should be 16)
            // Actually let me compute carefully:
            // gamma_rr = 4*(1-r2)^{-2}
            // d_rho = 4*(-2)*(1-r2)^{-3}*(-2rho) = 16rho*(1-r2)^{-3}
            // d2_rho = 16*(1-r2)^{-3} + 16rho*(-3)*(1-r2)^{-4}*(-2rho)
            //        = 16*(1-r2)^{-3} + 96*r2*(1-r2)^{-4}
            //        = 16/(1-r2)^4 * [(1-r2) + 6*r2]
            //        = 16*(1+5*r2)/(1-r2)^4
            double d2g_rr[4];
            d2g_rr[IDX_RR] = 16.0*(1.0+5.0*r2) / (onemr2*onemr2*onemr2*onemr2);
            d2g_rr[IDX_RT] = 0.0;
            // gamma_tt = 4*r2*(1-r2)^{-2}
            // d_rho = 8*rho*(1-r2)^{-2} + 4*r2*16*rho*(1-r2)^{-3}/4 ...let me redo
            // d_rho gamma_tt = d_rho[4*r2/(1-r2)^2]
            //   = [8rho*(1-r2)^2 + 4*r2*4rho*(1-r2)] / (1-r2)^4
            //   = 8rho/(1-r2)^2 + 16*r2*rho/(1-r2)^3
            //   = 8rho[(1-r2) + 2*r2]/(1-r2)^3
            //   = 8rho(1+r2)/(1-r2)^3   ✓
            // d2_rho gamma_tt:
            //   d_rho[8rho(1+r2)/(1-r2)^3]
            //   = [8(1+r2) + 8rho*2rho]/(1-r2)^3 + 8rho(1+r2)*6rho/(1-r2)^4
            //   = [8(1+r2) + 16r2]/(1-r2)^3 + 48r2(1+r2)/(1-r2)^4
            //   = [8+8r2+16r2]/(1-r2)^3 + 48r2(1+r2)/(1-r2)^4
            //   = (8+24r2)/(1-r2)^3 + 48r2(1+r2)/(1-r2)^4
            //   = [(8+24r2)(1-r2) + 48r2(1+r2)] / (1-r2)^4
            //   = [8 - 8r2 + 24r2 - 24r2*r2 + 48r2 + 48r2*r2] / (1-r2)^4
            //   = [8 + 64r2 + 24r2*r2] / (1-r2)^4
            //   = 8(1 + 8r2 + 3r2*r2) / (1-r2)^4
            // Hmm let me verify at rho=0: should be 8*1/1 = 8.
            // gamma_tt at rho=0 is 0, d_rho = 0 (by L'Hopital, or just: gamma_tt = 4rho^2/...)
            // Actually d_rho gamma_tt = 8*rho*(1+r2)/(1-r2)^3, at rho=0 this is 0.
            // d2_rho at rho=0: 8(1+0+0)/1 = 8.
            // And gamma_tt ≈ 4*rho^2 near rho=0, so d2_rho ≈ 8. ✓
            d2g_rr[IDX_TT] = 8.0*(1.0 + 8.0*r2 + 3.0*r2*r2) / (onemr2*onemr2*onemr2*onemr2);
            d2g_rr[IDX_PP] = d2g_rr[IDX_TT] * st*st;

            // d2_rhotheta gamma_{ij} — only gamma_pp depends on theta
            double d2g_rt[4] = {0, 0, 0, 0};
            d2g_rt[IDX_PP] = dg_r[IDX_TT] * 2.0*st*ct; // d_rho(d_theta gamma_pp) = d_rho(gamma_tt) * 2*st*ct

            // d2_thetatheta gamma_{ij}
            double d2g_tt[4] = {0, 0, 0, 0};
            d2g_tt[IDX_PP] = 4.0*r2*(2.0*ct*ct - 2.0*st*st) / (onemr2*onemr2);
            // d_theta^2 [4*r2*sin^2(theta)/(1-r2)^2] = 4*r2/(1-r2)^2 * d^2/dtheta^2 sin^2(theta)
            // d^2/dtheta^2 sin^2(theta) = 2(cos^2 - sin^2) = 2*cos(2theta)
            d2g_tt[IDX_PP] = 4.0*r2*2.0*(ct*ct - st*st) / (onemr2*onemr2);

            double R[4];
            computeRicci3D_raw(g, dg_r, dg_t, d2g_rr, d2g_rt, d2g_tt, R);

            // Check: R_{ij} should equal -2*gamma_{ij}
            // For the diagonal: R_{rr} = -2*g_{rr}, R_{tt} = -2*g_{tt}, R_{pp} = -2*g_{pp}
            // For off-diagonal: R_{rt} = -2*g_{rt} = 0
            double err_rr = std::abs(R[IDX_RR] - (-2.0)*g[IDX_RR]) /
                           std::max(std::abs(g[IDX_RR]), 1e-30);
            double err_rt = std::abs(R[IDX_RT]);
            double err_tt = std::abs(R[IDX_TT] - (-2.0)*g[IDX_TT]) /
                           std::max(std::abs(g[IDX_TT]), 1e-30);
            double err_pp = std::abs(R[IDX_PP] - (-2.0)*g[IDX_PP]) /
                           std::max(std::abs(g[IDX_PP]), 1e-30);

            double max_err = std::max({err_rr, err_rt, err_tt, err_pp});

            char name[80];
            snprintf(name, sizeof(name), "Test C: Ricci R_{ij}=-2g_{ij}, rho=%.2f, theta=%.1f",
                     rho, theta);
            check(max_err < 1e-10, name, max_err);

            if (max_err > 1e-10) {
                printf("    R_rr/g_rr = %.10f (expect -2), err=%.2e\n", R[IDX_RR]/g[IDX_RR], err_rr);
                printf("    R_rt     = %.10e\n", R[IDX_RT]);
                printf("    R_tt/g_tt = %.10f (expect -2), err=%.2e\n", R[IDX_TT]/g[IDX_TT], err_tt);
                printf("    R_pp/g_pp = %.10f (expect -2), err=%.2e\n", R[IDX_PP]/g[IDX_PP], err_pp);
            }
        }
    }
}

// ============================================================================
// Test D: Full residuals on pure AdS background (all should be zero)
// ============================================================================
void test_D_full_residuals_ads() {
    printf("\n--- Test D: Full residuals on pure AdS (should be zero) ---\n");
    using namespace geometry;

    for (double rho : {0.15, 0.3, 0.5, 0.7}) {
        for (double theta : {0.5, 1.0}) {
            // Build FullPointData for the AdS background
            FullPointData data;
            std::memset(&data, 0, sizeof(data));
            data.rho = rho;
            data.theta = theta;
            data.Delta = 6.0;
            data.Lambda = -3.0;

            // Hatted background values: N_hat = 1, beta = 0, gamma_hat = Omega^2 * gamma_bar
            data.fields[FLD_LAPSE] = 1.0; // N_hat = 1
            data.fields[FLD_SHIFT_R] = 0.0;
            data.fields[FLD_SHIFT_T] = 0.0;
            data.fields[FLD_GAMMA_RR] = AdS4Background::gammahat_bar_rhorho(rho);
            data.fields[FLD_GAMMA_RT] = 0.0;
            data.fields[FLD_GAMMA_TT] = AdS4Background::gammahat_bar_thetatheta(rho, theta);
            data.fields[FLD_GAMMA_PP] = AdS4Background::gammahat_bar_phiphi(rho, theta);
            data.fields[FLD_SCALAR] = 0.0;

            // Spatial derivatives of hatted background fields
            double r2 = rho*rho, s = 1.0+r2;
            // gammahat_rr = 4/(1+rho^2)^2
            // d_rho = -16*rho/(1+rho^2)^3
            data.dr[FLD_GAMMA_RR] = -16.0*rho / (s*s*s);
            // d2_rho = (-16*(1+r2)^3 + 16*rho*3*(1+r2)^2*2*rho)/(1+r2)^6
            //        = (-16 + 96*r2)/(1+r2)^4 = 16*(6r2-1)/(1+r2)^4
            //   Hmm, let me recompute:
            //   d_rho[-16*rho/(1+r2)^3] = -16/(1+r2)^3 + 16*rho*6*rho/(1+r2)^4
            //     = (-16*(1+r2) + 96*r2)/(1+r2)^4 = (-16+80*r2)/(1+r2)^4
            //     = 16*(5*r2-1)/(1+r2)^4
            data.d2rr[FLD_GAMMA_RR] = 16.0*(5.0*r2 - 1.0) / (s*s*s*s);

            // gammahat_tt = 4*rho^2/(1+rho^2)^2
            // d_rho = (8*rho*(1+r2)^2 - 4*r2*4*rho*(1+r2))/(1+r2)^4
            //       = 8*rho/(1+r2)^2 - 16*r2*rho/(1+r2)^3
            //       = 8*rho((1+r2) - 2*r2)/(1+r2)^3
            //       = 8*rho*(1-r2)/(1+r2)^3
            data.dr[FLD_GAMMA_TT] = 8.0*rho*(1.0-r2) / (s*s*s);
            // d2_rho: d_rho[8*rho*(1-r2)/(1+r2)^3]
            //   = [8*(1-r2) + 8*rho*(-2*rho)]/(1+r2)^3 + 8*rho*(1-r2)*(-6*rho)/(1+r2)^4
            //   = [8-8r2-16r2]/(1+r2)^3 - 48r2*(1-r2)/(1+r2)^4
            //   = (8-24r2)/(1+r2)^3 - 48r2*(1-r2)/(1+r2)^4
            //   = [(8-24r2)(1+r2) - 48r2+48r2*r2]/(1+r2)^4
            //   = [8+8r2-24r2-24r4-48r2+48r4]/(1+r2)^4
            //   = [8-64r2+24r4]/(1+r2)^4
            //   = 8*(1-8r2+3r4)/(1+r2)^4
            data.d2rr[FLD_GAMMA_TT] = 8.0*(1.0-8.0*r2+3.0*r2*r2) / (s*s*s*s);

            // gammahat_pp = gammahat_tt * sin^2(theta)
            double st = sin(theta), ct = cos(theta);
            data.dr[FLD_GAMMA_PP] = data.dr[FLD_GAMMA_TT] * st*st;
            data.d2rr[FLD_GAMMA_PP] = data.d2rr[FLD_GAMMA_TT] * st*st;

            data.dtheta[FLD_GAMMA_PP] = data.fields[FLD_GAMMA_TT] * 2.0*st*ct;
            data.d2rt[FLD_GAMMA_PP] = data.dr[FLD_GAMMA_TT] * 2.0*st*ct;
            data.d2tt[FLD_GAMMA_PP] = data.fields[FLD_GAMMA_TT] * 2.0*(ct*ct - st*st);

            // Lapse: N_hat = 1, all derivatives = 0

            EquationResiduals res;
            computeFullResiduals(data, res);

            double max_err = std::max({std::abs(res.hamiltonian),
                                        std::abs(res.momentum_r),
                                        std::abs(res.momentum_t),
                                        std::abs(res.evolution_rr),
                                        std::abs(res.evolution_rt),
                                        std::abs(res.evolution_tt),
                                        std::abs(res.evolution_pp),
                                        std::abs(res.scalar_wave)});

            char name[80];
            snprintf(name, sizeof(name), "Test D: all residuals=0 on AdS, rho=%.2f, theta=%.1f",
                     rho, theta);
            check(max_err < 1e-10, name, max_err);

            if (max_err > 1e-10) {
                printf("    hamiltonian = %.2e\n", res.hamiltonian);
                printf("    momentum_r  = %.2e\n", res.momentum_r);
                printf("    momentum_t  = %.2e\n", res.momentum_t);
                printf("    evolution_rr = %.2e\n", res.evolution_rr);
                printf("    evolution_rt = %.2e\n", res.evolution_rt);
                printf("    evolution_tt = %.2e\n", res.evolution_tt);
                printf("    evolution_pp = %.2e\n", res.evolution_pp);
                printf("    scalar_wave = %.2e\n", res.scalar_wave);
            }
        }
    }
}

// ============================================================================
// Test E: Full residuals with linear eigenmode seed
// ============================================================================
// On the linear seed (background metric + eps*phi_eigenmode):
//   - Constraints (trK, V^i) should be exactly 0 (metric is background)
//   - Scalar wave should be ~0 (eigenmode solves linearized eq, Box is linear)
//   - K evolution residuals should be O(eps^2) (matter stress is quadratic in phi)
void test_E_full_residuals_eigenmode() {
    printf("\n--- Test E: Full residuals with linear eigenmode seed ---\n");
    using namespace geometry;

    int ell = 2;
    double Delta = 6.0;
    double m_sq = Delta * (Delta - 3.0);
    ScalarEigenmode mode(ell, Delta);
    double omega0 = mode.omega0();

    // Set up Chebyshev grid for radial differentiation of hatted scalar
    int N_rho = 40;
    spectral::Chebyshev cheb(N_rho, 0.05, 0.95);

    double theta_test = 0.8;
    double ct = cos(theta_test), st = sin(theta_test);
    double P_ell = spectral::Legendre::P(ell, ct);
    double P_ell_m1 = spectral::Legendre::P(ell - 1, ct);
    double dP_dx = ell * (P_ell_m1 - ct * P_ell) / (1.0 - ct * ct);
    double dP_dtheta = -st * dP_dx;
    double d2P_dx2 = (2.0 * ct * dP_dx - ell * (ell + 1) * P_ell) / (1.0 - ct * ct);
    double d2P_dtheta2 = -ct * dP_dx + st * st * d2P_dx2;

    // Compute hatted radial profile: h(rho) = f(rho) / Omega^{Delta/2}
    // = [2*rho/(1+rho^2)]^ell * Omega^{Delta/2}
    std::vector<double> h_vals(N_rho + 1);
    for (int j = 0; j <= N_rho; j++) {
        double rho = cheb.grid(j);
        double Om = AdS4Background::Omega(rho);
        double sin_xbar = 2.0*rho / (1.0 + rho*rho);
        h_vals[j] = pow(sin_xbar, ell) * pow(Om, Delta/2.0);
    }

    std::vector<double> dh(N_rho + 1), d2h(N_rho + 1);
    cheb.differentiate(h_vals.data(), dh.data());
    cheb.differentiate2(h_vals.data(), d2h.data());

    // Test at several epsilon values
    std::vector<double> eps_vals = {1e-2, 1e-3, 1e-4, 1e-5};
    std::vector<double> max_evol_res(eps_vals.size());
    std::vector<double> max_scalar_res(eps_vals.size());
    std::vector<double> max_constraint_res(eps_vals.size());

    // Pick a representative interior point
    int j_test = N_rho / 3; // avoid boundaries
    double rho = cheb.grid(j_test);

    for (size_t ie = 0; ie < eps_vals.size(); ie++) {
        double eps = eps_vals[ie];

        FullPointData data;
        std::memset(&data, 0, sizeof(data));
        data.rho = rho;
        data.theta = theta_test;
        data.Delta = Delta;
        data.Lambda = -3.0;

        double r2 = rho*rho, s = 1.0 + r2;

        // --- Background metric (same as Test D) ---
        data.fields[FLD_LAPSE] = 1.0;
        data.fields[FLD_GAMMA_RR] = AdS4Background::gammahat_bar_rhorho(rho);
        data.fields[FLD_GAMMA_RT] = 0.0;
        data.fields[FLD_GAMMA_TT] = AdS4Background::gammahat_bar_thetatheta(rho, theta_test);
        data.fields[FLD_GAMMA_PP] = AdS4Background::gammahat_bar_phiphi(rho, theta_test);

        // Spatial derivatives of hatted background metric
        data.dr[FLD_GAMMA_RR] = -16.0*rho / (s*s*s);
        data.d2rr[FLD_GAMMA_RR] = 16.0*(5.0*r2 - 1.0) / (s*s*s*s);
        data.dr[FLD_GAMMA_TT] = 8.0*rho*(1.0-r2) / (s*s*s);
        data.d2rr[FLD_GAMMA_TT] = 8.0*(1.0-8.0*r2+3.0*r2*r2) / (s*s*s*s);
        data.dr[FLD_GAMMA_PP] = data.dr[FLD_GAMMA_TT] * st*st;
        data.d2rr[FLD_GAMMA_PP] = data.d2rr[FLD_GAMMA_TT] * st*st;
        data.dtheta[FLD_GAMMA_PP] = data.fields[FLD_GAMMA_TT] * 2.0*st*ct;
        data.d2rt[FLD_GAMMA_PP] = data.dr[FLD_GAMMA_TT] * 2.0*st*ct;
        data.d2tt[FLD_GAMMA_PP] = data.fields[FLD_GAMMA_TT] * 2.0*(ct*ct - st*st);

        // --- Scalar eigenmode (hatted): phi_hat = eps * h(rho) * P_ell(cos theta) ---
        data.fields[FLD_SCALAR] = eps * h_vals[j_test] * P_ell;
        data.dr[FLD_SCALAR] = eps * dh[j_test] * P_ell;
        data.dtheta[FLD_SCALAR] = eps * h_vals[j_test] * dP_dtheta;
        data.d2rr[FLD_SCALAR] = eps * d2h[j_test] * P_ell;
        data.d2rt[FLD_SCALAR] = eps * dh[j_test] * dP_dtheta;
        data.d2tt[FLD_SCALAR] = eps * h_vals[j_test] * d2P_dtheta2;

        // Time derivatives at t=0: phi ~ cos(omega*t), so d_t phi = 0, d2_t phi = -omega^2 phi
        data.dt[FLD_SCALAR] = 0.0;  // d_t phi_hat = 0 at t=0
        data.d2t[FLD_SCALAR] = -omega0*omega0 * data.fields[FLD_SCALAR]; // d2_t phi_hat

        EquationResiduals res;
        computeFullResiduals(data, res);

        max_constraint_res[ie] = std::max({std::abs(res.hamiltonian),
                                            std::abs(res.momentum_r),
                                            std::abs(res.momentum_t)});
        max_evol_res[ie] = std::max({std::abs(res.evolution_rr),
                                      std::abs(res.evolution_rt),
                                      std::abs(res.evolution_tt),
                                      std::abs(res.evolution_pp)});
        max_scalar_res[ie] = std::abs(res.scalar_wave);

        printf("    eps=%.0e: constraints=%.2e, evol=%.2e, scalar=%.2e\n",
               eps, max_constraint_res[ie], max_evol_res[ie], max_scalar_res[ie]);
    }

    // Constraints should be exactly 0 (metric is background, K=0)
    check(max_constraint_res[0] < 1e-10,
          "Test E: constraints ~0 on eigenmode seed", max_constraint_res[0]);

    // Scalar wave should be small (eigenmode solves linearized eq on background)
    // The residual comes from numerical differentiation errors, should be O(eps) not O(1)
    check(max_scalar_res[0] < 1e-4,
          "Test E: scalar wave residual small on eigenmode", max_scalar_res[0]);

    // Evolution residuals should scale as eps^2 (matter stress is quadratic)
    // Only check ratios where the signal is well above the background floor (~1e-13)
    printf("  Evolution residual scaling (expect eps^2):\n");
    for (size_t ie = 1; ie < eps_vals.size(); ie++) {
        double ratio = max_evol_res[ie] / max_evol_res[ie-1];
        double eps_ratio = eps_vals[ie] / eps_vals[ie-1];
        double expected_ratio = eps_ratio * eps_ratio; // quadratic scaling

        printf("    eps %.0e->%.0e: ratio=%.4f, expect=%.4f\n",
               eps_vals[ie-1], eps_vals[ie], ratio, expected_ratio);

        // Only test scaling when both values are well above background noise floor
        if (max_evol_res[ie] > 1e-12 && max_evol_res[ie-1] > 1e-12) {
            char name[100];
            snprintf(name, sizeof(name),
                     "Test E: evol scales as eps^2, eps=%.0e->%.0e",
                     eps_vals[ie-1], eps_vals[ie]);
            double deviation = std::abs(ratio - expected_ratio) / expected_ratio;
            check(deviation < 0.15, name, deviation);
        }
    }
}

int main() {
    printf("===== EQUATION TESTS =====\n");
    printf("Checkpoint 4: Tests A and B\n");

    test_A_pure_ads();
    test_B_linear_scalar();
    test_stress_tensor();
    test_background_christoffel();

    printf("\n===== CHECKPOINT 6: Full Nonlinear Equations =====\n");
    test_C_ricci_background();
    test_D_full_residuals_ads();
    test_E_full_residuals_eigenmode();

    printf("\n===================================\n");
    printf("Results: %d / %d tests passed\n", tests_passed, tests_total);
    printf("===================================\n");

    return (tests_passed == tests_total) ? 0 : 1;
}
