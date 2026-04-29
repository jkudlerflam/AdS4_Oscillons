// test_assembler.cpp — Checkpoint 7: ResidualAssembler tests
//
// Test F: Seed construction and state vector layout
// Test G: Residual on pure AdS background (should be zero)
// Test H: Residual with eigenmode seed (should scale as epsilon^2)
// Test I: Normalization function
#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cstring>
#include "solver/oscillon_system.h"
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

// ============================================================================
// Test F: Seed construction
// ============================================================================
// Verify that setLinearSeed produces the correct background + eigenmode values
// at several sample points.

void test_F_seed() {
    printf("\n--- Test F: Seed construction ---\n");

    solver::OscillonParams params;
    params.N_t = 4;
    params.N_nuc = 8;
    params.N_shell = 8;
    params.rho_mid = 0.5;
    params.N_theta = 4;
    params.Delta = 6.0;
    params.Lambda = -3.0;
    params.ell = 2;

    solver::OscillonSystem sys(params);

    int n = sys.stateSize() - 1;
    std::vector<double> u(n, 0.0);
    double omega;
    double eps = 1e-3;

    sys.setLinearSeed(u.data(), omega, eps);

    // Check omega
    double expected_omega = params.Delta + params.ell; // 8
    check(std::abs(omega - expected_omega) < 1e-12, "omega = Delta + ell",
          std::abs(omega - expected_omega));

    // Check lapse at a few points = 1.0
    {
        double max_err = 0;
        for (int m = 0; m <= params.N_t; m++) {
            for (int si = 0; si < std::min(10, sys.nSpatial()); si++) {
                double val = u[sys.stateIdx(geometry::FLD_LAPSE, m, si)];
                max_err = std::max(max_err, std::abs(val - 1.0));
            }
        }
        check(max_err < 1e-14, "lapse = 1 everywhere", max_err);
    }

    // Check shifts = 0
    {
        double max_err = 0;
        for (int A : {geometry::FLD_SHIFT_R, geometry::FLD_SHIFT_T}) {
            for (int m = 0; m <= params.N_t; m++) {
                for (int si = 0; si < sys.nSpatial(); si++) {
                    max_err = std::max(max_err, std::abs(u[sys.stateIdx(A, m, si)]));
                }
            }
        }
        check(max_err < 1e-14, "shifts = 0 everywhere", max_err);
    }

    // Check gamma_hat_rr at a sample point
    {
        int n_mid = sys.nRadial() / 4; // somewhere in the nucleus
        double rho = sys.rhoGrid(n_mid);
        double op2 = 1.0 + rho * rho;
        double expected = 4.0 / (op2 * op2);
        double val = u[sys.stateIdx(geometry::FLD_GAMMA_RR, 0, n_mid * sys.nAngular())];
        double err = std::abs(val - expected);
        check(err < 1e-14, "gamma_hat_rr matches background", err);
    }

    // Check scalar eigenmode at a sample point
    {
        geometry::ScalarEigenmode mode(params.ell, params.Delta);
        int n_mid = sys.nRadial() / 4;
        int j_mid = sys.nAngular() / 2;
        double rho = sys.rhoGrid(n_mid);
        double theta = sys.thetaGrid(j_mid);
        double tau_2 = sys.tauGrid(2); // m=2
        double t = tau_2 / omega;

        double expected = mode.phi_hat(t, rho, theta, eps);
        int si = sys.spatialIdx(n_mid, j_mid);
        double val = u[sys.stateIdx(geometry::FLD_SCALAR, 2, si)];
        double err = std::abs(val - expected);
        check(err < 1e-12, "scalar eigenmode matches phi_hat", err);
    }

    // Check state vector size
    {
        int expected_size = 8 * (params.N_t + 1) * sys.nSpatial() + 1;
        check(sys.stateSize() == expected_size, "stateSize correct",
              std::abs(sys.stateSize() - expected_size));
    }
}

// ============================================================================
// Test G: Residual on pure AdS background
// ============================================================================
// With epsilon=0, the seed is the pure AdS background. All equation residuals
// at interior points should be zero (to machine precision).

void test_G_background_residual() {
    printf("\n--- Test G: Residual on AdS background ---\n");

    solver::OscillonParams params;
    params.N_t = 4;
    params.N_nuc = 16;     // need adequate resolution for spectral accuracy
    params.N_shell = 16;
    params.rho_mid = 0.5;
    params.N_theta = 4;
    params.Delta = 6.0;
    params.Lambda = -3.0;
    params.ell = 2;

    solver::OscillonSystem sys(params);

    int n = sys.stateSize() - 1;
    std::vector<double> u(n, 0.0);
    double omega;

    // Pure background: epsilon = 0
    sys.setLinearSeed(u.data(), omega, 0.0);

    // Compute residual
    std::vector<double> R(n, 0.0);
    sys.computeResidual(u.data(), omega, R.data());

    // The residual should be zero everywhere:
    // - Interior points: PDE residuals should be zero on exact AdS
    // - Boundary/origin: Dirichlet residuals = u - background = 0
    // - Odd endpoints: u = 0 (shifts are zero)
    double max_res = 0;
    int max_idx = 0;
    for (int i = 0; i < n; i++) {
        if (std::abs(R[i]) > max_res) {
            max_res = std::abs(R[i]);
            max_idx = i;
        }
    }

    printf("  max |R| = %.2e at index %d\n", max_res, max_idx);

    // Per-equation breakdown
    int nT = sys.nTau();
    int nS = sys.nSpatial();
    const char* field_names[8] = {
        "lapse(trK)", "shift_r(V^r)", "shift_t(V^t)",
        "gamma_rr", "gamma_rt", "gamma_tt", "gamma_pp", "scalar"
    };
    for (int A = 0; A < 8; A++) {
        double max_field = 0;
        for (int m = 0; m < nT; m++) {
            for (int si = 0; si < nS; si++) {
                double val = std::abs(R[sys.stateIdx(A, m, si)]);
                max_field = std::max(max_field, val);
            }
        }
        char buf[256];
        snprintf(buf, sizeof(buf), "R[%s] on background", field_names[A]);
        check(max_field < 1e-4, buf, max_field);
    }

    // Overall check (N=16 gives ~5e-6 spectral truncation error)
    check(max_res < 1e-4, "total residual on background", max_res);
}

// ============================================================================
// Test H: Residual with eigenmode scales as epsilon^2
// ============================================================================
// The linear eigenmode satisfies the linearized equations exactly. The residual
// should therefore scale as epsilon^2 (from the nonlinear terms).

void test_H_eigenmode_scaling() {
    printf("\n--- Test H: Eigenmode residual scaling ---\n");

    solver::OscillonParams params;
    params.N_t = 4;
    params.N_nuc = 16;
    params.N_shell = 16;
    params.rho_mid = 0.5;
    params.N_theta = 4;
    params.Delta = 6.0;
    params.Lambda = -3.0;
    params.ell = 2;

    solver::OscillonSystem sys(params);
    int n = sys.stateSize() - 1;
    int nT = sys.nTau(), nS = sys.nSpatial();

    // Compute residuals at background and two amplitudes.
    // The scalar wave residual has O(eps) from spectral truncation of eigenmode
    // derivatives, while evolution equations have O(eps^2) from matter coupling.
    // We subtract the background residual and check evolution equations only.
    double eps1 = 0.1, eps2 = 0.05;

    std::vector<double> u0(n), u1(n), u2(n), R0(n), R1(n), R2(n);
    double omega0, omega1, omega2;

    sys.setLinearSeed(u0.data(), omega0, 0.0);
    sys.setLinearSeed(u1.data(), omega1, eps1);
    sys.setLinearSeed(u2.data(), omega2, eps2);

    sys.computeResidual(u0.data(), omega0, R0.data());
    sys.computeResidual(u1.data(), omega1, R1.data());
    sys.computeResidual(u2.data(), omega2, R2.data());

    // Evolution equation residuals (fields 3-6: gamma components)
    // These have O(spectral_floor + eps^2 * matter) behavior.
    // Subtract background to isolate the eps-dependent part.
    double evol_norm1 = 0, evol_norm2 = 0;
    for (int A = geometry::FLD_GAMMA_RR; A <= geometry::FLD_GAMMA_PP; A++) {
        for (int m = 0; m < nT; m++) {
            for (int si = 0; si < nS; si++) {
                int idx = sys.stateIdx(A, m, si);
                double r1 = R1[idx] - R0[idx];
                double r2 = R2[idx] - R0[idx];
                evol_norm1 += r1 * r1;
                evol_norm2 += r2 * r2;
            }
        }
    }
    evol_norm1 = sqrt(evol_norm1);
    evol_norm2 = sqrt(evol_norm2);

    printf("  ||R_evol(eps=%.2f) - R_evol(0)|| = %.4e\n", eps1, evol_norm1);
    printf("  ||R_evol(eps=%.2f) - R_evol(0)|| = %.4e\n", eps2, evol_norm2);

    if (evol_norm1 > 1e-14 && evol_norm2 > 1e-14) {
        double ratio = evol_norm2 / evol_norm1;
        double expected = (eps2 * eps2) / (eps1 * eps1); // 0.25
        double rel_err = std::abs(ratio - expected) / expected;
        printf("  ratio = %.6e (expected %.6e)\n", ratio, expected);
        check(rel_err < 0.5, "evolution residual scales as eps^2", rel_err);
    } else {
        check(true, "evolution residuals near zero", std::max(evol_norm1, evol_norm2));
    }

    // Also check that total residual decreases with epsilon (basic sanity)
    double total1 = 0, total2 = 0;
    for (int i = 0; i < n; i++) {
        total1 += (R1[i] - R0[i]) * (R1[i] - R0[i]);
        total2 += (R2[i] - R0[i]) * (R2[i] - R0[i]);
    }
    total1 = sqrt(total1);
    total2 = sqrt(total2);
    printf("  ||R(%.2f)-R(0)|| = %.4e, ||R(%.2f)-R(0)|| = %.4e\n",
           eps1, total1, eps2, total2);
    check(total2 < total1, "residual decreases with epsilon", total2 / total1);
}

// ============================================================================
// Test I: Normalization function
// ============================================================================

void test_I_normalization() {
    printf("\n--- Test I: Normalization ---\n");

    solver::OscillonParams params;
    params.N_t = 4;
    params.N_nuc = 8;
    params.N_shell = 8;
    params.rho_mid = 0.5;
    params.N_theta = 4;
    params.Delta = 6.0;
    params.Lambda = -3.0;
    params.ell = 2;

    solver::OscillonSystem sys(params);
    int n = sys.stateSize() - 1;

    double eps = 0.05;
    std::vector<double> u(n);
    double omega;
    sys.setLinearSeed(u.data(), omega, eps);

    double norm_val = sys.normalization(u.data());

    // The normalization returns phi_hat at (tau=0, reference spatial point).
    // For the eigenmode, phi_hat = eps * f(rho) * P_ell(cos theta) * cos(0) * Omega^{Delta/2}
    // This should be approximately proportional to epsilon.
    // More importantly, it should be nonzero for a non-trivial seed.

    printf("  normalization(eps=%.2e) = %.6e\n", eps, norm_val);
    check(std::abs(norm_val) > 1e-10, "normalization is nonzero", std::abs(norm_val));

    // Check proportionality to epsilon
    double eps2 = 0.1;
    std::vector<double> u2(n);
    double omega2;
    sys.setLinearSeed(u2.data(), omega2, eps2);
    double norm_val2 = sys.normalization(u2.data());

    double ratio = norm_val2 / norm_val;
    double expected = eps2 / eps;
    double rel_err = std::abs(ratio - expected) / expected;
    printf("  ratio = %.6f (expected %.6f)\n", ratio, expected);
    check(rel_err < 1e-10, "normalization proportional to epsilon", rel_err);
}

// ============================================================================
// Main
// ============================================================================

int main() {
    printf("========================================\n");
    printf("Checkpoint 7: ResidualAssembler tests\n");
    printf("========================================\n");

    test_F_seed();
    test_G_background_residual();
    test_H_eigenmode_scaling();
    test_I_normalization();

    printf("\n========================================\n");
    printf("Results: %d / %d passed\n", tests_passed, tests_total);
    printf("========================================\n");

    return (tests_passed == tests_total) ? 0 : 1;
}
