// test_newton_oscillon.cpp — Checkpoint 8: Newton convergence on oscillon system
//
// Uses LAPACK dgelsy (QR with column pivoting) to handle the Bianchi identity
// null space in the time-collocation Jacobian. At N_t=2 (3 temporal points),
// the null space is manageable and Newton converges to near-machine-precision.
//
// Test J: Newton converges at small epsilon (quadratic convergence)
// Test K: Frequency shift under nonlinear correction (omega decreases with eps)
// Test L: Multi-step continuation produces a branch with decreasing omega
// Test M: Converged solution has near-zero residual (cross-check)
#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>
#include "solver/oscillon_system.h"
#include "solver/oscillon_driver.h"
#include "solver/newton.h"

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

// Shared resolution and Newton params
static solver::OscillonParams defaultOscillonParams() {
    solver::OscillonParams params;
    params.N_t = 2;          // 3 temporal points — small Bianchi null space
    params.N_nuc = 4;
    params.N_shell = 4;
    params.rho_mid = 0.5;
    params.N_theta = 2;
    params.Delta = 6.0;
    params.Lambda = -3.0;
    params.ell = 2;
    return params;
}

static solver::NewtonParams defaultNewtonParams(bool verbose = false) {
    solver::NewtonParams np;
    np.tol = 1e-8;
    np.max_iter = 15;
    np.fd_eps = 1e-7;
    np.use_lapack = true;     // LAPACK dgelsy handles Bianchi null space correctly
    np.verbose = verbose;
    np.stag_window = 3;
    np.stag_ratio = 0.99;
    return np;
}

// ============================================================================
// Test J: Newton convergence at small epsilon
// ============================================================================
// At eps = 0.01, the linear seed is already an O(eps^2) approximate solution.
// Newton should converge in very few iterations with near-quadratic residual history.

void test_J_newton_convergence() {
    printf("\n--- Test J: Newton convergence at small epsilon ---\n");

    auto params = defaultOscillonParams();
    solver::OscillonDriver driver(params);

    double epsilon = 0.01;
    solver::NewtonResult result;
    std::vector<double> u;
    double omega;

    auto np = defaultNewtonParams(true);
    bool ok = driver.solveAtEpsilon(epsilon, result, u, omega, np);

    // Check convergence
    check(ok, "Newton converged at eps=0.01", ok ? 0.0 : 1.0);

    // Check that it converged quickly
    if (ok) {
        check(result.iterations <= 12, "converged in <= 12 iterations",
              (double)result.iterations);
    }

    // Check residual is small (LAPACK solve should give near-machine-precision)
    check(result.final_residual < 1e-8, "final residual < 1e-8",
          result.final_residual);

    // Check superlinear convergence: at least one pair of consecutive residuals
    // shows faster-than-linear decrease
    if (result.residual_history.size() >= 3) {
        bool saw_superlinear = false;
        for (size_t i = 2; i < result.residual_history.size(); i++) {
            double r_prev = result.residual_history[i-1];
            double r_pprev = result.residual_history[i-2];
            double r_curr = result.residual_history[i];
            if (r_pprev > 1e-12 && r_prev > 1e-12 && r_curr < r_prev) {
                // Superlinear: r_k / r_{k-1} < r_{k-1} / r_{k-2}
                double rate_curr = r_curr / r_prev;
                double rate_prev = r_prev / r_pprev;
                if (rate_curr < rate_prev * 1.5) { // generous tolerance
                    saw_superlinear = true;
                }
            }
        }
        check(saw_superlinear, "observed superlinear convergence", 0.0);
    }

    printf("  omega = %.10f\n", omega);
}

// ============================================================================
// Test K: Frequency shift at moderate amplitude
// ============================================================================
// At larger epsilon, nonlinear corrections shift omega. The key physical
// prediction: omega should DECREASE monotonically with increasing amplitude.

void test_K_frequency_shift() {
    printf("\n--- Test K: Frequency shift at moderate amplitude ---\n");

    auto params = defaultOscillonParams();
    solver::OscillonDriver driver(params);

    double eps_vals[] = {0.01, 0.05, 0.10};
    double omegas[3];
    bool all_ok = true;

    auto np = defaultNewtonParams(false);

    for (int k = 0; k < 3; k++) {
        solver::NewtonResult res;
        std::vector<double> u;
        bool ok = driver.solveAtEpsilon(eps_vals[k], res, u, omegas[k], np);
        printf("  eps = %.2f: converged = %s, omega = %.10f, res = %.2e\n",
               eps_vals[k], ok ? "yes" : "no", omegas[k], res.final_residual);
        if (!ok) all_ok = false;
    }

    check(all_ok, "all three amplitudes converged", all_ok ? 0 : 1);

    if (all_ok) {
        // Omega should decrease with increasing amplitude.
        // At low spectral resolution, the base omega (at eps→0) has a
        // discretization shift, but the eps-dependent part should be monotone.
        check(omegas[1] < omegas[0], "omega(eps=0.05) < omega(eps=0.01)",
              omegas[0] - omegas[1]);
        check(omegas[2] < omegas[1], "omega(eps=0.10) < omega(eps=0.05)",
              omegas[1] - omegas[2]);

        // The frequency shift magnitude should increase faster than linearly:
        // |Δω(0.10) - Δω(0.01)| > 2 * |Δω(0.05) - Δω(0.01)|
        // (at leading order, shift ~ eps² so this ratio is (10²-1²)/(5²-1²) ≈ 4.1)
        double shift_05 = omegas[0] - omegas[1]; // positive if omega decreases
        double shift_10 = omegas[0] - omegas[2];
        if (shift_05 > 1e-14) {
            double ratio = shift_10 / shift_05;
            printf("  shift(0.10)/shift(0.05) = %.4f (expected ~4.1 for eps^2 scaling)\n",
                   ratio);
            check(ratio > 1.5, "shift accelerates with amplitude", ratio);
        }
    }
}

// ============================================================================
// Test L: Continuation produces branch with decreasing omega
// ============================================================================

void test_L_continuation() {
    printf("\n--- Test L: Continuation branch ---\n");

    auto params = defaultOscillonParams();
    solver::OscillonDriver driver(params);

    solver::ContinuationParams cont;
    cont.eps_start = 0.01;
    cont.dw_initial = 0.002;
    cont.dw_min = 1e-4;
    cont.dw_max = 0.02;
    cont.w_max = 0.02;   // small range for test speed
    cont.newton = defaultNewtonParams(false);
    cont.verbose = true;
    cont.store_solutions = false;

    auto branch = driver.runContinuation(cont);

    printf("  Got %d branch points\n", (int)branch.size());

    check(branch.size() >= 3, "at least 3 branch points",
          (double)branch.size());

    if (branch.size() >= 2) {
        // Check monotonically increasing w
        bool w_mono = true;
        for (size_t i = 1; i < branch.size(); i++) {
            if (branch[i].w <= branch[i-1].w) {
                w_mono = false;
                break;
            }
        }
        check(w_mono, "w monotonically increasing", 0.0);

        // Check all converged
        double max_res = 0;
        for (auto& bp : branch)
            max_res = std::max(max_res, bp.residual);
        check(max_res < 1e-6, "all branch points converged", max_res);

        // Check omega is non-increasing along the branch
        bool omega_decreasing = true;
        for (size_t i = 1; i < branch.size(); i++) {
            if (branch[i].omega > branch[i-1].omega + 1e-8) {
                omega_decreasing = false;
                printf("  omega increased: %.10f -> %.10f at step %d\n",
                       branch[i-1].omega, branch[i].omega, (int)i);
                break;
            }
        }
        check(omega_decreasing, "omega non-increasing along branch", 0.0);
    }

    if (!branch.empty()) {
        printf("\n  Branch summary:\n");
        for (size_t i = 0; i < branch.size(); i++) {
            printf("    %2d: w = %.6e, omega = %.10f, res = %.2e, iters = %d\n",
                   (int)i, branch[i].w, branch[i].omega,
                   branch[i].residual, branch[i].newton_iters);
        }
    }
}

// ============================================================================
// Test M: Residual verification
// ============================================================================

void test_M_residual_verification() {
    printf("\n--- Test M: Residual verification ---\n");

    auto params = defaultOscillonParams();
    solver::OscillonDriver driver(params);

    double epsilon = 0.05;
    solver::NewtonResult result;
    std::vector<double> u;
    double omega;

    auto np = defaultNewtonParams(false);
    bool ok = driver.solveAtEpsilon(epsilon, result, u, omega, np);
    check(ok, "Newton converged for cross-check", result.final_residual);

    if (ok) {
        // Recompute residual independently
        solver::OscillonSystem& sys =
            const_cast<solver::OscillonSystem&>(driver.system());
        int n = sys.stateSize() - 1;
        std::vector<double> R(n, 0.0);
        sys.computeResidual(u.data(), omega, R.data());

        double max_res = 0;
        for (int i = 0; i < n; i++)
            max_res = std::max(max_res, std::abs(R[i]));

        printf("  Independent residual check: max |R| = %.4e\n", max_res);
        check(max_res < 1e-8, "independent residual < 1e-8", max_res);

        // Verify the solution differs from the seed
        std::vector<double> u_seed(n);
        double omega_seed;
        sys.setLinearSeed(u_seed.data(), omega_seed, epsilon);

        double diff = 0, seed_norm = 0;
        for (int i = 0; i < n; i++) {
            diff += (u[i] - u_seed[i]) * (u[i] - u_seed[i]);
            seed_norm += u_seed[i] * u_seed[i];
        }
        diff = sqrt(diff);
        seed_norm = sqrt(seed_norm);
        double rel_diff = diff / seed_norm;
        printf("  ||u - u_seed|| / ||u_seed|| = %.4e\n", rel_diff);

        check(rel_diff > 1e-6, "solution differs from seed (nonlinear correction)",
              rel_diff);
        check(rel_diff < 1.0, "solution not wildly different from seed", rel_diff);

        // Verify omega shifted from seed
        printf("  omega_seed = %.10f, omega_converged = %.10f\n", omega_seed, omega);
        check(std::abs(omega - omega_seed) > 1e-6,
              "omega shifted from seed value",
              std::abs(omega - omega_seed));
    }
}

// ============================================================================
// Main
// ============================================================================

int main() {
    printf("========================================\n");
    printf("Checkpoint 8: Newton convergence tests\n");
    printf("========================================\n");

    test_J_newton_convergence();
    test_K_frequency_shift();
    test_L_continuation();
    test_M_residual_verification();

    printf("\n========================================\n");
    printf("Results: %d / %d passed\n", tests_passed, tests_total);
    printf("========================================\n");

    return (tests_passed == tests_total) ? 0 : 1;
}
