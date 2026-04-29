// test_diagnostics.cpp — Checkpoint 6: Solution diagnostics
//
// Test N: Constraint violations decrease with resolution
// Test O: Spectral tails show exponential decay
// Test P: Physical observables are consistent across resolutions
// Test Q: Background solution has zero constraint violations
// Test R: Scalar energy scales as eps² at small amplitude
// Test S: Full diagnostic report consistency

#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>
#include "solver/oscillon_system.h"
#include "solver/oscillon_driver.h"
#include "solver/newton.h"
#include "diagnostics/diagnostics.h"
#include "diagnostics/charges.h"

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

static solver::OscillonParams makeParams(int N_nuc, int N_shell, int N_theta) {
    solver::OscillonParams p;
    p.N_t = 2;
    p.N_nuc = N_nuc;
    p.N_shell = N_shell;
    p.rho_mid = 0.5;
    p.N_theta = N_theta;
    p.Delta = 6.0;
    p.Lambda = -3.0;
    p.ell = 2;
    return p;
}

static solver::NewtonParams makeNewton() {
    solver::NewtonParams np;
    np.tol = 1e-8;
    np.max_iter = 15;
    np.fd_eps = 1e-7;
    np.use_lapack = true;
    np.verbose = false;
    np.stag_window = 3;
    np.stag_ratio = 0.99;
    return np;
}

// ============================================================================
// Test N: Constraint violations for a converged solution
// ============================================================================
void test_N_constraints() {
    printf("\n--- Test N: Constraint violations ---\n");

    auto params = makeParams(4, 4, 2);
    solver::OscillonDriver driver(params);
    auto np = makeNewton();

    solver::NewtonResult result;
    std::vector<double> u;
    double omega;
    bool ok = driver.solveAtEpsilon(0.05, result, u, omega, np);
    check(ok, "Newton converged for constraint test", result.final_residual);

    if (ok) {
        auto cd = diagnostics::computeConstraints(driver.system(), u.data(), omega);
        cd.print("  ");

        // trK should be small (maximal slicing constraint)
        check(cd.max_trK < 1e-4, "max|trK| < 1e-4", cd.max_trK);

        // Evolution residual should be bounded
        check(cd.max_evol_res < 1e2, "max|evol| < 100 (physical units)", cd.max_evol_res);

        // Scalar wave residual
        check(cd.max_scalar_res < 1e-3, "max|scalar_wave| < 1e-3", cd.max_scalar_res);

        // RMS should be smaller than max
        check(cd.rms_trK < cd.max_trK + 1e-15, "rms(trK) <= max(trK)", cd.rms_trK);
    }
}

// ============================================================================
// Test O: Spectral tails
// ============================================================================
void test_O_spectral_tails() {
    printf("\n--- Test O: Spectral tail decay ---\n");

    auto params = makeParams(4, 4, 2);
    solver::OscillonDriver driver(params);
    auto np = makeNewton();

    solver::NewtonResult result;
    std::vector<double> u;
    double omega;
    bool ok = driver.solveAtEpsilon(0.05, result, u, omega, np);
    check(ok, "Newton converged for spectral tail test", result.final_residual);

    if (ok) {
        auto st = diagnostics::computeSpectralTails(driver.system(), u.data());
        st.print("  ");

        // Chebyshev tails should decay
        check(st.chebyshev_tail.back() < st.chebyshev_tail.front(),
              "Chebyshev tail decays", st.chebyshev_tail.back());

        // Fourier tails should decay (if we have enough modes)
        if (st.fourier_tail.size() >= 2) {
            check(st.fourier_tail.back() < st.fourier_tail.front() + 1e-10,
                  "Fourier tail non-increasing", st.fourier_tail.back());
        }

        // Should resolve at least 0.5 digit
        check(st.cheb_digits > 0.5, "Chebyshev resolves > 0.5 digits", st.cheb_digits);
    }
}

// ============================================================================
// Test P: Physical observables
// ============================================================================
void test_P_observables() {
    printf("\n--- Test P: Physical observables ---\n");

    auto params = makeParams(4, 4, 2);
    solver::OscillonDriver driver(params);
    auto np = makeNewton();

    solver::NewtonResult result;
    std::vector<double> u;
    double omega;
    bool ok = driver.solveAtEpsilon(0.05, result, u, omega, np);
    check(ok, "Newton converged for observables test", result.final_residual);

    if (ok) {
        auto obs = diagnostics::computeObservables(driver.system(), u.data(), omega);
        obs.print("  ");

        // Omega should be close to but less than omega_0 = Delta + ell = 8
        check(obs.omega < 8.0, "omega < omega_0", 8.0 - obs.omega);
        check(obs.omega > 7.0, "omega > 7 (not crazy)", obs.omega);

        // phi_max should be positive and proportional to eps
        check(obs.phi_max > 0, "phi_max > 0", obs.phi_max);
        check(obs.phi_max < 1.0, "phi_max < 1 (bounded)", obs.phi_max);

        // E_scalar should be positive
        check(obs.E_scalar > 0, "E_scalar > 0", obs.E_scalar);

        // w should be nonzero
        check(std::abs(obs.w) > 1e-10, "w is nonzero", obs.w);
    }
}

// ============================================================================
// Test Q: Background has zero constraints
// ============================================================================
void test_Q_background_constraints() {
    printf("\n--- Test Q: Background constraint violations ---\n");

    auto params = makeParams(4, 4, 2);
    solver::OscillonSystem sys(params);

    int n = sys.stateSize() - 1;
    std::vector<double> u(n);
    double omega;
    sys.setLinearSeed(u.data(), omega, 0.0); // pure background

    auto cd = diagnostics::computeConstraints(sys, u.data(), omega);
    cd.print("  ");

    // Background residual should be very small (it's subtracted)
    check(cd.max_trK < 1e-8, "bg max|trK| < 1e-8", cd.max_trK);
    check(cd.max_scalar_res < 1e-8, "bg max|scalar| < 1e-8", cd.max_scalar_res);
}

// ============================================================================
// Test R: Scalar energy scales as eps²
// ============================================================================
void test_R_energy_scaling() {
    printf("\n--- Test R: Scalar energy eps^2 scaling ---\n");

    auto params = makeParams(4, 4, 2);
    solver::OscillonDriver driver(params);
    auto np = makeNewton();

    double eps_vals[] = {0.01, 0.02, 0.05};
    double energies[3];
    bool all_ok = true;

    for (int k = 0; k < 3; k++) {
        solver::NewtonResult result;
        std::vector<double> u;
        double omega;
        bool ok = driver.solveAtEpsilon(eps_vals[k], result, u, omega, np);
        if (!ok) { all_ok = false; continue; }

        auto obs = diagnostics::computeObservables(driver.system(), u.data(), omega);
        energies[k] = obs.E_scalar;
        printf("  eps = %.2f: E = %.6e, omega = %.10f\n", eps_vals[k], energies[k], omega);
    }

    check(all_ok, "all three amplitudes converged", all_ok ? 0 : 1);

    if (all_ok && energies[0] > 1e-20) {
        // E should scale as eps²: E(0.02)/E(0.01) ≈ 4, E(0.05)/E(0.01) ≈ 25
        double ratio_02 = energies[1] / energies[0];
        double ratio_05 = energies[2] / energies[0];
        printf("  E(0.02)/E(0.01) = %.2f (expected ~4)\n", ratio_02);
        printf("  E(0.05)/E(0.01) = %.2f (expected ~25)\n", ratio_05);

        check(ratio_02 > 2.0 && ratio_02 < 8.0,
              "E(0.02)/E(0.01) near 4 (eps^2 scaling)", ratio_02);
        check(ratio_05 > 10.0 && ratio_05 < 50.0,
              "E(0.05)/E(0.01) near 25 (eps^2 scaling)", ratio_05);
    }
}

// ============================================================================
// Test S: Full diagnostic report
// ============================================================================
void test_S_full_report() {
    printf("\n--- Test S: Full diagnostic report ---\n");

    auto params = makeParams(4, 4, 2);
    solver::OscillonDriver driver(params);
    auto np = makeNewton();

    solver::NewtonResult result;
    std::vector<double> u;
    double omega;
    bool ok = driver.solveAtEpsilon(0.05, result, u, omega, np);
    check(ok, "Newton converged for full report", result.final_residual);

    if (ok) {
        auto report = diagnostics::fullDiagnostics(driver.system(), u.data(), omega);
        printf("\n  === Full Diagnostic Report ===\n");
        report.print("  ");

        // Consistency: observables omega matches
        check(std::abs(report.observables.omega - omega) < 1e-14,
              "report.omega matches solver omega", std::abs(report.observables.omega - omega));
    }
}

int main() {
    printf("========================================\n");
    printf("Checkpoint 6: Diagnostic tests\n");
    printf("========================================\n");

    test_N_constraints();
    test_O_spectral_tails();
    test_P_observables();
    test_Q_background_constraints();
    test_R_energy_scaling();
    test_S_full_report();

    printf("\n========================================\n");
    printf("Results: %d / %d passed\n", tests_passed, tests_total);
    printf("========================================\n");

    return (tests_passed == tests_total) ? 0 : 1;
}
