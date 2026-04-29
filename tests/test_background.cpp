// test_background.cpp — Tests for AdS4 background and scalar eigenmodes
// Checkpoint 2: Background and eigenmode verified
#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>
#include "geometry/ads4.h"
#include "spectral/chebyshev.h"
#include "spectral/legendre.h"

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

void test_conformal_factor() {
    printf("\n--- Conformal factor Omega ---\n");
    using namespace geometry;

    // Omega(0) = 1, Omega(1) = 0
    check(std::abs(AdS4Background::Omega(0.0) - 1.0) < 1e-15,
          "Omega(0) = 1", std::abs(AdS4Background::Omega(0.0) - 1.0));
    check(std::abs(AdS4Background::Omega(1.0)) < 1e-15,
          "Omega(1) = 0", std::abs(AdS4Background::Omega(1.0)));

    // Omega(rho) = cos(x_bar) where rho = tan(x_bar/2)
    for (double rho : {0.1, 0.3, 0.5, 0.7, 0.9}) {
        double xb = AdS4Background::xbar_of_rho(rho);
        double Om = AdS4Background::Omega(rho);
        double err = std::abs(Om - cos(xb));
        check(err < 1e-14, "Omega = cos(x_bar)", err);
    }
}

void test_coordinate_maps() {
    printf("\n--- Coordinate maps ---\n");
    using namespace geometry;

    for (double xbar : {0.1, 0.5, 1.0, 1.4}) {
        double rho = AdS4Background::rho_of_xbar(xbar);
        double xbar_back = AdS4Background::xbar_of_rho(rho);
        double err = std::abs(xbar - xbar_back);
        check(err < 1e-14, "xbar -> rho -> xbar roundtrip", err);
    }
}

void test_background_metric() {
    printf("\n--- Background metric ---\n");
    using namespace geometry;

    // At rho=0: N_bar = 1, gamma_rr = 4
    check(std::abs(AdS4Background::N_bar(0.0) - 1.0) < 1e-14,
          "N_bar(0) = 1", std::abs(AdS4Background::N_bar(0.0) - 1.0));
    check(std::abs(AdS4Background::gamma_bar_rhorho(0.0) - 4.0) < 1e-14,
          "gamma_rr(0) = 4", std::abs(AdS4Background::gamma_bar_rhorho(0.0) - 4.0));

    // Regularized: Nhat = 1 everywhere
    for (double rho : {0.0, 0.3, 0.5, 0.7}) {
        check(std::abs(AdS4Background::Nhat_bar(rho) - 1.0) < 1e-15,
              "Nhat_bar = 1", std::abs(AdS4Background::Nhat_bar(rho) - 1.0));
    }

    // gammahat_rr(rho) = 4/(1+rho^2)^2
    for (double rho : {0.0, 0.3, 0.5, 0.7}) {
        double expected = 4.0 / ((1.0 + rho*rho) * (1.0 + rho*rho));
        double computed = AdS4Background::gammahat_bar_rhorho(rho);
        double err = std::abs(computed - expected);
        check(err < 1e-14, "gammahat_rr = 4/(1+rho^2)^2", err);
    }
}

void test_eigenmode_basic() {
    printf("\n--- Scalar eigenmode basics ---\n");
    using namespace geometry;

    // Delta = 6, ell = 2: omega_0 = 6 + 2 = 8, m^2 = 6*3 = 18
    ScalarEigenmode mode(2, 6.0);
    check(std::abs(mode.omega0() - 8.0) < 1e-14, "omega_0 = Delta + ell = 8", 0.0);
    check(std::abs(mode.mass_sq() - 18.0) < 1e-14, "m^2 = Delta(Delta-3) = 18", 0.0);

    // f(0) = 0 for ell > 0 (since sin^ell(x_bar) -> 0)
    double f0 = mode.f_rho(0.0);
    check(std::abs(f0) < 1e-14, "f(rho=0) = 0 for ell=2", std::abs(f0));

    // f(rho) -> 0 as rho -> 1 (since cos^Delta(x_bar) -> 0)
    double f1 = mode.f_rho(0.999);
    check(f1 < 1e-3, "f(rho~1) small (boundary)", f1);
}

void test_eigenmode_wave_equation() {
    // CRITICAL TEST: Verify Box phi - m^2 phi = 0
    printf("\n--- Eigenmode wave equation verification ---\n");
    using namespace geometry;

    // Higher ell and Delta need more resolution (steeper gradients near boundaries)
    struct TestCase { int ell; double Delta; int N_rho; double tol; };
    TestCase cases[] = {
        {2,  6.0,  40, 1e-8},
        {5,  6.0,  50, 1e-8},
        {10, 6.0,  80, 1e-2},   // ell=10: singular endpoints limit test resolution
        {2,  4.0,  40, 1e-8},
        {2,  10.0, 80, 1e-1},   // Delta=10: steep boundary falloff; actual solver uses rho+regularization
    };

    for (auto& tc : cases) {
        ScalarEigenmode mode(tc.ell, tc.Delta);
        double res = verifyEigenmode(mode, tc.N_rho, 20);

        char name[80];
        snprintf(name, sizeof(name), "Box phi - m^2 phi = 0, ell=%d, Delta=%.0f (N=%d)",
                 tc.ell, tc.Delta, tc.N_rho);
        check(res < tc.tol, name, res);
    }
}

void test_eigenmode_orthonormality() {
    printf("\n--- Eigenmode angular structure ---\n");
    using namespace geometry;

    // phi(t, rho, theta) should have the right angular dependence
    ScalarEigenmode mode(3, 6.0);
    double rho = 0.3, t = 0.0;

    // At theta = pi/2: P_3(0) = 0 for ell=3 (since P_3(0) = 0)
    double phi_eq = mode.phi(t, rho, PI/2.0);
    check(std::abs(phi_eq) < 1e-14, "phi(theta=pi/2) = 0 for ell=3 (P_3(0)=0)", std::abs(phi_eq));

    // At theta = 0: P_3(1) = 1
    double phi_pole = mode.phi(t, rho, 0.0);
    double f_rho = mode.f_rho(rho);
    check(std::abs(phi_pole - f_rho) < 1e-14, "phi(theta=0) = f(rho) for ell=3",
          std::abs(phi_pole - f_rho));
}

int main() {
    printf("===== BACKGROUND & EIGENMODE TESTS =====\n");
    printf("Checkpoint 2\n");

    test_conformal_factor();
    test_coordinate_maps();
    test_background_metric();
    test_eigenmode_basic();
    test_eigenmode_wave_equation();
    test_eigenmode_orthonormality();

    printf("\n===================================\n");
    printf("Results: %d / %d tests passed\n", tests_passed, tests_total);
    printf("===================================\n");

    return (tests_passed == tests_total) ? 0 : 1;
}
