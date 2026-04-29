// test_spectral.cpp — Tests for Chebyshev, Legendre, and Fourier modules
// Checkpoint 1: All spectral modules compile and pass tests
#include <cstdio>
#include <cmath>
#include <cassert>
#include <vector>
#include <algorithm>
#include "spectral/chebyshev.h"
#include "spectral/legendre.h"
#include "spectral/fourier.h"

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

// ===================== CHEBYSHEV TESTS =====================
void test_chebyshev_diff_sin3x() {
    // Test: differentiate sin(3x) on [-1,1]. Max error vs analytic.
    printf("\n--- Chebyshev: differentiate sin(3x) ---\n");

    // Test spectral convergence at multiple resolutions
    for (int N : {16, 20, 24}) {
        spectral::Chebyshev cheb(N, -1.0, 1.0);

        std::vector<double> f(N+1), df_num(N+1);
        for (int j = 0; j <= N; j++) {
            double x = cheb.grid(j);
            f[j] = sin(3.0 * x);
        }

        cheb.differentiate(f.data(), df_num.data());

        double max_err = 0.0;
        for (int j = 0; j <= N; j++) {
            double x = cheb.grid(j);
            double exact = 3.0 * cos(3.0 * x);
            max_err = std::max(max_err, std::abs(df_num[j] - exact));
        }

        char name[64];
        snprintf(name, sizeof(name), "d/dx sin(3x) with N=%d", N);
        printf("  max error = %.6e\n", max_err);
        if (N == 20) {
            check(max_err < 1e-10, name, max_err);
        }
    }
}

void test_chebyshev_diff_poly() {
    // Chebyshev should differentiate polynomials exactly
    printf("\n--- Chebyshev: differentiate x^5 ---\n");
    spectral::Chebyshev cheb(8, -1.0, 1.0);

    std::vector<double> f(9), df_num(9);
    for (int j = 0; j <= 8; j++) {
        double x = cheb.grid(j);
        f[j] = pow(x, 5);
    }

    cheb.differentiate(f.data(), df_num.data());

    double max_err = 0.0;
    for (int j = 0; j <= 8; j++) {
        double x = cheb.grid(j);
        double exact = 5.0 * pow(x, 4);
        max_err = std::max(max_err, std::abs(df_num[j] - exact));
    }

    check(max_err < 1e-12, "d/dx x^5 with N=8", max_err);
}

void test_chebyshev_second_deriv() {
    printf("\n--- Chebyshev: second derivative of sin(3x) ---\n");
    spectral::Chebyshev cheb(20, -1.0, 1.0);

    std::vector<double> f(21), d2f_num(21);
    for (int j = 0; j <= 20; j++) {
        double x = cheb.grid(j);
        f[j] = sin(3.0 * x);
    }

    cheb.differentiate2(f.data(), d2f_num.data());

    double max_err = 0.0;
    for (int j = 0; j <= 20; j++) {
        double x = cheb.grid(j);
        double exact = -9.0 * sin(3.0 * x);
        max_err = std::max(max_err, std::abs(d2f_num[j] - exact));
    }

    check(max_err < 1e-10, "d^2/dx^2 sin(3x) with N=20", max_err);
}

void test_chebyshev_transform() {
    printf("\n--- Chebyshev: forward/inverse transform ---\n");
    spectral::Chebyshev cheb(16, -1.0, 1.0);

    std::vector<double> f(17), coeffs(17), f_rec(17);
    for (int j = 0; j <= 16; j++) {
        double x = cheb.grid(j);
        f[j] = exp(-x*x);
    }

    cheb.forward(f.data(), coeffs.data());
    cheb.inverse(coeffs.data(), f_rec.data());

    double max_err = 0.0;
    for (int j = 0; j <= 16; j++) {
        max_err = std::max(max_err, std::abs(f[j] - f_rec[j]));
    }

    check(max_err < 1e-13, "forward/inverse roundtrip for exp(-x^2)", max_err);
}

void test_chebyshev_integration() {
    printf("\n--- Chebyshev: integration ---\n");
    spectral::Chebyshev cheb(20, -1.0, 1.0);

    // Integrate cos(x) from -1 to 1 = 2*sin(1)
    std::vector<double> f(21);
    for (int j = 0; j <= 20; j++) {
        f[j] = cos(cheb.grid(j));
    }

    double result = cheb.integrate(f.data());
    double exact = 2.0 * sin(1.0);
    double err = std::abs(result - exact);

    check(err < 1e-14, "integral of cos(x) on [-1,1]", err);
}

void test_chebyshev_mapped_domain() {
    printf("\n--- Chebyshev: mapped domain [0, 0.5] ---\n");
    spectral::Chebyshev cheb(16, 0.0, 0.5);

    std::vector<double> f(17), df_num(17);
    for (int j = 0; j <= 16; j++) {
        double x = cheb.grid(j);
        f[j] = sin(PI * x);
    }

    cheb.differentiate(f.data(), df_num.data());

    double max_err = 0.0;
    for (int j = 0; j <= 16; j++) {
        double x = cheb.grid(j);
        double exact = PI * cos(PI * x);
        max_err = std::max(max_err, std::abs(df_num[j] - exact));
    }

    check(max_err < 1e-11, "d/dx sin(pi*x) on [0, 0.5]", max_err);
}

void test_two_domain() {
    printf("\n--- Two-domain Chebyshev ---\n");
    spectral::TwoDomainChebyshev td(16, 16, 0.5);

    // Check that global grid covers [0, 1]
    const auto& grid = td.grid();
    double min_g = *std::min_element(grid.begin(), grid.end());
    double max_g = *std::max_element(grid.begin(), grid.end());

    check(std::abs(min_g) < 1e-14, "Two-domain: min grid ~ 0", std::abs(min_g));
    check(std::abs(max_g - 1.0) < 1e-14, "Two-domain: max grid ~ 1", std::abs(max_g - 1.0));

    // Differentiate a smooth function across both domains
    auto func = [](double rho) { return sin(PI * rho) * exp(-rho); };
    auto dfunc = [](double rho) { return (PI * cos(PI * rho) - sin(PI * rho)) * exp(-rho); };

    int Nn = td.nucleus().size(), Ns = td.shell().size();
    std::vector<double> fn(Nn), fs(Ns), dfn(Nn), dfs(Ns);

    for (int j = 0; j < Nn; j++) fn[j] = func(td.nucleus().grid(j));
    for (int j = 0; j < Ns; j++) fs[j] = func(td.shell().grid(j));

    td.differentiate(fn.data(), fs.data(), dfn.data(), dfs.data());

    double max_err = 0.0;
    for (int j = 1; j < Nn - 1; j++) {
        double exact = dfunc(td.nucleus().grid(j));
        max_err = std::max(max_err, std::abs(dfn[j] - exact));
    }
    for (int j = 1; j < Ns - 1; j++) {
        double exact = dfunc(td.shell().grid(j));
        max_err = std::max(max_err, std::abs(dfs[j] - exact));
    }

    check(max_err < 1e-10, "Two-domain diff of sin(pi*rho)*exp(-rho)", max_err);
}

// ===================== LEGENDRE TESTS =====================
void test_legendre_orthogonality() {
    // Verify orthogonality: int P_l(x) P_l'(x) dx = 2*delta/(2l+1)
    printf("\n--- Legendre: orthogonality ---\n");
    spectral::Legendre leg(16);

    double max_err = 0.0;
    for (int l1 = 0; l1 <= 10; l1++) {
        for (int l2 = l1; l2 <= 10; l2++) {
            // Compute integral using quadrature
            double integral = 0.0;
            for (int j = 0; j < leg.size(); j++) {
                double p1 = spectral::Legendre::P(l1, leg.mu(j));
                double p2 = spectral::Legendre::P(l2, leg.mu(j));
                integral += leg.weights()[j] * p1 * p2;
            }

            double exact = (l1 == l2) ? 2.0 / (2*l1 + 1) : 0.0;
            double err = std::abs(integral - exact);
            max_err = std::max(max_err, err);
        }
    }

    check(max_err < 1e-12, "Orthogonality for l,l' = 0,...,10 with N=16", max_err);
}

void test_legendre_transform() {
    printf("\n--- Legendre: forward/inverse transform ---\n");
    spectral::Legendre leg(12);

    // Test function: P_3(cos theta) + 0.5 * P_5(cos theta)
    std::vector<double> f(leg.size());
    for (int j = 0; j < leg.size(); j++) {
        f[j] = spectral::Legendre::P(3, leg.mu(j))
             + 0.5 * spectral::Legendre::P(5, leg.mu(j));
    }

    std::vector<double> coeffs(11, 0.0), f_rec(leg.size());
    leg.forward(f.data(), coeffs.data(), 10);
    leg.inverse(coeffs.data(), 10, f_rec.data());

    // Check coefficients
    double err_c3 = std::abs(coeffs[3] - 1.0);
    double err_c5 = std::abs(coeffs[5] - 0.5);
    double err_c0 = std::abs(coeffs[0]);
    double err_c1 = std::abs(coeffs[1]);

    check(err_c3 < 1e-12, "Legendre coeff a_3 = 1", err_c3);
    check(err_c5 < 1e-12, "Legendre coeff a_5 = 0.5", err_c5);
    check(err_c0 < 1e-12, "Legendre coeff a_0 = 0", err_c0);
    check(err_c1 < 1e-12, "Legendre coeff a_1 = 0", err_c1);

    // Roundtrip
    double max_err = 0.0;
    for (int j = 0; j < leg.size(); j++) {
        max_err = std::max(max_err, std::abs(f[j] - f_rec[j]));
    }
    check(max_err < 1e-12, "Legendre forward/inverse roundtrip", max_err);
}

void test_legendre_product() {
    // Verify P_3 * P_3 decomposition against known coefficients
    printf("\n--- Legendre: product P_3 * P_3 ---\n");

    auto coeffs = spectral::Legendre::productCoeffs(3, 3);
    // P_3^2 = sum_L (2L+1) (3 3 L; 0 0 0)^2 P_L for L = 0, 2, 4, 6

    // Known values from Wigner 3j:
    // (3 3 0; 0 0 0) = (-1)^3 * 3! / (3! 3! 3!) * ... let me just verify numerically
    // We verify by checking P_3(x)^2 at several points against the decomposition

    spectral::Legendre leg(12);
    double max_err = 0.0;
    for (int j = 0; j < leg.size(); j++) {
        double x = leg.mu(j);
        double p3sq = spectral::Legendre::P(3, x) * spectral::Legendre::P(3, x);

        double recon = 0.0;
        for (int L = 0; L < (int)coeffs.size(); L++) {
            if (std::abs(coeffs[L]) > 1e-15) {
                recon += coeffs[L] * spectral::Legendre::P(L, x);
            }
        }

        max_err = std::max(max_err, std::abs(p3sq - recon));
    }

    check(max_err < 1e-12, "P_3^2 = sum c_L P_L verified pointwise", max_err);

    // Also check P_3 * P_3 has only even L terms
    bool only_even = true;
    for (int L = 1; L < (int)coeffs.size(); L += 2) {
        if (std::abs(coeffs[L]) > 1e-14) only_even = false;
    }
    check(only_even, "P_3^2 has only even L", 0.0);

    // Print coefficients
    printf("  Product coefficients P_3 * P_3:\n");
    for (int L = 0; L < (int)coeffs.size(); L++) {
        if (std::abs(coeffs[L]) > 1e-14) {
            printf("    c_%d = %.15f\n", L, coeffs[L]);
        }
    }
}

void test_legendre_product_general() {
    // Verify P_ell * P_ell for ell = 2, 5
    printf("\n--- Legendre: product P_l * P_l for l=2,5 ---\n");

    for (int ell : {2, 5}) {
        auto coeffs = spectral::Legendre::productCoeffs(ell, ell);

        spectral::Legendre leg(2*ell + 4);
        double max_err = 0.0;
        for (int j = 0; j < leg.size(); j++) {
            double x = leg.mu(j);
            double plsq = spectral::Legendre::P(ell, x) * spectral::Legendre::P(ell, x);

            double recon = 0.0;
            for (int L = 0; L < (int)coeffs.size(); L++) {
                recon += coeffs[L] * spectral::Legendre::P(L, x);
            }

            max_err = std::max(max_err, std::abs(plsq - recon));
        }

        char name[64];
        snprintf(name, sizeof(name), "P_%d^2 product decomposition", ell);
        check(max_err < 1e-12, name, max_err);
    }
}

void test_wigner3j_known_values() {
    printf("\n--- Wigner 3j: known values ---\n");

    // (1 1 0; 0 0 0) = (-1)^1 * 1! / (1! 0! 0!) * sqrt(0! 0! 2! / 3!) = -1/sqrt(3)
    double w = spectral::Legendre::wigner3j_000(1, 1, 0);
    double exact = -1.0 / sqrt(3.0);
    check(std::abs(w - exact) < 1e-14, "(1 1 0; 0 0 0) = -1/sqrt(3)", std::abs(w - exact));

    // (1 1 2; 0 0 0) = (-1)^2 * 2! / (1! 1! 0!) * sqrt(2! 2! 0! / 5!) = 2/sqrt(5) * sqrt(4/120) = ...
    // Actually use the formula: (-1)^g g! / [(g-l1)!(g-l2)!(g-L)!] * sqrt[(2g-2l1)!(2g-2l2)!(2g-2L)!/(2g+1)!]
    // g = (1+1+2)/2 = 2
    // = (-1)^2 * 2! / [(2-1)!(2-1)!(2-2)!] * sqrt[(4-2)!(4-2)!(4-4)! / 5!]
    // = 1 * 2 / [1*1*1] * sqrt[2!*2!*0! / 5!] = 2 * sqrt(4/120) = 2/sqrt(30)
    w = spectral::Legendre::wigner3j_000(1, 1, 2);
    exact = 2.0 / sqrt(30.0);
    check(std::abs(w - exact) < 1e-14, "(1 1 2; 0 0 0) = 2/sqrt(30)", std::abs(w - exact));

    // (2 2 0; 0 0 0): g=2, = (-1)^2 * 2! / (0! 0! 2!) * sqrt(2!*2!*4!/5!) = 1 * 1 * sqrt(4*24/120) = sqrt(96/120)
    // Hmm let me recompute: g=2, l1=l2=2, L=0
    // = (-1)^2 * 2! / [(2-2)!(2-2)!(2-0)!] * sqrt[(4-4)!(4-4)!(4-0)! / 5!]
    // = 2 / [1*1*2] * sqrt[1*1*24/120] = 1 * sqrt(1/5) = 1/sqrt(5)
    w = spectral::Legendre::wigner3j_000(2, 2, 0);
    exact = 1.0 / sqrt(5.0);
    check(std::abs(w - exact) < 1e-14, "(2 2 0; 0 0 0) = 1/sqrt(5)", std::abs(w - exact));

    // Odd sum should give zero
    w = spectral::Legendre::wigner3j_000(1, 2, 2);
    check(std::abs(w) < 1e-14, "(1 2 2; 0 0 0) = 0 (odd sum)", std::abs(w));
}

// ===================== FOURIER TESTS =====================
void test_fourier_cos_transform() {
    printf("\n--- Fourier: cosine transform ---\n");
    spectral::Fourier four(8);

    // f(tau) = cos(tau) + 0.3*cos(3*tau)
    auto tau = four.collocTau();
    std::vector<double> f(9), coeffs(9);
    for (int j = 0; j <= 8; j++) {
        f[j] = cos(tau[j]) + 0.3 * cos(3 * tau[j]);
    }

    four.forwardCos(f.data(), coeffs.data());

    double err_0 = std::abs(coeffs[0]);
    double err_1 = std::abs(coeffs[1] - 1.0);
    double err_2 = std::abs(coeffs[2]);
    double err_3 = std::abs(coeffs[3] - 0.3);

    check(err_0 < 1e-14, "Fourier a_0 = 0", err_0);
    check(err_1 < 1e-14, "Fourier a_1 = 1", err_1);
    check(err_2 < 1e-14, "Fourier a_2 = 0", err_2);
    check(err_3 < 1e-14, "Fourier a_3 = 0.3", err_3);

    // Roundtrip
    std::vector<double> f_rec(9);
    four.inverseCos(coeffs.data(), f_rec.data());
    double max_err = 0.0;
    for (int j = 0; j <= 8; j++) {
        max_err = std::max(max_err, std::abs(f[j] - f_rec[j]));
    }
    check(max_err < 1e-13, "Fourier cos roundtrip", max_err);
}

void test_fourier_sin_transform() {
    printf("\n--- Fourier: sine transform ---\n");
    spectral::Fourier four(8);

    auto tau = four.collocTau();
    std::vector<double> f(9), coeffs(8);
    for (int j = 0; j <= 8; j++) {
        f[j] = 0.7 * sin(2 * tau[j]) + 0.4 * sin(5 * tau[j]);
    }

    four.forwardSin(f.data(), coeffs.data());

    // b_2 = 0.7, b_5 = 0.4 (stored at indices 1 and 4)
    double err_b2 = std::abs(coeffs[1] - 0.7);
    double err_b5 = std::abs(coeffs[4] - 0.4);
    double err_b1 = std::abs(coeffs[0]);

    check(err_b1 < 1e-13, "Fourier b_1 = 0", err_b1);
    check(err_b2 < 1e-13, "Fourier b_2 = 0.7", err_b2);
    check(err_b5 < 1e-13, "Fourier b_5 = 0.4", err_b5);
}

void test_fourier_time_derivative() {
    printf("\n--- Fourier: time derivative ---\n");
    spectral::Fourier four(8);

    // d/dt [cos(tau) + 0.3 cos(3 tau)] = -sin(tau) - 0.9 sin(3 tau)
    // In units of omega: sin coeffs = {-1, 0, -0.9, 0, ...}
    std::vector<double> cos_c(9, 0.0), sin_c(8);
    cos_c[1] = 1.0;
    cos_c[3] = 0.3;

    four.dtCos(cos_c.data(), sin_c.data());

    // sin_c[k-1] = -k * cos_c[k]
    double err_1 = std::abs(sin_c[0] - (-1.0));
    double err_3 = std::abs(sin_c[2] - (-0.9));
    double err_2 = std::abs(sin_c[1]);

    check(err_1 < 1e-14, "d/dt cos(tau): sin coeff k=1", err_1);
    check(err_2 < 1e-14, "d/dt cos(tau): sin coeff k=2 = 0", err_2);
    check(err_3 < 1e-14, "d/dt cos(tau): sin coeff k=3", err_3);
}

void test_fourier_product() {
    printf("\n--- Fourier: product of cosine series ---\n");
    spectral::Fourier four(10);

    // [cos(tau)]^2 = 0.5 + 0.5*cos(2*tau)
    std::vector<double> a(11, 0.0), c(11, 0.0);
    a[1] = 1.0;

    four.productCosCos(a.data(), a.data(), c.data());

    double err_0 = std::abs(c[0] - 0.5);
    double err_2 = std::abs(c[2] - 0.5);
    double err_1 = std::abs(c[1]);

    check(err_0 < 1e-12, "cos^2: a_0 = 0.5", err_0);
    check(err_1 < 1e-12, "cos^2: a_1 = 0", err_1);
    check(err_2 < 1e-12, "cos^2: a_2 = 0.5", err_2);
}

// ===================== CONVERGENCE TESTS =====================
void test_spectral_convergence() {
    printf("\n--- Spectral convergence: exp(-10*x^2) ---\n");
    // Test exponential convergence of Chebyshev coefficients

    for (int N : {8, 16, 24, 32}) {
        spectral::Chebyshev cheb(N, -1.0, 1.0);

        std::vector<double> f(N+1), coeffs(N+1);
        for (int j = 0; j <= N; j++) {
            f[j] = exp(-10.0 * cheb.grid(j) * cheb.grid(j));
        }

        cheb.forward(f.data(), coeffs.data());

        double tail = std::abs(coeffs[N]);
        printf("  N=%2d: |c_N| = %.6e\n", N, tail);
    }

    // Check N=32 tail is small
    spectral::Chebyshev cheb(32, -1.0, 1.0);
    std::vector<double> f(33), coeffs(33);
    for (int j = 0; j <= 32; j++) {
        f[j] = exp(-10.0 * cheb.grid(j) * cheb.grid(j));
    }
    cheb.forward(f.data(), coeffs.data());
    check(std::abs(coeffs[32]) < 1e-7, "Exponential convergence: |c_32| < 1e-7 for exp(-10x^2)",
          std::abs(coeffs[32]));
}

// ===================== MAIN =====================
int main() {
    printf("===== SPECTRAL MODULE TESTS =====\n");
    printf("Checkpoint 1: Chebyshev, Legendre, Fourier\n");

    // Chebyshev
    test_chebyshev_diff_sin3x();
    test_chebyshev_diff_poly();
    test_chebyshev_second_deriv();
    test_chebyshev_transform();
    test_chebyshev_integration();
    test_chebyshev_mapped_domain();
    test_two_domain();

    // Legendre
    test_legendre_orthogonality();
    test_legendre_transform();
    test_legendre_product();
    test_legendre_product_general();
    test_wigner3j_known_values();

    // Fourier
    test_fourier_cos_transform();
    test_fourier_sin_transform();
    test_fourier_time_derivative();
    test_fourier_product();

    // Convergence
    test_spectral_convergence();

    printf("\n===================================\n");
    printf("Results: %d / %d tests passed\n", tests_passed, tests_total);
    printf("===================================\n");

    return (tests_passed == tests_total) ? 0 : 1;
}
