// test_solver.cpp — Tests for Newton-Raphson and GMRES
#include <cstdio>
#include <cmath>
#include <vector>
#include <functional>
#include "solver/gmres.h"

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

void test_gmres_simple() {
    printf("\n--- GMRES: simple 3x3 system ---\n");
    // A = [[2, 1, 0], [1, 3, 1], [0, 1, 2]]
    // b = [1, 2, 3]
    // x = [0, 1/3, 4/3] ... solve by hand or numerically

    int n = 3;
    double A[] = {2, 1, 0, 1, 3, 1, 0, 1, 2};
    double b[] = {1, 2, 3};
    double x[] = {0, 0, 0};

    auto matvec = [&](const double* v, double* Av) {
        for (int i = 0; i < 3; i++) {
            Av[i] = 0;
            for (int j = 0; j < 3; j++) {
                Av[i] += A[i*3+j] * v[j];
            }
        }
    };

    int iters = solver::gmres(matvec, b, x, n, 10, 1e-12);

    // Verify Ax = b
    double Ax[3];
    matvec(x, Ax);
    double max_err = 0.0;
    for (int i = 0; i < 3; i++) {
        max_err = std::max(max_err, std::abs(Ax[i] - b[i]));
    }

    check(iters >= 0, "GMRES converged", (double)iters);
    check(max_err < 1e-10, "GMRES residual < 1e-10", max_err);

    printf("  Solution: x = [%.6f, %.6f, %.6f], iters = %d\n", x[0], x[1], x[2], iters);
}

void test_gmres_larger() {
    printf("\n--- GMRES: 100x100 diagonal system ---\n");
    int n = 100;
    std::vector<double> diag(n), b(n), x(n, 0.0);
    for (int i = 0; i < n; i++) {
        diag[i] = i + 1.0;
        b[i] = 1.0;
    }

    auto matvec = [&](const double* v, double* Av) {
        for (int i = 0; i < n; i++) Av[i] = diag[i] * v[i];
    };

    int iters = solver::gmres(matvec, b.data(), x.data(), n, 50, 1e-12);

    double max_err = 0.0;
    for (int i = 0; i < n; i++) {
        max_err = std::max(max_err, std::abs(x[i] - 1.0 / diag[i]));
    }

    check(iters >= 0, "GMRES 100x100 converged", (double)iters);
    check(max_err < 1e-10, "GMRES 100x100 solution accurate", max_err);
}

int main() {
    printf("===== SOLVER TESTS =====\n");
    printf("Checkpoint 5\n");

    test_gmres_simple();
    test_gmres_larger();

    printf("\n===================================\n");
    printf("Results: %d / %d tests passed\n", tests_passed, tests_total);
    printf("===================================\n");

    return (tests_passed == tests_total) ? 0 : 1;
}
