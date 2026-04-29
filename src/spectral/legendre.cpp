// legendre.cpp — Legendre spectral methods implementation
// Uses Gauss-Legendre quadrature (interior points, exact for poly deg 2n-1)
#include "spectral/legendre.h"
#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace spectral {

static const double PI = 3.14159265358979323846264338327950288;

Legendre::Legendre(int N_theta, int parity)
    : N_theta_(N_theta), parity_(parity)
{
    buildGaussLegendreGrid();
}

double Legendre::P(int ell, double x) {
    if (ell == 0) return 1.0;
    if (ell == 1) return x;
    double p0 = 1.0, p1 = x;
    for (int k = 2; k <= ell; k++) {
        double p2 = ((2*k - 1) * x * p1 - (k - 1) * p0) / k;
        p0 = p1;
        p1 = p2;
    }
    return p1;
}

void Legendre::P_all(int L, double x, double* result) {
    result[0] = 1.0;
    if (L == 0) return;
    result[1] = x;
    for (int k = 2; k <= L; k++) {
        result[k] = ((2*k - 1) * x * result[k-1] - (k - 1) * result[k-2]) / k;
    }
}

void Legendre::buildGaussLegendreGrid() {
    // Pure Gauss-Legendre quadrature: n points, exact for polynomials of degree 2n-1
    // Roots of P_n(x) on [-1, 1]
    int n = N_theta_ + 1;
    mu_.resize(n);
    w_.resize(n);
    theta_.resize(n);

    // Find roots of P_n(x) by Newton's method
    for (int i = 0; i < n; i++) {
        // Initial guess using Tricomi's approximation
        double xi = cos(PI * (i + 0.75) / (n + 0.5));

        for (int iter = 0; iter < 200; iter++) {
            // Evaluate P_n(xi) and P_{n-1}(xi) using recurrence
            double p0 = 1.0, p1 = xi;
            for (int k = 2; k <= n; k++) {
                double p2 = ((2*k - 1) * xi * p1 - (k - 1) * p0) / k;
                p0 = p1;
                p1 = p2;
            }
            // p1 = P_n(xi), p0 = P_{n-1}(xi)

            // Newton step: P'_n(x) = n * (P_{n-1}(x) - x * P_n(x)) / (1 - x^2)
            double dP = n * (p0 - xi * p1) / (1.0 - xi * xi);
            double dx = -p1 / dP;
            xi += dx;
            if (std::abs(dx) < 1e-16) break;
        }
        mu_[i] = xi;
    }

    // Sort in decreasing order (mu[0] = most positive)
    std::sort(mu_.begin(), mu_.end(), std::greater<double>());

    // Weights: w_i = 2 / ((1 - x_i^2) * [P'_n(x_i)]^2)
    for (int i = 0; i < n; i++) {
        double x = mu_[i];
        // Evaluate P_n(x) and P_{n-1}(x)
        double p0 = 1.0, p1 = x;
        for (int k = 2; k <= n; k++) {
            double p2 = ((2*k - 1) * x * p1 - (k - 1) * p0) / k;
            p0 = p1;
            p1 = p2;
        }
        // P'_n(x) = n * (P_{n-1}(x) - x * P_n(x)) / (1 - x^2)
        double dP = n * (p0 - x * p1) / (1.0 - x * x);
        w_[i] = 2.0 / ((1.0 - x * x) * dP * dP);
    }

    // theta grid: theta = arccos(mu)
    for (int i = 0; i < n; i++) {
        theta_[i] = acos(mu_[i]);
    }
}

void Legendre::forward(const double* values, double* coeffs, int L_max) const {
    // a_ell = (2*ell+1)/2 * sum_j w_j * f_j * P_ell(mu_j)
    // This uses the quadrature to compute the projection integral
    int n = N_theta_ + 1;
    std::vector<double> pvals(L_max + 1);

    for (int ell = 0; ell <= L_max; ell++) {
        coeffs[ell] = 0.0;
    }

    for (int j = 0; j < n; j++) {
        P_all(L_max, mu_[j], pvals.data());
        for (int ell = 0; ell <= L_max; ell++) {
            coeffs[ell] += w_[j] * values[j] * pvals[ell];
        }
    }

    for (int ell = 0; ell <= L_max; ell++) {
        coeffs[ell] *= (2*ell + 1) / 2.0;
    }
}

void Legendre::inverse(const double* coeffs, int L_max, double* values) const {
    int n = N_theta_ + 1;
    std::vector<double> pvals(L_max + 1);

    for (int j = 0; j < n; j++) {
        P_all(L_max, mu_[j], pvals.data());
        double s = 0.0;
        for (int ell = 0; ell <= L_max; ell++) {
            s += coeffs[ell] * pvals[ell];
        }
        values[j] = s;
    }
}

double Legendre::integrate(const double* f) const {
    double sum = 0.0;
    int n = N_theta_ + 1;
    for (int j = 0; j < n; j++) {
        sum += w_[j] * f[j];
    }
    return sum;
}

double Legendre::wigner3j_000(int l1, int l2, int L) {
    // Wigner 3j symbol (l1 l2 L; 0 0 0)
    // Zero unless l1 + l2 + L is even and triangle inequality holds
    int J = l1 + l2 + L;
    if (J % 2 != 0) return 0.0;
    if (L > l1 + l2 || L < std::abs(l1 - l2)) return 0.0;

    int g = J / 2;

    auto lgam = [](int n) -> double {
        return std::lgamma(n + 1.0);
    };

    double log_num = lgam(g) + 0.5 * (lgam(J - 2*l1) + lgam(J - 2*l2) + lgam(J - 2*L));
    double log_den = lgam(g - l1) + lgam(g - l2) + lgam(g - L) + 0.5 * lgam(J + 1);

    double sign = (g % 2 == 0) ? 1.0 : -1.0;
    return sign * exp(log_num - log_den);
}

std::vector<double> Legendre::productCoeffs(int l1, int l2) {
    // P_{l1}(x) * P_{l2}(x) = sum_L (2L+1) * (l1 l2 L; 0 0 0)^2 * P_L(x)
    int Lmax = l1 + l2;

    std::vector<double> result(Lmax + 1, 0.0);

    int Lmin = std::abs(l1 - l2);
    for (int L = Lmin; L <= Lmax; L += 2) {
        double w3j = wigner3j_000(l1, l2, L);
        result[L] = (2*L + 1) * w3j * w3j;
    }

    return result;
}

void Legendre::differentiate(const double* coeffs, int L_max, double* dvalues) const {
    int n = N_theta_ + 1;
    std::vector<double> pvals(L_max + 2);

    for (int j = 0; j < n; j++) {
        P_all(L_max, mu_[j], pvals.data());
        double s = 0.0;
        double sth = sin(theta_[j]);

        if (std::abs(sth) < 1e-14) {
            dvalues[j] = 0.0;
        } else {
            for (int ell = 1; ell <= L_max; ell++) {
                // d/dtheta P_ell(cos theta) = -sin(theta) * P'_ell(cos theta)
                // Using: (1-x^2) P'_ell(x) = ell * (P_{ell-1}(x) - x P_ell(x))
                // => -sin(theta) P'_ell = ell*(P_{ell-1} - cos(theta)*P_ell) / sin(theta) * (-sin(theta))
                // Wait, more carefully:
                // P'_ell(x) = ell * (P_{ell-1}(x) - x P_ell(x)) / (1 - x^2)
                //           = ell * (P_{ell-1}(mu) - mu P_ell(mu)) / sin^2(theta)
                // d/dtheta P_ell = -sin(theta) * P'_ell(mu)
                //                = -ell * (P_{ell-1}(mu) - mu*P_ell(mu)) / sin(theta)
                s += coeffs[ell] * (-ell) * (pvals[ell-1] - mu_[j] * pvals[ell]) / sth;
            }
            dvalues[j] = s;
        }
    }
}

} // namespace spectral
