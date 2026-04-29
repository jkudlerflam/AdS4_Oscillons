// legendre.h — Legendre polynomial spectral methods
// Gauss-Lobatto grids on [0, pi] (theta coordinate on S^2)
// Transforms, products, and Clebsch-Gordan decomposition
#pragma once

#include <vector>
#include <cmath>

namespace spectral {

class Legendre {
public:
    // Construct with N_theta+1 Gauss-Lobatto points for P_j(cos theta)
    // Parity: if specified, only use P_j with j having given parity (0=even, 1=odd, -1=both)
    Legendre(int N_theta, int parity = -1);

    int order() const { return N_theta_; }
    int size() const { return N_theta_ + 1; }
    int parity() const { return parity_; }

    // Grid points in theta in [0, pi]
    const std::vector<double>& thetaGrid() const { return theta_; }
    double theta(int j) const { return theta_[j]; }

    // cos(theta) grid
    const std::vector<double>& cosGrid() const { return mu_; }
    double mu(int j) const { return mu_[j]; }

    // Evaluate P_ell(x) using 3-term recurrence
    static double P(int ell, double x);

    // Evaluate P_ell for all ell = 0,...,L at point x
    static void P_all(int L, double x, double* result);

    // Forward Legendre transform: values at grid points -> coefficients a_ell
    // f(theta_j) = sum_{ell} a_ell P_ell(cos theta_j)
    // Uses quadrature: a_ell = (2ell+1)/2 * sum_j w_j f(theta_j) P_ell(mu_j)
    void forward(const double* values, double* coeffs, int L_max) const;

    // Inverse transform: coefficients -> values at grid points
    void inverse(const double* coeffs, int L_max, double* values) const;

    // Gauss-Legendre quadrature weights (for integration against sin(theta) dtheta)
    const std::vector<double>& weights() const { return w_; }

    // Integrate f(theta) sin(theta) dtheta using quadrature
    double integrate(const double* f) const;

    // Product linearization: P_{l1}(x) * P_{l2}(x) = sum_L c_L P_L(x)
    // Returns coefficients c_L for L = |l1-l2|, |l1-l2|+2, ..., l1+l2
    // Uses Clebsch-Gordan (Wigner 3j) formula
    static std::vector<double> productCoeffs(int l1, int l2);

    // Wigner 3j symbol ( l1 l2 L ; 0 0 0 )
    static double wigner3j_000(int l1, int l2, int L);

    // Differentiation: d/dtheta of Legendre expansion
    // Given coefficients a_ell, compute d/dtheta [sum a_ell P_ell(cos theta)]
    // at the grid points
    void differentiate(const double* coeffs, int L_max, double* dvalues) const;

private:
    int N_theta_;
    int parity_;  // -1 = both, 0 = even, 1 = odd
    std::vector<double> theta_;
    std::vector<double> mu_;     // cos(theta)
    std::vector<double> w_;      // quadrature weights
    std::vector<double> P_grid_; // P_ell(mu_j), stored [ell * size + j]

    void buildGaussLobattoGrid();
    void buildGaussLegendreGrid();
};

} // namespace spectral
