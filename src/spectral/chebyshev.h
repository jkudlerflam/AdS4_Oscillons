// chebyshev.h — Chebyshev spectral methods on Gauss-Lobatto grids
// Differentiation, interpolation, and transforms for domains [a,b]
#pragma once

#include <vector>
#include <cmath>
#include <cassert>

namespace spectral {

class Chebyshev {
public:
    // Construct for N+1 Gauss-Lobatto points on [a, b]
    // Points: x_j = (a+b)/2 + (b-a)/2 * cos(pi*j/N),  j = 0,...,N
    // Note: x_0 = b (right endpoint), x_N = a (left endpoint)
    Chebyshev(int N, double a = -1.0, double b = 1.0);

    int order() const { return N_; }
    int size() const { return N_ + 1; }
    double a() const { return a_; }
    double b() const { return b_; }

    // Grid points (N+1 of them)
    const std::vector<double>& grid() const { return x_; }
    double grid(int j) const { return x_[j]; }

    // Differentiation matrix D (maps values at grid points to derivative values)
    // D is (N+1) x (N+1), stored row-major
    const std::vector<double>& diffMatrix() const { return D_; }
    double D(int i, int j) const { return D_[i * (N_+1) + j]; }

    // Apply differentiation: given f values at grid points, compute f' at grid points
    void differentiate(const double* f, double* df) const;
    void differentiate(const std::vector<double>& f, std::vector<double>& df) const;

    // Second derivative
    void differentiate2(const double* f, double* d2f) const;

    // Forward transform: values at grid points -> Chebyshev coefficients
    // f(x) = sum_{k=0}^{N} c_k T_k(s(x)) where s maps [a,b] to [-1,1]
    void forward(const double* values, double* coeffs) const;

    // Inverse transform: Chebyshev coefficients -> values at grid points
    void inverse(const double* coeffs, double* values) const;

    // Evaluate the interpolant at an arbitrary point x in [a,b]
    double evaluate(const double* coeffs, double x) const;

    // Clenshaw evaluation from coefficients
    double clenshaw(const double* coeffs, double s) const;

    // Integration weights (Clenshaw-Curtis)
    const std::vector<double>& weights() const { return w_; }
    double integrate(const double* f) const;

    // Map from [a,b] to [-1,1] and back
    double to_standard(double x) const { return 2.0*(x - a_)/(b_ - a_) - 1.0; }
    double from_standard(double s) const { return a_ + (s + 1.0)*(b_ - a_)/2.0; }

private:
    int N_;
    double a_, b_;
    std::vector<double> x_;      // grid points
    std::vector<double> D_;      // differentiation matrix
    std::vector<double> D2_;     // second derivative matrix
    std::vector<double> w_;      // integration weights

    void buildGrid();
    void buildDiffMatrix();
    void buildWeights();
};

// Two-domain Chebyshev: nucleus [0, rho_mid] and shell [rho_mid, 1]
// with C^k matching at rho_mid
class TwoDomainChebyshev {
public:
    TwoDomainChebyshev(int N_nuc, int N_shell, double rho_mid = 0.5);

    const Chebyshev& nucleus() const { return nuc_; }
    const Chebyshev& shell() const { return shell_; }

    int totalPoints() const { return nuc_.size() + shell_.size() - 1; }
    // -1 because the interface point is shared

    double rho_mid() const { return rho_mid_; }

    // Global grid (nucleus points then shell points, interface shared)
    const std::vector<double>& grid() const { return global_grid_; }

    // Differentiate across both domains (with matching at interface)
    void differentiate(const double* f_nuc, const double* f_shell,
                       double* df_nuc, double* df_shell) const;

private:
    Chebyshev nuc_, shell_;
    double rho_mid_;
    std::vector<double> global_grid_;
};

} // namespace spectral
