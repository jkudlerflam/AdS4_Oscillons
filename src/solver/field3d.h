// field3d.h — 3D spectral field representation
// Fourier (time) x Chebyshev (radial) x Legendre (angular)
#pragma once

#include "spectral/chebyshev.h"
#include "spectral/legendre.h"
#include "spectral/fourier.h"
#include <vector>
#include <string>

namespace solver {

// A field on the 3D grid with specific time-parity
class Field3D {
public:
    // time_even: true for cosine series, false for sine series
    Field3D(const spectral::Fourier& four,
            const spectral::TwoDomainChebyshev& radial,
            const spectral::Legendre& angular,
            bool time_even, const std::string& name = "");

    const std::string& name() const { return name_; }
    bool isTimeEven() const { return time_even_; }

    int nFourier() const { return four_.modes(); }
    int nRadialNuc() const { return radial_.nucleus().size(); }
    int nRadialShell() const { return radial_.shell().size(); }
    int nAngular() const { return angular_.size(); }

    // Total number of coefficients
    int totalSize() const;

    // Access coefficient: c[k][domain][n][j]
    // k = Fourier mode (0..N_t for cos, 0..N_t-1 for sin)
    // domain = 0 (nucleus) or 1 (shell)
    // n = radial index, j = angular index
    double& coeff(int k, int domain, int n, int j);
    double coeff(int k, int domain, int n, int j) const;

    // Flat access to coefficient vector (for Newton-Raphson)
    double* data() { return coeffs_.data(); }
    const double* data() const { return coeffs_.data(); }
    int dataSize() const { return (int)coeffs_.size(); }

    // Set all coefficients to zero
    void setZero();

    // Evaluate at a point (tau, rho, theta)
    // tau = omega * t (normalized time)
    double evaluate(double tau, double rho, double theta) const;

    // Get values on the full collocation grid
    // grid[k][n][j] for Fourier x Radial x Angular
    void getGridValues(std::vector<double>& values) const;

    // Set from grid values
    void setFromGridValues(const std::vector<double>& values);

private:
    const spectral::Fourier& four_;
    const spectral::TwoDomainChebyshev& radial_;
    const spectral::Legendre& angular_;
    bool time_even_;
    std::string name_;

    std::vector<double> coeffs_;
    int nFourierModes() const;
    int offset(int k, int domain, int n, int j) const;
};

} // namespace solver
