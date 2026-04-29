// field3d.cpp — 3D spectral field implementation
#include "solver/field3d.h"
#include <cstring>
#include <stdexcept>
#include <cmath>

namespace solver {

Field3D::Field3D(const spectral::Fourier& four,
                 const spectral::TwoDomainChebyshev& radial,
                 const spectral::Legendre& angular,
                 bool time_even, const std::string& name)
    : four_(four), radial_(radial), angular_(angular),
      time_even_(time_even), name_(name)
{
    int sz = totalSize();
    coeffs_.resize(sz, 0.0);
}

int Field3D::nFourierModes() const {
    return time_even_ ? four_.modes() + 1 : four_.modes();
}

int Field3D::totalSize() const {
    int nf = nFourierModes();
    int n_nuc = radial_.nucleus().size();
    int n_shell = radial_.shell().size();
    int n_ang = angular_.size();
    return nf * (n_nuc + n_shell) * n_ang;
}

int Field3D::offset(int k, int domain, int n, int j) const {
    int n_nuc = radial_.nucleus().size();
    int n_shell = radial_.shell().size();
    int n_ang = angular_.size();
    int n_rad = (domain == 0) ? n_nuc : n_shell;

    int rad_offset = (domain == 0) ? 0 : n_nuc * n_ang;
    return k * (n_nuc + n_shell) * n_ang + rad_offset + n * n_ang + j;
}

double& Field3D::coeff(int k, int domain, int n, int j) {
    return coeffs_[offset(k, domain, n, j)];
}

double Field3D::coeff(int k, int domain, int n, int j) const {
    return coeffs_[offset(k, domain, n, j)];
}

void Field3D::setZero() {
    std::fill(coeffs_.begin(), coeffs_.end(), 0.0);
}

double Field3D::evaluate(double tau, double rho, double theta) const {
    // Determine which domain
    double rho_mid = radial_.rho_mid();
    int domain = (rho <= rho_mid) ? 0 : 1;

    const spectral::Chebyshev& cheb = (domain == 0) ? radial_.nucleus() : radial_.shell();
    int n_rad = cheb.size();
    int n_ang = angular_.size();
    int n_four = nFourierModes();

    // For each Fourier mode, evaluate the spatial part
    double result = 0.0;
    for (int k = 0; k < n_four; k++) {
        // Get radial coefficients for this Fourier mode and evaluate
        // First do the Legendre sum at the given theta
        double spatial = 0.0;
        for (int n = 0; n < n_rad; n++) {
            // Sum over angular modes
            double ang_sum = 0.0;
            for (int j = 0; j < n_ang; j++) {
                ang_sum += coeff(k, domain, n, j) *
                           spectral::Legendre::P(j, cos(theta));
            }
            // Evaluate Chebyshev at rho
            // This is approximate - should use Chebyshev interpolation properly
            // For now, just use the grid value if on grid
            spatial += ang_sum; // Placeholder
        }
        // TODO: proper Chebyshev interpolation

        // Apply Fourier factor
        if (time_even_) {
            result += spatial * cos(k * tau);
        } else {
            result += spatial * sin((k+1) * tau);
        }
    }

    return result;
}

void Field3D::getGridValues(std::vector<double>& values) const {
    // Not yet needed for initial tests
    values.resize(totalSize(), 0.0);
}

void Field3D::setFromGridValues(const std::vector<double>& values) {
    if ((int)values.size() != totalSize()) {
        throw std::runtime_error("Field3D::setFromGridValues: size mismatch");
    }
    coeffs_ = values;
}

} // namespace solver
