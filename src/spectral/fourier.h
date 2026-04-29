// fourier.h — Fourier cosine/sine series for time-periodic fields
// Period T = 2*pi/omega, with time-reversal parity (even: cosine, odd: sine)
#pragma once

#include <vector>
#include <cmath>

namespace spectral {

class Fourier {
public:
    // N_t Fourier modes
    // Even parity: f(t) = sum_{k=0}^{N_t} a_k cos(k*omega*t)
    // Odd parity:  f(t) = sum_{k=1}^{N_t} b_k sin(k*omega*t)
    Fourier(int N_t);

    int modes() const { return N_t_; }

    // Number of collocation points in half-period [0, T/2]
    // Using N_t + 1 equispaced points in [0, pi/omega]
    int nColloc() const { return N_t_ + 1; }

    // Collocation times: t_j = j * pi / (N_t * omega) for j = 0,...,N_t
    // In normalized variable tau = omega*t, collocation at tau_j = j*pi/N_t
    std::vector<double> collocTau() const;

    // Forward cosine transform: values at collocation points -> cosine coefficients
    // f(tau_j) = sum_{k=0}^{N_t} a_k cos(k * tau_j)
    void forwardCos(const double* values, double* coeffs) const;

    // Inverse cosine transform: coefficients -> values at collocation points
    void inverseCos(const double* coeffs, double* values) const;

    // Forward sine transform: values at collocation points -> sine coefficients
    // f(tau_j) = sum_{k=1}^{N_t} b_k sin(k * tau_j)
    void forwardSin(const double* values, double* coeffs) const;

    // Inverse sine transform: coefficients -> values at collocation points
    void inverseSin(const double* coeffs, double* values) const;

    // Time derivative of cosine series: d/dt [sum a_k cos(k*omega*t)]
    //   = -omega * sum k * a_k sin(k*omega*t)
    // Returns sine coefficients: b_k = -k * a_k (in units of omega)
    void dtCos(const double* cos_coeffs, double* sin_coeffs) const;

    // Time derivative of sine series: d/dt [sum b_k sin(k*omega*t)]
    //   = omega * sum k * b_k cos(k*omega*t)
    void dtSin(const double* sin_coeffs, double* cos_coeffs) const;

    // Product of two cosine series (truncated to N_t modes):
    // [sum a_k cos(k tau)] * [sum b_k cos(k tau)] = sum c_k cos(k tau)
    void productCosCos(const double* a, const double* b, double* c) const;

    // Product of cosine and sine series:
    // [sum a_k cos(k tau)] * [sum b_k sin(k tau)] = sum c_k sin(k tau)
    void productCosSin(const double* a, const double* b, double* c) const;

    // Product of two sine series gives cosine:
    // [sum a_k sin(k tau)] * [sum b_k sin(k tau)] = sum c_k cos(k tau)
    void productSinSin(const double* a, const double* b, double* c) const;

    // Evaluate cosine series at arbitrary tau
    double evalCos(const double* coeffs, double tau) const;

    // Evaluate sine series at arbitrary tau
    double evalSin(const double* coeffs, double tau) const;

private:
    int N_t_;
};

} // namespace spectral
