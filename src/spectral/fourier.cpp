// fourier.cpp — Fourier cosine/sine series implementation
#include "spectral/fourier.h"
#include <cstring>
#include <algorithm>

namespace spectral {

static const double PI = 3.14159265358979323846264338327950288;

Fourier::Fourier(int N_t) : N_t_(N_t) {
    if (N_t < 1) throw "Fourier modes must be >= 1";
}

std::vector<double> Fourier::collocTau() const {
    std::vector<double> tau(N_t_ + 1);
    for (int j = 0; j <= N_t_; j++) {
        tau[j] = PI * j / N_t_;
    }
    return tau;
}

void Fourier::forwardCos(const double* values, double* coeffs) const {
    // DCT-I: a_k = (2/N_t) sum_{j=0}^{N_t} '' values[j] cos(k * pi * j / N_t)
    // where '' means endpoints halved
    int N = N_t_;
    for (int k = 0; k <= N; k++) {
        double ck = 0.0;
        for (int j = 0; j <= N; j++) {
            double w = 1.0;
            if (j == 0 || j == N) w = 0.5;
            ck += w * values[j] * cos(PI * k * j / N);
        }
        ck *= 2.0 / N;
        if (k == 0 || k == N) ck *= 0.5;
        coeffs[k] = ck;
    }
}

void Fourier::inverseCos(const double* coeffs, double* values) const {
    int N = N_t_;
    for (int j = 0; j <= N; j++) {
        double s = coeffs[0];
        for (int k = 1; k < N; k++) {
            s += coeffs[k] * cos(PI * k * j / N);
        }
        s += coeffs[N] * cos(PI * j); // = coeffs[N] * (-1)^j
        values[j] = s;
    }
}

void Fourier::forwardSin(const double* values, double* coeffs) const {
    // DST-I: b_k = (2/N_t) sum_{j=1}^{N_t-1} values[j] sin(k * pi * j / N_t)
    // Note: values[0] = values[N_t] = 0 for a sine series
    int N = N_t_;
    for (int k = 1; k <= N; k++) {
        double bk = 0.0;
        for (int j = 0; j <= N; j++) {
            bk += values[j] * sin(PI * k * j / N);
        }
        bk *= 2.0 / N;
        // No endpoint correction needed for sine (sin=0 at endpoints)
        // Actually, the formula is just sum over interior:
        // but being safe, include all and let sin handle it
        coeffs[k-1] = bk; // store b_1,...,b_{N_t} in indices 0,...,N_t-1
    }
}

void Fourier::inverseSin(const double* coeffs, double* values) const {
    int N = N_t_;
    for (int j = 0; j <= N; j++) {
        double s = 0.0;
        for (int k = 1; k <= N; k++) {
            s += coeffs[k-1] * sin(PI * k * j / N);
        }
        values[j] = s;
    }
}

void Fourier::dtCos(const double* cos_coeffs, double* sin_coeffs) const {
    // d/dt [sum a_k cos(k omega t)] = -omega sum k a_k sin(k omega t)
    // Return in units of omega: sin_coeffs[k-1] = -k * cos_coeffs[k]
    for (int k = 1; k <= N_t_; k++) {
        sin_coeffs[k-1] = -k * cos_coeffs[k];
    }
}

void Fourier::dtSin(const double* sin_coeffs, double* cos_coeffs) const {
    // d/dt [sum b_k sin(k omega t)] = omega sum k b_k cos(k omega t)
    // cos_coeffs[k] = k * sin_coeffs[k-1]
    cos_coeffs[0] = 0.0; // no k=0 contribution from derivative of sine series
    for (int k = 1; k <= N_t_; k++) {
        cos_coeffs[k] = k * sin_coeffs[k-1];
    }
}

void Fourier::productCosCos(const double* a, const double* b, double* c) const {
    // cos(k tau) cos(j tau) = 0.5 [cos((k-j)tau) + cos((k+j)tau)]
    // Do in physical space for simplicity: transform to grid, multiply, transform back
    int N = N_t_;
    // Use 2*N_t points for dealiasing
    int M = 2 * N + 1;
    std::vector<double> va(M), vb(M), vc(M);

    // Evaluate on fine grid
    for (int j = 0; j < M; j++) {
        double tau = PI * j / (M - 1.0);
        double sa = 0.0, sb = 0.0;
        for (int k = 0; k <= N; k++) {
            double ct = cos(k * tau);
            sa += a[k] * ct;
            sb += b[k] * ct;
        }
        va[j] = sa;
        vb[j] = sb;
        vc[j] = sa * sb;
    }

    // Transform product back to cosine coefficients
    for (int k = 0; k <= N; k++) {
        double ck = 0.0;
        for (int j = 0; j < M; j++) {
            double w = 1.0;
            if (j == 0 || j == M-1) w = 0.5;
            ck += w * vc[j] * cos(PI * k * j / (M - 1.0));
        }
        ck *= 2.0 / (M - 1.0);
        if (k == 0) ck *= 0.5;
        c[k] = ck;
    }
}

void Fourier::productCosSin(const double* a, const double* b, double* c) const {
    int N = N_t_;
    int M = 2 * N + 1;
    std::vector<double> vc(M);

    for (int j = 0; j < M; j++) {
        double tau = PI * j / (M - 1.0);
        double sa = 0.0, sb = 0.0;
        for (int k = 0; k <= N; k++) sa += a[k] * cos(k * tau);
        for (int k = 1; k <= N; k++) sb += b[k-1] * sin(k * tau);
        vc[j] = sa * sb;
    }

    // Extract sine coefficients
    for (int k = 1; k <= N; k++) {
        double bk = 0.0;
        for (int j = 0; j < M; j++) {
            bk += vc[j] * sin(PI * k * j / (M - 1.0));
        }
        bk *= 2.0 / (M - 1.0);
        c[k-1] = bk;
    }
}

void Fourier::productSinSin(const double* a, const double* b, double* c) const {
    int N = N_t_;
    int M = 2 * N + 1;
    std::vector<double> vc(M);

    for (int j = 0; j < M; j++) {
        double tau = PI * j / (M - 1.0);
        double sa = 0.0, sb = 0.0;
        for (int k = 1; k <= N; k++) sa += a[k-1] * sin(k * tau);
        for (int k = 1; k <= N; k++) sb += b[k-1] * sin(k * tau);
        vc[j] = sa * sb;
    }

    for (int k = 0; k <= N; k++) {
        double ck = 0.0;
        for (int j = 0; j < M; j++) {
            double w = 1.0;
            if (j == 0 || j == M-1) w = 0.5;
            ck += w * vc[j] * cos(PI * k * j / (M - 1.0));
        }
        ck *= 2.0 / (M - 1.0);
        if (k == 0) ck *= 0.5;
        c[k] = ck;
    }
}

double Fourier::evalCos(const double* coeffs, double tau) const {
    double s = coeffs[0];
    for (int k = 1; k <= N_t_; k++) {
        s += coeffs[k] * cos(k * tau);
    }
    return s;
}

double Fourier::evalSin(const double* coeffs, double tau) const {
    double s = 0.0;
    for (int k = 1; k <= N_t_; k++) {
        s += coeffs[k-1] * sin(k * tau);
    }
    return s;
}

} // namespace spectral
