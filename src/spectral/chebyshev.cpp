// chebyshev.cpp — Chebyshev spectral methods implementation
#include "spectral/chebyshev.h"
#include <cstring>
#include <numeric>
#include <stdexcept>

namespace spectral {

static const double PI = 3.14159265358979323846264338327950288;

Chebyshev::Chebyshev(int N, double a, double b)
    : N_(N), a_(a), b_(b), x_(N+1), D_((N+1)*(N+1)), D2_((N+1)*(N+1)), w_(N+1)
{
    if (N < 1) throw std::invalid_argument("Chebyshev order must be >= 1");
    buildGrid();
    buildDiffMatrix();
    buildWeights();
}

void Chebyshev::buildGrid() {
    // Gauss-Lobatto: x_j = (a+b)/2 + (b-a)/2 * cos(pi*j/N)
    double mid = 0.5*(a_ + b_);
    double half = 0.5*(b_ - a_);
    for (int j = 0; j <= N_; j++) {
        x_[j] = mid + half * cos(PI * j / N_);
    }
}

void Chebyshev::buildDiffMatrix() {
    int Np = N_ + 1;
    // Standard Chebyshev differentiation matrix on [-1,1]
    // then rescale for [a,b]

    // Barycentric weights
    std::vector<double> c(Np);
    for (int j = 0; j <= N_; j++) {
        c[j] = 1.0;
        if (j == 0 || j == N_) c[j] = 2.0;
        if (j % 2 == 1) c[j] = -c[j];
    }

    // Standard grid points on [-1,1]
    std::vector<double> s(Np);
    for (int j = 0; j <= N_; j++) {
        s[j] = cos(PI * j / N_);
    }

    // Build D on [-1,1]
    for (int i = 0; i <= N_; i++) {
        for (int j = 0; j <= N_; j++) {
            if (i != j) {
                D_[i*Np + j] = (c[i] / c[j]) / (s[i] - s[j]);
            }
        }
        // Diagonal: negative row sum
        double rowsum = 0.0;
        for (int j = 0; j <= N_; j++) {
            if (j != i) rowsum += D_[i*Np + j];
        }
        D_[i*Np + i] = -rowsum;
    }

    // Rescale for [a,b]: d/dx = (2/(b-a)) * d/ds
    double scale = 2.0 / (b_ - a_);
    for (int i = 0; i < Np*Np; i++) {
        D_[i] *= scale;
    }

    // Second derivative matrix: D2 = D * D
    std::fill(D2_.begin(), D2_.end(), 0.0);
    for (int i = 0; i < Np; i++) {
        for (int j = 0; j < Np; j++) {
            double s = 0.0;
            for (int k = 0; k < Np; k++) {
                s += D_[i*Np + k] * D_[k*Np + j];
            }
            D2_[i*Np + j] = s;
        }
    }
}

void Chebyshev::buildWeights() {
    // Clenshaw-Curtis quadrature weights on [-1,1] then scaled to [a,b]
    // Reference: Waldvogel (2006), "Fast construction of the Fejer and Clenshaw-Curtis
    // quadrature rules", BIT Numerical Mathematics 46(1), 195-202.
    //
    // For N+1 Chebyshev-Lobatto points x_j = cos(j*pi/N):
    // w_j = (c_j/N) * sum_{k=0}^{N} '' b_k * cos(k*j*pi/N)
    // where b_k = 2/(1-k^2) for even k, 0 for odd k
    // and '' means first and last terms halved, c_0 = c_N = 0.5, else 1

    int N = N_;

    // Build b_k for k = 0, ..., N
    std::vector<double> b(N + 1, 0.0);
    for (int k = 0; k <= N; k += 2) {
        b[k] = 2.0 / (1.0 - (double)(k * k));
    }

    // w_j = (1/N) * sum_{k=0}^{N} '' b_k cos(k j pi / N)
    // where '' means b_0 and b_N are halved
    for (int j = 0; j <= N; j++) {
        double wj = 0.5 * b[0]; // k=0 term (halved)
        for (int k = 1; k < N; k++) {
            wj += b[k] * cos(k * j * PI / N);
        }
        wj += 0.5 * b[N] * cos(j * PI); // k=N term (halved)
        wj *= 2.0 / N;

        if (j == 0 || j == N) wj *= 0.5;
        // Scale to [a,b]
        w_[j] = wj * (b_ - a_) / 2.0;
    }
}

void Chebyshev::differentiate(const double* f, double* df) const {
    int Np = N_ + 1;
    for (int i = 0; i < Np; i++) {
        double s = 0.0;
        for (int j = 0; j < Np; j++) {
            s += D_[i*Np + j] * f[j];
        }
        df[i] = s;
    }
}

void Chebyshev::differentiate(const std::vector<double>& f, std::vector<double>& df) const {
    assert((int)f.size() == N_+1);
    df.resize(N_+1);
    differentiate(f.data(), df.data());
}

void Chebyshev::differentiate2(const double* f, double* d2f) const {
    int Np = N_ + 1;
    for (int i = 0; i < Np; i++) {
        double s = 0.0;
        for (int j = 0; j < Np; j++) {
            s += D2_[i*Np + j] * f[j];
        }
        d2f[i] = s;
    }
}

void Chebyshev::forward(const double* values, double* coeffs) const {
    // DCT-I based transform: c_k = (2/N) sum_{j=0}^{N} ''  f_j cos(pi*j*k/N)
    // where '' means first and last terms halved
    int Np = N_ + 1;
    for (int k = 0; k <= N_; k++) {
        double ck = 0.0;
        for (int j = 0; j <= N_; j++) {
            double w = 1.0;
            if (j == 0 || j == N_) w = 0.5;
            ck += w * values[j] * cos(PI * j * k / N_);
        }
        ck *= 2.0 / N_;
        if (k == 0 || k == N_) ck *= 0.5;
        coeffs[k] = ck;
    }
}

void Chebyshev::inverse(const double* coeffs, double* values) const {
    int Np = N_ + 1;
    for (int j = 0; j <= N_; j++) {
        double s = coeffs[0]; // T_0 = 1
        for (int k = 1; k < N_; k++) {
            s += coeffs[k] * cos(PI * j * k / N_);
        }
        s += coeffs[N_] * cos(PI * j); // T_N at x_j
        values[j] = s;
    }
}

double Chebyshev::evaluate(const double* coeffs, double x) const {
    double s = to_standard(x);
    return clenshaw(coeffs, s);
}

double Chebyshev::clenshaw(const double* coeffs, double s) const {
    // Clenshaw algorithm for sum c_k T_k(s)
    if (N_ == 0) return coeffs[0];
    double b1 = 0.0, b2 = 0.0;
    for (int k = N_; k >= 1; k--) {
        double tmp = 2.0 * s * b1 - b2 + coeffs[k];
        b2 = b1;
        b1 = tmp;
    }
    return coeffs[0] + s * b1 - b2;
}

double Chebyshev::integrate(const double* f) const {
    double sum = 0.0;
    for (int j = 0; j <= N_; j++) {
        sum += w_[j] * f[j];
    }
    return sum;
}

// --- Two-domain ---

TwoDomainChebyshev::TwoDomainChebyshev(int N_nuc, int N_shell, double rho_mid)
    : nuc_(N_nuc, 0.0, rho_mid), shell_(N_shell, rho_mid, 1.0), rho_mid_(rho_mid)
{
    // Build global grid: nucleus points (decreasing from rho_mid to 0)
    // then shell points (decreasing from 1 to rho_mid), skipping duplicate at rho_mid
    int total = nuc_.size() + shell_.size() - 1;
    global_grid_.resize(total);
    for (int j = 0; j < nuc_.size(); j++) {
        global_grid_[j] = nuc_.grid(j);
    }
    // shell_.grid(0) = 1.0 (right endpoint of shell)
    // shell_.grid(N_shell) = rho_mid (left endpoint = nuc_.grid(0))
    // Skip the last shell point (= rho_mid, already included)
    for (int j = 0; j < shell_.size() - 1; j++) {
        global_grid_[nuc_.size() + j] = shell_.grid(j);
    }
}

void TwoDomainChebyshev::differentiate(const double* f_nuc, const double* f_shell,
                                         double* df_nuc, double* df_shell) const {
    nuc_.differentiate(f_nuc, df_nuc);
    shell_.differentiate(f_shell, df_shell);
}

} // namespace spectral
