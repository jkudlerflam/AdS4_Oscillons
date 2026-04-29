// gmres.cpp — Restarted GMRES implementation
#include "solver/gmres.h"
#include <cmath>
#include <cstring>
#include <algorithm>
#include <cstdio>

namespace solver {

static double dot(const double* a, const double* b, int n) {
    double s = 0.0;
    for (int i = 0; i < n; i++) s += a[i] * b[i];
    return s;
}

static double norm2(const double* a, int n) {
    return sqrt(dot(a, a, n));
}

static void axpy(double alpha, const double* x, double* y, int n) {
    for (int i = 0; i < n; i++) y[i] += alpha * x[i];
}

static void scale(double alpha, double* x, int n) {
    for (int i = 0; i < n; i++) x[i] *= alpha;
}

int gmres(
    std::function<void(const double*, double*)> matvec,
    const double* b,
    double* x,
    int n,
    int restart,
    double tol,
    int max_iter)
{
    int m = std::min(restart, n);

    std::vector<double> r(n), w(n);
    // V: Krylov basis vectors, stored column-major V[j*n + i]
    std::vector<double> V((m+1) * n);
    // H: upper Hessenberg matrix, (m+1) x m
    std::vector<double> H((m+1) * m, 0.0);
    // Givens rotations
    std::vector<double> cs(m), sn(m), g(m+1);

    double bnorm = norm2(b, n);
    if (bnorm < 1e-30) {
        std::memset(x, 0, n * sizeof(double));
        return 0;
    }

    int total_iter = 0;

    for (int outer = 0; outer < max_iter / m + 1; outer++) {
        // r = b - A*x
        matvec(x, r.data());
        for (int i = 0; i < n; i++) r[i] = b[i] - r[i];

        double rnorm = norm2(r.data(), n);
        if (rnorm / bnorm < tol) return total_iter;

        // v_0 = r / ||r||
        for (int i = 0; i < n; i++) V[i] = r[i] / rnorm;

        std::fill(g.begin(), g.end(), 0.0);
        g[0] = rnorm;
        std::fill(H.begin(), H.end(), 0.0);

        int j;
        for (j = 0; j < m; j++) {
            total_iter++;

            // w = A * v_j
            matvec(&V[j*n], w.data());

            // Arnoldi: modified Gram-Schmidt
            for (int i = 0; i <= j; i++) {
                H[i*m + j] = dot(&V[i*n], w.data(), n);
                axpy(-H[i*m + j], &V[i*n], w.data(), n);
            }
            H[(j+1)*m + j] = norm2(w.data(), n);

            if (std::abs(H[(j+1)*m + j]) < 1e-30) {
                // Lucky breakdown
                break;
            }

            for (int i = 0; i < n; i++) {
                V[(j+1)*n + i] = w[i] / H[(j+1)*m + j];
            }

            // Apply previous Givens rotations
            for (int i = 0; i < j; i++) {
                double temp = cs[i] * H[i*m+j] + sn[i] * H[(i+1)*m+j];
                H[(i+1)*m+j] = -sn[i] * H[i*m+j] + cs[i] * H[(i+1)*m+j];
                H[i*m+j] = temp;
            }

            // Compute new Givens rotation
            double a = H[j*m+j], bb = H[(j+1)*m+j];
            double r = sqrt(a*a + bb*bb);
            cs[j] = a / r;
            sn[j] = bb / r;

            H[j*m+j] = r;
            H[(j+1)*m+j] = 0.0;

            g[j+1] = -sn[j] * g[j];
            g[j] = cs[j] * g[j];

            if (std::abs(g[j+1]) / bnorm < tol) {
                j++;
                break;
            }
        }

        // Back-substitution
        int k = std::min(j, m);
        std::vector<double> y(k);
        for (int i = k-1; i >= 0; i--) {
            y[i] = g[i];
            for (int l = i+1; l < k; l++) {
                y[i] -= H[i*m+l] * y[l];
            }
            y[i] /= H[i*m+i];
        }

        // x = x + V * y
        for (int i = 0; i < k; i++) {
            axpy(y[i], &V[i*n], x, n);
        }

        if (std::abs(g[k]) / bnorm < tol) return total_iter;
    }

    return -1; // failed to converge
}

} // namespace solver
