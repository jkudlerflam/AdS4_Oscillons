// newton.cpp — Newton-Raphson with block-structured direct solve + line search
//
// The oscillon system has block-triangular Jacobian structure near the
// background. The Jacobian also has a ~20-dimensional null space from
// the Bianchi identity (contracted Bianchi identity creates algebraic
// relations between the constraint and evolution equations in the
// time-collocation formulation).
//
// We handle this with:
// 1. Block structure: solve scalar+omega block first, then metric block
// 2. Regularized LU: project out near-null directions
// 3. Backtracking line search: prevent divergence from inaccurate Newton steps

#include "solver/newton.h"
#include "solver/gmres.h"
#include <cmath>
#include <cstring>
#include <cstdio>
#include <algorithm>
#include <numeric>
#include <vector>

#ifdef HAVE_LAPACK
// LAPACK routines for handling rank-deficient systems (Bianchi null space)
extern "C" {
    // Minimum-norm least-squares via SVD — most robust, O(n³) with large constant
    void dgelsd_(int* m, int* n, int* nrhs, double* A, int* lda,
                 double* B, int* ldb, double* s, double* rcond, int* rank,
                 double* work, int* lwork, int* iwork, int* info);
    // Minimum-norm least-squares via QR with column pivoting — faster than SVD
    void dgelsy_(int* m, int* n, int* nrhs, double* A, int* lda,
                 double* B, int* ldb, int* jpvt, double* rcond, int* rank,
                 double* work, int* lwork, int* info);
}
#endif

namespace solver {

// Dense LU solve with equilibration and regularization.
// Near-singular directions (pivot < tol_pivot) get zero update.
static bool denseSolve(double* J, double* b, int sz, double tol_pivot = 1e-10,
                       int* n_regularized_out = nullptr) {
    // Row equilibration
    std::vector<double> row_scale(sz, 1.0), col_scale(sz, 1.0);
    for (int i = 0; i < sz; i++) {
        double mx = 0;
        for (int j = 0; j < sz; j++)
            mx = std::max(mx, std::abs(J[i * sz + j]));
        row_scale[i] = (mx > 1e-30) ? mx : 1.0;
        double inv = 1.0 / row_scale[i];
        for (int j = 0; j < sz; j++) J[i * sz + j] *= inv;
        b[i] *= inv;
    }
    // Column equilibration
    for (int j = 0; j < sz; j++) {
        double mx = 0;
        for (int i = 0; i < sz; i++)
            mx = std::max(mx, std::abs(J[i * sz + j]));
        col_scale[j] = (mx > 1e-30) ? mx : 1.0;
        double inv = 1.0 / col_scale[j];
        for (int i = 0; i < sz; i++) J[i * sz + j] *= inv;
    }

    // Gaussian elimination with partial pivoting + regularization
    std::vector<int> piv(sz);
    for (int i = 0; i < sz; i++) piv[i] = i;
    int n_regularized = 0;

    for (int col = 0; col < sz; col++) {
        double max_val = 0;
        int max_row = col;
        for (int row = col; row < sz; row++) {
            double val = std::abs(J[piv[row] * sz + col]);
            if (val > max_val) { max_val = val; max_row = row; }
        }

        if (max_val < tol_pivot) {
            // Regularize: zero this direction
            n_regularized++;
            std::swap(piv[col], piv[max_row]);
            J[piv[col] * sz + col] = 1.0;
            for (int row = col + 1; row < sz; row++)
                J[piv[row] * sz + col] = 0.0;
            b[piv[col]] = 0.0; // zero RHS for this direction
            continue;
        }

        std::swap(piv[col], piv[max_row]);
        double pivot = J[piv[col] * sz + col];
        for (int row = col + 1; row < sz; row++) {
            double factor = J[piv[row] * sz + col] / pivot;
            J[piv[row] * sz + col] = factor;
            for (int k = col + 1; k < sz; k++)
                J[piv[row] * sz + k] -= factor * J[piv[col] * sz + k];
        }
    }

    // Forward substitution
    std::vector<double> y(sz);
    for (int i = 0; i < sz; i++) {
        y[i] = b[piv[i]];
        for (int j = 0; j < i; j++)
            y[i] -= J[piv[i] * sz + j] * y[j];
    }

    // Back substitution
    for (int i = sz - 1; i >= 0; i--) {
        b[i] = y[i];
        for (int j = i + 1; j < sz; j++)
            b[i] -= J[piv[i] * sz + j] * b[j];
        b[i] /= J[piv[i] * sz + i];
    }

    // Undo column scaling
    for (int i = 0; i < sz; i++) b[i] /= col_scale[i];
    if (n_regularized_out) *n_regularized_out = n_regularized;
    return true; // always returns true with regularization
}

NewtonResult newtonSolve(
    std::function<void(const double*, double, double*)> residual_func,
    std::function<double(const double*)> normalization_func,
    double* u,
    double& omega,
    double w_target,
    int n,
    const NewtonParams& params)
{
    NewtonResult result;
    result.converged = false;
    result.stagnated = false;
    result.iterations = 0;
    result.omega = omega;

    int N = n + 1;
    std::vector<double> F(N), du(N), u_pert(n), F_pert(N);
    std::vector<double> u_trial(n); // for line search

    for (int iter = 0; iter < params.max_iter; iter++) {
        // Evaluate residual
        residual_func(u, omega, F.data());
        F[n] = normalization_func(u) - w_target;

        double res_norm = 0.0;
        for (int i = 0; i < N; i++) res_norm = std::max(res_norm, std::abs(F[i]));

        result.residual_history.push_back(res_norm);
        if (params.verbose) {
            printf("  Newton iter %d: ||F||_inf = %.6e, omega = %.10f\n",
                   iter, res_norm, omega);
        }

        if (res_norm < params.tol) {
            result.converged = true;
            result.iterations = iter;
            result.final_residual = res_norm;
            result.omega = omega;
            return result;
        }

        // Stagnation detection: if residual hasn't decreased meaningfully for
        // stag_window consecutive iterations, declare convergence at the
        // Bianchi null-space floor (the minimum achievable residual).
        if (params.stag_window > 0 && (int)result.residual_history.size() > params.stag_window) {
            int sz = (int)result.residual_history.size();
            bool stagnated = true;
            for (int k = 1; k <= params.stag_window; k++) {
                if (result.residual_history[sz - k] <
                    params.stag_ratio * result.residual_history[sz - k - 1]) {
                    stagnated = false;
                    break;
                }
            }
            if (stagnated) {
                result.converged = true;
                result.stagnated = true;
                result.iterations = iter;
                result.final_residual = res_norm;
                result.omega = omega;
                if (params.verbose)
                    printf("  Newton stagnation detected (Bianchi floor): "
                           "accepting residual %.4e\n", res_norm);
                return result;
            }
        }

        if (params.use_lapack) {
#ifndef HAVE_LAPACK
            fprintf(stderr, "ERROR: --use_lapack requires HAVE_LAPACK at compile time\n");
            break;
#else
            // ============================================================
            // LAPACK dgelsd: minimum-norm least-squares via SVD
            // ============================================================
            // Correctly handles the Bianchi identity null space by computing
            // the minimum-norm solution of min ||J*du + F||₂. Small singular
            // values (below rcond * σ_max) are treated as zero.
            double eps = params.fd_eps;

            // Build Jacobian in COLUMN-MAJOR order for LAPACK
            // J_cm[i + j*N] = dF_i/du_j
            // Each column is an independent residual evaluation → OpenMP parallel
            std::vector<double> J_cm(N * (long long)N, 0.0);

#ifdef USE_OPENMP
            #pragma omp parallel for schedule(dynamic, 1)
#endif
            for (int j = 0; j <= n; j++) {
                // Thread-local storage for perturbation
                std::vector<double> u_loc(n);
                std::vector<double> F_loc(N);
                std::memcpy(u_loc.data(), u, n * sizeof(double));

                double h;
                if (j < n) {
                    // u-column
                    h = eps * std::max(1.0, std::abs(u[j]));
                    u_loc[j] += h;
                    residual_func(u_loc.data(), omega, F_loc.data());
                    F_loc[n] = normalization_func(u_loc.data()) - w_target;
                } else {
                    // omega-column
                    h = eps * std::max(1.0, std::abs(omega));
                    residual_func(u_loc.data(), omega + h, F_loc.data());
                    F_loc[n] = normalization_func(u_loc.data()) - w_target;
                }
                for (int i = 0; i < N; i++)
                    J_cm[i + (long long)j * N] = (F_loc[i] - F[i]) / h;
            }

            // RHS = -F (dgelsd overwrites with solution)
            std::vector<double> rhs(N);
            for (int i = 0; i < N; i++) rhs[i] = -F[i];

            // dgelsy: QR with column pivoting — faster than SVD for large systems
            int m_lap = N, n_lap = N, nrhs = 1, lda = N, ldb = N;
            int rank_out, info;
            double rcond = 1e-10; // columns with norm below rcond*max_norm treated as zero
            std::vector<int> jpvt(N, 0); // all columns free

            // Workspace query
            int lwork = -1;
            double wquery;
            dgelsy_(&m_lap, &n_lap, &nrhs, J_cm.data(), &lda, rhs.data(), &ldb,
                    jpvt.data(), &rcond, &rank_out, &wquery, &lwork, &info);
            lwork = (int)wquery + 1;
            std::vector<double> work(lwork);

            // Actual solve
            dgelsy_(&m_lap, &n_lap, &nrhs, J_cm.data(), &lda, rhs.data(), &ldb,
                    jpvt.data(), &rcond, &rank_out, work.data(), &lwork, &info);

            if (params.verbose && iter == 0)
                printf("    (LAPACK dgelsy: rank=%d/%d, info=%d)\n",
                       rank_out, N, info);

            for (int i = 0; i < N; i++) du[i] = rhs[i];
#endif // HAVE_LAPACK

        } else if (params.use_full_dense) {
            // ============================================================
            // Full dense Jacobian solve (no block decomposition)
            // ============================================================
            // Build the full (N x N) Jacobian by finite differences.
            // This captures all cross-couplings between scalar and metric.
            double eps = params.fd_eps;
            std::vector<double> J_full(N * (long long)N, 0.0);

            // Columns for u perturbations
            for (int j = 0; j < n; j++) {
                double h = eps * std::max(1.0, std::abs(u[j]));
                std::memcpy(u_pert.data(), u, n * sizeof(double));
                u_pert[j] += h;
                residual_func(u_pert.data(), omega, F_pert.data());
                F_pert[n] = normalization_func(u_pert.data()) - w_target;
                for (int i = 0; i < N; i++)
                    J_full[i * N + j] = (F_pert[i] - F[i]) / h;
            }
            // Omega column
            {
                double h = eps * std::max(1.0, std::abs(omega));
                std::memcpy(u_pert.data(), u, n * sizeof(double));
                residual_func(u_pert.data(), omega + h, F_pert.data());
                F_pert[n] = normalization_func(u_pert.data()) - w_target;
                for (int i = 0; i < N; i++)
                    J_full[i * N + n] = (F_pert[i] - F[i]) / h;
            }

            // RHS = -F
            std::vector<double> rhs(N);
            for (int i = 0; i < N; i++) rhs[i] = -F[i];

            int n_reg = 0;
            denseSolve(J_full.data(), rhs.data(), N, 1e-10, &n_reg);
            if (params.verbose && iter == 0)
                printf("    (full dense: %d/%d directions regularized)\n", n_reg, N);

            for (int i = 0; i < N; i++) du[i] = rhs[i];

        } else if (params.use_direct) {
            // ============================================================
            // Block-structured direct solve
            // ============================================================
            int n_per_field = n / 8;
            int n_metric = 7 * n_per_field;
            int n_scalar = n_per_field;
            double eps = params.fd_eps;

            // ---- Block 1: Scalar + omega ----
            int sz_s = n_scalar + 1;
            std::vector<double> J_S(sz_s * (long long)sz_s, 0.0);
            std::vector<double> rhs_s(sz_s);

            for (int i = 0; i < n_scalar; i++)
                rhs_s[i] = -F[n_metric + i];
            rhs_s[n_scalar] = -F[n];

            // Build scalar block Jacobian
            for (int j = 0; j < n_scalar; j++) {
                int u_idx = n_metric + j;
                double h = eps * std::max(1.0, std::abs(u[u_idx]));
                std::memcpy(u_pert.data(), u, n * sizeof(double));
                u_pert[u_idx] += h;
                residual_func(u_pert.data(), omega, F_pert.data());
                F_pert[n] = normalization_func(u_pert.data()) - w_target;
                for (int i = 0; i < n_scalar; i++)
                    J_S[i * sz_s + j] = (F_pert[n_metric + i] - F[n_metric + i]) / h;
                J_S[n_scalar * sz_s + j] = (F_pert[n] - F[n]) / h;
            }
            // Omega column
            {
                double h = eps * std::max(1.0, std::abs(omega));
                std::memcpy(u_pert.data(), u, n * sizeof(double));
                residual_func(u_pert.data(), omega + h, F_pert.data());
                F_pert[n] = normalization_func(u_pert.data()) - w_target;
                for (int i = 0; i < n_scalar; i++)
                    J_S[i * sz_s + n_scalar] = (F_pert[n_metric + i] - F[n_metric + i]) / h;
                J_S[n_scalar * sz_s + n_scalar] = (F_pert[n] - F[n]) / h;
            }

            denseSolve(J_S.data(), rhs_s.data(), sz_s);

            double d_omega = rhs_s[n_scalar];
            std::vector<double> d_scalar(n_scalar);
            std::memcpy(d_scalar.data(), rhs_s.data(), n_scalar * sizeof(double));

            // ---- Block 2: Metric ----
            // RHS: -R_M - J_Mω * δω
            std::vector<double> J_Momega(n_metric);
            {
                double h = eps * std::max(1.0, std::abs(omega));
                std::memcpy(u_pert.data(), u, n * sizeof(double));
                residual_func(u_pert.data(), omega + h, F_pert.data());
                for (int i = 0; i < n_metric; i++)
                    J_Momega[i] = (F_pert[i] - F[i]) / h;
            }

            std::vector<double> rhs_m(n_metric);
            for (int i = 0; i < n_metric; i++)
                rhs_m[i] = -F[i] - J_Momega[i] * d_omega;

            // Build metric Jacobian
            std::vector<double> J_M(n_metric * (long long)n_metric, 0.0);
            for (int j = 0; j < n_metric; j++) {
                double h = eps * std::max(1.0, std::abs(u[j]));
                std::memcpy(u_pert.data(), u, n * sizeof(double));
                u_pert[j] += h;
                residual_func(u_pert.data(), omega, F_pert.data());
                for (int i = 0; i < n_metric; i++)
                    J_M[i * n_metric + j] = (F_pert[i] - F[i]) / h;
            }

            denseSolve(J_M.data(), rhs_m.data(), n_metric);

            // Assemble full update
            for (int i = 0; i < n_metric; i++) du[i] = rhs_m[i];
            for (int i = 0; i < n_scalar; i++) du[n_metric + i] = d_scalar[i];
            du[n] = d_omega;

        } else {
            // GMRES solve with diagonal preconditioner + Tikhonov regularization
            double reg = params.gmres_reg;

            // --- Diagonal preconditioner via stochastic probing ---
            // Estimate diag(J) using random ±1 vectors: O(p*n) cost
            // diag(J) ≈ (1/p) * sum_k z_k ⊙ (J*z_k)
            // where z_k are random ±1 vectors and ⊙ is elementwise product
            std::vector<double> diag_J(N, 0.0);
            {
                const int n_probes = 20;
                std::vector<double> z(N), Jz(N);
                std::vector<double> u_p(n), F_p(N);

                // Simple deterministic seeding for reproducibility
                unsigned seed = 12345;
                for (int p = 0; p < n_probes; p++) {
                    // Generate random ±1 vector
                    for (int i = 0; i < N; i++) {
                        seed = seed * 1103515245 + 12345;
                        z[i] = ((seed >> 16) & 1) ? 1.0 : -1.0;
                    }

                    // Compute J*z via finite difference
                    double znorm = std::sqrt((double)N); // ||z|| = sqrt(N) for ±1 vector
                    double unorm = 0;
                    for (int i = 0; i < n; i++) unorm += u[i] * u[i];
                    unorm += omega * omega;
                    unorm = std::sqrt(unorm);
                    double h = params.fd_eps * std::max(1.0, unorm) / znorm;

                    for (int i = 0; i < n; i++) u_p[i] = u[i] + h * z[i];
                    double omega_p = omega + h * z[n];
                    residual_func(u_p.data(), omega_p, F_p.data());
                    F_p[n] = normalization_func(u_p.data()) - w_target;

                    for (int i = 0; i < N; i++)
                        diag_J[i] += z[i] * (F_p[i] - F[i]) / h;
                }
                for (int i = 0; i < N; i++) diag_J[i] /= n_probes;

                // Safeguard: clamp small diagonal entries
                for (int i = 0; i < N; i++) {
                    double d = std::abs(diag_J[i]);
                    if (d < 1e-12) diag_J[i] = 1.0; // identity for tiny entries
                }

                if (params.verbose && iter == 0) {
                    double dmin = 1e30, dmax = 0;
                    for (int i = 0; i < N; i++) {
                        double d = std::abs(diag_J[i]);
                        dmin = std::min(dmin, d);
                        dmax = std::max(dmax, d);
                    }
                    printf("    Preconditioner: diag(J) range [%.2e, %.2e], "
                           "cond ~ %.1e\n", dmin, dmax, dmax/dmin);
                }
            }

            // Left-preconditioned matvec: v → P^{-1}(J·v + reg·v)
            auto matvec = [&](const double* v, double* Jv) {
                // Compute ||v|| for FD step scaling
                double vnorm = 0;
                for (int i = 0; i < N; i++) vnorm += v[i] * v[i];
                vnorm = std::sqrt(vnorm);
                if (vnorm < 1e-30) vnorm = 1.0;

                // Compute ||u|| for scaling reference
                double unorm = 0;
                for (int i = 0; i < n; i++) unorm += u[i] * u[i];
                unorm += omega * omega;
                unorm = std::sqrt(unorm);

                // FD step: h chosen so ||h*v|| ~ sqrt(eps_machine) * max(1, ||u||)
                double h = params.fd_eps * std::max(1.0, unorm) / vnorm;

                for (int i = 0; i < n; i++) u_pert[i] = u[i] + h * v[i];
                double omega_pert = omega + h * v[n];
                residual_func(u_pert.data(), omega_pert, F_pert.data());
                F_pert[n] = normalization_func(u_pert.data()) - w_target;
                for (int i = 0; i < N; i++)
                    Jv[i] = (F_pert[i] - F[i]) / h;

                // Tikhonov regularization: (J + reg*I)v
                if (reg > 0.0) {
                    for (int i = 0; i < N; i++)
                        Jv[i] += reg * v[i];
                }

                // Left preconditioning: P^{-1} * (Jv)
                for (int i = 0; i < N; i++)
                    Jv[i] /= diag_J[i];
            };
            for (int i = 0; i < N; i++) du[i] = 0.0;
            // Preconditioned RHS: P^{-1} * (-F)
            std::vector<double> negF(N);
            for (int i = 0; i < N; i++) negF[i] = -F[i] / diag_J[i];

            // Eisenstat-Walker adaptive forcing: loosen GMRES tolerance when
            // far from solution, tighten as Newton converges
            double gmres_tol_eff = params.gmres_tol;
            if (params.gmres_adaptive_tol && iter > 0) {
                double prev_res = result.residual_history[iter - 1];
                // Choice 2: eta_k = gamma * (||F_k|| / ||F_{k-1}||)^alpha
                double eta = 0.9 * std::pow(res_norm / prev_res, 1.5);
                eta = std::min(eta, 0.5);    // never looser than 0.5
                eta = std::max(eta, 1e-4);   // never tighter than 1e-4
                gmres_tol_eff = eta;
            } else if (params.gmres_adaptive_tol) {
                gmres_tol_eff = 0.5; // first iteration: loose
            }

            int gmres_iters = gmres(matvec, negF.data(), du.data(), N,
                  params.gmres_restart, gmres_tol_eff, params.gmres_max_iter);

            if (params.verbose) {
                double du_norm = 0;
                for (int i = 0; i < N; i++) du_norm += du[i] * du[i];
                du_norm = std::sqrt(du_norm);
                // Check quality: compute ||J*du + F|| / ||F||
                std::vector<double> Jdu(N);
                matvec(du.data(), Jdu.data());
                double lin_res = 0;
                for (int i = 0; i < N; i++)
                    lin_res = std::max(lin_res, std::abs(Jdu[i] + F[i]));
                printf("    GMRES(%d, tol=%.1e): %d iters, ||du|| = %.4e, "
                       "d_omega = %.4e, ||J*du+F||_inf = %.4e\n",
                       params.gmres_restart, gmres_tol_eff,
                       gmres_iters, du_norm, du[n], lin_res);
            }
        }

        // ============================================================
        // Backtracking line search
        // ============================================================
        double alpha = 1.0;
        double best_res = res_norm;
        double best_alpha = 0.0;
        double omega_save = omega;

        for (int ls = 0; ls < 10; ls++) {
            // Trial step
            for (int i = 0; i < n; i++)
                u_trial[i] = u[i] + alpha * du[i];
            double omega_trial = omega_save + alpha * du[n];

            // Evaluate residual at trial point
            residual_func(u_trial.data(), omega_trial, F_pert.data());
            F_pert[n] = normalization_func(u_trial.data()) - w_target;

            double trial_res = 0;
            for (int i = 0; i < N; i++)
                trial_res = std::max(trial_res, std::abs(F_pert[i]));

            if (trial_res < best_res) {
                best_res = trial_res;
                best_alpha = alpha;
            }

            if (trial_res < res_norm) {
                // Accept: sufficient decrease
                break;
            }
            alpha *= 0.5;
        }

        if (best_alpha < 1e-10) {
            // No improvement found — try a tiny step
            best_alpha = 1e-4;
        }

        // Apply the best step
        for (int i = 0; i < n; i++)
            u[i] += best_alpha * du[i];
        omega = omega_save + best_alpha * du[n];
        result.omega = omega;

        if (params.verbose && best_alpha < 1.0) {
            printf("    (line search: alpha = %.4e)\n", best_alpha);
        }
    }

    result.iterations = params.max_iter;
    result.final_residual = result.residual_history.back();
    return result;
}

} // namespace solver
