// oscillon_driver.cpp — Amplitude continuation driver
//
// Two continuation strategies:
//
// 1. Fixed-w stepping (original): parameterize by w = phi_hat(tau=0, ref).
//    Simple but fails at folds.
//
// 2. Pseudo-arclength continuation: parameterize by arc length along the
//    solution branch. The Newton system is augmented with the constraint
//        dot · (x - x_pred) = 0
//    where dot is the tangent vector and x_pred is the predictor (linear
//    extrapolation). This navigates folds in any variable.
//
// Reference: Keller (1977), "Numerical Solution of Bifurcation and
// Nonlinear Eigenvalue Problems," in Applications of Bifurcation Theory.

#include "solver/oscillon_driver.h"
#include <cstdio>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <numeric>

#ifdef HAVE_LAPACK
extern "C" {
    void dgelsy_(int* m, int* n, int* nrhs, double* A, int* lda,
                 double* B, int* ldb, int* jpvt, double* rcond, int* rank,
                 double* work, int* lwork, int* info);
}
#endif

namespace solver {

OscillonDriver::OscillonDriver(const OscillonParams& params)
    : params_(params), sys_(params)
{}

// ============================================================================
// Solve at a single amplitude
// ============================================================================

bool OscillonDriver::solveAtEpsilon(
    double epsilon, NewtonResult& result,
    std::vector<double>& u_out, double& omega_out,
    const NewtonParams& np)
{
    int n = sys_.stateSize() - 1;
    u_out.resize(n);
    sys_.setLinearSeed(u_out.data(), omega_out, epsilon);

    double w_target = sys_.normalization(u_out.data());

    result = solveAtW(u_out.data(), omega_out, w_target, np);
    return result.converged;
}

// ============================================================================
// Solve at a given normalization target
// ============================================================================

NewtonResult OscillonDriver::solveAtW(
    double* u, double& omega, double w_target,
    const NewtonParams& np)
{
    int n = sys_.stateSize() - 1;

    auto residual_func = [this](const double* u_in, double om, double* R) {
        sys_.computeResidual(u_in, om, R);
    };

    auto norm_func = [this](const double* u_in) -> double {
        return sys_.normalization(u_in);
    };

    return newtonSolve(residual_func, norm_func,
                       u, omega, w_target, n, np);
}

// ============================================================================
// Newton solve with arclength constraint
// ============================================================================
//
// Solves the augmented system:
//   F_i(u, omega) = 0,   i = 0..n-1    (physics residuals)
//   dot_u · (u - u_pred) + dot_omega * (omega - omega_pred) = 0  (arclength)
//
// The arclength constraint replaces the normalization constraint.
// The Jacobian's last row is [dot_u, dot_omega] instead of [dw/du, 0].

NewtonResult OscillonDriver::solveArclength(
    double* u, double& omega,
    const double* u_pred, double omega_pred,
    const double* dot_u, double dot_omega,
    const NewtonParams& np)
{
    int n = sys_.stateSize() - 1;
    int N = n + 1; // augmented size

    NewtonResult result;
    result.converged = false;
    result.stagnated = false;
    result.iterations = 0;
    result.omega = omega;

    std::vector<double> F(N), du(N);
    std::vector<double> u_trial(n);

    for (int iter = 0; iter < np.max_iter; iter++) {
        // Evaluate physics residuals
        sys_.computeResidual(u, omega, F.data());

        // Arclength constraint: dot · (x - x_pred) = 0
        double arc_res = dot_omega * (omega - omega_pred);
        for (int i = 0; i < n; i++)
            arc_res += dot_u[i] * (u[i] - u_pred[i]);
        F[n] = arc_res;

        double res_norm = 0.0;
        for (int i = 0; i < N; i++)
            res_norm = std::max(res_norm, std::abs(F[i]));

        result.residual_history.push_back(res_norm);
        if (np.verbose) {
            printf("  Newton iter %d: ||F||_inf = %.6e, omega = %.10f\n",
                   iter, res_norm, omega);
        }

        if (res_norm < np.tol) {
            result.converged = true;
            result.iterations = iter;
            result.final_residual = res_norm;
            result.omega = omega;
            return result;
        }

        // Stagnation detection
        if (np.stag_window > 0 && (int)result.residual_history.size() > np.stag_window) {
            int sz = (int)result.residual_history.size();
            bool stag = true;
            for (int k = 1; k <= np.stag_window; k++) {
                if (result.residual_history[sz - k] <
                    np.stag_ratio * result.residual_history[sz - k - 1]) {
                    stag = false;
                    break;
                }
            }
            if (stag) {
                result.converged = true;
                result.stagnated = true;
                result.iterations = iter;
                result.final_residual = res_norm;
                result.omega = omega;
                if (np.verbose)
                    printf("  Newton stagnation (Bianchi floor): accepting %.4e\n",
                           res_norm);
                return result;
            }
        }

        // Build Jacobian and solve (LAPACK path only for now)
#ifndef HAVE_LAPACK
        fprintf(stderr, "ERROR: arclength continuation requires HAVE_LAPACK\n");
        break;
#else
        double eps = np.fd_eps;

        // Build Jacobian in column-major for LAPACK
        std::vector<double> J_cm(N * (long long)N, 0.0);

#ifdef USE_OPENMP
        #pragma omp parallel for schedule(dynamic, 1)
#endif
        for (int j = 0; j <= n; j++) {
            std::vector<double> u_loc(n);
            std::vector<double> F_loc(N);
            std::memcpy(u_loc.data(), u, n * sizeof(double));

            double h;
            if (j < n) {
                h = eps * std::max(1.0, std::abs(u[j]));
                u_loc[j] += h;
                sys_.computeResidual(u_loc.data(), omega, F_loc.data());
                // Arclength constraint derivative w.r.t. u[j] = dot_u[j]
                double arc_pert = dot_omega * (omega - omega_pred);
                for (int i = 0; i < n; i++)
                    arc_pert += dot_u[i] * (u_loc[i] - u_pred[i]);
                F_loc[n] = arc_pert;
            } else {
                h = eps * std::max(1.0, std::abs(omega));
                sys_.computeResidual(u_loc.data(), omega + h, F_loc.data());
                // Arclength constraint derivative w.r.t. omega = dot_omega
                double arc_pert = dot_omega * (omega + h - omega_pred);
                for (int i = 0; i < n; i++)
                    arc_pert += dot_u[i] * (u[i] - u_pred[i]);
                F_loc[n] = arc_pert;
            }

            for (int i = 0; i < N; i++)
                J_cm[i + (long long)j * N] = (F_loc[i] - F[i]) / h;
        }

        // Solve J * du = -F via LAPACK dgelsy
        std::vector<double> rhs(N);
        for (int i = 0; i < N; i++) rhs[i] = -F[i];

        {
            int m_lap = N, n_lap = N, nrhs = 1, lda = N, ldb = N;
            int rank_out, info;
            double rcond = 1e-10;
            std::vector<int> jpvt(N, 0);

            int lwork = -1;
            double wquery;
            dgelsy_(&m_lap, &n_lap, &nrhs, J_cm.data(), &lda, rhs.data(), &ldb,
                    jpvt.data(), &rcond, &rank_out, &wquery, &lwork, &info);
            lwork = (int)wquery + 1;
            std::vector<double> work(lwork);

            dgelsy_(&m_lap, &n_lap, &nrhs, J_cm.data(), &lda, rhs.data(), &ldb,
                    jpvt.data(), &rcond, &rank_out, work.data(), &lwork, &info);

            if (np.verbose && iter == 0)
                printf("    (LAPACK dgelsy: rank=%d/%d)\n", rank_out, N);
        }

        for (int i = 0; i < N; i++) du[i] = rhs[i];
#endif

        // Backtracking line search
        double alpha = 1.0;
        double best_res = res_norm;
        double best_alpha = 0.0;
        double omega_save = omega;

        for (int ls = 0; ls < 10; ls++) {
            for (int i = 0; i < n; i++)
                u_trial[i] = u[i] + alpha * du[i];
            double omega_trial = omega_save + alpha * du[n];

            std::vector<double> F_trial(N);
            sys_.computeResidual(u_trial.data(), omega_trial, F_trial.data());
            double arc_trial = dot_omega * (omega_trial - omega_pred);
            for (int i = 0; i < n; i++)
                arc_trial += dot_u[i] * (u_trial[i] - u_pred[i]);
            F_trial[n] = arc_trial;

            double trial_res = 0;
            for (int i = 0; i < N; i++)
                trial_res = std::max(trial_res, std::abs(F_trial[i]));

            if (trial_res < best_res) {
                best_res = trial_res;
                best_alpha = alpha;
            }
            if (trial_res < res_norm) break;
            alpha *= 0.5;
        }

        if (best_alpha < 1e-10) best_alpha = 1e-4;

        for (int i = 0; i < n; i++)
            u[i] += best_alpha * du[i];
        omega = omega_save + best_alpha * du[n];
        result.omega = omega;

        if (np.verbose && best_alpha < 1.0)
            printf("    (line search: alpha = %.4e)\n", best_alpha);
    }

    result.iterations = np.max_iter;
    result.final_residual = result.residual_history.back();
    return result;
}

// ============================================================================
// Pseudo-arclength continuation
// ============================================================================

std::vector<BranchPoint> OscillonDriver::runArclengthContinuation(
    const ArclengthParams& ap, BranchCallback cb)
{
    std::vector<BranchPoint> branch;
    int n = sys_.stateSize() - 1;

    if (ap.verbose) {
        printf("========================================\n");
        printf("Pseudo-arclength continuation driver\n");
        printf("  ell = %d, Delta = %.1f, Lambda = %.1f\n",
               params_.ell, params_.Delta, params_.Lambda);
        printf("  N_t = %d, N_nuc = %d, N_shell = %d, N_theta = %d\n",
               params_.N_t, params_.N_nuc, params_.N_shell, params_.N_theta);
        printf("  State size: %d\n", n + 1);
        printf("  eps_start = %.4e, eps_second = %.4e\n",
               ap.eps_start, ap.eps_second);
        printf("  ds_initial = %.4e, ds_min = %.4e, ds_max = %.4e\n",
               ap.ds_initial, ap.ds_min, ap.ds_max);
        printf("========================================\n\n");
    }

    // ---- Bootstrap: get two points via fixed-w to establish tangent ----

    // Point 0
    std::vector<double> u0(n), u1(n);
    double omega0, omega1;
    NewtonResult res0, res1;

    if (!solveAtEpsilon(ap.eps_start, res0, u0, omega0, ap.newton)) {
        if (ap.verbose)
            printf("FAILED: Newton did not converge at eps_start = %.4e\n",
                   ap.eps_start);
        return branch;
    }
    double w0 = sys_.normalization(u0.data());

    if (ap.verbose)
        printf("Bootstrap point 0: w = %.6e, omega = %.10f, res = %.2e\n",
               w0, omega0, res0.final_residual);

    {
        BranchPoint bp;
        bp.w = w0; bp.omega = omega0;
        bp.residual = res0.final_residual;
        bp.newton_iters = res0.iterations;
        if (ap.store_solutions) bp.u.assign(u0.begin(), u0.end());
        branch.push_back(bp);
        if (cb) cb(0, bp, u0.data(), sys_);
    }

    // Point 1
    if (!solveAtEpsilon(ap.eps_second, res1, u1, omega1, ap.newton)) {
        if (ap.verbose)
            printf("FAILED: Newton did not converge at eps_second = %.4e\n",
                   ap.eps_second);
        return branch;
    }
    double w1 = sys_.normalization(u1.data());

    if (ap.verbose)
        printf("Bootstrap point 1: w = %.6e, omega = %.10f, res = %.2e\n",
               w1, omega1, res1.final_residual);

    {
        BranchPoint bp;
        bp.w = w1; bp.omega = omega1;
        bp.residual = res1.final_residual;
        bp.newton_iters = res1.iterations;
        if (ap.store_solutions) bp.u.assign(u1.begin(), u1.end());
        branch.push_back(bp);
        if (cb) cb(1, bp, u1.data(), sys_);
    }

    // ---- Arclength continuation loop ----
    // Now we have (u0, omega0) and (u1, omega1). Compute secant tangent.

    // Call the shared implementation
    auto rest = resumeArclengthContinuation(
        u0.data(), omega0, w0,
        u1.data(), omega1, w1,
        ap, cb);

    // Append results (skip first two since resumeArclength records the start point)
    for (size_t i = 1; i < rest.size(); i++)
        branch.push_back(rest[i]);

    return branch;
}

std::vector<BranchPoint> OscillonDriver::resumeArclengthContinuation(
    const double* u0_in, double omega0, double w0,
    const double* u1_in, double omega1, double w1,
    const ArclengthParams& ap, BranchCallback cb)
{
    std::vector<BranchPoint> branch;
    int n = sys_.stateSize() - 1;

    std::vector<double> u_prev(u0_in, u0_in + n);
    std::vector<double> u_curr(u1_in, u1_in + n);
    double omega_prev = omega0;
    double omega_curr = omega1;

    // Record starting point
    {
        BranchPoint bp;
        bp.w = w1; bp.omega = omega1; bp.residual = 0; bp.newton_iters = 0;
        branch.push_back(bp);
    }

    // Compute initial secant tangent and normalize
    std::vector<double> dot_u(n);
    double dot_omega;
    {
        double norm_sq = 0;
        for (int i = 0; i < n; i++) {
            dot_u[i] = u_curr[i] - u_prev[i];
            norm_sq += dot_u[i] * dot_u[i];
        }
        dot_omega = omega_curr - omega_prev;
        norm_sq += dot_omega * dot_omega;
        double norm = std::sqrt(norm_sq);
        if (norm < 1e-30) {
            if (ap.verbose)
                printf("ERROR: zero tangent between bootstrap points\n");
            return branch;
        }
        for (int i = 0; i < n; i++) dot_u[i] /= norm;
        dot_omega /= norm;
    }

    double ds = ap.ds_initial;
    int fold_count = 0;    // counts points after fold detection
    bool fold_detected = false;
    double E_prev = 0;     // for fold detection via dE/ds sign change
    int branch_offset = (int)branch.size() - 1; // offset for callback indexing

    // Backup vectors for retry
    std::vector<double> u_backup(n), dot_u_backup(n);

    if (ap.verbose) {
        printf("\n--- Starting arclength continuation ---\n");
        printf("  Initial tangent: dot_omega = %.6e\n", dot_omega);
    }

    while ((int)branch.size() < ap.max_branch_points + 1) {
        // Save backup
        std::copy(u_curr.begin(), u_curr.end(), u_backup.begin());
        double omega_backup = omega_curr;
        std::copy(dot_u.begin(), dot_u.end(), dot_u_backup.begin());
        double dot_omega_backup = dot_omega;

        // Predictor: linear extrapolation along tangent
        std::vector<double> u_pred(n);
        for (int i = 0; i < n; i++)
            u_pred[i] = u_curr[i] + ds * dot_u[i];
        double omega_pred = omega_curr + ds * dot_omega;

        // Initial guess = predictor
        std::vector<double> u_next(u_pred);
        double omega_next = omega_pred;

        if (ap.verbose) {
            printf("\n--- Step %d: ds = %.4e, predicted omega = %.10f ---\n",
                   (int)branch.size(), ds, omega_pred);
        }

        // Corrector: Newton with arclength constraint
        NewtonResult result = solveArclength(
            u_next.data(), omega_next,
            u_pred.data(), omega_pred,
            dot_u.data(), dot_omega,
            ap.newton);

        if (result.converged && result.final_residual < ap.residual_max) {
            double w_next = sys_.normalization(u_next.data());

            BranchPoint bp;
            bp.w = w_next;
            bp.omega = omega_next;
            bp.residual = result.final_residual;
            bp.newton_iters = result.iterations;
            if (ap.store_solutions)
                bp.u.assign(u_next.begin(), u_next.end());
            branch.push_back(bp);

            if (ap.verbose) {
                printf("Branch point %d: w = %.6e, omega = %.10f, "
                       "res = %.2e, iters = %d\n",
                       (int)branch.size(), bp.w, bp.omega,
                       bp.residual, bp.newton_iters);
            }

            if (cb) cb(branch_offset + (int)branch.size() - 1, bp,
                       u_next.data(), sys_);

            // Update tangent: secant between current and new point
            {
                double norm_sq = 0;
                for (int i = 0; i < n; i++) {
                    dot_u[i] = u_next[i] - u_curr[i];
                    norm_sq += dot_u[i] * dot_u[i];
                }
                dot_omega = omega_next - omega_curr;
                norm_sq += dot_omega * dot_omega;
                double norm = std::sqrt(norm_sq);
                if (norm > 1e-30) {
                    for (int i = 0; i < n; i++) dot_u[i] /= norm;
                    dot_omega /= norm;
                }
            }

            // Shift: prev <- curr, curr <- next
            std::copy(u_curr.begin(), u_curr.end(), u_prev.begin());
            omega_prev = omega_curr;
            std::copy(u_next.begin(), u_next.end(), u_curr.begin());
            omega_curr = omega_next;

            // Adaptive step
            if (result.iterations <= ap.fast_iters)
                ds = std::min(ds * ap.step_increase, ap.ds_max);
            else if (result.iterations >= ap.slow_iters)
                ds = std::max(ds * ap.step_decrease, ap.ds_min);

            // Fold detection: check if E changed sign of slope
            // (We get E from the callback via diagnostics, but here we use
            // a simple proxy: the energy is ~ w^2 * E_2 at small amplitude,
            // so we track w as a fold indicator. A more robust check uses
            // the actual E_scalar from diagnostics.)
            // For now, detect fold by checking if omega started increasing
            // (it should be decreasing along the first branch).
            if (branch.size() >= 3) {
                double omega_m2 = branch[branch.size()-3].omega;
                double omega_m1 = branch[branch.size()-2].omega;
                double omega_m0 = branch[branch.size()-1].omega;
                bool was_decreasing = (omega_m1 < omega_m2);
                bool now_increasing = (omega_m0 > omega_m1);
                if (was_decreasing && now_increasing && !fold_detected) {
                    fold_detected = true;
                    if (ap.verbose)
                        printf("*** FOLD DETECTED at omega = %.10f ***\n",
                               omega_m1);
                }
            }

            if (fold_detected) {
                fold_count++;
                if (ap.stop_after_fold && fold_count >= ap.points_after_fold) {
                    if (ap.verbose)
                        printf("Stopping: %d points past fold\n", fold_count);
                    break;
                }
            }

            // Termination checks
            if (omega_curr < ap.omega_min) {
                if (ap.verbose)
                    printf("Stopping: omega = %.6f < omega_min = %.6f\n",
                           omega_curr, ap.omega_min);
                break;
            }

        } else {
            // Newton failed or residual too large — restore and reduce step
            if (ap.verbose) {
                if (!result.converged)
                    printf("Newton failed, reducing ds\n");
                else
                    printf("Residual %.2e > %.2e, reducing ds\n",
                           result.final_residual, ap.residual_max);
            }

            // Restore
            std::copy(u_backup.begin(), u_backup.end(), u_curr.begin());
            omega_curr = omega_backup;
            std::copy(dot_u_backup.begin(), dot_u_backup.end(), dot_u.begin());
            dot_omega = dot_omega_backup;

            ds *= ap.step_decrease;
            if (ds < ap.ds_min) {
                if (ap.verbose)
                    printf("Step size below minimum (%.2e < %.2e), stopping.\n",
                           ds, ap.ds_min);
                break;
            }
        }
    }

    if (ap.verbose) {
        printf("\n========================================\n");
        printf("Arclength continuation complete: %d branch points\n",
               (int)branch.size());
        if (!branch.empty()) {
            printf("  w range: [%.6e, %.6e]\n",
                   branch.front().w, branch.back().w);
            printf("  omega range: [%.10f, %.10f]\n",
                   branch.back().omega, branch.front().omega);
        }
        if (fold_detected)
            printf("  Fold detected: YES (%d points past fold)\n", fold_count);
        else
            printf("  Fold detected: NO\n");
        printf("========================================\n");
    }

    return branch;
}

// ============================================================================
// Fixed-w continuation (original, with convergence guard added)
// ============================================================================

std::vector<BranchPoint> OscillonDriver::runContinuation(
    const ContinuationParams& cont, BranchCallback cb)
{
    std::vector<BranchPoint> branch;
    int n = sys_.stateSize() - 1;

    if (cont.verbose) {
        printf("========================================\n");
        printf("Oscillon continuation driver (fixed-w)\n");
        printf("  ell = %d, Delta = %.1f, Lambda = %.1f\n",
               params_.ell, params_.Delta, params_.Lambda);
        printf("  N_t = %d, N_nuc = %d, N_shell = %d, N_theta = %d\n",
               params_.N_t, params_.N_nuc, params_.N_shell, params_.N_theta);
        printf("  eps_start = %.4e, dw_initial = %.4e, w_max = %.4e\n",
               cont.eps_start, cont.dw_initial, cont.w_max);
        printf("========================================\n\n");
    }

    std::vector<double> u(n);
    double omega;
    sys_.setLinearSeed(u.data(), omega, cont.eps_start);
    double w_current = sys_.normalization(u.data());

    if (cont.verbose) {
        printf("Initial seed: eps = %.4e, w = %.6e, omega = %.10f\n",
               cont.eps_start, w_current, omega);
    }

    NewtonResult result = solveAtW(u.data(), omega, w_current, cont.newton);

    if (!result.converged) {
        if (cont.verbose)
            printf("FAILED: Newton did not converge at initial amplitude.\n");
        return branch;
    }

    {
        BranchPoint bp;
        bp.w = w_current;
        bp.omega = omega;
        bp.residual = result.final_residual;
        bp.newton_iters = result.iterations;
        if (cont.store_solutions) bp.u = std::vector<double>(u.begin(), u.end());
        branch.push_back(bp);

        if (cont.verbose) {
            printf("\nBranch point %d: w = %.6e, omega = %.10f, "
                   "res = %.2e, iters = %d\n",
                   (int)branch.size(), bp.w, bp.omega,
                   bp.residual, bp.newton_iters);
        }
        if (cb) cb((int)branch.size() - 1, bp, u.data(), sys_);
    }

    double dw = cont.dw_initial;
    std::vector<double> u_prev(n);

    while (w_current < cont.w_max &&
           (int)branch.size() < cont.max_branch_points)
    {
        std::copy(u.begin(), u.end(), u_prev.begin());
        double omega_prev = omega;
        double w_prev = w_current;

        double w_next = w_current + dw;

        if (cont.verbose) {
            printf("\n--- Attempting w = %.6e (dw = %.4e) ---\n", w_next, dw);
        }

        result = solveAtW(u.data(), omega, w_next, cont.newton);

        if (result.converged && result.final_residual < cont.residual_max) {
            w_current = w_next;

            BranchPoint bp;
            bp.w = w_current;
            bp.omega = omega;
            bp.residual = result.final_residual;
            bp.newton_iters = result.iterations;
            if (cont.store_solutions)
                bp.u = std::vector<double>(u.begin(), u.end());
            branch.push_back(bp);

            if (cont.verbose) {
                printf("Branch point %d: w = %.6e, omega = %.10f, "
                       "res = %.2e, iters = %d\n",
                       (int)branch.size(), bp.w, bp.omega,
                       bp.residual, bp.newton_iters);
            }
            if (cb) cb((int)branch.size() - 1, bp, u.data(), sys_);

            if (result.iterations <= cont.fast_iters) {
                dw = std::min(dw * cont.step_increase, cont.dw_max);
            } else if (result.iterations >= cont.slow_iters) {
                dw = std::max(dw * cont.step_decrease, cont.dw_min);
            }

        } else {
            if (cont.verbose) {
                if (!result.converged)
                    printf("Newton failed at w = %.6e, reducing step\n", w_next);
                else
                    printf("Residual %.2e > guard %.2e at w = %.6e, reducing step\n",
                           result.final_residual, cont.residual_max, w_next);
            }

            std::copy(u_prev.begin(), u_prev.end(), u.begin());
            omega = omega_prev;
            w_current = w_prev;

            dw *= cont.step_decrease;

            if (dw < cont.dw_min) {
                if (cont.verbose)
                    printf("Step size below minimum (%.2e < %.2e), stopping.\n",
                           dw, cont.dw_min);
                break;
            }
        }
    }

    if (cont.verbose) {
        printf("\n========================================\n");
        printf("Continuation complete: %d branch points\n", (int)branch.size());
        if (!branch.empty()) {
            printf("  w range: [%.6e, %.6e]\n", branch.front().w, branch.back().w);
            printf("  omega range: [%.10f, %.10f]\n",
                   branch.back().omega, branch.front().omega);
        }
        printf("========================================\n");
    }

    return branch;
}

// ============================================================================
// Resume fixed-w continuation from checkpoint
// ============================================================================

std::vector<BranchPoint> OscillonDriver::resumeContinuation(
    const double* u_init, double omega_init, double w_init,
    const ContinuationParams& cont, BranchCallback cb)
{
    std::vector<BranchPoint> branch;
    int n = sys_.stateSize() - 1;

    if (cont.verbose) {
        printf("========================================\n");
        printf("Resuming continuation from checkpoint\n");
        printf("  ell = %d, Delta = %.1f, Lambda = %.1f\n",
               params_.ell, params_.Delta, params_.Lambda);
        printf("  w_start = %.6e, omega = %.10f\n", w_init, omega_init);
        printf("  w_max = %.4e\n", cont.w_max);
        printf("========================================\n\n");
    }

    std::vector<double> u(u_init, u_init + n);
    double omega = omega_init;
    double w_current = w_init;

    {
        BranchPoint bp;
        bp.w = w_current;
        bp.omega = omega;
        bp.residual = 0;
        bp.newton_iters = 0;
        branch.push_back(bp);
        if (cb) cb(0, bp, u.data(), sys_);
    }

    double dw = cont.dw_initial;
    std::vector<double> u_prev(n);
    NewtonResult result;

    while (w_current < cont.w_max &&
           (int)branch.size() < cont.max_branch_points)
    {
        std::copy(u.begin(), u.end(), u_prev.begin());
        double omega_prev = omega;
        double w_prev = w_current;

        double w_next = w_current + dw;

        if (cont.verbose) {
            printf("\n--- Attempting w = %.6e (dw = %.4e) ---\n", w_next, dw);
        }

        result = solveAtW(u.data(), omega, w_next, cont.newton);

        if (result.converged && result.final_residual < cont.residual_max) {
            w_current = w_next;

            BranchPoint bp;
            bp.w = w_current;
            bp.omega = omega;
            bp.residual = result.final_residual;
            bp.newton_iters = result.iterations;
            branch.push_back(bp);

            if (cont.verbose) {
                printf("Branch point %d: w = %.6e, omega = %.10f, "
                       "res = %.2e, iters = %d\n",
                       (int)branch.size(), bp.w, bp.omega,
                       bp.residual, bp.newton_iters);
            }
            if (cb) cb((int)branch.size() - 1, bp, u.data(), sys_);

            if (result.iterations <= cont.fast_iters) {
                dw = std::min(dw * cont.step_increase, cont.dw_max);
            } else if (result.iterations >= cont.slow_iters) {
                dw = std::max(dw * cont.step_decrease, cont.dw_min);
            }

        } else {
            if (cont.verbose) {
                if (!result.converged)
                    printf("Newton failed at w = %.6e, reducing step\n", w_next);
                else
                    printf("Residual %.2e > guard %.2e, reducing step\n",
                           result.final_residual, cont.residual_max);
            }

            std::copy(u_prev.begin(), u_prev.end(), u.begin());
            omega = omega_prev;
            w_current = w_prev;

            dw *= cont.step_decrease;

            if (dw < cont.dw_min) {
                if (cont.verbose)
                    printf("Step size below minimum (%.2e < %.2e), stopping.\n",
                           dw, cont.dw_min);
                break;
            }
        }
    }

    if (cont.verbose) {
        printf("\n========================================\n");
        printf("Continuation complete: %d branch points\n", (int)branch.size());
        if (!branch.empty()) {
            printf("  w range: [%.6e, %.6e]\n", branch.front().w, branch.back().w);
            printf("  omega range: [%.10f, %.10f]\n",
                   branch.back().omega, branch.front().omega);
        }
        printf("========================================\n");
    }

    return branch;
}

// ============================================================================
// Checkpoint save/load (fixed-w format)
// ============================================================================

bool OscillonDriver::saveCheckpoint(const std::string& filename, int n,
                                     double omega, double w, const double* u)
{
    FILE* f = fopen(filename.c_str(), "wb");
    if (!f) return false;
    fwrite(&n, sizeof(int), 1, f);
    fwrite(&omega, sizeof(double), 1, f);
    fwrite(&w, sizeof(double), 1, f);
    fwrite(u, sizeof(double), n, f);
    fclose(f);
    return true;
}

bool OscillonDriver::loadCheckpoint(const std::string& filename, int& n,
                                     double& omega, double& w,
                                     std::vector<double>& u)
{
    FILE* f = fopen(filename.c_str(), "rb");
    if (!f) return false;
    if (fread(&n, sizeof(int), 1, f) != 1) { fclose(f); return false; }
    if (fread(&omega, sizeof(double), 1, f) != 1) { fclose(f); return false; }
    if (fread(&w, sizeof(double), 1, f) != 1) { fclose(f); return false; }
    u.resize(n);
    if ((int)fread(u.data(), sizeof(double), n, f) != n) { fclose(f); return false; }
    fclose(f);
    return true;
}

// ============================================================================
// Arclength checkpoint save/load (two consecutive points)
// ============================================================================

bool OscillonDriver::saveArclengthCheckpoint(
    const std::string& filename, int n,
    double omega0, double w0, const double* u0,
    double omega1, double w1, const double* u1)
{
    FILE* f = fopen(filename.c_str(), "wb");
    if (!f) return false;
    int magic = 0xA4C1;  // distinguish from fixed-w checkpoints
    fwrite(&magic, sizeof(int), 1, f);
    fwrite(&n, sizeof(int), 1, f);
    fwrite(&omega0, sizeof(double), 1, f);
    fwrite(&w0, sizeof(double), 1, f);
    fwrite(u0, sizeof(double), n, f);
    fwrite(&omega1, sizeof(double), 1, f);
    fwrite(&w1, sizeof(double), 1, f);
    fwrite(u1, sizeof(double), n, f);
    fclose(f);
    return true;
}

bool OscillonDriver::loadArclengthCheckpoint(
    const std::string& filename, int& n,
    double& omega0, double& w0, std::vector<double>& u0,
    double& omega1, double& w1, std::vector<double>& u1)
{
    FILE* f = fopen(filename.c_str(), "rb");
    if (!f) return false;
    int magic;
    if (fread(&magic, sizeof(int), 1, f) != 1 || magic != 0xA4C1)
        { fclose(f); return false; }
    if (fread(&n, sizeof(int), 1, f) != 1) { fclose(f); return false; }
    if (fread(&omega0, sizeof(double), 1, f) != 1) { fclose(f); return false; }
    if (fread(&w0, sizeof(double), 1, f) != 1) { fclose(f); return false; }
    u0.resize(n);
    if ((int)fread(u0.data(), sizeof(double), n, f) != n) { fclose(f); return false; }
    if (fread(&omega1, sizeof(double), 1, f) != 1) { fclose(f); return false; }
    if (fread(&w1, sizeof(double), 1, f) != 1) { fclose(f); return false; }
    u1.resize(n);
    if ((int)fread(u1.data(), sizeof(double), n, f) != n) { fclose(f); return false; }
    fclose(f);
    return true;
}

} // namespace solver
