// main.cpp — AdS4 Oscillon solver: production CLI
//
// Usage:
//   ./ads4osc [options]
//
// Modes:
//   --branch       Run amplitude continuation and output branch data (default)
//   --convergence  Run resolution convergence study
//   --single       Solve at a single epsilon value
//
// Parameters:
//   --ell N        Angular mode (default: 2)
//   --Delta D      Conformal dimension (default: 6.0)
//   --Lambda L     Cosmological constant (default: -3.0)
//   --N_nuc N      Chebyshev order, nucleus (default: 4)
//   --N_shell N    Chebyshev order, shell (default: 4)
//   --N_theta N    Legendre order (default: 2)
//   --N_t N        Fourier modes (default: 2)
//   --eps E        Amplitude parameter (default: 0.01)
//   --eps_start E  Starting amplitude for continuation (default: 0.01)
//   --w_max W      Max normalization for continuation (default: 0.1)
//   --tol T        Newton tolerance (default: 1e-8)
//   --output FILE  Output CSV filename (default: branch.csv or convergence.csv)
//   --verbose      Print detailed output
//   --quiet        Minimal output
//
// Examples:
//   ./ads4osc --branch --ell 2 --Delta 6 --N_nuc 6 --N_shell 6 --N_theta 4 --w_max 0.5
//   ./ads4osc --convergence --ell 2 --Delta 6 --eps 0.05
//   ./ads4osc --single --eps 0.05 --N_nuc 8 --N_shell 8 --N_theta 4

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <chrono>
#include "solver/oscillon_system.h"
#include "solver/oscillon_driver.h"
#include "solver/newton.h"
#include "diagnostics/diagnostics.h"
#include "diagnostics/charges.h"
#include "io/output.h"

#ifdef USE_OPENMP
#include <omp.h>
#endif

enum RunMode { MODE_BRANCH, MODE_ARCLENGTH, MODE_CONVERGENCE, MODE_SINGLE };

struct Config {
    RunMode mode = MODE_BRANCH;
    int ell = 2;
    double Delta = 6.0;
    double Lambda = -3.0;
    int N_nuc = 4;
    int N_shell = 4;
    int N_theta = 2;
    int N_t = 2;
    double eps = 0.05;
    double eps_start = 0.01;
    double eps_second = 0.02;  // second bootstrap amplitude for arclength
    double w_max = 1000.0;
    double ds_initial = 0.01;  // arclength step
    double ds_min = 1e-7;
    double ds_max = 0.5;
    double tol = 1e-8;
    double residual_max = 1e-4; // convergence guard
    int max_iter = 15;
    int max_branch_points = 2000;
    int points_after_fold = 20;
    std::string output;
    std::string restart;    // checkpoint file to resume from
    bool verbose = false;
    bool quiet = false;
    bool store_solutions = false;
    bool use_gmres = false;
    int gmres_restart = 100;
    double gmres_tol = 1e-4;
};

static Config parseArgs(int argc, char** argv) {
    Config cfg;
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--branch")      cfg.mode = MODE_BRANCH;
        else if (arg == "--arclength") cfg.mode = MODE_ARCLENGTH;
        else if (arg == "--convergence") cfg.mode = MODE_CONVERGENCE;
        else if (arg == "--single") cfg.mode = MODE_SINGLE;
        else if (arg == "--verbose") cfg.verbose = true;
        else if (arg == "--quiet")   cfg.quiet = true;
        else if (arg == "--store")   cfg.store_solutions = true;
        else if (arg == "--gmres")   cfg.use_gmres = true;
        else if (i + 1 < argc) {
            std::string val = argv[++i];
            if (arg == "--ell")      cfg.ell = atoi(val.c_str());
            else if (arg == "--Delta") cfg.Delta = atof(val.c_str());
            else if (arg == "--Lambda") cfg.Lambda = atof(val.c_str());
            else if (arg == "--N_nuc") cfg.N_nuc = atoi(val.c_str());
            else if (arg == "--N_shell") cfg.N_shell = atoi(val.c_str());
            else if (arg == "--N_theta") cfg.N_theta = atoi(val.c_str());
            else if (arg == "--N_t") cfg.N_t = atoi(val.c_str());
            else if (arg == "--eps") cfg.eps = atof(val.c_str());
            else if (arg == "--eps_start") cfg.eps_start = atof(val.c_str());
            else if (arg == "--eps_second") cfg.eps_second = atof(val.c_str());
            else if (arg == "--w_max") cfg.w_max = atof(val.c_str());
            else if (arg == "--ds_initial") cfg.ds_initial = atof(val.c_str());
            else if (arg == "--ds_min") cfg.ds_min = atof(val.c_str());
            else if (arg == "--ds_max") cfg.ds_max = atof(val.c_str());
            else if (arg == "--tol") cfg.tol = atof(val.c_str());
            else if (arg == "--residual_max") cfg.residual_max = atof(val.c_str());
            else if (arg == "--max_iter") cfg.max_iter = atoi(val.c_str());
            else if (arg == "--max_branch_points") cfg.max_branch_points = atoi(val.c_str());
            else if (arg == "--points_after_fold") cfg.points_after_fold = atoi(val.c_str());
            else if (arg == "--output") cfg.output = val;
            else if (arg == "--restart") cfg.restart = val;
            else if (arg == "--gmres_restart") cfg.gmres_restart = atoi(val.c_str());
            else if (arg == "--gmres_tol") cfg.gmres_tol = atof(val.c_str());
            else { fprintf(stderr, "Unknown option: %s\n", arg.c_str()); exit(1); }
        } else {
            fprintf(stderr, "Unknown option: %s\n", arg.c_str()); exit(1);
        }
    }
    return cfg;
}

static solver::OscillonParams makeOscillonParams(const Config& cfg) {
    solver::OscillonParams p;
    p.N_t = cfg.N_t;
    p.N_nuc = cfg.N_nuc;
    p.N_shell = cfg.N_shell;
    p.rho_mid = 0.5;
    p.N_theta = cfg.N_theta;
    p.Delta = cfg.Delta;
    p.Lambda = cfg.Lambda;
    p.ell = cfg.ell;
    return p;
}

static solver::NewtonParams makeNewtonParams(const Config& cfg) {
    solver::NewtonParams np;
    np.tol = cfg.tol;
    np.max_iter = cfg.max_iter;
    np.fd_eps = 1e-7;
    if (cfg.use_gmres) {
        np.use_lapack = false;
        np.use_direct = false;
        np.use_full_dense = false;
        np.gmres_restart = cfg.gmres_restart;
        np.gmres_tol = cfg.gmres_tol;
        np.gmres_max_iter = 3000;
        np.gmres_reg = 1e-2;              // Tikhonov regularization for Bianchi null space
        np.gmres_adaptive_tol = true;     // Eisenstat-Walker forcing
        np.max_iter = 50;                 // more Newton iterations (inexact steps)
    } else {
        np.use_lapack = true;
    }
    np.verbose = cfg.verbose;
    np.stag_window = 3;
    np.stag_ratio = 0.99;
    return np;
}

// ============================================================================
// Mode: Branch continuation
// ============================================================================
static int runBranch(const Config& cfg) {
    auto params = makeOscillonParams(cfg);
    solver::OscillonDriver driver(params);

    solver::ContinuationParams cont;
    cont.eps_start = cfg.eps_start;
    cont.dw_initial = 0.002;
    cont.dw_min = 1e-5;
    cont.dw_max = 0.1;
    cont.w_max = cfg.w_max;
    cont.residual_max = cfg.residual_max;
    cont.newton = makeNewtonParams(cfg);
    cont.verbose = !cfg.quiet;
    cont.store_solutions = false; // no need to store — we compute diagnostics in the callback
    cont.max_branch_points = cfg.max_branch_points;

    // Open CSV file for incremental writing
    std::string outfile = cfg.output;
    if (outfile.empty()) {
        char buf[256];
        snprintf(buf, sizeof(buf), "branch_ell%d_D%.0f_N%d.csv",
                 cfg.ell, cfg.Delta, cfg.N_nuc);
        outfile = buf;
    }

    FILE* csv = fopen(outfile.c_str(), "w");
    if (!csv) {
        fprintf(stderr, "ERROR: Cannot open %s for writing\n", outfile.c_str());
        return 1;
    }
    fprintf(csv, "# AdS4 Oscillon Branch Data\n");
    fprintf(csv, "# ell=%d, Delta=%.1f, Lambda=%.1f\n", cfg.ell, cfg.Delta, cfg.Lambda);
    fprintf(csv, "# N_nuc=%d, N_shell=%d, N_theta=%d, N_t=%d\n",
            cfg.N_nuc, cfg.N_shell, cfg.N_theta, cfg.N_t);
    fprintf(csv, "index,w,omega,residual,newton_iters,"
                 "E_scalar,phi_max,max_trK,max_evol,dNhat_dr\n");
    fflush(csv);

    // Checkpoint filename (same base as CSV but .ckpt)
    std::string ckptfile = outfile;
    if (ckptfile.size() > 4 && ckptfile.substr(ckptfile.size()-4) == ".csv")
        ckptfile = ckptfile.substr(0, ckptfile.size()-4) + ".ckpt";
    else
        ckptfile += ".ckpt";

    int n_state = driver.system().stateSize() - 1;

    // Callback: compute diagnostics, write CSV row, save checkpoint
    auto callback = [&](int idx, const solver::BranchPoint& bp,
                        const double* u, const solver::OscillonSystem& sys) {
        auto report = diagnostics::fullDiagnostics(sys, u, bp.omega);
        fprintf(csv, "%d,%.15e,%.15e,%.6e,%d,"
                     "%.15e,%.15e,%.6e,%.6e,%.15e\n",
                idx, bp.w, bp.omega, bp.residual, bp.newton_iters,
                report.observables.E_scalar,
                report.observables.phi_max,
                report.constraints.max_trK,
                report.constraints.max_evol_res,
                report.observables.dNhat_dr_boundary);
        fflush(csv);  // flush after every row so data survives timeout

        // Save checkpoint (overwrite each time — only need the latest)
        solver::OscillonDriver::saveCheckpoint(ckptfile, n_state, bp.omega, bp.w, u);
    };

    auto t0 = std::chrono::high_resolution_clock::now();
    std::vector<solver::BranchPoint> branch;

    if (!cfg.restart.empty()) {
        // Resume from checkpoint
        int n_ckpt;
        double omega_ckpt, w_ckpt;
        std::vector<double> u_ckpt;
        if (!solver::OscillonDriver::loadCheckpoint(cfg.restart, n_ckpt, omega_ckpt, w_ckpt, u_ckpt)) {
            fprintf(stderr, "ERROR: Cannot load checkpoint %s\n", cfg.restart.c_str());
            fclose(csv);
            return 1;
        }
        if (n_ckpt != n_state) {
            fprintf(stderr, "ERROR: Checkpoint state size %d != expected %d\n", n_ckpt, n_state);
            fclose(csv);
            return 1;
        }
        printf("Resuming from checkpoint: w = %.6e, omega = %.10f\n", w_ckpt, omega_ckpt);
        branch = driver.resumeContinuation(u_ckpt.data(), omega_ckpt, w_ckpt, cont, callback);
    } else {
        branch = driver.runContinuation(cont, callback);
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();

    fclose(csv);

    if (branch.empty()) {
        fprintf(stderr, "ERROR: No branch points found\n");
        return 1;
    }

    printf("\nBranch data written to %s (%d rows)\n", outfile.c_str(), (int)branch.size());
    printf("\nBranch summary: %d points, omega [%.6f, %.6f], "
           "w [%.4e, %.4e], time %.1fs\n",
           (int)branch.size(), branch.back().omega, branch.front().omega,
           branch.front().w, branch.back().w, elapsed);

    return 0;
}

// ============================================================================
// Mode: Arclength continuation
// ============================================================================
static int runArclength(const Config& cfg) {
    auto params = makeOscillonParams(cfg);
    solver::OscillonDriver driver(params);

    solver::ArclengthParams ap;
    ap.eps_start = cfg.eps_start;
    ap.eps_second = cfg.eps_second;
    ap.ds_initial = cfg.ds_initial;
    ap.ds_min = cfg.ds_min;
    ap.ds_max = cfg.ds_max;
    ap.fast_iters = 5;
    ap.slow_iters = 15;
    ap.step_increase = 1.5;
    ap.step_decrease = 0.5;
    ap.newton = makeNewtonParams(cfg);
    ap.max_branch_points = cfg.max_branch_points;
    ap.residual_max = cfg.residual_max;
    ap.stop_after_fold = false;
    ap.points_after_fold = cfg.points_after_fold;
    ap.verbose = !cfg.quiet;
    ap.store_solutions = false;

    // Open CSV file
    std::string outfile = cfg.output;
    if (outfile.empty()) {
        char buf[256];
        snprintf(buf, sizeof(buf), "arclength_ell%d_D%.0f_N%d.csv",
                 cfg.ell, cfg.Delta, cfg.N_nuc);
        outfile = buf;
    }

    FILE* csv = fopen(outfile.c_str(), "w");
    if (!csv) {
        fprintf(stderr, "ERROR: Cannot open %s for writing\n", outfile.c_str());
        return 1;
    }
    fprintf(csv, "# AdS4 Oscillon Branch Data (arclength continuation)\n");
    fprintf(csv, "# ell=%d, Delta=%.1f, Lambda=%.1f\n", cfg.ell, cfg.Delta, cfg.Lambda);
    fprintf(csv, "# N_nuc=%d, N_shell=%d, N_theta=%d, N_t=%d\n",
            cfg.N_nuc, cfg.N_shell, cfg.N_theta, cfg.N_t);
    fprintf(csv, "index,w,omega,residual,newton_iters,"
                 "E_scalar,phi_max,max_trK,max_evol,dNhat_dr\n");
    fflush(csv);

    // Checkpoint filenames
    std::string ckptfile = outfile;
    if (ckptfile.size() > 4 && ckptfile.substr(ckptfile.size()-4) == ".csv")
        ckptfile = ckptfile.substr(0, ckptfile.size()-4) + ".ckpt";
    else
        ckptfile += ".ckpt";

    int n_state = driver.system().stateSize() - 1;

    // Track previous two points for arclength checkpoint
    std::vector<double> ckpt_u_prev, ckpt_u_curr;
    double ckpt_omega_prev = 0, ckpt_omega_curr = 0;
    double ckpt_w_prev = 0, ckpt_w_curr = 0;

    auto callback = [&](int idx, const solver::BranchPoint& bp,
                        const double* u, const solver::OscillonSystem& sys) {
        auto report = diagnostics::fullDiagnostics(sys, u, bp.omega);
        fprintf(csv, "%d,%.15e,%.15e,%.6e,%d,"
                     "%.15e,%.15e,%.6e,%.6e,%.15e\n",
                idx, bp.w, bp.omega, bp.residual, bp.newton_iters,
                report.observables.E_scalar,
                report.observables.phi_max,
                report.constraints.max_trK,
                report.constraints.max_evol_res,
                report.observables.dNhat_dr_boundary);
        fflush(csv);

        // Update checkpoint state
        ckpt_u_prev = ckpt_u_curr;
        ckpt_omega_prev = ckpt_omega_curr;
        ckpt_w_prev = ckpt_w_curr;
        ckpt_u_curr.assign(u, u + n_state);
        ckpt_omega_curr = bp.omega;
        ckpt_w_curr = bp.w;

        // Save arclength checkpoint (need two points)
        if (!ckpt_u_prev.empty()) {
            solver::OscillonDriver::saveArclengthCheckpoint(
                ckptfile, n_state,
                ckpt_omega_prev, ckpt_w_prev, ckpt_u_prev.data(),
                ckpt_omega_curr, ckpt_w_curr, ckpt_u_curr.data());
        }
    };

    auto t0 = std::chrono::high_resolution_clock::now();
    std::vector<solver::BranchPoint> branch;

    if (!cfg.restart.empty()) {
        // Resume from arclength checkpoint
        int n_ckpt;
        double om0, w0_ck, om1, w1_ck;
        std::vector<double> u0_ck, u1_ck;
        if (!solver::OscillonDriver::loadArclengthCheckpoint(
                cfg.restart, n_ckpt, om0, w0_ck, u0_ck, om1, w1_ck, u1_ck)) {
            fprintf(stderr, "ERROR: Cannot load arclength checkpoint %s\n",
                    cfg.restart.c_str());
            fclose(csv);
            return 1;
        }
        if (n_ckpt != n_state) {
            fprintf(stderr, "ERROR: Checkpoint state size %d != expected %d\n",
                    n_ckpt, n_state);
            fclose(csv);
            return 1;
        }
        printf("Resuming from arclength checkpoint: w = [%.6e, %.6e]\n",
               w0_ck, w1_ck);
        branch = driver.resumeArclengthContinuation(
            u0_ck.data(), om0, w0_ck,
            u1_ck.data(), om1, w1_ck,
            ap, callback);
    } else {
        branch = driver.runArclengthContinuation(ap, callback);
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();

    fclose(csv);

    if (branch.empty()) {
        fprintf(stderr, "ERROR: No branch points found\n");
        return 1;
    }

    printf("\nBranch data written to %s (%d rows)\n", outfile.c_str(),
           (int)branch.size());
    printf("Branch summary: %d points, omega [%.6f, %.6f], "
           "w [%.4e, %.4e], time %.1fs\n",
           (int)branch.size(), branch.back().omega, branch.front().omega,
           branch.front().w, branch.back().w, elapsed);

    return 0;
}

// ============================================================================
// Mode: Resolution convergence study
// ============================================================================
static int runConvergence(const Config& cfg) {
    printf("=== Resolution convergence study ===\n");
    printf("ell=%d, Delta=%.1f, Lambda=%.1f, eps=%.4f\n\n",
           cfg.ell, cfg.Delta, cfg.Lambda, cfg.eps);

    // Resolution levels: (N_nuc, N_shell, N_theta)
    struct ResLevel { int N_nuc, N_shell, N_theta, N_t; };
    std::vector<ResLevel> levels = {
        {4,  4,  2, 2},
        {6,  6,  2, 2},
        {6,  6,  4, 2},
        {8,  8,  2, 2},
        {8,  8,  4, 2},
        {10, 10, 4, 2},
    };

    std::vector<io::ConvergenceRow> rows;
    auto np = makeNewtonParams(cfg);
    np.verbose = cfg.verbose;

    for (const auto& lev : levels) {
        solver::OscillonParams params;
        params.N_t = lev.N_t;
        params.N_nuc = lev.N_nuc;
        params.N_shell = lev.N_shell;
        params.rho_mid = 0.5;
        params.N_theta = lev.N_theta;
        params.Delta = cfg.Delta;
        params.Lambda = cfg.Lambda;
        params.ell = cfg.ell;

        solver::OscillonDriver driver(params);
        int n = driver.system().stateSize() - 1;

        printf("N_nuc=%d, N_theta=%d, N_t=%d (n=%d) ... ",
               lev.N_nuc, lev.N_theta, lev.N_t, n + 1);
        fflush(stdout);

        solver::NewtonResult result;
        std::vector<double> u;
        double omega;

        auto t0 = std::chrono::high_resolution_clock::now();
        bool ok = driver.solveAtEpsilon(cfg.eps, result, u, omega, np);
        auto t1 = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(t1 - t0).count();

        io::ConvergenceRow row;
        row.N_nuc = lev.N_nuc;
        row.N_shell = lev.N_shell;
        row.N_theta = lev.N_theta;
        row.N_t = lev.N_t;
        row.state_size = n + 1;
        row.time_seconds = elapsed;

        if (ok) {
            auto report = diagnostics::fullDiagnostics(driver.system(), u.data(), omega);
            row.omega = omega;
            row.E_scalar = report.observables.E_scalar;
            row.max_trK = report.constraints.max_trK;
            row.max_evol = report.constraints.max_evol_res;
            row.residual = result.final_residual;

            printf("omega=%.10f, E=%.6e, trK=%.2e, time=%.1fs\n",
                   omega, row.E_scalar, row.max_trK, elapsed);
        } else {
            row.omega = 0;
            row.E_scalar = 0;
            row.max_trK = 0;
            row.max_evol = 0;
            row.residual = result.final_residual;
            printf("FAILED (res=%.2e, time=%.1fs)\n", result.final_residual, elapsed);
        }
        rows.push_back(row);
    }

    // Richardson extrapolation: if we have multiple converged results,
    // show the convergence of omega and E
    printf("\n--- Convergence table ---\n");
    printf("%6s %6s %6s %8s %15s %15s %10s %10s\n",
           "N_nuc", "N_th", "N_t", "n", "omega", "E_scalar", "max_trK", "time_s");
    for (const auto& r : rows) {
        printf("%6d %6d %6d %8d %15.10f %15.6e %10.2e %10.1f\n",
               r.N_nuc, r.N_theta, r.N_t, r.state_size,
               r.omega, r.E_scalar, r.max_trK, r.time_seconds);
    }

    if (rows.size() >= 2) {
        double omega_prev = rows[rows.size()-2].omega;
        double omega_last = rows[rows.size()-1].omega;
        printf("\nDelta(omega) between last two: %.6e\n",
               std::abs(omega_last - omega_prev));
    }

    // Write CSV
    std::string outfile = cfg.output;
    if (outfile.empty()) {
        char buf[256];
        snprintf(buf, sizeof(buf), "convergence_ell%d_D%.0f_eps%.3f.csv",
                 cfg.ell, cfg.Delta, cfg.eps);
        outfile = buf;
    }
    io::writeConvergenceCSV(outfile, rows, cfg.ell, cfg.Delta, cfg.Lambda, cfg.eps);

    return 0;
}

// ============================================================================
// Mode: Single solve
// ============================================================================
static int runSingle(const Config& cfg) {
    auto params = makeOscillonParams(cfg);
    solver::OscillonDriver driver(params);
    auto np = makeNewtonParams(cfg);
    np.verbose = true;

    int n = driver.system().stateSize() - 1;
    printf("=== Single solve ===\n");
    printf("ell=%d, Delta=%.1f, Lambda=%.1f, eps=%.4f\n", cfg.ell, cfg.Delta, cfg.Lambda, cfg.eps);
    printf("N_nuc=%d, N_shell=%d, N_theta=%d, N_t=%d (n=%d)\n\n",
           cfg.N_nuc, cfg.N_shell, cfg.N_theta, cfg.N_t, n + 1);

    solver::NewtonResult result;
    std::vector<double> u;
    double omega;

    auto t0 = std::chrono::high_resolution_clock::now();
    bool ok = driver.solveAtEpsilon(cfg.eps, result, u, omega, np);
    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();

    printf("\nNewton: %s in %d iterations, residual = %.4e, time = %.1fs\n",
           ok ? "converged" : "FAILED", result.iterations, result.final_residual, elapsed);

    if (ok) {
        printf("\n");
        auto report = diagnostics::fullDiagnostics(driver.system(), u.data(), omega);
        report.print("  ");
    }

    return ok ? 0 : 1;
}

// ============================================================================
// Main
// ============================================================================
int main(int argc, char** argv) {
    Config cfg = parseArgs(argc, argv);

    printf("========================================\n");
    printf("AdS4 Massive Scalar Oscillons\n");
    printf("  Martinon et al. (2017) [arXiv:1701.09100]\n");
#ifdef USE_OPENMP
    printf("  OpenMP: %d threads available\n", omp_get_max_threads());
#else
    printf("  OpenMP: disabled (serial Jacobian build)\n");
#endif
#ifdef HAVE_LAPACK
    printf("  LAPACK: enabled (dgelsy for rank-deficient Newton)\n");
#else
    printf("  LAPACK: disabled (GMRES fallback)\n");
#endif
    printf("========================================\n\n");

    switch (cfg.mode) {
        case MODE_BRANCH:      return runBranch(cfg);
        case MODE_ARCLENGTH:   return runArclength(cfg);
        case MODE_CONVERGENCE: return runConvergence(cfg);
        case MODE_SINGLE:      return runSingle(cfg);
    }
    return 1;
}
