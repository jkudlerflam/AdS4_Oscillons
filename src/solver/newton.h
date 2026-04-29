// newton.h — Newton-Raphson solver for the nonlinear spectral system
#pragma once
#include <vector>
#include <functional>

namespace solver {

struct NewtonParams {
    double tol = 1e-8;          // convergence tolerance
    int max_iter = 30;          // max Newton iterations
    int gmres_restart = 50;     // GMRES restart parameter
    double gmres_tol = 1e-6;    // GMRES tolerance
    int gmres_max_iter = 500;   // max GMRES iterations
    double gmres_reg = 0.0;     // Tikhonov regularization: solve (J + reg*I)du = -F
    bool gmres_adaptive_tol = false; // Eisenstat-Walker adaptive forcing
    double fd_eps = 1e-7;       // finite difference step for Jacobian
    bool use_direct = false;    // use block-structured dense solve instead of GMRES
    bool use_full_dense = false; // use full (non-block) dense Jacobian with regularized LU
    bool use_lapack = false;     // use LAPACK dgelsd (SVD least-squares) — handles null space correctly
    bool verbose = true;

    // Stagnation detection: declare convergence if the residual stops decreasing.
    // This handles the Bianchi identity null-space floor in the time-collocation
    // formulation, where the minimum achievable residual is set by the spectral
    // truncation error projected onto null(J^T).
    int stag_window = 3;        // number of stagnating iterations before declaring converged
    double stag_ratio = 0.99;   // ratio threshold: stagnation if res[i] > stag_ratio * res[i-1]
};

struct NewtonResult {
    bool converged;
    bool stagnated;             // true if convergence was by stagnation detection
    int iterations;
    double final_residual;
    double omega;               // converged frequency
    std::vector<double> residual_history;
};

// Solve F(u, omega) = 0 with u the state vector and omega the extra unknown
// residual_func(u, omega, F): computes F given u and omega
// normalization_func(u): returns the value that should equal the prescribed w
NewtonResult newtonSolve(
    std::function<void(const double*, double, double*)> residual_func,
    std::function<double(const double*)> normalization_func,
    double* u,              // initial guess (modified in place)
    double& omega,          // initial frequency guess (modified)
    double w_target,        // normalization target
    int n,                  // size of u
    const NewtonParams& params = NewtonParams()
);

} // namespace solver
