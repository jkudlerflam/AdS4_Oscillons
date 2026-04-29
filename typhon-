// oscillon_driver.h — Amplitude continuation driver for the oscillon system
//
// Two continuation modes:
//
// 1. Fixed-w stepping (runContinuation / resumeContinuation):
//    Steps the normalization target w = phi_hat(tau=0, ref point).
//    Simple but cannot navigate folds where dE/dw = 0.
//
// 2. Pseudo-arclength continuation (runArclengthContinuation):
//    Parameterizes the branch by arc length s. At each step, the Newton
//    system is augmented with:
//        dot_u · (u - u_prev) + dot_omega · (omega - omega_prev) = ds
//    where (dot_u, dot_omega) is the tangent vector. This navigates
//    folds in any variable, including E_max turning points.
//
// Both modes support adaptive stepping, convergence guards, and
// checkpoint/restart.
#pragma once

#include "solver/oscillon_system.h"
#include "solver/newton.h"
#include <vector>
#include <string>
#include <functional>

namespace solver {

// A single point on the solution branch
struct BranchPoint {
    double w;             // normalization value
    double omega;         // frequency
    double residual;      // final Newton residual
    int newton_iters;     // Newton iterations used
    std::vector<double> u; // full state vector (optional, can be empty to save memory)
};

// Parameters for the continuation driver
struct ContinuationParams {
    // Starting amplitude
    double eps_start = 0.01;

    // Normalization stepping (fixed-w mode)
    double dw_initial = 0.005;   // initial step in w
    double dw_min = 1e-6;        // minimum step
    double dw_max = 0.1;         // maximum step
    double w_max = 10.0;         // stop when w exceeds this

    // Adaptive stepping
    int fast_iters = 5;          // if Newton converges in <= this, increase step
    int slow_iters = 15;         // if Newton takes >= this, decrease step
    double step_increase = 1.5;  // factor to increase dw
    double step_decrease = 0.5;  // factor to decrease dw

    // Newton parameters
    NewtonParams newton;

    // Storage options
    bool store_solutions = false; // if true, store full u at each branch point
    int max_branch_points = 1000; // safety limit

    // Convergence guard: stop writing data if Newton residual exceeds this
    double residual_max = 1e-4;

    // Output
    bool verbose = true;
};

// Parameters specific to pseudo-arclength continuation
struct ArclengthParams {
    // Starting amplitude (used to get the first two points via fixed-w)
    double eps_start = 0.01;
    double eps_second = 0.02;    // second amplitude for initial tangent

    // Arclength stepping
    double ds_initial = 0.01;    // initial arclength step
    double ds_min = 1e-7;        // minimum step
    double ds_max = 0.5;         // maximum step

    // Adaptive stepping (same logic as ContinuationParams)
    int fast_iters = 5;
    int slow_iters = 15;
    double step_increase = 1.5;
    double step_decrease = 0.5;

    // Newton parameters
    NewtonParams newton;

    // Termination
    int max_branch_points = 2000;
    double residual_max = 1e-4;  // convergence guard
    double omega_min = 0.0;      // stop if omega drops below this
    bool stop_after_fold = true; // stop after detecting E_max (dE/ds changes sign)
    int points_after_fold = 20;  // how many points to compute past the fold

    // Storage
    bool store_solutions = false;

    // Output
    bool verbose = true;
};

class OscillonDriver {
public:
    OscillonDriver(const OscillonParams& params);

    // Callback type: called after each successful branch point.
    // Arguments: (branch_index, BranchPoint, pointer to state vector u, system reference)
    // The state vector is valid only during the callback.
    using BranchCallback = std::function<void(int, const BranchPoint&, const double*, const OscillonSystem&)>;

    // ---- Fixed-w continuation (original) ----

    std::vector<BranchPoint> runContinuation(const ContinuationParams& cont = ContinuationParams(),
                                             BranchCallback cb = nullptr);

    std::vector<BranchPoint> resumeContinuation(const double* u_init, double omega_init,
                                                double w_init,
                                                const ContinuationParams& cont = ContinuationParams(),
                                                BranchCallback cb = nullptr);

    // ---- Pseudo-arclength continuation ----

    // Run arclength continuation from a small-amplitude seed.
    // Bootstrap: solves at eps_start and eps_second via fixed-w to get
    // the initial tangent, then switches to arclength stepping.
    std::vector<BranchPoint> runArclengthContinuation(
        const ArclengthParams& params = ArclengthParams(),
        BranchCallback cb = nullptr);

    // Resume arclength continuation from two consecutive branch points.
    // u0/omega0/w0 and u1/omega1/w1 define the tangent direction.
    std::vector<BranchPoint> resumeArclengthContinuation(
        const double* u0, double omega0, double w0,
        const double* u1, double omega1, double w1,
        const ArclengthParams& params = ArclengthParams(),
        BranchCallback cb = nullptr);

    // ---- Single solves ----

    // Save/load checkpoint (binary: n, omega, w, u[0..n-1])
    static bool saveCheckpoint(const std::string& filename, int n, double omega,
                               double w, const double* u);
    static bool loadCheckpoint(const std::string& filename, int& n, double& omega,
                               double& w, std::vector<double>& u);

    // Save/load arclength checkpoint (two consecutive points + tangent)
    static bool saveArclengthCheckpoint(const std::string& filename, int n,
                                        double omega0, double w0, const double* u0,
                                        double omega1, double w1, const double* u1);
    static bool loadArclengthCheckpoint(const std::string& filename, int& n,
                                        double& omega0, double& w0, std::vector<double>& u0,
                                        double& omega1, double& w1, std::vector<double>& u1);

    // Solve at a single amplitude epsilon.
    // Returns true if Newton converged.
    bool solveAtEpsilon(double epsilon, NewtonResult& result,
                        std::vector<double>& u_out, double& omega_out,
                        const NewtonParams& np = NewtonParams());

    // Solve at a given normalization target w, starting from initial guess u, omega.
    // u and omega are modified in place.
    NewtonResult solveAtW(double* u, double& omega, double w_target,
                          const NewtonParams& np = NewtonParams());

    // Access the underlying system
    const OscillonSystem& system() const { return sys_; }

private:
    OscillonParams params_;
    OscillonSystem sys_;

    // Internal: Newton solve with arclength constraint.
    // Solves F(u, omega) = 0 subject to:
    //   dot_u · (u - u_pred) + dot_omega * (omega - omega_pred) = 0
    // where (u_pred, omega_pred) is the predictor and (dot_u, dot_omega) is
    // the tangent vector (unit norm in the (u, omega) space).
    NewtonResult solveArclength(double* u, double& omega,
                                const double* u_pred, double omega_pred,
                                const double* dot_u, double dot_omega,
                                const NewtonParams& np);
};

} // namespace solver
