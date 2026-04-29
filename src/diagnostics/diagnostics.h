// diagnostics.h — Solution diagnostics for oscillon solver
//
// Three categories:
//   1. Constraint monitoring: max|trK|, max|V^i| at interior points
//   2. Spectral convergence: Chebyshev/Legendre/Fourier tail decay
//   3. Physical observables: time-averaged scalar energy, AMD mass proxy
#pragma once

#include <vector>
#include <cstdio>
#include "solver/oscillon_system.h"
#include "geometry/adm.h"
#include "geometry/equations.h"

namespace diagnostics {

// ============================================================================
// Constraint violations
// ============================================================================
struct ConstraintData {
    double max_trK;        // max |trK| over interior collocation points
    double max_V_rho;      // max |V^rho| (gauge violation)
    double max_V_theta;    // max |V^theta|
    double max_evol_res;   // max evolution equation residual (raw, pre-regularization)
    double max_scalar_res; // max scalar wave residual

    // L2 norms (RMS over interior points)
    double rms_trK;
    double rms_evol;
    int n_interior;        // number of interior points used

    void print(const char* prefix = "") const {
        printf("%sConstraint violations:\n", prefix);
        printf("%s  max|trK| = %.4e, max|V^rho| = %.4e, max|V^theta| = %.4e\n",
               prefix, max_trK, max_V_rho, max_V_theta);
        printf("%s  max|evol| = %.4e, max|scalar| = %.4e\n",
               prefix, max_evol_res, max_scalar_res);
        printf("%s  rms(trK) = %.4e, rms(evol) = %.4e [%d interior pts]\n",
               prefix, rms_trK, rms_evol, n_interior);
    }
};

// Compute constraint violations from a converged solution
ConstraintData computeConstraints(
    const solver::OscillonSystem& sys,
    const double* u, double omega);

// ============================================================================
// Spectral convergence: tail decay in each spectral direction
// ============================================================================
struct SpectralTails {
    // For each field, the max coefficient magnitude at each spectral order
    std::vector<double> chebyshev_tail; // max |c_n| for Chebyshev order n
    std::vector<double> legendre_tail;  // max |a_l| for Legendre degree l
    std::vector<double> fourier_tail;   // max |a_k| for Fourier mode k

    // Effective decay rate (log10 of last / first)
    double cheb_decay_rate;
    double leg_decay_rate;
    double four_decay_rate;

    // Number of resolved digits (approx)
    double cheb_digits;
    double leg_digits;
    double four_digits;

    void print(const char* prefix = "") const {
        printf("%sSpectral tails (decay rates):\n", prefix);
        printf("%s  Chebyshev: %.1f digits (rate = %.2f/mode)\n",
               prefix, cheb_digits, cheb_decay_rate);
        printf("%s  Legendre:  %.1f digits (rate = %.2f/mode)\n",
               prefix, leg_digits, leg_decay_rate);
        printf("%s  Fourier:   %.1f digits (rate = %.2f/mode)\n",
               prefix, four_digits, four_decay_rate);
    }
};

// Compute spectral tail decay for scalar field
SpectralTails computeSpectralTails(
    const solver::OscillonSystem& sys,
    const double* u);

// ============================================================================
// Physical observables
// ============================================================================
struct PhysicalObservables {
    double omega;          // frequency
    double w;              // normalization (amplitude parameter)

    // Time-averaged scalar field energy (volume integral)
    double E_scalar;

    // Max scalar field value |phi| over the domain
    double phi_max;

    // Boundary data: d_rho(N_hat)|_{rho=1} (AMD mass proxy)
    // The true AMD mass requires careful holographic renormalization;
    // this proxy tracks the leading boundary correction.
    double dNhat_dr_boundary;

    void print(const char* prefix = "") const {
        printf("%sPhysical observables:\n", prefix);
        printf("%s  omega = %.10f, w = %.6e\n", prefix, omega, w);
        printf("%s  E_scalar = %.6e, phi_max = %.6e\n",
               prefix, E_scalar, phi_max);
        printf("%s  dNhat_dr|_{rho=1} = %.6e (AMD proxy)\n",
               prefix, dNhat_dr_boundary);
    }
};

PhysicalObservables computeObservables(
    const solver::OscillonSystem& sys,
    const double* u, double omega);

// ============================================================================
// Full diagnostic report for one solution point
// ============================================================================
struct DiagnosticReport {
    ConstraintData constraints;
    SpectralTails tails;
    PhysicalObservables observables;

    void print(const char* prefix = "") const {
        observables.print(prefix);
        constraints.print(prefix);
        tails.print(prefix);
    }
};

DiagnosticReport fullDiagnostics(
    const solver::OscillonSystem& sys,
    const double* u, double omega);

} // namespace diagnostics
