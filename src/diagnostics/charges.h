// charges.h — Conserved charges: AMD mass, angular momentum
//
// For time-periodic solutions in AdS, the AMD mass is defined from the
// asymptotic expansion of the conformal metric. Since our Dirichlet BCs
// force the metric to match AdS at the boundary, the mass information
// is encoded in the subleading radial derivatives.
//
// The full AMD mass computation requires careful holographic renormalization.
// For now we provide:
//   - A mass proxy from the boundary derivative of N_hat
//   - The Komar integral (volume integral, well-defined for stationary spacetimes)
#pragma once

#include "solver/oscillon_system.h"

namespace diagnostics {

// Komar mass via volume integral (time-averaged for periodic solutions)
// M_Komar = -(1/4πG) ∫_Σ R_{μν} n^μ ξ^ν √γ d³x
// With 8πG = 1 (Martinon convention): M_Komar = -2 ∫ (R_{μν} n^μ ξ^ν) √γ d³x
double computeKomarMassProxy(
    const solver::OscillonSystem& sys,
    const double* u, double omega);

} // namespace diagnostics
