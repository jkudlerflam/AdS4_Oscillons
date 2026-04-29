// charges.cpp — Conserved charges implementation
#include "diagnostics/charges.h"
#include "diagnostics/diagnostics.h"
#include <cmath>

namespace diagnostics {

double computeKomarMassProxy(
    const solver::OscillonSystem& sys,
    const double* u, double omega)
{
    // For now, use the scalar energy as a mass proxy.
    // The full Komar integral requires the 4D Ricci tensor which
    // needs additional infrastructure.
    auto obs = computeObservables(sys, u, omega);
    return obs.E_scalar;
}

} // namespace diagnostics
