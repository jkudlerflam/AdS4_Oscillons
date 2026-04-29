// ads4.h — AdS4 background geometry and scalar eigenmodes
// Conformal coordinates following Martinon et al. (2017)
#pragma once

#include <cmath>
#include <vector>
#include <functional>

namespace geometry {

// AdS4 with L=1, Lambda = -3
// Compactified coordinates: rho in [0,1), where rho = r/(1+sqrt(1+r^2))
// Or equivalently x_bar in [0, pi/2), with rho = sin(x_bar)/(1 + cos(x_bar))
//
// Conformal factor: Omega = (1 - rho^2)/(1 + rho^2)
// Background metric (eq 30 of Martinon et al.):
//   ds^2 = [(1+rho^2)/(1-rho^2)]^2 dt^2 + 4/(1-rho^2)^2 (drho^2 + rho^2 dOmega_2^2)

class AdS4Background {
public:
    AdS4Background() = default;

    // Conformal factor Omega(rho)
    static double Omega(double rho);
    static double dOmega_drho(double rho);

    // x_bar coordinate: cos(x_bar) = (1-rho^2)/(1+rho^2) = Omega
    static double rho_of_xbar(double xbar);
    static double xbar_of_rho(double rho);

    // Background metric components in (rho, theta, phi) coordinates
    // gamma_bar_{rho rho} = 4/(1-rho^2)^2
    // gamma_bar_{theta theta} = 4*rho^2/(1-rho^2)^2
    // gamma_bar_{phi phi} = 4*rho^2*sin^2(theta)/(1-rho^2)^2
    // N_bar = (1+rho^2)/(1-rho^2)
    static double N_bar(double rho);
    static double gamma_bar_rhorho(double rho);
    static double gamma_bar_thetatheta(double rho, double theta);
    static double gamma_bar_phiphi(double rho, double theta);

    // Regularized background: hat quantities (Table 1 of Martinon et al.)
    // N_hat = Omega * N_bar = 1
    // gamma_hat_{ij} = Omega^2 * gamma_bar_{ij}
    static double Nhat_bar(double rho) { return 1.0; }
    static double gammahat_bar_rhorho(double rho);
    static double gammahat_bar_thetatheta(double rho, double theta);
    static double gammahat_bar_phiphi(double rho, double theta);

    // Background Christoffel symbols Gamma_bar^i_{jk} in (rho, theta, phi)
    // These are needed for the gauge condition V^i = gamma^{kl}(Gamma^i_{kl} - Gamma_bar^i_{kl})
    struct Christoffel {
        double G_rho_rhorho, G_rho_thetatheta, G_rho_phiphi;
        double G_theta_rhotheta, G_theta_phiphi;
        double G_phi_rhophi, G_phi_thetaphi;
        // All others are zero by symmetry or obtained by symmetry of lower indices
    };
    static Christoffel christoffel_bar(double rho, double theta);
};

// Scalar eigenmode on AdS4
// (n_rad, ell, m=0) mode: phi = f(x_bar) * P_ell(cos theta) * cos(omega_0 * t)
// omega_0 = Delta + 2*n_rad + ell  (for AdS4, d=3, omega = Delta + 2n + ell)
// For n_rad = 0: f(x_bar) = sin^ell(x_bar) * cos^Delta(x_bar) [unnormalized]
//
// In rho coordinates: x_bar = 2*arctan(rho), sin(x_bar) = 2*rho/(1+rho^2),
//   cos(x_bar) = (1-rho^2)/(1+rho^2) = Omega
// So f(rho) = [2*rho/(1+rho^2)]^ell * [(1-rho^2)/(1+rho^2)]^Delta

class ScalarEigenmode {
public:
    ScalarEigenmode(int ell, double Delta, int n_rad = 0);

    int ell() const { return ell_; }
    double Delta() const { return Delta_; }
    double mass_sq() const { return Delta_ * (Delta_ - 3.0); }
    double omega0() const { return omega0_; }

    // Radial profile f(rho) [in rho coordinates, unnormalized]
    double f_rho(double rho) const;

    // Radial profile f(x_bar) [in x_bar coordinates]
    double f_xbar(double xbar) const;

    // Full eigenmode phi(t, rho, theta) = epsilon * f(rho) * P_ell(cos theta) * cos(omega_0 * t)
    double phi(double t, double rho, double theta, double epsilon = 1.0) const;

    // Regularized eigenmode: phi_hat = phi / Omega^(Delta/2)
    double phi_hat(double t, double rho, double theta, double epsilon = 1.0) const;

    // Normalization: set f(0) = 1 for ell=0, or appropriate limit for ell>0
    // For n_rad=0: f(rho) -> (2*rho)^ell as rho->0
    // Normalized so phi_hat(t=0, rho=0, theta=0) = epsilon for ell=0

private:
    int ell_, n_rad_;
    double Delta_, omega0_;
    double m_sq_;
};

// Verify the eigenmode satisfies the wave equation on AdS4
// Returns max |Box phi - m^2 phi| at a set of test points
double verifyEigenmode(const ScalarEigenmode& mode, int N_rho, int N_theta);

} // namespace geometry
