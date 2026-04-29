// ads4.cpp — AdS4 background geometry and scalar eigenmodes
#include "geometry/ads4.h"
#include "spectral/chebyshev.h"
#include "spectral/legendre.h"
#include <cmath>
#include <algorithm>
#include <cstdio>

namespace geometry {

static const double PI = 3.14159265358979323846264338327950288;

// ---- AdS4Background ----

double AdS4Background::Omega(double rho) {
    return (1.0 - rho*rho) / (1.0 + rho*rho);
}

double AdS4Background::dOmega_drho(double rho) {
    double r2 = rho*rho;
    return -4.0 * rho / ((1.0 + r2) * (1.0 + r2));
}

double AdS4Background::rho_of_xbar(double xbar) {
    // rho = tan(x_bar/2) = sin(x_bar)/(1 + cos(x_bar))
    return tan(xbar / 2.0);
}

double AdS4Background::xbar_of_rho(double rho) {
    return 2.0 * atan(rho);
}

double AdS4Background::N_bar(double rho) {
    return (1.0 + rho*rho) / (1.0 - rho*rho);
}

double AdS4Background::gamma_bar_rhorho(double rho) {
    double d = 1.0 - rho*rho;
    return 4.0 / (d * d);
}

double AdS4Background::gamma_bar_thetatheta(double rho, double /*theta*/) {
    double d = 1.0 - rho*rho;
    return 4.0 * rho * rho / (d * d);
}

double AdS4Background::gamma_bar_phiphi(double rho, double theta) {
    double d = 1.0 - rho*rho;
    double st = sin(theta);
    return 4.0 * rho * rho * st * st / (d * d);
}

double AdS4Background::gammahat_bar_rhorho(double rho) {
    // Omega^2 * gamma_bar_{rhorho}
    double Om = Omega(rho);
    return Om * Om * gamma_bar_rhorho(rho);
    // = [(1-rho^2)/(1+rho^2)]^2 * 4/(1-rho^2)^2
    // = 4/(1+rho^2)^2
}

double AdS4Background::gammahat_bar_thetatheta(double rho, double theta) {
    double Om = Omega(rho);
    return Om * Om * gamma_bar_thetatheta(rho, theta);
    // = 4*rho^2/(1+rho^2)^2
}

double AdS4Background::gammahat_bar_phiphi(double rho, double theta) {
    double Om = Omega(rho);
    return Om * Om * gamma_bar_phiphi(rho, theta);
    // = 4*rho^2*sin^2(theta)/(1+rho^2)^2
}

AdS4Background::Christoffel AdS4Background::christoffel_bar(double rho, double theta) {
    Christoffel G;
    // Flat metric in (rho, theta, phi): ds^2_flat = drho^2 + rho^2(dtheta^2 + sin^2 theta dphi^2)
    // Plus conformal factor: 4/(1-rho^2)^2
    // gamma_bar_{ij} = A(rho)^2 * delta_{ij} in flat polar, where A = 2/(1-rho^2)
    // A' = 4*rho/(1-rho^2)^2
    // For conformally flat: Gamma^i_{jk} = (delta^i_j A'/A + delta^i_k A'/A - delta_{jk} g^{il} A'/A)
    //   plus the flat Christoffel symbols

    double r2 = rho * rho;
    double onemr2 = 1.0 - r2;

    // d/drho [log(A^2)] = d/drho [log(4/(1-rho^2)^2)] = 4*rho/(1-rho^2)
    // = 2 * A'/A where A = 2/(1-rho^2), so A'/A = 2*rho/(1-rho^2)
    double ApA = 2.0 * rho / onemr2; // A'/A

    // Flat Christoffel in spherical: Gamma^rho_{theta theta} = -rho,
    // Gamma^rho_{phi phi} = -rho*sin^2(theta), Gamma^theta_{rho theta} = 1/rho,
    // Gamma^theta_{phi phi} = -sin(theta)*cos(theta), Gamma^phi_{rho phi} = 1/rho,
    // Gamma^phi_{theta phi} = cos(theta)/sin(theta)

    // Conformal Christoffel symbols:
    // Gamma^i_{jk} = Gamma_flat^i_{jk} + delta^i_j * d_k(log A) + delta^i_k * d_j(log A)
    //                - gamma_flat_{jk} * gamma_flat^{il} * d_l(log A)
    // where d_l(log A) = (A'/A) * delta_{l rho} (only radial dependence)
    // and A'/A = 2*rho/(1-rho^2)

    // Since only rho-derivative of A is nonzero:
    // If both j,k != rho: Gamma^rho_{jk} += -gamma_flat_{jk}/1 * (A'/A) [from last term with i=rho, l=rho, gamma^{rho rho}=1]
    // Actually in flat spherical, gamma_flat_{rho rho} = 1, gamma^{rho rho} = 1

    double ct = cos(theta), st = sin(theta);

    // Gamma^rho_{rho rho} = 0 + ApA + ApA - 1 * 1 * ApA = ApA
    G.G_rho_rhorho = ApA;

    // Gamma^rho_{theta theta} = -rho + 0 + 0 - rho^2 * 1 * ApA = -rho - rho^2 * ApA
    //  = -rho * (1 + rho * ApA) = -rho * (1 + 2*rho^2/(1-rho^2)) = -rho * (1+rho^2)/(1-rho^2)
    G.G_rho_thetatheta = -rho * (1.0 + r2) / onemr2;

    // Gamma^rho_{phi phi} = -rho*st^2 - rho^2*st^2 * ApA
    //  = -rho*st^2 * (1 + rho*ApA) = -rho*st^2*(1+rho^2)/(1-rho^2)
    G.G_rho_phiphi = -rho * st * st * (1.0 + r2) / onemr2;

    // Gamma^theta_{rho theta} = 1/rho + ApA + 0 - 0 = 1/rho + ApA
    //  = 1/rho + 2*rho/(1-rho^2) = (1-rho^2 + 2*rho^2) / (rho*(1-rho^2))
    //  = (1+rho^2)/(rho*(1-rho^2))
    G.G_theta_rhotheta = (1.0 + r2) / (rho * onemr2);

    // Gamma^theta_{phi phi} = -st*ct + 0 + 0 - 0 = -st*ct
    G.G_theta_phiphi = -st * ct;

    // Gamma^phi_{rho phi} = 1/rho + ApA = (1+rho^2)/(rho*(1-rho^2))
    G.G_phi_rhophi = (1.0 + r2) / (rho * onemr2);

    // Gamma^phi_{theta phi} = ct/st + 0 = ct/st
    G.G_phi_thetaphi = (std::abs(st) > 1e-14) ? ct / st : 0.0;

    return G;
}

// ---- ScalarEigenmode ----

ScalarEigenmode::ScalarEigenmode(int ell, double Delta, int n_rad)
    : ell_(ell), n_rad_(n_rad), Delta_(Delta),
      omega0_(Delta + 2.0*n_rad + ell),
      m_sq_(Delta * (Delta - 3.0))
{
}

double ScalarEigenmode::f_rho(double rho) const {
    // For n_rad = 0:
    // f(rho) = [2*rho/(1+rho^2)]^ell * [(1-rho^2)/(1+rho^2)]^Delta
    if (n_rad_ != 0) {
        // General case: need hypergeometric function
        // For now, only implement n_rad = 0
        return 0.0;
    }
    double r2 = rho * rho;
    double s = 1.0 + r2;
    double sinxb = 2.0 * rho / s;   // sin(x_bar)
    double cosxb = (1.0 - r2) / s;  // cos(x_bar) = Omega

    return pow(sinxb, ell_) * pow(cosxb, Delta_);
}

double ScalarEigenmode::f_xbar(double xbar) const {
    if (n_rad_ != 0) return 0.0;
    return pow(sin(xbar), ell_) * pow(cos(xbar), Delta_);
}

double ScalarEigenmode::phi(double t, double rho, double theta, double epsilon) const {
    double Pl = spectral::Legendre::P(ell_, cos(theta));
    return epsilon * f_rho(rho) * Pl * cos(omega0_ * t);
}

double ScalarEigenmode::phi_hat(double t, double rho, double theta, double epsilon) const {
    // phi_hat = phi / Omega^(Delta/2)
    double Om = AdS4Background::Omega(rho);
    if (Om < 1e-30) return 0.0;
    return phi(t, rho, theta, epsilon) / pow(Om, Delta_ / 2.0);
}

double verifyEigenmode(const ScalarEigenmode& mode, int N_rho, int N_theta) {
    // Verify Box phi - m^2 phi = 0 on AdS4 background
    // Box phi = -1/N^2 (d_t^2 phi) + spatial Laplacian
    // On AdS4 with the eigenmode: phi = f(rho) P_ell(cos theta) cos(omega t)
    // Box phi = [omega^2/N^2 * f - spatial terms] * P_ell * cos(omega t) - m^2 phi
    // Actually, let's work in x_bar coordinates where the equation is simpler.
    //
    // In x_bar coordinates: ds^2 = (1/cos^2 x_bar)[-dt^2 + dx_bar^2 + sin^2 x_bar dOmega_2^2]
    //
    // The d'Alembertian on this metric gives (after separating phi = f(x) P_l(cos th) cos(wt)):
    //
    //   cos^2(x) f'' + 2 cot(x) f' + [w^2 cos^2(x) - l(l+1) cos^2(x)/sin^2(x) - m^2] f = 0
    //
    // Dividing by cos^2(x):
    //   f'' + (2 cot(x)/cos^2(x)) f' + [w^2 - l(l+1)/sin^2(x) - m^2/cos^2(x)] f = 0
    //
    // where 2 cot(x)/cos^2(x) = 2/(sin(x)*cos(x))
    //
    // Derivation: Box phi = -(cos^2 x)(d_t^2 phi) + (cos^4 x / sin^2 x) d_x(sin^2 x / cos^2 x * d_x phi)
    //                      + (cos^2 x / sin^2 x) Delta_{S^2} phi
    // The radial part expands to: cos^2(x) f'' + 2 cot(x) f'

    // We verify numerically using Chebyshev on [eps, pi/2 - eps]
    double xbar_min = 0.02;
    double xbar_max = PI / 2.0 - 0.02;
    spectral::Chebyshev cheb(N_rho, xbar_min, xbar_max);

    double omega = mode.omega0();
    double m2 = mode.mass_sq();
    int ell = mode.ell();

    // Evaluate f at grid points
    std::vector<double> fvals(cheb.size());
    for (int j = 0; j < cheb.size(); j++) {
        fvals[j] = mode.f_xbar(cheb.grid(j));
    }

    // Compute f' and f''
    std::vector<double> df(cheb.size()), d2f(cheb.size());
    cheb.differentiate(fvals.data(), df.data());
    cheb.differentiate2(fvals.data(), d2f.data());

    // Check residual of the ODE:
    // f'' + 2/(sin x cos x) f' + [omega^2 - ell(ell+1)/sin^2(x) - m^2/cos^2(x)] f = 0
    double max_res = 0.0;
    for (int j = 1; j < cheb.size() - 1; j++) {  // skip boundary points
        double xb = cheb.grid(j);
        double sx = sin(xb), cx = cos(xb);
        if (std::abs(sx) < 1e-10 || std::abs(cx) < 1e-10) continue;

        double coeff1 = 2.0 / (sx * cx);  // 2/(sin x cos x)
        double pot = omega*omega - (double)(ell*(ell+1))/(sx*sx) - m2/(cx*cx);

        double res = d2f[j] + coeff1 * df[j] + pot * fvals[j];

        // Normalize by the magnitude of the largest term
        double scale = std::max({std::abs(d2f[j]), std::abs(coeff1 * df[j]),
                                  std::abs(pot * fvals[j]), 1e-30});
        double rel_res = std::abs(res) / scale;

        max_res = std::max(max_res, rel_res);
    }

    return max_res;
}

} // namespace geometry
