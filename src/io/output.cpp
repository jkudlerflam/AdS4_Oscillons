// output.cpp — I/O utilities implementation
#include "io/output.h"
#include <cstdio>

namespace io {

void writeBranchCSV(const std::string& filename,
                    const std::vector<BranchRow>& rows,
                    int ell, double Delta, double Lambda,
                    int N_nuc, int N_shell, int N_theta, int N_t)
{
    FILE* f = fopen(filename.c_str(), "w");
    if (!f) {
        fprintf(stderr, "ERROR: Cannot open %s for writing\n", filename.c_str());
        return;
    }

    // Header with metadata
    fprintf(f, "# AdS4 Oscillon Branch Data\n");
    fprintf(f, "# ell=%d, Delta=%.1f, Lambda=%.1f\n", ell, Delta, Lambda);
    fprintf(f, "# N_nuc=%d, N_shell=%d, N_theta=%d, N_t=%d\n",
            N_nuc, N_shell, N_theta, N_t);
    fprintf(f, "index,w,omega,residual,newton_iters,"
               "E_scalar,phi_max,max_trK,max_evol,dNhat_dr\n");

    for (const auto& r : rows) {
        fprintf(f, "%d,%.15e,%.15e,%.6e,%d,"
                   "%.15e,%.15e,%.6e,%.6e,%.15e\n",
                r.index, r.w, r.omega, r.residual, r.newton_iters,
                r.E_scalar, r.phi_max, r.max_trK, r.max_evol, r.dNhat_dr);
    }

    fclose(f);
    printf("Branch data written to %s (%d rows)\n", filename.c_str(), (int)rows.size());
}

void writeConvergenceCSV(const std::string& filename,
                         const std::vector<ConvergenceRow>& rows,
                         int ell, double Delta, double Lambda, double epsilon)
{
    FILE* f = fopen(filename.c_str(), "w");
    if (!f) {
        fprintf(stderr, "ERROR: Cannot open %s for writing\n", filename.c_str());
        return;
    }

    fprintf(f, "# Resolution Convergence Study\n");
    fprintf(f, "# ell=%d, Delta=%.1f, Lambda=%.1f, eps=%.6f\n", ell, Delta, Lambda, epsilon);
    fprintf(f, "N_nuc,N_shell,N_theta,N_t,state_size,"
               "omega,E_scalar,max_trK,max_evol,residual,time_s\n");

    for (const auto& r : rows) {
        fprintf(f, "%d,%d,%d,%d,%d,"
                   "%.15e,%.15e,%.6e,%.6e,%.6e,%.2f\n",
                r.N_nuc, r.N_shell, r.N_theta, r.N_t, r.state_size,
                r.omega, r.E_scalar, r.max_trK, r.max_evol, r.residual, r.time_seconds);
    }

    fclose(f);
    printf("Convergence data written to %s (%d rows)\n", filename.c_str(), (int)rows.size());
}

} // namespace io
