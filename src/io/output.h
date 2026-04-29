// output.h — I/O utilities for branch data and solution fields
#pragma once

#include <string>
#include <vector>
#include <cstdio>
#include "solver/oscillon_driver.h"
#include "diagnostics/diagnostics.h"

namespace io {

// Branch point with diagnostics for CSV output
struct BranchRow {
    int index;
    double w;
    double omega;
    double residual;
    int newton_iters;
    double E_scalar;
    double phi_max;
    double max_trK;
    double max_evol;
    double dNhat_dr;
};

// Write branch data to CSV
void writeBranchCSV(const std::string& filename,
                    const std::vector<BranchRow>& rows,
                    int ell, double Delta, double Lambda,
                    int N_nuc, int N_shell, int N_theta, int N_t);

// Write resolution convergence data to CSV
struct ConvergenceRow {
    int N_nuc, N_shell, N_theta, N_t;
    int state_size;
    double omega;
    double E_scalar;
    double max_trK;
    double max_evol;
    double residual;
    double time_seconds;
};

void writeConvergenceCSV(const std::string& filename,
                         const std::vector<ConvergenceRow>& rows,
                         int ell, double Delta, double Lambda, double epsilon);

} // namespace io
