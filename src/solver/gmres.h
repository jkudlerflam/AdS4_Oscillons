// gmres.h — GMRES iterative solver
#pragma once
#include <vector>
#include <functional>

namespace solver {

// GMRES(m) solver: solve A*x = b where A is given by matvec function
// matvec(x, Ax): computes Ax = A*x
// Returns number of iterations, or -1 if failed
int gmres(
    std::function<void(const double*, double*)> matvec,
    const double* b,
    double* x,          // initial guess on input, solution on output
    int n,              // system size
    int restart = 50,   // restart parameter
    double tol = 1e-8,
    int max_iter = 500
);

} // namespace solver
