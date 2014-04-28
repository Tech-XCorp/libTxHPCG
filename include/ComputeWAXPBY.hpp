#ifndef COMPUTE_AXPBY_HPP
#define COMPUTE_AXPBY_HPP

#include "Vector.hpp"

int ComputeWAXPBY(const local_int_t n, const double alpha, const Vector & x,
    const double beta, const Vector & y, Vector & w, bool & isOptimized,
    bool copyIn = true, bool copyOut = true);

#endif
