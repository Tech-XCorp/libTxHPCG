#ifndef COMPUTE_DOT_PRODUCT_HPP
#define COMPUTE_DOT_PRODUCT_HPP

#include "Vector.hpp"
int ComputeDotProduct(const local_int_t n, const Vector & x, const Vector & y,
    double & result, double & time_allreduce, bool & isOptimized,
    bool copyIn = true);

#endif
