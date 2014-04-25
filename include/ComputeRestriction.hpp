#ifndef COMPUTE_RESTRICTION_HPP
#define COMPUTE_RESTRICTION_HPP

#include "Vector.hpp"
#include "SparseMatrix.hpp"

int ComputeRestriction(const SparseMatrix& Af, const Vector& xf, bool copyIn = true,
                       bool copyOut = true);

#endif
