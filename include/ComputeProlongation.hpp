#ifndef COMPUTE_PROLONGATION_HPP
#define COMPUTE_PROLONGATION_HPP

#include "Vector.hpp"
#include "SparseMatrix.hpp"

int ComputeProlongation(const SparseMatrix& Af, Vector& xf, bool copyIn = true,
                        bool copyOut = true);

#endif

