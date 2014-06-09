
//@HEADER
// ***************************************************
//
// HPCG: High Performance Conjugate Gradient Benchmark
//
// Contact:
// Michael A. Heroux ( maherou@sandia.gov)
// Jack Dongarra     (dongarra@eecs.utk.edu)
// Piotr Luszczek    (luszczek@eecs.utk.edu)
//
// ***************************************************
//@HEADER

/*!
 @file ComputeSPMV.cpp

 HPCG routine
 */

#include <cuda_runtime.h>
#include <iostream>

#include "ComputeSPMV.hpp"
#include "ComputeSPMV_ref.hpp"
#include "TxVectorOptimizationDataBase.hpp"
#include "TxMatrixOptimizationDataBase.hpp"

/*!
  Routine to compute sparse matrix vector product y = Ax where:
  Precondition: First call exchange_externals to get off-processor values of x

  This routine calls the reference SpMV implementation by default, but
  can be replaced by a custom, optimized routine suited for
  the target system.

  @param[in]  A the known system matrix
  @param[in]  x the known vector
  @param[out] y the On exit contains the result: Ax.

  @return returns 0 upon success and non-zero otherwise

  @see ComputeSPMV_ref
*/
int ComputeSPMV(const SparseMatrix& A, Vector& x, Vector& y, bool copyIn, bool copyOut) {
  TxMatrixOptimizationDataBase* optData =
      (TxMatrixOptimizationDataBase*)A.optimizationData;
  return optData->ComputeSPMV(A, x, y, copyIn, copyOut);
}

