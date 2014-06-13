
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
 @file OptimizeProblem.cpp

 HPCG routine
 */

#include <OptimizeProblem.hpp>

#include <TxMatrixOptimizationDataBase.hpp>
#include <BackendFactory.hpp>
#include <config.h>

int setUpLocalMatrixOnGPU(SparseMatrix& A);
int initializeCusparse(SparseMatrix& A);

/*!
  Optimizes the data structures used for CG iteration to increase the
  performance of the benchmark version of the preconditioned CG algorithm.

  @param[inout] A      The known system matrix, also contains the MG hierarchy in attributes Ac and mgData.
  @param[inout] data   The data structure with all necessary CG vectors preallocated
  @param[inout] b      The known right hand side vector
  @param[inout] x      The solution vector to be computed in future CG iteration
  @param[inout] xexact The exact solution vector

  @return returns 0 upon success and non-zero otherwise

  @see GenerateGeometry
  @see GenerateProblem
*/
int OptimizeProblem(SparseMatrix &A, CGData &data, Vector &b, Vector &x,
                            Vector &xexact) {
  int err = 0;
  SparseMatrix* m = &A;
  while (m) {
    TxMatrixOptimizationDataBase *optimizationData = 
      getMatrixOptimizationData(BACKEND_TO_USE);
    if (!optimizationData) {
      throw std::runtime_error(std::string("Unknown backend \"") +
          std::string(BACKEND_TO_USE) + std::string("\"."));
    }
    err = optimizationData->ingestLocalMatrix(*m);
    m->optimizationData = optimizationData;
    m = m->Ac;
  }
  return err;
}

