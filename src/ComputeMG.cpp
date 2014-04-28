
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
 @file ComputeMG.cpp

 HPCG routine
 */

#include "ComputeMG.hpp"
#include "ComputeMG_ref.hpp"
#include "ComputeRestriction_ref.hpp"
#include "ComputeProlongation_ref.hpp"
#include "ComputeSYMGS.hpp"
#include "ComputeSPMV.hpp"
#include "ComputeRestriction.hpp"
#include "ComputeProlongation.hpp"

/*!
  @param[in] A the known system matrix
  @param[in] r the input vector
  @param[inout] x On exit contains the result of the multigrid V-cycle with r as the RHS, x is the approximation to Ax = r.

  @return returns 0 upon success and non-zero otherwise

  @see ComputeMG_ref
*/
int ComputeMG(const SparseMatrix& A, const Vector& r, Vector& x, bool copyIn,
              bool copyOut) {
  assert(x.localLength==A.localNumberOfColumns); // Make sure x contain space for halo values

  ZeroVector(x); // initialize x to zero

  int ierr = 0;
  if (A.mgData!=0) { // Go to next coarse level if defined
    int numberOfPresmootherSteps = A.mgData->numberOfPresmootherSteps;
    ierr += ComputeSYMGS(A, r, x, numberOfPresmootherSteps, copyIn, false);
    if (ierr!=0) return(ierr);
    ierr = ComputeSPMV(A, x, *A.mgData->Axf, false, false); if (ierr!=0) return(ierr);
    // Perform restriction operation using simple injection
    ierr = ComputeRestriction(A, r, false, false);  if (ierr!=0) return(ierr);
    ierr = ComputeMG(*A.Ac,*A.mgData->rc, *A.mgData->xc, false, false);  if (ierr!=0) return(ierr);
    ierr = ComputeProlongation(A, x, false, true);  if (ierr!=0) return(ierr);
    int numberOfPostsmootherSteps = A.mgData->numberOfPostsmootherSteps;
    ierr += ComputeSYMGS(A, r, x, numberOfPostsmootherSteps, true, copyOut);
    if (ierr!=0) return(ierr);
  }
  else {
    ierr = ComputeSYMGS(A, r, x, 1, copyIn, copyOut);
    if (ierr!=0) return(ierr);
  }
  return(0);
}

