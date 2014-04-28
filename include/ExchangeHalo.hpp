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
// modified by Dominic Meiser, Tech-X corporation
//
// ***************************************************
//@HEADER

#ifndef EXCHANGEHALO_HPP
#define EXCHANGEHALO_HPP

#include "SparseMatrix.hpp"
#include "Vector.hpp"

struct DataTransfer_;
typedef struct DataTransfer_* DataTransfer;

/*!
  Initiates communication of data that is at the border of the part of
  the domain assigned to this processor.  Data is not guaranteed to be
  received until EndExchangeHalo finishes.

  @param[in]    A The known system matrix
  @param[inout] x On entry: the local vector entries followed by entries
                to be communicated. It is save to modify the local data
                after BeginExchangeHalo returns.  It is assumed that the
                GPU data in x is up to date.
  @return       A handle to internal data transfer information needed
                for completing the halo exchange by means of
                EndExchangeHalo.

  @sa EndExchangeHalo
 */
DataTransfer BeginExchangeHalo(const SparseMatrix & A, Vector & x);

/**
   Finalizes halo data exchange.
 
   @param[in]    A The known system matrix
   @param[inout] x On exit: the local vector entries followed by entries
                 received from other processes.  Only the data on the
                 GPU is updated.
   @param[in]    transfer Handle to data transfer data obtained by call
                 to BeginExchangeHalo.

   @sa BeginExchangeHalo
  */
void EndExchangeHalo(const SparseMatrix & A, Vector & x, DataTransfer transfer);

/**
 * For convenience we provide a declaration of the original ExchangeHalo
 * */
void ExchangeHalo(const SparseMatrix& A, Vector& x);

#endif // EXCHANGEHALO_HPP
