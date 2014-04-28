#ifndef MATRIX_OPTIMIZATION_DATA_HPP
#define MATRIX_OPTIMIZATION_DATA_HPP

#include <cusparse_v2.h>
#include <cuGelus.h>
#include <gelusBase.h>
#include "CPCSR.hpp"

#include "SparseMatrix.hpp"
#include "Vector.hpp"

/**
 * @brief Class with Tx implementations and data for HPCG
 *
 * Instances of this class are stored in the optimizationData
 * pointer of the HPCG SparseMatrix class.
 *
 * @sa SparseMatrix, Vector, VectorOptimizationDataTx
 * */
class MatrixOptimizationDataTx {
 public:
  MatrixOptimizationDataTx();
  ~MatrixOptimizationDataTx();
  int setupLocalMatrixOnGPU(SparseMatrix &A);

  int ComputeSPMV(const SparseMatrix& A, Vector& x, Vector& y,
                  bool copyIn = true, bool copyOut = true);
  int ComputeSYMGS(const SparseMatrix& A, const Vector& x, Vector& y,
                   int numberOfSmootherSteps = 1, bool copyIn = true,
                   bool copyOut = true);
  int ComputeProlongation(const SparseMatrix& Af, Vector& xf, bool copyIn,
                          bool copyOut);
  int ComputeRestriction(const SparseMatrix& Af, const Vector& rf, bool copyIn,
                         bool copyOut);
  double* getSendBuffer_d() { return sendBuffer_d; }
  int* getElementsToSend_d() { return elementsToSend; } 

 private:
  cusparseHandle_t handle;              //!< cusparse context
  cusparseMatDescr_t matDescr;          //!< matrix Description
  cusparseHybMat_t localMatrix;         //!< cusparse matrix
  cugelusSorIterationData_t gsContext;  //!< gelus GS context
  CPCSR scatterFromHalo;                //!< sparse matrix needed for receiving
                                        //   data from other processes
  local_int_t* f2c;                     //!< restriction/prolongation operator
#ifndef HPCG_NOMPI
  local_int_t* elementsToSend;          //!< Indices of elements to send
  double* sendBuffer_d;                 //!< Send buffer on device.
#endif

  double* workvector;  //!< Work space for SYMGS

  // Disallow copy and assignment
  MatrixOptimizationDataTx(const MatrixOptimizationDataTx&);
  MatrixOptimizationDataTx& operator=(const MatrixOptimizationDataTx&);
};

void dumpMatrix(std::ostream& s, const std::vector<int>& i,
                const std::vector<int>& j, const std::vector<double>& a);

#endif

