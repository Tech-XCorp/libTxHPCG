#include "MatrixOptimizationDataTx.hpp"
#include "SparseMatrix.hpp"
#include "KernelWrappers.h"

#ifndef HPCG_NOMPI
#include "ExchangeHalo.hpp"
#include "mpi.h"
#endif

MatrixOptimizationDataTx::MatrixOptimizationDataTx()
    : handle(0), matDescr(0), localMatrix(0), gsContext(0), f2c(0),
      workvector(0) {
  cusparseStatus_t err = cusparseCreate(&handle);
  CHKCUSPARSEERR(err);
#ifndef HPCG_NOMPI
  elementsToSend = 0;
#endif
}

MatrixOptimizationDataTx::~MatrixOptimizationDataTx() {
  if (handle) {
    cusparseDestroy(handle);
    handle = 0;
  }
  if (matDescr) {
    cusparseDestroyMatDescr(matDescr);
    matDescr = 0;
  }
  if (localMatrix) {
    cusparseDestroyHybMat(localMatrix);
    localMatrix = 0;
  }
  if (gsContext) {
    cugelusDestroySorIterationData(gsContext);
    gsContext = 0;
  }
  if (f2c) {
    CHKCUDAERR(cudaFree(f2c));
  }
  if (workvector) {
    CHKCUDAERR(cudaFree(workvector));
  }
#ifndef HPCG_NOMPI
  CHKCUDAERR(cudaFree(elementsToSend));
  CHKCUDAERR(cudaFree(sendBuffer_d));
#endif
}

int MatrixOptimizationDataTx::setupLocalMatrixOnGPU(SparseMatrix& A) {
  std::vector<local_int_t> i(A.localNumberOfRows + 1, 0);
  // Slight overallocation for these arrays
  std::vector<local_int_t> j;
  j.reserve(A.localNumberOfNonzeros);
  std::vector<double> a;
  a.reserve(A.localNumberOfNonzeros);
  scatterFromHalo.setNumRows(A.localNumberOfRows);
  scatterFromHalo.setNumCols(A.localNumberOfColumns);
  scatterFromHalo.clear();
  // We're splitting the matrix into diagonal and off-diagonal block to
  // enable overlapping of computation and communication.
  i[0] = 0;
  for (local_int_t m = 0; m < A.localNumberOfRows; ++m) {
    local_int_t nonzerosInRow = 0;
    for (local_int_t n = 0; n < A.nonzerosInRow[m]; ++n) {
      local_int_t col = A.mtxIndL[m][n];
      if (col < A.localNumberOfRows) {
        j.push_back(col);
        a.push_back(A.matrixValues[m][n]);
        ++nonzerosInRow;
      } else {
        scatterFromHalo.addEntry(m, col, A.matrixValues[m][n]);
      }
    }
    i[m + 1] = i[m] + nonzerosInRow;
  }

  // Setup SpMV data on GPU
  cudaError_t err = cudaSuccess;
  int* i_d;
  err = cudaMalloc((void**)&i_d, i.size() * sizeof(i[0]));
  CHKCUDAERR(err);
  err = cudaMemcpy(i_d, &i[0], i.size() * sizeof(i[0]), cudaMemcpyHostToDevice);
  CHKCUDAERR(err);
  int* j_d;
  err = cudaMalloc((void**)&j_d, j.size() * sizeof(j[0]));
  CHKCUDAERR(err);
  err = cudaMemcpy(j_d, &j[0], j.size() * sizeof(j[0]), cudaMemcpyHostToDevice);
  CHKCUDAERR(err);
  double* a_d;
  err = cudaMalloc((void**)&a_d, a.size() * sizeof(a[0]));
  CHKCUDAERR(err);
  err = cudaMemcpy(a_d, &a[0], a.size() * sizeof(a[0]), cudaMemcpyHostToDevice);
  CHKCUDAERR(err);
  cusparseStatus_t cerr = CUSPARSE_STATUS_SUCCESS;
  cerr = cusparseCreateMatDescr(&matDescr);
  CHKCUSPARSEERR(cerr);
  cerr = cusparseSetMatIndexBase(matDescr, CUSPARSE_INDEX_BASE_ZERO);
  CHKCUSPARSEERR(cerr);
  cerr = cusparseSetMatType(matDescr, CUSPARSE_MATRIX_TYPE_GENERAL);
  CHKCUSPARSEERR(cerr);
  cerr = cusparseCreateHybMat(&localMatrix);
  CHKCUSPARSEERR(cerr);
  cerr = cusparseDcsr2hyb(handle, A.localNumberOfRows, A.localNumberOfColumns,
                          matDescr, a_d, i_d, j_d, localMatrix, 27,
                          CUSPARSE_HYB_PARTITION_USER);
  CHKCUSPARSEERR(cerr);

#ifndef HPCG_NOMPI
  err = cudaMalloc((void**)&elementsToSend,
                   A.totalToBeSent * sizeof(*elementsToSend));
  CHKCUDAERR(err);
  err = cudaMemcpy(elementsToSend, A.elementsToSend,
                   A.totalToBeSent * sizeof(*elementsToSend),
                   cudaMemcpyHostToDevice);
  CHKCUDAERR(err);
  err = cudaMalloc((void**)&sendBuffer_d, A.totalToBeSent * sizeof(double));
  CHKCUDAERR(err);
#endif

  // Set up the GS data.
  gelusStatus_t gerr = GELUS_STATUS_SUCCESS;
  gelusSolveDescription_t solveDescr;
  gerr = gelusCreateSolveDescr(&solveDescr);
  CHKGELUSERR(gerr);
  gerr = gelusSetSolveOperation(solveDescr, GELUS_OPERATION_NON_TRANSPOSE);
  CHKGELUSERR(gerr);
  gerr = gelusSetSolveFillMode(solveDescr, GELUS_FILL_MODE_FULL);
  CHKGELUSERR(gerr);
  gerr = gelusSetSolveStorageFormat(solveDescr, GELUS_STORAGE_FORMAT_HYB);
  CHKGELUSERR(gerr);
  gerr = gelusSetOptimizationLevel(solveDescr, GELUS_OPTIMIZATION_LEVEL_THREE);
  CHKGELUSERR(gerr);

  gerr = cugelusCreateSorIterationData(&gsContext);
  CHKGELUSERR(gerr);

#ifdef HPCG_DEBUG
  std::cout << A.localNumberOfRows << std::endl;
  std::cout << A.localNumberOfColumns << std::endl;
  std::cout << A.localNumberOfNonzeros << std::endl;
  int myrank;
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
  if (myrank == 0) {
    dumpMatrix(std::cout, i, j, a);
  }
#endif

  gerr = cugelusDcsrsor_iteration_analysis(
      A.localNumberOfRows, solveDescr, GELUS_SOR_SYMMETRIC, 1.0,
      &i[0], &j[0], &a[0], gsContext);
  gerr = gelusDestroySolveDescr(solveDescr);
  CHKGELUSERR(gerr);

  if (A.mgData) {
    err =
      cudaMalloc((void**)&f2c, A.mgData->rc->localLength * sizeof(local_int_t));
    CHKCUDAERR(err);
    err = cudaMemcpy(f2c, A.mgData->f2cOperator,
        A.mgData->rc->localLength * sizeof(local_int_t),
        cudaMemcpyHostToDevice);
    CHKCUDAERR(err);
  }

  err = cudaMalloc((void**)&workvector, A.localNumberOfRows * sizeof(double));
  CHKCUDAERR(err);

  return (int)cerr | (int)gerr | (int)err;
}

int MatrixOptimizationDataTx::ComputeSPMV(const SparseMatrix& A, Vector& x,
    Vector& y, bool copyIn,
    bool copyOut) {
  double* x_dev = 0;
  double* y_dev = 0;
  if (copyIn) {
    x_dev = transferDataToGPU(x);
    y_dev = transferDataToGPU(y);
  } else {
    x_dev = ((VectorOptimizationDataTx*)x.optimizationData)->devicePtr;
    y_dev = ((VectorOptimizationDataTx*)y.optimizationData)->devicePtr;
  }
#ifndef HPCG_NOMPI
  DataTransfer transfer = BeginExchangeHalo(A, x);
#endif
  double alpha = 1.0;
  double beta = 0.0;
  cusparseStatus_t cerr;
  cerr = cusparseDhybmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
      matDescr, localMatrix, x_dev, &beta, y_dev);
  CHKCUSPARSEERR(cerr);
#ifndef HPCG_NOMPI
  EndExchangeHalo(A, x, transfer);
  scatterFromHalo.spmv(x_dev, y_dev);
#endif
  if (copyOut) {
    transferDataFromGPU(y);
  }
  return 0;
}

int MatrixOptimizationDataTx::ComputeSYMGS(const SparseMatrix &A,
                                           const Vector &r, Vector &x,
                                           int numberOfSmootherSteps,
                                           bool copyIn, bool copyOut) {
  const double* r_dev = 0;  
  double* x_dev = 0;
  if (copyIn) {
    x_dev = transferDataToGPU(x);
    r_dev = transferDataToGPU(r);
  } else {
    x_dev = ((VectorOptimizationDataTx*)x.optimizationData)->devicePtr;
    r_dev = ((VectorOptimizationDataTx*)r.optimizationData)->devicePtr;
  }
#ifndef HPCG_NOMPI
  DataTransfer transfer = BeginExchangeHalo(A, x);
  EndExchangeHalo(A, x, transfer);
#endif
  launchDeviceCopy(workvector, r_dev, r.localLength);
  scatterFromHalo.spmv(x_dev, workvector, -1, 1);
  gelusStatus_t err =
      cugelusDcsrsor_iterate(x_dev, workvector, GELUS_SOR_INITIAL_GUESS_NONZERO,
                             numberOfSmootherSteps, gsContext);
  CHKGELUSERR(err);
  if (copyOut) {
    transferDataFromGPU(x);
  }
  return 0;
}

int MatrixOptimizationDataTx::ComputeProlongation(const SparseMatrix& Af,
    Vector& xf, bool copyIn, bool copyOut) {
  double* xf_d = 0;
  double* xc_d = 0;
  if (copyIn) {
    xf_d = transferDataToGPU(xf);
    xc_d = transferDataToGPU(*Af.mgData->xc);
  } else {
    xf_d = ((VectorOptimizationDataTx*)xf.optimizationData)->devicePtr;
    xc_d = ((VectorOptimizationDataTx*)Af.mgData->xc->optimizationData)->devicePtr;
  }

  local_int_t nc = Af.mgData->rc->localLength;
  launchProlongationKernel(xf_d, xc_d, nc, f2c); 

  if (copyOut) {
    transferDataFromGPU(xf);
    transferDataFromGPU(*Af.mgData->xc);
  }
  return 0;
}

int MatrixOptimizationDataTx::ComputeRestriction(const SparseMatrix& Af,
                                                 const Vector& rf, bool copyIn,
                                                 bool copyOut) {
  double* Axf_d = 0;
  double* rf_d = 0;
  double* rc_d = 0;
  if (copyIn) {
    Axf_d = transferDataToGPU(*Af.mgData->Axf);
    rf_d = transferDataToGPU(rf);
    rc_d = transferDataToGPU(*Af.mgData->rc);
  } else {
    Axf_d = ((VectorOptimizationDataTx*)Af.mgData->Axf->optimizationData)->devicePtr;
    rf_d = ((VectorOptimizationDataTx*)rf.optimizationData)->devicePtr;
    rc_d = ((VectorOptimizationDataTx*)Af.mgData->rc->optimizationData)->devicePtr;
  }

  local_int_t nc = Af.mgData->rc->localLength;
  launchRestrictionKernel(rc_d, rf_d, Axf_d, nc, f2c); 

  if (copyOut) {
    transferDataFromGPU(*Af.mgData->Axf);
    transferDataFromGPU(rf);
    transferDataFromGPU(*Af.mgData->rc);
  }
  return 0;
}

void dumpMatrix(std::ostream& s, const std::vector<int>& i,
    const std::vector<int>& j, const std::vector<double>& a)
{
  s << "%%MatrixMarket matrix coordinate real general\n";
  int numRows = i.size() - 1;
  int nnz = j.size();
  s << numRows << " " << numRows << " " << nnz << "\n";
  for (int m = 0; m < numRows; ++m) {
    for (int n = i[m]; n < i[m + 1]; ++n) {
      s << m + 1 << " " << j[n] + 1 << " " << a[n] << "\n";
    }
  }
}
