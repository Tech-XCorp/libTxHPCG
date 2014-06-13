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
// modified by Dominic Meiser, Tech-X Corporation.
//
// ***************************************************
//@HEADER
#ifndef HPCG_NOMPI
#include <mpi.h>
#include "Geometry.hpp"
#include "ExchangeHalo.hpp"
#include <cstdlib>
#include <vector>
#include "KernelWrappers.h"
#include <CU/chkcudaerror.hpp>
#include <CU/TxMatrixOptimizationDataCU.hpp>
#include <CU/TxVectorOptimizationDataCU.hpp>
#include <config.h>

struct DataTransfer_ {
  std::vector<MPI_Request> requests;
};

DataTransfer BeginExchangeHalo(const SparseMatrix &A, Vector &x) {
  local_int_t localNumberOfRows = A.localNumberOfRows;
  int num_neighbors = A.numberOfSendNeighbors;
  local_int_t *receiveLength = A.receiveLength;
  local_int_t *sendLength = A.sendLength;
  int *neighbors = A.neighbors;
#ifdef HAVE_GPU_AWARE_MPI
  double *sendBuffer =
      ((TxMatrixOptimizationDataCU *)A.optimizationData)->getSendBuffer_d();
#else
  double *sendBuffer = A.sendBuffer;
#endif
  local_int_t totalToBeSent = A.totalToBeSent;

  int size, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int MPI_MY_TAG = 99;

  DataTransfer transfer = new DataTransfer_;
  transfer->requests.resize(2 * num_neighbors);
  MPI_Request *request = transfer->requests.data();

  TxVectorOptimizationDataCU *vOptData =
    (TxVectorOptimizationDataCU*)x.optimizationData;
#ifdef HAVE_GPU_AWARE_MPI
  double *x_external = (double*)vOptData->getDevicePtr() + localNumberOfRows;
#else
  double *x_external = x.values + localNumberOfRows;
#endif

  cudaError_t cerr = cudaDeviceSynchronize();
  CHKCUDAERR(cerr);
  for (int i = 0; i < num_neighbors; i++) {
    local_int_t n_recv = receiveLength[i];
    MPI_Irecv(x_external, n_recv, MPI_DOUBLE, neighbors[i], MPI_MY_TAG,
              MPI_COMM_WORLD, request + i);
    x_external += n_recv;
  }

  TxMatrixOptimizationDataCU *optData =
      (TxMatrixOptimizationDataCU *)A.optimizationData;
  launchScatter(optData->getSendBuffer_d(), (const double*)vOptData->getDevicePtr(),
                optData->getElementsToSend_d(), totalToBeSent);
#ifdef HAVE_GPU_AWARE_MPI
#else
  cerr = cudaMemcpy(sendBuffer, optData->getSendBuffer_d(),
                    totalToBeSent * sizeof(double), cudaMemcpyDeviceToHost);
  CHKCUDAERR(cerr);
#endif
  request = transfer->requests.data() + num_neighbors;
  for (int i = 0; i < num_neighbors; i++) {
    local_int_t n_send = sendLength[i];
    MPI_Isend(sendBuffer, n_send, MPI_DOUBLE, neighbors[i], MPI_MY_TAG,
             MPI_COMM_WORLD, request + i);
    sendBuffer += n_send;
  }
  return transfer;
}

void EndExchangeHalo(const SparseMatrix &A, Vector &x, DataTransfer transfer) {
  MPI_Request *request = transfer->requests.data();
  int num_neighbors = A.numberOfSendNeighbors;
  int err = MPI_Waitall(2 * num_neighbors, request, MPI_STATUSES_IGNORE);
  if (err != MPI_SUCCESS) {
    std::cout << "Error in EndExchangeHalo." << std::endl;
    exit(-1);
  }
  delete transfer;
#ifdef HAVE_GPU_AWARE_MPI
#else
  TxVectorOptimizationDataCU *vOptData =
      (TxVectorOptimizationDataCU *)x.optimizationData;
  cudaError_t cerr;
  local_int_t numRecvd = A.localNumberOfColumns - A.localNumberOfRows;
  cerr = cudaMemcpy((double*)vOptData->getDevicePtr() + A.localNumberOfRows,
                    x.values + A.localNumberOfRows, numRecvd * sizeof(double),
                    cudaMemcpyHostToDevice);
  CHKCUDAERR(cerr);
#endif
}

#endif  // ifndef HPCG_NOMPI
