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
#include <chkcudaerror.hpp>
#include <MatrixOptimizationDataTx.hpp>
#include <VectorOptimizationDataTx.hpp>

struct DataTransfer_ {
  std::vector<MPI_Request> receive_requests;
};

DataTransfer BeginExchangeHalo(const SparseMatrix &A, Vector &x) {
  local_int_t localNumberOfRows = A.localNumberOfRows;
  int num_neighbors = A.numberOfSendNeighbors;
  local_int_t *receiveLength = A.receiveLength;
  local_int_t *sendLength = A.sendLength;
  int *neighbors = A.neighbors;
  double *sendBuffer = A.sendBuffer;
  local_int_t totalToBeSent = A.totalToBeSent;

  double *const xv = x.values;

  int size, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int MPI_MY_TAG = 99;

  DataTransfer transfer = new DataTransfer_;
  transfer->receive_requests.resize(num_neighbors);
  MPI_Request *request = transfer->receive_requests.data();

  double *x_external = (double *)xv + localNumberOfRows;

  for (int i = 0; i < num_neighbors; i++) {
    local_int_t n_recv = receiveLength[i];
    MPI_Irecv(x_external, n_recv, MPI_DOUBLE, neighbors[i], MPI_MY_TAG,
              MPI_COMM_WORLD, request + i);
    x_external += n_recv;
  }

  MatrixOptimizationDataTx *optData =
      (MatrixOptimizationDataTx *)A.optimizationData;
  VectorOptimizationDataTx *vOptData =
    (VectorOptimizationDataTx*)x.optimizationData;
  launchScatter(optData->getSendBuffer_d(), vOptData->devicePtr,
                optData->getElementsToSend_d(), totalToBeSent);
  cudaError_t cerr;
  cerr = cudaMemcpy(sendBuffer, optData->getSendBuffer_d(),
                    totalToBeSent * sizeof(double), cudaMemcpyDeviceToHost);
  CHKCUDAERR(cerr);
  for (int i = 0; i < num_neighbors; i++) {
    local_int_t n_send = sendLength[i];
    MPI_Send(sendBuffer, n_send, MPI_DOUBLE, neighbors[i], MPI_MY_TAG,
             MPI_COMM_WORLD);
    sendBuffer += n_send;
  }
  return transfer;
}

void EndExchangeHalo(const SparseMatrix &A, Vector &x, DataTransfer transfer) {
  MPI_Status status;
  MPI_Request *request = transfer->receive_requests.data();
  int num_neighbors = A.numberOfSendNeighbors;
  // TODO: Thread this loop
  for (int i = 0; i < num_neighbors; i++) {
    if (MPI_Wait(request + i, &status)) {
      std::exit(-1);  // TODO: have better error exit
    }
  }
  delete transfer;
  VectorOptimizationDataTx *vOptData =
      (VectorOptimizationDataTx *)x.optimizationData;
  cudaError_t cerr;
  local_int_t numRecvd = A.localNumberOfColumns - A.localNumberOfRows;
  cerr = cudaMemcpy(vOptData->devicePtr + A.localNumberOfRows,
                    x.values + A.localNumberOfRows, numRecvd * sizeof(double),
                    cudaMemcpyHostToDevice);
  CHKCUDAERR(cerr);
}


#endif  // ifndef HPCG_NOMPI
