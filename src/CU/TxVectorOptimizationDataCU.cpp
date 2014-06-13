#include <CU/TxVectorOptimizationDataCU.hpp>

#include <iostream>
#include <cuda_runtime.h>
#include <CU/chkcudaerror.hpp>

#include <Vector.hpp>
#include <CU/KernelWrappers.h>

TxVectorOptimizationDataCU::~TxVectorOptimizationDataCU() {
  if (devicePtr) {
    cudaFree(devicePtr);
  }
}

void TxVectorOptimizationDataCU::ZeroVector(int n) {
  launchZeroVector(devicePtr, n);
}

void TxVectorOptimizationDataCU::allocateResources(int n) {
  freeResources();
  cudaError_t err = cudaMalloc((void**)&devicePtr, n * sizeof(double));
  CHKCUDAERR(err);
}

void TxVectorOptimizationDataCU::freeResources() {
  if (devicePtr) {
    cudaError_t err = cudaFree(devicePtr);
    CHKCUDAERR(err);
  }
}

void TxVectorOptimizationDataCU::transferDataToDevice(const Vector& v) {
  cudaError_t err =
      cudaMemcpy(devicePtr, v.values, sizeof(double) * v.localLength,
                 cudaMemcpyHostToDevice);
  CHKCUDAERR(err);
}

void TxVectorOptimizationDataCU::transferDataFromDevice(Vector& v) {
  cudaError_t err = 
      cudaMemcpy(v.values, devicePtr, sizeof(double) * v.localLength,
                 cudaMemcpyDeviceToHost);
  CHKCUDAERR(err);
}

void* TxVectorOptimizationDataCU::getDevicePtr() {
  return devicePtr;
}

int TxVectorOptimizationDataCU::computeWAXPBY(
    int n, double alpha, const void* x, double beta, const void* y, void* w) const {
  launchComputeWAXPBY(n, alpha, (const double*)x, beta, (const double*)y, (double*)w);
  return 0;
}

void TxVectorOptimizationDataCU::computeDotProduct(
    int n, const void* x, const void* y, double* result) const {
  launchComputeDotProduct(n, (const double*)x, (const double*)y, result);
}

void TxVectorOptimizationDataCU::copyDeviceData(void* dest, int numEntries)
{
  cudaError_t err = cudaMemcpy(dest, devicePtr, numEntries * sizeof(double),
      cudaMemcpyDeviceToDevice);
  CHKCUDAERR(err);
}

TxVectorOptimizationDataCU* TxVectorOptimizationDataCU::create()
{
  return new TxVectorOptimizationDataCU;
}
