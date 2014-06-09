#include <CU/VectorOptimizationDataTx.hpp>

#include <iostream>
#include <cuda_runtime.h>
#include <CU/chkcudaerror.hpp>

#include <Vector.hpp>
#include <CU/KernelWrappers.h>

void VectorOptimizationDataTx::ZeroVector(int n) {
  launchZeroVector(devicePtr, n);
}

void VectorOptimizationDataTx::freeResources() {
  if (devicePtr) {
    cudaFree(devicePtr);
  }
}

void VectorOptimizationDataTx::transferDataToDevice(const Vector& v) {
  cudaError_t err =
      cudaMemcpy(devicePtr, v.values, sizeof(double) * v.localLength,
                 cudaMemcpyHostToDevice);
  CHKCUDAERR(err);
}

void VectorOptimizationDataTx::transferDataFromDevice(Vector& v) {
  cudaError_t err = 
      cudaMemcpy(v.values, devicePtr, sizeof(double) * v.localLength,
                 cudaMemcpyDeviceToHost);
  CHKCUDAERR(err);
}

void* VectorOptimizationDataTx::getDevicePtr() {
  return devicePtr;
}

int VectorOptimizationDataTx::computeWAXPBY(
    int n, double alpha, const void* x, double beta, const void* y, void* w) const {
  launchComputeWAXPBY(n, alpha, (const double*)x, beta, (const double*)y, (double*)w);
  return 0;
}

void VectorOptimizationDataTx::computeDotProduct(
    int n, const void* x, const void* y, double* result) const {
  launchComputeDotProduct(n, (const double*)x, (const double*)y, result);
}
