#include <VectorOptimizationDataTx.hpp>

#include <iostream>
#include <cuda_runtime.h>
#include <chkcudaerror.hpp>

#include <Vector.hpp>
#include <src/KernelWrappers.h>

double *transferDataToGPU(const Vector &v) {
  VectorOptimizationDataTx *opt =
      (VectorOptimizationDataTx *)v.optimizationData;
  cudaError_t err =
      cudaMemcpy(opt->devicePtr, v.values, sizeof(double) * v.localLength,
                 cudaMemcpyHostToDevice);
  CHKCUDAERR(err);
  return opt->devicePtr;
}

void transferDataFromGPU(const Vector &v) {
  VectorOptimizationDataTx *opt =
      (VectorOptimizationDataTx *)v.optimizationData;
  cudaError_t err = 
      cudaMemcpy(v.values, opt->devicePtr, sizeof(double) * v.localLength,
                 cudaMemcpyDeviceToHost);
  CHKCUDAERR(err);
}

void VectorOptimizationDataTx::ZeroVector(int n) {
  launchZeroVector(devicePtr, n);
}
