#include <ComputeWAXPBY.hpp>
#include <KernelWrappers.h>
#include "VectorOptimizationDataTx.hpp"

int ComputeWAXPBY(const local_int_t n, const double alpha, const Vector & x,
    const double beta, const Vector & y, Vector & w, bool & isOptimized,
    bool copyIn, bool copyOut)
{
  isOptimized = true;
  const double* x_d;
  const double* y_d;
  double* w_d = ((VectorOptimizationDataTx*)w.optimizationData)->devicePtr;
  if (copyIn) {
    x_d = transferDataToGPU(x);
    y_d = transferDataToGPU(y);
  } else {
    x_d = ((VectorOptimizationDataTx*)x.optimizationData)->devicePtr;
    y_d = ((VectorOptimizationDataTx*)y.optimizationData)->devicePtr;
  }
  launchComputeWAXPBY(n, alpha, x_d, beta, y_d, w_d);
  if (copyOut) {
    transferDataFromGPU(w);
  }
  return 0;
}

