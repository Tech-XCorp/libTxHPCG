#include <ComputeWAXPBY.hpp>
#include "TxVectorOptimizationDataBase.hpp"

int ComputeWAXPBY(const local_int_t n, const double alpha, const Vector & x,
    const double beta, const Vector & y, Vector & w, bool & isOptimized,
    bool copyIn, bool copyOut)
{
  isOptimized = true;
  const void* x_d;
  const void* y_d;
  void* w_d = ((TxVectorOptimizationDataBase*)w.optimizationData)->getDevicePtr();
  if (copyIn) {
    transferDataToDevice(x);
    x_d = ((TxVectorOptimizationDataBase*)x.optimizationData)->getDevicePtr();
    transferDataToDevice(y);
    y_d = ((TxVectorOptimizationDataBase*)y.optimizationData)->getDevicePtr();
  } else {
    x_d = ((TxVectorOptimizationDataBase*)x.optimizationData)->getDevicePtr();
    y_d = ((TxVectorOptimizationDataBase*)y.optimizationData)->getDevicePtr();
  }
  ((const TxVectorOptimizationDataBase*)w.optimizationData)->computeWAXPBY(
    n, alpha, x_d, beta, y_d, w_d);
  if (copyOut) {
    transferDataFromDevice(w);
  }
  return 0;
}

