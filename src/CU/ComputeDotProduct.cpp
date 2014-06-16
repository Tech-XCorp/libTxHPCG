#include "ComputeDotProduct.hpp"
#include "KernelWrappers.h"
#include "TxVectorOptimizationDataBase.hpp"
#ifndef HPCG_NOMPI
#include <mpi.h>
#include "mytimer.hpp"
#endif


int ComputeDotProduct(const local_int_t n, const Vector & x, const Vector & y,
    double & result, double & time_allreduce, bool & isOptimized, bool copyIn) {
  isOptimized = true;
  const void* x_d;
  const void* y_d;
  if (copyIn) {
    x_d = (double*)transferDataToDevice(x);
    y_d = (double*)transferDataToDevice(y);
  } else {
    x_d = ((TxVectorOptimizationDataBase*)x.optimizationData)->getDevicePtr();
    y_d = ((TxVectorOptimizationDataBase*)y.optimizationData)->getDevicePtr();
  }
  double local_result;
  ((TxVectorOptimizationDataBase*)x.optimizationData)->computeDotProduct(n, x_d, y_d, &local_result);

#ifndef HPCG_NOMPI
  double t0 = mytimer();
  double global_result = 0.0;
  MPI_Allreduce(&local_result, &global_result, 1, MPI_DOUBLE, MPI_SUM,
      MPI_COMM_WORLD);
  time_allreduce += mytimer() - t0;
  result = global_result;
#else
  result = local_result;
#endif
  return 0;
}

