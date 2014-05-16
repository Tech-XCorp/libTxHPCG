#include "ComputeDotProduct.hpp"
#include "KernelWrappers.h"
#include "VectorOptimizationDataTx.hpp"
#ifndef HPCG_NOMPI
#include <mpi.h>
#include "mytimer.hpp"
#endif


int ComputeDotProduct(const local_int_t n, const Vector & x, const Vector & y,
    double & result, double & time_allreduce, bool & isOptimized, bool copyIn) {
  isOptimized = true;
  const double* x_d;
  const double* y_d;
  if (copyIn) {
    x_d = transferDataToGPU(x);
    y_d = transferDataToGPU(y);
  } else {
    x_d = ((VectorOptimizationDataTx*)x.optimizationData)->devicePtr;
    y_d = ((VectorOptimizationDataTx*)y.optimizationData)->devicePtr;
  }
  double local_result;
  launchComputeDotProduct(n, x_d, y_d, &local_result);
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

