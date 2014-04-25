#ifndef SPMV_KERNEL_WRAPPER
#define SPMV_KERNEL_WRAPPER

#include <Geometry.hpp>

void launchSpmvKernel(local_int_t numNonZeroRows,
                      const local_int_t* nonZeroRows,
                      const local_int_t* offsets,
                      const local_int_t* columnIndices, const double* values,
                      const double* x, double* y, double alpha, double beta);

void launchProlongationKernel(double* xf, const double* xc, int nc,
                              const int* f2c);

void launchRestrictionKernel(double* rc, const double* rf, const double* Axf,
                             int nc, const int* f2c);

void launchZeroVector(double* v, int n);

/**
 * @brief Copy arrays of doubles
 *
 * src and dst must not overlap.
 * */
void launchDeviceCopy(double* dst, const double* src, int N);

#endif

