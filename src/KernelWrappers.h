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

/**
 * @brief Scatter values into a contiguous array
 *
 * This can be used to prepare for a scatter during the halo exchange.
 * Duplicate entries in indices and aliasing of dst and src + indices lead
 * to undefined behavior.
 *
 * This kernel corresponds to the following sequential loop
 * for (int i = 0; i < N; ++i) {
 *   dst[i] = src[indices[i]];
 * }
 *
 * @param dst     Buffer into which the src elements are scattered.
 * @param src     Source array.
 * @param indices Indices of array elements in src to be copied to
 *                buffer.
 * @param N       Number of elements to scatter.
 * */
void launchScatter(double* dst, const double* src, const int* indices, int N);

/**
 * w <- alpha * x + beta * y
 * */
void launchComputeWAXPBY(local_int_t N, double alpha, const double* x,
    double beta, const double* y, double* w);

#endif

