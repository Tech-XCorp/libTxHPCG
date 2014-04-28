#include "KernelWrappers.h"

size_t getNumBlocks(size_t numThreads, size_t blockSize) {
  size_t numBlocks = (numThreads + blockSize - 1) / blockSize;
  return numBlocks;
}

__global__ void spmvKernel(local_int_t numNonZeroRows,
                           const local_int_t* __restrict__ nonZeroRows,
                           const local_int_t* __restrict__ offsets,
                           const local_int_t* __restrict__ columnIndices,
                           const double* __restrict__ values,
                           const double* __restrict__ x,
                           double* __restrict__ y,
                           double alpha, double beta) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < numNonZeroRows) {
    int r = nonZeroRows[i];
    double result = 0;
    for (int j = offsets[i]; j < offsets[i + 1]; ++j) {
      result += values[j] * x[columnIndices[j]];
    }
    y[r] = alpha * result + beta * y[r];
  }
}

__global__ void prolongationKernel(double* __restrict__ xf,
                                   const double* __restrict__ xc, int nc,
                                   const int* __restrict__ f2c) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < nc) {
    xf[f2c[i]] += xc[i];
  }
}

__global__ void restrictionKernel(double* __restrict__ rc,
                                  const double* __restrict__ rf,
                                  const double* __restrict__ Axf, int nc,
                                  const int* __restrict__ f2c) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < nc) {
    rc[i] = rf[f2c[i]] - Axf[f2c[i]];
  }
}

__global__ void copyKernel(double* dst, const double* src, int N) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int numThreads = gridDim.x * blockDim.x;
  while (i < N) {
    dst[i] = src[i];
    i += numThreads;
  }
}

__global__ void zeroVectorKernel(double* __restrict__ v, int N) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int numThreads = gridDim.x * blockDim.x;
  while (i < N) {
    v[i] = 0;
    i += numThreads;
  }
}

__global__ void scatterKernel(double* __restrict__ dst,
    const double* __restrict__ src, const int* __restrict indices,
    int N) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int numThreads = gridDim.x * blockDim.x;
  while (i < N) {
    dst[i] = src[indices[i]];
    i += numThreads;
  }
}

void launchSpmvKernel(local_int_t numNonZeroRows,
                      const local_int_t *nonZeroRows,
                      const local_int_t *offsets,
                      const local_int_t *columnIndices, const double *values,
                      const double *x, double *y, double alpha,
                      double beta) {
  if (numNonZeroRows > 0) {
  static const size_t blockSize = 128;
  size_t numBlocks = getNumBlocks(numNonZeroRows, blockSize);
  spmvKernel<<<numBlocks, blockSize>>>(numNonZeroRows, nonZeroRows,
      offsets, columnIndices, values, x, y, alpha, beta);
  }
}

void launchProlongationKernel(double* xf, const double* xc, int nc,
                              const int* f2c)
{
  if (nc > 0) {
    static const size_t blockSize = 128;
    size_t numBlocks = getNumBlocks(nc, blockSize);
    prolongationKernel<<<numBlocks, blockSize>>> (xf, xc, nc, f2c);
  }
}

void launchRestrictionKernel(double* rc, const double* rf, const double*
    Axf, int nc, const int* f2c)
{
  if (nc > 0) {
    static const size_t blockSize = 128;
    size_t numBlocks = getNumBlocks(nc, blockSize);
    restrictionKernel<<<numBlocks, blockSize>>> (rc, rf, Axf, nc, f2c);
  }
}

void launchDeviceCopy(double* dst, const double* src, int N)
{
  if (N > 0) {
    static const size_t blockSize = 128;
    static const size_t MAX_NUM_BLOCKS = 128;
    size_t numBlocks = getNumBlocks(N, blockSize);
    if (numBlocks > MAX_NUM_BLOCKS) {
      numBlocks = MAX_NUM_BLOCKS;
    }
    copyKernel<<<numBlocks, blockSize>>> (dst, src, N);
  }
}

void launchZeroVector(double* v, int N) {
  if (N > 0) {
    static const size_t blockSize = 128;
    static const size_t MAX_NUM_BLOCKS = 128;
    size_t numBlocks = getNumBlocks(N, blockSize);
    if (numBlocks > MAX_NUM_BLOCKS) {
      numBlocks = MAX_NUM_BLOCKS;
    }
    zeroVectorKernel<<<numBlocks, blockSize>>> (v, N);
  }
}

void launchScatter(double* dst, const double* src, const int* indices, int N) {
  if (N > 0) {
    static const size_t blockSize = 128;
    static const size_t MAX_NUM_BLOCKS = 128;
    size_t numBlocks = getNumBlocks(N, blockSize);
    if (numBlocks > MAX_NUM_BLOCKS) {
      numBlocks = MAX_NUM_BLOCKS;
    }
    scatterKernel<<<numBlocks, blockSize>>>(dst, src, indices, N);
  }
}
