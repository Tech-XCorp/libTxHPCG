#ifndef VECTOR_OPTIMIZATION_DATA_TX_HPP
#define VECTOR_OPTIMIZATION_DATA_TX_HPP

struct Vector_STRUCT;
typedef struct Vector_STRUCT Vector;

/**
 * @brief Class with data for Tx implementation of HPCG
 *
 * Instances of this class are stored in the optimizationData
 * pointer of the HPCG Vector class.
 *
 * @sa Vector, SparseMatrix, MatrixOptimizationDataTx
 * */
struct VectorOptimizationDataTx {
  double* devicePtr;
  void ZeroVector(int N);
};

/**
 * @brief Transfer data in Vector from host to GPU
 *
 * The data on the GPU is overwritten by the host data.
 *
 * @return Device pointer to data on GPU
 *
 * @sa transferDataFromGPU
 * */
double* transferDataToGPU(const Vector& v);

/**
 * @brief Transfer data from GPU to host.
 *
 * The data on the host side is overwritten by the GPU data.
 *
 * @sa transferDataToGPU
 * */
void transferDataFromGPU(const Vector& v);

#endif

