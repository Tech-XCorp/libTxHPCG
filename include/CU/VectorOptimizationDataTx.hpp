#ifndef VECTOR_OPTIMIZATION_DATA_TX_HPP
#define VECTOR_OPTIMIZATION_DATA_TX_HPP

#include <TxVectorOptimizationDataBase.hpp>

/**
 * @brief Class with data for Tx implementation of HPCG
 *
 * Instances of this class are stored in the optimizationData
 * pointer of the HPCG Vector class.
 *
 * @sa Vector, SparseMatrix, MatrixOptimizationDataTx
 * */
class VectorOptimizationDataTx : public TxVectorOptimizationDataBase{
  public:
    void ZeroVector(int N);
    virtual void freeResources();
    virtual void transferDataToDevice(const Vector&);
    virtual void transferDataFromDevice(Vector&);
    virtual void copyDeviceData(void* dest, int numEntries) = 0;
    virtual int computeWAXPBY(int n, double alpha, const void* x, double beta, const void* y, void* w) const = 0;
    virtual void computeDotProduct(int n, const void* x, const void* y, double* result) const = 0;
    virtual void* getDevicePtr();
  private:
    double* devicePtr;
};

#endif

