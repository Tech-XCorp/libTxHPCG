#ifndef TX_VECTOR_OPTIMIZATION_DATA_CU_HPP
#define TX_VECTOR_OPTIMIZATION_DATA_CU_HPP

#include <TxVectorOptimizationDataBase.hpp>

/**
 * @brief Class with data for Tx implementation of HPCG
 *
 * Instances of this class are stored in the optimizationData
 * pointer of the HPCG Vector class.
 *
 * @sa Vector, SparseMatrix, TxMatrixOptimizationDataBase
 * */
class TxVectorOptimizationDataCU : public TxVectorOptimizationDataBase {
  public:
    void ZeroVector(int N);
    virtual void freeResources();
    virtual void transferDataToDevice(const Vector&);
    virtual void transferDataFromDevice(Vector&);
    virtual void copyDeviceData(void* dest, int numEntries);
    virtual int computeWAXPBY(int n, double alpha, const void* x, double beta, const void* y, void* w) const;
    virtual void computeDotProduct(int n, const void* x, const void* y, double* result) const;
    virtual void* getDevicePtr();
    virtual TxVectorOptimizationDataCU* create();
  private:
    double* devicePtr;
};

#endif

