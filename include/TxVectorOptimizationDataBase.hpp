#ifndef TX_VECTOR_OPTIMIZATION_DATA_BASE_HPP
#define TX_VECTOR_OPTIMIZATION_DATA_BASE_HPP

struct Vector_STRUCT;
typedef struct Vector_STRUCT Vector;

class TxVectorOptimizationDataBase {
  public:
    virtual void ZeroVector(int N) = 0;
    virtual void freeResources() = 0;
    virtual void transferDataToDevice(const double*) = 0;
    virtual void transferDataFromDevice(double*) = 0;
    virtual void copyDeviceData(void* dest, int numEntries) = 0;
    virtual int computeWAXPBY(int n, double alpha, const void* x, double beta, const void* y, void* w) const = 0;
    virtual void computeDotProduct(int n, const void* x, const void* y, double* result) const = 0;
    virtual void* getDevicePtr() = 0;
};

/**
 * @brief Release all resources needed by TxVectorOptimizationDataBase
 *
 * */
void freeResources(TxVectorOptimizationDataBase*);

/**
 * @brief Transfer data in Vector from host to device
 *
 * The data on the device is overwritten by the host data.
 *
 * @return Device pointer to data on device
 *
 * @sa transferDataFromDevice
 * */
double* transferDataToDevice(const Vector& v);

/**
 * @brief Transfer data from device to host.
 *
 * The data on the host side is overwritten by the device data.
 *
 * @sa transferDataToDevice
 * */
void transferDataFromDevice(const Vector& v);

#endif
