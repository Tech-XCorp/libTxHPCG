#include <TxVectorOptimizationDataBase.hpp>

#include <Vector.hpp>


void freeResources(TxVectorOptimizationDataBase* data) {
  data->freeResources();
}

void* transferDataToDevice(const Vector &v) {
  TxVectorOptimizationDataBase* opt = (TxVectorOptimizationDataBase*)v.optimizationData;
  opt->transferDataToDevice(v);
  return opt->getDevicePtr();
}

void transferDataFromDevice(Vector &v) {
  TxVectorOptimizationDataBase* opt = (TxVectorOptimizationDataBase*)v.optimizationData;
  opt->transferDataFromDevice(v);
}


