#include <TxVectorOptimizationDataBase.hpp>


void freeResources(TxVectorOptimizationDataBase* data) {
  data->freeResources();
}

void* transferDataToDevice(const Vector &v) {
  TxVectorOptimizationDataBase* opt = (TxVectorOptimizationDataBase*)v.optimizationData;
  opt->transferDataToDevice(v.values);
  return opt->getDevicePtr();
}

void transferDataFromDevice(const Vector &v) {
  TxVectorOptimizationDataBase* opt = (TxVectorOptimizationDataBase*)v.optimizationData;
  opt->transferDataFromDevice(v.values);
}


