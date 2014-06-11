#include <BackendFactory.hpp>

#include <Backend.hpp>
#include <BackendRegistry.hpp>

TxMatrixOptimizationDataBase* getMatrixOptimizationData(const char* backendName) {
  Backend be = BackendRegistry::getBackend(backendName);
  return be.getMatOptData();
}

TxVectorOptimizationDataBase* getVectorOptimizationData(const char* backendName) {
  Backend be = BackendRegistry::getBackend(backendName);
  return be.getVecOptData();
}

