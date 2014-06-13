#include <BackendFactory.hpp>

#include <Backend.hpp>
#include <BackendRegistry.hpp>

#include <TxMatrixOptimizationDataBase.hpp>
#include <TxVectorOptimizationDataBase.hpp>

TxMatrixOptimizationDataBase* getMatrixOptimizationData(const char* backendName) {
  Backend be = BackendRegistry::getInstance()->getBackend(backendName);
  return be.getMatOptData()->create();
}

TxVectorOptimizationDataBase* getVectorOptimizationData(const char* backendName) {
  Backend be = BackendRegistry::getInstance()->getBackend(backendName);
  return be.getVecOptData()->create();
}

