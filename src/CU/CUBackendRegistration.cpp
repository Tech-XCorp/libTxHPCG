#include <BackendRegistry.hpp>
#include <Backend.hpp>
#include <CU/TxMatrixOptimizationDataCU.hpp>
#include <CU/TxVectorOptimizationDataCU.hpp>


class CUBackendRegistration {
  public:
    CUBackendRegistration() : 
      be(new TxMatrixOptimizationDataCU, new TxVectorOptimizationDataCU) {
        BackendRegistry::addBackend("Tech-X CUDA backend", be);
      }
    ~CUBackendRegistration() {
      delete be.getMatOptData();
      delete be.getVecOptData();
    }

  private:
    Backend be;
};

static CUBackendRegistration registerCUBackend;
