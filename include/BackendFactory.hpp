#ifndef BACKEND_FACTORY_HPP
#define BACKEND_FACTORY_HPP

class TxMatrixOptimizationDataBase;
class TxVectorOptimizationDataBase;

TxMatrixOptimizationDataBase* getMatrixOptimizationData(const char* backendName);
TxVectorOptimizationDataBase* getVectorOptimizationData(const char* backendName);

#endif

