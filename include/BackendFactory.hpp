#ifndef BACKEND_FACTORY_HPP
#define BACKEND_FACTORY_HPP

class TxMatrixOptimizationDataBase;
class TxVectorOptimizationDataBase;
#include <config.h>

// The backend choosen at configure time
#if !defined(OPTIMIZED_BACKEND_NAME)
#error "OPTIMIZED_BACKEND_NAME not defined. libTxHPCG will not work without this!"
#endif
#define BACKEND_TO_USE OPTIMIZED_BACKEND_NAME

TxMatrixOptimizationDataBase* getMatrixOptimizationData(const char* backendName);
TxVectorOptimizationDataBase* getVectorOptimizationData(const char* backendName);

#endif

