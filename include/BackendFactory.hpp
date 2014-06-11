#ifndef BACKEND_FACTORY_HPP
#define BACKEND_FACTORY_HPP

#include <TxMatrixOptimizationDataBase.hpp>
#include <TxVectorOptimizationDataBase.hpp>
#include <config.h>

// Available Backends:
#define BACKEND_TX_CUDA "Tech-X CUDA Backend"

// The backend choosen at configure time
#if !defined(OPTIMIZED_BACKEND)
#error "OPTIMIZED_BACKEND not defined. libTxHPCG will not work without this!"
#endif
#define BACKEND_TO_USE OPTIMIZED_BACKEND

TxMatrixOptimizationDataBase* getMatrixOptimizationData(const char* backendName);
TxVectorOptimizationDataBase* getVectorOptimizationData(const char* backendName);

#endif

