include_directories(${CMAKE_CURRENT_SOURCE_DIR})
set(CUTxHPCG_SOURCES
    CPCSR.cpp
    ComputeDotProduct.cpp
    MatrixOptimizationDataTx.cpp
    VectorOptimizationDataTx.cpp
    ExchangeHalo.cpp
    )
set(CUTxHPCG_SOURCES_CU
    Kernels.cu
   )
cuda_add_library(CUTxHPCG
    ${CUTxHPCG_SOURCES}
    ${CUTxHPCG_SOURCES_CU}
    )
install(TARGETS CUTxHPCG DESTINATION lib)

