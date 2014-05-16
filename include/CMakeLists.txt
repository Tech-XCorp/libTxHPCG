set(TxHPCG_HEADERS
    CPCSR.hpp
    ComputeDotProduct.hpp
    ComputeMG.hpp
    ComputeProlongation.hpp
    ComputeRestriction.hpp
    ComputeSPMV.hpp
    ComputeSYMGS.hpp
    ComputeWAXPBY.hpp
    ExchangeHalo.hpp
    MatrixOptimizationDataTx.hpp
    VectorOptimizationDataTx.hpp
   )

install(FILES ${TxHPCG_HEADERS} DESTINATION include)
if(BUILD_CU_TXHPCG)
  add_subdirectory(CU)
endif()

