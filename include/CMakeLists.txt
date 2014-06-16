set(TxHPCG_HEADERS
    BackendFactory.hpp
    CPCSR.hpp
    ComputeDotProduct.hpp
    ComputeMG.hpp
    ComputeProlongation.hpp
    ComputeRestriction.hpp
    ComputeSPMV.hpp
    ComputeSYMGS.hpp
    ComputeWAXPBY.hpp
    ExchangeHalo.hpp
    TxVectorOptimizationDataBase.hpp
    TxMatrixOptimizationDataBase.hpp
   )

install(FILES ${TxHPCG_HEADERS} DESTINATION include)
if(BUILD_CU_TXHPCG)
  add_subdirectory(CU)
endif()

