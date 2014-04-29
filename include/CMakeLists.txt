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
    chkcudaerror.hpp
   )

install(FILES ${TxHPCG_HEADERS} DESTINATION include)

