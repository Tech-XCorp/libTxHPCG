set(TxHPCG_HEADERS
    CPCSR.hpp
    ComputeMG.hpp
    ComputeProlongation.hpp
    ComputeRestriction.hpp
    ComputeSPMV.hpp
    ComputeSYMGS.hpp
    MatrixOptimizationDataTx.hpp
    VectorOptimizationDataTx.hpp
    chkcudaerror.hpp
   )

install(FILES ${TxHPCG_HEADERS} DESTINATION include)

