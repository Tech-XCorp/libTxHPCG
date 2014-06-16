#include "ComputeProlongation.hpp"

#include "TxMatrixOptimizationDataBase.hpp"

int ComputeProlongation(const SparseMatrix& Af, Vector& xf, bool copyIn,
                        bool copyOut)
{
  TxMatrixOptimizationDataBase* optData =
    (TxMatrixOptimizationDataBase*)Af.optimizationData;
  return optData->ComputeProlongation(Af, xf, copyIn, copyOut);
}
