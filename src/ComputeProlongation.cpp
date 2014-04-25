#include "ComputeProlongation.hpp"

#include "MatrixOptimizationDataTx.hpp"

int ComputeProlongation(const SparseMatrix& Af, Vector& xf, bool copyIn,
                        bool copyOut)
{
  MatrixOptimizationDataTx* optData =
      (MatrixOptimizationDataTx*)Af.optimizationData;
  return optData->ComputeProlongation(Af, xf, copyIn, copyOut);
}
