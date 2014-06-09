#include "ComputeRestriction.hpp"

#include "TxMatrixOptimizationDataBase.hpp"

int ComputeRestriction(const SparseMatrix& Af, const Vector& rf, bool copyIn,
                       bool copyOut) {
  TxMatrixOptimizationDataBase* optData =
      (TxMatrixOptimizationDataBase*)Af.optimizationData;
  return optData->ComputeRestriction(Af, rf, copyIn, copyOut);
}

