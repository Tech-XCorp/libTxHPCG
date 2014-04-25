#include "ComputeRestriction.hpp"

#include "MatrixOptimizationDataTx.hpp"

int ComputeRestriction(const SparseMatrix& Af, const Vector& rf, bool copyIn,
                       bool copyOut) {
  MatrixOptimizationDataTx* optData =
      (MatrixOptimizationDataTx*)Af.optimizationData;
  return optData->ComputeRestriction(Af, rf, copyIn, copyOut);
}

