#include <GenerateGeometry.hpp>
#include <CU/TxMatrixOptimizationDataCU.hpp>
#include <testUtilsCU.hpp>
#include <SetupHalo.hpp>
#include <config.h>
#ifdef HAVE_MPI
#include <mpi.h>
#endif
#include <cstring>

SparseMatrix buildSparseMatrixCU(int nXFine, int nYFine, int nZFine) {
  int nXCoarse = nXFine / 2;
  int nYCoarse = nYFine / 2;
  int nZCoarse = nZFine / 2;
  int nFine = nXFine * nYFine * nZFine;
  int nCoarse = nXCoarse * nYCoarse * nZCoarse;

  Geometry* geom = new Geometry;
  int size = 1;
  int rank = 0;
  int numThreads = 1;
  GenerateGeometry(size, rank, numThreads, nXFine, nYFine, nZFine, geom);

  SparseMatrix m;
  m.geom = geom;
  m.localNumberOfRows = nFine;
  m.localNumberOfColumns = nFine;
  m.totalNumberOfRows = nFine;
  m.localNumberOfNonzeros = 0;
  InitializeSparseMatrix(m, 0);
  SetupHalo(m);
  m.mgData = new MGData;
  m.mgData->f2cOperator = new int[nFine];
  for (int i = 0; i < nFine; ++i) {
    m.mgData->f2cOperator[i] = 0;
  }
  for (local_int_t izc = 0; izc < nZCoarse; ++izc) {
    local_int_t izf = 2 * izc;
    for (local_int_t iyc = 0; iyc < nYCoarse; ++iyc) {
      local_int_t iyf = 2 * iyc;
      for (local_int_t ixc = 0; ixc < nXCoarse; ++ixc) {
        local_int_t ixf = 2 * ixc;
        local_int_t currentCoarseRow =
            izc * nXCoarse * nYCoarse + iyc * nXCoarse + ixc;
        local_int_t currentFineRow = izf * nXFine * nYFine + iyf * nXFine + ixf;
        m.mgData->f2cOperator[currentCoarseRow] = currentFineRow;
      }
    }
  }
  m.mgData->rc = new Vector;
  InitializeVector(*m.mgData->rc, nCoarse);
  FillRandomVector(*m.mgData->rc);
  m.mgData->xc = new Vector;
  InitializeVector(*m.mgData->xc, nCoarse);
  FillRandomVector(*m.mgData->xc);
  m.mgData->Axf = new Vector;
  InitializeVector(*m.mgData->Axf, nFine);
  FillRandomVector(*m.mgData->Axf);

  TxMatrixOptimizationDataCU *optimizationData = new TxMatrixOptimizationDataCU;
  optimizationData->ingestLocalMatrix(m);
  m.optimizationData = optimizationData;

  return m;
}

