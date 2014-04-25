#include "HPCGFixture.hpp"

#include <SetupHalo.hpp>
#include <GenerateGeometry.hpp>
#include <GenerateProblem.hpp>
#include <GenerateCoarseProblem.hpp>
#include <CGData.hpp>
#include <OptimizeProblem.hpp>

#ifdef HAVE_MPI
#include <mpi.h>
#endif

HPCG_Fixture::HPCG_Fixture(int argc, char* argv[]) {
  HPCG_Init(&argc, &argv, params);
#ifndef HPCG_NOMPI
  MPI_Barrier(MPI_COMM_WORLD);
#endif

  local_int_t nx, ny, nz;
  nx = (local_int_t)params.nx;
  ny = (local_int_t)params.ny;
  nz = (local_int_t)params.nz;
  int size = params.comm_size;
  int rank = params.comm_rank;

  Geometry* geom = new Geometry;
  GenerateGeometry(size, rank, params.numThreads, nx, ny, nz, geom);

  InitializeSparseMatrix(A, geom);
  GenerateProblem(A, &b, &x, &xexact);
  SetupHalo(A);
  int numberOfMgLevels = 4; // Number of levels including first
  SparseMatrix* curLevelMatrix = &A;
  for (int level = 1; level< numberOfMgLevels; ++level) {
	  GenerateCoarseProblem(*curLevelMatrix);
	  curLevelMatrix = curLevelMatrix->Ac;
  }
  CGData data;
  InitializeSparseCGData(A, data);
  OptimizeProblem(A, data, b, x, xexact);
}

HPCG_Fixture::~HPCG_Fixture() {
}

char* HPCG_Fixture::defaultArgv[] = {"HPCGFixture"};
