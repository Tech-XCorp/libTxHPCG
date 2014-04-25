#include "MPIFixture.hpp"
#include <cstdlib>
#include <config.h>
#ifdef HAVE_MPI
#include <mpi.h>
#endif

MPIFixture::MPIFixture() {
  if (mpiInitialized) {
    return;
  }
#ifdef HAVE_MPI
  int argc = 1;
  char **argv = (char **)malloc(argc * sizeof(char *));
  argv[0] = (char *)malloc(strlen(filename) + 1);
  strcpy(argv[0], filename);
  MPI_Init(&argc, &argv);
  atexit(mpi_finalize);
#endif
  mpiInitialized = true;
}

MPIFixture::~MPIFixture() {}

void MPIFixture::mpi_finalize() { MPI_Finalize(); }
const char *MPIFixture::filename = "./test_HPCGFixture";
bool MPIFixture::mpiInitialized = false;

