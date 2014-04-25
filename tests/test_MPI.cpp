#include <config.h>
#ifdef HAVE_MPI
#include <mpi.h>
#endif
#include <iostream>

int main(int argc, char** argv) {
#ifdef HAVE_MPI
  std::cout << "Initializing MPI." << std::endl;
  MPI_Init(&argc, &argv);
  MPI_Finalize();
#endif
  std::cout << "argc == " << argc << std::endl;
  for (int i = 0; i < argc; ++i ) {
    std::cout << argv[i] << std::endl;
  }
  return 0;
}


