#ifndef MPI_FIXTURE_HPP
#define MPI_FIXTURE_HPP

/**
 * @brief Fixture for MPI initialization and finalization.
 *
 * MPI_Init() is called when the class is created and a call
 * to MPI_Finalize() is registered with the c runtime using atexit().
 * In the boost unit_test_framework this class can be used with
 * BOOST_GLOBAL_FIXTURE.  This class is not thread safe.  When several
 * processes are run using mpirun then each process must create an
 * instance of MPIFixture.  
 * */
struct MPIFixture {
  MPIFixture();
  ~MPIFixture();
private:
  static void mpi_finalize();
  static const char *filename;
  static bool mpiInitialized;
};

#endif

