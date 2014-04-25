#ifndef HPCG_FIXTURE_HPP
#define HPCG_FIXTURE_HPP

#include <SparseMatrix.hpp>
#include <Geometry.hpp>
#include <Vector.hpp>
#include <hpcg.hpp>
#include <config.h>

/**
 * @brief Fixture for HPCG unit tests.
 *
 * This fixture mimicks the setup in testing/main.cpp.
 * It is assumed the MPI_Init and MPI_Finalize are being called outside
 * of this fixture.
 * */
class HPCG_Fixture {
  public:
    HPCG_Fixture(int argn = 1, char* argv[] = defaultArgv);
    ~HPCG_Fixture();

    SparseMatrix A;
    Vector b;
    Vector x;
    Vector xexact;
  private:
    HPCG_Params params;
    static char* defaultArgv[];
};

#endif

