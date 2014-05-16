#include <config.h>
#include <boost/test/unit_test.hpp>
#include <boost/test/parameterized_test.hpp>
#include <boost/test/output_test_stream.hpp>
using namespace boost::unit_test;
using boost::test_tools::output_test_stream;
#ifdef HAVE_MPI
#include <mpi.h>
#endif

#include <Vector.hpp>
#include <SparseMatrix.hpp>
#include <ComputeSYMGS.hpp>
#include <ExchangeHalo.hpp>

#include <MPIFixture.hpp>
#include <HPCGFixture.hpp>
#include <testUtils.hpp>


BOOST_GLOBAL_FIXTURE(MPIFixture);

/** 
 * @brief Reference implementation of SYMGS.
 *
 * Taken from HPCG v2.1
 * */
int ComputeSYMGS_ref( const SparseMatrix & A, const Vector & r, Vector & x);

/**
 * @brief Compare ComputeSYMGS with ComputeSYMGS_ref
 * */
void compareSYMGSWithRef(SparseMatrix& m);

/**
 * @brief Compare SYMGS with reference implementation.
 * */
void testComputeSYMGS(Dims d) {
  char** options = buildOptions(d.nx, d.ny, d.nz);
  HPCG_Fixture fixture(3, options);
  destroyOptions(options);

  BOOST_TEST_CHECKPOINT("Calling compareSYMGSWithRef in testComputeSYMGS");
  compareSYMGSWithRef(fixture.A);
}

/**
 * @brief Compare SYMGS with reference implementation for coarse level
 * matrix.
 * */
void testComputeSYMGSCoarse(Dims d) {
  char** options = buildOptions(d.nx, d.ny, d.nz);
  HPCG_Fixture fixture(3, options);
  destroyOptions(options);
  SparseMatrix& m = *fixture.A.Ac;

  BOOST_TEST_CHECKPOINT("Calling compareSYMGSWithRef in testComputeSYMGSCoarse");
  compareSYMGSWithRef(m);
}

void compareSYMGSWithRef(SparseMatrix& m) {
  Vector r;
  InitializeVector(r, m.localNumberOfRows);
  FillRandomVector(r);
  Vector x_ref;
  InitializeVector(x_ref, m.localNumberOfColumns);
  FillRandomVector(x_ref);
  Vector x;
  InitializeVector(x, m.localNumberOfColumns);
  CopyVector(x_ref, x);

  int err = ComputeSYMGS_ref(m, r, x_ref);
  BOOST_REQUIRE_EQUAL(0, err);

  err = ComputeSYMGS(m, r, x, 1);
  BOOST_REQUIRE_EQUAL(0, err);

  CHECK_RANGES_EQUAL(x_ref.values, x_ref.values + m.localNumberOfRows, x.values,
                     1.0e-8);

  DeleteVector(r);
  DeleteVector(x);
  DeleteVector(x_ref);
}

/**
 * @brief Test driver.
 * */
test_suite* init_unit_test_suite(int argc, char *argv[])
{
  std::vector<Dims> geometries;
  geometries.push_back(Dims(16, 16, 16));
  geometries.push_back(Dims(16, 16, 32));
  geometries.push_back(Dims(32, 16, 16));
  geometries.push_back(Dims(16, 32, 16));
  geometries.push_back(Dims(16, 48, 16));

  test_suite* fine_tests = BOOST_TEST_SUITE("Fine Matrix test suite");
  fine_tests->add(BOOST_PARAM_TEST_CASE(
      &testComputeSYMGS, geometries.begin(), geometries.end()));
  framework::master_test_suite().add(fine_tests);

  test_suite* coarse_tests = BOOST_TEST_SUITE("Coarse Matrix test suite");
  coarse_tests->add(BOOST_PARAM_TEST_CASE(
      &testComputeSYMGSCoarse, geometries.begin(), geometries.end()));
  framework::master_test_suite().add(coarse_tests);
  return 0;
}

int ComputeSYMGS_ref(const SparseMatrix &A, const Vector &r, Vector &x) {
  assert(x.localLength == A.localNumberOfColumns);
#ifndef HPCG_NOMPI
  ExchangeHalo(A, x);
#endif

  const local_int_t nrow = A.localNumberOfRows;
  double **matrixDiagonal = A.matrixDiagonal;
  const double *const rv = r.values;
  double *const xv = x.values;

  // Forward sweep
  for (local_int_t i = 0; i < nrow; i++) {
    const double *const currentValues = A.matrixValues[i];
    const local_int_t *const currentColIndices = A.mtxIndL[i];
    const int currentNumberOfNonzeros = A.nonzerosInRow[i];
    const double currentDiagonal = matrixDiagonal[i][0];
    double sum = rv[i];      
    for (int j = 0; j < currentNumberOfNonzeros; j++) {
      local_int_t curCol = currentColIndices[j];
      sum -= currentValues[j] * xv[curCol];
    }
    sum += xv[i] * currentDiagonal;
    xv[i] = sum / currentDiagonal;
  }

  // Now the back sweep.
  for (local_int_t i = nrow - 1; i >= 0; i--) {
    const double *const currentValues = A.matrixValues[i];
    const local_int_t *const currentColIndices = A.mtxIndL[i];
    const int currentNumberOfNonzeros = A.nonzerosInRow[i];
    const double currentDiagonal = matrixDiagonal[i][0];
    double sum = rv[i];       
    for (int j = 0; j < currentNumberOfNonzeros; j++) {
      local_int_t curCol = currentColIndices[j];
      sum -= currentValues[j] * xv[curCol];
    }
    sum += xv[i] * currentDiagonal;
    xv[i] = sum / currentDiagonal;
  }
  return (0);
}

