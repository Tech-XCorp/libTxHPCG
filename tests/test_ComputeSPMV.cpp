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
#include <ComputeSPMV.hpp>
#include <ExchangeHalo.hpp>

#include <MPIFixture.hpp>
#include <HPCGFixture.hpp>
#include <testUtils.hpp>


BOOST_GLOBAL_FIXTURE(MPIFixture);

/**
 * @brief Reference implementation of SPMV.
 *
 * Taken from hpcg source (v. 2.1)
 * */
int ComputeSPMV_ref(const SparseMatrix &A, Vector &x, Vector &y);

/**
 * @brief Compare ComputeSPMV with ComputeSPMV_ref
 * */
void compareSPMVWithRef(SparseMatrix& m);

/**
 * @brief Compare SPMV with reference implementation.
 * */
void testComputeSPMV(Dims d) {
  char** options = buildOptions(d.nx, d.ny, d.nz);
  HPCG_Fixture fixture(3, options);
  destroyOptions(options);

  BOOST_TEST_CHECKPOINT("Calling compareSPMVWithRef in testComputeSPMV");
  compareSPMVWithRef(fixture.A);
}

/**
 * @brief Compare SPMV with reference implementation on coarser level.
 * */
void testComputeSPMVCoarse(Dims d) {
  char** options = buildOptions(d.nx, d.ny, d.nz);
  HPCG_Fixture fixture(3, options);
  destroyOptions(options);
  SparseMatrix& m = *fixture.A.Ac;

  BOOST_TEST_CHECKPOINT("Calling compareSPMVWithRef in testComputeSPMVCoarse");
  compareSPMVWithRef(m);
}

void compareSPMVWithRef(SparseMatrix& m) {
  Vector x;
  InitializeVector(x, m.localNumberOfColumns);
  FillRandomVector(x);
  Vector y_ref;
  InitializeVector(y_ref, m.localNumberOfRows);
  int err = ComputeSPMV_ref(m, x, y_ref);
  BOOST_REQUIRE_EQUAL(0, err);

  Vector y;
  InitializeVector(y, m.localNumberOfRows);
  err = ComputeSPMV(m, x, y);
  BOOST_REQUIRE_EQUAL(0, err);

  CHECK_RANGES_EQUAL(y_ref.values, y_ref.values + y_ref.localLength, y.values, 1.0e-8);

  DeleteVector(x);
  DeleteVector(y_ref);
}

/**
 * @brief Test driver.
 * */
test_suite* init_unit_test_suite(int argc, char *argv[])
{
  std::vector<Dims> geometries;
  geometries.push_back(Dims(16, 16, 16));
  geometries.push_back(Dims(16, 16, 32));
  geometries.push_back(Dims(16, 32, 16));
  geometries.push_back(Dims(32, 32, 16));
  geometries.push_back(Dims(64, 16, 128));

  test_suite* fine_tests = BOOST_TEST_SUITE("Fine Matrix test suite");
  fine_tests->add(BOOST_PARAM_TEST_CASE(
      &testComputeSPMV, geometries.begin(), geometries.end()));
  framework::master_test_suite().add(fine_tests);

  test_suite* coarse_tests = BOOST_TEST_SUITE("Coarse Matrix test suite");
  coarse_tests->add(BOOST_PARAM_TEST_CASE(
      &testComputeSPMVCoarse, geometries.begin(), geometries.end()));
  framework::master_test_suite().add(coarse_tests);
  return 0;
}

int ComputeSPMV_ref(const SparseMatrix &A, Vector &x, Vector &y)
{
  assert(x.localLength >= A.localNumberOfColumns);
  assert(y.localLength >= A.localNumberOfRows);
#ifndef HPCG_NOMPI
  ExchangeHalo(A, x);
#endif
  const double *const xv = x.values;
  double *const yv = y.values;
  const local_int_t nrow = A.localNumberOfRows;
  for (local_int_t i = 0; i < nrow; i++) {
    double sum = 0.0;
    const double *const cur_vals = A.matrixValues[i];
    const local_int_t *const cur_inds = A.mtxIndL[i];
    const int cur_nnz = A.nonzerosInRow[i];

    for (int j = 0; j < cur_nnz; j++)
      sum += cur_vals[j] * xv[cur_inds[j]];
    yv[i] = sum;
  }
  return (0);
}

