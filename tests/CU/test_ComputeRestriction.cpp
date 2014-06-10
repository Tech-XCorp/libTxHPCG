#include <boost/test/unit_test.hpp>
#include <boost/test/parameterized_test.hpp>
using namespace boost::unit_test;

#include <Vector.hpp>
#include <SparseMatrix.hpp>
#include <ComputeRestriction.hpp>
#include <CU/TxMatrixOptimizationDataCU.hpp>

#include <testUtils.hpp>
#include <testUtilsCU.hpp>

/**
 * @brief Reference implementation of restriction.
 *
 * Taken from hpcg source (v. 2.1)
 * */
int ComputeRestriction_ref(const SparseMatrix &A, const Vector &rf) {

  double *Axfv = A.mgData->Axf->values;
  double *rfv = rf.values;
  double *rcv = A.mgData->rc->values;
  local_int_t *f2c = A.mgData->f2cOperator;
  local_int_t nc = A.mgData->rc->localLength;

  for (local_int_t i = 0; i < nc; ++i)
    rcv[i] = rfv[f2c[i]] - Axfv[f2c[i]];

  return (0);
}

/**
 * @brief Compare restriction with reference implementation.
 * */
void testComputeRestriction(Dims d) {
  BOOST_REQUIRE_EQUAL(0, d.nx % 2);
  BOOST_REQUIRE_EQUAL(0, d.ny % 2);
  BOOST_REQUIRE_EQUAL(0, d.nz % 2);
  int n = d.nx * d.ny * d.nz;
  BOOST_REQUIRE_EQUAL(0, n % 8);
  int nCoarse = n / 8;
  SparseMatrix m = buildSparseMatrixCU(d.nx, d.ny, d.nz);
  Vector rf;
  InitializeVector(rf, n);
  FillRandomVector(rf);

  Vector result_ref;
  InitializeVector(result_ref, nCoarse);
  ComputeRestriction_ref(m, rf);
  CopyVector(*m.mgData->rc, result_ref);

  Vector result;
  InitializeVector(result, nCoarse);
  ComputeRestriction(m, rf);
  CopyVector(*m.mgData->rc, result);

  CHECK_RANGES_EQUAL(result_ref.values,
                     result_ref.values + result_ref.localLength, result.values);

  DeleteVector(result_ref);
  DeleteVector(result);
  DeleteMatrix(m);
}

/**
 * @brief Test driver.
 * */
test_suite *init_unit_test_suite(int argc, char *argv[]) {
  std::vector<Dims> geometries;
  geometries.push_back(Dims(2, 2, 2));
  geometries.push_back(Dims(2, 2, 4));
  geometries.push_back(Dims(4, 2, 4));
  geometries.push_back(Dims(4, 20, 4));
  geometries.push_back(Dims(88, 20, 2));
  geometries.push_back(Dims(140, 30, 40));
  framework::master_test_suite().add(BOOST_PARAM_TEST_CASE(
      &testComputeRestriction, geometries.begin(), geometries.end()));
  return 0;
}
