#include <boost/test/unit_test.hpp>
#include <boost/test/parameterized_test.hpp>
using namespace boost::unit_test;

#include <Vector.hpp>
#include <SparseMatrix.hpp>
#include <ComputeProlongation.hpp>
#include <CU/TxMatrixOptimizationDataCU.hpp>

#include <testUtils.hpp>
#include <testUtilsCU.hpp>

/**
 * @brief Reference implementation of restriction.
 *
 * Taken from hpcg source (v. 2.1)
 * */
int ComputeProlongation_ref(const SparseMatrix & Af, Vector & xf) {

  double * xfv = xf.values;
  double * xcv = Af.mgData->xc->values;
  local_int_t * f2c = Af.mgData->f2cOperator;
  local_int_t nc = Af.mgData->rc->localLength;

  for (local_int_t i=0; i<nc; ++i) xfv[f2c[i]] += xcv[i];
  return 0;
}

/**
 * @brief Compare restriction with reference implementation.
 * */

void testComputeProlongation(Dims d) {
  BOOST_REQUIRE_EQUAL(0, d.nx % 2);
  BOOST_REQUIRE_EQUAL(0, d.ny % 2);
  BOOST_REQUIRE_EQUAL(0, d.nz % 2);
  int n = d.nx * d.ny * d.nz;
  BOOST_REQUIRE_EQUAL(0, n % 8);
  SparseMatrix m = buildSparseMatrixCU(d.nx, d.ny, d.nz);
  Vector xf_ref;
  InitializeVector(xf_ref, n);
  FillRandomVector(xf_ref);
  Vector xf;
  InitializeVector(xf, n);
  CopyVector(xf_ref, xf);

  ComputeProlongation_ref(m, xf_ref);

  ComputeProlongation(m, xf);

  CHECK_RANGES_EQUAL(xf_ref.values, xf_ref.values + xf_ref.localLength,
                     xf.values);

  DeleteVector(xf_ref);
  DeleteVector(xf);
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
      &testComputeProlongation, geometries.begin(), geometries.end()));
  return 0;
}
