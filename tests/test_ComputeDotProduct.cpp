#include <boost/test/unit_test.hpp>
#include <boost/test/parameterized_test.hpp>
using namespace boost::unit_test;

#include <Vector.hpp>
#include <ComputeDotProduct.hpp>
#include <ComputeDotProduct_ref.hpp>
#include <MPIFixture.hpp>

BOOST_GLOBAL_FIXTURE(MPIFixture);


void testComputeDotProduct(int N) {
  Vector x;
  Vector y;
  InitializeVector(x, N);
  InitializeVector(y, N);
  FillRandomVector(x);
  FillRandomVector(y);

  bool isOptimized;
  double result;
  double t;
  int err = ComputeDotProduct(N, x, y, result, t, isOptimized);
  BOOST_REQUIRE(isOptimized);
  BOOST_REQUIRE_EQUAL(err, 0);

  double result_ref;
  double t_ref;
  err = ComputeDotProduct_ref(N, x, y, result_ref, t_ref);
  BOOST_REQUIRE_EQUAL(err, 0);

  BOOST_CHECK_CLOSE(result, result_ref, 1.0e-7);
}

test_suite* init_unit_test_suite(int argc, char *argv[]) {
  std::vector<int> sizes;
  sizes.push_back(187);
  sizes.push_back(512);
  sizes.push_back(513);
  sizes.push_back(1023);
  sizes.push_back(1000000);
  test_suite* tests = BOOST_TEST_SUITE("DotProduct test suite");
  tests->add(BOOST_PARAM_TEST_CASE(
        &testComputeDotProduct, sizes.begin(), sizes.end()));
  framework::master_test_suite().add(tests);
  return 0;
}


