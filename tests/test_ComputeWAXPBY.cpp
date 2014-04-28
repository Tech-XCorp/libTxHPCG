#define BOOST_TEST_MODULE WAXPBY
#include <boost/test/unit_test.hpp>

#include <Vector.hpp>
#include <ComputeWAXPBY.hpp>
#include <ComputeWAXPBY_ref.hpp>
#include "testUtils.hpp"

BOOST_AUTO_TEST_CASE(test1) {
  static const int N = 187;
  static const double alpha = 137.0;
  static const double beta = 8.3;

  Vector x;
  Vector y;
  Vector w;
  Vector w_ref;
  InitializeVector(x, N);
  InitializeVector(y, N);
  InitializeVector(w, N);
  InitializeVector(w_ref, N);
  FillRandomVector(x);
  FillRandomVector(y);
  ZeroVector(w);
  ZeroVector(w_ref);
  
  bool isOptimized;
  int err = ComputeWAXPBY(N, alpha, x, beta, y, w, isOptimized);
  BOOST_REQUIRE(isOptimized);
  BOOST_REQUIRE_EQUAL(err, 0);

  err = ComputeWAXPBY_ref(N, alpha, x, beta, y, w_ref);
  BOOST_REQUIRE_EQUAL(err, 0);

  CHECK_RANGES_EQUAL(w_ref.values, w_ref.values + N, w.values);
}

