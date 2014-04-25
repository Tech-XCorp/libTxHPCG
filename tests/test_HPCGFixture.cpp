#define BOOST_TEST_MODULE HPCGFixture
#include <boost/test/unit_test.hpp>

#include "HPCGFixture.hpp"
#include "MPIFixture.hpp"
#include "Geometry.hpp"

#include <config.h>

BOOST_GLOBAL_FIXTURE(MPIFixture);

BOOST_AUTO_TEST_CASE(IncludeTest) {}

BOOST_AUTO_TEST_CASE(DefaultConstructor) {
  HPCG_Fixture fixture;
}

BOOST_AUTO_TEST_CASE(CustomSizes) {
  char** options = new char*[4];
  char optZero[] = "test_HPCGFixture";
  options[0] = optZero;
  char optOne[] = "--nx=64";
  options[1] = optOne; 
  char optTwo[] = "--ny=128";
  options[2] = optTwo;
  char optThree[] = "--nz=96";
  options[3] = optThree;
  HPCG_Fixture fixture(3, options);
  delete [] options;
  const Geometry* geometry = fixture.A.geom;
  BOOST_CHECK_EQUAL(64, geometry->nx);
  BOOST_CHECK_EQUAL(128, geometry->ny);
  BOOST_CHECK_EQUAL(96, geometry->nz);
}

