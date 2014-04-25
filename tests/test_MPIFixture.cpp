#define BOOST_TEST_MODULE MPIFixture
#include <boost/test/unit_test.hpp>

#include "MPIFixture.hpp"

BOOST_GLOBAL_FIXTURE(MPIFixture);


BOOST_AUTO_TEST_CASE(IncludeTest) {}

BOOST_AUTO_TEST_CASE(SecondFixture) {
  // Should be possible to create a second MPIFixture
  MPIFixture fixture;
}

