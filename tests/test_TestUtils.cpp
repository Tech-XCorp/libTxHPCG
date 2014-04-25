#define BOOST_TEST_MODULE TestUtils
#include <boost/test/unit_test.hpp>
#include <boost/test/output_test_stream.hpp>
using boost::test_tools::output_test_stream;

#include <testUtils.hpp>

BOOST_AUTO_TEST_CASE(BuildOptions)
{
  char** options = buildOptions(17, 32, 60);
  BOOST_CHECK(options);
  BOOST_CHECK(options[0]);
  BOOST_CHECK(options[1]);
  BOOST_CHECK(options[2]);
  BOOST_CHECK(options[3]);
  output_test_stream output;
  for (int i = 0; i < 4; ++i) {
    output << options[i] << "\n";
  }
  BOOST_CHECK(!output.is_empty(false));
  static const char expectedOutput[] = "test_ComputeSPMV\n"
                                       "--nx=17\n"
                                       "--ny=32\n"
                                       "--nz=60\n";
  BOOST_CHECK(output.check_length(strlen(expectedOutput), false));
  BOOST_CHECK(output.is_equal(expectedOutput));
  destroyOptions(options);
}
