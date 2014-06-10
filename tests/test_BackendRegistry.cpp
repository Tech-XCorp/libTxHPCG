#define BOOST_TEST_MODULE BackendRegistry
#include <boost/test/unit_test.hpp>

#include <BackendRegistry.hpp>
#include <TxMatrixOptimizationDataBase.hpp>
#include <TxVectorOptimizationDataBase.hpp>

class BackendRegistryFixture {
  public:
    BackendRegistryFixture() {
      matbe = new int;
      vecbe = new int;
      BackendRegistry::addBackend("Foo", 
          Backend((TxMatrixOptimizationDataBase*)matbe,
            (TxVectorOptimizationDataBase*)vecbe));
    }
    ~BackendRegistryFixture() {
      delete matbe;
      delete vecbe;
    }
  private:
    int* matbe;
    int* vecbe;
};

BOOST_FIXTURE_TEST_SUITE(s, BackendRegistryFixture)

BOOST_AUTO_TEST_CASE(IncludeTest) {}

BOOST_AUTO_TEST_CASE(AddBackend) {
}

BOOST_AUTO_TEST_CASE(GetBackend) {
  Backend be = BackendRegistry::getBackend("Bar");
  BOOST_CHECK(be.getMatOptData() == 0);
  BOOST_CHECK(be.getVecOptData() == 0);
  be = BackendRegistry::getBackend("Foo");
  BOOST_CHECK(be.getMatOptData() != 0);
  BOOST_CHECK(be.getVecOptData() != 0);
}

BOOST_AUTO_TEST_SUITE_END()
