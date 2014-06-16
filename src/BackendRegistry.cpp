#include <BackendRegistry.hpp>

#include <config.h>

#ifdef HAVE_CU_BACKEND
#include <CU/TxMatrixOptimizationDataCU.hpp>
#include <CU/TxVectorOptimizationDataCU.hpp>
#endif

#include <iostream>

std::map<std::string, Backend>* BackendRegistry::registry = 0;
BackendRegistry* BackendRegistry::instance = 0;


BackendRegistry::BackendRegistry() {

#ifdef HAVE_CU_BACKEND
  addBackend("Tech-X CUDA backend",
      Backend(new TxMatrixOptimizationDataCU, new TxVectorOptimizationDataCU));
#endif
}

BackendRegistry* BackendRegistry::getInstance() {
  if (!instance) {
    instance = new BackendRegistry;
  }
  return instance;
}

std::map<std::string, Backend>* BackendRegistry::getRegistry() {
  if (!registry) {
    registry = new BERegistry;
  }
  return registry;
}

Backend BackendRegistry::getBackend(const std::string& name) {
  BERegistry* reg = getRegistry();
  BERegistry::iterator b = reg->find(name);
  if (b != reg->end()) {
    return b->second;
  } else {
    std::cerr << "Didn't find backend \"" << name << "\" in registry!\n";
    std::cerr << "Available backends: \n";
    std::vector<std::string> names = getBackendNames();
    for (std::vector<std::string>::const_iterator i = names.begin();
        i != names.end(); ++i) {
      std::cerr << *i << "\n";
    }
    std::cerr << std::endl;
    return Backend(0, 0);
  }
}

std::vector<std::string> BackendRegistry::getBackendNames()
{
  std::vector<std::string> names;
  for (BERegistry::const_iterator i = getRegistry()->begin();
      i != getRegistry()->end(); ++i) {
    names.push_back(i->first);
  }
  return names;
}

void BackendRegistry::addBackend(const std::string& name, const Backend& backend) {
  BERegistry* reg = getRegistry();
  reg->insert(std::pair<std::string, Backend>(name, backend));
}


