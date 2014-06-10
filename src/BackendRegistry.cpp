#include <BackendRegistry.hpp>

std::map<std::string, Backend>* BackendRegistry::registry = 0;

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
    return Backend(0, 0);
  }
}

void BackendRegistry::addBackend(const std::string& name, const Backend& backend) {
  BERegistry* reg = getRegistry();
  reg->insert(std::pair<std::string, Backend>(name, backend));
}


