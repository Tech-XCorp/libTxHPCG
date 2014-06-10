#ifndef BACKEND_REGISTRY_HPP
#define BACKEND_REGISTRY_HPP

#include <map>
#include <string>

#include <Backend.hpp>

class BackendRegistry {
  public:
    static Backend getBackend(const std::string& name);
    static void addBackend(const std::string& name, const Backend& backend);

  private:
    typedef std::map<std::string, Backend> BERegistry;
    static BERegistry* getRegistry();
    static BERegistry* registry;
    BackendRegistry();
    BackendRegistry(const BackendRegistry&);
    BackendRegistry& operator=(const BackendRegistry&);
};


#endif
