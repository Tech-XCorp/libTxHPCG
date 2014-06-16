#ifndef BACKEND_REGISTRY_HPP
#define BACKEND_REGISTRY_HPP

#include <map>
#include <string>
#include <vector>

#include <Backend.hpp>

class BackendRegistry {
  public:
    static BackendRegistry* getInstance();
    Backend getBackend(const std::string& name);
    void addBackend(const std::string& name, const Backend& backend);
    std::vector<std::string> getBackendNames();

  private:
    typedef std::map<std::string, Backend> BERegistry;
    static BERegistry* getRegistry();
    static BERegistry* registry;

    static BackendRegistry* instance;
    BackendRegistry();
    BackendRegistry(const BackendRegistry&);
    BackendRegistry& operator=(const BackendRegistry&);
};

#endif
