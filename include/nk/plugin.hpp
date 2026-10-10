// include/nk/plugin.hpp
//
// nk::PluginContext — the plugin boundary protocol for NumKit modules.
//
// An L0 interface header: depends ONLY on the C++ standard library + nk_value.
// Shared between the host engine (numkit/numkit) and dynamic modules (numkit/*).

#pragma once

#include <nk/value/value.hpp>
#include <nk/value/span.hpp>
#include <nk/value/error.hpp>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory_resource>
#include <string>
#include <vector>

#if defined(_WIN32) || defined(__CYGWIN__)
  #if defined(NK_PLUGIN_BUILD)
    #define NK_PLUGIN_EXPORT __declspec(dllexport)
  #else
    #define NK_PLUGIN_EXPORT
  #endif
#else
  #if defined(__GNUC__) && __GNUC__ >= 4
    #define NK_PLUGIN_EXPORT __attribute__((visibility("default")))
  #else
    #define NK_PLUGIN_EXPORT
  #endif
#endif

namespace nk {

constexpr uint32_t NK_PLUGIN_ABI_VERSION = 1;

class Engine;
struct Environment;

struct CallContext
{
    Engine *engine{nullptr};
    Environment *env{nullptr};
};

// Signature of external functions registered into the engine
using ExternalFunc = std::function<void(Span<const Value> args,
                                        size_t nargout,
                                        Span<Value> outs,
                                        CallContext &ctx)>;

// Minimal engine abstraction required by adapters
class Engine
{
public:
    virtual ~Engine() = default;
    virtual std::pmr::memory_resource *resource() const noexcept = 0;
    virtual Value callFunctionHandle(const Value &handle, Span<const Value> args, void *env = nullptr) {
        (void)handle; (void)args; (void)env;
        return Value();
    }
    virtual std::vector<Value> callFunctionHandleMulti(const Value &handle, Span<const Value> args, size_t nargout) {
        (void)handle; (void)args; (void)nargout;
        return {};
    }
    virtual void registerFunction(const std::string &ns, const std::string &name, ExternalFunc func) = 0;
    virtual void registerFunction(const std::string &name, ExternalFunc func) = 0;
};

// Interface passed to nk_plugin_init by the host Engine.
// Inherits from Engine so that existing XLibrary::install(engine) calls work directly!
class PluginContext : public Engine {
public:
    virtual ~PluginContext() = default;
    virtual uint32_t abiVersion() const noexcept { return NK_PLUGIN_ABI_VERSION; }
};

} // namespace nk

// Entry point type of every dynamic module:
// extern "C" NK_PLUGIN_EXPORT bool nk_plugin_init(nk::PluginContext &ctx);
typedef bool (*NkPluginInitFn)(nk::PluginContext &ctx);
