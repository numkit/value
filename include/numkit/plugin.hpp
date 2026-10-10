// include/numkit/plugin.hpp
//
// numkit::PluginContext — the plugin boundary protocol for NumKit modules.
#pragma once

#include <numkit/value/value.hpp>
#include <numkit/value/span.hpp>
#include <numkit/value/error.hpp>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory_resource>
#include <string>
#include <vector>

#ifndef NUMKIT_PLUGIN_ABI_VERSION
#  define NUMKIT_PLUGIN_ABI_VERSION 1
#endif

#if !defined(NUMKIT_PLUGIN_EXPORT)
  #if defined(_WIN32) || defined(__CYGWIN__)
    #if defined(NUMKIT_PLUGIN_BUILD)
      #define NUMKIT_PLUGIN_EXPORT __declspec(dllexport)
    #else
      #define NUMKIT_PLUGIN_EXPORT
    #endif
  #else
    #if defined(__GNUC__) && __GNUC__ >= 4
      #define NUMKIT_PLUGIN_EXPORT __attribute__((visibility("default")))
    #else
      #define NUMKIT_PLUGIN_EXPORT
    #endif
  #endif
#endif

namespace numkit {

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

// PluginContext: host interface passed to numkit_plugin_init.
// Inherited by HostPluginContext in the host engine.
class PluginContext {
public:
    virtual ~PluginContext() = default;
    virtual std::pmr::memory_resource *resource() const noexcept = 0;
    virtual void registerFunction(const std::string &ns, const std::string &name, ExternalFunc func) = 0;
    virtual void registerFunction(const std::string &name, ExternalFunc func) = 0;
    virtual Value callFunctionHandle(const Value &handle, Span<const Value> args, void *env = nullptr) {
        (void)handle; (void)args; (void)env;
        return Value();
    }
    virtual std::vector<Value> callFunctionHandleMulti(const Value &handle, Span<const Value> args, size_t nargout) {
        (void)handle; (void)args; (void)nargout;
        return {};
    }
    virtual uint32_t abiVersion() const noexcept { return NUMKIT_PLUGIN_ABI_VERSION; }
};

} // namespace numkit

// Entry point type of every dynamic module:
// extern "C" NUMKIT_PLUGIN_EXPORT bool numkit_plugin_init(numkit::PluginContext &ctx);
typedef bool (*NumkitPluginInitFn)(numkit::PluginContext &ctx);
