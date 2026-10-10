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
#include <memory_resource>

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

// The signature of every registered function adapter in a dynamic module.
using PluginFunc = void (*)(Span<const Value> args,
                            size_t nargout,
                            Span<Value> outs,
                            std::pmr::memory_resource *mr);

// Interface passed to nk_plugin_init by the host Engine.
class PluginContext {
public:
    virtual ~PluginContext() = default;

    virtual uint32_t abiVersion() const noexcept { return NK_PLUGIN_ABI_VERSION; }

    // Register a function into the host engine.
    // `ns`: namespace prefix (e.g. "signal", "image"), or "" for bare/global.
    // `name`: function name (e.g. "filter", "fir1").
    // `func`: pointer to the adapter function.
    virtual void registerFunction(const char *ns,
                                  const char *name,
                                  PluginFunc func) = 0;

    // Default memory resource of the active engine session.
    virtual std::pmr::memory_resource *resource() noexcept = 0;
};

} // namespace nk

// Entry point type of every dynamic module:
// extern "C" NK_PLUGIN_EXPORT bool nk_plugin_init(nk::PluginContext &ctx);
typedef bool (*NkPluginInitFn)(nk::PluginContext &ctx);
