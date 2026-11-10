// include/numkit/value/c_api_bridge.hpp
//
// C++ bridge between `numkit::Value` and `numkit_val*` handles.
// Allocation (`boxValue`) and deallocation (`numkit_val_release`) always execute
// inside `numkit_value_c` so CRT heaps and handle counters stay unified across
// all shared libraries on Windows, Linux, and macOS.
#pragma once

#include <numkit/value/c_api.h>
#include <numkit/value/span.hpp>
#include <numkit/value/value.hpp>

#include <cstring>
#include <exception>
#include <stdexcept>
#include <vector>

namespace numkit::c_abi {

NUMKIT_VALUE_API numkit_val *boxValue(Value v);
NUMKIT_VALUE_API const Value &unboxValue(const numkit_val *v);
NUMKIT_VALUE_API Value &unboxValueMut(numkit_val *v);

inline void clearError(numkit_error *err) noexcept
{
    if (err) {
        err->code = 0;
        err->message[0] = '\0';
    }
}

inline void setError(numkit_error *err, int code, const char *msg) noexcept
{
    if (!err) return;
    err->code = code == 0 ? 1 : code;
    if (!msg) {
        err->message[0] = '\0';
        return;
    }
    std::strncpy(err->message, msg, sizeof(err->message) - 1);
    err->message[sizeof(err->message) - 1] = '\0';
}

inline void invokeHostCallback(numkit_host_fn fn,
                               void *userData,
                               Span<const Value> args,
                               size_t nargout,
                               Span<Value> outs)
{
    std::vector<numkit_val *> boxedArgs;
    boxedArgs.reserve(args.size());
    for (size_t i = 0; i < args.size(); ++i) {
        boxedArgs.push_back(boxValue(args[i]));
    }

    size_t actualOut = outs.empty() ? 1 : outs.size();
    std::vector<numkit_val *> rawOuts(actualOut, nullptr);
    numkit_error cbErr{};
    clearError(&cbErr);

    int rc = fn(userData,
                boxedArgs.empty() ? nullptr : boxedArgs.data(),
                boxedArgs.size(),
                nargout,
                rawOuts.data(),
                &cbErr);

    for (numkit_val *p : boxedArgs) {
        numkit_val_release(p);
    }

    if (rc != 0) {
        for (numkit_val *p : rawOuts) {
            numkit_val_release(p);
        }
        throw std::runtime_error(cbErr.message[0] ? cbErr.message : "Host callback failed");
    }

    for (size_t i = 0; i < rawOuts.size(); ++i) {
        if (rawOuts[i]) {
            if (i < outs.size()) {
                outs[i] = std::move(unboxValueMut(rawOuts[i]));
            }
            numkit_val_release(rawOuts[i]);
        }
    }
}

} // namespace numkit::c_abi
