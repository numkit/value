// include/numkit/value/fn_handle.hpp
//
// numkit::FnHandle — canonical callback type used by the numerical-library API
// for any function that takes a user-supplied function handle (fzero, integral,
// fminsearch, cellfun, structfun, …).

#pragma once

#include <keel/function_ref.h>
#include <keel/memory.h>
#include <keel/ref.h>

#include <numkit/value/value.hpp>

namespace numkit {

/// @brief Function-handle callback.
///
/// `args` carries the per-call inputs; `outs` is the caller-allocated
/// output slice (its `.size()` is the requested `nargout`). `alloc` is
/// the caller's allocator — callbacks should pass it when constructing
/// output Values.
using FnHandle = keel::FunctionRef<void(keel::ArrayRef<Value>    args,
                                        keel::MutArrayRef<Value> outs,
                                        keel::Allocator *        alloc)>;

} // namespace numkit
