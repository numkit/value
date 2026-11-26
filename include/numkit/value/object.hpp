// include/numkit/value/object.hpp
//
// Engine object model. A class instance is a ValueType::OBJECT heap value
// carrying a class name + shared ObjectState.
#pragma once

#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

#include <keel/array.h>
#include <keel/function.h>
#include <keel/hash_map.h>
#include <keel/ref.h>
#include <keel/smart_ptr.h>
#include <keel/string.h>


namespace numkit {

struct CallContext;   // defined in sys/plugin.hpp or core/types.hpp
struct UserFunction;  // defined in core/types.hpp

// ============================================================
// NativePayload — opaque C++ instance state for builtin classes
// ============================================================
struct NativePayload
{
    virtual ~NativePayload() = default;
    virtual keel::Arc<NativePayload> clone(keel::Allocator *alloc = nullptr) const = 0;
    virtual keel::f64 asDouble() const
    {
        throw numkit::Error("Cannot read element as double from type 'object'");
    }
    virtual keel::Arc<NativePayload> fromScalarDouble(keel::f64 v,
                                                      keel::Allocator *alloc = nullptr) const
    {
        (void)v;
        (void)alloc;
        return nullptr;
    }
    virtual keel::StringRef concreteClassName() const noexcept
    {
        return {};
    }
};

// ============================================================
// ObjectState — instance state, shared via keel::Arc.
// ============================================================
struct ObjectState
{
    keel::HashMap<keel::String, Value> props;
    keel::Arc<NativePayload>           native;
    keel::String                       enumName;

    explicit ObjectState(keel::Allocator *alloc = nullptr) noexcept
        : props(alloc)
        , enumName(alloc)
    {}

    keel::Arc<ObjectState> deepCopy(keel::Allocator *alloc = nullptr) const
    {
        keel::Allocator *a = alloc ? alloc : props.allocator();
        auto s = keel::make_arc_in<ObjectState>(a, a);
        s->props = props.clone(a);
        s->native = native ? native->clone(a) : nullptr;
        s->enumName = enumName.clone(a);
        return s;
    }
};

using ObjectMethod = keel::SmallFunction<void(Value &self, keel::ArrayRef<Value> args,
                                              keel::usize nargout, keel::MutArrayRef<Value> outs,
                                              CallContext &ctx), 32>;

struct PropMeta
{
    keel::String name;
    keel::String getAccess = keel::String("public");
    keel::String setAccess = keel::String("public");
    bool         isConstant = false;
    bool         isDependent = false;
};

struct MethodMeta
{
    keel::String name;
    bool         isStatic = false;
    keel::String access = keel::String("public");
    bool         isAbstract = false;
};

// ============================================================
// BuiltinClass — one registry entry. Instances reference it by name.
// ============================================================
struct BuiltinClass
{
    keel::String name;
    bool         isHandle = false;
    bool         isSealed = false;
    bool         isAbstract = false;
    // Classes sharing a non-empty heterogeneousGroup may coexist in the same
    // object array and promote numeric placeholder arrays (e.g. zeros(1,N))
    // on indexed assignment.
    keel::String heterogeneousGroup;
    keel::Array<keel::String> superclasses;

    keel::SmallFunction<Value(keel::ArrayRef<Value> args, CallContext &ctx), 32> construct;

    keel::HashMap<keel::String, ObjectMethod> methods;

    keel::Array<keel::String> propNames;
    keel::SmallFunction<bool(const Value &self, keel::StringRef name,
                             Value &out, CallContext &ctx), 32> propGet;
    keel::SmallFunction<bool(Value &self, keel::StringRef name,
                             const Value &val, CallContext &ctx), 32> propSet;

    ObjectMethod subsref;
    ObjectMethod subsasgn;
    keel::SmallFunction<keel::String(const Value &self), 32> dispText;
    keel::HashMap<keel::String, ObjectMethod> ops;
    keel::Array<keel::String> hidden;
    keel::Array<PropMeta>     propMeta;
    keel::Array<MethodMeta>   methodMeta;
    keel::HashMap<keel::String, keel::Arc<const UserFunction>> methodFns;
};

} // namespace numkit
