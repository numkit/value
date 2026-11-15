// include/numkit/value/object.hpp
//
// Engine object model. A class instance is a ValueType::OBJECT heap value
// carrying a class name + shared ObjectState.
#pragma once

#include <numkit/value/value.hpp>

#include <keel/ref.h>

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace numkit {

struct CallContext;   // defined in sys/plugin.hpp or core/types.hpp
struct UserFunction;  // defined in core/types.hpp

// ============================================================
// NativePayload — opaque C++ instance state for builtin classes
// ============================================================
struct NativePayload
{
    virtual ~NativePayload() = default;
    virtual std::shared_ptr<NativePayload> clone() const = 0;
    virtual keel::f64 asDouble() const
    {
        throw std::runtime_error("Cannot read element as double from type 'object'");
    }
    virtual std::shared_ptr<NativePayload> fromScalarDouble(keel::f64 v) const
    {
        (void)v;
        return nullptr;
    }
};

// ============================================================
// ObjectState — instance state, shared via shared_ptr.
// ============================================================
struct ObjectState
{
    std::map<std::string, Value>   props;
    std::shared_ptr<NativePayload> native;
    std::string                    enumName;

    explicit ObjectState(keel::Allocator * /*alloc*/ = nullptr) {}

    std::shared_ptr<ObjectState> deepCopy(keel::Allocator *alloc = nullptr) const
    {
        auto s = std::make_shared<ObjectState>(alloc);
        s->props = props;
        s->native = native ? native->clone() : nullptr;
        s->enumName = enumName;
        return s;
    }
};

using ObjectMethod = std::function<void(Value &self, keel::ArrayRef<Value> args,
                                        keel::usize nargout, keel::MutArrayRef<Value> outs,
                                        CallContext &ctx)>;

struct PropMeta
{
    std::string name;
    std::string getAccess = "public";
    std::string setAccess = "public";
    bool        isConstant = false;
    bool        isDependent = false;
};
struct MethodMeta
{
    std::string name;
    bool        isStatic = false;
    std::string access = "public";
    bool        isAbstract = false;
};

// ============================================================
// BuiltinClass — one registry entry. Instances reference it by name.
// ============================================================
struct BuiltinClass
{
    std::string name;
    bool        isHandle = false;
    bool        isSealed = false;
    bool        isAbstract = false;
    std::vector<std::string> superclasses;

    std::function<Value(keel::ArrayRef<Value> args, CallContext &ctx)> construct;

    std::unordered_map<std::string, ObjectMethod> methods;

    std::vector<std::string> propNames;
    std::function<bool(const Value &self, const std::string &name,
                       Value &out, CallContext &ctx)> propGet;
    std::function<bool(Value &self, const std::string &name,
                       const Value &val, CallContext &ctx)> propSet;

    ObjectMethod subsref;
    ObjectMethod subsasgn;
    std::function<std::string(const Value &self)> dispText;
    std::unordered_map<std::string, ObjectMethod> ops;
    std::vector<std::string> hidden;
    std::vector<PropMeta>   propMeta;
    std::vector<MethodMeta> methodMeta;
    std::unordered_map<std::string, std::shared_ptr<const UserFunction>> methodFns;
};

} // namespace numkit
