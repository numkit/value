// include/numkit/value/argcheck.hpp
//
// numkit::argcheck — engine-free argument-validation primitives with
// numkit-voiced diagnostics.

#pragma once

#include <cassert>
#include <cmath>
#include <sstream>
#include <string>
#include <string_view>

#include <keel/math.h>
#include <keel/ref.h>

#include <numkit/value/error.hpp>
#include <numkit/value/index.hpp>
#include <numkit/value/value.hpp>
#include <numkit/value/value_type.hpp>

namespace numkit::argcheck {

namespace detail {

[[noreturn]] inline void fail(const std::string &msg, std::string_view fn,
                              const std::string &id)
{
    throw Error(msg, 0, 0, std::string(fn), "", id);
}

// Value rendering for got-clauses: "2.5", "-5", "1e+30", "Inf", "NaN".
inline std::string num(keel::f64 v)
{
    if (keel::is_nan(v))
        return "NaN";
    if (keel::is_inf(v))
        return v > 0 ? "Inf" : "-Inf";
    std::ostringstream os;
    os << v;
    return os.str();
}

// Shape rendering for got-clauses: "1x2", "2x3x4", "0x0".
inline std::string dimsStr(const Value &v)
{
    const Dims &d = v.dims();
    std::string s = std::to_string(d.dim(0));
    for (keel::i32 k = 1; k < d.ndim(); ++k) {
        s += "x";
        s += std::to_string(d.dim(k));
    }
    return s;
}

} // namespace detail

// ============================================================
// Arity.
// ============================================================
inline void checkNargin(keel::usize actual, keel::usize minArgs, std::string_view fn)
{
    if (actual < minArgs)
        detail::fail(std::string(fn) + ": not enough input arguments — at "
                     "least " + std::to_string(minArgs) + " required (got "
                         + std::to_string(actual) + ").",
                     fn, "numkit:" + std::string(fn) + ":nargin");
}

inline void checkNargin(keel::usize actual, keel::usize minArgs, keel::usize maxArgs,
                        std::string_view fn)
{
    checkNargin(actual, minArgs, fn);
    if (actual > maxArgs)
        detail::fail(std::string(fn) + ": too many input arguments — at "
                     "most " + std::to_string(maxArgs) + " accepted (got "
                         + std::to_string(actual) + ").",
                     fn, "numkit:" + std::string(fn) + ":maxnargin");
}

// ============================================================
// Array-creation size inputs (zeros/ones/eye/true/false/inf/nan/
// rand/randn/randi/strings).
// ============================================================

inline keel::usize arrayDim(keel::f64 d, std::string_view fn)
{
    if (keel::is_nan(d) || keel::is_inf(d))
        detail::fail(std::string(fn) + ": size inputs must be finite (got "
                     + detail::num(d) + ").",
                     fn, "numkit:" + std::string(fn) + ":sizeNotFinite");
    if (d != std::floor(d))
        detail::fail(std::string(fn) + ": size inputs must be integers "
                     "(got " + detail::num(d) + ").",
                     fn, "numkit:" + std::string(fn) + ":sizeNotInteger");
    if (d < 0.0)
        return 0;
    keel::usize out = 0;
    if (!tryNonNegSize(d, out))
        detail::fail(std::string(fn) + ": requested array size exceeds the "
                     "supported maximum (got " + detail::num(d) + ").",
                     fn, "numkit:" + std::string(fn) + ":sizeTooLarge");
    return out;
}

inline void arrayDimReal(const Value &v, std::string_view fn)
{
    if (v.isComplex())
        detail::fail(std::string(fn) + ": size inputs must be real "
                     "(argument is complex).",
                     fn, "numkit:" + std::string(fn) + ":sizeNotReal");
}

inline keel::usize arrayDim(const Value &v, std::string_view fn)
{
    if (!v.isScalar())
        detail::fail(std::string(fn) + ": size inputs must be scalars "
                     "(argument is " + detail::dimsStr(v) + ").",
                     fn, "numkit:" + std::string(fn) + ":sizeNotScalar");
    arrayDimReal(v, fn);
    return arrayDim(v.toScalar(), fn);
}

// ============================================================
// reshape size inputs.
// ============================================================

inline void reshapeSizeReal(const Value &v, std::string_view fn)
{
    if (v.isComplex())
        detail::fail(std::string(fn) + ": size inputs must be real "
                     "(argument is complex).",
                     fn, "numkit:" + std::string(fn) + ":sizeNotReal");
}

inline keel::f64 reshapeSizeArg(const Value &v, std::string_view fn)
{
    if (!v.isScalar())
        detail::fail(std::string(fn) + ": size inputs must be scalars "
                     "(argument is " + detail::dimsStr(v) + ").",
                     fn, "numkit:" + std::string(fn) + ":sizeNotScalar");
    reshapeSizeReal(v, fn);
    return v.toScalar();
}

inline keel::usize reshapeSize(keel::f64 d, std::string_view fn)
{
    if (d < 0.0)
        detail::fail(std::string(fn) + ": size inputs must be non-negative "
                     "(got " + detail::num(d) + ").",
                     fn, "numkit:" + std::string(fn) + ":sizeNegative");
    if (keel::is_nan(d) || keel::is_inf(d) || d != std::floor(d))
        detail::fail(std::string(fn) + ": size inputs must be real "
                     "integers (got " + detail::num(d) + ").",
                     fn, "numkit:" + std::string(fn) + ":sizeNotInteger");
    keel::usize out = 0;
    if (!tryNonNegSize(d, out))
        detail::fail(std::string(fn) + ": requested array size exceeds the "
                     "supported maximum (got " + detail::num(d) + ").",
                     fn, "numkit:" + std::string(fn) + ":sizeTooLarge");
    return out;
}

// ============================================================
// Named-scalar extraction ("input <k> (<name>) must be …").
// ============================================================

inline keel::f64 expectedReal(const Value &v, keel::i32 argNum,
                              std::string_view argName, std::string_view fn)
{
    const std::string id = "numkit:" + std::string(fn) + ":"
                           + std::string(argName);
    if (!v.isScalar())
        detail::fail(std::string(fn) + ": input " + std::to_string(argNum)
                         + " (" + std::string(argName) + ") must be a "
                         "scalar (argument is " + detail::dimsStr(v) + ").",
                     fn, id + "NotScalar");
    if (v.isComplex())
        detail::fail(std::string(fn) + ": input " + std::to_string(argNum)
                         + " (" + std::string(argName) + ") must be real "
                         "(argument is complex).",
                     fn, id + "NotReal");
    return v.toScalar();
}

inline void expectedPositive(keel::f64 v, keel::i32 argNum, std::string_view argName,
                             std::string_view fn)
{
    if (!(v > 0.0))
        detail::fail(std::string(fn) + ": input " + std::to_string(argNum)
                         + " (" + std::string(argName) + ") must be "
                         "positive (got " + detail::num(v) + ").",
                     fn, "numkit:" + std::string(fn) + ":"
                         + std::string(argName) + "NotPositive");
}

// ============================================================
// Dimension arguments.
// ============================================================

inline keel::i32 dimArg(keel::f64 d, std::string_view fn)
{
    if (keel::is_nan(d) || keel::is_inf(d))
        detail::fail(std::string(fn) + ": dimension argument must be finite "
                     "(got " + detail::num(d) + ").",
                     fn, "numkit:" + std::string(fn) + ":dimNotFinite");
    if (d != std::floor(d))
        detail::fail(std::string(fn) + ": dimension argument must be an "
                     "integer (got " + detail::num(d) + ").",
                     fn, "numkit:" + std::string(fn) + ":dimNotInteger");
    if (d < 1.0)
        detail::fail(std::string(fn) + ": dimension argument must be a "
                     "positive integer (got " + detail::num(d) + ").",
                     fn, "numkit:" + std::string(fn) + ":dimNotPositive");
    if (d > static_cast<keel::f64>(keel::Limits<keel::i32>::kMax))
        detail::fail(std::string(fn) + ": dimension argument exceeds maximum "
                     "supported rank (got " + detail::num(d) + ").",
                     fn, "numkit:" + std::string(fn) + ":dimTooLarge");
    return static_cast<keel::i32>(d);
}

inline keel::i32 dimArg(const Value &v, std::string_view fn)
{
    if (!v.isScalar())
        detail::fail(std::string(fn) + ": dimension argument must be a "
                     "scalar (argument is " + detail::dimsStr(v) + ").",
                     fn, "numkit:" + std::string(fn) + ":dimNotScalar");
    if (v.isComplex())
        detail::fail(std::string(fn) + ": dimension argument must be real "
                     "(argument is complex).",
                     fn, "numkit:" + std::string(fn) + ":dimNotReal");
    return dimArg(v.toScalar(), fn);
}

// ============================================================
// Named integer-valued scalar inputs and 0/1 flags.
// ============================================================

inline keel::i32 expectedInteger(keel::f64 v, keel::i32 argNum, std::string_view argName,
                                 std::string_view fn)
{
    if (keel::is_nan(v) || keel::is_inf(v) || v != std::floor(v))
        detail::fail(std::string(fn) + ": input " + std::to_string(argNum)
                         + " (" + std::string(argName) + ") must be an "
                         "integer (got " + detail::num(v) + ").",
                     fn, "numkit:" + std::string(fn) + ":"
                         + std::string(argName) + "NotInteger");
    if (v < static_cast<keel::f64>(keel::Limits<keel::i32>::kMin)
        || v > static_cast<keel::f64>(keel::Limits<keel::i32>::kMax))
        detail::fail(std::string(fn) + ": input " + std::to_string(argNum)
                         + " (" + std::string(argName) + ") exceeds integer "
                         "range (got " + detail::num(v) + ").",
                     fn, "numkit:" + std::string(fn) + ":"
                         + std::string(argName) + "OutOfRange");
    return static_cast<keel::i32>(v);
}

inline void expectedNonNegative(keel::f64 v, keel::i32 argNum, std::string_view argName,
                                std::string_view fn)
{
    if (v < 0.0)
        detail::fail(std::string(fn) + ": input " + std::to_string(argNum)
                         + " (" + std::string(argName) + ") must be "
                         "non-negative (got " + detail::num(v) + ").",
                     fn, "numkit:" + std::string(fn) + ":"
                         + std::string(argName) + "NotNegative");
}

inline void expectedFlag01(keel::f64 v, keel::i32 argNum, std::string_view argName,
                           std::string_view fn)
{
    expectedInteger(v, argNum, argName, fn);
    if (v != 0.0 && v != 1.0)
        detail::fail(std::string(fn) + ": input " + std::to_string(argNum)
                         + " (" + std::string(argName) + ") must be "
                         "0 or 1 (got " + detail::num(v) + ").",
                     fn, "numkit:" + std::string(fn) + ":"
                         + std::string(argName) + "Not01");
}

inline keel::i32 expectedPositiveInteger(keel::f64 v, keel::i32 argNum,
                                         std::string_view argName,
                                         std::string_view fn)
{
    const keel::i32 i = expectedInteger(v, argNum, argName, fn);
    expectedPositive(v, argNum, argName, fn);
    return i;
}

inline void expectedBetween(keel::f64 v, keel::f64 lo, keel::f64 hi, keel::i32 argNum,
                            std::string_view argName, std::string_view fn)
{
    if (!(v >= lo && v <= hi))
        detail::fail(std::string(fn) + ": input " + std::to_string(argNum)
                         + " (" + std::string(argName) + ") must be between "
                         + detail::num(lo) + " and " + detail::num(hi)
                         + " inclusive (got " + detail::num(v) + ").",
                     fn, "numkit:" + std::string(fn) + ":"
                         + std::string(argName) + "OutOfRange");
}

inline keel::i32 expectedOneOf(keel::f64 v, std::initializer_list<keel::f64> choices,
                               keel::i32 argNum, std::string_view argName,
                               std::string_view fn)
{
    expectedInteger(v, argNum, argName, fn);
    for (keel::f64 c : choices)
        if (v == c)
            return static_cast<keel::i32>(v);
    std::string list;
    for (keel::f64 c : choices) {
        if (!list.empty())
            list += ", ";
        list += detail::num(c);
    }
    detail::fail(std::string(fn) + ": input " + std::to_string(argNum)
                     + " (" + std::string(argName) + ") must be one of: "
                     + list + " (got " + detail::num(v) + ").",
                 fn, "numkit:" + std::string(fn) + ":"
                     + std::string(argName) + "NotChoice");
}

inline keel::usize roundLength(const Value &v, keel::i32 argNum, std::string_view argName,
                               std::string_view fn)
{
    const keel::f64 d = expectedReal(v, argNum, argName, fn);
    const keel::f64 r = std::floor(d + 0.5);
    if (!(r >= 0.0))
        detail::fail(std::string(fn) + ": input " + std::to_string(argNum)
                         + " (" + std::string(argName) + ") must be "
                         "non-negative (got " + detail::num(d) + ").",
                     fn, "numkit:" + std::string(fn) + ":"
                         + std::string(argName) + "NotNegative");
    return static_cast<keel::usize>(r);
}

// ============================================================
// String-valued option inputs: one of a fixed choice set.
// ============================================================
inline keel::usize expectedChoice(const Value &v,
                                  std::initializer_list<std::string_view> choices,
                                  keel::i32 argNum, std::string_view argName,
                                  std::string_view fn)
{
    auto describe = [&]() -> std::string {
        if (v.isChar() || v.isString())
            return "got '" + v.toString() + "'";
        if (v.isScalar() && !v.isComplex())
            return "got " + detail::num(v.toScalar());
        return "argument is " + detail::dimsStr(v);
    };
    if (v.isChar() || v.isString()) {
        std::string s = v.toString();
        for (char &c : s)
            c = keel::detail::to_ascii_lower(c);
        keel::usize idx = 0;
        for (std::string_view c : choices) {
            if (s == c)
                return idx;
            ++idx;
        }
    }
    std::string list;
    for (std::string_view c : choices) {
        if (!list.empty())
            list += ", ";
        list += "'";
        list += c;
        list += "'";
    }
    detail::fail(std::string(fn) + ": input " + std::to_string(argNum)
                     + " (" + std::string(argName) + ") must be one of: "
                     + list + " (" + describe() + ").",
                 fn, "numkit:" + std::string(fn) + ":"
                     + std::string(argName) + "NotChoice");
}

// ============================================================
// String choice — prefix-matching semantics.
// Return codes: >=0 matched index; -1 no match; -2 ambiguous prefix.
// ============================================================
inline keel::i32 matchStringChoice(const Value &name,
                                   keel::ArrayRef<std::string_view> choices)
{
    std::string s = name.toString();
    for (char &c : s)
        c = keel::detail::to_ascii_lower(c);
    for (keel::usize i = 0; i < choices.size(); ++i)
        if (s == choices[i])
            return static_cast<keel::i32>(i);
    keel::i32 hit = -1;
    for (keel::usize i = 0; i < choices.size(); ++i) {
        if (choices[i].rfind(s, 0) == 0) {
            if (hit >= 0)
                return -2;
            hit = static_cast<keel::i32>(i);
        }
    }
    return hit;
}

inline keel::i32 matchStringChoice(const Value &name,
                                   keel::ArrayRef<keel::StringRef> choices)
{
    std::string s = name.toString();
    for (char &c : s)
        c = keel::detail::to_ascii_lower(c);
    const keel::StringRef sr(s);
    for (keel::usize i = 0; i < choices.size(); ++i)
        if (choices[i] == sr)
            return static_cast<keel::i32>(i);
    keel::i32 hit = -1;
    for (keel::usize i = 0; i < choices.size(); ++i) {
        if (choices[i].starts_with(sr)) {
            if (hit >= 0)
                return -2;
            hit = static_cast<keel::i32>(i);
        }
    }
    return hit;
}

// ============================================================
// Class-set gates — the accepted-array-class contract.
// ============================================================

enum Class : keel::u32 {
    ClsDbl = 1u << 0,
    ClsSgl = 1u << 1,
    ClsCpx = 1u << 2,
    ClsI8  = 1u << 3,
    ClsI16 = 1u << 4,
    ClsI32 = 1u << 5,
    ClsI64 = 1u << 6,
    ClsU8  = 1u << 7,
    ClsU16 = 1u << 8,
    ClsU32 = 1u << 9,
    ClsU64 = 1u << 10,
    ClsLog = 1u << 11,
    ClsChr = 1u << 12,
};

enum class ClassMask : keel::u32 {};
constexpr ClassMask operator|(ClassMask a, ClassMask b)
{
    return static_cast<ClassMask>(static_cast<keel::u32>(a)
                                  | static_cast<keel::u32>(b));
}
constexpr bool maskAccepts(ClassMask m, Class c)
{
    return (static_cast<keel::u32>(m) & static_cast<keel::u32>(c)) != 0;
}
constexpr ClassMask classMaskOf(Class c) { return static_cast<ClassMask>(c); }

inline Class classBitOf(ValueType t)
{
    switch (t) {
    case ValueType::DOUBLE:  return ClsDbl;
    case ValueType::SINGLE:  return ClsSgl;
    case ValueType::COMPLEX: return ClsCpx;
    case ValueType::INT8:    return ClsI8;
    case ValueType::INT16:   return ClsI16;
    case ValueType::INT32:   return ClsI32;
    case ValueType::INT64:   return ClsI64;
    case ValueType::UINT8:   return ClsU8;
    case ValueType::UINT16:  return ClsU16;
    case ValueType::UINT32:  return ClsU32;
    case ValueType::UINT64:  return ClsU64;
    case ValueType::LOGICAL: return ClsLog;
    case ValueType::CHAR:    return ClsChr;
    default: break;
    }
    return static_cast<Class>(0);
}

struct RejectionVoice {
    std::string_view notAccepted = "invalidType";
    std::string_view complexRejected = "expectedReal";
    std::string_view singleRejected = "";
};

inline void requireClassSet(const Value &x, ClassMask accept,
                            std::string_view fn,
                            std::string_view argName = "A",
                            const RejectionVoice &voice = {})
{
    assert(static_cast<keel::u32>(accept) != 0 && "empty class mask");
    const Class c = classBitOf(x.type());
    if (maskAccepts(accept, c))
        return;
    const std::string got = mtypeName(x.type());
    if (c == ClsCpx && !voice.complexRejected.empty())
        detail::fail(std::string(fn) + ": " + std::string(argName)
                     + " must be real (got a complex array).",
                     fn, "numkit:" + std::string(fn) + ":"
                         + std::string(voice.complexRejected));
    if (c == ClsSgl && maskAccepts(accept, ClsDbl)
        && !voice.singleRejected.empty())
        detail::fail(std::string(fn) + ": " + std::string(argName)
                     + " is not supported in single precision in this "
                       "revision (pass a double array; got "
                     + got + ").",
                     fn, "numkit:" + std::string(fn) + ":"
                         + std::string(voice.singleRejected));
    detail::fail(std::string(fn) + ": " + std::string(argName)
                 + " has an unsupported class for this function (got "
                 + got + ").",
                 fn, "numkit:" + std::string(fn) + ":"
                     + std::string(voice.notAccepted));
}

// ============================================================
// Name-value tail walker.
// ============================================================
template <typename F>
void forEachNvPair(keel::ArrayRef<Value> in, keel::usize start,
                   keel::ArrayRef<std::string_view> names,
                   std::string_view fn, std::string_view tag, F &&dispatch)
{
    if ((in.size() - start) % 2 != 0)
        detail::fail(std::string(fn) + ": each parameter name needs a "
                     "value (unmatched value in the name-value list).",
                     fn, "numkit:" + std::string(fn) + ":nvMismatch");
    for (keel::usize i = start; i + 1 < in.size(); i += 2) {
        const Value &name = in[i];
        if (!(name.isChar() || name.isString()))
            detail::fail(std::string(fn) + ": expected a parameter name "
                         "(unmatched value in the name-value list).",
                         fn, "numkit:" + std::string(fn) + ":nvMismatch");
        const keel::i32 hit = matchStringChoice(name, names);
        if (hit < 0)
            detail::fail(std::string(fn) + ": unrecognized parameter "
                         "name '" + name.toString() + "'.",
                         fn, std::string(tag));
        dispatch(static_cast<keel::usize>(hit), in[i + 1]);
    }
}

template <typename F>
void forEachNvPair(keel::ArrayRef<Value> in, keel::usize start,
                   keel::ArrayRef<keel::StringRef> names,
                   std::string_view fn, std::string_view tag, F &&dispatch)
{
    if ((in.size() - start) % 2 != 0)
        detail::fail(std::string(fn) + ": each parameter name needs a "
                     "value (unmatched value in the name-value list).",
                     fn, "numkit:" + std::string(fn) + ":nvMismatch");
    for (keel::usize i = start; i + 1 < in.size(); i += 2) {
        const Value &name = in[i];
        if (!(name.isChar() || name.isString()))
            detail::fail(std::string(fn) + ": expected a parameter name "
                         "(unmatched value in the name-value list).",
                         fn, "numkit:" + std::string(fn) + ":nvMismatch");
        const keel::i32 hit = matchStringChoice(name, names);
        if (hit < 0)
            detail::fail(std::string(fn) + ": unrecognized parameter "
                         "name '" + name.toString() + "'.",
                         fn, std::string(tag));
        dispatch(static_cast<keel::usize>(hit), in[i + 1]);
    }
}

} // namespace numkit::argcheck
