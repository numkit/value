// include/numkit/value/argcheck.hpp
//
// numkit::argcheck — engine-free argument-validation primitives with
// numkit-voiced diagnostics.

#pragma once

#include <cassert>
#include <initializer_list>

#include <keel/format.h>
#include <keel/math.h>
#include <keel/number.h>
#include <keel/ref.h>
#include <keel/string.h>

#include <numkit/value/error.hpp>
#include <numkit/value/index.hpp>
#include <numkit/value/value.hpp>
#include <numkit/value/value_type.hpp>

namespace numkit::argcheck {

namespace detail {

[[noreturn]] inline void fail(keel::StringRef msg, keel::StringRef fn,
                              keel::StringRef id)
{
    throw Error(msg, 0, 0, fn, "", id);
}

// Value rendering for got-clauses: "2.5", "-5", "1e+30", "Inf", "NaN".
inline keel::FixedString<64> num(keel::f64 v) noexcept
{
    if (keel::is_nan(v))
        return keel::FixedString<64>("NaN");
    if (keel::is_inf(v))
        return keel::FixedString<64>(v > 0 ? "Inf" : "-Inf");
    keel::FixedString<64> out;
    const keel::usize len = keel::format_f64_general(out.inline_buffer(), 63, v, 6, false);
    out.set_inline_size(len);
    return out;
}

// Shape rendering for got-clauses: "1x2", "2x3x4", "0x0".
inline keel::SmallString<64> dimsStr(const Value &v)
{
    const Dims &d = v.dims();
    keel::SmallString<64> s = keel::format_small<64>("{}", d.dim(0));
    for (keel::i32 k = 1; k < d.ndim(); ++k)
        keel::format_into(s, "x{}", d.dim(k));
    return s;
}

} // namespace detail

// ============================================================
// Arity.
// ============================================================
inline void checkNargin(keel::usize actual, keel::usize minArgs, keel::StringRef fn)
{
    if (actual < minArgs)
        detail::fail(keel::format_small<192>(
                         "{}: not enough input arguments — at least {} required (got {}).",
                         fn, minArgs, actual),
                     fn, keel::format_small<96>("numkit:{}:nargin", fn));
}

inline void checkNargin(keel::usize actual, keel::usize minArgs, keel::usize maxArgs,
                        keel::StringRef fn)
{
    checkNargin(actual, minArgs, fn);
    if (actual > maxArgs)
        detail::fail(keel::format_small<192>(
                         "{}: too many input arguments — at most {} accepted (got {}).",
                         fn, maxArgs, actual),
                     fn, keel::format_small<96>("numkit:{}:maxnargin", fn));
}

// ============================================================
// Array-creation size inputs (zeros/ones/eye/true/false/inf/nan/
// rand/randn/randi/strings).
// ============================================================

inline keel::usize arrayDim(keel::f64 d, keel::StringRef fn)
{
    if (keel::is_nan(d) || keel::is_inf(d))
        detail::fail(keel::format_small<192>(
                         "{}: size inputs must be finite (got {}).", fn, detail::num(d)),
                     fn, keel::format_small<96>("numkit:{}:sizeNotFinite", fn));
    if (d != keel::floor(d))
        detail::fail(keel::format_small<192>(
                         "{}: size inputs must be integers (got {}).", fn, detail::num(d)),
                     fn, keel::format_small<96>("numkit:{}:sizeNotInteger", fn));
    if (d < 0.0)
        return 0;
    keel::usize out = 0;
    if (!tryNonNegSize(d, out))
        detail::fail(keel::format_small<192>(
                         "{}: requested array size exceeds the supported maximum (got {}).",
                         fn, detail::num(d)),
                     fn, keel::format_small<96>("numkit:{}:sizeTooLarge", fn));
    return out;
}

inline void arrayDimReal(const Value &v, keel::StringRef fn)
{
    if (v.isComplex())
        detail::fail(keel::format_small<192>(
                         "{}: size inputs must be real (argument is complex).", fn),
                     fn, keel::format_small<96>("numkit:{}:sizeNotReal", fn));
}

inline keel::usize arrayDim(const Value &v, keel::StringRef fn)
{
    if (!v.isScalar())
        detail::fail(keel::format_small<192>(
                         "{}: size inputs must be scalars (argument is {}).",
                         fn, detail::dimsStr(v)),
                     fn, keel::format_small<96>("numkit:{}:sizeNotScalar", fn));
    arrayDimReal(v, fn);
    return arrayDim(v.toScalar(), fn);
}

// ============================================================
// reshape size inputs.
// ============================================================

inline void reshapeSizeReal(const Value &v, keel::StringRef fn)
{
    if (v.isComplex())
        detail::fail(keel::format_small<192>(
                         "{}: size inputs must be real (argument is complex).", fn),
                     fn, keel::format_small<96>("numkit:{}:sizeNotReal", fn));
}

inline keel::f64 reshapeSizeArg(const Value &v, keel::StringRef fn)
{
    if (!v.isScalar())
        detail::fail(keel::format_small<192>(
                         "{}: size inputs must be scalars (argument is {}).",
                         fn, detail::dimsStr(v)),
                     fn, keel::format_small<96>("numkit:{}:sizeNotScalar", fn));
    reshapeSizeReal(v, fn);
    return v.toScalar();
}

inline keel::usize reshapeSize(keel::f64 d, keel::StringRef fn)
{
    if (d < 0.0)
        detail::fail(keel::format_small<192>(
                         "{}: size inputs must be non-negative (got {}).", fn, detail::num(d)),
                     fn, keel::format_small<96>("numkit:{}:sizeNegative", fn));
    if (!keel::is_integer(d))
        detail::fail(keel::format_small<192>(
                         "{}: size inputs must be real integers (got {}).", fn, detail::num(d)),
                     fn, keel::format_small<96>("numkit:{}:sizeNotInteger", fn));
    keel::usize out = 0;
    if (!tryNonNegSize(d, out))
        detail::fail(keel::format_small<192>(
                         "{}: requested array size exceeds the supported maximum (got {}).",
                         fn, detail::num(d)),
                     fn, keel::format_small<96>("numkit:{}:sizeTooLarge", fn));
    return out;
}

// ============================================================
// Named-scalar extraction ("input <k> (<name>) must be …").
// ============================================================

inline keel::f64 expectedReal(const Value &v, keel::i32 argNum,
                              keel::StringRef argName, keel::StringRef fn)
{
    if (!v.isScalar())
        detail::fail(keel::format_small<192>(
                         "{}: input {} ({}) must be a scalar (argument is {}).",
                         fn, argNum, argName, detail::dimsStr(v)),
                     fn, keel::format_small<96>("numkit:{}:{}NotScalar", fn, argName));
    if (v.isComplex())
        detail::fail(keel::format_small<192>(
                         "{}: input {} ({}) must be real (argument is complex).",
                         fn, argNum, argName),
                     fn, keel::format_small<96>("numkit:{}:{}NotReal", fn, argName));
    return v.toScalar();
}

inline void expectedPositive(keel::f64 v, keel::i32 argNum, keel::StringRef argName,
                             keel::StringRef fn)
{
    if (!(v > 0.0))
        detail::fail(keel::format_small<192>(
                         "{}: input {} ({}) must be positive (got {}).",
                         fn, argNum, argName, detail::num(v)),
                     fn, keel::format_small<96>("numkit:{}:{}NotPositive", fn, argName));
}

// ============================================================
// Dimension arguments.
// ============================================================

inline keel::i32 dimArg(keel::f64 d, keel::StringRef fn)
{
    if (keel::is_nan(d) || keel::is_inf(d))
        detail::fail(keel::format_small<192>(
                         "{}: dimension argument must be finite (got {}).", fn, detail::num(d)),
                     fn, keel::format_small<96>("numkit:{}:dimNotFinite", fn));
    if (d != keel::floor(d))
        detail::fail(keel::format_small<192>(
                         "{}: dimension argument must be an integer (got {}).", fn, detail::num(d)),
                     fn, keel::format_small<96>("numkit:{}:dimNotInteger", fn));
    if (d < 1.0)
        detail::fail(keel::format_small<192>(
                         "{}: dimension argument must be a positive integer (got {}).",
                         fn, detail::num(d)),
                     fn, keel::format_small<96>("numkit:{}:dimNotPositive", fn));
    if (d > static_cast<keel::f64>(keel::Limits<keel::i32>::kMax))
        detail::fail(keel::format_small<192>(
                         "{}: dimension argument exceeds maximum supported rank (got {}).",
                         fn, detail::num(d)),
                     fn, keel::format_small<96>("numkit:{}:dimTooLarge", fn));
    return static_cast<keel::i32>(d);
}

inline keel::i32 dimArg(const Value &v, keel::StringRef fn)
{
    if (!v.isScalar())
        detail::fail(keel::format_small<192>(
                         "{}: dimension argument must be a scalar (argument is {}).",
                         fn, detail::dimsStr(v)),
                     fn, keel::format_small<96>("numkit:{}:dimNotScalar", fn));
    if (v.isComplex())
        detail::fail(keel::format_small<192>(
                         "{}: dimension argument must be real (argument is complex).", fn),
                     fn, keel::format_small<96>("numkit:{}:dimNotReal", fn));
    return dimArg(v.toScalar(), fn);
}

// ============================================================
// Named integer-valued scalar inputs and 0/1 flags.
// ============================================================

inline keel::i32 expectedInteger(keel::f64 v, keel::i32 argNum, keel::StringRef argName,
                                 keel::StringRef fn)
{
    if (!keel::is_integer(v))
        detail::fail(keel::format_small<192>(
                         "{}: input {} ({}) must be an integer (got {}).",
                         fn, argNum, argName, detail::num(v)),
                     fn, keel::format_small<96>("numkit:{}:{}NotInteger", fn, argName));
    if (v < static_cast<keel::f64>(keel::Limits<keel::i32>::kMin)
        || v > static_cast<keel::f64>(keel::Limits<keel::i32>::kMax))
        detail::fail(keel::format_small<192>(
                         "{}: input {} ({}) exceeds integer range (got {}).",
                         fn, argNum, argName, detail::num(v)),
                     fn, keel::format_small<96>("numkit:{}:{}OutOfRange", fn, argName));
    return static_cast<keel::i32>(v);
}

inline void expectedNonNegative(keel::f64 v, keel::i32 argNum, keel::StringRef argName,
                                keel::StringRef fn)
{
    if (v < 0.0)
        detail::fail(keel::format_small<192>(
                         "{}: input {} ({}) must be non-negative (got {}).",
                         fn, argNum, argName, detail::num(v)),
                     fn, keel::format_small<96>("numkit:{}:{}NotNegative", fn, argName));
}

inline void expectedFlag01(keel::f64 v, keel::i32 argNum, keel::StringRef argName,
                           keel::StringRef fn)
{
    expectedInteger(v, argNum, argName, fn);
    if (v != 0.0 && v != 1.0)
        detail::fail(keel::format_small<192>(
                         "{}: input {} ({}) must be 0 or 1 (got {}).",
                         fn, argNum, argName, detail::num(v)),
                     fn, keel::format_small<96>("numkit:{}:{}Not01", fn, argName));
}

inline keel::i32 expectedPositiveInteger(keel::f64 v, keel::i32 argNum,
                                         keel::StringRef argName,
                                         keel::StringRef fn)
{
    const keel::i32 i = expectedInteger(v, argNum, argName, fn);
    expectedPositive(v, argNum, argName, fn);
    return i;
}

inline void expectedBetween(keel::f64 v, keel::f64 lo, keel::f64 hi, keel::i32 argNum,
                            keel::StringRef argName, keel::StringRef fn)
{
    if (!(v >= lo && v <= hi))
        detail::fail(keel::format_small<192>(
                         "{}: input {} ({}) must be between {} and {} inclusive (got {}).",
                         fn, argNum, argName, detail::num(lo), detail::num(hi), detail::num(v)),
                     fn, keel::format_small<96>("numkit:{}:{}OutOfRange", fn, argName));
}

inline keel::i32 expectedOneOf(keel::f64 v, std::initializer_list<keel::f64> choices,
                               keel::i32 argNum, keel::StringRef argName,
                               keel::StringRef fn)
{
    expectedInteger(v, argNum, argName, fn);
    for (keel::f64 c : choices)
        if (v == c)
            return static_cast<keel::i32>(v);
    keel::SmallString<128> list;
    for (keel::f64 c : choices) {
        if (!list.empty())
            list.append(", ");
        list.append(detail::num(c));
    }
    detail::fail(keel::format_small<192>(
                     "{}: input {} ({}) must be one of: {} (got {}).",
                     fn, argNum, argName, list, detail::num(v)),
                 fn, keel::format_small<96>("numkit:{}:{}NotChoice", fn, argName));
}

inline keel::usize roundLength(const Value &v, keel::i32 argNum, keel::StringRef argName,
                               keel::StringRef fn)
{
    const keel::f64 d = expectedReal(v, argNum, argName, fn);
    const keel::f64 r = keel::floor(d + 0.5);
    if (!(r >= 0.0))
        detail::fail(keel::format_small<192>(
                         "{}: input {} ({}) must be non-negative (got {}).",
                         fn, argNum, argName, detail::num(d)),
                     fn, keel::format_small<96>("numkit:{}:{}NotNegative", fn, argName));
    return static_cast<keel::usize>(r);
}

// ============================================================
// String-valued option inputs: one of a fixed choice set.
// ============================================================
inline keel::usize expectedChoice(const Value &v,
                                  std::initializer_list<keel::StringRef> choices,
                                  keel::i32 argNum, keel::StringRef argName,
                                  keel::StringRef fn)
{
    auto describe = [&]() -> keel::SmallString<96> {
        if (v.isChar() || v.isString())
            return keel::format_small<96>("got '{}'", v.asStringRef());
        if (v.isScalar() && !v.isComplex())
            return keel::format_small<96>("got {}", detail::num(v.toScalar()));
        return keel::format_small<96>("argument is {}", detail::dimsStr(v));
    };
    if (v.isChar() || v.isString()) {
        const keel::StringRef sr = v.asStringRef();
        keel::usize idx = 0;
        for (keel::StringRef c : choices) {
            if (sr.equals_ignore_ascii_case(c))
                return idx;
            ++idx;
        }
    }
    keel::SmallString<128> list;
    for (keel::StringRef c : choices) {
        if (!list.empty())
            list.append(", ");
        list.push_back('\'');
        list.append(c);
        list.push_back('\'');
    }
    detail::fail(keel::format_small<256>(
                     "{}: input {} ({}) must be one of: {} ({}).",
                     fn, argNum, argName, list, describe()),
                 fn, keel::format_small<96>("numkit:{}:{}NotChoice", fn, argName));
}

// ============================================================
// String choice — prefix-matching semantics.
// Return codes: >=0 matched index; -1 no match; -2 ambiguous prefix.
// ============================================================
template <typename Str = keel::StringRef>
    requires keel::IsConvertible<const Str &, keel::StringRef>
inline keel::i32 matchStringChoice(const Value &name,
                                   keel::ArrayRef<Str> choices)
{
    const keel::StringRef sr = name.asStringRef();
    for (keel::usize i = 0; i < choices.size(); ++i)
        if (keel::StringRef(choices[i]).equals_ignore_ascii_case(sr))
            return static_cast<keel::i32>(i);
    keel::i32 hit = -1;
    for (keel::usize i = 0; i < choices.size(); ++i) {
        const keel::StringRef cand(choices[i]);
        if (cand.size() >= sr.size() && cand.substr(0, sr.size()).equals_ignore_ascii_case(sr)) {
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
    keel::StringRef notAccepted = "invalidType";
    keel::StringRef complexRejected = "expectedReal";
    keel::StringRef singleRejected = "";
};

inline void requireClassSet(const Value &x, ClassMask accept,
                            keel::StringRef fn,
                            keel::StringRef argName = "A",
                            const RejectionVoice &voice = {})
{
    assert(static_cast<keel::u32>(accept) != 0 && "empty class mask");
    const Class c = classBitOf(x.type());
    if (maskAccepts(accept, c))
        return;
    const char *got = mtypeName(x.type());
    if (c == ClsCpx && !voice.complexRejected.empty())
        detail::fail(keel::format_small<192>(
                         "{}: {} must be real (got a complex array).", fn, argName),
                     fn, keel::format_small<96>("numkit:{}:{}", fn, voice.complexRejected));
    if (c == ClsSgl && maskAccepts(accept, ClsDbl)
        && !voice.singleRejected.empty())
        detail::fail(keel::format_small<256>(
                         "{}: {} is not supported in single precision in this revision "
                         "(pass a double array; got {}).",
                         fn, argName, got),
                     fn, keel::format_small<96>("numkit:{}:{}", fn, voice.singleRejected));
    detail::fail(keel::format_small<192>(
                     "{}: {} has an unsupported class for this function (got {}).",
                     fn, argName, got),
                 fn, keel::format_small<96>("numkit:{}:{}", fn, voice.notAccepted));
}

// ============================================================
// Name-value tail walker.
// ============================================================
template <typename Str = keel::StringRef, typename F>
    requires keel::IsConvertible<const Str &, keel::StringRef>
void forEachNvPair(keel::ArrayRef<Value> in, keel::usize start,
                   keel::ArrayRef<Str> names,
                   keel::StringRef fn, keel::StringRef tag, F &&dispatch)
{
    if ((in.size() - start) % 2 != 0)
        detail::fail(keel::format_small<192>(
                         "{}: each parameter name needs a value (unmatched value in the name-value list).",
                         fn),
                     fn, keel::format_small<96>("numkit:{}:nvMismatch", fn));
    for (keel::usize i = start; i + 1 < in.size(); i += 2) {
        const Value &name = in[i];
        if (!(name.isChar() || name.isString()))
            detail::fail(keel::format_small<192>(
                             "{}: expected a parameter name (unmatched value in the name-value list).",
                             fn),
                         fn, keel::format_small<96>("numkit:{}:nvMismatch", fn));
        const keel::i32 hit = matchStringChoice(name, names);
        if (hit < 0)
            detail::fail(keel::format_small<192>(
                             "{}: unrecognized parameter name '{}'.", fn, name.asStringRef()),
                         fn, tag);
        dispatch(static_cast<keel::usize>(hit), in[i + 1]);
    }
}

} // namespace numkit::argcheck
