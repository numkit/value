// value/include/nk/value/argcheck.hpp
//
// nk::argcheck — engine-free argument-validation primitives with
// nk-voiced diagnostics.
//
// Part of the explicit argcheck separation (dev-docs/todo/
// argcheck_separation.md): reg adapters marshal, compute kernels
// validate through <fn>_argcheck built on these primitives, so
// interpreter and direct-C++ paths cross the same checks.
//
// Diagnostic policy (user rule 2026-09-17):
//   * WHICH inputs are rejected (and which argument gets blamed) is
//     validated against the reference engine — same inputs must fail
//     with the same error class (behavior parity, full_fidelity.md);
//   * the STRINGS are ours: identifiers use the nk:<fn>:<tag>
//     namespace, messages are more detailed than the reference's
//     (function, argument, expectation, and the received value/shape).
//     The reference's name or version never appears in shipped strings
//     or code comments here.
//   * a primitive joins this header only when a real consumer needs it
//     (no speculative vocabulary).

#pragma once

#include <cassert>
#include <cmath>
#include <cstddef>
#include <limits>
#include <sstream>
#include <string>
#include <string_view>

#include <nk/value/error.hpp>
#include <nk/value/span.hpp>
#include <nk/value/value.hpp>
#include <nk/value/value_type.hpp>   // mtypeName (class got-clauses)

namespace nk::argcheck {

namespace detail {

[[noreturn]] inline void fail(const std::string &msg, std::string_view fn,
                              const std::string &id)
{
    throw Error(msg, 0, 0, std::string(fn), "", id);
}

// Value rendering for got-clauses: "2.5", "-5", "1e+30", "Inf", "NaN".
inline std::string num(double v)
{
    if (std::isnan(v))
        return "NaN";
    if (std::isinf(v))
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
    for (int k = 1; k < d.ndim(); ++k) {
        s += "x";
        s += std::to_string(d.dim(k));
    }
    return s;
}

} // namespace detail

// ============================================================
// Arity.
// ============================================================
inline void checkNargin(size_t actual, size_t minArgs, std::string_view fn)
{
    if (actual < minArgs)
        detail::fail(std::string(fn) + ": not enough input arguments — at "
                     "least " + std::to_string(minArgs) + " required (got "
                         + std::to_string(actual) + ").",
                     fn, "nk:" + std::string(fn) + ":nargin");
}

inline void checkNargin(size_t actual, size_t minArgs, size_t maxArgs,
                        std::string_view fn)
{
    checkNargin(actual, minArgs, fn);
    if (actual > maxArgs)
        detail::fail(std::string(fn) + ": too many input arguments — at "
                     "most " + std::to_string(maxArgs) + " accepted (got "
                         + std::to_string(actual) + ").",
                     fn, "nk:" + std::string(fn) + ":maxnargin");
}

// ============================================================
// Array-creation size inputs (zeros/ones/eye/true/false/inf/nan/
// rand/randn/randi/strings).
//
// Reference-probed contract (2026-09-16): a negative integer size
// CLAMPS to an empty array (zeros(-1,3) -> 0x3); NaN/Inf, fractional,
// non-scalar and complex size inputs are each rejected as their own
// error class.
// ============================================================

// Numeric tail — use when the Value-level checks are already done (e.g.
// looping over the elements of a [m n p] size vector).
inline size_t arrayDim(double d, std::string_view fn)
{
    if (std::isnan(d) || std::isinf(d))
        detail::fail(std::string(fn) + ": size inputs must be finite (got "
                     + detail::num(d) + ").",
                     fn, "nk:" + std::string(fn) + ":sizeNotFinite");
    if (d != std::floor(d))
        detail::fail(std::string(fn) + ": size inputs must be integers "
                     "(got " + detail::num(d) + ").",
                     fn, "nk:" + std::string(fn) + ":sizeNotInteger");
    if (d < 0.0)
        return 0;
    if (d >= static_cast<double>(std::numeric_limits<size_t>::max()))
        detail::fail(std::string(fn) + ": requested array size exceeds the "
                     "supported maximum (got " + detail::num(d) + ").",
                     fn, "nk:" + std::string(fn) + ":sizeTooLarge");
    return static_cast<size_t>(d);
}

// Realness guard for a size input of ANY shape (scalar or [m n p]
// vector) — shared by the scalar and vector parsing forms.
inline void arrayDimReal(const Value &v, std::string_view fn)
{
    if (v.isComplex())
        detail::fail(std::string(fn) + ": size inputs must be real "
                     "(argument is complex).",
                     fn, "nk:" + std::string(fn) + ":sizeNotReal");
}

// Whole-argument form — scalar/real checks first (a complex SCALAR is
// isScalar()==true, so the complex check must come second to blame
// realness, not shape), then the numeric tail.
inline size_t arrayDim(const Value &v, std::string_view fn)
{
    if (!v.isScalar())
        detail::fail(std::string(fn) + ": size inputs must be scalars "
                     "(argument is " + detail::dimsStr(v) + ").",
                     fn, "nk:" + std::string(fn) + ":sizeNotScalar");
    arrayDimReal(v, fn);
    return arrayDim(v.toScalar(), fn);
}

// ============================================================
// reshape size inputs.
//
// Reference-probed contract (2026-09-16/17): reshape REJECTS what the
// creation family tolerates — negative and fractional sizes are each
// their own error class, and the negative check fires first.
// ============================================================

// Realness guard for a reshape size input of ANY shape.
inline void reshapeSizeReal(const Value &v, std::string_view fn)
{
    if (v.isComplex())
        detail::fail(std::string(fn) + ": size inputs must be real "
                     "(argument is complex).",
                     fn, "nk:" + std::string(fn) + ":sizeNotReal");
}

// Value-level extraction for one scalar size position. The caller checks
// for [] inference (an empty argument) BEFORE calling — empties never
// reach this function.
inline double reshapeSizeArg(const Value &v, std::string_view fn)
{
    if (!v.isScalar())
        detail::fail(std::string(fn) + ": size inputs must be scalars "
                     "(argument is " + detail::dimsStr(v) + ").",
                     fn, "nk:" + std::string(fn) + ":sizeNotScalar");
    reshapeSizeReal(v, fn);
    return v.toScalar();
}

inline size_t reshapeSize(double d, std::string_view fn)
{
    if (d < 0.0)
        detail::fail(std::string(fn) + ": size inputs must be non-negative "
                     "(got " + detail::num(d) + ").",
                     fn, "nk:" + std::string(fn) + ":sizeNegative");
    if (std::isnan(d) || std::isinf(d) || d != std::floor(d))
        detail::fail(std::string(fn) + ": size inputs must be real "
                     "integers (got " + detail::num(d) + ").",
                     fn, "nk:" + std::string(fn) + ":sizeNotInteger");
    // Cast-safety guard only (the numel-preservation check in reshape
    // catches unrealistic sizes first).
    if (d >= static_cast<double>(std::numeric_limits<size_t>::max()))
        detail::fail(std::string(fn) + ": requested array size exceeds the "
                     "supported maximum (got " + detail::num(d) + ").",
                     fn, "nk:" + std::string(fn) + ":sizeTooLarge");
    return static_cast<size_t>(d);
}

// ============================================================
// Named-scalar extraction ("input <k> (<name>) must be …").
// The richer twin of the reference's validateattributes family: the
// message names the position, the argument, the expectation AND the
// received value/shape.
// ============================================================

// Extract a real scalar. argNum/argName name the input position.
inline double expectedReal(const Value &v, int argNum,
                           std::string_view argName, std::string_view fn)
{
    const std::string id = "nk:" + std::string(fn) + ":"
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

// Domain check on an already-extracted value. `!(v > 0)` so NaN is
// rejected here too.
inline void expectedPositive(double v, int argNum, std::string_view argName,
                             std::string_view fn)
{
    if (!(v > 0.0))
        detail::fail(std::string(fn) + ": input " + std::to_string(argNum)
                         + " (" + std::string(argName) + ") must be "
                         "positive (got " + detail::num(v) + ").",
                     fn, "nk:" + std::string(fn) + ":"
                         + std::string(argName) + "NotPositive");
}

// ============================================================
// Dimension arguments — the dim=2 / vecdim inputs of the reduction
// families (bounds, iqr, median, mode, moment, trimmean, range,
// geomean, harmmean, mad, zscore …).
//
// Reference-probed contract (2026-09-17): a positive integer scalar
// (fractional / zero / negative are all rejected); vecdim entries carry
// the same per-element contract plus the caller's range check.
// ============================================================

// Numeric tail — use when the Value-level checks are already done (e.g.
// looping over the elements of a vecdim vector).
inline int dimArg(double d, std::string_view fn)
{
    if (std::isnan(d) || std::isinf(d))
        detail::fail(std::string(fn) + ": dimension argument must be finite "
                     "(got " + detail::num(d) + ").",
                     fn, "nk:" + std::string(fn) + ":dimNotFinite");
    if (d != std::floor(d))
        detail::fail(std::string(fn) + ": dimension argument must be an "
                     "integer (got " + detail::num(d) + ").",
                     fn, "nk:" + std::string(fn) + ":dimNotInteger");
    if (d < 1.0)
        detail::fail(std::string(fn) + ": dimension argument must be a "
                     "positive integer (got " + detail::num(d) + ").",
                     fn, "nk:" + std::string(fn) + ":dimNotPositive");
    return static_cast<int>(d);
}

// Whole-argument form — scalar / real checks first (a complex SCALAR is
// isScalar()==true, so the complex check must come second to blame
// realness, not shape), then the numeric tail.
inline int dimArg(const Value &v, std::string_view fn)
{
    if (!v.isScalar())
        detail::fail(std::string(fn) + ": dimension argument must be a "
                     "scalar (argument is " + detail::dimsStr(v) + ").",
                     fn, "nk:" + std::string(fn) + ":dimNotScalar");
    if (v.isComplex())
        detail::fail(std::string(fn) + ": dimension argument must be real "
                     "(argument is complex).",
                     fn, "nk:" + std::string(fn) + ":dimNotReal");
    return dimArg(v.toScalar(), fn);
}

// ============================================================
// Named integer-valued scalar inputs (k, order, NumLags, …) and 0/1
// flags. Integer-ness is one class; the sign policy is the caller's
// separate check (k and order are NON-negative — 0 is legal; NumLags is
// positive; cov/var/mad/zscore flags are exactly 0 or 1).
// Reference-probed 2026-09-17.
// ============================================================

inline int expectedInteger(double v, int argNum, std::string_view argName,
                           std::string_view fn)
{
    if (std::isnan(v) || std::isinf(v) || v != std::floor(v))
        detail::fail(std::string(fn) + ": input " + std::to_string(argNum)
                         + " (" + std::string(argName) + ") must be an "
                         "integer (got " + detail::num(v) + ").",
                     fn, "nk:" + std::string(fn) + ":"
                         + std::string(argName) + "NotInteger");
    return static_cast<int>(v);
}

inline void expectedNonNegative(double v, int argNum, std::string_view argName,
                                std::string_view fn)
{
    if (v < 0.0)
        detail::fail(std::string(fn) + ": input " + std::to_string(argNum)
                         + " (" + std::string(argName) + ") must be "
                         "non-negative (got " + detail::num(v) + ").",
                     fn, "nk:" + std::string(fn) + ":"
                         + std::string(argName) + "NotNegative");
}

inline void expectedFlag01(double v, int argNum, std::string_view argName,
                           std::string_view fn)
{
    expectedInteger(v, argNum, argName, fn);
    if (v != 0.0 && v != 1.0)
        detail::fail(std::string(fn) + ": input " + std::to_string(argNum)
                         + " (" + std::string(argName) + ") must be "
                         "0 or 1 (got " + detail::num(v) + ").",
                     fn, "nk:" + std::string(fn) + ":"
                         + std::string(argName) + "Not01");
}

// The dominant count/order family (k, order, window, span, level, N …):
// positive integer. Integer-ness fires first, so 2.5 reports NotInteger
// and -3 reports NotPositive — same rejection classes as the reference's
// "must be a positive integer", split into richer nk-voiced diagnostics.
inline int expectedPositiveInteger(double v, int argNum, std::string_view argName,
                                   std::string_view fn)
{
    const int i = expectedInteger(v, argNum, argName, fn);
    expectedPositive(v, argNum, argName, fn);
    return i;
}

// Inclusive range check on an already-extracted value. `lo`/`hi` are
// included on both ends; NaN fails (comparison with itself).
inline void expectedBetween(double v, double lo, double hi, int argNum,
                            std::string_view argName, std::string_view fn)
{
    if (!(v >= lo && v <= hi))
        detail::fail(std::string(fn) + ": input " + std::to_string(argNum)
                         + " (" + std::string(argName) + ") must be between "
                         + detail::num(lo) + " and " + detail::num(hi)
                         + " inclusive (got " + detail::num(v) + ").",
                     fn, "nk:" + std::string(fn) + ":"
                         + std::string(argName) + "OutOfRange");
}

// Numeric choice set — connectivity {4,8} / {6,18,26}, bit depths
// {8,10,12,16}, LUT sizes {2,3} … Integer-ness first (2.5 reports
// NotInteger), then membership. Reference-probed (2026-09-17):
// connectivity 5 / 2.5 / 0 are all rejected as one class.
inline int expectedOneOf(double v, std::initializer_list<double> choices,
                         int argNum, std::string_view argName,
                         std::string_view fn)
{
    expectedInteger(v, argNum, argName, fn);
    for (double c : choices)
        if (v == c)
            return static_cast<int>(v);
    std::string list;
    for (double c : choices) {
        if (!list.empty())
            list += ", ";
        list += detail::num(c);
    }
    detail::fail(std::string(fn) + ": input " + std::to_string(argNum)
                     + " (" + std::string(argName) + ") must be one of: "
                     + list + " (got " + detail::num(v) + ").",
                 fn, "nk:" + std::string(fn) + ":"
                     + std::string(argName) + "NotChoice");
}

// Length inputs whose reference ROUNDS a fractional value with a warning
// instead of rejecting (probed 2026-09-17: hamming(2.5) -> length 3,
// lpc(x, 2.5) rounds) — rejection parity therefore REQUIRES accepting
// them here too. Real check, round half away from zero, then the
// negative class (hamming(-1) errors as nonnegative).
inline size_t roundLength(const Value &v, int argNum, std::string_view argName,
                          std::string_view fn)
{
    const double d = expectedReal(v, argNum, argName, fn);
    const double r = std::floor(d + 0.5);
    if (!(r >= 0.0))
        detail::fail(std::string(fn) + ": input " + std::to_string(argNum)
                         + " (" + std::string(argName) + ") must be "
                         "non-negative (got " + detail::num(d) + ").",
                     fn, "nk:" + std::string(fn) + ":"
                         + std::string(argName) + "NotNegative");
    return static_cast<size_t>(r);
}

// ============================================================
// String-valued option inputs: one of a fixed choice set
// ('WidthReference' → 'halfprom'|'halfheight', SortStr, criteria,
// methods …). Case-insensitive; returns the matched index. Unifies the
// hand-rolled lowercase-loop + switch + throw blocks the option parsers
// used to copy around.
// ============================================================
inline size_t expectedChoice(const Value &v,
                             std::initializer_list<std::string_view> choices,
                             int argNum, std::string_view argName,
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
            c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        size_t idx = 0;
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
                 fn, "nk:" + std::string(fn) + ":"
                     + std::string(argName) + "NotChoice");
}


// ============================================================
// String choice — validatestring (prefix) semantics.
//
// expectedChoice above requires the EXACT (case-insensitive) string.
// The reference's method/flag parsers use validatestring: an exact
// match wins, otherwise a UNIQUE case-insensitive prefix. This probe
// owns the lowercasing (boundary rule: callers pass the RAW value).
// Return codes: >=0 matched index; -1 no match; -2 ambiguous prefix.
// An empty string matches nothing unless it is the single choice.
// ============================================================
inline int matchStringChoice(const Value &name,
                             Span<const std::string_view> choices)
{
    std::string s = name.toString();
    for (char &c : s)
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    for (std::size_t i = 0; i < choices.size(); ++i)
        if (s == choices[i])
            return static_cast<int>(i);
    int hit = -1;
    for (std::size_t i = 0; i < choices.size(); ++i) {
        if (choices[i].rfind(s, 0) == 0) {
            if (hit >= 0)
                return -2;
            hit = static_cast<int>(i);
        }
    }
    return hit;
}

// ============================================================
// Class-set gates — the accepted-array-class contract.
//
// The reference's type matrix is PER-FUNCTION inconsistent (probed
// per family: one entry takes single+double, its sibling rejects
// logical, another coerces char — see the family portions in
// dev-docs/todo/argcheck_separation.md). So the MECHANISM lives here
// (parameterized by a mask) and the PROBED masks stay in each
// family's detail header next to the tests that pin them.
//
// Preconditions / invariants:
//   * the class is read via type() ONLY — never through element
//     readers (the linalg lesson: dispatch predicates must not touch
//     data);
//   * the Cpx bit test is exact because the engine collapses a +0i
//     imaginary part to REAL storage, so type()==COMPLEX means a
//     nonzero imaginary part is present (if that collapse invariant
//     ever changes, this gate changes meaning — single-complex
//     storage is bugs/opened/core/single-complex-storage.md);
//   * ordering: this class gate runs BEFORE any extraction primitive
//     (it is the cheapest check and owns the type-blaming);
//   * empty masks are a programming error (assert in debug builds).
//
// Voice policy: the DEFAULTS are the parity voices (the reference
// itself rejects the class). A family whose KERNEL cannot compute a
// class the reference accepts (honest capability gap, e.g. single
// where the kernel is double-only — bugs/opened/linalg/
// single-float-inputs.md) overrides singleRejected with its gap id;
// that override must carry a filed bug. complexRejected defaults to
// the expectedReal class the reference's validators use.
// ============================================================

enum Class : uint32_t {
    ClsDbl = 1u << 0,
    ClsSgl = 1u << 1,
    ClsCpx = 1u << 2,   // complex storage (double-complex; see invariant above)
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

// Strong mask type: bitwise combinable, not accidentally an int.
enum class ClassMask : uint32_t {};
constexpr ClassMask operator|(ClassMask a, ClassMask b)
{
    return static_cast<ClassMask>(static_cast<uint32_t>(a)
                                  | static_cast<uint32_t>(b));
}
constexpr bool maskAccepts(ClassMask m, Class c)
{
    return (static_cast<uint32_t>(m) & static_cast<uint32_t>(c)) != 0;
}
constexpr ClassMask classMaskOf(Class c) { return static_cast<ClassMask>(c); }

// The single mapping point ValueType -> Class bit. One switch, tested.
inline Class classBitOf(ValueType t)
{
    switch (t) {
    case ValueType::DOUBLE: return ClsDbl;
    case ValueType::SINGLE: return ClsSgl;
    case ValueType::COMPLEX: return ClsCpx;
    case ValueType::INT8:   return ClsI8;
    case ValueType::INT16:  return ClsI16;
    case ValueType::INT32:  return ClsI32;
    case ValueType::INT64:  return ClsI64;
    case ValueType::UINT8:  return ClsU8;
    case ValueType::UINT16: return ClsU16;
    case ValueType::UINT32: return ClsU32;
    case ValueType::UINT64: return ClsU64;
    case ValueType::LOGICAL: return ClsLog;
    case ValueType::CHAR:   return ClsChr;
    default: break;
    }
    return static_cast<Class>(0);
}

struct RejectionVoice {
    // Parity defaults; singleRejected empty -> falls back to notAccepted
    // (override ONLY for kernel capability gaps with a filed bug).
    std::string_view notAccepted = "invalidType";
    std::string_view complexRejected = "expectedReal";
    std::string_view singleRejected = "";
};

inline void requireClassSet(const Value &x, ClassMask accept,
                            std::string_view fn,
                            std::string_view argName = "A",
                            const RejectionVoice &voice = {})
{
    assert(static_cast<uint32_t>(accept) != 0 && "empty class mask");
    const Class c = classBitOf(x.type());
    if (maskAccepts(accept, c))
        return;
    const std::string got = mtypeName(x.type());
    if (c == ClsCpx && !voice.complexRejected.empty())
        detail::fail(std::string(fn) + ": " + std::string(argName)
                     + " must be real (got a complex array).",
                     fn, "nk:" + std::string(fn) + ":"
                         + std::string(voice.complexRejected));
    if (c == ClsSgl && maskAccepts(accept, ClsDbl)
        && !voice.singleRejected.empty())
        detail::fail(std::string(fn) + ": " + std::string(argName)
                     + " is not supported in single precision in this "
                       "revision (pass a double array; got "
                     + got + ").",
                     fn, "nk:" + std::string(fn) + ":"
                         + std::string(voice.singleRejected));
    detail::fail(std::string(fn) + ": " + std::string(argName)
                 + " has an unsupported class for this function (got "
                 + got + ").",
                 fn, "nk:" + std::string(fn) + ":"
                     + std::string(voice.notAccepted));
}


// ============================================================
// Name-value tail walker.
//
// The last shared pattern of the option parsers: walk (name, value)
// pairs over a raw tail, validate the pair parity and the name
// (prefix semantics of matchStringChoice), and hand each matched
// (nameIdx, value) to the caller's dispatch lambda. Callers keep
// their per-name domain validation. Unknown names and odd tails are
// voiced here with the caller's tag.
// ============================================================
template <typename F>
void forEachNvPair(Span<const Value> in, std::size_t start,
                   Span<const std::string_view> names,
                   std::string_view fn, std::string_view tag, F &&dispatch)
{
    if ((in.size() - start) % 2 != 0)
        detail::fail(std::string(fn) + ": each parameter name needs a "
                     "value (unmatched value in the name-value list).",
                     fn, "nk:" + std::string(fn) + ":nvMismatch");
    for (std::size_t i = start; i + 1 < in.size(); i += 2) {
        const Value &name = in[i];
        if (!(name.isChar() || name.isString()))
            detail::fail(std::string(fn) + ": expected a parameter name "
                         "(unmatched value in the name-value list).",
                         fn, "nk:" + std::string(fn) + ":nvMismatch");
        const int hit = matchStringChoice(name, names);
        if (hit < 0)
            detail::fail(std::string(fn) + ": unrecognized parameter "
                         "name '" + name.toString() + "'.",
                         fn, std::string(tag));
        dispatch(static_cast<std::size_t>(hit), in[i + 1]);
    }
}

} // namespace nk::argcheck
