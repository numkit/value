#include <numkit/value/shape_ops.hpp>
#include <numkit/value/value.hpp>
#include <numkit/value/object.hpp> // ObjectState (deepCopy) for object arrays

#include <keel/algo.h>
#include <keel/bits.h>
#include <keel/format.h>
#include <keel/math.h>
#include <keel/memory.h>
#include <keel/number.h>
#include <keel/small_array.h>
#include <keel/string.h>

#include <cmath>
#include <stdexcept>

namespace numkit {

namespace {

inline keel::usize checkedMulSize(keel::usize a, keel::usize b)
{
    keel::usize out = 0;
    if (!keel::checked_mul(a, b, out))
        throw std::runtime_error("Maximum variable size allowed by the program is exceeded");
    return out;
}

inline keel::usize checkedMulDims(const keel::usize *dims, keel::i32 nd)
{
    keel::usize total = 1;
    for (keel::i32 i = 0; i < nd; ++i)
        total = checkedMulSize(total, dims[i]);
    return total;
}

template <typename T>
inline T roundSaturateInt(keel::f64 v) noexcept
{
    return keel::round_sat<T>(v);
}

inline char doubleToCharSafe(keel::f64 v) noexcept
{
    return static_cast<char>(keel::round_sat<keel::u8>(v));
}

} // namespace

const Dims Value::sScalarDims{1, 1};
const Dims Value::sEmptyDims{};

// MATLAB-style empty: 0×0 DOUBLE matrix with a real HeapObject (NOT
// the emptyTag sentinel). isEmpty() == true, isUnset() == false.
// Used as the canonical default for optional `const Value &` arguments
// in toolboxes/ public API.
const Value Value::Empty = Value::matrix(0, 0, ValueType::DOUBLE, nullptr);

Value Value::deleted()
{
    Value v;
    v.heap_ = deletedTag();
    return v;
}
Value::Value()
    : scalar_(0.0)
    , heap_(emptyTag())
{}
Value::~Value() noexcept
{
    releaseHeap();
}
Value::Value(const Value &o)
    : scalar_(o.scalar_)
    , heap_(o.heap_)
{
    if (o.isHeap())
        heap_->addRef();
}
Value &Value::operator=(const Value &o)
{
    if (this != &o) {
        releaseHeap();
        scalar_ = o.scalar_;
        heap_ = o.heap_;
        if (isHeap())
            heap_->addRef();
    }
    return *this;
}
Value::Value(Value &&o) noexcept
    : scalar_(o.scalar_)
    , heap_(o.heap_)
{
    o.heap_ = emptyTag();
    o.scalar_ = 0.0;
}
Value &Value::operator=(Value &&o) noexcept
{
    if (this != &o) {
        releaseHeap();
        scalar_ = o.scalar_;
        heap_ = o.heap_;
        o.heap_ = emptyTag();
        o.scalar_ = 0.0;
    }
    return *this;
}
void Value::swap(Value &o) noexcept
{
    keel::swap(scalar_, o.scalar_);
    keel::swap(heap_, o.heap_);
}
void Value::releaseHeap() noexcept
{
    if (isHeap()) {
        if (heap_->release())
            HeapObject::destroy(heap_);
    }
    heap_ = emptyTag();
}
void Value::detach()
{
    if (!isHeap())
        return;
    // Views (non-owning buffers wrapped via Value::view) must clone
    // unconditionally — refcount alone doesn't capture the "external
    // memory we promised not to write" invariant.
    const bool isView = heap_->buffer && !heap_->buffer->owns();
    if (!isView && heap_->loadRefCount() <= 1)
        return;
    HeapObject *c = heap_->clone();
    releaseHeap();
    heap_ = c;
}
HeapObject *Value::mutableHeap()
{
    detach();
    return heap_;
}

// ── Ergonomic ctors / view ─────────────────────────────────────
// Shared between Value(initializer_list) / Value(Span) / Value(vector)
// / Value(array). Allocates a 1×count row of `type` and memcpy's the
// source bytes in. count=0 returns an empty 1×0 matrix.
Value Value::makeContiguous(const void *src, size_t count,
                            ValueType type, size_t elemSize,
                            keel::Allocator *mr)
{
    Value v = Value::matrix(1, count, type, mr);
    if (count > 0 && src != nullptr) {
        keel::mem_copy(v.rawDataMut(), src, count * elemSize);
    }
    return v;
}

void Value::throwRankMismatch(keel::i32 actualRank, keel::usize requestedRank)
{
    keel::SmallString<128> msg;
    keel::format_into(msg, "Value::asMdRef: rank mismatch (value has ndim={}, requested Rank={})",
                      actualRank, requestedRank);
    throw std::runtime_error(msg.c_str());
}

Value::Value(std::initializer_list<double> il, keel::Allocator *mr)
    : Value()
{
    *this = makeContiguous(il.begin(), il.size(),
                            ValueType::DOUBLE, sizeof(double), mr);
}

Value Value::view(const void *data, ValueType type, Dims dims)
{
    Value v;
    HeapGuard h(HeapObject::create(keel::default_allocator()));
    h->type = type;
    h->dims = keel::move(dims);
    const size_t bytes = h->dims.numel() * elementSize(type);
    h->buffer = DataBuffer::createView(const_cast<void *>(data), bytes);
    v.heap_ = h.release();
    return v;
}

Value Value::scalar(double v, keel::Allocator *)
{
    Value m;
    m.scalar_ = v;
    m.heap_ = nullptr;
    return m;
}
Value Value::logicalScalar(bool v, keel::Allocator *)
{
    Value m;
    m.heap_ = v ? logicalTrueTag() : logicalFalseTag();
    return m;
}
Value Value::matrix(size_t rows, size_t cols, ValueType t, keel::Allocator *mr)
{
    if (rows == 1 && cols == 1 && t == ValueType::DOUBLE)
        return scalar(0.0, mr);
    const size_t total = checkedMulSize(rows, cols);
    const size_t bytes = checkedMulSize(total, elementSize(t));
    Value m;
    HeapGuard h(HeapObject::create(mr));
    h->type = t;
    h->dims = {rows, cols};
    if (bytes > 0) {
        h->buffer = DataBuffer::create(bytes, h->alloc);
        keel::mem_zero(h->buffer->data(), bytes);
    }
    m.heap_ = h.release();
    return m;
}
Value Value::matrix3d(size_t rows, size_t cols, size_t pages, ValueType t, keel::Allocator *mr)
{
    const size_t total = checkedMulSize(checkedMulSize(rows, cols), pages);
    const size_t bytes = checkedMulSize(total, elementSize(t));
    Value m;
    HeapGuard h(HeapObject::create(mr));
    h->type = t;
    h->dims = {rows, cols, pages};
    if (bytes > 0) {
        h->buffer = DataBuffer::create(bytes, h->alloc);
        keel::mem_zero(h->buffer->data(), bytes);
    }
    m.heap_ = h.release();
    return m;
}
Value Value::matrixND(const size_t *dims, int nd, ValueType t, keel::Allocator *mr)
{
    if (nd <= 0) return Value();
    if (nd == 1)
        return matrix(dims[0], 1, t, mr);
    if (nd == 2)
        return matrix(dims[0], dims[1], t, mr);
    if (nd == 3)
        return matrix3d(dims[0], dims[1], dims[2], t, mr);
    // nd >= 4 — go through the ND Dims ctor. Heap allocation in Dims only
    // kicks in for nd > kInlineCap (4); 4D stays fully inline.
    Dims newDims(dims, nd);
    const size_t total = checkedMulDims(dims, nd);
    const size_t bytes = checkedMulSize(total, elementSize(t));
    Value m;
    HeapGuard h(HeapObject::create(mr));
    h->type = t;
    h->dims = keel::move(newDims);
    if (bytes > 0) {
        h->buffer = DataBuffer::create(bytes, h->alloc);
        keel::mem_zero(h->buffer->data(), bytes);
    }
    m.heap_ = h.release();
    return m;
}
Value Value::fromString(keel::StringRef s, keel::Allocator *mr)
{
    Value m;
    HeapGuard h(HeapObject::create(mr));
    h->type = ValueType::CHAR;
    h->dims = {1, s.size()};
    if (!s.empty()) {
        h->buffer = DataBuffer::create(s.size(), h->alloc);
        keel::mem_copy(h->buffer->data(), s.data(), s.size());
    }
    m.heap_ = h.release();
    return m;
}
Value Value::cell(size_t rows, size_t cols, keel::Allocator *mr)
{
    const size_t total = checkedMulSize(rows, cols);
    Value m;
    HeapGuard h(HeapObject::create(mr));
    h->type = ValueType::CELL;
    h->dims = {rows, cols};
    h->cellData = h->allocNew<keel::Array<Value>>(total, h->alloc);
    m.heap_ = h.release();
    return m;
}
Value Value::cell3D(size_t rows, size_t cols, size_t pages, keel::Allocator *mr)
{
    const size_t total = checkedMulSize(checkedMulSize(rows, cols), pages);
    Value m;
    HeapGuard h(HeapObject::create(mr));
    h->type = ValueType::CELL;
    h->dims = {rows, cols, pages};
    h->cellData = h->allocNew<keel::Array<Value>>(total, h->alloc);
    m.heap_ = h.release();
    return m;
}
Value Value::cellND(const size_t *dims, int nd, keel::Allocator *mr)
{
    if (nd <= 0) return cell(0, 0, mr);
    if (nd == 1) return cell(dims[0], 1, mr);
    if (nd == 2) return cell(dims[0], dims[1], mr);
    if (nd == 3) return cell3D(dims[0], dims[1], dims[2], mr);
    Dims newDims(dims, nd);
    const size_t total = checkedMulDims(dims, nd);
    Value m;
    HeapGuard h(HeapObject::create(mr));
    h->type = ValueType::CELL;
    h->dims = keel::move(newDims);
    h->cellData = h->allocNew<keel::Array<Value>>(total, h->alloc);
    m.heap_ = h.release();
    return m;
}
Value Value::csl(size_t n, keel::Allocator *mr)
{
    // A comma-separated list is stored exactly like a 1xn cell (shared cellData
    // backing); only the type tag distinguishes it. Producers fill the slots via
    // cslAt()/cellAt(); splicing contexts read cslCount() + cslAt().
    Value m;
    HeapGuard h(HeapObject::create(mr));
    h->type = ValueType::CSL;
    h->dims = {1, n};
    h->cellData = h->allocNew<keel::Array<Value>>(n, h->alloc);
    m.heap_ = h.release();
    return m;
}
Value collapseCsl(Value v)
{
    if (!v.isCsl())
        return v;  // pass-through: a non-CSL value is already a single value
    size_t n = v.cslCount();
    if (n == 1)
        return keel::move(v.cslAt(0));  // unwrap the sole element
    if (n == 0) {
        throw std::runtime_error(
            "Insufficient number of values: a comma-separated list expanded to 0 "
            "values where 1 is required");
    }
    throw std::runtime_error(
        keel::format_small<128>(
            "Too many values: a comma-separated list expanded to {} values where 1 is required",
            n).c_str());
}

Value Value::stringScalar(keel::StringRef s, keel::Allocator *mr)
{
    Value m;
    HeapGuard h(HeapObject::create(mr));
    h->type = ValueType::STRING;
    h->dims = {1, 1};
    h->cellData = h->allocNew<keel::Array<Value>>(1, Value::fromString(s, h->alloc), h->alloc);
    m.heap_ = h.release();
    return m;
}
Value Value::stringArray(size_t rows, size_t cols, keel::Allocator *mr)
{
    const size_t total = checkedMulSize(rows, cols);
    Value m;
    HeapGuard h(HeapObject::create(mr));
    h->type = ValueType::STRING;
    h->dims = {rows, cols};
    // Initialize with empty strings
    h->cellData = h->allocNew<keel::Array<Value>>(total, Value::fromString("", h->alloc), h->alloc);
    m.heap_ = h.release();
    return m;
}
Value Value::stringArray3D(size_t rows, size_t cols, size_t pages, keel::Allocator *mr)
{
    const size_t total = checkedMulSize(checkedMulSize(rows, cols), pages);
    Value m;
    HeapGuard h(HeapObject::create(mr));
    h->type = ValueType::STRING;
    h->dims = {rows, cols, pages};
    h->cellData = h->allocNew<keel::Array<Value>>(total, Value::fromString("", h->alloc), h->alloc);
    m.heap_ = h.release();
    return m;
}
Value Value::stringArrayND(const size_t *dims, int nd, keel::Allocator *mr)
{
    if (nd <= 0) return stringArray(0, 0, mr);
    if (nd == 1) return stringArray(dims[0], 1, mr);
    if (nd == 2) return stringArray(dims[0], dims[1], mr);
    if (nd == 3) return stringArray3D(dims[0], dims[1], dims[2], mr);
    Dims newDims(dims, nd);
    const size_t total = checkedMulDims(dims, nd);
    Value m;
    HeapGuard h(HeapObject::create(mr));
    h->type = ValueType::STRING;
    h->dims = keel::move(newDims);
    h->cellData = h->allocNew<keel::Array<Value>>(total, Value::fromString("", h->alloc), h->alloc);
    m.heap_ = h.release();
    return m;
}
keel::String Value::stringElem(size_t i) const
{
    if (!isHeap() || heap_->type != ValueType::STRING || !heap_->cellData || i >= heap_->cellData->size())
        return {};
    return (*heap_->cellData)[i].toString();
}
void Value::stringElemSet(size_t i, keel::StringRef s)
{
    if (!isHeap() || heap_->type != ValueType::STRING || !heap_->cellData)
        throw std::runtime_error("Not a string array");
    detach();
    if (i >= heap_->cellData->size())
        heap_->cellData->resize(i + 1, Value::fromString("", heap_->alloc));
    (*heap_->cellData)[i] = Value::fromString(s, heap_->alloc);
}
Value Value::structure(keel::Allocator *mr)
{
    return structArray(1, 1, mr);
}

Value Value::structArray(size_t rows, size_t cols, keel::Allocator *mr)
{
    const size_t n = checkedMulSize(rows, cols);
    Value m;
    HeapGuard h(HeapObject::create(mr));
    h->type = ValueType::STRUCT;
    h->dims = {rows, cols};
    h->structArray = h->allocNew<keel::Array<keel::HashMap<keel::String, Value>>>(h->alloc);
    h->structArray->reserve(n);
    for (size_t i = 0; i < n; ++i)
        h->structArray->push_back(keel::HashMap<keel::String, Value>(h->alloc));
    h->fieldOrder = h->allocNew<keel::Array<keel::String>>(h->alloc);
    m.heap_ = h.release();
    return m;
}
Value Value::funcHandle(keel::StringRef name, keel::Allocator *mr)
{
    Value m;
    HeapGuard h(HeapObject::create(mr));
    h->type = ValueType::FUNC_HANDLE;
    h->dims = {1, 1};
    h->funcName = h->allocNew<keel::String>(name, h->alloc);
    m.heap_ = h.release();
    return m;
}
Value Value::object(keel::StringRef className,
                    keel::Arc<ObjectState> state, bool isHandle,
                    keel::Allocator *mr)
{
    Value m;
    HeapGuard h(HeapObject::create(mr));
    h->type = ValueType::OBJECT;
    h->dims = {1, 1}; // scalar object = 1×1 array of one state
    h->objClass = h->allocNew<keel::String>(className, h->alloc);
    h->objStates.push_back(keel::move(state));
    h->objIsHandle = isHandle;
    m.heap_ = h.release();
    return m;
}
Value Value::objectArray(keel::StringRef className, const Dims &dims,
                         keel::Array<keel::Arc<ObjectState>> states, bool isHandle,
                         keel::Allocator *mr)
{
    Value m;
    HeapGuard h(HeapObject::create(mr));
    h->type = ValueType::OBJECT;
    h->dims = dims;
    h->objClass = h->allocNew<keel::String>(className, h->alloc);
    if (states.allocator() == h->alloc) {
        h->objStates = keel::move(states);
    } else {
        h->objStates.reserve(states.size());
        for (auto &s : states)
            h->objStates.push_back(keel::move(s));
    }
    h->objIsHandle = isHandle;
    m.heap_ = h.release();
    return m;
}
// ============================================================
// Colon range: start:step:stop → row vector
// ============================================================
size_t Value::colonCount(double start, double step, double stop)
{
    if (keel::is_nan(start) || keel::is_nan(step) || keel::is_nan(stop))
        return 0; // MATLAB: NaN in colon produces empty
    if (keel::is_inf(start) || keel::is_inf(stop))
        throw std::runtime_error("Maximum variable size allowed by the program is exceeded");
    if (step == 0.0)
        throw std::runtime_error("Colon step cannot be zero");
    if (keel::is_inf(step))
        return (step > 0 && stop >= start) || (step < 0 && stop <= start) ? 1 : 0;
    if ((step > 0 && stop < start) || (step < 0 && stop > start))
        return 0;
    // FP-noise tolerance: scale-relative epsilon. Avoids dropping the
    // last element when (stop-start)/step lands a few ULP shy of an
    // integer (e.g. 0:0.1:1 gives 11, not 10), without overshooting on
    // clean integer steps (e.g. 1:2:10 gives 5, not 6 — see Value's
    // own gtest and MATLAB parity).
    const double rng = (stop - start) / step;
    const double tol = 3.0 * keel::Limits<double>::kEpsilon
                           * keel::max(keel::abs(start), keel::abs(stop))
                           / keel::abs(step);
    double n = std::floor(rng + tol) + 1;
    if (n < 0)
        n = 0;
    double last = start + (n - 1) * step;
    // Trim only when truly past the stop (beyond tol).
    if (step > 0 && last > stop + tol * keel::abs(step))
        n--;
    if (step < 0 && last < stop - tol * keel::abs(step))
        n--;
    if (n < 0)
        n = 0;
    return static_cast<size_t>(n);
}

Value Value::colonRange(double start, double stop, keel::Allocator *mr)
{
    return colonRange(start, 1.0, stop, mr);
}

Value Value::colonRange(double start, double step, double stop, keel::Allocator *mr)
{
    size_t count = Value::colonCount(start, step, stop);
    auto result = Value::matrix(1, count, ValueType::DOUBLE, mr);
    if (count > 0) {
        double *d = result.doubleDataMut();
        for (size_t i = 0; i < count; ++i)
            d[i] = start + static_cast<double>(i) * step;
        if (count >= 2) {
            double last = start + static_cast<double>(count - 1) * step;
            if ((step > 0 && last > stop) || (step < 0 && last < stop))
                d[count - 1] = stop;
        }
    }
    return result;
}

Value Value::colonRangeTyped(double start, double stop, ValueType t,
                              keel::Allocator *mr)
{
    return colonRangeTyped(start, 1.0, stop, t, mr);
}

Value Value::colonRangeTyped(double start, double step, double stop,
                              ValueType t, keel::Allocator *mr)
{
    if (t == ValueType::DOUBLE)
        return colonRange(start, step, stop, mr);
    size_t count = Value::colonCount(start, step, stop);
    Value result = Value::matrix(1, count, t, mr);
    if (count == 0) return result;
    auto write_loop = [&](auto *dp, auto /*tag*/) {
        using T = keel::RemovePointer<decltype(dp)>;
        for (size_t i = 0; i < count; ++i)
            dp[i] = static_cast<T>(start + static_cast<double>(i) * step);
        if (count >= 2) {
            double last = start + static_cast<double>(count - 1) * step;
            if ((step > 0 && last > stop) || (step < 0 && last < stop))
                dp[count - 1] = static_cast<T>(stop);
        }
    };
    switch (t) {
      case ValueType::CHAR: {
        char *cp = result.charDataMut();
        for (size_t i = 0; i < count; ++i) {
            double v = std::floor(start + static_cast<double>(i) * step);
            if (v < 0.0) v = 0.0;
            else if (v > 255.0) v = 255.0;
            cp[i] = static_cast<char>(static_cast<unsigned char>(v));
        }
        break;
      }
      case ValueType::SINGLE:  write_loop(result.singleDataMut(),  float{});    break;
      case ValueType::LOGICAL: write_loop(result.logicalDataMut(), uint8_t{});  break;
      case ValueType::INT8:    write_loop(result.int8DataMut(),    int8_t{});   break;
      case ValueType::INT16:   write_loop(result.int16DataMut(),   int16_t{});  break;
      case ValueType::INT32:   write_loop(result.int32DataMut(),   int32_t{});  break;
      case ValueType::INT64:   write_loop(result.int64DataMut(),   int64_t{});  break;
      case ValueType::UINT8:   write_loop(result.uint8DataMut(),   uint8_t{});  break;
      case ValueType::UINT16:  write_loop(result.uint16DataMut(),  uint16_t{}); break;
      case ValueType::UINT32:  write_loop(result.uint32DataMut(),  uint32_t{}); break;
      case ValueType::UINT64:  write_loop(result.uint64DataMut(),  uint64_t{}); break;
      default:
        throw std::runtime_error("colonRangeTyped: unsupported output type");
    }
    return result;
}

// ============================================================
// Concat helpers: type promotion and element access
// ============================================================

// Type promotion order for concatenation. An INTEGER operand dominates: MATLAB
// R2025b returns the class of the FIRST integer operand, casting every other
// operand (double / logical / a different integer class / the real part of a
// complex) to it with round-half-away + saturate — it is NOT an error. With no
// integer present: COMPLEX > DOUBLE > LOGICAL. (char / cell / complex have their
// own dedicated concat paths before this is reached; single stays unsupported.)
static ValueType promoteNumericType(const Value *elems, size_t count)
{
    bool hasDouble = false;
    bool hasSingle = false;
    bool hasComplex = false;
    bool hasInt = false;
    ValueType firstInt = ValueType::DOUBLE;
    for (size_t i = 0; i < count; ++i) {
        if (elems[i].isEmpty())
            continue;
        const ValueType t = elems[i].type();
        if (isIntegerType(t)) {
            if (!hasInt) { hasInt = true; firstInt = t; }
            continue;
        }
        switch (t) {
        case ValueType::COMPLEX:
            hasComplex = true;
            break;
        case ValueType::SINGLE:
            hasSingle = true;
            break;
        case ValueType::DOUBLE:
            hasDouble = true;
            break;
        case ValueType::LOGICAL:
            break;
        default:
            throw std::runtime_error(
                keel::format_small<96>("Concatenation not supported for type '{}'",
                                       mtypeName(elems[i].type())).c_str());
        }
    }
    if (hasInt)
        return firstInt;          // integer class dominates (cast the rest)
    if (hasComplex)
        return ValueType::COMPLEX;
    if (hasSingle)
        return ValueType::SINGLE;
    if (hasDouble)
        return ValueType::DOUBLE;
    return ValueType::LOGICAL; // all-logical stays logical
}

// Cast a DOUBLE working buffer to an integer class with MATLAB's conversion
// (round half away from zero, then saturate to the class range). Used to finish
// an integer concatenation that was computed in double precision.
static Value castConcatToInteger(const Value &d, ValueType vt,
                                 keel::Allocator *mr)
{
    const auto &dd = d.dims();
    size_t dBuf[Dims::kMaxRank];
    const int nd = dd.ndim();
    for (int i = 0; i < nd; ++i)
        dBuf[i] = dd.dim(i);
    Value r = Value::matrixND(dBuf, nd, vt, mr);
    const size_t n = d.numel();
    const double *src = d.doubleData();
    auto fill = [&](auto *dst) {
        using T = keel::RemovePointer<keel::RemoveCvRef<decltype(dst)>>;
        for (size_t i = 0; i < n; ++i)
            dst[i] = roundSaturateInt<T>(src[i]);
    };
    switch (vt) {
    case ValueType::INT8:   fill(r.int8DataMut());   break;
    case ValueType::INT16:  fill(r.int16DataMut());  break;
    case ValueType::INT32:  fill(r.int32DataMut());  break;
    case ValueType::INT64:  fill(r.int64DataMut());  break;
    case ValueType::UINT8:  fill(r.uint8DataMut());  break;
    case ValueType::UINT16: fill(r.uint16DataMut()); break;
    case ValueType::UINT32: fill(r.uint32DataMut()); break;
    case ValueType::UINT64: fill(r.uint64DataMut()); break;
    default: break;
    }
    return r;
}

// Read one element as double. Supports every numeric ValueType plus CHAR
// (returns the character's ASCII value) and LOGICAL (0/1). COMPLEX
// contributes the real part only, matching MATLAB's double(complex).
double Value::elemAsDouble(size_t idx) const
{
    switch (type()) {
    case ValueType::DOUBLE:  return doubleData()[idx];
    case ValueType::LOGICAL: return static_cast<double>(logicalData()[idx]);
    case ValueType::COMPLEX: return complexData()[idx].real();
    case ValueType::CHAR:
        return static_cast<double>(static_cast<unsigned char>(charData()[idx]));
    case ValueType::SINGLE:  return static_cast<double>(singleData()[idx]);
    case ValueType::INT8:    return static_cast<double>(int8Data()[idx]);
    case ValueType::INT16:   return static_cast<double>(int16Data()[idx]);
    case ValueType::INT32:   return static_cast<double>(int32Data()[idx]);
    case ValueType::INT64:   return static_cast<double>(int64Data()[idx]);
    case ValueType::UINT8:   return static_cast<double>(uint8Data()[idx]);
    case ValueType::UINT16:  return static_cast<double>(uint16Data()[idx]);
    case ValueType::UINT32:  return static_cast<double>(uint32Data()[idx]);
    case ValueType::UINT64:  return static_cast<double>(uint64Data()[idx]);
    case ValueType::OBJECT: {
        const auto *st = objectStateAt(idx);
        if (st && st->native)
            return st->native->asDouble();
        throw std::runtime_error(
            keel::format_small<96>("Cannot read element as double from type '{}'",
                                   mtypeName(type())).c_str());
    }
    default:
        throw std::runtime_error(
            keel::format_small<96>("Cannot read element as double from type '{}'",
                                   mtypeName(type())).c_str());
    }
}

// Read one element as keel::c64. Real-only types contribute zero
// imaginary part, matching MATLAB's real→complex promotion rule.
static keel::c64 readElemAsComplex(const Value &v, size_t idx)
{
    switch (v.type()) {
    case ValueType::COMPLEX:
        return v.complexData()[idx];
    case ValueType::DOUBLE:
    case ValueType::LOGICAL:
    case ValueType::CHAR:
    case ValueType::SINGLE:
    case ValueType::INT8: case ValueType::INT16: case ValueType::INT32: case ValueType::INT64:
    case ValueType::UINT8: case ValueType::UINT16: case ValueType::UINT32: case ValueType::UINT64:
        return keel::c64(v.elemAsDouble(idx), 0.0);
    default:
        throw std::runtime_error(
            keel::format_small<96>("Cannot read element as complex from type '{}'",
                                   mtypeName(v.type())).c_str());
    }
}

// Read one element as uint8_t logical. Every real numeric type is
// accepted — any non-zero is true. Matches MATLAB so concatenation
// of mixed-type arrays into a logical result behaves uniformly.
static uint8_t readElemAsLogical(const Value &v, size_t idx)
{
    switch (v.type()) {
    case ValueType::LOGICAL:
        return v.logicalData()[idx];
    case ValueType::DOUBLE:
    case ValueType::CHAR:
    case ValueType::SINGLE:
    case ValueType::INT8: case ValueType::INT16: case ValueType::INT32: case ValueType::INT64:
    case ValueType::UINT8: case ValueType::UINT16: case ValueType::UINT32: case ValueType::UINT64:
        return v.elemAsDouble(idx) != 0.0 ? 1 : 0;
    case ValueType::COMPLEX:
        return (v.complexData()[idx] != keel::c64(0.0, 0.0)) ? 1 : 0;
    default:
        throw std::runtime_error(
            keel::format_small<96>("Cannot read element as logical from type '{}'",
                                   mtypeName(v.type())).c_str());
    }
}

// Get dimensions for a concat element: rows, cols, outer slice product (axes >= 2).
// Scalars are treated as 1×1×1.
static void getElemDims(const Value &v, size_t &r, size_t &c, size_t &p)
{
    if (v.isScalar()) {
        r = c = p = 1;
    } else {
        auto &d = v.dims();
        r = d.rows();
        c = d.cols();
        p = 1;
        for (int k = 2; k < d.ndim(); ++k)
            p *= d.dim(k);
    }
}

// Validate and accumulate one dimension (rows or pages) that must match across elements.
// First non-empty element sets the value; subsequent must agree (broadcast 1 allowed).
static void matchDim(size_t &out, size_t elem, bool &set)
{
    if (!set) {
        out = elem;
        set = true;
    } else if (out != elem && elem != 1 && out != 1) {
        throw std::runtime_error(
            "Dimensions of arrays being concatenated are not consistent");
    }
    if (elem > 1)
        out = elem;
}

// Validate and compute higher dimensions (axes >= 2) across non-empty concat operands.
// Returns maxNdim (>= 2) and populates higherDims[2..maxNdim-1].
static int collectHigherDims(const Value *elems, size_t count, size_t (&higherDims)[Dims::kMaxRank])
{
    int maxNd = 2;
    for (size_t i = 0; i < count; ++i)
        if (!elems[i].isEmpty())
            maxNd = keel::max(maxNd, elems[i].dims().ndim());
    if (maxNd > Dims::kMaxRank)
        throw std::runtime_error("Maximum array dimensionality exceeded");
    for (int k = 2; k < maxNd; ++k) {
        size_t dk = 1;
        bool set = false;
        for (size_t i = 0; i < count; ++i) {
            if (elems[i].isEmpty())
                continue;
            const size_t ed = elems[i].isScalar() ? 1 : elems[i].dims().dim(k);
            matchDim(dk, ed, set);
        }
        higherDims[k] = dk;
    }
    return maxNd;
}

// Copy elements from source into a column-major destination buffer.
// Template avoids duplicating the loop structure for double vs keel::c64.
template<typename T, typename ReadFunc>
static void copyBlock(T *dst, size_t dstRows, size_t dstCols,
                      const Value &src, size_t srcRows, size_t srcCols, size_t srcPages,
                      size_t rowOff, size_t colOff, size_t pages,
                      ReadFunc read)
{
    size_t dstSlice = dstRows * dstCols;
    size_t srcSlice = srcRows * srcCols;
    for (size_t p = 0; p < pages; ++p) {
        const size_t sp = (srcPages <= 1) ? 0 : p;
        for (size_t c = 0; c < srcCols; ++c)
            for (size_t r = 0; r < srcRows; ++r)
                dst[(rowOff + r) + (colOff + c) * dstRows + p * dstSlice] =
                    read(src, r + c * srcRows + sp * srcSlice);
    }
}

// ============================================================
// Horizontal concatenation: [a, b, c]
// Concatenates along dimension 2 (columns).
// Rows and pages must match across all elements.
// ============================================================
// Concatenate struct operands (probed R2025b): every non-empty operand
// must be a struct (anything else is a conversion error:
// "Conversion to struct from <class> is not possible"); the field SETS
// must match (order-insensitive — the result keeps the FIRST operand's
// field order; mismatch is the reference's structFieldBad diagnostic);
// empties are neutral. Struct arrays concatenate element-wise with the
// usual dimension rules.
Value Value::concatStructs(const Value *elems, size_t count, bool vertical,
                           keel::Allocator *mr)
{
    if (!mr)
        mr = keel::default_allocator();
    keel::SmallArray<const Value *, 8> ops(mr);
    for (size_t i = 0; i < count; ++i) {
        if (elems[i].isEmpty())
            continue;
        if (!elems[i].isStruct())
            throw std::runtime_error(
                keel::format_small<128>("Conversion to struct from {} is not possible",
                                        mtypeName(elems[i].type())).c_str());
        ops.push_back(&elems[i]);
    }
    if (ops.empty())
        return Value::matrix(0, 0, ValueType::DOUBLE, mr);

    const keel::Array<keel::String> fields0 = ops[0]->fieldNamesInOrder();
    {
        keel::Array<keel::String> sorted0 = fields0.clone(mr);
        keel::sort(sorted0);
        for (size_t k = 1; k < ops.size(); ++k) {
            keel::Array<keel::String> fk = ops[k]->fieldNamesInOrder();
            keel::sort(fk);
            bool same = (fk.size() == sorted0.size());
            if (same) {
                for (size_t j = 0; j < fk.size(); ++j) {
                    if (fk[j] != sorted0[j]) {
                        same = false;
                        break;
                    }
                }
            }
            if (!same)
                throw std::runtime_error(
                    "Names of fields in structure arrays being concatenated "
                    "do not match. Concatenation of structure arrays "
                    "requires that these arrays have the same set of "
                    "fields.");
        }
    }

    auto srcElem = [](const Value &v, size_t flat) -> const keel::HashMap<keel::String, Value> & {
        return v.isStructArray() ? v.structArrayElem(flat) : v.structFields();
    };

    size_t totalA = 0, b = 0;
    bool bSet = false;
    for (const Value *op : ops) {
        const size_t r = op->dims().rows(), c = op->dims().cols();
        totalA += vertical ? r : c;
        if (op->numel() == 0) {
            b = keel::max(b, vertical ? c : r);
        } else {
            const size_t other = vertical ? c : r;
            if (bSet && b != other)
                throw std::runtime_error(
                    "Dimensions of arrays being concatenated are not "
                    "consistent");
            b = other;
            bSet = true;
        }
    }
    // Shape: vertical → (totalA × b); horizontal → (b × totalA).
    const size_t rows = vertical ? totalA : b;
    const size_t cols = vertical ? b : totalA;
    Value out = Value::structArray(rows, cols, mr);
    size_t off = 0;
    for (const Value *op : ops) {
        const size_t r = op->dims().rows(), c = op->dims().cols();
        for (size_t cc = 0; cc < c; ++cc) {
            for (size_t rr = 0; rr < r; ++rr) {
                const size_t src = cc * r + rr;
                const size_t dst = vertical ? cc * rows + (off + rr)
                                            : (off + cc) * rows + rr;
                const auto &m = srcElem(*op, src);
                for (const keel::String &name : fields0) {
                    if (const Value *vp = m.get(name))
                        out.setField(dst, name, *vp);
                }
            }
        }
        off += vertical ? r : c;
    }
    return out;
}

// Concatenate object elements (all the same class) into a 1×N row
// (vertical=false) or N×1 column object array, applying the per-element
// value/handle rule (value classes deep-copy, handle classes alias).
// Throws if objects are mixed with non-objects or different classes.
Value Value::concatObjects(const Value *elems, size_t count, bool vertical,
                           keel::Allocator *mr)
{
    if (!mr)
        mr = keel::default_allocator();
    keel::String clsName(mr);
    bool handle = false;
    const NativePayload *protoNative = nullptr;
    for (size_t i = 0; i < count; ++i) {
        if (!elems[i].isEmpty() && elems[i].isObject()) {
            clsName = elems[i].objectClassName();
            handle = elems[i].objectIsHandle();
            if (const ObjectState *st = elems[i].objectStateConst())
                protoNative = st->native.get();
            break;
        }
    }
    const bool canPromoteNumeric =
        protoNative && (protoNative->fromScalarDouble(0.0, mr) != nullptr);

    keel::Array<Value> promotedStorage(mr);
    promotedStorage.reserve(count);
    keel::SmallArray<const Value *, 8> ops(mr);
    ops.reserve(count);

    for (size_t i = 0; i < count; ++i) {
        if (elems[i].isEmpty())
            continue;
        if (!elems[i].isObject()) {
            if (canPromoteNumeric && (elems[i].isNumeric() || elems[i].isLogical())) {
                const size_t n = elems[i].numel();
                keel::Array<keel::Arc<ObjectState>> states(mr);
                states.reserve(n);
                for (size_t k = 0; k < n; ++k) {
                    auto st = keel::make_arc_in<ObjectState>(mr, mr);
                    st->native = protoNative->fromScalarDouble(elems[i].elemAt(k).toScalar(), mr);
                    states.push_back(keel::move(st));
                }
                promotedStorage.push_back(
                    Value::objectArray(clsName, elems[i].dims(), keel::move(states), handle, mr));
                continue;
            }
            throw std::runtime_error(
                "Cannot concatenate an object with a non-object value");
        }
        const keel::String otherCls = elems[i].objectClassName();
        if (otherCls != clsName) {
            auto isGraphicsCls = [](keel::StringRef c) {
                return c.starts_with("matlab.graphics.") || c.starts_with("matlab.ui.");
            };
            if (isGraphicsCls(clsName) && isGraphicsCls(otherCls)) {
                clsName.assign("matlab.graphics.Graphics");
            } else {
                throw std::runtime_error(
                    "Cannot concatenate objects of different classes");
            }
        }
    }
    HeapGuard h(HeapObject::create(mr));
    h->type = ValueType::OBJECT;
    h->objClass = h->allocNew<keel::String>(clsName, h->alloc);
    h->objIsHandle = handle;
    auto pushState = [&](const keel::Arc<ObjectState> &st) {
        h->objStates.push_back((handle || !st) ? st : st->deepCopy(mr));
    };
    // Gather the non-empty object operands (column-major storage + dims).
    size_t promIdx = 0;
    for (size_t i = 0; i < count; ++i) {
        if (elems[i].isEmpty())
            continue;
        if (elems[i].isObject())
            ops.push_back(&elems[i]);
        else
            ops.push_back(&promotedStorage[promIdx++]);
    }

    if (!vertical) {
        // horzcat: operands share #rows; result is R×(Σcols). Column-major
        // layout makes this a straight append of each operand's states.
        size_t rows = ops.empty() ? 0 : ops[0]->heap_->dims.rows();
        size_t cols = 0;
        for (const Value *o : ops) {
            if (o->heap_->dims.rows() != rows)
                throw std::runtime_error(
                    "Dimensions of arrays being concatenated are not consistent");
            cols += o->heap_->dims.cols();
            for (const auto &st : o->heap_->objStates)
                pushState(st);
        }
        h->dims = Dims(rows, cols);
    } else {
        // vertcat: operands share #cols; result is (Σrows)×C. Column-major
        // requires interleaving each operand's j-th column in turn.
        size_t cols = ops.empty() ? 0 : ops[0]->heap_->dims.cols();
        size_t rows = 0;
        for (const Value *o : ops) {
            if (o->heap_->dims.cols() != cols)
                throw std::runtime_error(
                    "Dimensions of arrays being concatenated are not consistent");
            rows += o->heap_->dims.rows();
        }
        for (size_t j = 0; j < cols; ++j)
            for (const Value *o : ops) {
                size_t orow = o->heap_->dims.rows();
                for (size_t r = 0; r < orow; ++r)
                    pushState(o->heap_->objStates[j * orow + r]);
            }
        h->dims = Dims(rows, cols);
    }
    Value out;
    out.heap_ = h.release();
    return out;
}

Value Value::horzcat(const Value *elems, size_t count, keel::Allocator *mr)
{
    // OBJECT concatenation: [a b] of same-class objects → a 1×N row array.
    for (size_t i = 0; i < count; ++i)
        if (elems[i].isObject())
            return concatObjects(elems, count, /*vertical=*/false, mr);

    // STRUCT concatenation: same field set → struct array (probed R2025b).
    for (size_t i = 0; i < count; ++i)
        if (elems[i].isStruct())
            return concatStructs(elems, count, /*vertical=*/false, mr);

    // String array horzcat: ["a", "b", "c"] → 1×3 string
    bool hasString = false;
    for (size_t i = 0; i < count; ++i)
        if (!elems[i].isEmpty() && elems[i].type() == ValueType::STRING) {
            hasString = true;
            break;
        }
    if (hasString) {
        size_t totalCols = 0;
        for (size_t i = 0; i < count; ++i) {
            if (elems[i].isEmpty())
                continue;
            if (elems[i].isString())
                totalCols += elems[i].dims().cols();
            else
                totalCols += 1; // scalar char or other → one string element
        }
        auto result = Value::stringArray(1, totalCols);
        size_t pos = 0;
        for (size_t i = 0; i < count; ++i) {
            if (elems[i].isEmpty())
                continue;
            if (elems[i].isString()) {
                for (size_t j = 0; j < elems[i].numel(); ++j)
                    result.stringElemSet(pos++, elems[i].stringElem(j));
            } else {
                result.stringElemSet(pos++, elems[i].toString());
            }
        }
        return result;
    }

    // Char concatenation path
    bool hasChar = false;
    for (size_t i = 0; i < count; ++i)
        if (!elems[i].isEmpty() && elems[i].type() == ValueType::CHAR) {
            hasChar = true;
            break;
        }

    if (hasChar) {
        size_t rows = 0, totalCols = 0;
        bool rowsSet = false, multiRow = false, sameRows = true;
        for (size_t i = 0; i < count; ++i) {
            if (elems[i].isEmpty()) continue;
            size_t eR = elems[i].dims().rows();
            size_t eC = elems[i].dims().cols();
            if (eR > 1) multiRow = true;
            if (!rowsSet) { rows = eR; rowsSet = true; }
            else if (rows != eR) sameRows = false;
            totalCols += eC;
        }
        keel::String result(mr);
        for (size_t i = 0; i < count; ++i) {
            if (elems[i].isEmpty())
                continue;
            if (elems[i].type() == ValueType::CHAR) {
                result.append(keel::StringRef(elems[i].charData(), elems[i].numel()));
            } else if (elems[i].isScalar()) {
                result.push_back(static_cast<char>(keel::round_i32(elems[i].toScalar())));
            } else {
                // Non-scalar double/logical: convert each element to char
                const double *d = elems[i].doubleData();
                for (size_t k = 0; k < elems[i].numel(); ++k)
                    result.push_back(static_cast<char>(keel::round_i32(d[k])));
            }
        }
        if (multiRow && sameRows && rows > 1 && rows * totalCols == result.size()) {
            Value out = Value::matrix(rows, totalCols, ValueType::CHAR, mr);
            keel::mem_copy(out.charDataMut(), result.data(), result.size());
            return out;
        }
        return Value::fromString(result, mr);
    }

    // Cell horzcat: [c1, c2] concatenates cell contents side by side
    // (MATLAB semantics; was a "Concatenation not supported" error).
    bool hasCellH = false;
    for (size_t i = 0; i < count; ++i)
        if (!elems[i].isEmpty() && elems[i].type() == ValueType::CELL) {
            hasCellH = true;
            break;
        }
    if (hasCellH) {
        size_t totalCols = 0, rows = 0;
        bool rowsSet = false;
        for (size_t i = 0; i < count; ++i) {
            size_t eR, eC, eP;
            getElemDims(elems[i], eR, eC, eP);
            totalCols += eC;
            if (elems[i].isEmpty()) {
                rows = keel::max(rows, eR);
            } else {
                matchDim(rows, eR, rowsSet);
            }
        }
        auto result = Value::cell(rows, totalCols, mr);
        size_t colOff = 0;
        for (size_t i = 0; i < count; ++i) {
            if (elems[i].isEmpty() || elems[i].type() != ValueType::CELL)
                continue;
            size_t eR = elems[i].dims().rows();
            size_t eC = elems[i].dims().cols();
            for (size_t c = 0; c < eC; ++c)
                for (size_t r = 0; r < eR; ++r)
                    result.cellAt((colOff + c) * rows + r) =
                        elems[i].cellAt(c * eR + r);
            colOff += eC;
        }
        return result;
    }

    // Collect dimensions
    size_t totalCols = 0, emptyCols = 0, rows = 0, pages = 1;
    bool rowsSet = false, pagesSet = false;

    for (size_t i = 0; i < count; ++i) {
        size_t eR, eC, eP;
        getElemDims(elems[i], eR, eC, eP);
        if (elems[i].isEmpty()) {
            // Mirror of the vertcat empty rule (probed R2025b): empties
            // add 0 columns and participate with MAX height —
            // [zeros(0,2), zeros(0,2)] is 0x4, [zeros(3,0), 5] is 3x1.
            emptyCols += eC;
            if (!rowsSet)
                rows = keel::max(rows, eR);
            pages = keel::max(pages, eP);
        } else {
            totalCols += eC;
            matchDim(rows, eR, rowsSet);
            matchDim(pages, eP, pagesSet);
        }
    }
    if (!rowsSet)
        totalCols += emptyCols;
    if (totalCols == 0) {
        // All-empty concat returns a PROPER empty (never an unset Value —
        // the old fallthrough left the assignment target unbound).
        return Value::matrix(rows, 0, ValueType::DOUBLE, mr);
    }
    // rows can only still be 0 for an ALL-empty concat with columns
    // (e.g. [zeros(0,1), zeros(0,1)] -> 0x2): MATLAB keeps the 0 height.

    size_t outDimsBuf[Dims::kMaxRank] = {0};
    const int outNd = collectHigherDims(elems, count, outDimsBuf);
    outDimsBuf[0] = rows;
    outDimsBuf[1] = totalCols;
    pages = 1;
    for (int k = 2; k < outNd; ++k)
        pages = checkedMulSize(pages, outDimsBuf[k]);

    ValueType outType = promoteNumericType(elems, count);
    // Integer output is computed in DOUBLE then cast (round+saturate) at the
    // end — the elemAsDouble copy path already reads every operand class.
    const bool intOut = isIntegerType(outType);
    const ValueType workType = intOut ? ValueType::DOUBLE : outType;
    bool anySingle = false;
    for (size_t i = 0; i < count; ++i)
        if (!elems[i].isEmpty() && (elems[i].type() == ValueType::SINGLE || elems[i].isSingleComplex()))
            anySingle = true;

    auto result = (outNd > 3)  ? Value::matrixND(outDimsBuf, outNd, workType, mr)
                : (pages > 1) ? Value::matrix3d(rows, totalCols, pages, workType, mr)
                              : Value::matrix(rows, totalCols, workType, mr);
    if (workType == ValueType::COMPLEX && anySingle)
        result.setSingleComplex(true);

    size_t colOff = 0;
    for (size_t i = 0; i < count; ++i) {
        if (elems[i].isEmpty())
            continue;
        size_t eR, eC, eP;
        getElemDims(elems[i], eR, eC, eP);

        if (workType == ValueType::COMPLEX)
            copyBlock(result.complexDataMut(), rows, totalCols,
                      elems[i], eR, eC, eP, 0, colOff, pages, readElemAsComplex);
        else if (workType == ValueType::SINGLE)
            copyBlock(result.singleDataMut(), rows, totalCols,
                      elems[i], eR, eC, eP, 0, colOff, pages, [](const Value &v, size_t i) { return static_cast<float>(v.elemAsDouble(i)); });
        else if (workType == ValueType::LOGICAL)
            copyBlock(result.logicalDataMut(), rows, totalCols,
                      elems[i], eR, eC, eP, 0, colOff, pages, readElemAsLogical);
        else
            copyBlock(result.doubleDataMut(), rows, totalCols,
                      elems[i], eR, eC, eP, 0, colOff, pages, [](const Value &v, size_t i) { return v.elemAsDouble(i); });
        colOff += eC;
    }
    return intOut ? castConcatToInteger(result, outType, mr) : result;
}

// ============================================================
// Vertical concatenation: [a; b; c]
// Concatenates along dimension 1 (rows).
// Columns and pages must match across all elements.
// ============================================================
Value Value::vertcat(const Value *elems, size_t count, keel::Allocator *mr)
{
    // OBJECT concatenation: [a; b] of same-class objects → an N×1 column.
    for (size_t i = 0; i < count; ++i)
        if (elems[i].isObject())
            return concatObjects(elems, count, /*vertical=*/true, mr);

    // STRUCT concatenation: same field set → struct array (probed R2025b).
    for (size_t i = 0; i < count; ++i)
        if (elems[i].isStruct())
            return concatStructs(elems, count, /*vertical=*/true, mr);

    size_t totalRows = 0, emptyRows = 0, cols = 0, pages = 1;
    bool colsSet = false, pagesSet = false;
    bool hasCell = false;

    for (size_t i = 0; i < count; ++i) {
        if (elems[i].isCell())
            hasCell = true;
        size_t eR, eC, eP;
        getElemDims(elems[i], eR, eC, eP);
        if (elems[i].isEmpty()) {
            // MATLAB (probed R2025b): empties contribute 0 rows and
            // participate with MAX width — [zeros(0,1); zeros(0,3)] is a
            // legal 0x3, [[ ]; [ ]] is 0x0. They used to be skipped
            // entirely, so an all-empty concat fell through to an UNSET
            // Value() that never bound its assignment target
            // (bugs/closed/lang/vertcat-all-empty).
            emptyRows += eR;
            if (!colsSet)
                cols = keel::max(cols, eC);
            pages = keel::max(pages, eP);
        } else {
            totalRows += eR;
            matchDim(cols, eC, colsSet);
            matchDim(pages, eP, pagesSet);
        }
    }
    if (!colsSet)
        totalRows += emptyRows;
    if (totalRows == 0) {
        // All-empty concat returns a PROPER empty, never an unset Value.
        if (hasCell)
            return Value::matrix(0, cols, ValueType::CELL, mr);
        return Value::matrix(0, cols, ValueType::DOUBLE, mr);
    }

    // Char vertcat
    bool hasChar = false;
    for (size_t i = 0; i < count; ++i)
        if (!elems[i].isEmpty() && elems[i].type() == ValueType::CHAR) {
            hasChar = true;
            break;
        }
    if (hasChar) {
        auto result = Value::matrix(totalRows, cols, ValueType::CHAR, mr);
        char *dst = result.charDataMut();
        size_t rowOff = 0;
        for (size_t i = 0; i < count; ++i) {
            if (elems[i].isEmpty())
                continue;
            size_t eR, eC, eP;
            getElemDims(elems[i], eR, eC, eP);
            if (elems[i].type() == ValueType::CHAR) {
                const char *src = elems[i].charData();
                for (size_t c = 0; c < eC; ++c)
                    for (size_t r = 0; r < eR; ++r)
                        dst[c * totalRows + rowOff + r] = src[c * eR + r];
            } else {
                // Double/logical to char conversion
                for (size_t c = 0; c < eC; ++c)
                    for (size_t r = 0; r < eR; ++r)
                        dst[c * totalRows + rowOff + r] =
                            static_cast<char>(keel::round_i32(
                                elems[i].elemAsDouble(c * eR + r)));
            }
            rowOff += eR;
        }
        return result;
    }

    // Cell vertcat: combine rows into a 2D cell (column-major)
    if (hasCell) {
        auto result = Value::cell(totalRows, cols, mr);
        size_t rowOff = 0;
        for (size_t i = 0; i < count; ++i) {
            if (elems[i].isEmpty())
                continue;
            size_t eR = elems[i].dims().rows();
            size_t eC = elems[i].dims().cols();
            for (size_t c = 0; c < eC; ++c)
                for (size_t r = 0; r < eR; ++r)
                    result.cellAt((c) * totalRows + (rowOff + r)) =
                        elems[i].cellAt(c * eR + r);
            rowOff += eR;
        }
        return result;
    }

    // String-array vertcat: 2-D literal `["a","b"; "c","d"]` lands here.
    // Each row arg is a 1×N (or M×N) string; stack into a (sum-rows)×N
    // string array. Char operands auto-promote to single-element strings.
    // See BUGS.md #8.
    bool hasString = false;
    for (size_t i = 0; i < count; ++i)
        if (!elems[i].isEmpty() && elems[i].type() == ValueType::STRING) {
            hasString = true;
            break;
        }
    if (hasString) {
        auto result = Value::stringArray(totalRows, cols, mr);
        size_t rowOff = 0;
        for (size_t i = 0; i < count; ++i) {
            if (elems[i].isEmpty())
                continue;
            size_t eR, eC, eP;
            getElemDims(elems[i], eR, eC, eP);
            if (elems[i].type() == ValueType::STRING) {
                for (size_t c = 0; c < eC; ++c)
                    for (size_t r = 0; r < eR; ++r)
                        result.stringElemSet((c) * totalRows + (rowOff + r),
                                              elems[i].stringElem(c * eR + r));
            } else if (elems[i].isChar() || elems[i].isScalar()) {
                // single char / numeric scalar → broadcast 1 string per cell.
                const keel::String s = elems[i].toString();
                for (size_t c = 0; c < eC; ++c)
                    for (size_t r = 0; r < eR; ++r)
                        result.stringElemSet((c) * totalRows + (rowOff + r), s);
            } else {
                throw std::runtime_error(
                    "String concatenation: row element must be string, char, or scalar");
            }
            rowOff += eR;
        }
        return result;
    }

    size_t outDimsBuf[Dims::kMaxRank] = {0};
    const int outNd = collectHigherDims(elems, count, outDimsBuf);
    outDimsBuf[0] = totalRows;
    outDimsBuf[1] = cols;
    pages = 1;
    for (int k = 2; k < outNd; ++k)
        pages = checkedMulSize(pages, outDimsBuf[k]);

    ValueType outType = promoteNumericType(elems, count);
    // Integer output computed in DOUBLE then cast (round+saturate) — see horzcat.
    const bool intOut = isIntegerType(outType);
    const ValueType workType = intOut ? ValueType::DOUBLE : outType;
    bool anySingle = false;
    for (size_t i = 0; i < count; ++i)
        if (!elems[i].isEmpty() && (elems[i].type() == ValueType::SINGLE || elems[i].isSingleComplex()))
            anySingle = true;

    auto result = (outNd > 3)  ? Value::matrixND(outDimsBuf, outNd, workType, mr)
                : (pages > 1) ? Value::matrix3d(totalRows, cols, pages, workType, mr)
                              : Value::matrix(totalRows, cols, workType, mr);
    if (workType == ValueType::COMPLEX && anySingle)
        result.setSingleComplex(true);

    size_t rowOff = 0;
    for (size_t i = 0; i < count; ++i) {
        if (elems[i].isEmpty())
            continue;
        size_t eR, eC, eP;
        getElemDims(elems[i], eR, eC, eP);

        if (workType == ValueType::COMPLEX)
            copyBlock(result.complexDataMut(), totalRows, cols,
                      elems[i], eR, eC, eP, rowOff, 0, pages, readElemAsComplex);
        else if (workType == ValueType::SINGLE)
            copyBlock(result.singleDataMut(), totalRows, cols,
                      elems[i], eR, eC, eP, rowOff, 0, pages, [](const Value &v, size_t i) { return static_cast<float>(v.elemAsDouble(i)); });
        else if (workType == ValueType::LOGICAL)
            copyBlock(result.logicalDataMut(), totalRows, cols,
                      elems[i], eR, eC, eP, rowOff, 0, pages, readElemAsLogical);
        else
            copyBlock(result.doubleDataMut(), totalRows, cols,
                      elems[i], eR, eC, eP, rowOff, 0, pages, [](const Value &v, size_t i) { return v.elemAsDouble(i); });
        rowOff += eR;
    }
    return intOut ? castConcatToInteger(result, outType, mr) : result;
}

// ============================================================
// Type-preserving indexing
// ============================================================

// Helper: create a scalar Value of the same type as this array at linear index.
Value Value::elemAt(size_t idx, keel::Allocator *mr) const
{
    ValueType t = type();
    switch (t) {
    case ValueType::DOUBLE:
        return Value::scalar(doubleData()[idx], mr);
    case ValueType::COMPLEX: {
        // A complex element whose imaginary part is zero narrows to real (MATLAB
        // isreal(z(k)) per element). Covers VM/TW scalar-index fast paths too.
        Value s = Value::complexScalar(complexData()[idx], mr);
        s.setSingleComplex(isSingleComplex());
        return narrowComplex(keel::move(s), mr);
    }
    case ValueType::LOGICAL:
        return Value::logicalScalar(logicalData()[idx] != 0, mr);
    case ValueType::CHAR: {
        const char ch = charData()[idx];
        return Value::fromString(keel::StringRef(&ch, 1), mr);
    }
    case ValueType::CELL:
        return cellAt(idx);
    case ValueType::OBJECT: {
        // Object-array element → a 1×1 scalar object (value/handle rule).
        size_t s = idx;
        return objectGather(&s, Dims(1, 1), mr);
    }
    case ValueType::STRING:
        // String-array element access — return a 1×1 string scalar
        // holding the string at the linear index. See BUGS.md #7.
        return Value::stringScalar(stringElem(idx), mr);
    case ValueType::STRUCT: {
        // s(i) on a struct → a 1×1 struct holding the i-th element's
        // fields. With AoS storage there's no fast-path: even for a
        // size-1 array we copy element[0]'s fields into a new struct so
        // mutation paths (`s(1).x = ...`) operate on a fresh map.
        Value out = Value::structure(mr);
        if (heap_->fieldOrder && out.heap_->fieldOrder)
            *out.heap_->fieldOrder = heap_->fieldOrder->clone(out.heap_->alloc);
        (*out.heap_->structArray)[0] = (*heap_->structArray)[idx].clone(out.heap_->alloc);
        return out;
    }
    // Typed integer / single return a 1×1 array of the same type so the
    // user's type sticks through indexing. Raw memcpy from the source
    // buffer keeps the representation byte-exact.
    case ValueType::INT8: case ValueType::INT16: case ValueType::INT32: case ValueType::INT64:
    case ValueType::UINT8: case ValueType::UINT16: case ValueType::UINT32: case ValueType::UINT64:
    case ValueType::SINGLE: {
        size_t es = elementSize(t);
        Value r = Value::matrix(1, 1, t, mr);
        keel::mem_copy(r.rawDataMut(),
                       static_cast<const char *>(rawData()) + idx * es, es);
        return r;
    }
    default:
        throw std::runtime_error(
            keel::format_small<96>("elemAt not supported for type '{}'", mtypeName(t)).c_str());
    }
}

// 1D slice: extract elements at given linear indices.
Value Value::reshape(size_t rows, size_t cols, keel::Allocator *mr) const
{
    if (rows * cols != numel())
        throw std::runtime_error("reshape: number of elements must not change");
    if (numel() == 1)
        return *this;  // 1x1 reshape is a no-op (matrix(1,1,DOUBLE) is inline, no buffer)
    if (isObject())
        return objectReshape(Dims(rows, cols), mr);
    if (type() == ValueType::STRUCT) {
        Value r = Value::structArray(rows, cols, mr);
        if (heap_->fieldOrder && r.heap_->fieldOrder)
            *r.heap_->fieldOrder = heap_->fieldOrder->clone(r.heap_->alloc);
        const auto &src = *heap_->structArray;
        auto &dst = *r.heap_->structArray;
        for (size_t i = 0; i < src.size() && i < dst.size(); ++i)
            dst[i] = src[i].clone(r.heap_->alloc);
        return r;
    }
    if (type() == ValueType::CELL || type() == ValueType::STRING) {
        Value r = (type() == ValueType::CELL) ? Value::cell(rows, cols, mr)
                                              : Value::stringArray(rows, cols, mr);
        const auto &src = cellDataVec();
        auto &dst = r.cellDataVec();
        for (size_t i = 0; i < src.size() && i < dst.size(); ++i)
            dst[i] = src[i];
        return r;
    }
    Value r = Value::matrix(rows, cols, type(), mr);  // numel>1 -> heap-backed
    r.setSingleComplex(isSingleComplex());
    if (rawBytes() > 0)
        keel::mem_copy(r.rawDataMut(), rawData(), rawBytes());
    return r;
}

// MATLAB narrows a complex result with an all-zero imaginary part back to a real
// double (declared in value.hpp). No-op unless COMPLEX with every imaginary
// component exactly zero (a NaN imaginary part keeps it complex).
Value narrowComplex(Value v, keel::Allocator *mr)
{
    if (v.type() != ValueType::COMPLEX) return v;
    const keel::c64 *d = v.complexData();
    const size_t n = v.numel();
    for (size_t i = 0; i < n; ++i)
        if (d[i].imag() != 0.0) return v;          // genuinely complex (NaN imag too)
    const Dims &dm = v.dims();
    const int nd = dm.ndim();
    size_t dimsBuf[Dims::kMaxRank];
    for (int i = 0; i < nd; ++i) dimsBuf[i] = dm.dim(i);
    if (v.isSingleComplex()) {
        Value r = Value::matrixND(dimsBuf, nd, ValueType::SINGLE, mr);
        float *o = r.singleDataMut();
        for (size_t i = 0; i < n; ++i) o[i] = static_cast<float>(d[i].real());
        return r;
    }
    Value r = Value::matrixND(dimsBuf, nd, ValueType::DOUBLE, mr);
    double *o = r.doubleDataMut();
    for (size_t i = 0; i < n; ++i) o[i] = d[i].real();
    return r;
}

// Shape rule: column vector source → column result; otherwise → row result.
// CELL: always returns sub-cell (even for count==1), matching MATLAB c(i) semantics.
Value Value::indexGet(const size_t *indices, size_t count, keel::Allocator *mr) const
{
    // For non-CELL scalar result, return a scalar value (narrow a complex
    // element whose imaginary part is zero — MATLAB: isreal(z(k)) per element).
    if (count == 1 && type() != ValueType::CELL)
        return narrowComplex(elemAt(indices[0], mr), mr);

    // Shape: column vector source → column result
    bool colResult = (dims().cols() == 1 && dims().rows() > 1);
    size_t rr = colResult ? count : 1;
    size_t cc = colResult ? 1 : count;

    ValueType t = type();
    switch (t) {
    case ValueType::DOUBLE: {
        auto result = Value::matrix(rr, cc, ValueType::DOUBLE, mr);
        double *dst = result.doubleDataMut();
        const double *src = doubleData();
        for (size_t i = 0; i < count; ++i)
            dst[i] = src[indices[i]];
        return result;
    }
    case ValueType::COMPLEX: {
        auto result = Value::complexMatrix(rr, cc, mr);
        result.setSingleComplex(isSingleComplex());
        keel::c64 *dst = result.complexDataMut();
        const keel::c64 *src = complexData();
        for (size_t i = 0; i < count; ++i)
            dst[i] = src[indices[i]];
        return narrowComplex(keel::move(result), mr);   // all-real slice -> real
    }
    case ValueType::LOGICAL: {
        auto result = Value::matrix(rr, cc, ValueType::LOGICAL, mr);
        uint8_t *dst = result.logicalDataMut();
        const uint8_t *src = logicalData();
        for (size_t i = 0; i < count; ++i)
            dst[i] = src[indices[i]];
        return result;
    }
    case ValueType::CHAR: {
        auto result = Value::matrix(1, count, ValueType::CHAR, mr);
        if (count > 0) {
            char *dst = result.charDataMut();
            const char *src = charData();
            for (size_t i = 0; i < count; ++i)
                dst[i] = src[indices[i]];
        }
        return result;
    }
    case ValueType::CELL: {
        auto result = Value::cell(rr, cc, mr);
        for (size_t i = 0; i < count; ++i)
            result.cellAt(i) = cellAt(indices[i]);
        return result;
    }
    case ValueType::OBJECT:
        // Object-array slice: gather selected states into the result shape.
        return objectGather(indices, Dims(rr, cc), mr);
    case ValueType::STRING: {
        // String-array slice: copy strings at the requested indices
        // into a fresh string array of the result shape. See BUGS.md #7.
        auto result = Value::stringArray(rr, cc, mr);
        for (size_t i = 0; i < count; ++i)
            result.stringElemSet(i, stringElem(indices[i]));
        return result;
    }
    case ValueType::STRUCT: {
        auto result = Value::structArray(rr, cc, mr);
        if (result.heap_->fieldOrder) {
            if (heap_->fieldOrder && !heap_->fieldOrder->empty()) {
                *result.heap_->fieldOrder = heap_->fieldOrder->clone(result.heap_->alloc);
            } else {
                auto fnms = fieldNamesInOrder();
                *result.heap_->fieldOrder = fnms.clone(result.heap_->alloc);
            }
        }
        for (size_t i = 0; i < count; ++i) {
            const auto &src = (*heap_->structArray)[indices[i]];
            (*result.heap_->structArray)[i] = src.clone(result.heap_->alloc);
        }
        return result;
    }
    // Typed integer / single: raw memcpy per element, same target type.
    case ValueType::INT8: case ValueType::INT16: case ValueType::INT32: case ValueType::INT64:
    case ValueType::UINT8: case ValueType::UINT16: case ValueType::UINT32: case ValueType::UINT64:
    case ValueType::SINGLE: {
        size_t es = elementSize(t);
        auto result = Value::matrix(rr, cc, t, mr);
        const char *src = static_cast<const char *>(rawData());
        char *dst = static_cast<char *>(result.rawDataMut());
        for (size_t i = 0; i < count; ++i)
            keel::mem_copy(dst + i * es, src + indices[i] * es, es);
        return result;
    }
    default:
        throw std::runtime_error(
            keel::format_small<96>("indexGet not supported for type '{}'", mtypeName(t)).c_str());
    }
}

static inline size_t subIdx(const size_t *idx, size_t k) noexcept
{
    return idx ? idx[k] : k;
}

// 2D slice: extract sub-matrix at given row/col indices (nullptr = full 0..n-1).
Value Value::indexGet2D(const size_t *rowIdx, size_t nrows,
                          const size_t *colIdx, size_t ncols,
                          keel::Allocator *mr) const
{
    // Scalar shortcut for non-CELL types
    if (nrows == 1 && ncols == 1 && type() != ValueType::CELL) {
        size_t idx = dims().sub2ind(subIdx(rowIdx, 0), subIdx(colIdx, 0));
        return narrowComplex(elemAt(idx, mr), mr);
    }

    ValueType t = type();

    if (t == ValueType::CELL) {
        auto &d = dims();
        auto result = Value::cell(nrows, ncols, mr);
        for (size_t c = 0; c < ncols; ++c)
            for (size_t r = 0; r < nrows; ++r)
                result.cellAt(c * nrows + r) = cellAt(d.sub2ind(subIdx(rowIdx, r), subIdx(colIdx, c)));
        return result;
    }
    if (t == ValueType::STRING) {
        // String 2-D slice: copy strings element-by-element. STRING
        // elements live in cellData, not a contiguous byte buffer, so the
        // memcpy fast path doesn't apply. See BUGS.md #7.
        auto &d = dims();
        auto result = Value::stringArray(nrows, ncols, mr);
        for (size_t c = 0; c < ncols; ++c)
            for (size_t r = 0; r < nrows; ++r)
                result.stringElemSet(c * nrows + r,
                                      stringElem(d.sub2ind(subIdx(rowIdx, r), subIdx(colIdx, c))));
        return result;
    }
    if (t == ValueType::STRUCT) {
        auto &d = dims();
        auto result = Value::structArray(nrows, ncols, mr);
        if (result.heap_->fieldOrder) {
            if (heap_->fieldOrder && !heap_->fieldOrder->empty()) {
                *result.heap_->fieldOrder = heap_->fieldOrder->clone(result.heap_->alloc);
            } else {
                auto fnms = fieldNamesInOrder();
                *result.heap_->fieldOrder = fnms.clone(result.heap_->alloc);
            }
        }
        for (size_t c = 0; c < ncols; ++c) {
            for (size_t r = 0; r < nrows; ++r) {
                const auto &src = (*heap_->structArray)[d.sub2ind(subIdx(rowIdx, r), subIdx(colIdx, c))];
                (*result.heap_->structArray)[c * nrows + r] = src.clone(result.heap_->alloc);
            }
        }
        return result;
    }
    if (t == ValueType::OBJECT) {
        auto &d = dims();
        keel::SmallArray<size_t, 32> src(nrows * ncols, mr ? mr : keel::default_allocator());
        for (size_t c = 0; c < ncols; ++c)
            for (size_t r = 0; r < nrows; ++r)
                src[c * nrows + r] = d.sub2ind(subIdx(rowIdx, r), subIdx(colIdx, c));
        return objectGather(src.data(), Dims(nrows, ncols), mr);
    }

    size_t es = elementSize(t);
    if (es == 0)
        throw std::runtime_error(
            keel::format_small<96>("indexGet2D not supported for type '{}'", mtypeName(t)).c_str());

    auto &d = dims();
    auto result = Value::matrix(nrows, ncols, t, mr);
    result.setSingleComplex(isSingleComplex());
    if (nrows == 0 || ncols == 0)
        return result;

    const char *src = static_cast<const char *>(rawData());
    char *dst = static_cast<char *>(result.rawDataMut());
    const size_t srcRows = d.rows();

    // Contiguous full-matrix or column-slice fast path when all rows are selected
    if (rowIdx == nullptr && nrows == srcRows) {
        if (colIdx == nullptr && ncols == (srcRows > 0 ? d.numel() / srcRows : d.cols())) {
            keel::mem_copy(dst, src, nrows * ncols * es);
        } else {
            const size_t colBytes = nrows * es;
            for (size_t c = 0; c < ncols; ++c) {
                const size_t sc = subIdx(colIdx, c);
                keel::mem_copy(dst + c * colBytes, src + d.sub2ind(0, sc) * es, colBytes);
            }
        }
        return narrowComplex(keel::move(result), mr);
    }

    // Typed double fast path (avoids per-element memcpy call overhead)
    if (t == ValueType::DOUBLE) {
        const double *s = doubleData();
        double *out = result.doubleDataMut();
        if (nrows == 1 && colIdx == nullptr) {
            const size_t r0 = subIdx(rowIdx, 0);
            const double *sr = s + r0;
            for (size_t c = 0; c < ncols; ++c)
                out[c] = sr[c * srcRows];
        } else {
            for (size_t c = 0; c < ncols; ++c) {
                const double *sc = s + subIdx(colIdx, c) * srcRows;
                double *dc = out + c * nrows;
                for (size_t r = 0; r < nrows; ++r)
                    dc[r] = sc[subIdx(rowIdx, r)];
            }
        }
        return result;
    }
    if (t == ValueType::COMPLEX) {
        const keel::c64 *s = complexData();
        keel::c64 *out = result.complexDataMut();
        bool anyImag = false;
        for (size_t c = 0; c < ncols; ++c) {
            const keel::c64 *sc = s + subIdx(colIdx, c) * srcRows;
            keel::c64 *dc = out + c * nrows;
            for (size_t r = 0; r < nrows; ++r) {
                keel::c64 v = sc[subIdx(rowIdx, r)];
                dc[r] = v;
                if (v.imag() != 0.0)
                    anyImag = true;
            }
        }
        return anyImag ? result : narrowComplex(keel::move(result), mr);
    }

    for (size_t c = 0; c < ncols; ++c) {
        const size_t sc = subIdx(colIdx, c);
        for (size_t r = 0; r < nrows; ++r)
            keel::mem_copy(dst + (c * nrows + r) * es,
                           src + d.sub2ind(subIdx(rowIdx, r), sc) * es, es);
    }
    return narrowComplex(keel::move(result), mr);   // all-real complex slice -> real
}

// 3D slice: extract sub-array at given row/col/page indices (nullptr = full 0..n-1).
Value Value::indexGet3D(const size_t *rowIdx, size_t nrows,
                          const size_t *colIdx, size_t ncols,
                          const size_t *pageIdx, size_t npages,
                          keel::Allocator *mr) const
{
    // Scalar shortcut for non-CELL types
    if (nrows == 1 && ncols == 1 && npages == 1 && type() != ValueType::CELL) {
        size_t idx = dims().sub2ind(subIdx(rowIdx, 0), subIdx(colIdx, 0), subIdx(pageIdx, 0));
        return narrowComplex(elemAt(idx, mr), mr);
    }

    ValueType t = type();

    // MATLAB convention: trailing singleton dims are stripped from a
    // slice (`A(:,:,k)` is 2-D, not [m,n,1]). Middle singletons stay
    // (`A(:,k,:)` is [m,1,p]). Arrays are always ≥ 2-D, so we never
    // strip below 2.
    const bool squeezePage = (npages == 1);

    if (t == ValueType::CELL) {
        auto &d = dims();
        auto result = squeezePage ? Value::cell(nrows, ncols, mr)
                                  : Value::cell3D(nrows, ncols, npages, mr);
        Dims rd(nrows, ncols, npages);
        for (size_t p = 0; p < npages; ++p)
            for (size_t c = 0; c < ncols; ++c)
                for (size_t r = 0; r < nrows; ++r)
                    result.cellAt(rd.sub2ind(r, c, p)) =
                        cellAt(d.sub2ind(subIdx(rowIdx, r), subIdx(colIdx, c), subIdx(pageIdx, p)));
        return result;
    }
    if (t == ValueType::OBJECT) {
        auto &d = dims();
        Dims rd(nrows, ncols, npages);
        keel::SmallArray<size_t, 32> src(nrows * ncols * npages, mr ? mr : keel::default_allocator());
        for (size_t p = 0; p < npages; ++p)
            for (size_t c = 0; c < ncols; ++c)
                for (size_t r = 0; r < nrows; ++r)
                    src[rd.sub2ind(r, c, p)] =
                        d.sub2ind(subIdx(rowIdx, r), subIdx(colIdx, c), subIdx(pageIdx, p));
        Dims resD = squeezePage ? Dims(nrows, ncols) : Dims(nrows, ncols, npages);
        return objectGather(src.data(), resD, mr);
    }

    size_t es = elementSize(t);
    if (es == 0)
        throw std::runtime_error(
            keel::format_small<96>("indexGet3D not supported for type '{}'", mtypeName(t)).c_str());

    auto &d = dims();
    auto result = squeezePage ? Value::matrix(nrows, ncols, t, mr)
                              : Value::matrix3d(nrows, ncols, npages, t, mr);
    result.setSingleComplex(isSingleComplex());
    if (nrows == 0 || ncols == 0 || npages == 0)
        return result;

    const char *src = static_cast<const char *>(rawData());
    char *dst = static_cast<char *>(result.rawDataMut());

    // Contiguous page or column-fiber fast paths when all rows are selected
    if (rowIdx == nullptr && nrows == d.rows()) {
        if (colIdx == nullptr && ncols == d.cols()) {
            const size_t pageBytes = nrows * ncols * es;
            for (size_t p = 0; p < npages; ++p) {
                const size_t sp = subIdx(pageIdx, p);
                keel::mem_copy(dst + p * pageBytes, src + d.sub2ind(0, 0, sp) * es, pageBytes);
            }
        } else {
            const size_t colBytes = nrows * es;
            for (size_t p = 0; p < npages; ++p) {
                const size_t sp = subIdx(pageIdx, p);
                for (size_t c = 0; c < ncols; ++c) {
                    const size_t sc = subIdx(colIdx, c);
                    keel::mem_copy(dst + (p * ncols + c) * colBytes,
                                   src + d.sub2ind(0, sc, sp) * es, colBytes);
                }
            }
        }
        return narrowComplex(keel::move(result), mr);
    }

    Dims rd(nrows, ncols, npages);
    for (size_t p = 0; p < npages; ++p) {
        const size_t sp = subIdx(pageIdx, p);
        for (size_t c = 0; c < ncols; ++c) {
            const size_t sc = subIdx(colIdx, c);
            for (size_t r = 0; r < nrows; ++r)
                keel::mem_copy(dst + rd.sub2ind(r, c, p) * es,
                               src + d.sub2ind(subIdx(rowIdx, r), sc, sp) * es, es);
        }
    }
    return narrowComplex(keel::move(result), mr);   // all-real complex slice -> real
}

// ND slice: extract sub-tensor at given per-dim indices.
Value Value::indexGetND(const size_t *const *perDimIdx,
                          const size_t *perDimCount,
                          int nd,
                          keel::Allocator *mr) const
{
    if (nd <= 0) return Value();
    if (nd == 1) return indexGet(perDimIdx[0], perDimCount[0], mr);
    if (nd == 2)
        return indexGet2D(perDimIdx[0], perDimCount[0],
                          perDimIdx[1], perDimCount[1], mr);
    if (nd == 3)
        return indexGet3D(perDimIdx[0], perDimCount[0],
                          perDimIdx[1], perDimCount[1],
                          perDimIdx[2], perDimCount[2], mr);

    ValueType t = type();
    const bool isCell = (t == ValueType::CELL);
    const bool isObj = (t == ValueType::OBJECT);
    size_t es = (isCell || isObj) ? 0 : elementSize(t);
    if (!isCell && !isObj && es == 0)
        throw std::runtime_error(
            keel::format_small<96>("indexGetND not supported for type '{}'", mtypeName(t)).c_str());

    // Build output shape from perDimCount; trailing zero-extent dims
    // produce an empty tensor. MATLAB convention: trailing singleton
    // dims are stripped from a slice (so `A(:,:,k,1)` on a 4-D array
    // squeezes to 2-D). Middle singletons stay (`A(:,k,:)` is [m,1,p]).
    // Arrays are always ≥ 2-D, so we never strip below 2.
    size_t totalOut = 1;
    for (int i = 0; i < nd; ++i) totalOut *= perDimCount[i];
    int outNd = nd;
    while (outNd > 2 && perDimCount[outNd - 1] == 1) --outNd;
    Value result;
    if (!isObj) {
        result = isCell ? Value::cellND(perDimCount, outNd, mr)
                        : Value::matrixND(perDimCount, outNd, t, mr);
        result.setSingleComplex(isSingleComplex());
    }
    if (totalOut == 0)
        return isObj ? objectGather(nullptr, Dims(perDimCount, outNd), mr) : result;

    // Source strides (column-major) for the existing tensor's actual rank.
    auto &srcDims = dims();
    constexpr int kMaxNd = Dims::kMaxRank;
    size_t srcStrides[kMaxNd];
    if (srcDims.ndim() > kMaxNd)
        throw std::runtime_error("indexGetND: source rank exceeds 32");
    computeStridesColMajor(srcDims, srcStrides);
    const int srcNd = srcDims.ndim();

    // Bounds check: for any dim i picked from perDimIdx[i], idx < srcDims.dim(i).
    // For dims i beyond the source's actual ndim, the source is implicitly
    // extended with trailing singletons — only idx == 0 is valid.
    for (int i = 0; i < nd; ++i) {
        const size_t lim = (i < srcNd) ? srcDims.dim(i) : 1;
        for (size_t k = 0; k < perDimCount[i]; ++k)
            if (perDimIdx[i][k] >= lim)
                throw std::runtime_error(
                    keel::format_small<128>(
                        "Index exceeds array dimensions (dim {}: {} > {})",
                        i + 1, perDimIdx[i][k] + 1, lim).c_str());
    }

    Dims outIter(perDimCount, nd);
    size_t coords[kMaxNd] = {0};
    size_t outIdx = 0;
    if (isObj) {
        // Object N-D slice: collect source linear offsets, then gather with
        // the per-element value/handle rule into the output-shaped array.
        keel::SmallArray<size_t, 32> src(totalOut, mr ? mr : keel::default_allocator());
        do {
            size_t srcOff = 0;
            const int loopN = keel::min(nd, srcNd);
            for (int i = 0; i < loopN; ++i)
                srcOff += perDimIdx[i][coords[i]] * srcStrides[i];
            src[outIdx++] = srcOff;
        } while (incrementCoords(coords, outIter));
        return objectGather(src.data(), Dims(perDimCount, outNd), mr);
    }
    if (isCell) {
        do {
            size_t srcOff = 0;
            const int loopN = keel::min(nd, srcNd);
            for (int i = 0; i < loopN; ++i) {
                const size_t pickedIdx = perDimIdx[i][coords[i]];
                srcOff += pickedIdx * srcStrides[i];
            }
            result.cellAt(outIdx) = cellAt(srcOff);
            ++outIdx;
        } while (incrementCoords(coords, outIter));
    } else {
        const char *src = static_cast<const char *>(rawData());
        char *dst = static_cast<char *>(result.rawDataMut());
        do {
            size_t srcOff = 0;
            const int loopN = keel::min(nd, srcNd);
            for (int i = 0; i < loopN; ++i) {
                const size_t pickedIdx = perDimIdx[i][coords[i]];
                srcOff += pickedIdx * srcStrides[i];
            }
            // Dims i ≥ srcNd contribute 0 (already verified by bounds check).
            keel::mem_copy(dst + outIdx * es, src + srcOff * es, es);
            ++outIdx;
        } while (incrementCoords(coords, outIter));
    }

    return narrowComplex(keel::move(result), mr);   // all-real complex slice -> real
}

// Logical indexing: extract elements where mask is true → row vector of same type.
Value Value::logicalIndex(const uint8_t *mask, size_t maskLen, keel::Allocator *mr) const
{
    // Check for true values beyond array bounds
    for (size_t i = numel(); i < maskLen; ++i)
        if (mask[i])
            throw std::runtime_error("Index exceeds the number of array elements");
    size_t n = keel::min(maskLen, numel());
    // Count selected elements
    size_t selected = 0;
    for (size_t i = 0; i < n; ++i)
        if (mask[i])
            selected++;

    // Shape: column vector source → column result
    bool colResult = (dims().cols() == 1 && dims().rows() > 1);
    size_t rr = colResult ? selected : 1;
    size_t cc = colResult ? 1 : selected;

    ValueType t = type();
    switch (t) {
    case ValueType::DOUBLE: {
        auto result = Value::matrix(rr, cc, ValueType::DOUBLE, mr);
        double *dst = result.doubleDataMut();
        const double *src = doubleData();
        size_t k = 0;
        for (size_t i = 0; i < n; ++i)
            if (mask[i])
                dst[k++] = src[i];
        return result;
    }
    case ValueType::COMPLEX: {
        auto result = Value::complexMatrix(rr, cc, mr);
        result.setSingleComplex(isSingleComplex());
        keel::c64 *dst = result.complexDataMut();
        const keel::c64 *src = complexData();
        size_t k = 0;
        for (size_t i = 0; i < n; ++i)
            if (mask[i])
                dst[k++] = src[i];
        return result;
    }
    case ValueType::LOGICAL: {
        auto result = Value::matrix(rr, cc, ValueType::LOGICAL, mr);
        uint8_t *dst = result.logicalDataMut();
        const uint8_t *src = logicalData();
        size_t k = 0;
        for (size_t i = 0; i < n; ++i)
            if (mask[i])
                dst[k++] = src[i];
        return result;
    }
    case ValueType::CHAR: {
        auto result = Value::matrix(rr, cc, ValueType::CHAR, mr);
        char *dst = result.charDataMut();
        const char *src = charData();
        size_t k = 0;
        for (size_t i = 0; i < n; ++i)
            if (mask[i])
                dst[k++] = src[i];
        return result;
    }
    case ValueType::CELL: {
        auto result = Value::cell(rr, cc, mr);
        size_t k = 0;
        for (size_t i = 0; i < n; ++i)
            if (mask[i])
                result.cellAt(k++) = cellAt(i);
        return result;
    }
    case ValueType::STRING: {
        auto result = Value::stringArray(rr, cc, mr);
        size_t k = 0;
        for (size_t i = 0; i < n; ++i)
            if (mask[i])
                result.stringElemSet(k++, stringElem(i));
        return result;
    }
    case ValueType::STRUCT: {
        auto result = Value::structArray(rr, cc, mr);
        if (result.heap_->fieldOrder) {
            if (heap_->fieldOrder && !heap_->fieldOrder->empty()) {
                *result.heap_->fieldOrder = heap_->fieldOrder->clone(result.heap_->alloc);
            } else {
                auto fnms = fieldNamesInOrder();
                *result.heap_->fieldOrder = fnms.clone(result.heap_->alloc);
            }
        }
        size_t k = 0;
        for (size_t i = 0; i < n; ++i) {
            if (mask[i]) {
                const auto &src = (*heap_->structArray)[i];
                (*result.heap_->structArray)[k++] = src.clone(result.heap_->alloc);
            }
        }
        return result;
    }
    case ValueType::OBJECT: {
        keel::SmallArray<size_t, 32> sel(mr ? mr : keel::default_allocator());
        sel.reserve(rr * cc);
        for (size_t i = 0; i < n; ++i)
            if (mask[i])
                sel.push_back(i);
        return objectGather(sel.data(), Dims(rr, cc), mr);
    }
    // Typed integer / single: raw memcpy per selected element.
    case ValueType::INT8: case ValueType::INT16: case ValueType::INT32: case ValueType::INT64:
    case ValueType::UINT8: case ValueType::UINT16: case ValueType::UINT32: case ValueType::UINT64:
    case ValueType::SINGLE: {
        size_t es = elementSize(t);
        auto result = Value::matrix(rr, cc, t, mr);
        const char *src = static_cast<const char *>(rawData());
        char *dst = static_cast<char *>(result.rawDataMut());
        size_t k = 0;
        for (size_t i = 0; i < n; ++i)
            if (mask[i])
                keel::mem_copy(dst + k++ * es, src + i * es, es);
        return result;
    }
    default:
        throw std::runtime_error(
            keel::format_small<96>("logicalIndex not supported for type '{}'", mtypeName(t)).c_str());
    }
}

// ============================================================
// Index resolution — convert Value index to keel::Array<keel::usize> (0-based)
// ============================================================

keel::usize Value::checkedScalarIndex(double v)
{
    keel::usize out0 = 0;
    if (!tryIndex0(v, out0))
        throw std::runtime_error(
            "Array indices must be positive integers or logical values.");
    return out0;
}

keel::Array<keel::usize> Value::resolveIndices(const Value &idx, size_t dimSize)
{
    keel::Array<keel::usize> out;
    if (idx.isChar() && idx.numel() == 1 && idx.charData()[0] == ':') {
        out.resize(dimSize);
        for (size_t i = 0; i < dimSize; ++i)
            out[i] = i;
    } else if (idx.isLogical()) {
        const uint8_t *m = idx.logicalData();
        for (size_t i = dimSize; i < idx.numel(); ++i)
            if (m[i])
                throw std::runtime_error("Index exceeds the number of array elements");
        size_t n = keel::min(idx.numel(), dimSize);
        for (size_t i = 0; i < n; ++i)
            if (m[i])
                out.push_back(i);
    } else if (idx.isDoubleScalar()) {
        keel::usize ii = Value::checkedScalarIndex(idx.toScalar());
        if (ii >= dimSize)
            throw std::runtime_error("Index exceeds array dimensions");
        out.push_back(ii);
    } else {
        // Any non-logical numeric index: double, single, integer types, or a
        // char vector (code points) — MATLAB accepts them all. elemAsDouble
        // reads each generically (doubleData() throws "Not a double array" for
        // non-DOUBLE index arrays, e.g. A(int32([1 3]))).
        out.reserve(idx.numel());
        for (size_t i = 0; i < idx.numel(); ++i) {
            keel::usize ii = Value::checkedScalarIndex(idx.elemAsDouble(i));
            if (ii >= dimSize)
                throw std::runtime_error("Index exceeds array dimensions");
            out.push_back(ii);
        }
    }
    return out;
}

keel::Array<keel::usize> Value::resolveIndicesUnchecked(const Value &idx)
{
    keel::Array<keel::usize> out;
    if (idx.isChar() && idx.numel() == 1 && idx.charData()[0] == ':') {
        // Colon in unchecked context — caller must handle this separately
        return out;
    }
    if (idx.isLogical()) {
        const uint8_t *m = idx.logicalData();
        for (size_t i = 0; i < idx.numel(); ++i)
            if (m[i])
                out.push_back(i);
    } else if (idx.isDoubleScalar() || idx.isLogicalScalar()) {
        out.push_back(Value::checkedScalarIndex(idx.toScalar()));
    } else {
        // See resolveIndices: integer/single/char index arrays read via the
        // generic elemAsDouble, not the DOUBLE-only doubleData().
        out.reserve(idx.numel());
        for (size_t i = 0; i < idx.numel(); ++i)
            out.push_back(Value::checkedScalarIndex(idx.elemAsDouble(i)));
    }
    return out;
}

// ============================================================
// Type-preserving indexed assignment
// ============================================================

// Helper: write one element from val into dst array at linear index.
// Converts val to the destination type. COW detach handled by *DataMut().
static void writeElem(Value &dst, size_t idx, const Value &val, size_t valIdx)
{
    ValueType t = dst.type();
    switch (t) {
    case ValueType::DOUBLE:
        dst.doubleDataMut()[idx] = val.elemAsDouble(valIdx);
        break;
    case ValueType::COMPLEX:
        dst.complexDataMut()[idx] = readElemAsComplex(val, valIdx);
        break;
    case ValueType::LOGICAL: {
        if (val.type() == ValueType::COMPLEX) // MATLAB errors; don't drop imag silently
            throw std::runtime_error("keel::c64 values cannot be converted to logicals.");
        double dv = val.elemAsDouble(valIdx);
        dst.logicalDataMut()[idx] = static_cast<uint8_t>(dv != 0.0);
        break;
    }
    case ValueType::CHAR:
        dst.charDataMut()[idx] = doubleToCharSafe(val.elemAsDouble(valIdx));
        break;
    case ValueType::CELL:
        dst.cellAt(idx) = val.cellAt(valIdx);
        break;
    case ValueType::STRING:
        // String-array element write. val can be string, char, or
        // numeric scalar (MATLAB allows any of these). See BUGS.md #5.
        if (val.isString())      dst.stringElemSet(idx, val.stringElem(valIdx));
        else if (val.isChar())   dst.stringElemSet(idx, val.toString());
        else                     dst.stringElemSet(idx, val.toString());
        break;
    // Typed integer / single: narrowing conversion from the source
    // element's double value. MATLAB rounds half away from zero and
    // saturates on overflow.
    case ValueType::INT8:   dst.int8DataMut()[idx]   = roundSaturateInt<int8_t  >(val.elemAsDouble(valIdx)); break;
    case ValueType::INT16:  dst.int16DataMut()[idx]  = roundSaturateInt<int16_t >(val.elemAsDouble(valIdx)); break;
    case ValueType::INT32:  dst.int32DataMut()[idx]  = roundSaturateInt<int32_t >(val.elemAsDouble(valIdx)); break;
    case ValueType::INT64:  dst.int64DataMut()[idx]  = roundSaturateInt<int64_t >(val.elemAsDouble(valIdx)); break;
    case ValueType::UINT8:  dst.uint8DataMut()[idx]  = roundSaturateInt<uint8_t >(val.elemAsDouble(valIdx)); break;
    case ValueType::UINT16: dst.uint16DataMut()[idx] = roundSaturateInt<uint16_t>(val.elemAsDouble(valIdx)); break;
    case ValueType::UINT32: dst.uint32DataMut()[idx] = roundSaturateInt<uint32_t>(val.elemAsDouble(valIdx)); break;
    case ValueType::UINT64: dst.uint64DataMut()[idx] = roundSaturateInt<uint64_t>(val.elemAsDouble(valIdx)); break;
    case ValueType::SINGLE: dst.singleDataMut()[idx] = static_cast<float   >(val.elemAsDouble(valIdx)); break;
    default:
        throw std::runtime_error(
            keel::format_small<96>("Indexed assignment not supported for type '{}'",
                                   mtypeName(t)).c_str());
    }
}

// Write a scalar val (broadcast) into dst at linear index.
static void writeScalar(Value &dst, size_t idx, const Value &val)
{
    ValueType t = dst.type();
    switch (t) {
    case ValueType::DOUBLE:
        dst.doubleDataMut()[idx] = val.toScalar();
        break;
    case ValueType::COMPLEX:
        dst.complexDataMut()[idx] = val.toComplex();
        break;
    case ValueType::LOGICAL:
        if (val.type() == ValueType::COMPLEX) // MATLAB errors; don't drop imag silently
            throw std::runtime_error("keel::c64 values cannot be converted to logicals.");
        dst.logicalDataMut()[idx] = static_cast<uint8_t>(val.toScalar() != 0.0);
        break;
    case ValueType::CHAR:
        dst.charDataMut()[idx] = doubleToCharSafe(val.toScalar());
        break;
    case ValueType::CELL:
        dst.cellAt(idx) = val;
        break;
    case ValueType::STRING:
        // Broadcast assignment: write the same string value into idx.
        // String, char, or numeric-scalar source. See BUGS.md #5.
        if (val.isString())     dst.stringElemSet(idx, val.stringElem(0));
        else if (val.isChar())  dst.stringElemSet(idx, val.toString());
        else                    dst.stringElemSet(idx, val.toString());
        break;
    case ValueType::INT8:   dst.int8DataMut()[idx]   = roundSaturateInt<int8_t  >(val.toScalar()); break;
    case ValueType::INT16:  dst.int16DataMut()[idx]  = roundSaturateInt<int16_t >(val.toScalar()); break;
    case ValueType::INT32:  dst.int32DataMut()[idx]  = roundSaturateInt<int32_t >(val.toScalar()); break;
    case ValueType::INT64:  dst.int64DataMut()[idx]  = roundSaturateInt<int64_t >(val.toScalar()); break;
    case ValueType::UINT8:  dst.uint8DataMut()[idx]  = roundSaturateInt<uint8_t >(val.toScalar()); break;
    case ValueType::UINT16: dst.uint16DataMut()[idx] = roundSaturateInt<uint16_t>(val.toScalar()); break;
    case ValueType::UINT32: dst.uint32DataMut()[idx] = roundSaturateInt<uint32_t>(val.toScalar()); break;
    case ValueType::UINT64: dst.uint64DataMut()[idx] = roundSaturateInt<uint64_t>(val.toScalar()); break;
    case ValueType::SINGLE: dst.singleDataMut()[idx] = static_cast<float   >(val.toScalar()); break;
    default:
        throw std::runtime_error(
            keel::format_small<96>("Indexed assignment not supported for type '{}'",
                                   mtypeName(t)).c_str());
    }
}

void Value::elemSet(size_t idx, const Value &val)
{
    // Promote double/single→complex if assigning complex value
    if ((type() == ValueType::DOUBLE || type() == ValueType::SINGLE) && val.isComplex())
        promoteToComplex();
    writeScalar(*this, idx, val);
}

void Value::indexSet(const size_t *indices, size_t count, const Value &val)
{
    // Promote double/single→complex if assigning complex value
    if ((type() == ValueType::DOUBLE || type() == ValueType::SINGLE) && val.isComplex())
        promoteToComplex();
    if (val.isScalar()) {
        for (size_t k = 0; k < count; ++k)
            writeScalar(*this, indices[k], val);
    } else {
        if (count != val.numel())
            throw std::runtime_error(
                "Unable to perform assignment because the left and right sides have a different number of elements");
        for (size_t k = 0; k < count; ++k)
            writeElem(*this, indices[k], val, k);
    }
}

void Value::indexSet2D(const size_t *rowIdx, size_t nrows,
                        const size_t *colIdx, size_t ncols,
                        const Value &val)
{
    if ((type() == ValueType::DOUBLE || type() == ValueType::SINGLE) && val.isComplex())
        promoteToComplex();
    auto &d = dims();
    if (val.isScalar()) {
        if (type() == ValueType::DOUBLE && !val.isComplex() && nrows > 0 && ncols > 0) {
            double *dst = doubleDataMut();
            const double sv = val.toScalar();
            const size_t dstRows = d.rows();
            for (size_t c = 0; c < ncols; ++c) {
                double *dc = dst + subIdx(colIdx, c) * dstRows;
                if (rowIdx == nullptr) {
                    for (size_t r = 0; r < nrows; ++r)
                        dc[r] = sv;
                } else {
                    for (size_t r = 0; r < nrows; ++r)
                        dc[rowIdx[r]] = sv;
                }
            }
            return;
        }
        if (type() == ValueType::COMPLEX && nrows > 0 && ncols > 0) {
            keel::c64 *dst = complexDataMut();
            const keel::c64 sv = val.toComplex();
            const size_t dstRows = d.rows();
            for (size_t c = 0; c < ncols; ++c) {
                keel::c64 *dc = dst + subIdx(colIdx, c) * dstRows;
                if (rowIdx == nullptr) {
                    for (size_t r = 0; r < nrows; ++r)
                        dc[r] = sv;
                } else {
                    for (size_t r = 0; r < nrows; ++r)
                        dc[rowIdx[r]] = sv;
                }
            }
            return;
        }
        for (size_t c = 0; c < ncols; ++c)
            for (size_t r = 0; r < nrows; ++r)
                writeScalar(*this, d.sub2ind(subIdx(rowIdx, r), subIdx(colIdx, c)), val);
    } else {
        size_t total = nrows * ncols;
        if (total != val.numel())
            throw std::runtime_error(
                "Unable to perform assignment because the left and right sides have a different number of elements");
        if (total == 0)
            return;
        if (type() == ValueType::DOUBLE && val.type() == ValueType::DOUBLE && !val.isComplex()) {
            double *dst = doubleDataMut();
            const double *src = val.doubleData();
            const size_t dstRows = d.rows();
            if (rowIdx == nullptr && nrows == dstRows && colIdx == nullptr && ncols == d.cols()) {
                keel::mem_copy(dst, src, total * sizeof(double));
                return;
            }
            if (rowIdx == nullptr) {
                for (size_t c = 0; c < ncols; ++c)
                    keel::mem_copy(dst + subIdx(colIdx, c) * dstRows,
                                   src + c * nrows,
                                   nrows * sizeof(double));
                return;
            }
            size_t k = 0;
            for (size_t c = 0; c < ncols; ++c) {
                double *dc = dst + subIdx(colIdx, c) * dstRows;
                for (size_t r = 0; r < nrows; ++r)
                    dc[rowIdx[r]] = src[k++];
            }
            return;
        }
        if (type() == ValueType::COMPLEX && val.type() == ValueType::COMPLEX) {
            keel::c64 *dst = complexDataMut();
            const keel::c64 *src = val.complexData();
            const size_t dstRows = d.rows();
            if (rowIdx == nullptr && nrows == dstRows && colIdx == nullptr && ncols == d.cols()) {
                keel::mem_copy(dst, src, total * sizeof(keel::c64));
                return;
            }
            if (rowIdx == nullptr) {
                for (size_t c = 0; c < ncols; ++c)
                    keel::mem_copy(dst + subIdx(colIdx, c) * dstRows,
                                   src + c * nrows,
                                   nrows * sizeof(keel::c64));
                return;
            }
            size_t k = 0;
            for (size_t c = 0; c < ncols; ++c) {
                keel::c64 *dc = dst + subIdx(colIdx, c) * dstRows;
                for (size_t r = 0; r < nrows; ++r)
                    dc[rowIdx[r]] = src[k++];
            }
            return;
        }
        size_t k = 0;
        for (size_t c = 0; c < ncols; ++c)
            for (size_t r = 0; r < nrows; ++r)
                writeElem(*this, d.sub2ind(subIdx(rowIdx, r), subIdx(colIdx, c)), val, k++);
    }
}

void Value::indexSet3D(const size_t *rowIdx, size_t nrows,
                        const size_t *colIdx, size_t ncols,
                        const size_t *pageIdx, size_t npages,
                        const Value &val)
{
    if ((type() == ValueType::DOUBLE || type() == ValueType::SINGLE) && val.isComplex())
        promoteToComplex();
    auto &d = dims();
    if (val.isScalar()) {
        if (type() == ValueType::DOUBLE && !val.isComplex() && nrows > 0 && ncols > 0 && npages > 0) {
            double *dst = doubleDataMut();
            const double sv = val.toScalar();
            for (size_t p = 0; p < npages; ++p) {
                const size_t sp = subIdx(pageIdx, p);
                for (size_t c = 0; c < ncols; ++c) {
                    double *dc = dst + d.sub2ind(0, subIdx(colIdx, c), sp);
                    if (rowIdx == nullptr) {
                        for (size_t r = 0; r < nrows; ++r)
                            dc[r] = sv;
                    } else {
                        for (size_t r = 0; r < nrows; ++r)
                            dc[rowIdx[r]] = sv;
                    }
                }
            }
            return;
        }
        for (size_t p = 0; p < npages; ++p)
            for (size_t c = 0; c < ncols; ++c)
                for (size_t r = 0; r < nrows; ++r)
                    writeScalar(*this, d.sub2ind(subIdx(rowIdx, r), subIdx(colIdx, c), subIdx(pageIdx, p)), val);
    } else {
        size_t total = nrows * ncols * npages;
        if (total != val.numel())
            throw std::runtime_error(
                "Unable to perform assignment because the left and right sides have a different number of elements");
        if (total == 0)
            return;
        if (type() == ValueType::DOUBLE && val.type() == ValueType::DOUBLE && !val.isComplex()) {
            double *dst = doubleDataMut();
            const double *src = val.doubleData();
            if (rowIdx == nullptr && nrows == d.rows() && colIdx == nullptr && ncols == d.cols()) {
                const size_t pageElems = nrows * ncols;
                for (size_t p = 0; p < npages; ++p)
                    keel::mem_copy(dst + d.sub2ind(0, 0, subIdx(pageIdx, p)),
                                   src + p * pageElems,
                                   pageElems * sizeof(double));
                return;
            }
            if (rowIdx == nullptr) {
                for (size_t p = 0; p < npages; ++p) {
                    const size_t sp = subIdx(pageIdx, p);
                    for (size_t c = 0; c < ncols; ++c)
                        keel::mem_copy(dst + d.sub2ind(0, subIdx(colIdx, c), sp),
                                       src + (p * ncols + c) * nrows,
                                       nrows * sizeof(double));
                }
                return;
            }
            size_t k = 0;
            for (size_t p = 0; p < npages; ++p) {
                const size_t sp = subIdx(pageIdx, p);
                for (size_t c = 0; c < ncols; ++c) {
                    double *dc = dst + d.sub2ind(0, subIdx(colIdx, c), sp);
                    for (size_t r = 0; r < nrows; ++r)
                        dc[rowIdx[r]] = src[k++];
                }
            }
            return;
        }
        if (type() == ValueType::COMPLEX && val.type() == ValueType::COMPLEX) {
            keel::c64 *dst = complexDataMut();
            const keel::c64 *src = val.complexData();
            if (rowIdx == nullptr && nrows == d.rows() && colIdx == nullptr && ncols == d.cols()) {
                const size_t pageElems = nrows * ncols;
                for (size_t p = 0; p < npages; ++p)
                    keel::mem_copy(dst + d.sub2ind(0, 0, subIdx(pageIdx, p)),
                                   src + p * pageElems,
                                   pageElems * sizeof(keel::c64));
                return;
            }
            if (rowIdx == nullptr) {
                for (size_t p = 0; p < npages; ++p) {
                    const size_t sp = subIdx(pageIdx, p);
                    for (size_t c = 0; c < ncols; ++c)
                        keel::mem_copy(dst + d.sub2ind(0, subIdx(colIdx, c), sp),
                                       src + (p * ncols + c) * nrows,
                                       nrows * sizeof(keel::c64));
                }
                return;
            }
            size_t k = 0;
            for (size_t p = 0; p < npages; ++p) {
                const size_t sp = subIdx(pageIdx, p);
                for (size_t c = 0; c < ncols; ++c) {
                    keel::c64 *dc = dst + d.sub2ind(0, subIdx(colIdx, c), sp);
                    for (size_t r = 0; r < nrows; ++r)
                        dc[rowIdx[r]] = src[k++];
                }
            }
            return;
        }
        size_t k = 0;
        for (size_t p = 0; p < npages; ++p)
            for (size_t c = 0; c < ncols; ++c)
                for (size_t r = 0; r < nrows; ++r)
                    writeElem(*this, d.sub2ind(subIdx(rowIdx, r), subIdx(colIdx, c), subIdx(pageIdx, p)), val, k++);
    }
}

void Value::indexSetND(const size_t *const *perDimIdx,
                        const size_t *perDimCount,
                        int nd,
                        const Value &val)
{
    if (nd <= 0) return;
    if (nd == 1) { indexSet(perDimIdx[0], perDimCount[0], val); return; }
    if (nd == 2) {
        indexSet2D(perDimIdx[0], perDimCount[0],
                   perDimIdx[1], perDimCount[1], val);
        return;
    }
    if (nd == 3) {
        indexSet3D(perDimIdx[0], perDimCount[0],
                   perDimIdx[1], perDimCount[1],
                   perDimIdx[2], perDimCount[2], val);
        return;
    }

    // CELL is handled element-wise via writeScalar/writeElem (which use
    // cellAt under the hood) — no special-casing needed here past the
    // shared scalar/elementwise loops below.
    if ((type() == ValueType::DOUBLE || type() == ValueType::SINGLE) && val.isComplex())
        promoteToComplex();

    constexpr int kMaxNd = Dims::kMaxRank;
    if (dims().ndim() > kMaxNd)
        throw std::runtime_error("indexSetND: target rank exceeds 32");

    // Auto-grow: expand any axis whose largest assignment index exceeds
    // current dim, including new trailing axes (rank goes up). Matches
    // MATLAB: `A(i,j,k,l) = v` creates / extends as needed.
    {
        const int curNd = dims().ndim();
        const int newNd = keel::max(nd, curNd);
        if (newNd > kMaxNd)
            throw std::runtime_error("indexSetND: target rank exceeds 32");
        size_t need[kMaxNd];
        for (int i = 0; i < newNd; ++i)
            need[i] = (i < curNd) ? dims().dim(i) : 1;
        bool grow = false;
        for (int i = 0; i < nd; ++i) {
            for (size_t k = 0; k < perDimCount[i]; ++k) {
                if (perDimIdx[i][k] + 1 > need[i]) {
                    need[i] = perDimIdx[i][k] + 1;
                    grow = true;
                }
            }
        }
        if (grow)
            resizeND(need, newNd, nullptr);
    }

    auto &d = dims();
    size_t dstStrides[kMaxNd];
    computeStridesColMajor(d, dstStrides);
    const int dstNd = d.ndim();

    size_t total = 1;
    for (int i = 0; i < nd; ++i) total *= perDimCount[i];
    if (total == 0) return;

    Dims iter(perDimCount, nd);
    size_t coords[kMaxNd] = {0};

    if (val.isScalar()) {
        do {
            size_t off = 0;
            const int loopN = keel::min(nd, dstNd);
            for (int i = 0; i < loopN; ++i)
                off += perDimIdx[i][coords[i]] * dstStrides[i];
            writeScalar(*this, off, val);
        } while (incrementCoords(coords, iter));
    } else {
        if (total != val.numel())
            throw std::runtime_error(
                "Unable to perform assignment because the left and right sides have a different number of elements");
        size_t k = 0;
        do {
            size_t off = 0;
            const int loopN = keel::min(nd, dstNd);
            for (int i = 0; i < loopN; ++i)
                off += perDimIdx[i][coords[i]] * dstStrides[i];
            writeElem(*this, off, val, k++);
        } while (incrementCoords(coords, iter));
    }
}

// ============================================================
// Type-preserving index deletion
// ============================================================

void Value::indexDelete(const size_t *indices, size_t count, keel::Allocator *mr)
{
    ValueType t = type();
    // STRING elements are variable-length strings, not fixed-width bytes, so the
    // memcpy-based path below would silently produce zero-sized output
    // (elementSize(STRING) == 0). Reject up-front alongside the other
    // structurally-incompatible types.
    if (t == ValueType::STRUCT || t == ValueType::FUNC_HANDLE || t == ValueType::EMPTY
        || t == ValueType::STRING)
        throw std::runtime_error(
            keel::format_small<96>("Delete indexing not supported for type '{}'", mtypeName(t)).c_str());

    size_t n = numel();
    keel::Allocator *tmpAlloc = mr ? mr : keel::default_allocator();
    keel::SmallArray<keel::u8, 64> del(n, keel::u8{0}, tmpAlloc);
    for (size_t k = 0; k < count; ++k)
        if (indices[k] < n)
            del[indices[k]] = 1;

    size_t remaining = 0;
    for (size_t i = 0; i < n; ++i)
        if (!del[i])
            ++remaining;
    // MATLAB: 1D delete always produces row vector, except column vectors stay column
    bool isRow = !(dims().cols() == 1 && dims().rows() > 1);

    if (t == ValueType::CELL) {
        auto &src = cellDataVec();
        auto result = isRow ? Value::cell(1, remaining, mr) : Value::cell(remaining, 1, mr);
        auto &dst = result.cellDataVec();
        size_t j = 0;
        for (size_t i = 0; i < n; ++i)
            if (!del[i])
                dst[j++] = src[i];
        *this = keel::move(result);
        return;
    }

    if (t == ValueType::OBJECT) {
        keel::SmallArray<size_t, 64> kept(tmpAlloc);
        kept.reserve(remaining);
        for (size_t i = 0; i < n; ++i)
            if (!del[i])
                kept.push_back(i);
        Dims rd = isRow ? Dims(size_t{1}, remaining) : Dims(remaining, size_t{1});
        *this = objectGather(kept.data(), rd, mr);
        return;
    }

    size_t es = elementSize(t);
    const bool sc = isSingleComplex();
    auto result = isRow ? Value::matrix(1, remaining, t, mr)
                        : Value::matrix(remaining, 1, t, mr);
    result.setSingleComplex(sc);
    if (remaining > 0 && es > 0) {
        const char *src = static_cast<const char *>(rawData());
        char *dst = static_cast<char *>(result.rawDataMut());
        size_t j = 0;
        for (size_t i = 0; i < n; ++i)
            if (!del[i])
                keel::mem_copy(dst + j++ * es, src + i * es, es);
    }
    *this = keel::move(result);
}

void Value::indexDelete2D(const size_t *rowIdx, size_t nrows,
                           const size_t *colIdx, size_t ncols,
                           keel::Allocator *mr)
{
    ValueType t = type();
    if (t == ValueType::STRUCT || t == ValueType::FUNC_HANDLE || t == ValueType::EMPTY
        || t == ValueType::STRING)
        throw std::runtime_error(
            keel::format_small<96>("Delete indexing not supported for type '{}'", mtypeName(t)).c_str());

    size_t R = dims().rows(), C = dims().cols();
    const bool sc = isSingleComplex();
    keel::Allocator *tmpAlloc = mr ? mr : keel::default_allocator();

    if (ncols == C) {
        // Delete rows
        keel::SmallArray<keel::u8, 64> delR(R, keel::u8{0}, tmpAlloc);
        for (size_t k = 0; k < nrows; ++k)
            if (rowIdx[k] < R)
                delR[rowIdx[k]] = 1;
        size_t newR = 0;
        for (size_t r = 0; r < R; ++r)
            if (!delR[r])
                ++newR;

        if (t == ValueType::CELL) {
            auto &src = cellDataVec();
            auto result = Value::cell(newR, C, mr);
            auto &dst = result.cellDataVec();
            size_t ri = 0;
            for (size_t r = 0; r < R; ++r)
                if (!delR[r]) {
                    for (size_t c = 0; c < C; ++c)
                        dst[c * newR + ri] = src[c * R + r];
                    ri++;
                }
            *this = keel::move(result);
        } else if (t == ValueType::OBJECT) {
            keel::SmallArray<size_t, 64> kept(tmpAlloc);
            kept.reserve(newR * C);
            for (size_t c = 0; c < C; ++c)
                for (size_t r = 0; r < R; ++r)
                    if (!delR[r])
                        kept.push_back(c * R + r);
            *this = objectGather(kept.data(), Dims(newR, C), mr);
        } else {
            size_t es = elementSize(t);
            auto result = Value::matrix(newR, C, t, mr);
            result.setSingleComplex(sc);
            if (newR > 0 && es > 0) {
                const char *src = static_cast<const char *>(rawData());
                char *dst = static_cast<char *>(result.rawDataMut());
                size_t ri = 0;
                for (size_t r = 0; r < R; ++r)
                    if (!delR[r]) {
                        for (size_t c = 0; c < C; ++c)
                            keel::mem_copy(dst + (c * newR + ri) * es,
                                           src + (c * R + r) * es, es);
                        ri++;
                    }
            }
            *this = keel::move(result);
        }
    } else if (nrows == R) {
        // Delete columns
        keel::SmallArray<keel::u8, 64> delC(C, keel::u8{0}, tmpAlloc);
        for (size_t k = 0; k < ncols; ++k)
            if (colIdx[k] < C)
                delC[colIdx[k]] = 1;
        size_t newC = 0;
        for (size_t c = 0; c < C; ++c)
            if (!delC[c])
                ++newC;

        if (t == ValueType::CELL) {
            auto &src = cellDataVec();
            auto result = Value::cell(R, newC, mr);
            auto &dst = result.cellDataVec();
            size_t ci = 0;
            for (size_t c = 0; c < C; ++c)
                if (!delC[c]) {
                    for (size_t r = 0; r < R; ++r)
                        dst[ci * R + r] = src[c * R + r];
                    ci++;
                }
            *this = keel::move(result);
        } else if (t == ValueType::OBJECT) {
            keel::SmallArray<size_t, 64> kept(tmpAlloc);
            kept.reserve(R * newC);
            for (size_t c = 0; c < C; ++c)
                if (!delC[c])
                    for (size_t r = 0; r < R; ++r)
                        kept.push_back(c * R + r);
            *this = objectGather(kept.data(), Dims(R, newC), mr);
        } else {
            size_t es = elementSize(t);
            auto result = Value::matrix(R, newC, t, mr);
            result.setSingleComplex(sc);
            if (newC > 0 && es > 0) {
                const char *src = static_cast<const char *>(rawData());
                char *dst = static_cast<char *>(result.rawDataMut());
                size_t ci = 0;
                for (size_t c = 0; c < C; ++c)
                    if (!delC[c]) {
                        keel::mem_copy(dst + ci * R * es, src + c * R * es, R * es);
                        ci++;
                    }
            }
            *this = keel::move(result);
        }
    } else {
        throw std::runtime_error(
            "Cannot delete from both rows and columns simultaneously");
    }
}

void Value::indexDelete3D(const size_t *rowIdx, size_t nrows,
                           const size_t *colIdx, size_t ncols,
                           const size_t *pageIdx, size_t npages,
                           keel::Allocator *mr)
{
    ValueType t = type();
    if (t == ValueType::STRUCT || t == ValueType::FUNC_HANDLE || t == ValueType::EMPTY
        || t == ValueType::STRING)
        throw std::runtime_error(
            keel::format_small<96>("Delete indexing not supported for type '{}'", mtypeName(t)).c_str());

    size_t R = dims().rows(), C = dims().cols(), P = dims().pages();
    const bool sc = isSingleComplex();
    keel::Allocator *tmpAlloc = mr ? mr : keel::default_allocator();

    // Exactly one dimension must be a proper subset; the other two must be fully selected
    bool fullR = (nrows == R), fullC = (ncols == C), fullP = (npages == P);
    int partialCount = (!fullR ? 1 : 0) + (!fullC ? 1 : 0) + (!fullP ? 1 : 0);
    if (partialCount != 1)
        throw std::runtime_error(
            "3D delete requires exactly one dimension to be partially selected");

    if (!fullP) {
        // Delete pages
        keel::SmallArray<keel::u8, 64> delP(P, keel::u8{0}, tmpAlloc);
        for (size_t k = 0; k < npages; ++k)
            if (pageIdx[k] < P) delP[pageIdx[k]] = 1;
        size_t newP = 0;
        for (size_t p = 0; p < P; ++p)
            if (!delP[p])
                ++newP;

        if (t == ValueType::CELL) {
            auto result = Value::cell3D(R, C, newP, mr);
            Dims rd(R, C, newP);
            size_t pi = 0;
            for (size_t p = 0; p < P; ++p)
                if (!delP[p]) {
                    for (size_t c = 0; c < C; ++c)
                        for (size_t r = 0; r < R; ++r)
                            result.cellAt(rd.sub2ind(r, c, pi)) =
                                cellAt(dims().sub2ind(r, c, p));
                    pi++;
                }
            *this = keel::move(result);
        } else {
            size_t es = elementSize(t);
            auto result = Value::matrix3d(R, C, newP, t, mr);
            result.setSingleComplex(sc);
            size_t sliceBytes = R * C * es;
            const char *src = static_cast<const char *>(rawData());
            char *dst = static_cast<char *>(result.rawDataMut());
            size_t pi = 0;
            for (size_t p = 0; p < P; ++p)
                if (!delP[p])
                    keel::mem_copy(dst + pi++ * sliceBytes, src + p * sliceBytes, sliceBytes);
            *this = keel::move(result);
        }
    } else if (!fullR) {
        // Delete rows (across all pages)
        keel::SmallArray<keel::u8, 64> delR(R, keel::u8{0}, tmpAlloc);
        for (size_t k = 0; k < nrows; ++k)
            if (rowIdx[k] < R) delR[rowIdx[k]] = 1;
        size_t newR = 0;
        for (size_t r = 0; r < R; ++r)
            if (!delR[r])
                ++newR;

        if (t == ValueType::CELL) {
            auto result = Value::cell3D(newR, C, P, mr);
            Dims rd(newR, C, P);
            for (size_t p = 0; p < P; ++p) {
                size_t ri = 0;
                for (size_t r = 0; r < R; ++r)
                    if (!delR[r]) {
                        for (size_t c = 0; c < C; ++c)
                            result.cellAt(rd.sub2ind(ri, c, p)) =
                                cellAt(dims().sub2ind(r, c, p));
                        ri++;
                    }
            }
            *this = keel::move(result);
        } else {
            size_t es = elementSize(t);
            auto result = Value::matrix3d(newR, C, P, t, mr);
            result.setSingleComplex(sc);
            const char *src = static_cast<const char *>(rawData());
            char *dst = static_cast<char *>(result.rawDataMut());
            for (size_t p = 0; p < P; ++p) {
                size_t ri = 0;
                for (size_t r = 0; r < R; ++r)
                    if (!delR[r]) {
                        for (size_t c = 0; c < C; ++c)
                            keel::mem_copy(
                                dst + (p * newR * C + c * newR + ri) * es,
                                src + (p * R * C + c * R + r) * es, es);
                        ri++;
                    }
            }
            *this = keel::move(result);
        }
    } else {
        // Delete columns (across all pages)
        keel::SmallArray<keel::u8, 64> delC(C, keel::u8{0}, tmpAlloc);
        for (size_t k = 0; k < ncols; ++k)
            if (colIdx[k] < C) delC[colIdx[k]] = 1;
        size_t newC = 0;
        for (size_t c = 0; c < C; ++c)
            if (!delC[c])
                ++newC;

        if (t == ValueType::CELL) {
            auto result = Value::cell3D(R, newC, P, mr);
            Dims rd(R, newC, P);
            for (size_t p = 0; p < P; ++p) {
                size_t ci = 0;
                for (size_t c = 0; c < C; ++c)
                    if (!delC[c]) {
                        for (size_t r = 0; r < R; ++r)
                            result.cellAt(rd.sub2ind(r, ci, p)) =
                                cellAt(dims().sub2ind(r, c, p));
                        ci++;
                    }
            }
            *this = keel::move(result);
        } else {
            size_t es = elementSize(t);
            auto result = Value::matrix3d(R, newC, P, t, mr);
            result.setSingleComplex(sc);
            const char *src = static_cast<const char *>(rawData());
            char *dst = static_cast<char *>(result.rawDataMut());
            for (size_t p = 0; p < P; ++p) {
                size_t ci = 0;
                for (size_t c = 0; c < C; ++c)
                    if (!delC[c]) {
                        keel::mem_copy(
                            dst + (p * R * newC + ci * R) * es,
                            src + (p * R * C + c * R) * es, R * es);
                        ci++;
                    }
            }
            *this = keel::move(result);
        }
    }
}

void Value::indexDeleteND(const size_t *const *perDimIdx,
                           const size_t *perDimCount,
                           int nd,
                           keel::Allocator *mr)
{
    if (nd <= 0) return;
    if (nd == 1) { indexDelete(perDimIdx[0], perDimCount[0], mr); return; }
    if (nd == 2) {
        indexDelete2D(perDimIdx[0], perDimCount[0],
                      perDimIdx[1], perDimCount[1], mr);
        return;
    }
    if (nd == 3) {
        indexDelete3D(perDimIdx[0], perDimCount[0],
                      perDimIdx[1], perDimCount[1],
                      perDimIdx[2], perDimCount[2], mr);
        return;
    }

    ValueType t = type();
    if (t == ValueType::STRUCT || t == ValueType::FUNC_HANDLE || t == ValueType::EMPTY
        || t == ValueType::STRING)
        throw std::runtime_error(
            keel::format_small<96>("Delete indexing not supported for type '{}'", mtypeName(t)).c_str());

    auto &d = dims();
    constexpr int kMaxNd = Dims::kMaxRank;
    if (d.ndim() > kMaxNd)
        throw std::runtime_error("indexDeleteND: target rank exceeds 32");
    const int srcNd = d.ndim();

    // Find the partial axis. All other axes must be the full range.
    int partial = -1;
    for (int i = 0; i < nd; ++i) {
        const size_t lim = (i < srcNd) ? d.dim(i) : 1;
        if (perDimCount[i] != lim) {
            if (partial != -1)
                throw std::runtime_error(
                    "ND delete requires exactly one dimension to be partially selected");
            partial = i;
        }
    }
    if (partial == -1)
        throw std::runtime_error(
            "ND delete requires exactly one dimension to be partially selected");
    if (partial >= srcNd)
        throw std::runtime_error(
            "ND delete: partial axis exceeds source rank");

    keel::Allocator *tmpAlloc = mr ? mr : keel::default_allocator();
    const size_t axisLen = d.dim(partial);
    keel::SmallArray<keel::u8, 64> delMask(axisLen, keel::u8{0}, tmpAlloc);
    for (size_t k = 0; k < perDimCount[partial]; ++k)
        if (perDimIdx[partial][k] < axisLen)
            delMask[perDimIdx[partial][k]] = 1;
    size_t newLen = 0;
    for (size_t i = 0; i < axisLen; ++i)
        if (!delMask[i])
            ++newLen;

    size_t newShape[kMaxNd];
    for (int i = 0; i < srcNd; ++i)
        newShape[i] = (i == partial) ? newLen : d.dim(i);

    const bool isCell = (t == ValueType::CELL);
    const size_t es = isCell ? 0 : elementSize(t);
    if (!isCell && es == 0)
        throw std::runtime_error(
            keel::format_small<96>("indexDeleteND not supported for type '{}'", mtypeName(t)).c_str());

    const bool sc = isSingleComplex();
    Value result = isCell ? Value::cellND(newShape, srcNd, mr)
                           : Value::matrixND(newShape, srcNd, t, mr);
    result.setSingleComplex(sc);
    if (newLen == 0) {
        *this = keel::move(result);
        return;
    }

    size_t srcStrides[kMaxNd], dstStrides[kMaxNd];
    computeStridesColMajor(d, srcStrides);
    Dims newDims(newShape, srcNd);
    computeStridesColMajor(newDims, dstStrides);

    // Per-axis-value compressed-output index.
    keel::SmallArray<size_t, 64> remap(axisLen, size_t{0}, tmpAlloc);
    {
        size_t j = 0;
        for (size_t i = 0; i < axisLen; ++i)
            if (!delMask[i]) remap[i] = j++;
    }

    char *dstRaw = isCell ? nullptr : static_cast<char *>(result.rawDataMut());
    const char *srcRaw = isCell ? nullptr : static_cast<const char *>(rawData());

    size_t coords[kMaxNd] = {0};
    do {
        if (delMask[coords[partial]])
            continue;
        size_t srcOff = 0, dstOff = 0;
        for (int i = 0; i < srcNd; ++i) {
            srcOff += coords[i] * srcStrides[i];
            const size_t dc = (i == partial) ? remap[coords[i]] : coords[i];
            dstOff += dc * dstStrides[i];
        }
        if (isCell)
            result.cellAt(dstOff) = cellAt(srcOff);
        else
            keel::mem_copy(dstRaw + dstOff * es, srcRaw + srcOff * es, es);
    } while (incrementCoords(coords, d));

    *this = keel::move(result);
}

Value Value::complexScalar(keel::c64 v, keel::Allocator *mr)
{
    Value m;
    HeapGuard h(HeapObject::create(mr));
    h->type = ValueType::COMPLEX;
    h->dims = {1, 1};
    h->buffer = DataBuffer::create(sizeof(keel::c64), h->alloc);
    *static_cast<keel::c64 *>(h->buffer->data()) = v;
    m.heap_ = h.release();
    return m;
}
Value Value::complexScalar(double re, double im, keel::Allocator *mr)
{
    return complexScalar(keel::c64(re, im), mr);
}
Value Value::complexMatrix(size_t rows, size_t cols, keel::Allocator *mr)
{
    return matrix(rows, cols, ValueType::COMPLEX, mr);
}

ValueType Value::type() const
{
    if (heap_ == nullptr)
        return ValueType::DOUBLE;
    if (heap_ == emptyTag() || heap_ == deletedTag())
        return ValueType::EMPTY;
    if (heap_ == logicalTrueTag() || heap_ == logicalFalseTag())
        return ValueType::LOGICAL;
    return heap_->type;
}
const Dims &Value::dims() const
{
    if (heap_ == nullptr)
        return sScalarDims;
    if (heap_ == emptyTag() || heap_ == deletedTag())
        return sEmptyDims;
    if (heap_ == logicalTrueTag() || heap_ == logicalFalseTag())
        return sScalarDims;
    return heap_->dims;
}
size_t Value::numel() const
{
    return dims().numel();
}
bool Value::isScalar() const
{
    if (heap_ == nullptr || heap_ == logicalTrueTag() || heap_ == logicalFalseTag())
        return true;
    if (heap_ == emptyTag() || heap_ == deletedTag())
        return false;
    return heap_->dims.isScalar();
}
bool Value::isEmpty() const
{
    if (heap_ == emptyTag() || heap_ == deletedTag())
        return true;
    if (heap_ == nullptr || heap_ == logicalTrueTag() || heap_ == logicalFalseTag())
        return false;
    return heap_->type == ValueType::EMPTY || heap_->dims.isEmpty();
}
bool Value::isNumeric() const
{
    ValueType t = type();
    return t == ValueType::DOUBLE || t == ValueType::SINGLE || t == ValueType::COMPLEX || t == ValueType::INT8
           || t == ValueType::INT16 || t == ValueType::INT32 || t == ValueType::INT64 || t == ValueType::UINT8
           || t == ValueType::UINT16 || t == ValueType::UINT32 || t == ValueType::UINT64;
}
bool Value::isComplex() const
{
    return type() == ValueType::COMPLEX;
}
bool Value::isSingleComplex() const
{
    return isHeap() && heap_->type == ValueType::COMPLEX && heap_->isSingleComplex;
}
void Value::setSingleComplex(bool v)
{
    if (isHeap() && heap_->type == ValueType::COMPLEX) {
        detach();
        heap_->isSingleComplex = v;
    }
}
bool Value::isLogical() const
{
    return type() == ValueType::LOGICAL;
}
bool Value::isChar() const
{
    return type() == ValueType::CHAR;
}
bool Value::isCell() const
{
    return type() == ValueType::CELL;
}
bool Value::isCsl() const
{
    return type() == ValueType::CSL;
}
size_t Value::cslCount() const
{
    return (isHeap() && heap_->cellData) ? heap_->cellData->size() : 0;
}
bool Value::isStruct() const
{
    return type() == ValueType::STRUCT;
}
bool Value::isFuncHandle() const
{
    return type() == ValueType::FUNC_HANDLE;
}
bool Value::isString() const
{
    return type() == ValueType::STRING;
}
bool Value::isObject() const
{
    return type() == ValueType::OBJECT;
}
keel::String Value::objectClassName() const
{
    if (isObject() && heap_->objClass)
        return heap_->objClass->clone();
    return {};
}
bool Value::objectIsHandle() const
{
    return isObject() && heap_->objIsHandle;
}
size_t Value::objectCount() const
{
    return isObject() ? heap_->objStates.size() : 0;
}
const ObjectState *Value::objectStateConst() const
{
    return (isObject() && !heap_->objStates.empty()) ? heap_->objStates.front().get()
                                                     : nullptr;
}
ObjectState *Value::objectStateMut()
{
    if (!isObject() || heap_->objStates.empty())
        return nullptr;
    detach(); // COW: applies the value/handle clone rule before mutation
    return heap_->objStates.front().get();
}
const ObjectState *Value::objectStateAt(size_t i) const
{
    return (isObject() && i < heap_->objStates.size()) ? heap_->objStates[i].get()
                                                       : nullptr;
}
ObjectState *Value::objectStateMutAt(size_t i)
{
    if (!isObject() || i >= heap_->objStates.size())
        return nullptr;
    detach(); // COW before mutation (value/handle rule applies per element)
    return heap_->objStates[i].get();
}

Value Value::objectGather(const size_t *srcLinear, const Dims &resultDims,
                          keel::Allocator *mr) const
{
    if (!isObject())
        return Value();
    if (!mr)
        mr = heap_->alloc ? heap_->alloc : keel::default_allocator();
    const bool handle = heap_->objIsHandle;
    const size_t n = resultDims.numel();
    HeapGuard h(HeapObject::create(mr));
    h->type = ValueType::OBJECT;
    h->objClass = h->allocNew<keel::String>(heap_->objClass ? heap_->objClass->as_ref() : keel::StringRef{}, h->alloc);
    h->objIsHandle = handle;
    h->dims = resultDims;
    h->objStates.reserve(n);
    for (size_t k = 0; k < n; ++k) {
        size_t s = srcLinear[k];
        if (s >= heap_->objStates.size())
            throw std::runtime_error("Index exceeds the number of array elements.");
        const auto &src = heap_->objStates[s];
        // value class → independent copy (safe to mutate); handle → alias.
        h->objStates.push_back((handle || !src) ? src : src->deepCopy(mr));
    }
    Value out;
    out.heap_ = h.release();
    return out;
}

Value Value::objectSubArray(keel::ArrayRef<keel::usize> idxs,
                            keel::Allocator *mr) const
{
    if (!isObject())
        return Value();
    bool colResult = (dims().cols() == 1 && dims().rows() > 1);
    Dims d = (idxs.size() == 1) ? Dims(1, 1)
           : (colResult ? Dims(idxs.size(), size_t{1}) : Dims(size_t{1}, idxs.size()));
    return objectGather(idxs.data(), d, mr);
}

Value Value::objectReshape(const Dims &newDims, keel::Allocator *mr) const
{
    if (!isObject())
        return Value();
    // Identity gather (column-major order preserved) into the new shape.
    const size_t n = heap_->objStates.size();
    keel::SmallArray<size_t, 32> idx(n, mr ? mr : keel::default_allocator());
    for (size_t i = 0; i < n; ++i)
        idx[i] = i;
    return objectGather(idx.data(), newDims, mr);
}

Value Value::objectConcatN(const Value *parts, size_t count, const Dims &outDims,
                           keel::Allocator *mr)
{
    if (!mr)
        mr = keel::default_allocator();
    keel::String clsName(mr);
    bool found = false, handle = false;
    for (size_t i = 0; i < count; ++i) {
        if (parts[i].isEmpty())
            continue;
        if (!parts[i].isObject())
            throw std::runtime_error("Cannot concatenate an object with a non-object value");
        if (!found) {
            clsName = parts[i].objectClassName();
            handle = parts[i].objectIsHandle();
            found = true;
        } else {
            const keel::String otherCls = parts[i].objectClassName();
            if (otherCls != clsName) {
                auto isGraphicsCls = [](keel::StringRef c) {
                    return c.starts_with("matlab.graphics.") || c.starts_with("matlab.ui.");
                };
                if (isGraphicsCls(clsName) && isGraphicsCls(otherCls)) {
                    clsName.assign("matlab.graphics.Graphics");
                } else {
                    throw std::runtime_error("Cannot concatenate objects of different classes");
                }
            }
        }
    }
    HeapGuard h(HeapObject::create(mr));
    h->type = ValueType::OBJECT;
    h->objClass = h->allocNew<keel::String>(clsName.as_ref(), h->alloc);
    h->objIsHandle = handle;
    h->dims = outDims;
    for (size_t i = 0; i < count; ++i) {
        if (!parts[i].isObject())
            continue;
        for (const auto &st : parts[i].heap_->objStates)
            h->objStates.push_back((handle || !st) ? st : st->deepCopy(mr));
    }
    Value out;
    out.heap_ = h.release();
    return out;
}

Value Value::objectTranspose(keel::Allocator *mr) const
{
    if (!isObject())
        return Value();
    const size_t R = heap_->dims.rows(), C = heap_->dims.cols();
    keel::SmallArray<size_t, 32> src(R * C, mr ? mr : keel::default_allocator());
    // result is C×R; result(i,j) [col-major k=j*C+i] = src(j,i) [k=i*R+j].
    for (size_t j = 0; j < R; ++j)     // result columns
        for (size_t i = 0; i < C; ++i) // result rows
            src[j * C + i] = i * R + j;
    return objectGather(src.data(), Dims(C, R), mr);
}

// Make *this a fresh empty (0×0) object array of `proto`'s class, if it is
// not already an object. Shared by the slice-assign entry points.
void Value::objectEnsureArrayLike(const Value &proto, keel::Allocator *mr)
{
    if (isObject())
        return;
    const keel::String protoCls = proto.objectClassName();
    HeapGuard h(HeapObject::create(mr));
    h->type = ValueType::OBJECT;
    h->objClass = h->allocNew<keel::String>(protoCls.as_ref(), h->alloc);
    h->objIsHandle = proto.objectIsHandle();
    h->dims = Dims(size_t{0}, size_t{0}); // 0×0; `{0,0}` would pick the ptr ctor
    Value fresh;
    fresh.heap_ = h.release();
    *this = keel::move(fresh);
}

void Value::objectAssignLinear(keel::ArrayRef<keel::usize> pos, const Value &rhs,
                               const Value &fill, keel::Allocator *mr)
{
    if (!mr)
        mr = (isHeap() && heap_->alloc) ? heap_->alloc
                                        : (rhs.isHeap() && rhs.heap_->alloc
                                               ? rhs.heap_->alloc
                                               : keel::default_allocator());
    objectEnsureArrayLike(rhs, mr);
    detach(); // HeapObject-level COW before in-place growth/store
    const bool handle = heap_->objIsHandle;
    auto sharedOf = [](const Value &v) -> keel::Arc<ObjectState> {
        return (v.isObject() && !v.heap_->objStates.empty()) ? v.heap_->objStates.front()
                                                             : keel::Arc<ObjectState>{};
    };
    auto indep = [&](const keel::Arc<ObjectState> &s) {
        return s ? s->deepCopy(mr) : s; // independent default for gaps
    };
    // RHS feeds one element per target, or a scalar broadcast to all.
    const size_t rn = rhs.isObject() ? rhs.objectCount() : 0;
    const bool broadcast = (rn == 1);
    if (!broadcast && rn != pos.size())
        throw std::runtime_error(
            "Unable to perform assignment because the left and right sides have a "
            "different number of elements.");
    auto rhsSlot = [&](size_t k) -> keel::Arc<ObjectState> {
        const auto &sp = rhs.heap_->objStates[broadcast ? 0 : k];
        return handle ? sp : (sp ? sp->deepCopy(mr) : sp); // alias vs deep copy
    };

    size_t maxPos = 0;
    for (size_t p : pos)
        maxPos = keel::max(maxPos, p + 1);
    auto &states = heap_->objStates;
    if (maxPos > states.size()) {
        // Out-of-bounds linear growth is only defined for a vector (or empty);
        // growing a proper matrix via a single linear index is ambiguous.
        if (heap_->dims.rows() > 1 && heap_->dims.cols() > 1)
            throw std::runtime_error(
                "Unable to grow a 2-D object array via a single linear index");
        const bool col = heap_->dims.cols() == 1 && heap_->dims.rows() > 1;
        keel::Arc<ObjectState> fillState = sharedOf(fill);
        while (states.size() < maxPos)
            states.push_back(indep(fillState));
        heap_->dims = col ? Dims(states.size(), size_t{1}) : Dims(size_t{1}, states.size());
    }
    for (size_t k = 0; k < pos.size(); ++k)
        states[pos[k]] = rhsSlot(k);
}

void Value::objectAssignND(keel::ArrayRef<keel::Array<keel::usize>> perDim,
                           const Value &rhs, const Value &fill,
                           keel::Allocator *mr)
{
    if (!mr)
        mr = (isHeap() && heap_->alloc) ? heap_->alloc
                                        : (rhs.isHeap() && rhs.heap_->alloc
                                               ? rhs.heap_->alloc
                                               : keel::default_allocator());
    objectEnsureArrayLike(rhs, mr);
    detach();
    const bool handle = heap_->objIsHandle;
    auto sharedOf = [](const Value &v) -> keel::Arc<ObjectState> {
        return (v.isObject() && !v.heap_->objStates.empty()) ? v.heap_->objStates.front()
                                                             : keel::Arc<ObjectState>{};
    };
    auto indep = [&](const keel::Arc<ObjectState> &s) {
        return s ? s->deepCopy(mr) : s;
    };

    const int curNd = heap_->dims.ndim();
    const int nd = keel::max(static_cast<int>(perDim.size()), curNd);
    keel::SmallArray<size_t, Dims::kMaxRank> newSize(static_cast<size_t>(nd), mr);
    for (int d = 0; d < nd; ++d) {
        size_t cur = (d < curNd) ? heap_->dims.dim(d) : 1;
        size_t need = 1;
        if (d < static_cast<int>(perDim.size()))
            for (size_t v : perDim[static_cast<size_t>(d)])
                need = keel::max(need, v + 1);
        newSize[static_cast<size_t>(d)] = keel::max(cur, need);
    }
    Dims newDims(newSize.data(), nd);

    size_t ntargets = 1;
    for (const auto &l : perDim)
        ntargets *= l.size();
    const size_t rn = rhs.isObject() ? rhs.objectCount() : 0;
    const bool broadcast = (rn == 1);
    if (!broadcast && rn != ntargets)
        throw std::runtime_error(
            "Unable to perform assignment because the left and right sides have a "
            "different number of elements.");
    auto rhsSlot = [&](size_t k) -> keel::Arc<ObjectState> {
        const auto &sp = rhs.heap_->objStates[broadcast ? 0 : k];
        return handle ? sp : (sp ? sp->deepCopy(mr) : sp);
    };

    if (!(newDims == heap_->dims)) {
        // Re-layout into the grown shape: default-fill, then move each old
        // element from its old column-major slot to its new one.
        keel::Array<keel::Arc<ObjectState>> neu(newDims.numel(), heap_->alloc);
        keel::Arc<ObjectState> fillState = sharedOf(fill);
        for (auto &s : neu)
            s = indep(fillState);
        size_t oldStr[Dims::kMaxRank];
        size_t newStr[Dims::kMaxRank];
        computeStridesColMajor(heap_->dims, oldStr);
        computeStridesColMajor(newDims, newStr);
        const size_t oldN = heap_->objStates.size();
        for (size_t k = 0; k < oldN; ++k) {
            size_t newLin = 0;
            for (int d = 0; d < curNd; ++d)
                newLin += ((k / oldStr[d]) % heap_->dims.dim(d)) * newStr[d];
            neu[newLin] = keel::move(heap_->objStates[k]);
        }
        heap_->objStates = keel::move(neu);
        heap_->dims = newDims;
    }

    if (ntargets == 0)
        return;
    size_t str[Dims::kMaxRank];
    computeStridesColMajor(heap_->dims, str);
    const int pnd = static_cast<int>(perDim.size());
    keel::SmallArray<size_t, Dims::kMaxRank> counts(static_cast<size_t>(pnd), mr);
    for (int d = 0; d < pnd; ++d)
        counts[static_cast<size_t>(d)] = perDim[static_cast<size_t>(d)].size();
    Dims iter(counts.data(), pnd);
    keel::SmallArray<size_t, Dims::kMaxRank> coords(static_cast<size_t>(pnd), size_t{0}, mr);
    size_t k = 0;
    do {
        size_t lin = 0;
        for (int d = 0; d < pnd; ++d)
            lin += perDim[static_cast<size_t>(d)][coords[static_cast<size_t>(d)]] * str[d];
        heap_->objStates[lin] = rhsSlot(k++);
    } while (incrementCoords(coords.data(), iter));
}

const void *Value::rawData() const
{
    if (heap_ == nullptr)
        return &scalar_;
    if (heap_ == logicalTrueTag()) {
        static const uint8_t t = 1;
        return &t;
    }
    if (heap_ == logicalFalseTag()) {
        static const uint8_t f = 0;
        return &f;
    }
    if (isTag())
        return nullptr;
    return heap_->buffer ? heap_->buffer->data() : nullptr;
}
size_t Value::rawBytes() const
{
    if (heap_ == nullptr)
        return sizeof(double);
    if (heap_ == logicalTrueTag() || heap_ == logicalFalseTag())
        return sizeof(uint8_t);
    if (isTag())
        return 0;
    if (!heap_->buffer)
        return 0;
    const size_t es = elementSize(heap_->type);
    if (es > 0)
        return keel::min(heap_->dims.numel() * es, heap_->buffer->bytes());
    return heap_->buffer->bytes();
}

size_t Value::deepBytes() const
{
    // Structs/cells have no flat buffer (rawBytes() == 0), so their reported
    // size must be the sum of everything they contain, recursively, plus
    // their own schema — mirroring MATLAB's whos.
    if (isStruct()) {
        // Field names are the struct's own metadata, stored once for the
        // whole array.
        size_t total = 0;
        const auto order = fieldNamesInOrder();
        for (const auto &nm : order)
            total += nm.size();
        const size_t n = numel();
        const bool arr = isStructArray();
        for (size_t e = 0; e < n; ++e) {
            const auto &fm = arr ? structArrayElem(e) : structFields();
            for (const auto &nm : order) {
                if (const Value *vp = fm.get(nm))
                    total += vp->deepBytes();
            }
        }
        return total;
    }
    if (isCell()) {
        size_t total = 0;
        const size_t n = numel();
        for (size_t i = 0; i < n; ++i)
            total += cellAt(i).deepBytes();
        return total;
    }
    return rawBytes();
}

const double *Value::doubleData() const
{
    if (heap_ == nullptr)
        return &scalar_;
    if (isTag())
        throw std::runtime_error("Not a double array");
    if (heap_->type != ValueType::DOUBLE)
        throw std::runtime_error("Not a double array");
    return heap_->buffer ? static_cast<const double *>(heap_->buffer->data()) : nullptr;
}
const uint8_t *Value::logicalData() const
{
    if (heap_ == logicalTrueTag()) {
        static const uint8_t t = 1;
        return &t;
    }
    if (heap_ == logicalFalseTag()) {
        static const uint8_t f = 0;
        return &f;
    }
    if (!isHeap() || heap_->type != ValueType::LOGICAL)
        throw std::runtime_error("Not a logical array");
    return heap_->buffer ? static_cast<const uint8_t *>(heap_->buffer->data()) : nullptr;
}
const char *Value::charData() const
{
    if (!isHeap() || heap_->type != ValueType::CHAR)
        throw std::runtime_error("Not a char array");
    return heap_->buffer ? static_cast<const char *>(heap_->buffer->data()) : nullptr;
}
double Value::toScalar() const
{
    if (heap_ == nullptr)
        return scalar_;
    if (heap_ == logicalTrueTag())
        return 1.0;
    if (heap_ == logicalFalseTag())
        return 0.0;
    if (isHeap() && heap_->type == ValueType::OBJECT && heap_->dims.isScalar()) {
        const auto *st = objectStateConst();
        if (st && st->native)
            return st->native->asDouble();
    }
    if (!isHeap() || !heap_->buffer)
        throw std::runtime_error(
            keel::format_small<96>("Cannot convert {} to scalar", mtypeName(type())).c_str());
    auto *h = heap_;
    if (h->type == ValueType::DOUBLE && h->dims.isScalar())
        return *static_cast<const double *>(h->buffer->data());
    if (h->type == ValueType::COMPLEX && h->dims.isScalar()) {
        auto c = *static_cast<const keel::c64 *>(h->buffer->data());
        if (c.imag() != 0.0)
            throw std::runtime_error(
                "Cannot convert complex with nonzero imaginary part to double scalar");
        return c.real();
    }
    if (h->type == ValueType::LOGICAL && h->dims.isScalar())
        return (double) *static_cast<const uint8_t *>(h->buffer->data());
    if (h->type == ValueType::CHAR && h->dims.isScalar())
        return (double) (unsigned char) *static_cast<const char *>(h->buffer->data());
    if (h->type == ValueType::SINGLE && h->dims.isScalar())
        return (double) *static_cast<const float *>(h->buffer->data());
    if (h->type == ValueType::INT8 && h->dims.isScalar())
        return (double) *static_cast<const int8_t *>(h->buffer->data());
    if (h->type == ValueType::INT16 && h->dims.isScalar())
        return (double) *static_cast<const int16_t *>(h->buffer->data());
    if (h->type == ValueType::INT32 && h->dims.isScalar())
        return (double) *static_cast<const int32_t *>(h->buffer->data());
    if (h->type == ValueType::INT64 && h->dims.isScalar())
        return (double) *static_cast<const int64_t *>(h->buffer->data());
    if (h->type == ValueType::UINT8 && h->dims.isScalar())
        return (double) *static_cast<const uint8_t *>(h->buffer->data());
    if (h->type == ValueType::UINT16 && h->dims.isScalar())
        return (double) *static_cast<const uint16_t *>(h->buffer->data());
    if (h->type == ValueType::UINT32 && h->dims.isScalar())
        return (double) *static_cast<const uint32_t *>(h->buffer->data());
    if (h->type == ValueType::UINT64 && h->dims.isScalar())
        return (double) *static_cast<const uint64_t *>(h->buffer->data());
    if (h->type == ValueType::OBJECT && h->dims.isScalar()) {
        const auto *st = objectStateConst();
        if (st && st->native)
            return st->native->asDouble();
    }
    throw std::runtime_error(
        keel::format_small<96>("Cannot convert {} to scalar", mtypeName(type())).c_str());
}
bool Value::toBool() const
{
    // MATLAB: a NaN in a conditional ("if"/"while"/logical()) is an error,
    // "NaN values cannot be converted to logicals." — NOT silently true/false.
    auto nanErr = [] {
        throw std::runtime_error("NaN values cannot be converted to logicals.");
    };
    if (heap_ == nullptr) {
        if (keel::is_nan(scalar_))
            nanErr();
        return scalar_ != 0.0;
    }
    if (heap_ == logicalTrueTag())
        return true;
    if (heap_ == logicalFalseTag())
        return false;
    if (heap_ == emptyTag() || heap_ == deletedTag())
        throw std::runtime_error("Cannot convert empty to bool");
    auto *h = heap_;
    if (h->type == ValueType::LOGICAL && h->dims.isScalar() && h->buffer)
        return *static_cast<const uint8_t *>(h->buffer->data()) != 0;
    if (h->type == ValueType::DOUBLE && h->dims.isScalar() && h->buffer) {
        double v = *static_cast<const double *>(h->buffer->data());
        if (keel::is_nan(v))
            nanErr();
        return v != 0.0;
    }
    if (h->type == ValueType::COMPLEX && h->dims.isScalar() && h->buffer) {
        auto c = *static_cast<const keel::c64 *>(h->buffer->data());
        if (keel::is_nan(c.real()) || keel::is_nan(c.imag()))
            nanErr();
        return c.real() != 0.0 || c.imag() != 0.0;
    }
    if (h->type == ValueType::DOUBLE && h->buffer) {
        const double *dd = static_cast<const double *>(h->buffer->data());
        size_t n = h->dims.numel();
        bool hasZero = false;
        for (size_t i = 0; i < n; ++i) {
            if (keel::is_nan(dd[i])) // scan all — `if [0 NaN]` errors in MATLAB
                nanErr();
            if (dd[i] == 0.0)
                hasZero = true;
        }
        return n > 0 && !hasZero;
    }
    if (h->type == ValueType::LOGICAL && h->buffer) {
        const uint8_t *ld = static_cast<const uint8_t *>(h->buffer->data());
        size_t n = h->dims.numel();
        for (size_t i = 0; i < n; ++i)
            if (!ld[i])
                return false;
        return n > 0;
    }
    if (h->type == ValueType::COMPLEX && h->buffer) {
        const keel::c64 *cd = static_cast<const keel::c64 *>(h->buffer->data());
        size_t n = h->dims.numel();
        bool hasZero = false;
        for (size_t i = 0; i < n; ++i) {
            if (keel::is_nan(cd[i].real()) || keel::is_nan(cd[i].imag()))
                nanErr();
            if (cd[i].real() == 0.0 && cd[i].imag() == 0.0)
                hasZero = true;
        }
        return n > 0 && !hasZero;
    }
    // Any other numeric type (single / int* / uint* / char): scalar → nonzero,
    // array → ALL elements nonzero, matching MATLAB's `if`/`while`
    // scalar-context truth (same rule as the DOUBLE path above). Without
    // this, `if uint8(...)`, `if single(...)`, even `if int8(5)` threw.
    if (h->buffer
        && (isIntegerType(h->type) || h->type == ValueType::SINGLE
            || h->type == ValueType::CHAR)) {
        size_t n = h->dims.numel();
        bool hasZero = false;
        for (size_t i = 0; i < n; ++i) {
            double v = elemAsDouble(i);
            if (keel::is_nan(v))
                nanErr();
            if (v == 0.0)
                hasZero = true;
        }
        return n > 0 && !hasZero;
    }
    throw std::runtime_error(
        keel::format_small<96>("Cannot convert {} to bool", mtypeName(type())).c_str());
}
keel::String Value::toString() const
{
    if (isHeap() && heap_->type == ValueType::CHAR) {
        if (heap_->buffer)
            return keel::String(keel::StringRef(static_cast<const char *>(heap_->buffer->data()),
                                                heap_->dims.numel()));
        return {}; // empty string (1x0 char)
    }
    if (isHeap() && heap_->type == ValueType::STRING) {
        if (heap_->cellData && !heap_->cellData->empty())
            return (*heap_->cellData)[0].toString();
        return {};
    }
    if (isHeap() && heap_->type == ValueType::FUNC_HANDLE && heap_->funcName)
        return heap_->funcName->clone();
    throw std::runtime_error("Not a char array");
}
keel::String Value::funcHandleName() const
{
    if (isHeap() && heap_->funcName)
        return heap_->funcName->clone();
    return {};
}
keel::String Value::funcHandleSource() const
{
    if (isHeap() && heap_->funcSource)
        return heap_->funcSource->clone();
    return {};
}
void Value::setFuncHandleSource(keel::StringRef source)
{
    if (!isHeap()) return;
    detach();
    heap_->allocDelete(heap_->funcSource);
    heap_->funcSource = heap_->allocNew<keel::String>(source, heap_->alloc);
}
size_t Value::funcHandleCaptureCount() const
{
    if (isHeap() && heap_->type == ValueType::FUNC_HANDLE && heap_->cellData)
        return heap_->cellData->size();
    return 0;
}
const Value &Value::funcHandleCapture(size_t i) const
{
    if (!isHeap() || heap_->type != ValueType::FUNC_HANDLE || !heap_->cellData
        || i >= heap_->cellData->size())
        throw std::runtime_error("funcHandleCapture: index out of range");
    return (*heap_->cellData)[i];
}
void Value::setFuncHandleCaptures(const Value *captures, size_t count)
{
    if (!isHeap() || heap_->type != ValueType::FUNC_HANDLE)
        return;
    detach();
    heap_->allocDelete(heap_->cellData);
    if (count > 0 && captures) {
        heap_->cellData = heap_->allocNew<keel::Array<Value>>(
            keel::ArrayRef<Value>(captures, count), heap_->alloc);
    }
}

const keel::c64 *Value::complexData() const
{
    if (!isHeap() || heap_->type != ValueType::COMPLEX)
        throw std::runtime_error("Not a complex array");
    return heap_->buffer ? static_cast<const keel::c64 *>(heap_->buffer->data()) : nullptr;
}
keel::c64 Value::toComplex() const
{
    if (heap_ == nullptr)
        return keel::c64(scalar_, 0.0);
    if (heap_ == logicalTrueTag())
        return keel::c64(1.0, 0.0);
    if (heap_ == logicalFalseTag())
        return keel::c64(0.0, 0.0);
    if (!isHeap() || !heap_->buffer)
        throw std::runtime_error("Cannot convert to complex");
    if (heap_->type == ValueType::COMPLEX && heap_->dims.isScalar())
        return *static_cast<const keel::c64 *>(heap_->buffer->data());
    if (heap_->dims.isScalar() && (isNumeric() || isLogical() || isChar()))
        return keel::c64(toScalar(), 0.0);
    throw std::runtime_error("Cannot convert to complex");
}
keel::c64 Value::complexElem(size_t i) const
{
    if (i >= numel())
        throw std::runtime_error("keel::c64 index out of bounds");
    return complexData()[i];
}
keel::c64 Value::complexElem(size_t r, size_t c) const
{
    return complexData()[dims().sub2indChecked(r, c)];
}

double *Value::doubleDataMut()
{
    if (heap_ == nullptr)
        return &scalar_;
    if (!isHeap() || heap_->type != ValueType::DOUBLE)
        throw std::runtime_error("Not a double array");
    detach();
    return heap_->buffer ? static_cast<double *>(heap_->buffer->data()) : nullptr;
}
uint8_t *Value::logicalDataMut()
{
    if (!isHeap() || heap_->type != ValueType::LOGICAL)
        throw std::runtime_error("Not a logical array");
    detach();
    return heap_->buffer ? static_cast<uint8_t *>(heap_->buffer->data()) : nullptr;
}
char *Value::charDataMut()
{
    if (!isHeap() || heap_->type != ValueType::CHAR)
        throw std::runtime_error("Not a char array");
    detach();
    return heap_->buffer ? static_cast<char *>(heap_->buffer->data()) : nullptr;
}
void *Value::rawDataMut()
{
    if (heap_ == nullptr)
        return &scalar_;
    if (!isHeap())
        return nullptr;
    detach();
    return heap_->buffer ? heap_->buffer->data() : nullptr;
}
keel::c64 *Value::complexDataMut()
{
    if (!isHeap() || heap_->type != ValueType::COMPLEX)
        throw std::runtime_error("Not a complex array");
    detach();
    return heap_->buffer ? static_cast<keel::c64 *>(heap_->buffer->data()) : nullptr;
}

#define NUMKIT_DEFINE_TYPED_DATA_ACCESSORS(CType, ConstFn, MutFn, VType, ErrLabel) \
    const CType *Value::ConstFn() const                                             \
    {                                                                               \
        if (!isHeap() || heap_->type != ValueType::VType)                           \
            throw std::runtime_error("Not a " ErrLabel " array");                   \
        return heap_->buffer ? static_cast<const CType *>(heap_->buffer->data())    \
                             : nullptr;                                             \
    }                                                                               \
    CType *Value::MutFn()                                                           \
    {                                                                               \
        if (!isHeap() || heap_->type != ValueType::VType)                           \
            throw std::runtime_error("Not a " ErrLabel " array");                   \
        detach();                                                                   \
        return heap_->buffer ? static_cast<CType *>(heap_->buffer->data())          \
                             : nullptr;                                             \
    }

NUMKIT_DEFINE_TYPED_DATA_ACCESSORS(float,    singleData, singleDataMut, SINGLE, "single")
NUMKIT_DEFINE_TYPED_DATA_ACCESSORS(int8_t,   int8Data,   int8DataMut,   INT8,   "int8")
NUMKIT_DEFINE_TYPED_DATA_ACCESSORS(int16_t,  int16Data,  int16DataMut,  INT16,  "int16")
NUMKIT_DEFINE_TYPED_DATA_ACCESSORS(int32_t,  int32Data,  int32DataMut,  INT32,  "int32")
NUMKIT_DEFINE_TYPED_DATA_ACCESSORS(int64_t,  int64Data,  int64DataMut,  INT64,  "int64")
NUMKIT_DEFINE_TYPED_DATA_ACCESSORS(uint8_t,  uint8Data,  uint8DataMut,  UINT8,  "uint8")
NUMKIT_DEFINE_TYPED_DATA_ACCESSORS(uint16_t, uint16Data, uint16DataMut, UINT16, "uint16")
NUMKIT_DEFINE_TYPED_DATA_ACCESSORS(uint32_t, uint32Data, uint32DataMut, UINT32, "uint32")
NUMKIT_DEFINE_TYPED_DATA_ACCESSORS(uint64_t, uint64Data, uint64DataMut, UINT64, "uint64")

#undef NUMKIT_DEFINE_TYPED_DATA_ACCESSORS

void Value::promoteToComplex(keel::Allocator *mr)
{
    ValueType t = type();
    if (t == ValueType::COMPLEX)
        return;
    if (t != ValueType::DOUBLE && t != ValueType::SINGLE)
        throw std::runtime_error("Can only promote double or single to complex");
    size_t n = numel();
    if (heap_ == nullptr) {
        double v = scalar_;
        *this = complexScalar(keel::c64(v, 0.0), mr);
        return;
    }
    if (!isHeap())
        throw std::runtime_error("Cannot promote to complex");
    detach(); // COW: ensure we have our own copy before mutating
    if (!mr)
        mr = heap_->alloc;
    auto *newBuf = DataBuffer::create(n * sizeof(keel::c64), mr);
    keel::c64 *dst = static_cast<keel::c64 *>(newBuf->data());
    if (n > 0 && heap_->buffer) {
        if (t == ValueType::SINGLE) {
            const float *src = static_cast<const float *>(heap_->buffer->data());
            for (size_t i = 0; i < n; ++i)
                dst[i] = keel::c64(static_cast<double>(src[i]), 0.0);
        } else {
            const double *src = static_cast<const double *>(heap_->buffer->data());
            for (size_t i = 0; i < n; ++i)
                dst[i] = keel::c64(src[i], 0.0);
        }
    }
    if (heap_->buffer && heap_->buffer->release())
        DataBuffer::destroy(heap_->buffer);
    heap_->buffer = newBuf;
    heap_->type = ValueType::COMPLEX;
    if (t == ValueType::SINGLE)
        heap_->isSingleComplex = true;
}

double Value::operator()(size_t i) const
{
    if (i >= numel())
        throw std::runtime_error("Index out of bounds");
    return doubleData()[i];
}
double Value::operator()(size_t r, size_t c) const
{
    return doubleData()[dims().sub2indChecked(r, c)];
}
double Value::operator()(size_t r, size_t c, size_t p) const
{
    return doubleData()[dims().sub2indChecked(r, c, p)];
}
double &Value::elem(size_t i)
{
    if (i >= numel())
        throw std::runtime_error("Index out of bounds");
    return doubleDataMut()[i];
}
double &Value::elem(size_t r, size_t c)
{
    return doubleDataMut()[dims().sub2indChecked(r, c)];
}
double &Value::elem(size_t r, size_t c, size_t p)
{
    return doubleDataMut()[dims().sub2indChecked(r, c, p)];
}

char Value::charElem(size_t i) const
{
    if (i >= numel())
        throw std::runtime_error("Char index out of bounds");
    return charData()[i];
}
char &Value::charElemMut(size_t i)
{
    if (i >= numel())
        throw std::runtime_error("Char index out of bounds");
    return charDataMut()[i];
}

keel::String Value::charRow(size_t r) const
{
    if (type() != ValueType::CHAR)
        throw std::runtime_error("charRow: value is not a char array");
    auto &d = dims();
    if (r >= d.rows())
        throw std::runtime_error("charRow: row index out of bounds");
    const char *cd = charData();
    size_t R = d.rows(), C = d.cols();
    keel::String s = keel::String::with_capacity(C);
    for (size_t c = 0; c < C; ++c)
        s.push_back(cd[c * R + r]);
    return s;
}

void Value::resize(size_t newRows, size_t newCols, keel::Allocator *mr)
{
    if (heap_ == nullptr) {
        double v = scalar_;
        HeapGuard h(HeapObject::create(mr));
        h->type = ValueType::DOUBLE;
        h->dims = {1, 1};
        h->buffer = DataBuffer::create(sizeof(double), h->alloc);
        *static_cast<double *>(h->buffer->data()) = v;
        heap_ = h.release();
    }
    if (heap_ == emptyTag() || heap_ == deletedTag()) {
        *this = matrix(newRows, newCols, ValueType::DOUBLE, mr);
        return;
    }
    if (!isHeap()) {
        bool lv = isLogicalTrue();
        *this = matrix(1, 1, ValueType::LOGICAL, mr);
        *logicalDataMut() = lv ? 1 : 0;
    }
    if (heap_->dims.ndim() > 3) {
        size_t ndBuf[Dims::kMaxRank];
        const int nd = heap_->dims.ndim();
        ndBuf[0] = newRows;
        ndBuf[1] = newCols;
        for (int k = 2; k < nd; ++k)
            ndBuf[k] = heap_->dims.dim(k);
        resizeND(ndBuf, nd, mr);
        return;
    }
    if (heap_->dims.is3D()) {
        resize3d(newRows, newCols, heap_->dims.pages(), mr);
        return;
    }
    detach();
    if (!mr)
        mr = heap_->alloc;
    const size_t oldR = heap_->dims.rows(), oldC = heap_->dims.cols();
    const size_t cr = keel::min(oldR, newRows), cc = keel::min(oldC, newCols);
    if (heap_->type == ValueType::CELL || heap_->type == ValueType::STRING) {
        Value grown = (heap_->type == ValueType::CELL)
                          ? Value::cell(newRows, newCols, mr)
                          : Value::stringArray(newRows, newCols, mr);
        if (heap_->cellData) {
            auto &dst = *grown.heap_->cellData;
            const auto &src = *heap_->cellData;
            for (size_t c = 0; c < cc; ++c)
                for (size_t r = 0; r < cr; ++r)
                    dst[c * newRows + r] = src[c * oldR + r];
        }
        *this = keel::move(grown);
        return;
    }
    if (heap_->type == ValueType::STRUCT) {
        Value grown = Value::structArray(newRows, newCols, mr);
        if (heap_->fieldOrder && grown.heap_->fieldOrder)
            *grown.heap_->fieldOrder = heap_->fieldOrder->clone(grown.heap_->alloc);
        if (heap_->structArray) {
            auto &dst = *grown.heap_->structArray;
            const auto &src = *heap_->structArray;
            auto *cmr = grown.heap_->alloc;
            for (size_t c = 0; c < cc; ++c)
                for (size_t r = 0; r < cr; ++r)
                    dst[c * newRows + r] = src[c * oldR + r].clone(cmr);
            if (heap_->fieldOrder) {
                for (size_t i = 0; i < dst.size(); ++i)
                    for (const auto &fn : *heap_->fieldOrder)
                        if (!dst[i].contains(fn))
                            dst[i].put(fn.clone(cmr), Value::matrix(0, 0, ValueType::DOUBLE, cmr));
            }
        }
        *this = keel::move(grown);
        return;
    }
    const size_t es = elementSize(heap_->type);
    if (es == 0)
        throw std::runtime_error(
            keel::format_small<96>("Cannot resize array of type '{}'", mtypeName(heap_->type)).c_str());
    const size_t nb = checkedMulSize(checkedMulSize(newRows, newCols), es);
    auto *nb2 = (nb > 0) ? DataBuffer::create(nb, mr) : nullptr;
    if (nb > 0) {
        // CHAR arrays fill with spaces; everything else with zeros
        keel::u8 fill = (heap_->type == ValueType::CHAR) ? static_cast<keel::u8>(' ') : keel::u8{0};
        keel::mem_set(nb2->data(), fill, nb);
    }
    if (heap_->buffer && nb2 && cr > 0 && cc > 0) {
        const char *s = static_cast<const char *>(heap_->buffer->data());
        char *d = static_cast<char *>(nb2->data());
        for (size_t c = 0; c < cc; ++c)
            keel::mem_copy(d + c * newRows * es, s + c * oldR * es, cr * es);
    }
    if (heap_->buffer && heap_->buffer->release())
        DataBuffer::destroy(heap_->buffer);
    heap_->buffer = nb2;
    heap_->dims = {newRows, newCols};
    heap_->appendCapacity = 0;
}

void Value::resize3d(size_t nr, size_t nc, size_t np, keel::Allocator *mr)
{
    if (np <= 1 && (!isHeap() || heap_->dims.ndim() <= 2)) {
        resize(nr, nc, mr);
        return;
    }
    if (heap_ == nullptr) {
        double v = scalar_;
        HeapGuard h(HeapObject::create(mr));
        h->type = ValueType::DOUBLE;
        h->dims = {1, 1};
        h->buffer = DataBuffer::create(sizeof(double), h->alloc);
        *static_cast<double *>(h->buffer->data()) = v;
        heap_ = h.release();
    }
    if (heap_ == emptyTag() || heap_ == deletedTag()) {
        *this = matrix3d(nr, nc, np, ValueType::DOUBLE, mr);
        return;
    }
    if (!isHeap()) {
        bool lv = isLogicalTrue();
        *this = matrix(1, 1, ValueType::LOGICAL, mr);
        *logicalDataMut() = lv ? 1 : 0;
    }
    detach();
    if (!mr)
        mr = heap_->alloc;
    const size_t oR = heap_->dims.rows(), oC = heap_->dims.cols(),
                 oP = heap_->dims.dim(2);
    const size_t cr = keel::min(oR, nr), cc = keel::min(oC, nc), cp = keel::min(oP, np);
    if (heap_->type == ValueType::CELL || heap_->type == ValueType::STRING) {
        Value grown = (heap_->type == ValueType::CELL)
                          ? Value::cell3D(nr, nc, np, mr)
                          : Value::stringArray3D(nr, nc, np, mr);
        if (heap_->cellData) {
            auto &dst = *grown.heap_->cellData;
            const auto &src = *heap_->cellData;
            for (size_t p = 0; p < cp; ++p)
                for (size_t c = 0; c < cc; ++c)
                    for (size_t r = 0; r < cr; ++r)
                        dst[p * nr * nc + c * nr + r] = src[p * oR * oC + c * oR + r];
        }
        *this = keel::move(grown);
        return;
    }
    const size_t es = elementSize(heap_->type);
    if (es == 0)
        throw std::runtime_error(
            keel::format_small<96>("Cannot resize array of type '{}'", mtypeName(heap_->type)).c_str());
    const size_t nb = checkedMulSize(checkedMulSize(checkedMulSize(nr, nc), np), es);
    auto *nb2 = (nb > 0) ? DataBuffer::create(nb, mr) : nullptr;
    if (nb > 0) {
        keel::u8 fill = (heap_->type == ValueType::CHAR) ? static_cast<keel::u8>(' ') : keel::u8{0};
        keel::mem_set(nb2->data(), fill, nb);
    }
    if (heap_->buffer && nb2 && cr > 0 && cc > 0 && cp > 0) {
        const char *s = static_cast<const char *>(heap_->buffer->data());
        char *d = static_cast<char *>(nb2->data());
        for (size_t p = 0; p < cp; ++p)
            for (size_t c = 0; c < cc; ++c)
                keel::mem_copy(d + (p * nr * nc + c * nr) * es,
                               s + (p * oR * oC + c * oR) * es,
                               cr * es);
    }
    if (heap_->buffer && heap_->buffer->release())
        DataBuffer::destroy(heap_->buffer);
    heap_->buffer = nb2;
    heap_->dims = (np <= 1) ? Dims{nr, nc} : Dims{nr, nc, np};
    heap_->appendCapacity = 0;
}

void Value::resizeND(const size_t *newDims, int nd, keel::Allocator *mr)
{
    constexpr int kMaxNd = Dims::kMaxRank;
    if (nd <= 0 || nd > kMaxNd)
        throw std::runtime_error("resizeND: rank out of range");
    if (nd == 1) { resize(newDims[0], 1, mr); return; }
    if (nd == 2) { resize(newDims[0], newDims[1], mr); return; }
    if (nd == 3) { resize3d(newDims[0], newDims[1], newDims[2], mr); return; }

    // Promote scalar (no heap_) to a 1×1 DOUBLE array so the rest can
    // assume heap_ is set and detach() is meaningful.
    if (heap_ == nullptr) {
        double v = scalar_;
        HeapGuard h(HeapObject::create(mr));
        h->type = ValueType::DOUBLE;
        h->dims = {1, 1};
        h->buffer = DataBuffer::create(sizeof(double), h->alloc);
        *static_cast<double *>(h->buffer->data()) = v;
        heap_ = h.release();
    }
    // Empty (tag) → fresh DOUBLE heap with the requested shape; no
    // existing data to preserve. Same path is used for the auto-grow
    // case `A(i,j,k,l) = v` when A doesn't exist yet.
    if (heap_ == emptyTag() || heap_ == deletedTag()) {
        *this = matrixND(newDims, nd, ValueType::DOUBLE, mr);
        return;
    }
    if (!isHeap()) {
        bool lv = isLogicalTrue();
        *this = matrix(1, 1, ValueType::LOGICAL, mr);
        *logicalDataMut() = lv ? 1 : 0;
    }
    detach();
    if (!mr) mr = heap_->alloc;

    const ValueType t = heap_->type;
    const Dims oldDims = heap_->dims;
    const int oldNd = oldDims.ndim();

    const size_t newTotal = checkedMulDims(newDims, nd);

    size_t oldStrides[kMaxNd], newStrides[kMaxNd];
    computeStridesColMajor(oldDims, oldStrides);
    Dims newD(newDims, nd);
    computeStridesColMajor(newD, newStrides);

    const int commonNd = keel::min(nd, oldNd);
    size_t commonDims[kMaxNd];
    size_t commonTotal = 1;
    for (int i = 0; i < commonNd; ++i) {
        commonDims[i] = keel::min(oldDims.dim(i), newDims[i]);
        commonTotal *= commonDims[i];
    }

    if (t == ValueType::CELL || t == ValueType::STRING) {
        auto newArr = (t == ValueType::CELL)
                          ? Value::cellND(newDims, nd, mr)
                          : Value::stringArrayND(newDims, nd, mr);
        if (commonTotal > 0 && heap_->cellData) {
            auto &dst = *newArr.heap_->cellData;
            const auto &src = *heap_->cellData;
            Dims commonD(commonDims, commonNd);
            size_t coords[kMaxNd] = {0};
            do {
                size_t oldOff = 0, newOff = 0;
                for (int i = 0; i < commonNd; ++i) {
                    oldOff += coords[i] * oldStrides[i];
                    newOff += coords[i] * newStrides[i];
                }
                dst[newOff] = src[oldOff];
            } while (incrementCoords(coords, commonD));
        }
        *this = keel::move(newArr);
        return;
    }

    const size_t es = elementSize(t);
    if (es == 0)
        throw std::runtime_error(
            keel::format_small<96>("resizeND not supported for type '{}'", mtypeName(t)).c_str());

    const size_t nb = checkedMulSize(newTotal, es);
    auto *nb2 = (nb > 0) ? DataBuffer::create(nb, mr) : nullptr;
    if (nb > 0) {
        const keel::u8 fill = (t == ValueType::CHAR) ? static_cast<keel::u8>(' ') : keel::u8{0};
        keel::mem_set(nb2->data(), fill, nb);
    }
    if (heap_->buffer && nb2 && commonTotal > 0) {
        const char *src = static_cast<const char *>(heap_->buffer->data());
        char *dst = static_cast<char *>(nb2->data());
        Dims commonD(commonDims, commonNd);
        size_t coords[kMaxNd] = {0};
        do {
            size_t oldOff = 0, newOff = 0;
            for (int i = 0; i < commonNd; ++i) {
                oldOff += coords[i] * oldStrides[i];
                newOff += coords[i] * newStrides[i];
            }
            keel::mem_copy(dst + newOff * es, src + oldOff * es, es);
        } while (incrementCoords(coords, commonD));
    }
    if (heap_->buffer && heap_->buffer->release())
        DataBuffer::destroy(heap_->buffer);
    heap_->buffer = nb2;
    heap_->dims = newD;
    heap_->appendCapacity = 0;
}

void Value::ensureSize(size_t idx, keel::Allocator *mr)
{
    if (heap_ == emptyTag() || heap_ == deletedTag() || (heap_ == nullptr && idx > 0)) {
        double old = (heap_ == nullptr) ? scalar_ : 0.0;
        *this = matrix(1, idx + 1, ValueType::DOUBLE, mr);
        if (old != 0.0)
            static_cast<double *>(heap_->buffer->data())[0] = old;
        return;
    }
    size_t need = idx + 1;
    if (need > numel()) {
        bool isColVec = (dims().cols() == 1 && dims().rows() > 1);
        if (isColVec)
            resize(need, 1, mr); // preserve column vector shape
        else if (dims().isVector() || dims().rows() <= 1)
            resize(1, need, mr);
        else
            throw std::runtime_error("Index exceeds array dimensions");
    }
}

void Value::appendScalar(double v, keel::Allocator *mr)
{
    size_t oldN = numel(), newN = oldN + 1;

    // Empty → fresh 1-element heap double with headroom.
    if (isEmpty()) {
        if (!mr && isHeap())
            mr = heap_->alloc;
        releaseHeap();
        size_t cap = 8;
        HeapGuard h(HeapObject::create(mr));
        h->type = ValueType::DOUBLE;
        h->dims = {1, 1};
        h->buffer = DataBuffer::create(cap * sizeof(double), h->alloc);
        h->appendCapacity = cap;
        double *d = static_cast<double *>(h->buffer->data());
        keel::mem_zero(d, cap * sizeof(double));
        d[0] = v;
        heap_ = h.release();
        return;
    }

    // Inline scalar → promote to a 2-element heap double with headroom.
    if (heap_ == nullptr) {
        double old = scalar_;
        size_t cap = keel::max(size_t(8), newN * 2);
        HeapGuard h(HeapObject::create(mr));
        h->type = ValueType::DOUBLE;
        h->dims = {1, newN};
        h->buffer = DataBuffer::create(cap * sizeof(double), h->alloc);
        h->appendCapacity = cap;
        double *d = static_cast<double *>(h->buffer->data());
        keel::mem_zero(d, cap * sizeof(double));
        d[0] = old;
        d[1] = v;
        heap_ = h.release();
        return;
    }
    if (!isHeap())
        throw std::runtime_error("Cannot append");
    detach();
    if (!mr)
        mr = heap_->alloc;
    size_t cap = heap_->appendCapacity;
    if (!cap && heap_->buffer)
        cap = heap_->buffer->bytes() / sizeof(double);
    if (newN <= cap && heap_->buffer) {
        static_cast<double *>(heap_->buffer->data())[oldN] = v;
        heap_->dims = {1, newN};
        return;
    }
    size_t nc = keel::max(newN, cap * 2);
    if (nc < 8)
        nc = 8;
    auto *nb = DataBuffer::create(nc * sizeof(double), mr);
    double *d = static_cast<double *>(nb->data());
    keel::mem_zero(d, nc * sizeof(double));
    if (oldN > 0 && heap_->buffer)
        keel::mem_copy(d, heap_->buffer->data(), oldN * sizeof(double));
    d[oldN] = v;
    if (heap_->buffer && heap_->buffer->release())
        DataBuffer::destroy(heap_->buffer);
    heap_->buffer = nb;
    heap_->dims = {1, newN};
    heap_->appendCapacity = nc;
}

void Value::appendScalarCol(double v, keel::Allocator *mr)
{
    // Column twin of appendScalar: same buffer/growth mechanics, but the
    // dims stay {newN, 1} — a column vector grows DOWN, not right.
    size_t oldN = numel(), newN = oldN + 1;

    if (isEmpty()) {
        if (!mr && isHeap())
            mr = heap_->alloc;
        releaseHeap();
        size_t cap = 8;
        HeapGuard h(HeapObject::create(mr));
        h->type = ValueType::DOUBLE;
        h->dims = {newN, 1};
        h->buffer = DataBuffer::create(cap * sizeof(double), h->alloc);
        h->appendCapacity = cap;
        double *d = static_cast<double *>(h->buffer->data());
        keel::mem_zero(d, cap * sizeof(double));
        d[0] = v;
        heap_ = h.release();
        return;
    }
    if (heap_ == nullptr) {
        double old = scalar_;
        size_t cap = keel::max(size_t(8), newN * 2);
        HeapGuard h(HeapObject::create(mr));
        h->type = ValueType::DOUBLE;
        h->dims = {newN, 1};
        h->buffer = DataBuffer::create(cap * sizeof(double), h->alloc);
        h->appendCapacity = cap;
        double *d = static_cast<double *>(h->buffer->data());
        keel::mem_zero(d, cap * sizeof(double));
        d[0] = old;
        d[1] = v;
        heap_ = h.release();
        return;
    }
    if (!isHeap())
        throw std::runtime_error("Cannot append");
    detach();
    if (!mr)
        mr = heap_->alloc;
    size_t cap = heap_->appendCapacity;
    if (!cap && heap_->buffer)
        cap = heap_->buffer->bytes() / sizeof(double);
    if (newN <= cap && heap_->buffer) {
        static_cast<double *>(heap_->buffer->data())[oldN] = v;
        heap_->dims = {newN, 1};
        return;
    }
    size_t nc = keel::max(newN, cap * 2);
    if (nc < 8)
        nc = 8;
    auto *nb = DataBuffer::create(nc * sizeof(double), mr);
    double *d = static_cast<double *>(nb->data());
    keel::mem_zero(d, nc * sizeof(double));
    if (oldN > 0 && heap_->buffer)
        keel::mem_copy(d, heap_->buffer->data(), oldN * sizeof(double));
    d[oldN] = v;
    if (heap_->buffer && heap_->buffer->release())
        DataBuffer::destroy(heap_->buffer);
    heap_->buffer = nb;
    heap_->appendCapacity = nc;
    heap_->dims = {newN, 1};
}

Value &Value::cellAt(size_t i)
{
    if (!isHeap() || !heap_->cellData)
        throw std::runtime_error("Not a cell");
    if (i >= heap_->cellData->size())
        throw std::runtime_error("Cell index out of bounds");
    detach(); // COW: ensure we have our own copy before mutation
    return (*heap_->cellData)[i];
}
const Value &Value::cellAt(size_t i) const
{
    if (!isHeap() || !heap_->cellData)
        throw std::runtime_error("Not a cell");
    if (i >= heap_->cellData->size())
        throw std::runtime_error("Cell index out of bounds");
    return (*heap_->cellData)[i];
}
keel::Array<Value> &Value::cellDataVec()
{
    if (!isHeap() || !heap_->cellData)
        throw std::runtime_error("Not a cell");
    detach(); // COW
    return *heap_->cellData;
}
const keel::Array<Value> &Value::cellDataVec() const
{
    if (!isHeap() || !heap_->cellData)
        throw std::runtime_error("Not a cell");
    return *heap_->cellData;
}

Value &Value::field(keel::StringRef n)
{
    if (!isHeap() || heap_->type != ValueType::STRUCT || !heap_->structArray)
        throw std::runtime_error("Not a struct");
    if (heap_->structArray->size() != 1)
        throw std::runtime_error(
            keel::format_small<192>(
                "Cannot dot-index a non-scalar struct array directly; "
                "use s(i).{} or arrayfun.", n).c_str());
    detach();
    // BUG #15: append to insertion-order tracker BEFORE auto-create.
    auto &m = (*heap_->structArray)[0];
    if (Value *vp = m.get(n))
        return *vp;
    if (heap_->fieldOrder)
        heap_->fieldOrder->push_back(keel::String(n, heap_->alloc));
    return *m.try_insert(keel::String(n, heap_->alloc), Value()).value;
}
const Value &Value::field(keel::StringRef n) const
{
    if (!isHeap() || heap_->type != ValueType::STRUCT || !heap_->structArray)
        throw std::runtime_error("Not a struct");
    if (heap_->structArray->size() != 1)
        throw std::runtime_error(
            keel::format_small<192>(
                "Cannot dot-index a non-scalar struct array directly; "
                "use s(i).{} or arrayfun.", n).c_str());
    const auto &m = (*heap_->structArray)[0];
    if (const Value *vp = m.get(n))
        return *vp;
    throw std::runtime_error(keel::format_small<128>("Field not found: {}", n).c_str());
}
bool Value::hasField(keel::StringRef n) const
{
    if (!isHeap() || heap_->type != ValueType::STRUCT || !heap_->structArray)
        return false;
    if (heap_->structArray->empty()) {
        if (heap_->fieldOrder) {
            for (const auto &f : *heap_->fieldOrder)
                if (f == n) return true;
        }
        return false;
    }
    return (*heap_->structArray)[0].contains(n);
}
keel::HashMap<keel::String, Value> &Value::structFields()
{
    if (!isHeap() || heap_->type != ValueType::STRUCT || !heap_->structArray
        || heap_->structArray->size() != 1)
        throw std::runtime_error("structFields() is only valid for a 1×1 struct");
    detach();
    return (*heap_->structArray)[0];
}
const keel::HashMap<keel::String, Value> &Value::structFields() const
{
    if (!isHeap() || heap_->type != ValueType::STRUCT || !heap_->structArray
        || heap_->structArray->size() != 1)
        throw std::runtime_error("structFields() is only valid for a 1×1 struct");
    return (*heap_->structArray)[0];
}

bool Value::isStructArray() const
{
    // Multi-element struct array (numel > 1). Single struct (numel == 1)
    // returns false to preserve existing call-site semantics that
    // distinguish "scalar struct" from "array".
    return isHeap() && heap_->type == ValueType::STRUCT && heap_->structArray
           && heap_->structArray->size() > 1;
}

void Value::growStructArrayTo(size_t idx, keel::Allocator *mr)
{
    if (!mr) {
        mr = (isHeap() && heap_->alloc) ? heap_->alloc
                                        : keel::default_allocator();
    }
    if (isEmpty() || !isHeap() || heap_->type != ValueType::STRUCT
        || !heap_->structArray) {
        *this = Value::structArray(0, 0, mr);
    }
    if (idx < numel())
        return;

    const size_t newSize = idx + 1;
    const bool isCol = (dims().cols() == 1 && dims().rows() > 1);
    const size_t newRows = isCol ? newSize : 1;
    const size_t newCols = isCol ? 1       : newSize;

    detach();
    const size_t oldSize = heap_->structArray->size();
    heap_->dims = Dims(newRows, newCols);
    if (!heap_->fieldOrder)
        heap_->fieldOrder = heap_->allocNew<keel::Array<keel::String>>(heap_->alloc);
    if (newSize > heap_->structArray->capacity())
        heap_->structArray->reserve(keel::max(newSize, heap_->structArray->capacity() * 2));
    for (size_t k = oldSize; k < newSize; ++k) {
        heap_->structArray->push_back(keel::HashMap<keel::String, Value>(heap_->alloc));
        auto &dst = heap_->structArray->back();
        for (const auto &fn : *heap_->fieldOrder)
            dst.try_insert(fn.clone(heap_->alloc), Value());
    }
}
size_t Value::growStructArrayND(const size_t *coords, int nd,
                                keel::Allocator *mr)
{
    if (nd <= 0 || nd > Dims::kMaxRank)
        throw std::runtime_error("struct-array subscript rank out of range");
    if (!mr) {
        mr = (isHeap() && heap_->alloc) ? heap_->alloc
                                        : keel::default_allocator();
    }
    if (isEmpty() || !isHeap() || heap_->type != ValueType::STRUCT
        || !heap_->structArray) {
        *this = Value::structArray(0, 0, mr);
    }

    const Dims oldD = dims();
    const int oldNd = oldD.ndim();
    const int newNd = keel::max(nd, oldNd);

    size_t need[Dims::kMaxRank];
    for (int i = 0; i < newNd; ++i)
        need[i] = (i < oldNd) ? oldD.dim(i) : 1;
    bool grow = (newNd != oldNd);
    for (int i = 0; i < nd; ++i)
        if (coords[i] + 1 > need[i]) {
            need[i] = coords[i] + 1;
            grow = true;
        }

    // Column-major linear index of `coords` within the (grown) shape.
    auto linearIn = [&](const size_t *dimsArr) {
        size_t lin = 0, stride = 1;
        for (int i = 0; i < nd; ++i) {
            lin += coords[i] * stride;
            stride *= dimsArr[i];
        }
        return lin;
    };

    if (!grow)
        return linearIn(need); // `need` == current dims here

    // Build a fresh struct heap with the N-D shape `need` and copy the
    // existing elements to their column-major positions in the new shape.
    const size_t newTotal = checkedMulDims(need, newNd);

    Value grown;
    {
        HeapGuard h(HeapObject::create(mr));
        h->type = ValueType::STRUCT;
        h->dims = Dims(need, newNd);
        h->structArray = h->allocNew<keel::Array<keel::HashMap<keel::String, Value>>>(h->alloc);
        h->structArray->reserve(newTotal);
        for (size_t i = 0; i < newTotal; ++i)
            h->structArray->push_back(keel::HashMap<keel::String, Value>(h->alloc));
        h->fieldOrder = h->allocNew<keel::Array<keel::String>>(h->alloc);
        grown.heap_ = h.release();
    }
    size_t oldStride[Dims::kMaxRank], newStride[Dims::kMaxRank];
    size_t s = 1;
    for (int i = 0; i < oldNd; ++i) { oldStride[i] = s; s *= oldD.dim(i); }
    s = 1;
    for (int i = 0; i < newNd; ++i) { newStride[i] = s; s *= need[i]; }
    const size_t oldTotal = numel();
    const Value &self = *this; // read via const → no spurious COW detach of the discarded array
    for (size_t lin = 0; lin < oldTotal; ++lin) {
        size_t rem = lin, newLin = 0;
        for (int i = 0; i < oldNd; ++i) {
            size_t c = (rem / oldStride[i]) % oldD.dim(i);
            newLin += c * newStride[i];
        }
        grown.structArrayElem(newLin) = self.structArrayElem(lin).clone(grown.heap_->alloc);
    }
    if (isHeap() && heap_->fieldOrder && grown.heap_ && grown.heap_->fieldOrder) {
        *grown.heap_->fieldOrder = heap_->fieldOrder->clone(grown.heap_->alloc);
        for (size_t k = 0; k < newTotal; ++k) {
            auto &dst = (*grown.heap_->structArray)[k];
            for (const auto &fn : *grown.heap_->fieldOrder)
                dst.try_insert(fn.clone(grown.heap_->alloc), Value());
        }
    }
    *this = keel::move(grown);
    return linearIn(need);
}

size_t Value::growCellTo(const size_t *coords, int nd, keel::Allocator *mr)
{
    if (nd <= 0 || nd > Dims::kMaxRank)
        throw std::runtime_error("cell subscript rank out of range");
    if (!mr) {
        mr = (isHeap() && heap_->alloc) ? heap_->alloc
                                        : keel::default_allocator();
    }
    if (isUnset() || isEmpty())
        *this = Value::cell(0, 0, mr);
    if (!isCell())
        throw std::runtime_error("Cell contents indexing on a non-cell value");

    auto linearIn = [&](const Dims &d) {
        size_t lin = 0, stride = 1;
        for (int i = 0; i < nd; ++i) {
            lin += coords[i] * stride;
            stride *= d.dim(i);
        }
        return lin;
    };

    if (nd == 1) {
        // 1-D: preserve row/column orientation, grow in-place (amortized O(1)).
        size_t i = coords[0];
        if (i >= numel()) {
            const bool isCol = (dims().cols() == 1 && dims().rows() > 1);
            detach();
            const size_t newSize = i + 1;
            heap_->dims = isCol ? Dims(newSize, 1) : Dims(1, newSize);
            if (newSize > heap_->cellData->capacity())
                heap_->cellData->reserve(keel::max(newSize, heap_->cellData->capacity() * 2));
            heap_->cellData->resize(newSize);
        }
        return i;
    }

    const Dims oldD = dims();
    const int oldNd = oldD.ndim();
    const int newNd = keel::max(nd, oldNd);
    size_t need[Dims::kMaxRank];
    for (int i = 0; i < newNd; ++i)
        need[i] = (i < oldNd) ? oldD.dim(i) : 1;
    bool grow = (newNd != oldNd);
    for (int i = 0; i < nd; ++i)
        if (coords[i] + 1 > need[i]) {
            need[i] = coords[i] + 1;
            grow = true;
        }
    if (!grow)
        return linearIn(oldD);

    // Rebuild via cellND (Value::resize delegates 2-D/3-D to the
    // buffer-only path which would corrupt a cell, so don't route through
    // it). Copy existing contents to their column-major positions.
    Value grown = Value::cellND(need, newNd, mr);
    size_t oldStride[Dims::kMaxRank], newStride[Dims::kMaxRank];
    size_t s = 1;
    for (int i = 0; i < oldNd; ++i) { oldStride[i] = s; s *= oldD.dim(i); }
    s = 1;
    for (int i = 0; i < newNd; ++i) { newStride[i] = s; s *= need[i]; }
    const size_t oldTotal = numel();
    const Value &self = *this; // read via const → no spurious COW detach
    for (size_t lin = 0; lin < oldTotal; ++lin) {
        size_t rem = lin, newLin = 0;
        for (int i = 0; i < oldNd; ++i) {
            size_t c = (rem / oldStride[i]) % oldD.dim(i);
            newLin += c * newStride[i];
        }
        grown.cellAt(newLin) = self.cellAt(lin);
    }
    *this = keel::move(grown);
    return linearIn(dims());
}
keel::HashMap<keel::String, Value> &Value::structArrayElem(size_t i)
{
    if (!isHeap() || heap_->type != ValueType::STRUCT || !heap_->structArray)
        throw std::runtime_error("Not a struct");
    if (i >= heap_->structArray->size())
        throw std::runtime_error("struct array index out of range");
    detach();
    return (*heap_->structArray)[i];
}
const keel::HashMap<keel::String, Value> &Value::structArrayElem(size_t i) const
{
    if (!isHeap() || heap_->type != ValueType::STRUCT || !heap_->structArray)
        throw std::runtime_error("Not a struct");
    if (i >= heap_->structArray->size())
        throw std::runtime_error("struct array index out of range");
    return (*heap_->structArray)[i];
}

// ── BUG #15: insertion-order helpers ───────────────────────────
namespace {
inline void appendIfNew(keel::Array<keel::String> *order,
                        keel::StringRef name,
                        keel::Allocator *alloc)
{
    if (!order) return;
    for (const auto &n : *order)
        if (n == name) return;
    order->push_back(keel::String(name, alloc));
}

inline keel::FixedString<64> fmtGeneral(double v) noexcept
{
    keel::FixedString<64> out;
    const keel::usize len = keel::format_f64_general(out.inline_buffer(), 63, v, 6, false);
    out.set_inline_size(len);
    return out;
}
} // namespace

void Value::setField(size_t linearIdx, keel::StringRef name, const Value &v)
{
    if (!isHeap() || heap_->type != ValueType::STRUCT || !heap_->structArray)
        throw std::runtime_error("Not a struct");
    if (linearIdx >= heap_->structArray->size())
        throw std::runtime_error("struct array index out of range");
    detach();
    // After detach heap_ may differ; re-fetch via heap_.
    appendIfNew(heap_->fieldOrder, name, heap_->alloc);
    if (heap_->structArray->size() > 1) {
        for (size_t k = 0; k < heap_->structArray->size(); ++k) {
            if (k != linearIdx && !(*heap_->structArray)[k].contains(name))
                (*heap_->structArray)[k].put(keel::String(name, heap_->alloc), Value());
        }
    }
    if (Value *vp = (*heap_->structArray)[linearIdx].get(name))
        *vp = v;
    else
        (*heap_->structArray)[linearIdx].put(keel::String(name, heap_->alloc), v);
}

void Value::setFieldAll(keel::StringRef name, const Value &v)
{
    if (!isHeap() || heap_->type != ValueType::STRUCT || !heap_->structArray)
        throw std::runtime_error("Not a struct");
    detach();
    appendIfNew(heap_->fieldOrder, name, heap_->alloc);
    for (auto &m : *heap_->structArray) {
        if (Value *vp = m.get(name))
            *vp = v;
        else
            m.put(keel::String(name, heap_->alloc), v);
    }
}

void Value::removeField(keel::StringRef name)
{
    if (!isHeap() || heap_->type != ValueType::STRUCT || !heap_->structArray)
        return;
    detach();
    for (auto &m : *heap_->structArray)
        m.remove(name);
    if (heap_->fieldOrder) {
        auto &v = *heap_->fieldOrder;
        for (size_t i = 0; i < v.size(); ++i) {
            if (v[i] == name) {
                v.erase(i);
                break;
            }
        }
    }
}

keel::Array<keel::String> Value::fieldNamesInOrder() const
{
    keel::Array<keel::String> out;
    if (!isHeap() || heap_->type != ValueType::STRUCT) return out;
    if (heap_->fieldOrder && !heap_->fieldOrder->empty()) {
        out = heap_->fieldOrder->clone();
        if (heap_->structArray && !heap_->structArray->empty()
            && (*heap_->structArray)[0].size() > out.size()) {
            keel::Array<keel::String> extra;
            for (const auto &entry : (*heap_->structArray)[0]) {
                bool seen = false;
                for (const auto &existing : out) {
                    if (existing == entry.key) {
                        seen = true;
                        break;
                    }
                }
                if (!seen)
                    extra.push_back(entry.key.clone());
            }
            keel::sort(extra);
            for (auto &e : extra)
                out.push_back(keel::move(e));
        }
        return out;
    }
    // Fallback: derive from element-0 map (alphabetical) for legacy
    // structs created via raw map insertion (no setField call).
    if (heap_->structArray && !heap_->structArray->empty()) {
        out.reserve((*heap_->structArray)[0].size());
        for (const auto &entry : (*heap_->structArray)[0])
            out.push_back(entry.key.clone());
        keel::sort(out);
    }
    return out;
}

keel::String Value::debugString() const
{
    keel::String os;
    ValueType t = type();
    keel::format_into(os, "{} [{}x{}", mtypeName(t), dims().rows(), dims().cols());
    if (dims().is3D())
        keel::format_into(os, "x{}", dims().pages());
    os.push_back(']');
    if (t == ValueType::DOUBLE && numel() <= 20 && numel() > 0) {
        os.append(" = ");
        const double *dd = doubleData();
        if (isScalar()) {
            os.append(fmtGeneral(dd[0]));
        } else {
            os.push_back('[');
            for (size_t i = 0; i < numel(); ++i) {
                if (i)
                    os.append(", ");
                os.append(fmtGeneral(dd[i]));
            }
            os.push_back(']');
        }
    }
    if (t == ValueType::COMPLEX && numel() <= 20 && numel() > 0 && isHeap() && heap_->buffer) {
        os.append(" = ");
        const keel::c64 *cd = complexData();
        if (isScalar()) {
            os.append(fmtGeneral(cd[0].real()));
            if (cd[0].imag() >= 0)
                os.push_back('+');
            os.append(fmtGeneral(cd[0].imag()));
            os.push_back('i');
        } else {
            os.push_back('[');
            for (size_t i = 0; i < numel(); ++i) {
                if (i)
                    os.append(", ");
                os.append(fmtGeneral(cd[i].real()));
                if (cd[i].imag() >= 0)
                    os.push_back('+');
                os.append(fmtGeneral(cd[i].imag()));
                os.push_back('i');
            }
            os.push_back(']');
        }
    }
    if (t == ValueType::LOGICAL && numel() > 0) {
        os.append(" = ");
        if (isScalar())
            os.append(toBool() ? "true" : "false");
        else if (isHeap() && heap_->buffer) {
            const uint8_t *ld = logicalData();
            os.push_back('[');
            for (size_t i = 0; i < numel(); ++i) {
                if (i)
                    os.append(", ");
                os.push_back(ld[i] ? '1' : '0');
            }
            os.push_back(']');
        }
    }
    if (t == ValueType::CHAR && isHeap() && heap_->buffer)
        keel::format_into(os, " = '{}'", toString());
    if (t == ValueType::STRING) {
        if (isScalar())
            keel::format_into(os, " = \"{}\"", toString());
        else if (isHeap() && heap_->cellData) {
            os.append(" [");
            for (size_t i = 0; i < heap_->cellData->size() && i < 10; ++i) {
                if (i)
                    os.append(", ");
                keel::format_into(os, "\"{}\"", (*heap_->cellData)[i].toString());
            }
            os.push_back(']');
        }
    }
    if (t == ValueType::FUNC_HANDLE)
        keel::format_into(os, " = @{}", funcHandleName());
    if (t == ValueType::CELL && isHeap() && heap_->cellData) {
        os.append(" {");
        for (size_t i = 0; i < heap_->cellData->size() && i < 10; ++i) {
            if (i)
                os.append(", ");
            os.append((*heap_->cellData)[i].debugString());
        }
        if (heap_->cellData->size() > 10)
            os.append(", ...");
        os.push_back('}');
    }
    return os;
}

// ============================================================
// Display formatting
// ============================================================

static keel::String fmtDouble(double v)
{
    if (keel::is_nan(v)) return keel::String::from("NaN");
    if (keel::is_inf(v)) return keel::String::from(v > 0 ? "Inf" : "-Inf");
    if (v == 0.0) return keel::String::from("0");
    // Integer check
    if (keel::abs(v) < 1e15 && v == static_cast<double>(keel::floor_i64(v)))
        return keel::format("{}", static_cast<int64_t>(v));
    return keel::String::from(fmtGeneral(v));
}

static keel::String fmtComplex(const keel::c64 &c)
{
    keel::String os;
    if (c.real() != 0.0 || c.imag() == 0.0)
        os.append(fmtDouble(c.real()));
    if (c.imag() != 0.0) {
        if (c.real() != 0.0 && c.imag() > 0) os.append(" + ");
        else if (c.real() != 0.0 && c.imag() < 0) os.append(" - ");
        double ai = keel::abs(c.imag());
        if (ai == 1.0) {
            os.push_back('i');
        } else {
            os.append(fmtDouble(ai));
            os.push_back('i');
        }
    }
    return os;
}

keel::String Value::formatDisplay(keel::StringRef name) const
{
    keel::String os;

    if (!name.empty())
        keel::format_into(os, "{} =\n", name);

    ValueType t = type();

    switch (t) {
    case ValueType::DOUBLE: {
        if (isScalar()) {
            keel::format_into(os, "   {}\n", fmtDouble(toScalar()));
        } else if (isEmpty()) {
            os.append("     []\n");
        } else {
            auto &d = dims();
            const size_t R = d.rows(), C = d.cols();
            const int nd = d.ndim();
            const double *base0 = doubleData();
            forEachOuterPage(d, [&](size_t plin, const size_t *outerCoords) {
                if (nd >= 3) {
                    os.append("\n(:,:");
                    for (int i = 2; i < nd; ++i)
                        keel::format_into(os, ",{}", outerCoords[i - 2] + 1);
                    os.append(") =\n\n");
                }
                const double *page = base0 + plin * R * C;
                keel::Array<keel::String> cells(R * C);
                keel::SmallArray<size_t, 32> colWidth(C, size_t{0});
                for (size_t r = 0; r < R; ++r) {
                    for (size_t c = 0; c < C; ++c) {
                        keel::String cell = fmtDouble(page[c * R + r]);
                        colWidth[c] = keel::max(colWidth[c], cell.size());
                        cells[r * C + c] = keel::move(cell);
                    }
                }
                for (size_t r = 0; r < R; ++r) {
                    os.append("   ");
                    for (size_t c = 0; c < C; ++c) {
                        const keel::String &cell = cells[r * C + c];
                        size_t pad = colWidth[c] > cell.size()
                                         ? colWidth[c] - cell.size()
                                         : 0;
                        for (size_t i = 0; i < pad + 1; ++i) os.push_back(' ');
                        os.append(cell);
                    }
                    os.push_back('\n');
                }
            });
        }
        break;
    }
    case ValueType::CHAR: {
        auto &d = dims();
        if (d.rows() <= 1) {
            keel::format_into(os, "   '{}'\n", toString());
        } else {
            for (size_t r = 0; r < d.rows(); ++r)
                keel::format_into(os, "   '{}'\n", charRow(r));
        }
        break;
    }
    case ValueType::STRING:
        if (isScalar()) {
            keel::format_into(os, "   \"{}\"\n", toString());
        } else {
            auto &d = dims();
            keel::format_into(os, "  {}x{} string array\n", d.rows(), d.cols());
            for (size_t i = 0; i < numel() && i < 20; ++i)
                keel::format_into(os, "    \"{}\"\n", stringElem(i));
        }
        break;
    case ValueType::LOGICAL: {
        if (isScalar()) {
            keel::format_into(os, "   {}\n", toBool() ? "1" : "0");
        } else {
            auto &d = dims();
            for (size_t p = 0; p < d.pages(); ++p) {
                if (d.is3D())
                    keel::format_into(os, "\n(:,:,{}) =\n\n", p + 1);
                for (size_t r = 0; r < d.rows(); ++r) {
                    os.append("   ");
                    for (size_t c = 0; c < d.cols(); ++c) {
                        size_t idx = d.is3D() ? (p * d.rows() * d.cols() + c * d.rows() + r)
                                              : d.sub2ind(r, c);
                        os.append(logicalData()[idx] ? " 1" : " 0");
                    }
                    os.push_back('\n');
                }
            }
        }
        break;
    }
    case ValueType::COMPLEX: {
        if (isScalar()) {
            keel::format_into(os, "   {}\n", fmtComplex(toComplex()));
        } else {
            auto &d = dims();
            const size_t R = d.rows(), C = d.cols();
            keel::Array<keel::String> cells(R * C);
            keel::SmallArray<size_t, 32> colWidth(C, size_t{0});
            for (size_t r = 0; r < R; ++r) {
                for (size_t c = 0; c < C; ++c) {
                    keel::String cell = fmtComplex(complexElem(r, c));
                    colWidth[c] = keel::max(colWidth[c], cell.size());
                    cells[r * C + c] = keel::move(cell);
                }
            }
            for (size_t r = 0; r < R; ++r) {
                os.append("   ");
                for (size_t c = 0; c < C; ++c) {
                    const keel::String &cell = cells[r * C + c];
                    size_t pad = colWidth[c] > cell.size()
                                     ? colWidth[c] - cell.size()
                                     : 0;
                    for (size_t i = 0; i < pad + 1; ++i) os.push_back(' ');
                    os.append(cell);
                }
                os.push_back('\n');
            }
        }
        break;
    }
    case ValueType::SINGLE: {
        if (isScalar()) {
            keel::format_into(os, "   {}\n", fmtGeneral(static_cast<double>(*singleData())));
        } else if (isEmpty()) {
            os.append("     []\n");
        } else {
            auto &d = dims();
            const float *fd = singleData();
            for (size_t p = 0; p < d.pages(); ++p) {
                if (d.is3D())
                    keel::format_into(os, "\n(:,:,{}) =\n\n", p + 1);
                for (size_t r = 0; r < d.rows(); ++r) {
                    os.append("   ");
                    for (size_t c = 0; c < d.cols(); ++c) {
                        size_t idx = d.is3D() ? d.sub2ind(r, c, p) : d.sub2ind(r, c);
                        keel::format_into(os, " {}", fmtGeneral(static_cast<double>(fd[idx])));
                    }
                    os.push_back('\n');
                }
            }
        }
        break;
    }
    case ValueType::INT8:
    case ValueType::INT16:
    case ValueType::INT32:
    case ValueType::INT64:
    case ValueType::UINT8:
    case ValueType::UINT16:
    case ValueType::UINT32:
    case ValueType::UINT64: {
        if (isScalar()) {
            keel::format_into(os, "   {}\n", static_cast<int64_t>(toScalar()));
        } else if (isEmpty()) {
            os.append("     []\n");
        } else {
            auto &d = dims();
            for (size_t p = 0; p < d.pages(); ++p) {
                if (d.is3D())
                    keel::format_into(os, "\n(:,:,{}) =\n\n", p + 1);
                for (size_t r = 0; r < d.rows(); ++r) {
                    os.append("   ");
                    for (size_t c = 0; c < d.cols(); ++c) {
                        size_t idx = d.is3D() ? d.sub2ind(r, c, p) : d.sub2ind(r, c);
                        const void *raw = rawData();
                        int64_t val = 0;
                        switch (t) {
                        case ValueType::INT8:   val = static_cast<const int8_t*>(raw)[idx]; break;
                        case ValueType::INT16:  val = static_cast<const int16_t*>(raw)[idx]; break;
                        case ValueType::INT32:  val = static_cast<const int32_t*>(raw)[idx]; break;
                        case ValueType::INT64:  val = static_cast<const int64_t*>(raw)[idx]; break;
                        case ValueType::UINT8:  val = static_cast<const uint8_t*>(raw)[idx]; break;
                        case ValueType::UINT16: val = static_cast<const uint16_t*>(raw)[idx]; break;
                        case ValueType::UINT32: val = static_cast<const uint32_t*>(raw)[idx]; break;
                        case ValueType::UINT64: val = static_cast<int64_t>(static_cast<const uint64_t*>(raw)[idx]); break;
                        default: break;
                        }
                        keel::format_into(os, " {}", val);
                    }
                    os.push_back('\n');
                }
            }
        }
        break;
    }
    case ValueType::STRUCT: {
        const auto names = fieldNamesInOrder();
        if (isStructArray()) {
            const auto &d = dims();
            keel::format_into(os, "  {}x{} struct array with fields:\n\n", d.rows(), d.cols());
            if (numel() > 0) {
                for (const auto &k : names)
                    keel::format_into(os, "    {}\n", k);
            }
        } else {
            os.append("  struct with fields:\n\n");
            if (numel() > 0) {
                const auto &m = structArrayElem(0);
                for (const auto &k : names) {
                    if (const Value *vp = m.get(k))
                        keel::format_into(os, "    {}: {}\n", k, vp->debugString());
                }
            }
        }
        break;
    }
    case ValueType::FUNC_HANDLE:
        keel::format_into(os, "   @{}\n", funcHandleName());
        break;
    case ValueType::CELL: {
        auto &d = dims();
        keel::format_into(os, "  {{{}x{}", d.rows(), d.cols());
        if (d.is3D()) keel::format_into(os, "x{}", d.pages());
        os.append(" cell}\n");
        for (size_t i = 0; i < numel() && i < 20; ++i)
            keel::format_into(os, "    {{{}}}: {}\n", i + 1, cellAt(i).debugString());
        if (numel() > 20)
            keel::format_into(os, "    ... ({} more)\n", numel() - 20);
        break;
    }
    case ValueType::EMPTY:
        os.append("     []\n");
        break;
    default:
        keel::format_into(os, "   {}\n", debugString());
        break;
    }

    os.push_back('\n');
    return os;
}

} // namespace numkit