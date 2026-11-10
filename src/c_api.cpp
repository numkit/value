// src/c_api.cpp
//
// Foundation `numkit_value_c` C-ABI implementation.
#include <numkit/value/c_api_bridge.hpp>
#include <numkit/value/dims.hpp>
#include <numkit/value/value_type.hpp>

#include <algorithm>
#include <atomic>
#include <cstring>
#include <string>

struct numkit_val_s {
    numkit::Value val;
    explicit numkit_val_s(numkit::Value v) : val(std::move(v)) {}
};

namespace {

std::atomic<int64_t> g_liveValues{0};

numkit::ValueType toValueType(numkit_dtype dt)
{
    using numkit::ValueType;
    switch (dt) {
    case NUMKIT_DTYPE_DOUBLE:      return ValueType::DOUBLE;
    case NUMKIT_DTYPE_SINGLE:      return ValueType::SINGLE;
    case NUMKIT_DTYPE_COMPLEX:     return ValueType::COMPLEX;
    case NUMKIT_DTYPE_LOGICAL:     return ValueType::LOGICAL;
    case NUMKIT_DTYPE_CHAR:        return ValueType::CHAR;
    case NUMKIT_DTYPE_STRING:      return ValueType::STRING;
    case NUMKIT_DTYPE_INT8:        return ValueType::INT8;
    case NUMKIT_DTYPE_INT16:       return ValueType::INT16;
    case NUMKIT_DTYPE_INT32:       return ValueType::INT32;
    case NUMKIT_DTYPE_INT64:       return ValueType::INT64;
    case NUMKIT_DTYPE_UINT8:       return ValueType::UINT8;
    case NUMKIT_DTYPE_UINT16:      return ValueType::UINT16;
    case NUMKIT_DTYPE_UINT32:      return ValueType::UINT32;
    case NUMKIT_DTYPE_UINT64:      return ValueType::UINT64;
    case NUMKIT_DTYPE_CELL:        return ValueType::CELL;
    case NUMKIT_DTYPE_STRUCT:      return ValueType::STRUCT;
    case NUMKIT_DTYPE_FUNC_HANDLE: return ValueType::FUNC_HANDLE;
    case NUMKIT_DTYPE_OBJECT:      return ValueType::OBJECT;
    default:
        throw std::invalid_argument("Unsupported numkit_dtype");
    }
}

numkit_dtype fromValueType(numkit::ValueType vt) noexcept
{
    using numkit::ValueType;
    switch (vt) {
    case ValueType::DOUBLE:      return NUMKIT_DTYPE_DOUBLE;
    case ValueType::SINGLE:      return NUMKIT_DTYPE_SINGLE;
    case ValueType::COMPLEX:     return NUMKIT_DTYPE_COMPLEX;
    case ValueType::LOGICAL:     return NUMKIT_DTYPE_LOGICAL;
    case ValueType::CHAR:        return NUMKIT_DTYPE_CHAR;
    case ValueType::STRING:      return NUMKIT_DTYPE_STRING;
    case ValueType::INT8:        return NUMKIT_DTYPE_INT8;
    case ValueType::INT16:       return NUMKIT_DTYPE_INT16;
    case ValueType::INT32:       return NUMKIT_DTYPE_INT32;
    case ValueType::INT64:       return NUMKIT_DTYPE_INT64;
    case ValueType::UINT8:       return NUMKIT_DTYPE_UINT8;
    case ValueType::UINT16:      return NUMKIT_DTYPE_UINT16;
    case ValueType::UINT32:      return NUMKIT_DTYPE_UINT32;
    case ValueType::UINT64:      return NUMKIT_DTYPE_UINT64;
    case ValueType::CELL:        return NUMKIT_DTYPE_CELL;
    case ValueType::STRUCT:      return NUMKIT_DTYPE_STRUCT;
    case ValueType::FUNC_HANDLE: return NUMKIT_DTYPE_FUNC_HANDLE;
    case ValueType::OBJECT:      return NUMKIT_DTYPE_OBJECT;
    default:                     return NUMKIT_DTYPE_UNKNOWN;
    }
}

size_t writeStrBuf(const std::string &s, char *buf, size_t bufCap) noexcept
{
    if (!buf || bufCap == 0) return s.size();
    size_t n = std::min(s.size(), bufCap - 1);
    if (n > 0) std::memcpy(buf, s.data(), n);
    buf[n] = '\0';
    return s.size();
}

std::string extractStringValue(const numkit::Value &v)
{
    if (v.isString()) {
        return v.numel() > 0 ? v.stringElem(0) : std::string{};
    }
    if (v.isChar()) {
        return v.toString();
    }
    return v.toString();
}

} // namespace

namespace numkit::c_abi {

numkit_val *boxValue(Value v)
{
    auto *p = new numkit_val_s(std::move(v));
    g_liveValues.fetch_add(1, std::memory_order_relaxed);
    return p;
}

const Value &unboxValue(const numkit_val *v)
{
    if (!v) return Value::Empty;
    return v->val;
}

Value &unboxValueMut(numkit_val *v)
{
    if (!v) throw std::invalid_argument("Null numkit_val* pointer");
    return v->val;
}

} // namespace numkit::c_abi

using numkit::Dims;
using numkit::Value;
using numkit::ValueType;
using numkit::c_abi::boxValue;
using numkit::c_abi::clearError;
using numkit::c_abi::setError;

extern "C" {

uint32_t numkit_abi_version(void)
{
    return NUMKIT_C_ABI_VERSION;
}

int64_t numkit_debug_live_values(void)
{
    return g_liveValues.load(std::memory_order_relaxed);
}

numkit_val *numkit_val_empty(void)
{
    return boxValue(Value::Empty);
}

numkit_val *numkit_val_scalar(double v)
{
    return boxValue(Value::scalar(v));
}

numkit_val *numkit_val_logical(int b)
{
    return boxValue(Value::logicalScalar(b != 0));
}

numkit_val *numkit_val_complex_scalar(double re, double im)
{
    return boxValue(Value::complexScalar(re, im));
}

numkit_val *numkit_val_string(const char *utf8)
{
    return boxValue(Value::stringScalar(utf8 ? utf8 : ""));
}

numkit_val *numkit_val_char(const char *utf8)
{
    return boxValue(Value::fromString(utf8 ? utf8 : ""));
}

numkit_val *numkit_val_func_handle(const char *name)
{
    return boxValue(Value::funcHandle(name ? name : ""));
}

numkit_val *numkit_val_array_copy(const void *data,
                                  numkit_dtype dtype,
                                  const size_t *dims,
                                  int ndims,
                                  numkit_error *err)
{
    clearError(err);
    try {
        if (!dims || ndims <= 0) {
            throw std::invalid_argument("dims must be non-null and ndims > 0");
        }
        ValueType vt = toValueType(dtype);
        Value arr = Value::matrixND(dims, ndims, vt);
        size_t bytes = arr.rawBytes();
        if (bytes > 0) {
            if (!data) {
                throw std::invalid_argument("data pointer is null for non-empty array");
            }
            std::memcpy(arr.rawDataMut(), data, bytes);
        }
        return boxValue(std::move(arr));
    } catch (const std::exception &ex) {
        setError(err, 1, ex.what());
        return nullptr;
    }
}

numkit_val *numkit_val_view(const void *data,
                            numkit_dtype dtype,
                            const size_t *dims,
                            int ndims,
                            numkit_error *err)
{
    clearError(err);
    try {
        if (!dims || ndims <= 0) {
            throw std::invalid_argument("dims must be non-null and ndims > 0");
        }
        ValueType vt = toValueType(dtype);
        Dims d(dims, ndims);
        if (d.numel() > 0 && !data) {
            throw std::invalid_argument("data pointer is null for non-empty view");
        }
        return boxValue(Value::view(data, vt, d));
    } catch (const std::exception &ex) {
        setError(err, 1, ex.what());
        return nullptr;
    }
}

numkit_val *numkit_val_alloc_array(numkit_dtype dtype,
                                   const size_t *dims,
                                   int ndims,
                                   void **out_data_ptr,
                                   numkit_error *err)
{
    clearError(err);
    try {
        if (!dims || ndims <= 0) {
            throw std::invalid_argument("dims must be non-null and ndims > 0");
        }
        ValueType vt = toValueType(dtype);
        Value arr = Value::matrixND(dims, ndims, vt);
        void *mutPtr = arr.numel() > 0 ? arr.rawDataMut() : nullptr;
        if (out_data_ptr) {
            *out_data_ptr = mutPtr;
        }
        return boxValue(std::move(arr));
    } catch (const std::exception &ex) {
        if (out_data_ptr) *out_data_ptr = nullptr;
        setError(err, 1, ex.what());
        return nullptr;
    }
}

numkit_val *numkit_val_clone(const numkit_val *v)
{
    if (!v) return nullptr;
    return boxValue(v->val);
}

void numkit_val_release(numkit_val *v)
{
    if (!v) return;
    delete v;
    g_liveValues.fetch_sub(1, std::memory_order_relaxed);
}

numkit_dtype numkit_val_type(const numkit_val *v)
{
    if (!v) return NUMKIT_DTYPE_UNKNOWN;
    return fromValueType(v->val.type());
}

size_t numkit_val_numel(const numkit_val *v)
{
    return v ? v->val.numel() : 0;
}

size_t numkit_val_ndims(const numkit_val *v)
{
    return v ? v->val.dims().ndim() : 0;
}

size_t numkit_val_dims(const numkit_val *v, size_t *out_dims, size_t max_dims)
{
    if (!v) return 0;
    const Dims &d = v->val.dims();
    size_t nd = d.ndim();
    if (out_dims && max_dims > 0) {
        size_t n = std::min(nd, max_dims);
        for (size_t i = 0; i < n; ++i) {
            out_dims[i] = d.dim(static_cast<int>(i));
        }
    }
    return nd;
}

int numkit_val_is_empty(const numkit_val *v)
{
    return (!v || v->val.isEmpty()) ? 1 : 0;
}

int numkit_val_is_scalar(const numkit_val *v)
{
    return (v && v->val.isScalar()) ? 1 : 0;
}

const void *numkit_val_raw_data(const numkit_val *v)
{
    if (!v) return nullptr;
    try {
        return v->val.rawData();
    } catch (...) {
        return nullptr;
    }
}

size_t numkit_val_raw_bytes(const numkit_val *v)
{
    return v ? v->val.rawBytes() : 0;
}

size_t numkit_val_copy_bytes(const numkit_val *v, void *dst, size_t max_bytes)
{
    if (!v || !dst || max_bytes == 0) return 0;
    const void *src = numkit_val_raw_data(v);
    if (!src) return 0;
    size_t avail = v->val.rawBytes();
    size_t n = std::min(avail, max_bytes);
    if (n > 0) std::memcpy(dst, src, n);
    return n;
}

double numkit_val_as_double(const numkit_val *v, numkit_error *err)
{
    clearError(err);
    if (!v) {
        setError(err, 1, "Null value handle");
        return 0.0;
    }
    try {
        return v->val.toScalar();
    } catch (const std::exception &ex) {
        setError(err, 1, ex.what());
        return 0.0;
    }
}

int numkit_val_as_bool(const numkit_val *v, numkit_error *err)
{
    clearError(err);
    if (!v) {
        setError(err, 1, "Null value handle");
        return 0;
    }
    try {
        return v->val.toBool() ? 1 : 0;
    } catch (const std::exception &ex) {
        setError(err, 1, ex.what());
        return 0;
    }
}

int numkit_val_as_complex(const numkit_val *v,
                          double *out_re,
                          double *out_im,
                          numkit_error *err)
{
    clearError(err);
    if (!v) {
        setError(err, 1, "Null value handle");
        return 1;
    }
    try {
        auto c = v->val.toComplex();
        if (out_re) *out_re = c.real();
        if (out_im) *out_im = c.imag();
        return 0;
    } catch (const std::exception &ex) {
        setError(err, 1, ex.what());
        return 1;
    }
}

size_t numkit_val_string_bytes(const numkit_val *v)
{
    if (!v) return 0;
    try {
        return extractStringValue(v->val).size();
    } catch (...) {
        return 0;
    }
}

size_t numkit_val_write_string(const numkit_val *v, char *buf, size_t buf_cap)
{
    if (!v) {
        if (buf && buf_cap > 0) buf[0] = '\0';
        return 0;
    }
    try {
        return writeStrBuf(extractStringValue(v->val), buf, buf_cap);
    } catch (...) {
        if (buf && buf_cap > 0) buf[0] = '\0';
        return 0;
    }
}

size_t numkit_val_object_class(const numkit_val *v, char *buf, size_t buf_cap)
{
    if (!v || !v->val.isObject()) {
        if (buf && buf_cap > 0) buf[0] = '\0';
        return 0;
    }
    return writeStrBuf(v->val.objectClassName(), buf, buf_cap);
}

numkit_val *numkit_cell_create(const size_t *dims, int ndims, numkit_error *err)
{
    clearError(err);
    try {
        if (!dims || ndims <= 0) {
            throw std::invalid_argument("dims must be non-null and ndims > 0");
        }
        return boxValue(Value::cellND(dims, ndims));
    } catch (const std::exception &ex) {
        setError(err, 1, ex.what());
        return nullptr;
    }
}

numkit_val *numkit_cell_get(const numkit_val *cell, size_t linear_idx, numkit_error *err)
{
    clearError(err);
    if (!cell || !cell->val.isCell()) {
        setError(err, 1, "Value is not a cell array");
        return nullptr;
    }
    try {
        if (linear_idx >= cell->val.numel()) {
            throw std::out_of_range("Cell index out of bounds");
        }
        return boxValue(cell->val.cellAt(linear_idx));
    } catch (const std::exception &ex) {
        setError(err, 1, ex.what());
        return nullptr;
    }
}

int numkit_cell_set(numkit_val *cell,
                    size_t linear_idx,
                    const numkit_val *elem,
                    numkit_error *err)
{
    clearError(err);
    if (!cell || !cell->val.isCell()) {
        setError(err, 1, "Value is not a cell array");
        return 1;
    }
    try {
        if (linear_idx >= cell->val.numel()) {
            throw std::out_of_range("Cell index out of bounds");
        }
        cell->val.cellAt(linear_idx) = elem ? elem->val : Value::Empty;
        return 0;
    } catch (const std::exception &ex) {
        setError(err, 1, ex.what());
        return 1;
    }
}

numkit_val *numkit_struct_create(size_t rows, size_t cols, numkit_error *err)
{
    clearError(err);
    try {
        if (rows == 1 && cols == 1) {
            return boxValue(Value::structure());
        }
        return boxValue(Value::structArray(rows, cols));
    } catch (const std::exception &ex) {
        setError(err, 1, ex.what());
        return nullptr;
    }
}

size_t numkit_struct_num_fields(const numkit_val *s)
{
    if (!s || !s->val.isStruct()) return 0;
    return s->val.fieldNamesInOrder().size();
}

size_t numkit_struct_field_name(const numkit_val *s,
                                size_t field_idx,
                                char *buf,
                                size_t buf_cap)
{
    if (!s || !s->val.isStruct()) {
        if (buf && buf_cap > 0) buf[0] = '\0';
        return 0;
    }
    auto names = s->val.fieldNamesInOrder();
    if (field_idx >= names.size()) {
        if (buf && buf_cap > 0) buf[0] = '\0';
        return 0;
    }
    return writeStrBuf(names[field_idx], buf, buf_cap);
}

numkit_val *numkit_struct_get(const numkit_val *s,
                              size_t linear_idx,
                              const char *field,
                              numkit_error *err)
{
    clearError(err);
    if (!s || !s->val.isStruct() || !field) {
        setError(err, 1, "Invalid struct or field name");
        return nullptr;
    }
    try {
        if (s->val.isStructArray()) {
            if (linear_idx >= s->val.numel()) {
                throw std::out_of_range("Struct array index out of bounds");
            }
            const auto &m = s->val.structArrayElem(linear_idx);
            auto it = m.find(field);
            if (it == m.end()) {
                throw std::runtime_error(std::string("Non-existent field '") + field + "'");
            }
            return boxValue(it->second);
        }
        if (linear_idx != 0) {
            throw std::out_of_range("Struct index out of bounds");
        }
        return boxValue(s->val.field(field));
    } catch (const std::exception &ex) {
        setError(err, 1, ex.what());
        return nullptr;
    }
}

int numkit_struct_set(numkit_val *s,
                      size_t linear_idx,
                      const char *field,
                      const numkit_val *val,
                      numkit_error *err)
{
    clearError(err);
    if (!s || !s->val.isStruct() || !field) {
        setError(err, 1, "Invalid struct or field name");
        return 1;
    }
    try {
        s->val.setField(linear_idx, field, val ? val->val : Value::Empty);
        return 0;
    } catch (const std::exception &ex) {
        setError(err, 1, ex.what());
        return 1;
    }
}

} // extern "C"
