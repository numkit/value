// include/numkit/value/value_type.hpp
#pragma once

#include <keel/core.h>

namespace numkit {

// ============================================================
// ValueType — element type enum
// ============================================================
enum class ValueType : keel::u8 {
    EMPTY,
    DOUBLE,
    COMPLEX,
    LOGICAL,
    CHAR,
    CELL,
    STRUCT,
    FUNC_HANDLE,
    INT8,
    INT16,
    INT32,
    INT64,
    UINT8,
    UINT16,
    UINT32,
    UINT64,
    SINGLE,
    STRING,
    OBJECT,  // class instance (builtin or user classdef)
    CSL      // comma-separated list — transient value-list produced by c{:} / c{idx} /
             // s.field expansion that splicing contexts flatten. Reuses HeapObject::cellData.
};

const char *mtypeName(ValueType t);
keel::usize elementSize(ValueType t);
bool isIntegerType(ValueType t);
bool isFloatType(ValueType t); // double or single

} // namespace numkit
