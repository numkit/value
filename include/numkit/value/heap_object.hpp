// include/numkit/value/heap_object.hpp
#pragma once

#include <numkit/value/data_buffer.hpp>
#include <numkit/value/dims.hpp>
#include <numkit/value/value_type.hpp>

#include <keel/array.h>

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace numkit {

class Value;        // forward decl
struct ObjectState; // OBJECT instance state (see object.hpp); held by shared_ptr

// ============================================================
// HeapObject — ref-counted storage for non-scalar values
//
// Holds arrays, cells, structs, func handles, and class instances.
// Shared via COW (copy-on-write) through refCount.
// ============================================================
struct HeapObject
{
    detail::RefCount refCount{1};
    ValueType type = ValueType::EMPTY;
    Dims dims;
    keel::Allocator *alloc = nullptr;  // not owned; outlives the heap

    // Array data (DOUBLE, SINGLE, INT*, UINT*, LOGICAL, CHAR, COMPLEX)
    DataBuffer *buffer = nullptr;

    // Extended data — allocated only for CELL, STRING, CSL, STRUCT, FUNC_HANDLE.
    // CELL / STRING / CSL element slots flow through `alloc` via `keel::Array<Value>`.
    // STRUCT layout: array of structs (AoS). `structArray` holds `numel()` per-element
    // field maps; a single struct has `structArray->size() == 1`.
    keel::Array<Value> *cellData = nullptr;
    std::vector<std::map<std::string, Value>> *structArray = nullptr;
    // Insertion-order tracker for struct fields so fieldnames() preserves creation order.
    std::vector<std::string> *fieldOrder = nullptr;
    std::string *funcName = nullptr;
    // Reconstructed source text for anonymous-function handles (e.g. "@(x)x+1").
    std::string *funcSource = nullptr;

    // OBJECT (class instance / array)
    std::string *objClass = nullptr;
    std::vector<std::shared_ptr<ObjectState>> objStates;
    bool objIsHandle = false;
    bool isSingleComplex = false;

    // Capacity for appendScalar amortization
    keel::usize appendCapacity = 0;

    HeapObject() = default;
    ~HeapObject();

    HeapObject(const HeapObject &) = delete;
    HeapObject &operator=(const HeapObject &) = delete;

    void addRef() noexcept { detail::refInc(refCount); }
    bool release() noexcept { return detail::refDecIsZero(refCount); }
    keel::i32 loadRefCount() const noexcept { return detail::refLoadAcquire(refCount); }

    // Deep clone for COW
    HeapObject *clone() const;
};

} // namespace numkit
