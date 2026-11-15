// include/numkit/value/heap_object.hpp
#pragma once

#include <numkit/value/data_buffer.hpp>
#include <numkit/value/dims.hpp>
#include <numkit/value/value_type.hpp>

#include <keel/array.h>
#include <keel/hash_map.h>
#include <keel/smart_ptr.h>
#include <keel/string.h>

#include <new>

namespace numkit {

class Value;        // forward decl
struct ObjectState; // OBJECT instance state (see object.hpp); held by keel::Arc

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
    keel::Allocator *alloc = nullptr;  // immutable after creation; outlives HeapObject

    // Array data (DOUBLE, SINGLE, INT*, UINT*, LOGICAL, CHAR, COMPLEX)
    DataBuffer *buffer = nullptr;

    // Extended data — allocated only for CELL, STRING, CSL, STRUCT, FUNC_HANDLE.
    // CELL / STRING / CSL element slots flow through `alloc` via `keel::Array<Value>`.
    // STRUCT layout: array of structs (AoS). `structArray` holds `numel()` per-element
    // field maps; a single struct has `structArray->size() == 1`.
    keel::Array<Value> *cellData = nullptr;
    keel::Array<keel::HashMap<keel::String, Value>> *structArray = nullptr;
    // Insertion-order tracker for struct fields so fieldnames() preserves creation order.
    keel::Array<keel::String> *fieldOrder = nullptr;
    keel::String *funcName = nullptr;
    // Reconstructed source text for anonymous-function handles (e.g. "@(x)x+1").
    keel::String *funcSource = nullptr;

    // OBJECT (class instance / array)
    keel::String *objClass = nullptr;
    keel::Array<keel::Arc<ObjectState>> objStates;
    bool objIsHandle = false;
    bool isSingleComplex = false;

    // Capacity for appendScalar amortization
    keel::usize appendCapacity = 0;

    explicit HeapObject(keel::Allocator *a = nullptr) noexcept
        : alloc(a ? a : keel::default_allocator())
        , objStates(alloc)
    {}
    ~HeapObject() noexcept;

    HeapObject(const HeapObject &) = delete;
    HeapObject &operator=(const HeapObject &) = delete;

    static HeapObject *create(keel::Allocator *alloc = nullptr);
    static void destroy(HeapObject *h) noexcept;

    template <typename T, typename... Args>
    T *allocNew(Args &&...args)
    {
        keel::Allocator *a = alloc ? alloc : keel::default_allocator();
        void *mem = a->try_alloc_bytes(sizeof(T), alignof(T));
        if (!mem)
            throw std::bad_alloc();
        try {
            return ::new (mem) T(keel::forward<Args>(args)...);
        } catch (...) {
            a->free_bytes(mem, sizeof(T), alignof(T));
            throw;
        }
    }

    template <typename T>
    void allocDelete(T *p) noexcept
    {
        if (!p)
            return;
        keel::Allocator *a = alloc ? alloc : keel::default_allocator();
        p->~T();
        a->free_bytes(p, sizeof(T), alignof(T));
    }

    void addRef() noexcept { detail::refInc(refCount); }
    bool release() noexcept { return detail::refDecIsZero(refCount); }
    keel::i32 loadRefCount() const noexcept { return detail::refLoadAcquire(refCount); }

    // Deep clone for COW
    HeapObject *clone() const;
};

class HeapGuard
{
public:
    explicit HeapGuard(HeapObject *h = nullptr) noexcept : h_(h) {}
    ~HeapGuard() noexcept { HeapObject::destroy(h_); }

    HeapGuard(const HeapGuard &) = delete;
    HeapGuard &operator=(const HeapGuard &) = delete;

    HeapObject *get() const noexcept { return h_; }
    HeapObject *operator->() const noexcept { return h_; }
    HeapObject *release() noexcept
    {
        HeapObject *p = h_;
        h_ = nullptr;
        return p;
    }

private:
    HeapObject *h_ = nullptr;
};

} // namespace numkit
