// include/numkit/value/data_buffer.hpp
#pragma once

#include <keel/memory.h>
#include <keel/smart_ptr.h>

namespace numkit {

namespace detail {

using RefCount = keel::detail::AtomicRefCount;

inline void refInc(RefCount &r) noexcept
{
    (void)keel::detail::arc_inc(r);
}

inline bool refDecIsZero(RefCount &r) noexcept
{
    return keel::detail::arc_dec(r) == 0u;
}

inline keel::i32 refLoadAcquire(const RefCount &r) noexcept
{
#if defined(_MSC_VER) && !defined(__clang__)
    return keel::as<keel::i32>(keel::detail::arc_load(r));
#else
    return keel::as<keel::i32>(__atomic_load_n(&r, __ATOMIC_ACQUIRE));
#endif
}

} // namespace detail

// ============================================================
// DataBuffer — raw memory block with atomic ref-counting
//
// Allocated via a `keel::Allocator*` passed in at construction.
// The same allocator is used for deallocation in the destructor — caller
// must keep it alive while the DataBuffer (and any cloned chain) lives.
//
// nullptr `alloc` is mapped to `keel::default_allocator()` so engine-free
// Value factories (`Value::matrix(..., nullptr)`) work out of the box.
//
// Non-owning mode: construct via the ViewTag ctor to wrap an external
// pointer without ownership; destructor will not deallocate. Used by
// Value::view() for zero-copy borrowing of caller-managed buffers.
// ============================================================
class DataBuffer
{
public:
    struct ViewTag {};

    DataBuffer(keel::usize bytes, keel::Allocator *alloc = nullptr);
    /// Non-owning ctor: wraps `external` for `bytes` bytes. Destructor
    /// will not free; caller manages lifetime.
    DataBuffer(ViewTag, void *external, keel::usize bytes) noexcept;
    ~DataBuffer() noexcept;

    static DataBuffer *create(keel::usize bytes, keel::Allocator *alloc = nullptr);
    static DataBuffer *createView(void *external, keel::usize bytes);
    static void destroy(DataBuffer *buf) noexcept;

    DataBuffer(const DataBuffer &) = delete;
    DataBuffer &operator=(const DataBuffer &) = delete;

    void *data() noexcept { return data_; }
    const void *data() const noexcept { return data_; }
    keel::usize bytes() const noexcept { return bytes_; }
    keel::Allocator *allocator() const noexcept { return alloc_; }
    bool owns() const noexcept { return owns_; }

    void addRef() noexcept;
    bool release() noexcept;
    keel::i32 refCount() const noexcept;

private:
    void *data_ = nullptr;
    keel::usize bytes_ = 0;
    detail::RefCount refCount_{1};
    keel::Allocator *alloc_ = nullptr;
    bool owns_ = true;
};

} // namespace numkit
