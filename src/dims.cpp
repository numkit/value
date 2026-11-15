// src/dims.cpp
#include <numkit/value/dims.hpp>

#include <keel/format.h>
#include <keel/memory.h>

#include <new>
#include <stdexcept>

namespace numkit {

Dims::Dims() : heap_(nullptr), nd_(2)
{
    inline_[0] = 0;
    inline_[1] = 0;
    inline_[2] = 1;
    inline_[3] = 1;
}

Dims::Dims(keel::usize r, keel::usize c) : heap_(nullptr), nd_(2)
{
    inline_[0] = r;
    inline_[1] = c;
    inline_[2] = 1;
    inline_[3] = 1;
}

Dims::Dims(keel::usize r, keel::usize c, keel::usize p) : heap_(nullptr), nd_(3)
{
    inline_[0] = r;
    inline_[1] = c;
    inline_[2] = p;
    inline_[3] = 1;
}

Dims::Dims(const keel::usize *dims, keel::i32 nd) : heap_(nullptr)
{
    setND(dims, nd);
}

void Dims::setND(const keel::usize *dims, keel::i32 nd)
{
    if (nd < 1) nd = 1;
    if (nd > kMaxRank)
        throw std::runtime_error(
            keel::format_fixed<96>("Maximum array dimensionality ({}) exceeded", kMaxRank).c_str());
    nd_ = static_cast<keel::u8>(nd);
    for (keel::i32 i = 0; i < kInlineCap; ++i)
        inline_[i] = (i < nd) ? dims[i] : 1;
    if (nd > kInlineCap) {
        const keel::usize extra = static_cast<keel::usize>(nd - kInlineCap);
        heap_ = keel::default_allocator()->try_alloc_array<keel::usize>(extra);
        if (!heap_)
            throw std::bad_alloc();
        keel::mem_copy(heap_, dims + kInlineCap, extra * sizeof(keel::usize));
    }
}

void Dims::freeHeap() noexcept
{
    if (heap_) {
        const keel::usize extra = static_cast<keel::usize>(nd_ - kInlineCap);
        keel::default_allocator()->free_array(heap_, extra);
        heap_ = nullptr;
    }
}

Dims::Dims(const Dims &o) : heap_(nullptr), nd_(o.nd_)
{
    keel::mem_copy(inline_, o.inline_, sizeof(inline_));
    if (o.heap_) {
        const keel::usize extra = static_cast<keel::usize>(nd_ - kInlineCap);
        heap_ = keel::default_allocator()->try_alloc_array<keel::usize>(extra);
        if (!heap_)
            throw std::bad_alloc();
        keel::mem_copy(heap_, o.heap_, extra * sizeof(keel::usize));
    }
}

Dims::Dims(Dims &&o) noexcept : heap_(o.heap_), nd_(o.nd_)
{
    keel::mem_copy(inline_, o.inline_, sizeof(inline_));
    o.heap_ = nullptr;
}

Dims &Dims::operator=(const Dims &o)
{
    if (this == &o) return *this;
    freeHeap();
    nd_ = o.nd_;
    keel::mem_copy(inline_, o.inline_, sizeof(inline_));
    if (o.heap_) {
        const keel::usize extra = static_cast<keel::usize>(nd_ - kInlineCap);
        heap_ = keel::default_allocator()->try_alloc_array<keel::usize>(extra);
        if (!heap_)
            throw std::bad_alloc();
        keel::mem_copy(heap_, o.heap_, extra * sizeof(keel::usize));
    }
    return *this;
}

Dims &Dims::operator=(Dims &&o) noexcept
{
    if (this == &o) return *this;
    freeHeap();
    nd_     = o.nd_;
    keel::mem_copy(inline_, o.inline_, sizeof(inline_));
    heap_   = o.heap_;
    o.heap_ = nullptr;
    return *this;
}

Dims::~Dims()
{
    freeHeap();
}

keel::usize Dims::sub2indChecked(keel::usize r, keel::usize c) const
{
    if (r >= inline_[0] || c >= inline_[1])
        throw std::runtime_error(
            keel::format_fixed<96>("Index ({},{}) exceeds dimensions", r + 1, c + 1).c_str());
    return c * inline_[0] + r;
}

keel::usize Dims::sub2indChecked(keel::usize r, keel::usize c, keel::usize p) const
{
    if (r >= inline_[0] || c >= inline_[1] || p >= inline_[2])
        throw std::runtime_error("Index exceeds dimensions");
    return p * inline_[0] * inline_[1] + c * inline_[0] + r;
}

bool Dims::operator==(const Dims &o) const noexcept
{
    if (!keel::mem_eq(inline_, o.inline_, sizeof(inline_)))
        return false;
    const keel::i32 n = keel::max(static_cast<keel::i32>(nd_), static_cast<keel::i32>(o.nd_));
    for (keel::i32 i = kInlineCap; i < n; ++i)
        if (dim(i) != o.dim(i)) return false;
    return true;
}

} // namespace numkit
