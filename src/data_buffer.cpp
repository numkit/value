// src/data_buffer.cpp
#include <numkit/value/data_buffer.hpp>

#include <new>

namespace numkit {

namespace {
constexpr keel::usize bufferAlign(keel::usize bytes) noexcept
{
    return (bytes >= keel::kSimdAlign) ? keel::kSimdAlign : keel::kDefaultAlign;
}
} // namespace

DataBuffer::DataBuffer(keel::usize bytes, keel::Allocator *alloc)
    : bytes_(bytes)
    , refCount_(1)
    , alloc_(alloc ? alloc : keel::default_allocator())
    , owns_(true)
{
    if (bytes > 0) {
        data_ = alloc_->try_alloc_bytes(bytes, bufferAlign(bytes));
        if (!data_)
            throw std::bad_alloc();
    }
}

DataBuffer::DataBuffer(ViewTag, void *external, keel::usize bytes) noexcept
    : data_(external)
    , bytes_(bytes)
    , refCount_(1)
    , alloc_(keel::default_allocator())
    , owns_(false)
{}

DataBuffer::~DataBuffer() noexcept
{
    if (data_ && owns_)
        alloc_->free_bytes(data_, bytes_, bufferAlign(bytes_));
}

DataBuffer *DataBuffer::create(keel::usize bytes, keel::Allocator *alloc)
{
    keel::Allocator *a = alloc ? alloc : keel::default_allocator();
    void *mem = a->try_alloc_bytes(sizeof(DataBuffer), alignof(DataBuffer));
    if (!mem)
        throw std::bad_alloc();
    try {
        return ::new (mem) DataBuffer(bytes, a);
    } catch (...) {
        a->free_bytes(mem, sizeof(DataBuffer), alignof(DataBuffer));
        throw;
    }
}

DataBuffer *DataBuffer::createView(void *external, keel::usize bytes)
{
    keel::Allocator *a = keel::default_allocator();
    void *mem = a->try_alloc_bytes(sizeof(DataBuffer), alignof(DataBuffer));
    if (!mem)
        throw std::bad_alloc();
    return ::new (mem) DataBuffer(ViewTag{}, external, bytes);
}

void DataBuffer::destroy(DataBuffer *buf) noexcept
{
    if (!buf)
        return;
    keel::Allocator *a = buf->alloc_;
    buf->~DataBuffer();
    a->free_bytes(buf, sizeof(DataBuffer), alignof(DataBuffer));
}

void DataBuffer::addRef() noexcept
{
    detail::refInc(refCount_);
}

bool DataBuffer::release() noexcept
{
    return detail::refDecIsZero(refCount_);
}

keel::i32 DataBuffer::refCount() const noexcept
{
    return detail::refLoadAcquire(refCount_);
}

} // namespace numkit
