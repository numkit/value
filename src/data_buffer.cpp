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

DataBuffer::~DataBuffer()
{
    if (data_ && owns_)
        alloc_->free_bytes(data_, bytes_, bufferAlign(bytes_));
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
