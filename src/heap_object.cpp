// src/heap_object.cpp
#include <numkit/value/heap_object.hpp>
#include <numkit/value/object.hpp>
#include <numkit/value/value.hpp>

namespace numkit {

HeapObject *HeapObject::create(keel::Allocator *alloc)
{
    keel::Allocator *a = alloc ? alloc : keel::default_allocator();
    void *mem = a->try_alloc_bytes(sizeof(HeapObject), alignof(HeapObject));
    if (!mem)
        throw std::bad_alloc();
    return ::new (mem) HeapObject(a);
}

void HeapObject::destroy(HeapObject *h) noexcept
{
    if (!h)
        return;
    keel::Allocator *a = h->alloc ? h->alloc : keel::default_allocator();
    h->~HeapObject();
    a->free_bytes(h, sizeof(HeapObject), alignof(HeapObject));
}

HeapObject::~HeapObject() noexcept
{
    if (buffer) {
        if (buffer->release())
            DataBuffer::destroy(buffer);
    }
    allocDelete(cellData);
    allocDelete(structArray);
    allocDelete(fieldOrder);
    allocDelete(funcName);
    allocDelete(funcSource);
    allocDelete(objClass);
}

HeapObject *HeapObject::clone() const
{
    keel::Allocator *ca = alloc ? alloc : keel::default_allocator();
    HeapGuard h(HeapObject::create(ca));
    h->type = type;
    h->dims = dims;
    h->isSingleComplex = isSingleComplex;
    if (buffer) {
        const keel::usize es = elementSize(type);
        const keel::usize copyBytes =
            (es > 0) ? keel::min(dims.numel() * es, buffer->bytes()) : buffer->bytes();
        if (copyBytes > 0) {
            h->buffer = DataBuffer::create(copyBytes, ca);
            keel::mem_copy(h->buffer->data(), buffer->data(), copyBytes);
        }
    }
    if (cellData)
        h->cellData = h->allocNew<keel::Array<Value>>(cellData->clone(ca));
    if (structArray)
        h->structArray = h->allocNew<keel::Array<keel::HashMap<keel::String, Value>>>(structArray->clone(ca));
    if (fieldOrder)
        h->fieldOrder = h->allocNew<keel::Array<keel::String>>(fieldOrder->clone(ca));
    if (funcName)
        h->funcName = h->allocNew<keel::String>(funcName->clone(ca));
    if (funcSource)
        h->funcSource = h->allocNew<keel::String>(funcSource->clone(ca));
    if (objClass) {
        h->objClass = h->allocNew<keel::String>(objClass->clone(ca));
        h->objIsHandle = objIsHandle;
        h->objStates.reserve(objStates.size());
        for (const auto &st : objStates) {
            if (objIsHandle || !st)
                h->objStates.push_back(st);
            else
                h->objStates.push_back(st->deepCopy(ca));
        }
    }
    return h.release();
}

} // namespace numkit
