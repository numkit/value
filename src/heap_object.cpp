// src/heap_object.cpp
#include <numkit/value/heap_object.hpp>
#include <numkit/value/object.hpp>
#include <numkit/value/value.hpp>

namespace numkit {

HeapObject::~HeapObject()
{
    if (buffer) {
        if (buffer->release())
            delete buffer;
    }
    delete cellData;
    delete structArray;
    delete fieldOrder;
    delete funcName;
    delete funcSource;
    delete objClass;
}

HeapObject *HeapObject::clone() const
{
    auto *h = new HeapObject();
    h->type = type;
    h->dims = dims;
    h->alloc = alloc;
    h->isSingleComplex = isSingleComplex;
    keel::Allocator *ca = alloc ? alloc : keel::default_allocator();
    if (buffer) {
        const keel::usize es = elementSize(type);
        const keel::usize copyBytes =
            (es > 0) ? keel::min(dims.numel() * es, buffer->bytes()) : buffer->bytes();
        if (copyBytes > 0) {
            h->buffer = new DataBuffer(copyBytes, alloc);
            keel::mem_copy(h->buffer->data(), buffer->data(), copyBytes);
        }
    }
    if (cellData)
        h->cellData = new keel::Array<Value>(cellData->clone(ca));
    if (structArray)
        h->structArray = new std::vector<std::map<std::string, Value>>(*structArray);
    if (fieldOrder)
        h->fieldOrder = new std::vector<std::string>(*fieldOrder);
    if (funcName)
        h->funcName = new std::string(*funcName);
    if (funcSource)
        h->funcSource = new std::string(*funcSource);
    if (objClass) {
        h->objClass = new std::string(*objClass);
        h->objIsHandle = objIsHandle;
        h->objStates.reserve(objStates.size());
        for (const auto &st : objStates) {
            if (objIsHandle || !st)
                h->objStates.push_back(st);
            else
                h->objStates.push_back(st->deepCopy(ca));
        }
    }
    return h;
}

} // namespace numkit
