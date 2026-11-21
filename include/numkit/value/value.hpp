#pragma once

#include <numkit/value/heap_object.hpp>

#include <keel/complex.h>
#include <keel/md_ref.h>
#include <keel/ref.h>

#include <initializer_list>

namespace numkit {

namespace detail {
/// Compile-time map from C++ element type -> `numkit::ValueType`.
template<class T> struct value_type_for;

template<> struct value_type_for<keel::f64> { static constexpr ValueType type = ValueType::DOUBLE;  };
template<> struct value_type_for<keel::f32> { static constexpr ValueType type = ValueType::SINGLE;  };
template<> struct value_type_for<keel::i8>  { static constexpr ValueType type = ValueType::INT8;    };
template<> struct value_type_for<keel::i16> { static constexpr ValueType type = ValueType::INT16;   };
template<> struct value_type_for<keel::i32> { static constexpr ValueType type = ValueType::INT32;   };
template<> struct value_type_for<keel::i64> { static constexpr ValueType type = ValueType::INT64;   };
template<> struct value_type_for<keel::u8>  { static constexpr ValueType type = ValueType::UINT8;   };
template<> struct value_type_for<keel::u16> { static constexpr ValueType type = ValueType::UINT16;  };
template<> struct value_type_for<keel::u32> { static constexpr ValueType type = ValueType::UINT32;  };
template<> struct value_type_for<keel::u64> { static constexpr ValueType type = ValueType::UINT64;  };
template<> struct value_type_for<keel::c64> { static constexpr ValueType type = ValueType::COMPLEX; };

// Zero-header support for binary64 std::complex<double> ranges when callers use Eigen / STL complex.
template <class C>
    requires(!keel::IsSame<keel::RemoveCvRef<C>, keel::c64>)
         && keel::IsTriviallyCopyable<C>
         && (sizeof(C) == 2 * sizeof(keel::f64))
         && requires(const C &z) {
                typename C::value_type;
                requires keel::IsSame<keel::RemoveCv<typename C::value_type>, keel::f64>;
                requires keel::IsSame<keel::RemoveCvRef<decltype(z.real())>, keel::f64>;
                requires keel::IsSame<keel::RemoveCvRef<decltype(z.imag())>, keel::f64>;
            }
struct value_type_for<C> { static constexpr ValueType type = ValueType::COMPLEX; };

template <class T> inline constexpr bool is_array_ref_v = false;
template <class U> inline constexpr bool is_array_ref_v<keel::ArrayRef<U>> = true;

template <class T> inline constexpr bool is_md_ref_v = false;
template <class U, keel::usize R, keel::IsMdLayout L> inline constexpr bool is_md_ref_v<keel::MdRef<U, R, L>> = true;
template <class U, keel::usize R, keel::IsMdLayout L> inline constexpr bool is_md_ref_v<keel::MutMdRef<U, R, L>> = true;
template <class U, keel::usize R> inline constexpr bool is_md_ref_v<keel::StridedMdRef<U, R>> = true;
template <class U, keel::usize R> inline constexpr bool is_md_ref_v<keel::MutStridedMdRef<U, R>> = true;

template <class T> inline constexpr bool is_native_value_elem_v = false;
template <> inline constexpr bool is_native_value_elem_v<keel::f64> = true;
template <> inline constexpr bool is_native_value_elem_v<keel::f32> = true;
template <> inline constexpr bool is_native_value_elem_v<keel::i8>  = true;
template <> inline constexpr bool is_native_value_elem_v<keel::i16> = true;
template <> inline constexpr bool is_native_value_elem_v<keel::i32> = true;
template <> inline constexpr bool is_native_value_elem_v<keel::i64> = true;
template <> inline constexpr bool is_native_value_elem_v<keel::u8>  = true;
template <> inline constexpr bool is_native_value_elem_v<keel::u16> = true;
template <> inline constexpr bool is_native_value_elem_v<keel::u32> = true;
template <> inline constexpr bool is_native_value_elem_v<keel::u64> = true;
template <> inline constexpr bool is_native_value_elem_v<keel::c64> = true;
} // namespace detail

// ============================================================
// Value — 16-byte tagged pointer value
//
// Layout:
//   keel::f64   scalar_ (8 bytes) — inline scalar value
//   HeapObject *heap_   (8 bytes) — tag / heap pointer
// ============================================================
class Value
{
public:
    Value();
    ~Value() noexcept;

    Value(const Value &other);
    Value &operator=(const Value &other);
    Value(Value &&other) noexcept;
    Value &operator=(Value &&other) noexcept;

    void swap(Value &other) noexcept;

    // ── Ergonomic ctors — owning, COPY ────────────────────────
    // Build a 1xN row vector from common C++ / keel sources.

    Value(std::initializer_list<keel::f64> il,
          keel::Allocator *alloc = nullptr);

    template<class T>
        requires requires { detail::value_type_for<T>::type; }
    Value(keel::ArrayRef<T> data,
          keel::Allocator *alloc = nullptr);

    template<class R>
        requires(!keel::IsSame<keel::RemoveCvRef<R>, Value>)
             && (!detail::is_array_ref_v<keel::RemoveCvRef<R>>)
             && (!detail::is_md_ref_v<keel::RemoveCvRef<R>>)
             && requires(const R &r) {
                    r.data();
                    r.size();
                    requires keel::IsPointer<decltype(r.data())>;
                    detail::value_type_for<keel::RemoveCv<keel::RemovePointer<decltype(r.data())>>>::type;
                }
    Value(const R &range,
          keel::Allocator *alloc = nullptr);

    /// Non-owning view over an externally-owned buffer.
    static Value view(const void *data,
                      ValueType type,
                      Dims dims);

    /// Zero-copy non-owning view over a column-major keel::MdRef (LayoutLeft only).
    template <class T, keel::usize Rank>
        requires detail::is_native_value_elem_v<T>
    static Value view(keel::MdRef<T, Rank, keel::LayoutLeft> src);

    template <class T, keel::usize Rank>
        requires detail::is_native_value_elem_v<T>
    static Value view(keel::MutMdRef<T, Rank, keel::LayoutLeft> src)
    {
        return view(src.as_ref());
    }

    /// Owning copy from any keel::MdRef (preserves logical shape; converts LayoutRight -> column-major).
    template <class T, keel::usize Rank, keel::IsMdLayout Layout>
        requires detail::is_native_value_elem_v<T>
    static Value fromMdRef(keel::MdRef<T, Rank, Layout> src,
                           keel::Allocator *alloc = nullptr);

    template <class T, keel::usize Rank, keel::IsMdLayout Layout>
        requires detail::is_native_value_elem_v<T>
    static Value fromMdRef(keel::MutMdRef<T, Rank, Layout> src,
                           keel::Allocator *alloc = nullptr)
    {
        return fromMdRef(src.as_ref(), alloc);
    }

    /// Empty 0x0 DOUBLE matrix sentinel (`[]`).
    static const Value Empty;

    // ── Factories — real ─────────────────────────────────────
    static Value scalar(keel::f64 v, keel::Allocator *alloc = nullptr);
    static Value logicalScalar(bool v, keel::Allocator *alloc = nullptr);
    static Value matrix(keel::usize rows,
                        keel::usize cols,
                        ValueType t = ValueType::DOUBLE,
                        keel::Allocator *alloc = nullptr);
    static Value matrix3d(keel::usize rows,
                          keel::usize cols,
                          keel::usize pages,
                          ValueType t = ValueType::DOUBLE,
                          keel::Allocator *alloc = nullptr);
    static Value matrixND(const keel::usize *dims,
                          keel::i32 nd,
                          ValueType t = ValueType::DOUBLE,
                          keel::Allocator *alloc = nullptr);
    static Value fromString(keel::StringRef s, keel::Allocator *alloc = nullptr);
    static Value cell(keel::usize rows, keel::usize cols, keel::Allocator *alloc = nullptr);
    static Value cell3D(keel::usize rows, keel::usize cols, keel::usize pages, keel::Allocator *alloc = nullptr);
    static Value cellND(const keel::usize *dims, keel::i32 nd, keel::Allocator *alloc = nullptr);
    static Value csl(keel::usize n, keel::Allocator *alloc = nullptr);
    static Value structure(keel::Allocator *alloc = nullptr);
    static Value structArray(keel::usize rows, keel::usize cols,
                             keel::Allocator *alloc = nullptr);
    static Value funcHandle(keel::StringRef name, keel::Allocator *alloc = nullptr);
    static Value object(keel::StringRef className,
                        keel::Arc<ObjectState> state, bool isHandle,
                        keel::Allocator *alloc = nullptr);
    static Value objectArray(keel::StringRef className, const Dims &dims,
                             keel::Array<keel::Arc<ObjectState>> states, bool isHandle,
                             keel::Allocator *alloc = nullptr);
    static Value deleted();

    // ── Factories — compound operations ────────────────────────
    static Value colonRange(keel::f64 start, keel::f64 stop, keel::Allocator *alloc = nullptr);
    static Value colonRange(keel::f64 start, keel::f64 step, keel::f64 stop, keel::Allocator *alloc = nullptr);
    static Value colonRangeTyped(keel::f64 start, keel::f64 stop, ValueType t,
                                 keel::Allocator *alloc = nullptr);
    static Value colonRangeTyped(keel::f64 start, keel::f64 step, keel::f64 stop,
                                 ValueType t, keel::Allocator *alloc = nullptr);
    static keel::usize colonCount(keel::f64 start, keel::f64 step, keel::f64 stop);

    static Value horzcat(const Value *elems, keel::usize count, keel::Allocator *alloc = nullptr);
    static Value vertcat(const Value *elems, keel::usize count, keel::Allocator *alloc = nullptr);

    // ── Type-preserving indexing ─────────────────────────────
    Value elemAt(keel::usize linearIdx, keel::Allocator *alloc = nullptr) const;
    keel::f64 elemAsDouble(keel::usize idx) const;
    Value indexGet(const keel::usize *indices, keel::usize count, keel::Allocator *alloc = nullptr) const;
    Value reshape(keel::usize rows, keel::usize cols, keel::Allocator *alloc = nullptr) const;
    Value indexGet2D(const keel::usize *rowIdx, keel::usize nrows,
                     const keel::usize *colIdx, keel::usize ncols, keel::Allocator *alloc = nullptr) const;
    Value indexGet3D(const keel::usize *rowIdx, keel::usize nrows,
                     const keel::usize *colIdx, keel::usize ncols,
                     const keel::usize *pageIdx, keel::usize npages, keel::Allocator *alloc = nullptr) const;
    Value indexGetND(const keel::usize *const *perDimIdx,
                     const keel::usize *perDimCount,
                     keel::i32 nd,
                     keel::Allocator *alloc = nullptr) const;
    Value logicalIndex(const keel::u8 *mask, keel::usize maskLen, keel::Allocator *alloc = nullptr) const;

    // ── Index resolution ────────────────────────────────────
    static keel::Array<keel::usize> resolveIndices(const Value &idx, keel::usize dimSize,
                                                   keel::Allocator *alloc = nullptr);
    static keel::Array<keel::usize> resolveIndicesUnchecked(const Value &idx,
                                                            keel::Allocator *alloc = nullptr);
    static keel::usize checkedScalarIndex(keel::f64 oneBased);

    // ── Type-preserving indexed assignment ──────────────────
    void elemSet(keel::usize linearIdx, const Value &val);
    void indexSet(const keel::usize *indices, keel::usize count, const Value &val);
    void indexSet2D(const keel::usize *rowIdx, keel::usize nrows,
                    const keel::usize *colIdx, keel::usize ncols,
                    const Value &val);
    void indexSet3D(const keel::usize *rowIdx, keel::usize nrows,
                    const keel::usize *colIdx, keel::usize ncols,
                    const keel::usize *pageIdx, keel::usize npages,
                    const Value &val);
    void indexSetND(const keel::usize *const *perDimIdx,
                    const keel::usize *perDimCount,
                    keel::i32 nd,
                    const Value &val);

    // ── Type-preserving deletion (v(idx) = []) ─────────────
    void indexDelete(const keel::usize *indices, keel::usize count, keel::Allocator *alloc = nullptr);
    void indexDelete2D(const keel::usize *rowIdx, keel::usize nrows,
                       const keel::usize *colIdx, keel::usize ncols,
                       keel::Allocator *alloc = nullptr);
    void indexDelete3D(const keel::usize *rowIdx, keel::usize nrows,
                       const keel::usize *colIdx, keel::usize ncols,
                       const keel::usize *pageIdx, keel::usize npages,
                       keel::Allocator *alloc = nullptr);
    void indexDeleteND(const keel::usize *const *perDimIdx,
                       const keel::usize *perDimCount,
                       keel::i32 nd,
                       keel::Allocator *alloc = nullptr);

    // ── Factories — complex ──────────────────────────────────
    static Value complexScalar(keel::c64 v, keel::Allocator *alloc = nullptr);
    static Value complexScalar(keel::f64 re, keel::f64 im, keel::Allocator *alloc = nullptr);
    static Value complexMatrix(keel::usize rows, keel::usize cols, keel::Allocator *alloc = nullptr);

    // ── Factories — string ───────────────────────────────────
    static Value stringScalar(keel::StringRef s, keel::Allocator *alloc = nullptr);
    static Value stringArray(keel::usize rows, keel::usize cols, keel::Allocator *alloc = nullptr);
    static Value stringArray3D(keel::usize rows, keel::usize cols, keel::usize pages, keel::Allocator *alloc = nullptr);
    static Value stringArrayND(const keel::usize *dims, keel::i32 nd, keel::Allocator *alloc = nullptr);

    // ── String accessors ────────────────────────────────────
    keel::StringRef stringElemRef(keel::usize i) const;
    keel::String stringElem(keel::usize i, keel::Allocator *alloc = nullptr) const;
    void stringElemSet(keel::usize i, keel::StringRef s);

    // ── Type queries ─────────────────────────────────────────
    ValueType type() const;
    const Dims &dims() const;
    keel::usize numel() const;
    bool isScalar() const;
    bool isEmpty() const;
    bool isNumeric() const;

    bool isUnset() const { return heap_ == emptyTag(); }
    bool isDeleted() const { return heap_ == deletedTag(); }
    bool isComplex() const;
    bool isSingleComplex() const;
    void setSingleComplex(bool v);
    bool isLogical() const;
    bool isChar() const;
    bool isCell() const;
    bool isStruct() const;
    bool isFuncHandle() const;
    bool isString() const;
    bool isObject() const;
    bool isCsl() const;

    // ── OBJECT accessors (see object.hpp) ────────────────────
    keel::StringRef objectClassNameRef() const;
    keel::String objectClassName(keel::Allocator *alloc = nullptr) const;
    bool objectIsHandle() const;
    keel::usize objectCount() const;
    const ObjectState *objectStateConst() const;
    ObjectState *objectStateMut();
    const ObjectState *objectStateAt(keel::usize i) const;
    ObjectState *objectStateMutAt(keel::usize i);

    Value objectSubArray(keel::ArrayRef<keel::usize> idxs,
                         keel::Allocator *alloc = nullptr) const;
    Value objectReshape(const Dims &newDims, keel::Allocator *alloc = nullptr) const;
    Value objectTranspose(keel::Allocator *alloc = nullptr) const;
    Value objectGather(const keel::usize *srcLinear, const Dims &resultDims,
                       keel::Allocator *alloc = nullptr) const;
    static Value objectConcatN(const Value *parts, keel::usize count, const Dims &outDims,
                               keel::Allocator *alloc = nullptr);
    void objectAssignLinear(keel::ArrayRef<keel::usize> pos, const Value &rhs,
                            const Value &fill, keel::Allocator *alloc = nullptr);
    void objectAssignND(keel::ArrayRef<keel::Array<keel::usize>> perDim, const Value &rhs,
                        const Value &fill, keel::Allocator *alloc = nullptr);

    // ── Const raw access ─────────────────────────────────────
    const void *rawData() const;
    keel::usize rawBytes() const;
    keel::usize deepBytes() const;

    // ── Const typed access — double ──────────────────────────
    const keel::f64 *doubleData() const;
    const keel::u8  *logicalData() const;
    const char      *charData() const;
    keel::f64 toScalar() const;
    bool toBool() const;
    keel::StringRef asStringRef() const;
    keel::String toString(keel::Allocator *alloc = nullptr) const;

    // ── Const typed access — single ────────────────────────────
    const keel::f32 *singleData() const;

    // ── Const typed access — integer types ───────────────────
    const keel::i8  *int8Data()  const;
    const keel::i16 *int16Data() const;
    const keel::i32 *int32Data() const;
    const keel::i64 *int64Data() const;
    const keel::u8  *uint8Data()  const;
    const keel::u16 *uint16Data() const;
    const keel::u32 *uint32Data() const;
    const keel::u64 *uint64Data() const;

    // ── Const typed access — complex ─────────────────────────
    const keel::c64 *complexData() const;
    keel::c64 toComplex() const;
    keel::c64 complexElem(keel::usize i) const;
    keel::c64 complexElem(keel::usize r, keel::usize c) const;

    // ── Mutable typed access (calls detach for COW) ──────────
    keel::f64 *doubleDataMut();
    keel::f32 *singleDataMut();
    keel::u8  *logicalDataMut();
    char      *charDataMut();
    void      *rawDataMut();
    keel::c64 *complexDataMut();
    keel::i8  *int8DataMut();
    keel::i16 *int16DataMut();
    keel::i32 *int32DataMut();
    keel::i64 *int64DataMut();
    keel::u8  *uint8DataMut();
    keel::u16 *uint16DataMut();
    keel::u32 *uint32DataMut();
    keel::u64 *uint64DataMut();

    // ── Zero-copy typed keel views (column-major LayoutLeft) ─
    template <class T>
        requires detail::is_native_value_elem_v<T>
    keel::ArrayRef<T> asArrayRef() const;

    template <class T>
        requires detail::is_native_value_elem_v<T>
    keel::MutArrayRef<T> asMutArrayRef();

    template <class T, keel::usize Rank>
        requires detail::is_native_value_elem_v<T> && (Rank >= 1)
    keel::MdRef<T, Rank, keel::LayoutLeft> asMdRef() const;

    template <class T, keel::usize Rank>
        requires detail::is_native_value_elem_v<T> && (Rank >= 1)
    keel::MutMdRef<T, Rank, keel::LayoutLeft> asMutMdRef();

    // ── Const indexing (column-major) ────────────────────────
    keel::f64 operator()(keel::usize i) const;
    keel::f64 operator()(keel::usize r, keel::usize c) const;
    keel::f64 operator()(keel::usize r, keel::usize c, keel::usize p) const;

    // ── Mutable indexing (calls detach) ──────────────────────
    keel::f64 &elem(keel::usize i);
    keel::f64 &elem(keel::usize r, keel::usize c);
    keel::f64 &elem(keel::usize r, keel::usize c, keel::usize p);

    // ── Char element access ──────────────────────────────────
    char charElem(keel::usize i) const;
    char &charElemMut(keel::usize i);
    keel::String charRow(keel::usize r, keel::Allocator *alloc = nullptr) const;

    // ── Resize ───────────────────────────────────────────────
    void resize(keel::usize newRows, keel::usize newCols, keel::Allocator *alloc = nullptr);
    void resize3d(keel::usize newRows, keel::usize newCols, keel::usize newPages, keel::Allocator *alloc = nullptr);
    void resizeND(const keel::usize *newDims, keel::i32 nd, keel::Allocator *alloc = nullptr);
    void ensureSize(keel::usize linearIdx, keel::Allocator *alloc = nullptr);
    void appendScalar(keel::f64 v, keel::Allocator *alloc = nullptr);
    void appendScalarCol(keel::f64 v, keel::Allocator *alloc = nullptr);

    // ── Promote double → complex ─────────────────────────────
    void promoteToComplex(keel::Allocator *alloc = nullptr);

    // ── Cell ─────────────────────────────────────────────────
    Value &cellAt(keel::usize i);
    const Value &cellAt(keel::usize i) const;
    // ── CSL (comma-separated list) ───────────────────────────
    keel::usize cslCount() const;
    Value &cslAt(keel::usize i) { return cellAt(i); }
    const Value &cslAt(keel::usize i) const { return cellAt(i); }
    keel::usize growCellTo(const keel::usize *coords, keel::i32 nd,
                           keel::Allocator *alloc = nullptr);
    keel::Array<Value> &cellDataVec();
    const keel::Array<Value> &cellDataVec() const;

    // ── Struct ───────────────────────────────────────────────
    Value &field(keel::StringRef name);
    const Value &field(keel::StringRef name) const;
    bool hasField(keel::StringRef name) const;
    keel::HashMap<keel::String, Value> &structFields();
    const keel::HashMap<keel::String, Value> &structFields() const;

    void setField(keel::usize linearIdx, keel::StringRef name, const Value &v);
    void setFieldAll(keel::StringRef name, const Value &v);
    void removeField(keel::StringRef name);
    keel::Array<keel::String> fieldNamesInOrder(keel::Allocator *alloc = nullptr) const;

    bool isStructArray() const;
    keel::HashMap<keel::String, Value> &structArrayElem(keel::usize i);
    void growStructArrayTo(keel::usize idx, keel::Allocator *alloc = nullptr);
    keel::usize growStructArrayND(const keel::usize *coords, keel::i32 nd,
                                  keel::Allocator *alloc = nullptr);
    const keel::HashMap<keel::String, Value> &structArrayElem(keel::usize i) const;

    // ── Func handle ──────────────────────────────────────────
    keel::StringRef funcHandleNameRef() const;
    keel::String funcHandleName(keel::Allocator *alloc = nullptr) const;
    keel::StringRef funcHandleSourceRef() const;
    keel::String funcHandleSource(keel::Allocator *alloc = nullptr) const;
    void setFuncHandleSource(keel::StringRef source);
    keel::usize funcHandleCaptureCount() const;
    const Value &funcHandleCapture(keel::usize i) const;
    void setFuncHandleCaptures(const Value *captures, keel::usize count);

    // ── Debug ────────────────────────────────────────────────
    keel::String debugString(keel::Allocator *alloc = nullptr) const;
    keel::String formatDisplay(keel::StringRef name, keel::Allocator *alloc = nullptr) const;

    // ── Fast scalar access for VM ────────────────────────────
    bool isDoubleScalar() const { return heap_ == nullptr; }
    keel::f64 scalarVal() const { return scalar_; }
    void setScalarVal(keel::f64 v)
    {
        releaseHeap();
        scalar_ = v;
        heap_ = nullptr;
    }
    void setScalarFast(keel::f64 v)
    {
        if (isHeap())
            releaseHeap();
        scalar_ = v;
        heap_ = nullptr;
    }

    void setLogicalFast(bool v)
    {
        if (isHeap())
            releaseHeap();
        scalar_ = 0.0;
        heap_ = v ? logicalTrueTag() : logicalFalseTag();
    }

    bool isLogicalScalar() const { return heap_ == logicalTrueTag() || heap_ == logicalFalseTag(); }
    bool isLogicalTrue() const { return heap_ == logicalTrueTag(); }
    bool isLogicalFalse() const { return heap_ == logicalFalseTag(); }

    keel::f64 fastScalarVal() const
    {
        if (heap_ == nullptr)
            return scalar_;
        return heap_ == logicalTrueTag() ? 1.0 : 0.0;
    }

    // ── Ultra-fast VM hot-path accessors ─────────────────────
    bool isHeapDouble() const { return isHeap() && heap_->type == ValueType::DOUBLE; }

    const Dims &heapDims() const { return heap_->dims; }

    keel::i32 heapRefCount() const
    {
        if (!isHeap())
            return 1;
        keel::i32 rc = heap_->loadRefCount();
        if (rc > 1)
            return rc;
        if (heap_->buffer && (!heap_->buffer->owns() || heap_->buffer->refCount() > 1))
            return 2;
        return 1;
    }

    bool hasHeap() const
    {
        return isHeap();
    }

    keel::f64 *doubleDataMutFast()
    {
        if (heap_->loadRefCount() == 1
            && heap_->buffer && heap_->buffer->owns()
            && heap_->buffer->refCount() == 1)
            return static_cast<keel::f64 *>(heap_->buffer->data());
        detach();
        return static_cast<keel::f64 *>(heap_->buffer->data());
    }

    const keel::f64 *doubleDataFast() const
    {
        return static_cast<const keel::f64 *>(heap_->buffer->data());
    }

    void swapHeapBufferUnchecked(Value &other) noexcept
    {
        DataBuffer *tmp = heap_->buffer;
        heap_->buffer = other.heap_->buffer;
        other.heap_->buffer = tmp;
        keel::usize capTmp = heap_->appendCapacity;
        heap_->appendCapacity = other.heap_->appendCapacity;
        other.heap_->appendCapacity = capTmp;
    }

private:
    // ── 16-byte layout ───────────────────────────────────────
    keel::f64 scalar_ = 0.0;
    HeapObject *heap_;

    // ── Tag constants ────────────────────────────────────────
    static HeapObject *emptyTag()        { return reinterpret_cast<HeapObject *>(keel::usize(1)); }
    static HeapObject *logicalTrueTag()  { return reinterpret_cast<HeapObject *>(keel::usize(2)); }
    static HeapObject *logicalFalseTag() { return reinterpret_cast<HeapObject *>(keel::usize(3)); }
    static HeapObject *deletedTag()      { return reinterpret_cast<HeapObject *>(keel::usize(4)); }

    // ── Internal helpers ─────────────────────────────────────
    bool isTag() const
    {
        keel::usize u = reinterpret_cast<keel::usize>(heap_);
        return u >= 1 && u <= 4;
    }
    bool isHeap() const { return reinterpret_cast<keel::usize>(heap_) > 4; }

    void releaseHeap() noexcept;
    void detach();
    HeapObject *mutableHeap();
    static Value concatStructs(const Value *elems, keel::usize count, bool vertical,
                               keel::Allocator *alloc = nullptr);
    static Value concatObjects(const Value *elems, keel::usize count, bool vertical,
                               keel::Allocator *alloc);
    void objectEnsureArrayLike(const Value &proto, keel::Allocator *alloc);

    static const Dims sScalarDims;
    static const Dims sEmptyDims;

    static Value makeContiguous(const void *src, keel::usize count,
                                ValueType type, keel::usize elemSize,
                                keel::Allocator *alloc);
    [[noreturn]] static void throwRankMismatch(keel::i32 actualRank, keel::usize requestedRank);

    template <class T> const T *typedDataPtr() const;
    template <class T> T *typedDataMutPtr();
};

Value narrowComplex(Value v, keel::Allocator *alloc = nullptr);
Value collapseCsl(Value v);

// ── Inline template bodies ──────────────────────────────────────

template <class T>
inline const T *Value::typedDataPtr() const
{
    if constexpr (keel::IsSame<T, keel::f64>)      return doubleData();
    else if constexpr (keel::IsSame<T, keel::f32>) return singleData();
    else if constexpr (keel::IsSame<T, keel::i8>)  return int8Data();
    else if constexpr (keel::IsSame<T, keel::i16>) return int16Data();
    else if constexpr (keel::IsSame<T, keel::i32>) return int32Data();
    else if constexpr (keel::IsSame<T, keel::i64>) return int64Data();
    else if constexpr (keel::IsSame<T, keel::u8>)  return uint8Data();
    else if constexpr (keel::IsSame<T, keel::u16>) return uint16Data();
    else if constexpr (keel::IsSame<T, keel::u32>) return uint32Data();
    else if constexpr (keel::IsSame<T, keel::u64>) return uint64Data();
    else if constexpr (keel::IsSame<T, keel::c64>) return complexData();
}

template <class T>
inline T *Value::typedDataMutPtr()
{
    if constexpr (keel::IsSame<T, keel::f64>)      return doubleDataMut();
    else if constexpr (keel::IsSame<T, keel::f32>) return singleDataMut();
    else if constexpr (keel::IsSame<T, keel::i8>)  return int8DataMut();
    else if constexpr (keel::IsSame<T, keel::i16>) return int16DataMut();
    else if constexpr (keel::IsSame<T, keel::i32>) return int32DataMut();
    else if constexpr (keel::IsSame<T, keel::i64>) return int64DataMut();
    else if constexpr (keel::IsSame<T, keel::u8>)  return uint8DataMut();
    else if constexpr (keel::IsSame<T, keel::u16>) return uint16DataMut();
    else if constexpr (keel::IsSame<T, keel::u32>) return uint32DataMut();
    else if constexpr (keel::IsSame<T, keel::u64>) return uint64DataMut();
    else if constexpr (keel::IsSame<T, keel::c64>) return complexDataMut();
}

template <class T>
    requires detail::is_native_value_elem_v<T>
inline keel::ArrayRef<T> Value::asArrayRef() const
{
    const T *ptr = typedDataPtr<T>();
    const keel::usize n = numel();
    return (ptr != nullptr && n > 0) ? keel::ArrayRef<T>(ptr, n) : keel::ArrayRef<T>{};
}

template <class T>
    requires detail::is_native_value_elem_v<T>
inline keel::MutArrayRef<T> Value::asMutArrayRef()
{
    T *ptr = typedDataMutPtr<T>();
    const keel::usize n = numel();
    return (ptr != nullptr && n > 0) ? keel::MutArrayRef<T>(ptr, n) : keel::MutArrayRef<T>{};
}

template <class T, keel::usize Rank>
    requires detail::is_native_value_elem_v<T> && (Rank >= 1)
inline keel::MdRef<T, Rank, keel::LayoutLeft> Value::asMdRef() const
{
    const keel::ArrayRef<T> flat = asArrayRef<T>();
    if constexpr (Rank == 1) {
        return keel::MdRef<T, 1, keel::LayoutLeft>(flat);
    } else {
        const Dims &d = dims();
        if (d.ndim() > keel::as<keel::i32>(Rank)) {
            throwRankMismatch(d.ndim(), Rank);
        }
        if (flat.empty()) {
            return {};
        }
        keel::StaticArray<keel::usize, Rank> shape{};
        for (keel::usize k = 0; k < Rank; ++k) {
            shape[k] = d.dim(keel::as<keel::i32>(k));
        }
        return keel::MdRef<T, Rank, keel::LayoutLeft>(flat, shape);
    }
}

template <class T, keel::usize Rank>
    requires detail::is_native_value_elem_v<T> && (Rank >= 1)
inline keel::MutMdRef<T, Rank, keel::LayoutLeft> Value::asMutMdRef()
{
    const keel::MutArrayRef<T> flat = asMutArrayRef<T>();
    if constexpr (Rank == 1) {
        return keel::MutMdRef<T, 1, keel::LayoutLeft>(flat);
    } else {
        const Dims &d = dims();
        if (d.ndim() > keel::as<keel::i32>(Rank)) {
            throwRankMismatch(d.ndim(), Rank);
        }
        if (flat.empty()) {
            return {};
        }
        keel::StaticArray<keel::usize, Rank> shape{};
        for (keel::usize k = 0; k < Rank; ++k) {
            shape[k] = d.dim(keel::as<keel::i32>(k));
        }
        return keel::MutMdRef<T, Rank, keel::LayoutLeft>(flat, shape);
    }
}

template <class T, keel::usize Rank>
    requires detail::is_native_value_elem_v<T>
inline Value Value::view(keel::MdRef<T, Rank, keel::LayoutLeft> src)
{
    constexpr ValueType vt = detail::value_type_for<T>::type;
    if (src.empty()) {
        return view(nullptr, vt, Dims(0, 0));
    }
    if constexpr (Rank == 1) {
        return view(src.data(), vt, Dims(1, src.extent(0)));
    } else if constexpr (Rank == 2) {
        return view(src.data(), vt, Dims(src.extent(0), src.extent(1)));
    } else if constexpr (Rank == 3) {
        return view(src.data(), vt, Dims(src.extent(0), src.extent(1), src.extent(2)));
    } else {
        return view(src.data(), vt, Dims(src.shape().data(), keel::as<keel::i32>(Rank)));
    }
}

template <class T, keel::usize Rank, keel::IsMdLayout Layout>
    requires detail::is_native_value_elem_v<T>
inline Value Value::fromMdRef(keel::MdRef<T, Rank, Layout> src, keel::Allocator *alloc)
{
    constexpr ValueType vt = detail::value_type_for<T>::type;
    if (src.empty()) {
        return Value::matrix(0, 0, vt, alloc);
    }
    if constexpr (Rank == 1) {
        return makeContiguous(src.data(), src.size(), vt, sizeof(T), alloc);
    } else {
        Value out = (Rank == 2)
                        ? Value::matrix(src.extent(0), src.extent(1), vt, alloc)
                        : Value::matrixND(src.shape().data(), keel::as<keel::i32>(Rank), vt, alloc);
        T *dstPtr = out.typedDataMutPtr<T>();
        if constexpr (keel::IsSame<Layout, keel::LayoutLeft>) {
            keel::mem_copy(dstPtr, src.data(), src.size() * sizeof(T));
        } else if constexpr (Rank == 2) {
            const keel::usize rows = src.extent(0);
            const keel::usize cols = src.extent(1);
            const T *srcPtr = src.data();
            for (keel::usize r = 0; r < rows; ++r) {
                for (keel::usize c = 0; c < cols; ++c) {
                    dstPtr[c * rows + r] = srcPtr[r * cols + c];
                }
            }
        } else {
            keel::StaticArray<keel::usize, Rank> colStrides{};
            keel::usize cs = 1;
            for (keel::usize d = 0; d < Rank; ++d) {
                colStrides[d] = cs;
                cs *= src.extent(d);
            }
            keel::StaticArray<keel::usize, Rank> coords{};
            const T *srcPtr = src.data();
            const keel::usize total = src.size();
            for (keel::usize srcIdx = 0; srcIdx < total; ++srcIdx) {
                keel::usize dstIdx = 0;
                for (keel::usize d = 0; d < Rank; ++d) {
                    dstIdx += coords[d] * colStrides[d];
                }
                dstPtr[dstIdx] = srcPtr[srcIdx];
                for (keel::usize ax = Rank; ax > 0; --ax) {
                    const keel::usize d = ax - 1;
                    if (++coords[d] < src.extent(d)) break;
                    coords[d] = 0;
                }
            }
        }
        return out;
    }
}

template<class T>
    requires requires { detail::value_type_for<T>::type; }
inline Value::Value(keel::ArrayRef<T> data, keel::Allocator *alloc)
    : Value()
{
    constexpr ValueType vt = detail::value_type_for<T>::type;
    *this = makeContiguous(data.data(), data.size(), vt, sizeof(T), alloc);
}

template<class R>
    requires(!keel::IsSame<keel::RemoveCvRef<R>, Value>)
         && (!detail::is_array_ref_v<keel::RemoveCvRef<R>>)
         && (!detail::is_md_ref_v<keel::RemoveCvRef<R>>)
         && requires(const R &r) {
                r.data();
                r.size();
                requires keel::IsPointer<decltype(r.data())>;
                detail::value_type_for<keel::RemoveCv<keel::RemovePointer<decltype(r.data())>>>::type;
            }
inline Value::Value(const R &range, keel::Allocator *alloc)
    : Value()
{
    using ElemT = keel::RemoveCv<keel::RemovePointer<decltype(range.data())>>;
    constexpr ValueType vt = detail::value_type_for<ElemT>::type;
    *this = makeContiguous(range.data(), keel::as<keel::usize>(range.size()), vt, sizeof(ElemT), alloc);
}

} // namespace numkit
