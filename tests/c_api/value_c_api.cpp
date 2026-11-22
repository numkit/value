// tests/c_api/value_c_api.cpp
//
// Standalone C-ABI and keel::Allocator / keel::c64 tests for `numkit/value`.
#include <numkit/value/argcheck.hpp>
#include <keel/core.h>
#include <numkit/value/c_api.h>
#include <numkit/value/c_api_bridge.hpp>
#include <numkit/value/fn_handle.hpp>
#include <numkit/value/index.hpp>
#include <numkit/value/object.hpp>
#include <numkit/value/shape_ops.hpp>
#include <numkit/value/value.hpp>

#include <keel/arena.h>

#include <gtest/gtest.h>
#include <array>
#include <limits>
#include <string>
#include <type_traits>

TEST(NumkitValueCApi, ScalarsStringsAndHandleLifecycle)
{
    const keel::i64 base = numkit_debug_live_values();
    EXPECT_EQ(numkit_abi_version(), static_cast<keel::u32>(NUMKIT_C_ABI_VERSION));

    numkit_error err{};
    numkit_val *vScalar = numkit_val_scalar(42.5);
    EXPECT_EQ(numkit_val_type(vScalar), NUMKIT_DTYPE_DOUBLE);
    EXPECT_EQ(numkit_val_is_scalar(vScalar), 1);
    EXPECT_DOUBLE_EQ(numkit_val_as_double(vScalar, &err), 42.5);
    EXPECT_EQ(err.code, 0);

    numkit_val *vBool = numkit_val_logical(1);
    EXPECT_EQ(numkit_val_as_bool(vBool, &err), 1);

    numkit_val *vCplx = numkit_val_complex_scalar(3.0, -4.0);
    keel::f64 re = 0.0, im = 0.0;
    EXPECT_EQ(numkit_val_as_complex(vCplx, &re, &im, &err), 0);
    EXPECT_DOUBLE_EQ(re, 3.0);
    EXPECT_DOUBLE_EQ(im, -4.0);

    numkit_val *vStr = numkit_val_string("hello numkit");
    char buf[64]{};
    EXPECT_EQ(numkit_val_write_string(vStr, buf, sizeof(buf)), 12u);
    EXPECT_STREQ(buf, "hello numkit");

    numkit_val_release(vScalar);
    numkit_val_release(vBool);
    numkit_val_release(vCplx);
    numkit_val_release(vStr);
    EXPECT_EQ(numkit_debug_live_values(), base);
}

TEST(NumkitValueCApi, ZeroCopyViewAllocArrayAndContainers)
{
    const keel::i64 base = numkit_debug_live_values();
    numkit_error err{};

    std::array<keel::f64, 4> hostBuf = {1.0, 2.0, 3.0, 4.0};
    keel::usize dims[2] = {2, 2};
    numkit_val *view = numkit_val_view(hostBuf.data(), NUMKIT_DTYPE_DOUBLE, dims, 2, &err);
    ASSERT_NE(view, nullptr);
    EXPECT_EQ(numkit_val_raw_data(view), static_cast<const void *>(hostBuf.data()));

    void *mutPtr = nullptr;
    numkit_val *owned = numkit_val_alloc_array(NUMKIT_DTYPE_DOUBLE, dims, 2, &mutPtr, &err);
    ASSERT_NE(owned, nullptr);
    ASSERT_NE(mutPtr, nullptr);
    auto *dp = static_cast<keel::f64 *>(mutPtr);
    for (keel::i32 i = 0; i < 4; ++i) dp[i] = 10.0 * (i + 1);
    EXPECT_EQ(numkit_val_raw_data(owned), mutPtr);

    numkit_val *st = numkit_struct_create(1, 1, &err);
    numkit_val *fv = numkit_val_scalar(99.0);
    EXPECT_EQ(numkit_struct_set(st, 0, "gain", fv, &err), 0);
    numkit_val *got = numkit_struct_get(st, 0, "gain", &err);
    EXPECT_DOUBLE_EQ(numkit_val_as_double(got, &err), 99.0);

    numkit_val_release(got);
    numkit_val_release(fv);
    numkit_val_release(st);
    numkit_val_release(owned);
    numkit_val_release(view);
    EXPECT_EQ(numkit_debug_live_values(), base);
}

TEST(NumkitValueCApi, KeelAllocatorAndContainersIntegration)
{
    keel::Arena arena(64 * 1024);
    keel::Allocator *alloc = arena.allocator();
    EXPECT_EQ(arena.used(), 0u);

    {
        numkit::Value mat = numkit::Value::matrix(4, 4, numkit::ValueType::DOUBLE, alloc);
        EXPECT_GT(arena.used(), 0u);
        mat.elem(1, 2) = 13.5;
        EXPECT_DOUBLE_EQ(mat(1, 2), 13.5);

        keel::Array<keel::c64> cvec(alloc);
        cvec.push_back(keel::c64{1.0, 2.0});
        cvec.push_back(keel::c64{3.0, -4.0});
        numkit::Value cv(cvec, alloc);
        EXPECT_TRUE(cv.isComplex());
        EXPECT_EQ(cv.numel(), 2u);
        EXPECT_DOUBLE_EQ(cv.complexElem(1).real(), 3.0);
        EXPECT_DOUBLE_EQ(cv.complexElem(1).imag(), -4.0);

        numkit::Value cell = numkit::Value::cell(1, 2, alloc);
        cell.cellAt(0) = mat;
        cell.cellAt(1) = cv;
        numkit::Value cellCopy = cell;
        cellCopy.cellAt(0) = numkit::Value::scalar(7.0); // COW detach
        EXPECT_DOUBLE_EQ(cell.cellAt(0)(1, 2), 13.5);
        EXPECT_DOUBLE_EQ(cellCopy.cellAt(0).toScalar(), 7.0);
    }
    arena.reset();
    EXPECT_EQ(arena.used(), 0u);
}

namespace {
constexpr bool checkConstexprIndexGate()
{
    keel::usize idx0 = 99;
    if (!numkit::tryIndex0(1.0, idx0) || idx0 != 0)
        return false;
    if (!numkit::tryIndex0(numkit::kMaxIndexDouble, idx0)
        || idx0 != numkit::kMaxIndex - 1)
        return false;
    if (numkit::tryIndex0(0.0, idx0) || numkit::tryIndex0(-1.0, idx0)
        || numkit::tryIndex0(1.5, idx0)
        || numkit::tryIndex0(numkit::kMaxExactDoubleInt, idx0))
        return false;

    keel::usize sz = 99;
    if (!numkit::tryNonNegSize(0.0, sz) || sz != 0)
        return false;
    if (!numkit::tryNonNegSize(numkit::kMaxIndexDouble, sz)
        || sz != numkit::kMaxIndex)
        return false;
    if (numkit::tryNonNegSize(-0.5, sz)
        || numkit::tryNonNegSize(numkit::kMaxExactDoubleInt, sz))
        return false;
    return true;
}
static_assert(checkConstexprIndexGate(), "numkit::tryIndex0 / tryNonNegSize constexpr check");
} // namespace

TEST(NumkitValueCApi, CanonicalIndexAndDimBounds)
{
    EXPECT_LE(numkit::kMaxIndex,
              static_cast<keel::usize>(keel::Limits<keel::isize>::kMax));

    EXPECT_EQ(numkit::Value::checkedScalarIndex(1.0), keel::usize{0});
    EXPECT_EQ(numkit::Value::checkedScalarIndex(numkit::kMaxIndexDouble),
              numkit::kMaxIndex - 1);

    const keel::f64 nanv = keel::Limits<keel::f64>::quiet_nan();
    const keel::f64 infv = keel::Limits<keel::f64>::infinity();
    for (keel::f64 bad : {0.0, -0.0, -1.0, 0.5, 1.5, nanv, infv, -infv,
                          numkit::kMaxExactDoubleInt, 1e300}) {
        EXPECT_THROW((void)numkit::Value::checkedScalarIndex(bad), std::runtime_error);
    }
}

TEST(NumkitValueCApi, MdRefLayoutLeftViewsAndInterop)
{
    using numkit::Value;

    // 1D range ctor must NOT silently flatten multi-dimensional MdRef / StridedMdRef views.
    static_assert(!std::is_constructible_v<Value, keel::MdRef<keel::f64, 2>>);
    static_assert(!std::is_constructible_v<Value, keel::MutMdRef<keel::f64, 2>>);
    static_assert(!std::is_constructible_v<Value, keel::ColMdRef<keel::f64, 2>>);
    static_assert(!std::is_constructible_v<Value, keel::StridedMdRef<keel::f64, 2>>);
    static_assert(!std::is_constructible_v<Value, keel::MutStridedMdRef<keel::f64, 2>>);

    // 1. Inline double scalar -> 1D & 2D LayoutLeft views
    Value s = Value::scalar(7.5);
    auto sArr = s.asArrayRef<keel::f64>();
    ASSERT_EQ(sArr.size(), keel::usize{1});
    EXPECT_DOUBLE_EQ(sArr[0], 7.5);
    auto s2d = s.asMdRef<keel::f64, 2>();
    ASSERT_TRUE(s2d);
    EXPECT_EQ(s2d.extent(0), keel::usize{1});
    EXPECT_EQ(s2d.extent(1), keel::usize{1});
    EXPECT_DOUBLE_EQ(s2d(0, 0), 7.5);
    s.asMutArrayRef<keel::f64>()[0] = 9.25;
    EXPECT_DOUBLE_EQ(s.toScalar(), 9.25);

    // 2. 2D column-major matrix -> zero-copy asMdRef<f64, 2> + COW detach on asMutMdRef
    Value m = Value::matrix(2, 3);
    m.elem(0, 0) = 1.0; m.elem(0, 1) = 3.0; m.elem(0, 2) = 5.0;
    m.elem(1, 0) = 2.0; m.elem(1, 1) = 4.0; m.elem(1, 2) = 6.0;

    keel::ColMdRef<keel::f64, 2> mv = m.asMdRef<keel::f64, 2>();
    ASSERT_TRUE(mv);
    EXPECT_EQ(mv.data(), m.doubleData());
    EXPECT_EQ(mv.extent(0), keel::usize{2});
    EXPECT_EQ(mv.extent(1), keel::usize{3});
    EXPECT_DOUBLE_EQ(mv(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(mv(1, 0), 2.0);
    EXPECT_DOUBLE_EQ(mv(0, 2), 5.0);
    EXPECT_DOUBLE_EQ(mv(1, 2), 6.0);
    EXPECT_DOUBLE_EQ(mv.col(1)[0], 3.0);
    EXPECT_DOUBLE_EQ(mv.col(1)[1], 4.0);

    // Trailing singleton allowed: 2x3 matrix viewed as 2x3x1
    auto mv3 = m.asMdRef<keel::f64, 3>();
    ASSERT_TRUE(mv3);
    EXPECT_EQ(mv3.extent(2), keel::usize{1});
    EXPECT_DOUBLE_EQ(mv3(1, 2, 0), 6.0);

    // COW detach via asMutMdRef
    Value copy = m;
    auto mutView = copy.asMutMdRef<keel::f64, 2>();
    mutView(1, 1) = 42.0;
    EXPECT_DOUBLE_EQ(copy(1, 1), 42.0);
    EXPECT_DOUBLE_EQ(m(1, 1), 4.0);

    // Type mismatch throws
    EXPECT_THROW(((void)m.asArrayRef<keel::f32>()), std::runtime_error);
    EXPECT_THROW(((void)m.asMdRef<keel::i32, 2>()), std::runtime_error);

    // 3. 3D volume -> asMdRef<i32, 3> works; asMdRef<i32, 2> throws rank mismatch
    Value vol = Value::matrix3d(2, 3, 4, numkit::ValueType::INT32);
    auto v3mut = vol.asMutMdRef<keel::i32, 3>();
    for (keel::usize p = 0; p < 4; ++p)
        for (keel::usize c = 0; c < 3; ++c)
            for (keel::usize r = 0; r < 2; ++r)
                v3mut(r, c, p) = static_cast<keel::i32>(p * 100 + c * 10 + r);
    EXPECT_THROW(((void)vol.asMdRef<keel::i32, 2>()), std::runtime_error);
    auto v3c = vol.asMdRef<keel::i32, 3>();
    EXPECT_EQ(v3c.subview(2)(1, 2), 221);

    // 4. fromMdRef: LayoutRight (row-major -> column-major transpose) & LayoutLeft (direct copy)
    const keel::f64 rowData[6] = {10.0, 20.0, 30.0, 40.0, 50.0, 60.0}; // 2x3 row-major
    keel::MdRef<keel::f64, 2, keel::LayoutRight> rowView(keel::ArrayRef<keel::f64>(rowData, 6), 2, 3);
    Value fromRow = Value::fromMdRef(rowView);
    EXPECT_EQ(fromRow.dims().rows(), keel::usize{2});
    EXPECT_EQ(fromRow.dims().cols(), keel::usize{3});
    EXPECT_DOUBLE_EQ(fromRow(0, 0), 10.0);
    EXPECT_DOUBLE_EQ(fromRow(0, 1), 20.0);
    EXPECT_DOUBLE_EQ(fromRow(0, 2), 30.0);
    EXPECT_DOUBLE_EQ(fromRow(1, 0), 40.0);
    EXPECT_DOUBLE_EQ(fromRow(1, 1), 50.0);
    EXPECT_DOUBLE_EQ(fromRow(1, 2), 60.0);

    // 3D LayoutRight -> column-major Value
    keel::f32 rowVol[24];
    for (keel::usize r = 0; r < 2; ++r)
        for (keel::usize c = 0; c < 3; ++c)
            for (keel::usize p = 0; p < 4; ++p)
                rowVol[(r * 3 + c) * 4 + p] = static_cast<keel::f32>(r * 100 + c * 10 + p);
    keel::MdRef<keel::f32, 3, keel::LayoutRight> rowVolView(keel::ArrayRef<keel::f32>(rowVol, 24), 2, 3, 4);
    Value fromRow3 = Value::fromMdRef(rowVolView);
    auto back3 = fromRow3.asMdRef<keel::f32, 3>();
    for (keel::usize r = 0; r < 2; ++r)
        for (keel::usize c = 0; c < 3; ++c)
            for (keel::usize p = 0; p < 4; ++p)
                EXPECT_FLOAT_EQ(back3(r, c, p), static_cast<keel::f32>(r * 100 + c * 10 + p));

    // LayoutLeft direct copy + zero-copy Value::view(ColMdRef)
    const keel::c64 colCplx[4] = {{1, 2}, {3, 4}, {5, 6}, {7, 8}};
    keel::ColMdRef<keel::c64, 2> cplxView(keel::ArrayRef<keel::c64>(colCplx, 4), 2, 2);
    Value ownedCplx = Value::fromMdRef(cplxView);
    auto ownedView = ownedCplx.asMdRef<keel::c64, 2>();
    EXPECT_EQ(ownedView(1, 0), keel::c64(3, 4));
    EXPECT_EQ(ownedView(0, 1), keel::c64(5, 6));

    Value zeroCopy = Value::view(cplxView);
    EXPECT_EQ(zeroCopy.complexData(), colCplx);
    auto zcView = zeroCopy.asMdRef<keel::c64, 2>();
    EXPECT_EQ(zcView(1, 1), keel::c64(7, 8));

    // Empty matrix -> empty MdRef
    Value empty = Value::matrix(0, 0);
    EXPECT_TRUE(empty.asArrayRef<keel::f64>().empty());
    auto emptyView = empty.asMdRef<keel::f64, 2>();
    EXPECT_TRUE(emptyView.empty());
}

TEST(NumkitValueCApi, ZeroAllocStringRefsAndAllocatorRouting)
{
    using numkit::Value;

    // ── 1. Zero-alloc *Ref accessors ─────────────────────────
    Value vChar = Value::fromString("hello world");
    keel::StringRef charRef = vChar.asStringRef();
    EXPECT_EQ(charRef, "hello world");
    EXPECT_EQ(charRef.data(), vChar.charData());

    Value vEmptyChar = Value::fromString("");
    EXPECT_TRUE(vEmptyChar.asStringRef().empty());

    Value vStrArr = Value::stringArray(1, 3);
    vStrArr.stringElemSet(0, "alpha");
    vStrArr.stringElemSet(1, "beta");
    vStrArr.stringElemSet(2, "gamma");
    EXPECT_EQ(vStrArr.asStringRef(), "alpha");
    EXPECT_EQ(vStrArr.stringElemRef(0), "alpha");
    EXPECT_EQ(vStrArr.stringElemRef(1), "beta");
    EXPECT_EQ(vStrArr.stringElemRef(2), "gamma");
    EXPECT_TRUE(vStrArr.stringElemRef(99).empty());
    EXPECT_TRUE(vChar.stringElemRef(0).empty());

    Value vFh = Value::funcHandle("myfunc");
    vFh.setFuncHandleSource("@(x) x + 1");
    EXPECT_EQ(vFh.funcHandleNameRef(), "myfunc");
    EXPECT_EQ(vFh.asStringRef(), "myfunc");
    EXPECT_EQ(vFh.funcHandleSourceRef(), "@(x) x + 1");
    EXPECT_TRUE(vChar.funcHandleNameRef().empty());
    EXPECT_TRUE(vChar.funcHandleSourceRef().empty());

    auto objState = keel::make_arc_in<numkit::ObjectState>(
        keel::default_allocator(), keel::default_allocator());
    Value vObj = Value::object("datetime", objState, false);
    EXPECT_EQ(vObj.objectClassNameRef(), "datetime");
    EXPECT_TRUE(vChar.objectClassNameRef().empty());

    EXPECT_THROW((void)Value::scalar(1.0).asStringRef(), std::runtime_error);

    // ── 2. Allocator routing on owning Value methods & Error ─
    keel::Arena arena(64 * 1024);
    keel::Allocator *alloc = arena.allocator();

    {
        keel::usize before = arena.used();
        keel::String s1 = vChar.toString(alloc);
        EXPECT_EQ(s1.as_ref(), "hello world");
        EXPECT_EQ(s1.allocator(), alloc);
        EXPECT_GT(arena.used(), before);

        before = arena.used();
        keel::String s2 = vStrArr.stringElem(1, alloc);
        EXPECT_EQ(s2.as_ref(), "beta");
        EXPECT_EQ(s2.allocator(), alloc);
        EXPECT_GT(arena.used(), before);

        before = arena.used();
        keel::String s3 = vObj.objectClassName(alloc);
        EXPECT_EQ(s3.as_ref(), "datetime");
        EXPECT_EQ(s3.allocator(), alloc);
        EXPECT_GT(arena.used(), before);

        before = arena.used();
        keel::String s4 = vFh.funcHandleName(alloc);
        EXPECT_EQ(s4.as_ref(), "myfunc");
        EXPECT_EQ(s4.allocator(), alloc);
        EXPECT_GT(arena.used(), before);

        before = arena.used();
        keel::String s5 = vFh.funcHandleSource(alloc);
        EXPECT_EQ(s5.as_ref(), "@(x) x + 1");
        EXPECT_EQ(s5.allocator(), alloc);
        EXPECT_GT(arena.used(), before);

        // 2x3 char matrix (column-major): row 0 = "abc", row 1 = "xyz"
        Value vCharMat = Value::matrix(2, 3, numkit::ValueType::CHAR);
        char *cd = vCharMat.charDataMut();
        cd[0] = 'a'; cd[1] = 'x';
        cd[2] = 'b'; cd[3] = 'y';
        cd[4] = 'c'; cd[5] = 'z';
        before = arena.used();
        keel::String row1 = vCharMat.charRow(1, alloc);
        EXPECT_EQ(row1.as_ref(), "xyz");
        EXPECT_EQ(row1.allocator(), alloc);
        EXPECT_GT(arena.used(), before);

        // fieldNamesInOrder: outer array + every inner String in alloc
        Value st = Value::structure();
        st.setField(0, "beta", Value::scalar(2.0));
        st.setField(0, "alpha", Value::scalar(1.0));
        before = arena.used();
        keel::Array<keel::String> fields = st.fieldNamesInOrder(alloc);
        ASSERT_EQ(fields.size(), keel::usize{2});
        EXPECT_EQ(fields.allocator(), alloc);
        EXPECT_EQ(fields[0].as_ref(), "beta");
        EXPECT_EQ(fields[0].allocator(), alloc);
        EXPECT_EQ(fields[1].as_ref(), "alpha");
        EXPECT_EQ(fields[1].allocator(), alloc);
        EXPECT_GT(arena.used(), before);

        // resolveIndices / resolveIndicesUnchecked
        Value idx = Value::matrix(1, 3);
        idx.elem(0) = 2.0; idx.elem(1) = 4.0; idx.elem(2) = 1.0;
        before = arena.used();
        keel::Array<keel::usize> r1 = Value::resolveIndices(idx, 5, alloc);
        ASSERT_EQ(r1.size(), keel::usize{3});
        EXPECT_EQ(r1[0], keel::usize{1});
        EXPECT_EQ(r1[1], keel::usize{3});
        EXPECT_EQ(r1[2], keel::usize{0});
        EXPECT_EQ(r1.allocator(), alloc);
        EXPECT_GT(arena.used(), before);

        before = arena.used();
        keel::Array<keel::usize> r2 = Value::resolveIndicesUnchecked(idx, alloc);
        ASSERT_EQ(r2.size(), keel::usize{3});
        EXPECT_EQ(r2[0], keel::usize{1});
        EXPECT_EQ(r2.allocator(), alloc);
        EXPECT_GT(arena.used(), before);

        // debugString & formatDisplay
        before = arena.used();
        keel::String dbg = vChar.debugString(alloc);
        EXPECT_EQ(dbg.allocator(), alloc);
        EXPECT_TRUE(dbg.as_ref().contains("hello world"));
        EXPECT_GT(arena.used(), before);

        before = arena.used();
        keel::String disp = vChar.formatDisplay("msg", alloc);
        EXPECT_EQ(disp.allocator(), alloc);
        EXPECT_TRUE(disp.as_ref().contains("msg ="));
        EXPECT_TRUE(disp.as_ref().contains("hello world"));
        EXPECT_GT(arena.used(), before);

        // Error::formattedWhat
        numkit::Error errFull("bad input", 12, 5, "solve", "iter=3");
        before = arena.used();
        keel::String fw1 = errFull.formattedWhat(alloc);
        EXPECT_EQ(fw1.allocator(), alloc);
        EXPECT_TRUE(fw1.as_ref().contains("line 12"));
        EXPECT_TRUE(fw1.as_ref().contains("bad input"));
        EXPECT_GT(arena.used(), before);

        numkit::Error errSimple("plain error");
        before = arena.used();
        keel::String fw2 = errSimple.formattedWhat(alloc);
        EXPECT_EQ(fw2.allocator(), alloc);
        EXPECT_EQ(fw2.as_ref(), "plain error");
        EXPECT_GT(arena.used(), before);
    }
    arena.reset();
    EXPECT_EQ(arena.used(), 0u);
}
