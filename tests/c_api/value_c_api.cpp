// tests/c_api/value_c_api.cpp
//
// Standalone C-ABI and keel::Allocator / keel::c64 tests for `numkit/value`.
#include <numkit/value/argcheck.hpp>
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
              static_cast<keel::usize>(std::numeric_limits<keel::isize>::max()));

    EXPECT_EQ(numkit::Value::checkedScalarIndex(1.0), keel::usize{0});
    EXPECT_EQ(numkit::Value::checkedScalarIndex(numkit::kMaxIndexDouble),
              numkit::kMaxIndex - 1);

    const keel::f64 nanv = std::numeric_limits<keel::f64>::quiet_NaN();
    const keel::f64 infv = std::numeric_limits<keel::f64>::infinity();
    for (keel::f64 bad : {0.0, -0.0, -1.0, 0.5, 1.5, nanv, infv, -infv,
                          numkit::kMaxExactDoubleInt, 1e300}) {
        EXPECT_THROW((void)numkit::Value::checkedScalarIndex(bad), std::runtime_error);
    }
}
