// tests/c_api/value_c_api.cpp
//
// Standalone C-ABI tests for `numkit/value` (`numkit_value_c`).
#include <numkit/value/c_api.h>

#include <gtest/gtest.h>
#include <array>
#include <string>

TEST(NumkitValueCApi, ScalarsStringsAndHandleLifecycle)
{
    const int64_t base = numkit_debug_live_values();
    EXPECT_EQ(numkit_abi_version(), static_cast<uint32_t>(NUMKIT_C_ABI_VERSION));

    numkit_error err{};
    numkit_val *vScalar = numkit_val_scalar(42.5);
    EXPECT_EQ(numkit_val_type(vScalar), NUMKIT_DTYPE_DOUBLE);
    EXPECT_EQ(numkit_val_is_scalar(vScalar), 1);
    EXPECT_DOUBLE_EQ(numkit_val_as_double(vScalar, &err), 42.5);
    EXPECT_EQ(err.code, 0);

    numkit_val *vBool = numkit_val_logical(1);
    EXPECT_EQ(numkit_val_as_bool(vBool, &err), 1);

    numkit_val *vCplx = numkit_val_complex_scalar(3.0, -4.0);
    double re = 0.0, im = 0.0;
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
    const int64_t base = numkit_debug_live_values();
    numkit_error err{};

    std::array<double, 4> hostBuf = {1.0, 2.0, 3.0, 4.0};
    size_t dims[2] = {2, 2};
    numkit_val *view = numkit_val_view(hostBuf.data(), NUMKIT_DTYPE_DOUBLE, dims, 2, &err);
    ASSERT_NE(view, nullptr);
    EXPECT_EQ(numkit_val_raw_data(view), static_cast<const void *>(hostBuf.data()));

    void *mutPtr = nullptr;
    numkit_val *owned = numkit_val_alloc_array(NUMKIT_DTYPE_DOUBLE, dims, 2, &mutPtr, &err);
    ASSERT_NE(owned, nullptr);
    ASSERT_NE(mutPtr, nullptr);
    auto *dp = static_cast<double *>(mutPtr);
    for (int i = 0; i < 4; ++i) dp[i] = 10.0 * (i + 1);
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
