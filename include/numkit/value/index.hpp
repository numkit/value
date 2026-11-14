// include/numkit/value/index.hpp
//
// Canonical pointer-sized dimension/index aliases (`numkit::Size`,
// `numkit::Index`) and platform-safe `double -> Size` conversion gates.
//
// Invariant: any accepted index or array dimension `n` satisfies
// `n <= kMaxIndex`, where:
//   * on 64-bit targets: `2^53 - 1` (`9007199254740991`), the largest
//     contiguous exact integer in IEEE-754 binary64 (`0x1p53 - 1.0`, so
//     `n + 1` never aliases `n` under `double` rounding);
//   * on 32-bit targets (e.g. WASM32): `0x7fffffff` (`INT32_MAX`), so every
//     dimension and 0-based offset fits simultaneously in both `Size`
//     (`std::size_t`) and signed `Index` (`std::ptrdiff_t` / `Eigen::Index`).
#pragma once

#include <cstddef>
#include <cstdint>

namespace numkit {

using Size  = std::size_t;
using Index = std::ptrdiff_t;

// Exact integer ceiling of IEEE-754 binary64 (`2^53`, i.e. `flintmax('double')`).
inline constexpr double kMaxExactDoubleInt = 0x1p53;

// Platform-safe upper bound for any 1-based index or array dimension.
inline constexpr Size kMaxIndex =
    (sizeof(Size) >= 8) ? static_cast<Size>((1ULL << 53) - 1ULL)
                        : static_cast<Size>(0x7fffffffULL);

inline constexpr double kMaxIndexDouble = static_cast<double>(kMaxIndex);

// Convert a 1-based `double` subscript into a 0-based `Size` without UB or
// `std::floor`. Rejects NaN, +/-Inf, `<= 0`, fractional values, and values
// above `kMaxIndexDouble` prior to casting.
constexpr bool tryIndex0(double idx1, Size &out0) noexcept
{
    if (idx1 >= 1.0 && idx1 <= kMaxIndexDouble) {
        const Size u = static_cast<Size>(idx1);
        if (static_cast<double>(u) == idx1) {
            out0 = u - 1;
            return true;
        }
    }
    return false;
}

// Convert a non-negative `double` size/count into `Size` without UB or
// `std::floor`. Rejects NaN, +/-Inf, `< 0`, fractional values, and values
// above `kMaxIndexDouble` prior to casting.
constexpr bool tryNonNegSize(double d, Size &out) noexcept
{
    if (d >= 0.0 && d <= kMaxIndexDouble) {
        const Size u = static_cast<Size>(d);
        if (static_cast<double>(u) == d) {
            out = u;
            return true;
        }
    }
    return false;
}

} // namespace numkit
