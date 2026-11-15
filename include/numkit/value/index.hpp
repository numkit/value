// include/numkit/value/index.hpp
//
// Platform-safe `keel::f64 -> keel::usize` index/dimension conversion gates.
//
// Invariant: any accepted index or array dimension `n` satisfies
// `n <= kMaxIndex`, where:
//   * on 64-bit targets: `2^53 - 1` (`9007199254740991`), the largest
//     contiguous exact integer in IEEE-754 binary64 (`0x1p53 - 1.0`, so
//     `n + 1` never aliases `n` under `double` rounding);
//   * on 32-bit targets (e.g. WASM32): `0x7fffffff` (`INT32_MAX`), so every
//     dimension and 0-based offset fits simultaneously in both `keel::usize`
//     and signed `keel::isize` (`Eigen::Index`).
#pragma once

#include <keel/core.h>

namespace numkit {

// Exact integer ceiling of IEEE-754 binary64 (`2^53`, i.e. `flintmax('double')`).
inline constexpr keel::f64 kMaxExactDoubleInt = 0x1p53;

// Platform-safe upper bound for any 1-based index or array dimension.
inline constexpr keel::usize kMaxIndex =
    (sizeof(keel::usize) >= 8) ? static_cast<keel::usize>((1ULL << 53) - 1ULL)
                               : static_cast<keel::usize>(0x7fffffffULL);

inline constexpr keel::f64 kMaxIndexDouble = static_cast<keel::f64>(kMaxIndex);

// Convert a 1-based `f64` subscript into a 0-based `usize` without UB or
// `std::floor`. Rejects NaN, +/-Inf, `<= 0`, fractional values, and values
// above `kMaxIndexDouble` prior to casting.
constexpr bool tryIndex0(keel::f64 idx1, keel::usize &out0) noexcept
{
    if (idx1 >= 1.0 && idx1 <= kMaxIndexDouble) {
        const keel::usize u = static_cast<keel::usize>(idx1);
        if (static_cast<keel::f64>(u) == idx1) {
            out0 = u - 1;
            return true;
        }
    }
    return false;
}

// Convert a non-negative `f64` size/count into `usize` without UB or
// `std::floor`. Rejects NaN, +/-Inf, `< 0`, fractional values, and values
// above `kMaxIndexDouble` prior to casting.
constexpr bool tryNonNegSize(keel::f64 d, keel::usize &out) noexcept
{
    if (d >= 0.0 && d <= kMaxIndexDouble) {
        const keel::usize u = static_cast<keel::usize>(d);
        if (static_cast<keel::f64>(u) == d) {
            out = u;
            return true;
        }
    }
    return false;
}

} // namespace numkit
