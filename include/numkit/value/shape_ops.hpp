// include/numkit/value/shape_ops.hpp
//
// Pure-function helpers for ND tensor shape ops: broadcasting, strides,
// coord-walking. Used by builtin and DSP ops that need to generalise
// past 2D/3D. Header-only inline so the optimiser can fold the ndim
// branches in 2D/3D fast paths.
//
// All routines use column-major strides (rows = innermost), matching
// the Value layout. Trailing-singleton normalisation is implicit: a
// missing trailing dim is treated as 1.

#pragma once

#include <numkit/value/dims.hpp>

namespace numkit {

// ============================================================
// broadcastDimsND
// ============================================================
inline bool broadcastDimsND(const Dims &a, const Dims &b, Dims &out)
{
    const keel::i32 na = a.ndim();
    const keel::i32 nb = b.ndim();
    const keel::i32 nOut = keel::max(na, nb);

    constexpr keel::i32 kMaxStack = Dims::kMaxRank;
    keel::usize buf[kMaxStack];
    if (nOut > kMaxStack) return false;

    for (keel::i32 i = 0; i < nOut; ++i) {
        const keel::usize da = (i < na) ? a.dim(i) : 1;
        const keel::usize db = (i < nb) ? b.dim(i) : 1;
        if (da != db && da != 1 && db != 1) return false;
        buf[i] = keel::max(da, db);
    }
    out = Dims(buf, nOut);
    return true;
}

// ============================================================
// broadcastOffsetND
// ============================================================
inline keel::usize broadcastOffsetND(const keel::usize *coords, keel::i32 outNdim,
                                     const Dims &operand)
{
    const keel::i32 nOp = operand.ndim();
    keel::usize off    = 0;
    keel::usize stride = 1;
    for (keel::i32 i = 0; i < outNdim; ++i) {
        const keel::usize opDim = (i < nOp) ? operand.dim(i) : 1;
        const keel::usize c     = (opDim == 1) ? 0 : coords[i];
        off += c * stride;
        stride *= opDim;
    }
    return off;
}

// ============================================================
// computeStridesColMajor
// ============================================================
inline void computeStridesColMajor(const Dims &d, keel::usize *strides)
{
    const keel::i32 n = d.ndim();
    if (n == 0) return;
    strides[0] = 1;
    for (keel::i32 i = 1; i < n; ++i)
        strides[i] = strides[i - 1] * d.dim(i - 1);
}

// ============================================================
// linearizeFromCoords
// ============================================================
inline keel::usize linearizeFromCoords(const keel::usize *coords,
                                       const keel::usize *strides,
                                       keel::i32 nd)
{
    keel::usize off = 0;
    for (keel::i32 i = 0; i < nd; ++i)
        off += coords[i] * strides[i];
    return off;
}

// ============================================================
// incrementCoords
// ============================================================
inline bool incrementCoords(keel::usize *coords, const Dims &d)
{
    const keel::i32 n = d.ndim();
    for (keel::i32 i = 0; i < n; ++i) {
        if (++coords[i] < d.dim(i)) return true;
        coords[i] = 0;
    }
    return false;
}

// ============================================================
// forEachNDPair
// ============================================================
template <typename Fn>
inline void forEachNDPair(const Dims &aDims, const Dims &bDims,
                          const Dims &outDims, Fn &&fn)
{
    if (aDims == bDims && aDims == outDims) {
        const keel::usize n = outDims.numel();
        for (keel::usize i = 0; i < n; ++i) fn(i, i, i);
        return;
    }
    const keel::i32 nd = outDims.ndim();
    keel::usize coords[Dims::kMaxRank] = {0};
    keel::usize outIdx = 0;
    do {
        const keel::usize aOff = broadcastOffsetND(coords, nd, aDims);
        const keel::usize bOff = broadcastOffsetND(coords, nd, bDims);
        fn(outIdx++, aOff, bOff);
    } while (incrementCoords(coords, outDims));
}

// ============================================================
// forEachOuterPage
// ============================================================
template <typename Fn>
inline void forEachOuterPage(const Dims &d, Fn &&fn)
{
    const keel::i32 nd = d.ndim();
    keel::usize outerCount = 1;
    for (keel::i32 i = 2; i < nd; ++i) outerCount *= d.dim(i);
    keel::usize outerCoords[Dims::kMaxRank] = {0};
    for (keel::usize plin = 0; plin < outerCount; ++plin) {
        fn(plin, outerCoords);
        for (keel::i32 i = 2; i < nd; ++i) {
            if (++outerCoords[i - 2] < d.dim(i)) break;
            outerCoords[i - 2] = 0;
        }
    }
}

} // namespace numkit
