// include/dims.hpp
#pragma once

#include <cstddef>
#include <cstdint>

#include <numkit/value/index.hpp>

namespace numkit {

// ============================================================
// Dims — array dimensions, ND with SBO storage.
//
// Layout:
//   * inline_ holds dims [0, kInlineCap). Slots beyond ndim() are
//     always = 1 (so trailing-singleton equality semantics work
//     without per-call branching).
//   * For ndim() > kInlineCap, the extra dims live on heap_, indexed
//     as heap_[i - kInlineCap]. heap_ == nullptr when ndim() <= kInlineCap.
//
// The SBO threshold of 4 covers MATLAB's typical use (scalar / vector /
// matrix / 3D-volume / 4D-batch tensor) without any allocation. Heap is
// reserved for the genuinely ND case (5D+). Same pattern as
// llvm::SmallVector / pytorch::DimVector.
//
// rows()/cols()/pages() are inline aliases on inline_[0/1/2] — no
// branch on the hot accessor path. dim(i) branches on i < kInlineCap;
// for compile-time-constant i (the common case in 2D/3D code) the
// optimiser folds the branch away.
// ============================================================
struct Dims
{
    static constexpr int kInlineCap = 4;
    // Maximum supported rank for any ND op. Used as a stack-buffer
    // bound (e.g. `size_t coords[Dims::kMaxRank]`) and a sanity-check
    // ceiling. 32 is well above any practical use.
    static constexpr int kMaxRank   = 32;

    Dims();
    Dims(Size r, Size c);
    Dims(Size r, Size c, Size p);
    Dims(const Size *dims, int nd);

    Dims(const Dims &o);
    Dims(Dims &&o) noexcept;
    Dims &operator=(const Dims &o);
    Dims &operator=(Dims &&o) noexcept;
    ~Dims();

    int  ndims() const noexcept { return nd_; }
    int  ndim()  const noexcept { return nd_; }
    Size rows()  const noexcept { return inline_[0]; }
    Size cols()  const noexcept { return inline_[1]; }
    Size pages() const noexcept { return inline_[2]; }

    Size dim(int i) const noexcept
    {
        if (i < 0)             return 1;
        if (i < kInlineCap)    return inline_[i];
        if (i < nd_)           return heap_[i - kInlineCap];
        return 1;
    }

    Size dimSize(int i) const noexcept { return dim(i); }

    Size numel() const noexcept
    {
        Size n = inline_[0] * inline_[1] * inline_[2] * inline_[3];
        if (heap_) {
            const int extra = nd_ - kInlineCap;
            for (int i = 0; i < extra; ++i)
                n *= heap_[i];
        }
        return n;
    }

    bool isScalar() const noexcept { return numel() == 1; }
    bool isEmpty()  const noexcept { return numel() == 0; }
    bool isVector() const noexcept
    {
        return nd_ == 2 && (inline_[0] == 1 || inline_[1] == 1);
    }
    bool is3D() const noexcept { return nd_ == 3 && inline_[2] > 1; }

    Size sub2ind(Size r, Size c) const noexcept
    {
        return c * inline_[0] + r;
    }
    Size sub2ind(Size r, Size c, Size p) const noexcept
    {
        return p * inline_[0] * inline_[1] + c * inline_[0] + r;
    }
    Size sub2indChecked(Size r, Size c) const;
    Size sub2indChecked(Size r, Size c, Size p) const;

    bool operator==(const Dims &o) const noexcept;
    bool operator!=(const Dims &o) const noexcept { return !(*this == o); }

private:
    void setND(const Size *dims, int nd);
    void freeHeap() noexcept;

    Size    inline_[kInlineCap];
    Size   *heap_;
    uint8_t nd_;
};

} // namespace numkit
