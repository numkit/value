// include/numkit/value/dims.hpp
#pragma once

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
// The SBO threshold of 4 covers scalar / vector / matrix / 3D-volume /
// 4D-batch tensors without any heap allocation. Heap is reserved for 5D+.
// ============================================================
struct Dims
{
    static constexpr keel::i32 kInlineCap = 4;
    // Maximum supported rank for any ND op. Used as a stack-buffer
    // bound (e.g. `keel::usize coords[Dims::kMaxRank]`) and sanity ceiling.
    static constexpr keel::i32 kMaxRank   = 32;

    Dims();
    Dims(keel::usize r, keel::usize c);
    Dims(keel::usize r, keel::usize c, keel::usize p);
    Dims(const keel::usize *dims, keel::i32 nd);

    Dims(const Dims &o);
    Dims(Dims &&o) noexcept;
    Dims &operator=(const Dims &o);
    Dims &operator=(Dims &&o) noexcept;
    ~Dims();

    keel::i32   ndims() const noexcept { return nd_; }
    keel::i32   ndim()  const noexcept { return nd_; }
    keel::usize rows()  const noexcept { return inline_[0]; }
    keel::usize cols()  const noexcept { return inline_[1]; }
    keel::usize pages() const noexcept { return inline_[2]; }

    keel::usize dim(keel::i32 i) const noexcept
    {
        if (i < 0)          return 1;
        if (i < kInlineCap) return inline_[i];
        if (i < nd_)        return heap_[i - kInlineCap];
        return 1;
    }

    keel::usize dimSize(keel::i32 i) const noexcept { return dim(i); }

    keel::usize numel() const noexcept
    {
        keel::usize n = inline_[0] * inline_[1] * inline_[2] * inline_[3];
        if (heap_) {
            const keel::i32 extra = nd_ - kInlineCap;
            for (keel::i32 i = 0; i < extra; ++i)
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

    keel::usize sub2ind(keel::usize r, keel::usize c) const noexcept
    {
        return c * inline_[0] + r;
    }
    keel::usize sub2ind(keel::usize r, keel::usize c, keel::usize p) const noexcept
    {
        return p * inline_[0] * inline_[1] + c * inline_[0] + r;
    }
    keel::usize sub2indChecked(keel::usize r, keel::usize c) const;
    keel::usize sub2indChecked(keel::usize r, keel::usize c, keel::usize p) const;

    bool operator==(const Dims &o) const noexcept;
    bool operator!=(const Dims &o) const noexcept { return !(*this == o); }

private:
    void setND(const keel::usize *dims, keel::i32 nd);
    void freeHeap() noexcept;

    keel::usize  inline_[kInlineCap];
    keel::usize *heap_;
    keel::u8     nd_;
};

} // namespace numkit
