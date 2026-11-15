# numkit/value

[![License: 0BSD](https://img.shields.io/badge/License-0BSD-blue.svg)](LICENSE)
[![C++20](https://img.shields.io/badge/C%2B%2B-20-blue.svg)](https://en.wikipedia.org/wiki/C%2B%2B20)

The core data substrate, memory model, and contract validation layer for **[NumKit](https://github.com/numkit/numkit)**, built directly atop **[keelcpp/keel](https://github.com/keelcpp/keel)**.

`numkit/value` is an engine-free C++20 foundation library implementing a 16-byte tagged variant array and matrix model with Copy-on-Write (`DataBuffer` / `HeapObject`), `keel::c64` complex storage, and explicit `keel::Allocator*` memory management.

---

## Features

- **16-byte tagged `Value`**: Zero-allocation `keel::f64` scalars, logical tags, and empty sentinels.
- **Copy-on-Write (COW)**: Lazy 64-byte SIMD-aligned buffer cloning with atomic reference counting.
- **Column-Major Multi-dimensional Arrays (`Dims`)**: Compatible with BLAS/LAPACK/Fortran layouts.
- **Explicit Allocators**: Full control over allocation via `keel::Allocator*` (arenas, pools, system heap).
- **Formal Argument Contracts (`argcheck`)**: Standardized runtime shape, class-mask, and domain validation.
- **Stable C-ABI (`numkit_value_c`)**: Shared value representation across all dynamic NumKit modules.

---

## Building

```bash
cmake -B build -S .
cmake --build build --config Release
```

---

## License

This software is released under the **[BSD Zero Clause License (0BSD)](LICENSE)**.
