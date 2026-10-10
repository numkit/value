# numkit/value

[![License: 0BSD](https://img.shields.io/badge/License-0BSD-blue.svg)](LICENSE)
[![C++17](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://en.wikipedia.org/wiki/C%2B%2B17)

The core data substrate, memory model, and contract validation layer for **[NumKit](https://github.com/numkit/numkit)**.

`numkit/value` is an engine-free, standalone C++17 foundation library. It implements a 16-byte tagged variant array and matrix model with deterministic memory management (Copy-on-Write / PMR allocators) and MATLAB-compatible semantics.

---

## Features

- **16-byte tagged `Value`**: Zero-allocation double scalars, boolean masks, and empty arrays.
- **Copy-on-Write (COW)**: Lazy buffer cloning with atomic reference counting.
- **Column-Major Multi-dimensional Arrays (`Dims`)**: Compatible with BLAS/LAPACK/Fortran layouts.
- **PMR Memory Resources**: Full control over allocation via `std::pmr::memory_resource*` (arenas, pools, system heap).
- **Formal Argument Contracts (`argcheck`)**: Standardized runtime shape and domain validation.
- **Plugin Boundary Protocol (`plugin.hpp`)**: Minimal interface for dynamic NumKit modules (`.dll` / `.so`).
- **Zero External Dependencies**: Depends only on the C++17 standard library.

---

## Building

```bash
cmake -B build -S .
cmake --build build --config Release
```

Or consume directly via CMake `FetchContent`:

```cmake
include(FetchContent)
FetchContent_Declare(
    numkit_value
    GIT_REPOSITORY https://github.com/numkit/value.git
    GIT_TAG v0.1.0
)
FetchContent_MakeAvailable(numkit_value)

target_link_libraries(my_project PRIVATE nk::value)
```

---

## License

This software is released under the **[BSD Zero Clause License (0BSD)](LICENSE)**.
You may use, modify, and distribute it for any purpose with or without fee, with zero attribution required.
