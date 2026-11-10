/* include/numkit/value/c_api.h
 *
 * Foundation C-ABI for `numkit::Value` (`numkit_val*`), error reporting,
 * zero-copy views, and container access (`cell`, `struct`).
 *
 * Every NumKit module (`signal`, `stats`, `control`, `wavelet`, `audio`,
 * `comm`, `image`, `cas`, and the `numkit` core engine) shares this exact
 * binary value representation so values pass across modules with zero copying.
 */
#ifndef NUMKIT_VALUE_C_API_H
#define NUMKIT_VALUE_C_API_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32) || defined(__CYGWIN__)
#  if defined(NUMKIT_VALUE_C_BUILD)
#    define NUMKIT_VALUE_API __declspec(dllexport)
#  elif defined(NUMKIT_VALUE_C_STATIC)
#    define NUMKIT_VALUE_API
#  else
#    define NUMKIT_VALUE_API __declspec(dllimport)
#  endif
#else
#  if defined(__GNUC__) && __GNUC__ >= 4
#    define NUMKIT_VALUE_API __attribute__((visibility("default")))
#  else
#    define NUMKIT_VALUE_API
#  endif
#endif

#define NUMKIT_C_ABI_VERSION 1

/* Opaque handle to a `numkit::Value` */
typedef struct numkit_val_s numkit_val;

/* Opaque pre-resolved function pointer returned by `numkit_<module>_resolve` */
typedef struct numkit_fn_s numkit_fn;

/* Element / container type tag (mirrors `numkit::ValueType`) */
typedef enum numkit_dtype {
    NUMKIT_DTYPE_DOUBLE      = 0,
    NUMKIT_DTYPE_SINGLE      = 1,
    NUMKIT_DTYPE_COMPLEX     = 2,  /* interleaved double (re, im) pairs */
    NUMKIT_DTYPE_LOGICAL     = 3,  /* uint8_t (0 or 1) */
    NUMKIT_DTYPE_CHAR        = 4,  /* char matrix */
    NUMKIT_DTYPE_STRING      = 5,  /* string array */
    NUMKIT_DTYPE_INT8        = 6,
    NUMKIT_DTYPE_INT16       = 7,
    NUMKIT_DTYPE_INT32       = 8,
    NUMKIT_DTYPE_INT64       = 9,
    NUMKIT_DTYPE_UINT8       = 10,
    NUMKIT_DTYPE_UINT16      = 11,
    NUMKIT_DTYPE_UINT32      = 12,
    NUMKIT_DTYPE_UINT64      = 13,
    NUMKIT_DTYPE_CELL        = 14,
    NUMKIT_DTYPE_STRUCT      = 15,
    NUMKIT_DTYPE_FUNC_HANDLE = 16,
    NUMKIT_DTYPE_OBJECT      = 17,
    NUMKIT_DTYPE_UNKNOWN     = 255
} numkit_dtype;

/* Error reporting structure. `code == 0` indicates success. */
typedef struct numkit_error {
    int  code;
    char message[512];
} numkit_error;

/* Host-defined callback function signature for `register_fn` across modules.
 * Note: `outs` always points to `max(1, nargout)` zero-initialized slots, so
 * writing `outs[0]` even when `nargout == 0` populates the default `ans` result. */
typedef int (*numkit_host_fn)(void *user_data,
                              const numkit_val *const *args,
                              size_t nargs,
                              size_t nargout,
                              numkit_val **outs,
                              numkit_error *err);

/* =========================================================================
 * 1. ABI Version & Diagnostics
 * ========================================================================= */

/* Return `NUMKIT_C_ABI_VERSION`. */
NUMKIT_VALUE_API uint32_t numkit_abi_version(void);

/* Return the number of currently live `numkit_val*` handles across the process. */
NUMKIT_VALUE_API int64_t numkit_debug_live_values(void);

/* =========================================================================
 * 2. Value Creation, Zero-Copy Views & Lifecycle
 * ========================================================================= */

/* Construct an empty 0x0 double matrix `[]`. */
NUMKIT_VALUE_API numkit_val *numkit_val_empty(void);

/* Construct a 1x1 double scalar (inline representation, no array heap alloc). */
NUMKIT_VALUE_API numkit_val *numkit_val_scalar(double v);

/* Construct a 1x1 logical scalar (0 -> false, non-zero -> true). */
NUMKIT_VALUE_API numkit_val *numkit_val_logical(int b);

/* Construct a 1x1 complex double scalar (re + i*im). */
NUMKIT_VALUE_API numkit_val *numkit_val_complex_scalar(double re, double im);

/* Construct a 1x1 string scalar (MATLAB double-quoted "string"). */
NUMKIT_VALUE_API numkit_val *numkit_val_string(const char *utf8);

/* Construct a 1xN char row vector (MATLAB single-quoted 'char'). */
NUMKIT_VALUE_API numkit_val *numkit_val_char(const char *utf8);

/* Construct a function handle `@name`. */
NUMKIT_VALUE_API numkit_val *numkit_val_func_handle(const char *name);

/* Construct an owning N-D array by copying `data` (column-major / Fortran order). */
NUMKIT_VALUE_API numkit_val *numkit_val_array_copy(const void *data,
                                                   numkit_dtype dtype,
                                                   const size_t *dims,
                                                   int ndims,
                                                   numkit_error *err);

/* Construct a ZERO-COPY non-owning view over externally owned column-major memory.
 * Caller MUST keep `data` alive while this `numkit_val` (or any shallow clone) is
 * used. Any internal mutation automatically detaches via Copy-on-Write. */
NUMKIT_VALUE_API numkit_val *numkit_val_view(const void *data,
                                             numkit_dtype dtype,
                                             const size_t *dims,
                                             int ndims,
                                             numkit_error *err);

/* Allocate an owning N-D array and return its mutable data pointer in `*out_data_ptr`
 * so callers can populate it directly without an extra copy. */
NUMKIT_VALUE_API numkit_val *numkit_val_alloc_array(numkit_dtype dtype,
                                                    const size_t *dims,
                                                    int ndims,
                                                    void **out_data_ptr,
                                                    numkit_error *err);

/* Clone a value handle (O(1) Copy-on-Write share for heap arrays). */
NUMKIT_VALUE_API numkit_val *numkit_val_clone(const numkit_val *v);

/* Release an owned value handle. Safe to call with NULL. */
NUMKIT_VALUE_API void numkit_val_release(numkit_val *v);

/* =========================================================================
 * 3. Value Introspection & Zero-Copy Data Access
 * ========================================================================= */

NUMKIT_VALUE_API numkit_dtype numkit_val_type(const numkit_val *v);
NUMKIT_VALUE_API size_t       numkit_val_numel(const numkit_val *v);
NUMKIT_VALUE_API size_t       numkit_val_ndims(const numkit_val *v);
NUMKIT_VALUE_API size_t       numkit_val_dims(const numkit_val *v, size_t *out_dims, size_t max_dims);
NUMKIT_VALUE_API int          numkit_val_is_empty(const numkit_val *v);
NUMKIT_VALUE_API int          numkit_val_is_scalar(const numkit_val *v);

/* Return a const pointer to the contiguous column-major buffer of `v`
 * (including inline 1x1 double scalars, where it points to the internal scalar). */
NUMKIT_VALUE_API const void *numkit_val_raw_data(const numkit_val *v);
NUMKIT_VALUE_API size_t      numkit_val_raw_bytes(const numkit_val *v);

/* Copy up to `max_bytes` of flat element data into `dst`. Returns bytes copied. */
NUMKIT_VALUE_API size_t numkit_val_copy_bytes(const numkit_val *v, void *dst, size_t max_bytes);

/* Scalar extraction helpers */
NUMKIT_VALUE_API double numkit_val_as_double(const numkit_val *v, numkit_error *err);
NUMKIT_VALUE_API int    numkit_val_as_bool(const numkit_val *v, numkit_error *err);
NUMKIT_VALUE_API int    numkit_val_as_complex(const numkit_val *v,
                                              double *out_re,
                                              double *out_im,
                                              numkit_error *err);

/* String / Char / display extraction */
NUMKIT_VALUE_API size_t numkit_val_string_bytes(const numkit_val *v);
NUMKIT_VALUE_API size_t numkit_val_write_string(const numkit_val *v, char *buf, size_t buf_cap);

/* Return the class name of an `NUMKIT_DTYPE_OBJECT` value (e.g. "sym", "tf"), or "". */
NUMKIT_VALUE_API size_t numkit_val_object_class(const numkit_val *v, char *buf, size_t buf_cap);

/* =========================================================================
 * 4. Cell & Struct Containers
 * ========================================================================= */

NUMKIT_VALUE_API numkit_val *numkit_cell_create(const size_t *dims, int ndims, numkit_error *err);
NUMKIT_VALUE_API numkit_val *numkit_cell_get(const numkit_val *cell, size_t linear_idx, numkit_error *err);
NUMKIT_VALUE_API int         numkit_cell_set(numkit_val *cell,
                                             size_t linear_idx,
                                             const numkit_val *elem,
                                             numkit_error *err);

NUMKIT_VALUE_API numkit_val *numkit_struct_create(size_t rows, size_t cols, numkit_error *err);
NUMKIT_VALUE_API size_t      numkit_struct_num_fields(const numkit_val *s);
NUMKIT_VALUE_API size_t      numkit_struct_field_name(const numkit_val *s,
                                                      size_t field_idx,
                                                      char *buf,
                                                      size_t buf_cap);
NUMKIT_VALUE_API numkit_val *numkit_struct_get(const numkit_val *s,
                                               size_t linear_idx,
                                               const char *field,
                                               numkit_error *err);
NUMKIT_VALUE_API int         numkit_struct_set(numkit_val *s,
                                               size_t linear_idx,
                                               const char *field,
                                               const numkit_val *val,
                                               numkit_error *err);

#ifdef __cplusplus
}
#endif

#endif /* NUMKIT_VALUE_C_API_H */
