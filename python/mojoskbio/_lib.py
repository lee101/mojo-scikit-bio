from __future__ import annotations

import ctypes
import os
import subprocess

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "src", "kernels.mojo")
LIB = os.environ.get("MOJOSK_BIO_LIB") or os.path.join(
    ROOT, "dist", "libmojo-scikit-bio.so"
)

I = ctypes.c_int64
F = ctypes.c_double

_SIGNATURES = {
    "msb_hamming": ([I, I, I], I),
    "msb_kmer_distance": ([I] * 11, F),
    "msb_alpha": ([I, I, I, F, F, I], F),
    "msb_alpha_batch": ([I, I, I, I, I, F, F, I], None),
    "msb_alpha_batch_i64": ([I, I, I, I, I, F, I], None),
    "msb_beta": ([I, I, I, I, I, F], None),
    "msb_beta_bool": ([I, I, I, I, I], None),
}


class BuildError(RuntimeError):
    pass


def build(force: bool = False) -> str:
    if (
        not force
        and os.path.exists(LIB)
        and os.path.getmtime(LIB) >= os.path.getmtime(SRC)
    ):
        return LIB
    if os.environ.get("MOJOSK_BIO_LIB"):
        raise BuildError(f"shared library does not exist: {LIB}")
    proc = subprocess.run(
        ["bash", os.path.join(ROOT, "build", "build.sh")],
        capture_output=True,
        text=True,
        timeout=1800,
    )
    if proc.returncode or not os.path.exists(LIB):
        raise BuildError((proc.stderr or proc.stdout).strip()[:4000])
    return LIB


_library: ctypes.CDLL | None = None


def lib() -> ctypes.CDLL:
    global _library
    if _library is None:
        _library = ctypes.CDLL(build())
        for name, (argtypes, restype) in _SIGNATURES.items():
            function = getattr(_library, name)
            function.argtypes = argtypes
            function.restype = restype
    return _library


def addr(
    array: np.ndarray,
    *,
    dtype: np.dtype | type | None = None,
    writable: bool = False,
) -> int:
    """Return a checked, non-null address for a synchronous FFI call."""
    if not isinstance(array, np.ndarray):
        raise TypeError("FFI buffers must be NumPy arrays.")
    if dtype is not None and array.dtype != np.dtype(dtype):
        raise TypeError(f"FFI buffer must have dtype {np.dtype(dtype)}.")
    if not array.flags.c_contiguous or not array.flags.aligned:
        raise ValueError("FFI buffers must be aligned and C-contiguous.")
    if writable and not array.flags.writeable:
        raise ValueError("FFI output buffers must be writable.")
    address = int(array.ctypes.data)
    if array.size == 0 or address == 0:
        raise ValueError("FFI buffers must be non-empty and non-null.")
    return address


def f64(values) -> np.ndarray:
    source = np.asarray(values)
    result = np.ascontiguousarray(source, dtype=np.float64)
    if source.dtype.itemsize > result.dtype.itemsize or np.issubdtype(
        source.dtype, np.integer
    ):
        with np.errstate(invalid="ignore", over="ignore"):
            restored = result.astype(source.dtype)
        if not np.array_equal(source, restored, equal_nan=True):
            raise ValueError("Counts cannot be represented exactly as float64.")
    return result
