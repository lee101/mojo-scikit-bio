from __future__ import annotations

import math

import numpy as np

from .._lib import addr, f64, lib


def _counts(counts) -> np.ndarray:
    array = np.asarray(counts)
    if array.ndim != 1:
        raise ValueError("`counts` must be a 1-D array (vector).")
    if not (
        np.issubdtype(array.dtype, np.integer)
        or np.issubdtype(array.dtype, np.floating)
        or array.dtype == np.dtype("bool")
    ):
        raise ValueError("Counts must be integers or floating-point numbers.")
    if array.size and array.min() < 0:
        raise ValueError("Counts cannot contain negative values.")
    return f64(array)


def _value(counts, code, parameter1=0.0, parameter2=0.0, flag=0):
    array = _counts(counts)
    if array.size == 0:
        array = np.zeros(1, dtype=np.float64)
        length = 0
    else:
        length = len(array)
    return float(
        lib().msb_alpha(
            addr(array, dtype=np.float64),
            length,
            code,
            float(parameter1),
            float(parameter2),
            int(flag),
        )
    )


def observed_features(counts):
    return int(_value(counts, 0))


def sobs(counts):
    return int(_value(counts, 0))


def singles(counts):
    return int(_value(counts, 1))


def doubles(counts):
    return int(_value(counts, 2))


def osd(counts):
    return sobs(counts), singles(counts), doubles(counts)


def chao1(counts, bias_corrected=True):
    return _value(counts, 3, flag=bool(bias_corrected))


def berger_parker_d(counts):
    return _value(counts, 4)


def dominance(counts, finite=False):
    return _value(counts, 5, flag=bool(finite))


def simpson_d(counts, finite=False):
    return dominance(counts, finite=finite)


def shannon(counts, base=None, exp=False):
    return _value(counts, 6, 0.0 if base is None else base, flag=bool(exp))


def simpson(counts, finite=False):
    return _value(counts, 7, flag=bool(finite))


def inv_simpson(counts, finite=False):
    return _value(counts, 8, flag=bool(finite))


def enspie(counts, finite=False):
    return inv_simpson(counts, finite=finite)


def goods_coverage(counts):
    return _value(counts, 9)


def margalef(counts):
    return _value(counts, 10)


def menhinick(counts):
    return _value(counts, 11)


def mcintosh_d(counts):
    return _value(counts, 12)


def mcintosh_e(counts):
    return _value(counts, 13)


def pielou_e(counts, base=None):
    return _value(counts, 14, 0.0 if base is None else base)


def simpson_e(counts):
    return _value(counts, 15)


def heip_e(counts):
    return _value(counts, 16)


def robbins(counts):
    return _value(counts, 17)


def hill(counts, order=2):
    return _value(counts, 18, order if math.isfinite(order) else 0.0, flag=math.isinf(order))


def renyi(counts, order=2, base=None):
    return _value(
        counts,
        19,
        order if math.isfinite(order) else 0.0,
        0.0 if base is None else base,
        math.isinf(order),
    )


def tsallis(counts, order=2):
    return _value(counts, 20, order if math.isfinite(order) else 0.0, flag=math.isinf(order))


__all__ = [
    "berger_parker_d",
    "chao1",
    "dominance",
    "doubles",
    "enspie",
    "goods_coverage",
    "heip_e",
    "hill",
    "inv_simpson",
    "margalef",
    "mcintosh_d",
    "mcintosh_e",
    "menhinick",
    "observed_features",
    "osd",
    "pielou_e",
    "renyi",
    "robbins",
    "shannon",
    "simpson",
    "simpson_d",
    "simpson_e",
    "singles",
    "sobs",
    "tsallis",
]
