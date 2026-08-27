from __future__ import annotations

import inspect

import numpy as np
import pandas as pd

from .._lib import addr, f64, lib
from ..stats.distance import DistanceMatrix
from . import alpha

_ALPHA_METRICS = set(alpha.__all__) - {"osd"}

_BETA_CODES = {
    "euclidean": 0,
    "sqeuclidean": 1,
    "cityblock": 2,
    "manhattan": 2,
    "braycurtis": 3,
    "canberra": 4,
    "chebyshev": 5,
    "correlation": 6,
    "cosine": 7,
    "hamming": 8,
    "jaccard": 9,
    "dice": 10,
    "matching": 11,
    "rogerstanimoto": 12,
    "russellrao": 13,
    "sokalsneath": 14,
    "yule": 15,
    "minkowski": 18,
    "jensenshannon": 19,
}

_QUALITATIVE = {
    "dice",
    "jaccard",
    "matching",
    "rogerstanimoto",
    "russellrao",
    "sokalsneath",
    "yule",
}


def _matrix(
    counts, validate=True, *, preserve_int64=False, qualitative=False
) -> np.ndarray:
    array = np.asarray(counts)
    if array.ndim > 2:
        raise ValueError(
            f"`counts` has {array.ndim} dimensions whereas up to 2 dimensions are allowed."
        )
    array = np.atleast_2d(array)
    if validate:
        if not (
            np.issubdtype(array.dtype, np.integer)
            or np.issubdtype(array.dtype, np.floating)
            or array.dtype == np.dtype("bool")
        ):
            raise ValueError("Counts must be integers or floating-point numbers.")
        if array.size and array.min() < 0:
            raise ValueError("Counts cannot contain negative values.")
    if qualitative:
        return np.ascontiguousarray(array != 0, dtype=np.uint8)
    if preserve_int64 and array.dtype == np.dtype(np.int64):
        if array.size and array.max() > 2**53:
            raise ValueError("Counts cannot be represented exactly as float64.")
        if array.flags.c_contiguous and array.flags.aligned:
            return array
    return f64(array)


def _ids(ids, rows, beta=False):
    if ids is None:
        return [str(i) for i in range(rows)] if beta else list(range(rows))
    values = list(ids)
    if len(values) != rows:
        raise ValueError("Number of IDs must match the number of samples.")
    return values


def _take(kwargs, name, default):
    return kwargs.pop(name, default)


def _alpha_spec(metric, kwargs):
    options = dict(kwargs)
    code = parameter1 = parameter2 = flag = 0
    if metric in ("observed_features", "sobs"):
        code = 0
    elif metric == "singles":
        code = 1
    elif metric == "doubles":
        code = 2
    elif metric == "chao1":
        code, flag = 3, bool(_take(options, "bias_corrected", True))
    elif metric == "berger_parker_d":
        code = 4
    elif metric in ("dominance", "simpson_d"):
        code, flag = 5, bool(_take(options, "finite", False))
    elif metric == "shannon":
        code = 6
        base = _take(options, "base", None)
        parameter1 = 0.0 if base is None else base
        flag = bool(_take(options, "exp", False))
    elif metric == "simpson":
        code, flag = 7, bool(_take(options, "finite", False))
    elif metric in ("inv_simpson", "enspie"):
        code, flag = 8, bool(_take(options, "finite", False))
    elif metric == "goods_coverage":
        code = 9
    elif metric == "margalef":
        code = 10
    elif metric == "menhinick":
        code = 11
    elif metric == "mcintosh_d":
        code = 12
    elif metric == "mcintosh_e":
        code = 13
    elif metric == "pielou_e":
        code = 14
        base = _take(options, "base", None)
        parameter1 = 0.0 if base is None else base
    elif metric == "simpson_e":
        code = 15
    elif metric == "heip_e":
        code = 16
    elif metric == "robbins":
        code = 17
    elif metric == "hill":
        code = 18
        order = _take(options, "order", 2)
        flag = np.isposinf(order)
        parameter1 = 0.0 if flag else order
    elif metric == "renyi":
        code = 19
        order = _take(options, "order", 2)
        flag = np.isposinf(order)
        parameter1 = 0.0 if flag else order
        base = _take(options, "base", None)
        parameter2 = 0.0 if base is None else base
    elif metric == "tsallis":
        code = 20
        order = _take(options, "order", 2)
        flag = np.isposinf(order)
        parameter1 = 0.0 if flag else order
    else:
        raise ValueError(
            f'"{metric}" is not an available alpha diversity metric name. '
            "Refer to `get_alpha_diversity_metrics` for a list of available metrics."
        )
    if options:
        key = next(iter(options))
        raise TypeError(f"got an unexpected keyword argument {key!r}")
    return int(code), float(parameter1), float(parameter2), int(flag)


def alpha_diversity(metric, counts, ids=None, validate=True, **kwargs):
    if callable(metric):
        matrix = _matrix(counts, validate=validate)
        sample_ids = _ids(ids, len(matrix))
        values = [metric(row, **kwargs) for row in matrix]
        return pd.Series(values, index=sample_ids)
    if not isinstance(metric, str):
        raise ValueError(f"Invalid metric provided: {metric!r}.")
    code, parameter1, parameter2, flag = _alpha_spec(metric, kwargs)
    matrix = _matrix(
        counts, validate=validate, preserve_int64=validate and code in (0, 6)
    )
    sample_ids = _ids(ids, len(matrix))
    result = np.empty(len(matrix), dtype=np.float64)
    if len(matrix) == 0:
        return pd.Series(result, index=sample_ids)
    if matrix.shape[1] == 0:
        matrix = np.zeros((len(matrix), 1), dtype=np.float64)
        columns = 0
    else:
        columns = matrix.shape[1]
    if matrix.dtype == np.dtype(np.int64) and code in (0, 6):
        lib().msb_alpha_batch_i64(
            addr(matrix, dtype=np.int64),
            len(matrix),
            columns,
            addr(result, dtype=np.float64, writable=True),
            code,
            parameter1,
            flag,
        )
    else:
        lib().msb_alpha_batch(
            addr(matrix, dtype=np.float64),
            len(matrix),
            columns,
            addr(result, dtype=np.float64, writable=True),
            code,
            parameter1,
            parameter2,
            flag,
        )
    if code in (0, 1, 2):
        result = result.astype(np.int64)
    return pd.Series(result, index=sample_ids)


def beta_diversity(
    metric,
    counts,
    ids=None,
    validate=True,
    pairwise_func=None,
    **kwargs,
):
    qualitative = isinstance(metric, str) and metric in _QUALITATIVE
    matrix = _matrix(counts, validate=validate, qualitative=qualitative)
    sample_ids = _ids(ids, len(matrix), beta=True)
    if not matrix.size:
        return DistanceMatrix(np.zeros((len(matrix), len(matrix))), sample_ids)
    if pairwise_func is not None:
        result = pairwise_func(matrix, metric=metric, **kwargs)
        return DistanceMatrix(result, sample_ids)
    if callable(metric):
        result = np.zeros((len(matrix), len(matrix)), dtype=np.float64)
        signature = inspect.signature(metric)
        for i in range(len(matrix)):
            for j in range(i + 1, len(matrix)):
                value = metric(matrix[i], matrix[j], **kwargs)
                result[i, j] = result[j, i] = value
        return DistanceMatrix(result, sample_ids, validate=False)
    if not isinstance(metric, str) or metric not in _BETA_CODES:
        raise ValueError(f"Unknown Distance Metric: {metric}")
    parameter = kwargs.pop("p", 2.0) if metric == "minkowski" else 0.0
    if kwargs:
        key = next(iter(kwargs))
        raise TypeError(f"got an unexpected keyword argument {key!r}")
    result = np.empty((len(matrix), len(matrix)), dtype=np.float64)
    if qualitative:
        lib().msb_beta_bool(
            addr(matrix, dtype=np.uint8),
            len(matrix),
            matrix.shape[1],
            addr(result, dtype=np.float64, writable=True),
            _BETA_CODES[metric],
        )
    else:
        lib().msb_beta(
            addr(matrix, dtype=np.float64),
            len(matrix),
            matrix.shape[1],
            addr(result, dtype=np.float64, writable=True),
            _BETA_CODES[metric],
            float(parameter),
        )
    return DistanceMatrix(result, sample_ids, validate=False)


def get_alpha_diversity_metrics():
    return sorted(_ALPHA_METRICS)


def get_beta_diversity_metrics():
    return sorted(_BETA_CODES)
