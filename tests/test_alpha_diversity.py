import numpy as np
import pandas as pd
import pytest

from skbio.diversity import alpha_diversity as sk_alpha_diversity
from skbio.diversity import alpha as sk_alpha

from mojoskbio.diversity import alpha_diversity
from mojoskbio.diversity import alpha as mojo_alpha

pytestmark = pytest.mark.filterwarnings(
    "ignore:invalid value encountered in scalar divide:RuntimeWarning"
)


METRIC_CASES = [
    ("observed_features", {}),
    ("sobs", {}),
    ("singles", {}),
    ("doubles", {}),
    ("chao1", {}),
    ("chao1", {"bias_corrected": False}),
    ("berger_parker_d", {}),
    ("dominance", {}),
    ("dominance", {"finite": True}),
    ("simpson_d", {}),
    ("shannon", {}),
    ("shannon", {"base": 2}),
    ("shannon", {"exp": True}),
    ("simpson", {}),
    ("simpson", {"finite": True}),
    ("inv_simpson", {}),
    ("inv_simpson", {"finite": True}),
    ("enspie", {}),
    ("goods_coverage", {}),
    ("margalef", {}),
    ("menhinick", {}),
    ("mcintosh_d", {}),
    ("mcintosh_e", {}),
    ("pielou_e", {}),
    ("pielou_e", {"base": 2}),
    ("simpson_e", {}),
    ("heip_e", {}),
    ("robbins", {}),
    ("hill", {}),
    ("hill", {"order": 0}),
    ("hill", {"order": 1}),
    ("hill", {"order": np.inf}),
    ("renyi", {}),
    ("renyi", {"order": 0, "base": 2}),
    ("renyi", {"order": 1}),
    ("renyi", {"order": np.inf}),
    ("tsallis", {}),
    ("tsallis", {"order": 0}),
    ("tsallis", {"order": 1}),
    ("tsallis", {"order": np.inf}),
]


@pytest.mark.parametrize("metric,kwargs", METRIC_CASES)
@pytest.mark.parametrize(
    "counts",
    [
        np.array([1, 2, 3, 0, 8, 1]),
        np.array([0, 0, 0]),
        np.array([1]),
        np.array([0.5, 1.5, 3.0, 0.0]),
    ],
)
def test_direct_alpha_metric_parity(metric, kwargs, counts):
    actual = getattr(mojo_alpha, metric)(counts, **kwargs)
    expected = getattr(sk_alpha, metric)(counts, **kwargs)
    assert actual == pytest.approx(expected, rel=2e-8, abs=1e-11, nan_ok=True)


def test_osd_parity():
    counts = [0, 1, 1, 2, 3, 9]
    assert mojo_alpha.osd(counts) == sk_alpha.osd(counts)


def test_all_advertised_alpha_metrics_are_listed_and_tested():
    expected = {name for name, _ in METRIC_CASES}
    assert set(alpha_diversity.__globals__["get_alpha_diversity_metrics"]()) == expected


@pytest.mark.parametrize("metric,kwargs", METRIC_CASES)
def test_alpha_diversity_batch_parity(metric, kwargs):
    rng = np.random.default_rng(4)
    counts = rng.integers(0, 30, size=(75, 120))
    counts[0] = 0
    ids = [f"sample-{i}" for i in range(len(counts))]
    actual = alpha_diversity(metric, counts, ids=ids, **kwargs)
    expected = sk_alpha_diversity(metric, counts, ids=ids, **kwargs)
    pd.testing.assert_index_equal(actual.index, expected.index)
    assert np.allclose(actual, expected, rtol=2e-8, atol=1e-11, equal_nan=True)


@pytest.mark.parametrize("metric", ["observed_features", "shannon"])
@pytest.mark.parametrize(
    ("rows", "columns", "dtype"),
    [
        (5, 13, np.float64),
        (5, 13, np.int64),
        (2048, 137, np.int64),
    ],
)
def test_alpha_simd_tail_and_parallel_threshold(metric, rows, columns, dtype):
    rng = np.random.default_rng(rows + columns)
    counts = rng.integers(0, 20, size=(rows, columns)).astype(dtype)
    actual = alpha_diversity(metric, counts)
    expected = sk_alpha_diversity(metric, counts)
    assert np.allclose(actual, expected, rtol=2e-8, atol=1e-11, equal_nan=True)


def test_alpha_diversity_callable_and_vector():
    metric = lambda row, scale=1: np.count_nonzero(row) * scale
    actual = alpha_diversity(metric, [1, 0, 2], scale=3)
    assert actual.index.tolist() == [0]
    assert actual.tolist() == [6]


def test_alpha_diversity_zero_rows_does_not_cross_ffi_with_null_pointers():
    actual = alpha_diversity("shannon", np.empty((0, 3)))
    assert actual.empty


@pytest.mark.parametrize(
    "counts",
    [
        np.array([2**53 + 1], dtype=np.uint64),
        np.array([np.longdouble("1.0000000000000000001")]),
    ],
)
def test_alpha_rejects_silent_float64_narrowing(counts):
    with pytest.raises(ValueError, match="represented exactly"):
        mojo_alpha.shannon(counts)


@pytest.mark.parametrize("counts", [[1, -1], [["not", "counts"]], np.ones((2, 2, 2))])
def test_alpha_validation(counts):
    with pytest.raises(ValueError):
        mojo_alpha.shannon(counts)
