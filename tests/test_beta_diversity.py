import numpy as np
import pandas as pd
import pytest
from scipy.spatial.distance import pdist

from skbio.diversity import beta_diversity as sk_beta_diversity

from mojoskbio import DistanceMatrix
from mojoskbio.diversity import beta_diversity, get_beta_diversity_metrics


METRICS = [
    "euclidean",
    "sqeuclidean",
    "cityblock",
    "manhattan",
    "braycurtis",
    "canberra",
    "chebyshev",
    "correlation",
    "cosine",
    "hamming",
    "jaccard",
    "dice",
    "matching",
    "rogerstanimoto",
    "russellrao",
    "sokalsneath",
    "yule",
    "minkowski",
    "jensenshannon",
]


@pytest.mark.parametrize("metric", METRICS)
def test_beta_diversity_parity(metric):
    rng = np.random.default_rng(12)
    counts = rng.integers(0, 20, size=(24, 35))
    kwargs = {"p": 3.25} if metric == "minkowski" else {}
    ids = [f"s{i}" for i in range(len(counts))]
    actual = beta_diversity(metric, counts, ids=ids, **kwargs)
    expected = sk_beta_diversity(metric, counts, ids=ids, **kwargs)
    assert actual.ids == expected.ids
    assert np.allclose(
        actual.data, expected.data, rtol=2e-8, atol=1e-11, equal_nan=True
    )


@pytest.mark.parametrize(
    ("rows", "columns"),
    [
        (4, 11),
        (48, 137),
    ],
)
def test_braycurtis_simd_tail_and_parallel_threshold(rows, columns):
    rng = np.random.default_rng(rows + columns)
    counts = rng.integers(0, 20, size=(rows, columns))
    actual = beta_diversity("braycurtis", counts)
    expected = sk_beta_diversity("braycurtis", counts)
    assert np.allclose(
        actual.data, expected.data, rtol=2e-8, atol=1e-11, equal_nan=True
    )


def test_beta_callable_parity():
    counts = np.array([[1, 2], [3, 7], [0, 4]], dtype=float)

    def scaled_l1(left, right, scale=1):
        return np.abs(left - right).sum() * scale

    actual = beta_diversity(scaled_l1, counts, scale=2)
    expected = sk_beta_diversity(scaled_l1, counts, scale=2)
    assert np.array_equal(actual.data, expected.data)


def test_pairwise_func_condensed():
    counts = np.array([[1, 2], [3, 7], [0, 4]], dtype=float)
    actual = beta_diversity("euclidean", counts, pairwise_func=pdist)
    expected = sk_beta_diversity("euclidean", counts)
    assert np.allclose(actual.data, expected.data)


def test_empty_beta_matrix():
    actual = beta_diversity("euclidean", np.empty((3, 0)), ids=["a", "b", "c"])
    assert actual.ids == ("a", "b", "c")
    assert np.array_equal(actual.data, np.zeros((3, 3)))


def test_distance_matrix_surface():
    matrix = DistanceMatrix([[0, 1, 2], [1, 0, 3], [2, 3, 0]], ["a", "b", "c"])
    assert matrix["a", "c"] == 2
    assert np.array_equal(matrix.condensed_form(), [1, 2, 3])
    assert matrix.copy() == matrix
    frame = matrix.to_data_frame()
    assert isinstance(frame, pd.DataFrame)
    assert frame.loc["b", "c"] == 3
    restored = DistanceMatrix(matrix.condensed_form(), matrix.ids)
    assert restored == matrix


def test_metric_list_and_validation():
    assert get_beta_diversity_metrics() == sorted(METRICS)
    with pytest.raises(ValueError):
        beta_diversity("unweighted_unifrac", [[1, 2], [2, 1]])
    with pytest.raises(ValueError):
        beta_diversity("euclidean", [[1, -1], [2, 3]])
