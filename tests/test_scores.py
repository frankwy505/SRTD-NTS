import sys

import numpy as np
import scipy.sparse as sps
from scipy.spatial.distance import pdist, squareform

from srtd_nts import (
    linear_cka,
    max_rtd_lite,
    nts_metrics,
    nts_scores,
    rtd_srtd_lite_metrics,
    rtd_srtd_lite_scores,
    srtd_lite,
    srtd_lite_barcode,
    srtd_score,
    symmetric_auxiliary_matrix,
    symmetric_auxiliary_sparse_matrix,
)


def _distance_matrices():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(16, 4))
    y = x + 0.1 * rng.normal(size=(16, 4))
    return squareform(pdist(x)), squareform(pdist(y))


def test_nts_identical_matrices_are_one():
    distance_x, _ = _distance_matrices()
    scores = nts_scores(distance_x, distance_x)
    assert np.isclose(scores["NTS_E"], 1.0)
    assert np.isclose(scores["NTS_M"], 1.0)


def test_lite_scores_are_nonnegative():
    distance_x, distance_y = _distance_matrices()
    scores = rtd_srtd_lite_scores(distance_x, distance_y)
    assert scores["RTD_lite"] >= 0.0
    assert scores["SRTD_lite"] >= 0.0
    assert scores["Max_RTD_lite"] >= 0.0
    assert np.isclose(scores["Max_RTD_lite"], max_rtd_lite(distance_x, distance_y))


def test_legacy_named_aliases_remain_available():
    distance_x, distance_y = _distance_matrices()
    assert nts_metrics(distance_x, distance_y) == nts_scores(distance_x, distance_y)
    assert rtd_srtd_lite_metrics(distance_x, distance_y) == rtd_srtd_lite_scores(distance_x, distance_y)


def test_srtd_lite_matches_barcode_sum():
    distance_x, distance_y = _distance_matrices()
    intervals = srtd_lite_barcode(distance_x, distance_y)
    barcode_sum = sum(death - birth for birth, death in intervals)
    assert np.isclose(srtd_lite(distance_x, distance_y), barcode_sum)


def test_lite_scores_satisfy_paper_identity():
    distance_x, distance_y = _distance_matrices()
    scores = rtd_srtd_lite_scores(distance_x, distance_y)
    assert np.isclose(
        scores["SRTD_lite"],
        scores["RTD_lite"] + scores["Max_RTD_lite"],
    )


def test_symmetric_auxiliary_matrix_matches_paper_blocks():
    distance_1 = np.array(
        [
            [0.0, 2.0, 5.0],
            [2.0, 0.0, 3.0],
            [5.0, 3.0, 0.0],
        ]
    )
    distance_2 = np.array(
        [
            [0.0, 4.0, 1.0],
            [4.0, 0.0, 6.0],
            [1.0, 6.0, 0.0],
        ]
    )

    auxiliary = symmetric_auxiliary_matrix(distance_1, distance_2, q=1.0)
    normalized_1 = distance_1 / 5.0
    normalized_2 = distance_2 / 6.0
    matrix_min = np.minimum(normalized_1, normalized_2)
    matrix_max = np.maximum(normalized_1, normalized_2)
    max_plus = matrix_max.copy()
    max_plus[np.triu_indices(3, k=1)] = np.inf

    expected = np.block(
        [
            [matrix_max, max_plus.T, np.zeros((3, 1))],
            [max_plus, matrix_min, np.full((3, 1), np.inf)],
            [np.zeros((1, 3)), np.full((1, 3), np.inf), np.zeros((1, 1))],
        ]
    )
    assert np.array_equal(np.isinf(auxiliary), np.isinf(expected))
    assert np.allclose(
        auxiliary[np.isfinite(expected)],
        expected[np.isfinite(expected)],
    )


def test_symmetric_auxiliary_sparse_matrix_preserves_finite_edges():
    distance_x, distance_y = _distance_matrices()
    dense = symmetric_auxiliary_matrix(distance_x, distance_y)
    sparse = symmetric_auxiliary_sparse_matrix(distance_x, distance_y)
    assert sps.isspmatrix_coo(sparse)

    lower_rows, lower_cols = np.tril_indices(dense.shape[0], k=-1)
    finite = np.isfinite(dense[lower_rows, lower_cols])
    expected = {
        (int(row), int(col), float(value))
        for row, col, value in zip(
            lower_rows[finite],
            lower_cols[finite],
            dense[lower_rows, lower_cols][finite],
        )
    }
    observed = {
        (int(row), int(col), float(value))
        for row, col, value in zip(sparse.row, sparse.col, sparse.data)
    }
    assert observed == expected
    assert any(value == 0.0 for _, _, value in observed)


def test_srtd_score_uses_sparse_full_symmetric_auxiliary(monkeypatch):
    distance_x, distance_y = _distance_matrices()
    captured = {}

    class FakeRipserPlusPlus:
        @staticmethod
        def run(args, data):
            captured["args"] = args
            captured["data"] = data
            return {
                "dgms": {
                    0: np.array([[0.0, 1.0], [0.2, np.inf]]),
                    1: np.array([[0.5, 0.75]]),
                }
            }

    monkeypatch.setitem(sys.modules, "ripserplusplus", FakeRipserPlusPlus)
    scores = srtd_score(distance_x, distance_y, max_dim=1)

    assert captured["args"] == "--format sparse --dim 1"
    assert sps.isspmatrix_coo(captured["data"])
    assert np.isclose(scores[0], 1.0)
    assert np.isclose(scores[1], 0.25)


def test_linear_cka_identical_representations_are_one():
    rng = np.random.default_rng(11)
    x = rng.normal(size=(16, 4))
    assert np.isclose(linear_cka(x, x), 1.0)


def test_synthetic_cluster_k2_regression():
    rng = np.random.RandomState(42)
    groups = [rng.multivariate_normal(np.zeros(2), np.eye(2), 5) for _ in range(60)]
    base = np.concatenate(groups)
    shifted = []
    for index, group in enumerate(groups):
        angle = 2 * np.pi * (index % 2) / 2
        shifted.append(group + 10 * np.array([np.cos(angle), np.sin(angle)]))

    distance_base = squareform(pdist(base))
    distance_shifted = squareform(pdist(np.concatenate(shifted)))
    nts = nts_scores(distance_base, distance_shifted)
    lite = rtd_srtd_lite_scores(distance_base, distance_shifted)

    assert np.isclose(nts["NTS_E"], 0.286550, atol=1e-6)
    assert np.isclose(nts["NTS_M"], 0.336100, atol=1e-6)
    assert np.isclose(lite["RTD_lite"], 7.164670, atol=1e-6)
    assert np.isclose(lite["SRTD_lite"], 19.950334, atol=1e-6)
    assert np.isclose(lite["Max_RTD_lite"], 12.785664, atol=1e-6)
