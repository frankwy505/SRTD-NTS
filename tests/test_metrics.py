import numpy as np
from scipy.spatial.distance import pdist, squareform

from srtd_nts import (
    linear_cka,
    max_rtd_lite,
    nts_metrics,
    rtd_srtd_lite_metrics,
    srtd_lite,
    srtd_lite_barcode,
)


def _distance_matrices():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(16, 4))
    y = x + 0.1 * rng.normal(size=(16, 4))
    return squareform(pdist(x)), squareform(pdist(y))


def test_nts_identical_matrices_are_one():
    distance_x, _ = _distance_matrices()
    metrics = nts_metrics(distance_x, distance_x)
    assert np.isclose(metrics["NTS_E"], 1.0)
    assert np.isclose(metrics["NTS_M"], 1.0)


def test_lite_metrics_are_nonnegative():
    distance_x, distance_y = _distance_matrices()
    metrics = rtd_srtd_lite_metrics(distance_x, distance_y)
    assert metrics["RTD_lite"] >= 0.0
    assert metrics["SRTD_lite"] >= 0.0
    assert metrics["Max_RTD_lite"] >= 0.0
    assert np.isclose(metrics["Max_RTD_lite"], max_rtd_lite(distance_x, distance_y))


def test_srtd_lite_matches_barcode_sum():
    distance_x, distance_y = _distance_matrices()
    intervals = srtd_lite_barcode(distance_x, distance_y)
    barcode_sum = sum(death - birth for birth, death in intervals)
    assert np.isclose(srtd_lite(distance_x, distance_y), barcode_sum)


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
    nts = nts_metrics(distance_base, distance_shifted)
    lite = rtd_srtd_lite_metrics(distance_base, distance_shifted)

    assert np.isclose(nts["NTS_E"], 0.286550, atol=1e-6)
    assert np.isclose(nts["NTS_M"], 0.336100, atol=1e-6)
    assert np.isclose(lite["RTD_lite"], 7.164670, atol=1e-6)
    assert np.isclose(lite["SRTD_lite"], 19.950334, atol=1e-6)
    assert np.isclose(lite["Max_RTD_lite"], 12.785664, atol=1e-6)
