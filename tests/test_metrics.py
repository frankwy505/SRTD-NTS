import numpy as np
from scipy.spatial.distance import pdist, squareform

from srtd_nts import nts_metrics, rtd_srtd_lite_metrics, srtd_lite, srtd_lite_barcode


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


def test_srtd_lite_matches_barcode_sum():
    distance_x, distance_y = _distance_matrices()
    intervals = srtd_lite_barcode(distance_x, distance_y)
    barcode_sum = sum(death - birth for birth, death in intervals)
    assert np.isclose(srtd_lite(distance_x, distance_y), barcode_sum)

