import numpy as np
from scipy.spatial.distance import pdist, squareform

from srtd_nts import nts_scores, rtd_srtd_lite_scores, srtd_lite_barcode


def main() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(64, 8))
    y = x + 0.15 * rng.normal(size=(64, 8))

    distance_x = squareform(pdist(x, metric="euclidean"))
    distance_y = squareform(pdist(y, metric="euclidean"))

    print("NTS:")
    print(nts_scores(distance_x, distance_y))

    print("\nRTD-lite / SRTD-lite:")
    print(rtd_srtd_lite_scores(distance_x, distance_y))

    print("\nFirst five SRTD-lite barcode intervals:")
    print(srtd_lite_barcode(distance_x, distance_y)[:5])


if __name__ == "__main__":
    main()
