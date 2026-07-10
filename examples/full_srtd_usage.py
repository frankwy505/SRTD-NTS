import numpy as np
from scipy.spatial.distance import pdist, squareform

from srtd_nts import srtd_score


def main() -> None:
    rng = np.random.default_rng(123)
    x = rng.normal(size=(12, 4))
    y = x + 0.2 * rng.normal(size=(12, 4))

    distance_x = squareform(pdist(x, metric="euclidean"))
    distance_y = squareform(pdist(y, metric="euclidean"))

    print(srtd_score(distance_x, distance_y, max_dim=1))


if __name__ == "__main__":
    main()
