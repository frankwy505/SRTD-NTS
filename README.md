# SRTD-NTS

Code for **Symmetric Divergence and Normalized Similarity: A Unified Topological Framework for Representation Analysis**.

This repository currently contains a clean minimal implementation of the core metrics from the paper:

- **NTS-E** and **NTS-M**: normalized topological similarity scores based on MST core edges.
- **RTD-lite**, **SRTD-lite**, and **Max-RTD-lite**: MST-based topological divergence scores.
- **Linear CKA**: a standard representation-similarity baseline used in the experiments.
- **Full SRTD score**: optional persistent-homology implementation when `ripserplusplus` is installed.

The repository intentionally excludes datasets, model checkpoints, notebooks, cached representations, and training logs.

## Installation

```bash
pip install -r requirements.txt
pip install -e .
```

For full SRTD persistent-homology scores, install `ripserplusplus` separately. The lite metrics and NTS metrics only require NumPy and SciPy.

## Quick Example

```python
import numpy as np
from scipy.spatial.distance import pdist, squareform
from srtd_nts import nts_metrics, rtd_srtd_lite_metrics

rng = np.random.default_rng(0)
x = rng.normal(size=(64, 8))
y = x + 0.15 * rng.normal(size=(64, 8))

w_x = squareform(pdist(x, metric="euclidean"))
w_y = squareform(pdist(y, metric="euclidean"))

print(nts_metrics(w_x, w_y))
print(rtd_srtd_lite_metrics(w_x, w_y))
```

You can also run:

```bash
python examples/basic_usage.py
```

For the synthetic cluster experiment:

```bash
python examples/synthetic_clusters.py --out outputs/synthetic_clusters.csv
```

The script recreates the cluster-splitting experiment from the research notebooks with the cleaned package implementations. It reports NTS-E/NTS-M, RTD-lite, SRTD-lite, SRTD-lite/2, Max-RTD-lite, and linear CKA. Add `--plot` to save figures if `matplotlib` is installed.

## Repository Layout

```text
src/srtd_nts/
  metrics.py        Core NTS, CKA, RTD-lite, SRTD-lite, and optional full SRTD scores
examples/
  basic_usage.py    Minimal runnable example on synthetic representations
  synthetic_clusters.py
                    Clean synthetic cluster experiment
tests/
  test_metrics.py   Lightweight sanity checks
```

## Notes

Inputs are pairwise dissimilarity matrices for the same ordered samples. Matrices should be square, symmetric, nonnegative, and have a zero diagonal.
