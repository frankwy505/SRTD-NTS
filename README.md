# SRTD-NTS

Code for **Symmetric Divergence and Normalized Similarity: A Unified Topological Framework for Representation Analysis**.

Paper status: the [OpenReview record](https://openreview.net/forum?id=pGgJ9qB2Io) currently lists the submission as decision pending for TMLR, with a public recommendation of **Accept as is**.

This repository currently contains a clean minimal implementation of the core representation-analysis scores from the paper:

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

For full SRTD persistent-homology scores, install `ripserplusplus` separately. The lite scores and NTS scores only require NumPy and SciPy.

## Quick Example

```python
import numpy as np
from scipy.spatial.distance import pdist, squareform
from srtd_nts import nts_scores, rtd_srtd_lite_scores

rng = np.random.default_rng(0)
x = rng.normal(size=(64, 8))
y = x + 0.15 * rng.normal(size=(64, 8))

w_x = squareform(pdist(x, metric="euclidean"))
w_y = squareform(pdist(y, metric="euclidean"))

print(nts_scores(w_x, w_y))
print(rtd_srtd_lite_scores(w_x, w_y))
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

## References and Acknowledgements

This code builds on ideas and tools from prior topological representation analysis work:

- **RTD**: the original Representation Topology Divergence paper by Barannikov, Trofimov, Balabin, and Burnaev, with the official repository at [IlyaTrofimov/RTD](https://github.com/IlyaTrofimov/RTD).
- **RTD-Lite**: the scalable MST-based RTD-Lite method by Tulchinskii, Voronkova, Trofimov, Burnaev, and Barannikov, with the official repository at [ArGintum/RTD-Lite](https://github.com/ArGintum/RTD-Lite).
- **Ripser++**: the optional full persistent-homology path uses `ripserplusplus`, the GPU-accelerated Vietoris-Rips persistence software by Simon Zhang, Mengbai Xiao, and Hao Wang. The Ripser++ project also credits Birkan Gokbag and Ryan DeMilt as contributors. Ripser++ is built on **Ripser**, written by Ulrich Bauer.

See [CITATION.bib](CITATION.bib) for BibTeX entries. The lite scores and NTS scores in this repository are implemented independently in NumPy/SciPy; upstream RTD, RTD-Lite, Ripser, and Ripser++ code is not vendored here.

We use "scores", "similarities", and "divergences" for the returned quantities. They should not be interpreted as mathematical distances unless explicitly stated.

## Repository Layout

```text
CITATION.bib       BibTeX entries for related work and optional dependencies
src/srtd_nts/
  scores.py         Core NTS, CKA, RTD-lite, SRTD-lite, and optional full SRTD scores
  metrics.py        Backward-compatible import shim
examples/
  basic_usage.py    Minimal runnable example on synthetic representations
  synthetic_clusters.py
                    Clean synthetic cluster experiment
tests/
  test_scores.py    Lightweight sanity checks
```

## Notes

Inputs are pairwise dissimilarity matrices for the same ordered samples. Matrices should be square, symmetric, nonnegative, and have a zero diagonal.
