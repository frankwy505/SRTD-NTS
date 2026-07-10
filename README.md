# SRTD-NTS

Code for **Symmetric Divergence and Normalized Similarity: A Unified Topological Framework for Representation Analysis**.Accepted by TMLR.

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

On a typical CUDA Linux server, the minimal setup is:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install -e .
```

### Optional full SRTD backend

The default installation is sufficient for NTS, RTD-lite, SRTD-lite,
Max-RTD-lite, and linear CKA. The full persistent-homology SRTD entry point
(`srtd_score`) additionally requires the `ripserplusplus` Python bindings.

Following the [RTD-AE repository](https://github.com/danchern97/RTD_AE),
we use the [RipserZeros](https://github.com/ArGintum/RipserZeros) fork of
Ripser++:

```bash
pip install -e ".[full]"
```

This installs the pinned optional dependency:

```bash
pip install "ripserplusplus @ git+https://github.com/ArGintum/RipserZeros.git@bac8a96f56e9e3ed46202323accbeeee11c4b54c"
```

Equivalently, install the optional requirements file directly:

```bash
pip install -r requirements-full.txt
```

If the pinned install fails on your server, install RipserZeros directly by
following the upstream repository instructions at
[ArGintum/RipserZeros](https://github.com/ArGintum/RipserZeros), then return here
and run the smoke test below.

RipserZeros/Ripser++ is a compiled optional dependency intended for 64-bit Linux
systems with CMake, CUDA/NVCC, GCC, NumPy, and SciPy. If this backend is
unavailable, use `srtd_lite` or `rtd_srtd_lite_scores` instead of `srtd_score`.
After installing the optional backend, run the smoke test:

```bash
python - <<'PY'
import ripserplusplus as rpp_py
print("ripserplusplus import OK")
PY

python examples/full_srtd_usage.py
```

### Which SRTD entry point should I use?

For most users, start with `srtd_lite` or `rtd_srtd_lite_scores`. These functions
are MST-based, require only NumPy/SciPy, and are the intended fast path in this
package.

Use `srtd_score` only when full persistent-homology SRTD is specifically needed.
It constructs the symmetric auxiliary distance matrix and then calls
RipserZeros/Ripser++, so it is more demanding to install and much more expensive
to run. The helper `symmetric_auxiliary_matrix` implements the paper's `M_sym`
block matrix, while `srtd_score` passes the finite lower-triangular entries to
RipserZeros as a sparse COO matrix, preserving the zero-weight off-diagonal
edges required by the mapping-cone construction.

The implementation expects dense input pairwise distance matrices, so memory is
quadratic in the number of samples: one `float64` `5000 x 5000` matrix is about
200 MB before temporary copies. The full SRTD auxiliary matrix has size
`(2n + 1) x (2n + 1)`, and persistent-homology computation can dominate runtime.

In short: `SRTD-lite` is the practical default; full `SRTD` is an optional,
heavier backend for smaller or well-provisioned Linux/CUDA runs.

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
- **Ripser++ / RipserZeros**: the optional full persistent-homology path uses `ripserplusplus`, the GPU-accelerated Vietoris-Rips persistence software by Simon Zhang, Mengbai Xiao, and Hao Wang. The pinned installation command follows the RipserZeros dependency used in the RTD-AE repository. The Ripser++ project also credits Birkan Gokbag and Ryan DeMilt as contributors. Ripser++ is built on **Ripser**, written by Ulrich Bauer.
- **RTD-AE**: the optional RipserZeros dependency is referenced from the official RTD-AE repository by Trofimov, Cherniavskii, Tulchinskii, Balabin, Barannikov, and Burnaev.

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
  full_srtd_usage.py
                    Full SRTD smoke test; requires ripserplusplus
  synthetic_clusters.py
                    Clean synthetic cluster experiment
tests/
  test_scores.py    Lightweight sanity checks
```

## Notes

Inputs are pairwise dissimilarity matrices for the same ordered samples. Matrices should be square, symmetric, nonnegative, and have a zero diagonal.
