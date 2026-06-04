from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.stats import spearmanr


Edge = tuple[int, int]


@dataclass(frozen=True)
class MST:
    edges: list[Edge]
    weights: np.ndarray
    total_weight: float


class UnionFind:
    def __init__(self, n_vertices: int):
        self.parent = list(range(n_vertices))
        self.rank = [0] * n_vertices

    def find(self, vertex: int) -> int:
        root = vertex
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[vertex] != vertex:
            parent = self.parent[vertex]
            self.parent[vertex] = root
            vertex = parent
        return root

    def connected(self, a: int, b: int) -> bool:
        return self.find(a) == self.find(b)

    def union(self, a: int, b: int) -> bool:
        root_a = self.find(a)
        root_b = self.find(b)
        if root_a == root_b:
            return False
        if self.rank[root_a] < self.rank[root_b]:
            root_a, root_b = root_b, root_a
        self.parent[root_b] = root_a
        if self.rank[root_a] == self.rank[root_b]:
            self.rank[root_a] += 1
        return True

    def copy(self) -> "UnionFind":
        other = UnionFind(0)
        other.parent = self.parent.copy()
        other.rank = self.rank.copy()
        return other


def validate_distance_matrix(distance_matrix: np.ndarray, *, name: str = "distance_matrix") -> np.ndarray:
    matrix = np.asarray(distance_matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{name} must be a square matrix.")
    if not np.all(np.isfinite(matrix[np.triu_indices_from(matrix, k=1)])):
        raise ValueError(f"{name} contains non-finite off-diagonal values.")
    if np.any(matrix < 0):
        raise ValueError(f"{name} must be nonnegative.")
    if not np.allclose(matrix, matrix.T, atol=1e-8):
        raise ValueError(f"{name} must be symmetric.")
    matrix = matrix.copy()
    np.fill_diagonal(matrix, 0.0)
    return matrix


def validate_pair(distance_matrix_1: np.ndarray, distance_matrix_2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    matrix_1 = validate_distance_matrix(distance_matrix_1, name="distance_matrix_1")
    matrix_2 = validate_distance_matrix(distance_matrix_2, name="distance_matrix_2")
    if matrix_1.shape != matrix_2.shape:
        raise ValueError("Distance matrices must have the same shape.")
    return matrix_1, matrix_2


def upper_triangular_values(distance_matrix: np.ndarray) -> np.ndarray:
    matrix = validate_distance_matrix(distance_matrix)
    return matrix[np.triu_indices_from(matrix, k=1)]


def normalize_by_quantile(distance_matrix: np.ndarray, q: float = 0.90) -> np.ndarray:
    if not 0.0 < q <= 1.0:
        raise ValueError("q must be in (0, 1].")
    matrix = validate_distance_matrix(distance_matrix)
    values = matrix[np.triu_indices_from(matrix, k=1)]
    values = values[values > 0]
    if values.size == 0:
        return matrix
    scale = float(np.quantile(values, q))
    if scale <= 1e-12:
        return matrix
    return matrix / scale


def mst(distance_matrix: np.ndarray) -> MST:
    matrix = validate_distance_matrix(distance_matrix)
    tree = minimum_spanning_tree(matrix).tocoo()
    edges: list[Edge] = []
    weights: list[float] = []
    for i, j, weight in zip(tree.row, tree.col, tree.data):
        a, b = int(i), int(j)
        if a > b:
            a, b = b, a
        edges.append((a, b))
        weights.append(float(weight))
    weights_array = np.asarray(weights, dtype=float)
    return MST(edges=edges, weights=weights_array, total_weight=float(weights_array.sum()))


def core_mst_edges(distance_matrix_1: np.ndarray, distance_matrix_2: np.ndarray) -> list[Edge]:
    matrix_1, matrix_2 = validate_pair(distance_matrix_1, distance_matrix_2)
    return sorted(set(mst(matrix_1).edges).union(mst(matrix_2).edges))


def _safe_spearman(x: Sequence[float], y: Sequence[float]) -> float:
    x_array = np.asarray(x, dtype=float)
    y_array = np.asarray(y, dtype=float)
    if len(x_array) < 2 or np.std(x_array) <= 1e-12 or np.std(y_array) <= 1e-12:
        return np.nan
    return float(spearmanr(x_array, y_array).statistic)


def _tree_adjacency(num_nodes: int, edges: Sequence[Edge], weights: Sequence[float]) -> list[list[tuple[int, float]]]:
    adjacency: list[list[tuple[int, float]]] = [[] for _ in range(num_nodes)]
    for (a, b), weight in zip(edges, weights):
        adjacency[a].append((b, float(weight)))
        adjacency[b].append((a, float(weight)))
    return adjacency


def _bottleneck_path_value(adjacency: list[list[tuple[int, float]]], source: int, target: int) -> float:
    stack = [(source, -1, 0.0)]
    while stack:
        node, parent, current_max = stack.pop()
        if node == target:
            return current_max
        for next_node, weight in adjacency[node]:
            if next_node != parent:
                stack.append((next_node, node, max(current_max, weight)))
    raise RuntimeError("Tree path not found.")


def _bottleneck_values(num_nodes: int, tree: MST, pairs: Iterable[Edge]) -> np.ndarray:
    adjacency = _tree_adjacency(num_nodes, tree.edges, tree.weights)
    return np.asarray(
        [_bottleneck_path_value(adjacency, a, b) for a, b in pairs],
        dtype=float,
    )


def nts_e(distance_matrix_1: np.ndarray, distance_matrix_2: np.ndarray) -> float:
    matrix_1, matrix_2 = validate_pair(distance_matrix_1, distance_matrix_2)
    core_edges = core_mst_edges(matrix_1, matrix_2)
    values_1 = np.asarray([matrix_1[a, b] for a, b in core_edges], dtype=float)
    values_2 = np.asarray([matrix_2[a, b] for a, b in core_edges], dtype=float)
    return _safe_spearman(values_1, values_2)


def nts_m(distance_matrix_1: np.ndarray, distance_matrix_2: np.ndarray) -> float:
    matrix_1, matrix_2 = validate_pair(distance_matrix_1, distance_matrix_2)
    tree_1 = mst(matrix_1)
    tree_2 = mst(matrix_2)
    core_edges = sorted(set(tree_1.edges).union(tree_2.edges))
    values_1 = _bottleneck_values(matrix_1.shape[0], tree_1, core_edges)
    values_2 = _bottleneck_values(matrix_2.shape[0], tree_2, core_edges)
    return _safe_spearman(values_1, values_2)


def nts_metrics(distance_matrix_1: np.ndarray, distance_matrix_2: np.ndarray) -> dict[str, float]:
    nts_e_value = nts_e(distance_matrix_1, distance_matrix_2)
    nts_m_value = nts_m(distance_matrix_1, distance_matrix_2)
    return {
        "NTS_E": nts_e_value,
        "NTS_M": nts_m_value,
        "D_NTS_E": (1.0 - nts_e_value) / 2.0 if np.isfinite(nts_e_value) else np.nan,
        "D_NTS_M": (1.0 - nts_m_value) / 2.0 if np.isfinite(nts_m_value) else np.nan,
    }


def rtd_lite(
    distance_matrix_1: np.ndarray,
    distance_matrix_2: np.ndarray,
    *,
    q: float = 0.90,
    symmetric: bool = True,
) -> float:
    matrix_1, matrix_2 = validate_pair(distance_matrix_1, distance_matrix_2)
    matrix_1 = normalize_by_quantile(matrix_1, q=q)
    matrix_2 = normalize_by_quantile(matrix_2, q=q)
    matrix_min = np.minimum(matrix_1, matrix_2)

    mst_1 = mst(matrix_1).total_weight
    mst_min = mst(matrix_min).total_weight
    directional_12 = mst_1 - mst_min
    if not symmetric:
        return float(max(0.0, directional_12))

    mst_2 = mst(matrix_2).total_weight
    directional_21 = mst_2 - mst_min
    return float(max(0.0, 0.5 * (directional_12 + directional_21)))


def srtd_lite(distance_matrix_1: np.ndarray, distance_matrix_2: np.ndarray, *, q: float = 0.90) -> float:
    matrix_1, matrix_2 = validate_pair(distance_matrix_1, distance_matrix_2)
    matrix_1 = normalize_by_quantile(matrix_1, q=q)
    matrix_2 = normalize_by_quantile(matrix_2, q=q)
    matrix_min = np.minimum(matrix_1, matrix_2)
    matrix_max = np.maximum(matrix_1, matrix_2)
    value = mst(matrix_max).total_weight - mst(matrix_min).total_weight
    return float(max(0.0, value))


def srtd_lite_barcode(
    distance_matrix_1: np.ndarray,
    distance_matrix_2: np.ndarray,
    *,
    q: float = 0.90,
) -> list[tuple[float, float]]:
    matrix_1, matrix_2 = validate_pair(distance_matrix_1, distance_matrix_2)
    matrix_1 = normalize_by_quantile(matrix_1, q=q)
    matrix_2 = normalize_by_quantile(matrix_2, q=q)
    matrix_min = np.minimum(matrix_1, matrix_2)
    matrix_max = np.maximum(matrix_1, matrix_2)

    n_vertices = matrix_1.shape[0]
    min_tree = mst(matrix_min)
    max_tree = mst(matrix_max)
    min_edges = sorted(zip(min_tree.edges, min_tree.weights), key=lambda item: (item[1], item[0]))
    max_edges = sorted(zip(max_tree.edges, max_tree.weights), key=lambda item: (item[1], item[0]))

    subtree = UnionFind(n_vertices)
    intervals: list[tuple[float, float]] = []
    for (u, v), birth in min_edges:
        if subtree.connected(u, v):
            continue
        temporary = subtree.copy()
        death = float(birth)
        for (a, b), candidate_death in max_edges:
            temporary.union(a, b)
            if temporary.connected(u, v):
                death = float(candidate_death)
                break
        intervals.append((float(birth), death))
        subtree.union(u, v)
    return intervals


def rtd_srtd_lite_metrics(distance_matrix_1: np.ndarray, distance_matrix_2: np.ndarray, *, q: float = 0.90) -> dict[str, float]:
    matrix_1, matrix_2 = validate_pair(distance_matrix_1, distance_matrix_2)
    normalized_1 = normalize_by_quantile(matrix_1, q=q)
    normalized_2 = normalize_by_quantile(matrix_2, q=q)
    matrix_min = np.minimum(normalized_1, normalized_2)
    matrix_max = np.maximum(normalized_1, normalized_2)

    mst_1 = mst(normalized_1).total_weight
    mst_2 = mst(normalized_2).total_weight
    mst_min = mst(matrix_min).total_weight
    mst_max = mst(matrix_max).total_weight

    rtd_12 = mst_1 - mst_min
    rtd_21 = mst_2 - mst_min
    return {
        "RTD_lite": float(max(0.0, 0.5 * (rtd_12 + rtd_21))),
        "RTD_lite_dir_12": float(max(0.0, rtd_12)),
        "RTD_lite_dir_21": float(max(0.0, rtd_21)),
        "SRTD_lite": float(max(0.0, mst_max - mst_min)),
    }


def symmetric_auxiliary_matrix(
    distance_matrix_1: np.ndarray,
    distance_matrix_2: np.ndarray,
    *,
    q: float = 0.90,
) -> np.ndarray:
    matrix_1, matrix_2 = validate_pair(distance_matrix_1, distance_matrix_2)
    matrix_1 = normalize_by_quantile(matrix_1, q=q)
    matrix_2 = normalize_by_quantile(matrix_2, q=q)

    n_vertices = matrix_1.shape[0]
    matrix_min = np.minimum(matrix_1, matrix_2)
    matrix_max = np.maximum(matrix_1, matrix_2)

    max_plus = matrix_max.copy()
    max_plus[np.triu_indices(n_vertices, k=1)] = np.inf

    zeros_col = np.zeros((n_vertices, 1), dtype=float)
    inf_col = np.full((n_vertices, 1), np.inf, dtype=float)
    zero_row = np.zeros((1, n_vertices), dtype=float)
    inf_row = np.full((1, n_vertices), np.inf, dtype=float)

    row_1 = np.concatenate([matrix_max, max_plus.T, zeros_col], axis=1)
    row_2 = np.concatenate([max_plus, matrix_min, inf_col], axis=1)
    row_3 = np.concatenate([zero_row, inf_row, np.zeros((1, 1), dtype=float)], axis=1)
    return np.concatenate([row_1, row_2, row_3], axis=0)


def srtd_score(
    distance_matrix_1: np.ndarray,
    distance_matrix_2: np.ndarray,
    *,
    max_dim: int = 1,
    q: float = 0.90,
) -> dict[int, float]:
    try:
        import ripserplusplus as rpp_py
    except ImportError as exc:
        raise ImportError("srtd_score requires ripserplusplus to be installed.") from exc

    auxiliary = symmetric_auxiliary_matrix(distance_matrix_1, distance_matrix_2, q=q)
    auxiliary = (auxiliary + auxiliary.T) / 2.0
    np.fill_diagonal(auxiliary, 0.0)
    result = rpp_py.run(f"--format distance --dim {max_dim}", auxiliary)

    scores: dict[int, float] = {}
    for dim in range(max_dim + 1):
        diagram = np.asarray(result["dgms"][dim], dtype=float)
        if diagram.size == 0:
            scores[dim] = 0.0
            continue
        finite = np.isfinite(diagram[:, 0]) & np.isfinite(diagram[:, 1])
        scores[dim] = float(np.sum(diagram[finite, 1] - diagram[finite, 0]))
    return scores

