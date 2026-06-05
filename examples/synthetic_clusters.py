from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.spatial.distance import pdist, squareform

from srtd_nts import linear_cka, nts_scores, rtd_srtd_lite_scores


DEFAULT_K_VALUES = (1, 2, 3, 4, 5, 6, 10, 12)


def make_cluster_base(
    *,
    n_groups: int = 60,
    points_per_group: int = 5,
    seed: int = 42,
) -> tuple[np.ndarray, list[np.ndarray]]:
    rng = np.random.RandomState(seed)
    groups = [
        rng.multivariate_normal(np.zeros(2), np.eye(2), points_per_group)
        for _ in range(n_groups)
    ]
    return np.concatenate(groups), groups


def split_clusters(groups: list[np.ndarray], k: int, *, radius: float = 10.0) -> np.ndarray:
    if k < 1:
        raise ValueError("k must be positive.")
    if k == 1:
        return np.concatenate(groups)

    shifted_groups = []
    for index, group in enumerate(groups):
        angle = 2.0 * np.pi * (index % k) / k
        offset = radius * np.array([np.cos(angle), np.sin(angle)])
        shifted_groups.append(group + offset)
    return np.concatenate(shifted_groups)


def distance_matrix(points: np.ndarray) -> np.ndarray:
    return squareform(pdist(points, metric="euclidean"))


def run_experiment(
    *,
    k_values: tuple[int, ...] = DEFAULT_K_VALUES,
    q: float = 0.90,
    seed: int = 42,
) -> list[dict[str, int | float]]:
    base_points, groups = make_cluster_base(seed=seed)
    base_distance = distance_matrix(base_points)

    rows: list[dict[str, int | float]] = []
    for k in k_values:
        shifted_points = split_clusters(groups, k)
        shifted_distance = distance_matrix(shifted_points)

        nts = nts_scores(base_distance, shifted_distance)
        lite = rtd_srtd_lite_scores(base_distance, shifted_distance, q=q)
        cka = linear_cka(base_points, shifted_points)

        rows.append(
            {
                "k": k,
                "NTS_E": nts["NTS_E"],
                "NTS_M": nts["NTS_M"],
                "RTD_lite": lite["RTD_lite"],
                "SRTD_lite": lite["SRTD_lite"],
                "SRTD_lite_half": 0.5 * lite["SRTD_lite"],
                "Max_RTD_lite": lite["Max_RTD_lite"],
                "CKA": cka,
            }
        )
    return rows


def write_csv(rows: list[dict[str, int | float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def print_table(rows: list[dict[str, int | float]]) -> None:
    headers = list(rows[0])
    widths = {
        header: max(len(header), *(len(format_value(row[header], header)) for row in rows))
        for header in headers
    }
    print("  ".join(header.rjust(widths[header]) for header in headers))
    print("  ".join("-" * widths[header] for header in headers))
    for row in rows:
        print("  ".join(format_value(row[header], header).rjust(widths[header]) for header in headers))


def format_value(value: int | float, header: str) -> str:
    if header == "k":
        return str(int(value))
    return f"{value:.6f}"


def plot_rows(rows: list[dict[str, int | float]], output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("Plotting requires matplotlib. Install it or omit --plot.") from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    k_values = [row["k"] for row in rows]

    figure_specs = [
        ("nts", ("NTS_E", "NTS_M"), "Similarity"),
        ("rtd_lite", ("RTD_lite", "SRTD_lite_half", "Max_RTD_lite"), "Divergence"),
        ("cka", ("CKA",), "Similarity"),
    ]
    for name, series_names, ylabel in figure_specs:
        fig, ax = plt.subplots(figsize=(7, 5))
        for series_name in series_names:
            ax.plot(k_values, [row[series_name] for row in rows], marker="o", label=series_name)
        ax.set_xlabel("Number of clusters")
        ax.set_ylabel(ylabel)
        ax.set_xticks(k_values)
        ax.legend()
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(output_dir / f"synthetic_clusters_{name}.png", dpi=200)
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the synthetic cluster representation experiment.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--q", type=float, default=0.90, help="Quantile normalization for lite scores.")
    parser.add_argument(
        "--k-values",
        type=int,
        nargs="+",
        default=list(DEFAULT_K_VALUES),
        help="Cluster counts to evaluate.",
    )
    parser.add_argument("--out", type=Path, default=Path("outputs/synthetic_clusters.csv"))
    parser.add_argument("--plot", action="store_true", help="Save matplotlib plots next to the CSV output.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = run_experiment(k_values=tuple(args.k_values), q=args.q, seed=args.seed)
    print_table(rows)
    write_csv(rows, args.out)
    print(f"\nSaved results to {args.out}")
    if args.plot:
        plot_rows(rows, args.out.parent)
        print(f"Saved plots to {args.out.parent}")


if __name__ == "__main__":
    main()
