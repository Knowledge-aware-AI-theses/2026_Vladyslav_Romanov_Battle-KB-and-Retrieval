from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


RESULTS_DIR = Path("results/retrieval")


def load_metrics(file_path: Path) -> dict:
    if not file_path.exists():
        raise FileNotFoundError(
            f"Retrieval metrics file not found: {file_path.resolve()}"
        )

    with file_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def metrics_file(kb: int, slice_name: str = "all") -> Path:
    if kb not in {1, 2}:
        raise ValueError("kb must be 1 or 2.")

    suffix = f"kb{kb}" if slice_name == "all" else f"kb{kb}_{slice_name}"

    return (
        RESULTS_DIR
        / f"retrieval_experiment_average_metrics_{suffix}.json"
    )


def plot_kb_comparison(
    metric_name: str,
    kb1_file: Path | None = None,
    kb2_file: Path | None = None,
) -> None:
    """Compare BM25, dense, and hybrid performance for KB1 and KB2."""

    kb1_file = kb1_file or metrics_file(1)
    kb2_file = kb2_file or metrics_file(2)

    kb1_results = load_metrics(kb1_file)
    kb2_results = load_metrics(kb2_file)

    labels = ["KB1", "KB2"]
    methods = ["bm25", "dense", "hybrid"]

    x = np.arange(len(labels))
    width = 0.24

    fig, ax = plt.subplots(figsize=(8, 5))

    for index, method in enumerate(methods):
        values = [
            kb1_results[method][metric_name],
            kb2_results[method][metric_name],
        ]

        ax.bar(
            x + (index - 1) * width,
            values,
            width,
            label=method.upper() if method == "bm25" else method.capitalize(),
        )

    ax.set_ylabel(metric_name)
    ax.set_xlabel("Knowledge base")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)

    fig.tight_layout()
    plt.show()


def plot_metric_by_rarity(
    metric_name: str,
    kb: int = 2,
) -> None:
    """Compare retrieval performance for common and rare queries."""

    common_results = load_metrics(metrics_file(kb, "common"))
    rare_results = load_metrics(metrics_file(kb, "rare"))

    labels = ["Common", "Rare"]
    methods = ["bm25", "dense", "hybrid"]
    results = [common_results, rare_results]

    x = np.arange(len(labels))
    width = 0.24

    fig, ax = plt.subplots(figsize=(8, 5))

    for index, method in enumerate(methods):
        values = [
            result[method][metric_name]
            for result in results
        ]

        ax.bar(
            x + (index - 1) * width,
            values,
            width,
            label=method.upper() if method == "bm25" else method.capitalize(),
        )

    ax.set_ylabel(metric_name)
    ax.set_xlabel("Context rarity")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)

    fig.tight_layout()
    plt.show()


def plot_metric_by_difficulty(
    metric_name: str,
    kb: int = 2,
) -> None:
    """Compare retrieval performance across easy, medium, and hard queries."""

    slice_names = ["easy", "medium", "hard"]
    labels = ["Easy", "Medium", "Hard"]
    methods = ["bm25", "dense", "hybrid"]

    results = [
        load_metrics(metrics_file(kb, slice_name))
        for slice_name in slice_names
    ]

    x = np.arange(len(labels))
    width = 0.22

    fig, ax = plt.subplots(figsize=(8, 5))

    for index, method in enumerate(methods):
        values = [
            result[method][metric_name]
            for result in results
        ]

        ax.bar(
            x + (index - 1) * width,
            values,
            width,
            label=method.upper() if method == "bm25" else method.capitalize(),
        )

    ax.set_ylabel(metric_name)
    ax.set_xlabel("Query difficulty")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)

    fig.tight_layout()
    plt.show()


def plot_metric_by_query_type(
    metric_name: str,
    kb: int = 2,
) -> None:
    """Compare retrieval performance across all query-type slices."""

    query_types = [
        ("participant", "Participant-related"),
        ("similarity", "Similarity"),
        ("narrative", "Narrative / Tactical"),
        ("temporal", "Temporal"),
        ("spatial", "Spatial / Terrain"),
        ("outcome", "Outcome-related"),
        ("weapon", "Weapon / Unit-related"),
    ]

    methods = ["bm25", "dense", "hybrid"]
    labels = [label for _, label in query_types]
    results = [
        load_metrics(metrics_file(kb, slice_name))
        for slice_name, _ in query_types
    ]

    x = np.arange(len(labels))
    width = 0.24

    fig, ax = plt.subplots(figsize=(13, 6))

    for index, method in enumerate(methods):
        values = [
            result[method][metric_name]
            for result in results
        ]

        ax.bar(
            x + (index - 1) * width,
            values,
            width,
            label=method.upper() if method == "bm25" else method.capitalize(),
        )

    ax.set_ylabel(metric_name)
    ax.set_xlabel("Query type")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)

    fig.tight_layout()
    plt.show()


def plot_similarity(
    metric_name: str,
    kb: int = 2,
) -> None:
    """Plot the similarity-query slice for BM25, dense, and hybrid."""

    results = load_metrics(metrics_file(kb, "similarity"))

    methods = ["bm25", "dense", "hybrid"]
    labels = ["BM25", "Dense", "Hybrid"]
    values = [
        results[method][metric_name]
        for method in methods
    ]

    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(x, values, width=0.6)

    ax.set_ylabel(metric_name)
    ax.set_xlabel("Retrieval method")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)

    fig.tight_layout()
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot retrieval experiment results."
    )

    parser.add_argument(
        "--plot",
        choices=[
            "kb",
            "rarity",
            "difficulty",
            "query-type",
            "similarity",
        ],
        required=True,
        help="Diagram to generate.",
    )

    parser.add_argument(
        "--metric",
        default="Recall@10",
        help="Metric name stored in the retrieval result files.",
    )

    parser.add_argument(
        "--kb",
        type=int,
        choices=[1, 2],
        default=2,
        help="Knowledge base version for slice-based diagrams.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.plot == "kb":
        plot_kb_comparison(args.metric)
    elif args.plot == "rarity":
        plot_metric_by_rarity(args.metric, kb=args.kb)
    elif args.plot == "difficulty":
        plot_metric_by_difficulty(args.metric, kb=args.kb)
    elif args.plot == "query-type":
        plot_metric_by_query_type(args.metric, kb=args.kb)
    elif args.plot == "similarity":
        plot_similarity(args.metric, kb=args.kb)


if __name__ == "__main__":
    main()
