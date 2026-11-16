from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator


RESULTS_DIR = "results/e2e_rag"

TOP_K_RESULTS_DIR = os.path.join(RESULTS_DIR, "top_k")

top_k_results_baseline_file = os.path.join(TOP_K_RESULTS_DIR, "baseline/average_metrics.json")
top_5_results_json_file = os.path.join(TOP_K_RESULTS_DIR, "json/top_5/average_metrics.json")
top_10_results_json_file = os.path.join(TOP_K_RESULTS_DIR, "json/top_10/average_metrics.json")
top_15_results_json_file = os.path.join(TOP_K_RESULTS_DIR, "json/top_15/average_metrics.json")
top_20_results_json_file = os.path.join(TOP_K_RESULTS_DIR, "json/top_20/average_metrics.json")
top_25_results_json_file = os.path.join(TOP_K_RESULTS_DIR, "json/top_25/average_metrics.json")
top_30_results_json_file = os.path.join(TOP_K_RESULTS_DIR, "json/top_30/average_metrics.json")
top_35_results_json_file = os.path.join(TOP_K_RESULTS_DIR, "json/top_35/average_metrics.json")
top_40_results_json_file = os.path.join(TOP_K_RESULTS_DIR, "json/top_40/average_metrics.json")
top_5_results_rawtext_file = os.path.join(TOP_K_RESULTS_DIR, "raw_text/top_5/average_metrics.json")
top_10_results_rawtext_file = os.path.join(TOP_K_RESULTS_DIR, "raw_text/top_10/average_metrics.json")
top_20_results_rawtext_file = os.path.join(TOP_K_RESULTS_DIR, "raw_text/top_20/average_metrics.json")

RARITY_RESULTS_DIR = os.path.join(RESULTS_DIR, "rarity")

rarity_results_baseline_common_file = os.path.join(RARITY_RESULTS_DIR, "baseline/common/average_metrics.json")
rarity_results_baseline_rare_file = os.path.join(RARITY_RESULTS_DIR, "baseline/rare/average_metrics.json")
rarity_results_json_common_file = os.path.join(RARITY_RESULTS_DIR, "json/common/average_metrics.json")
rarity_results_json_rare_file = os.path.join(RARITY_RESULTS_DIR, "json/rare/average_metrics.json")
rarity_results_rawtext_common_file = os.path.join(RARITY_RESULTS_DIR, "raw_text/common/average_metrics.json")
rarity_results_rawtext_rare_file = os.path.join(RARITY_RESULTS_DIR, "raw_text/rare/average_metrics.json")

QUERY_TYPE_RESULTS_DIR = os.path.join(RESULTS_DIR, "query_type")

querytype_results_baseline_file = os.path.join(QUERY_TYPE_RESULTS_DIR, "baseline/average_metrics.json")
querytype_results_json_file = os.path.join(QUERY_TYPE_RESULTS_DIR, "json/top_20/average_metrics.json")
querytype_results_rawtext_file = os.path.join(QUERY_TYPE_RESULTS_DIR, "raw_text/top_20/average_metrics.json")

top_k_plot_source = {
    "baseline": top_k_results_baseline_file,
    "raw_text": {
        5: top_5_results_rawtext_file,
        10: top_10_results_rawtext_file,
        20: top_20_results_rawtext_file,
    },
    "json": {
        5: top_5_results_json_file,
        10: top_10_results_json_file,
        20: top_20_results_json_file,
        30: top_30_results_json_file,
    },
}

rarity_plot_source = {
    "baseline": {
        "common": rarity_results_baseline_common_file,
        "rare": rarity_results_baseline_rare_file,
    },
    "raw_text": {
        "common": rarity_results_rawtext_common_file,
        "rare": rarity_results_rawtext_rare_file,
    },
    "json": {
        "common": rarity_results_json_common_file,
        "rare": rarity_results_json_rare_file,
    },
}

query_type_plot_source = {
    "baseline": querytype_results_baseline_file,
    "raw_text": querytype_results_rawtext_file,
    "json": querytype_results_json_file,
}

json_error_count_source = {
    5: top_5_results_json_file,
    10: top_10_results_json_file,
    15: top_15_results_json_file,
    20: top_20_results_json_file,
    25: top_25_results_json_file,
    30: top_30_results_json_file,
    35: top_35_results_json_file,
    40: top_40_results_json_file,
}


### Plot functions

def plot_metric_top_k(
    metric: str,
    source: dict,
) -> None:
    """
    Plot one end-to-end RAG metric for baseline, raw-text, and JSON.

    Expected source structure:
        {
            "baseline": path,
            "raw_text": {
                5: path,
                10: path,
                20: path,
            },
            "json": {
                5: path,
                10: path,
                20: path,
                30: path,
            },
        }
    """
    supported_metrics = {
        "precision",
        "recall",
        "f1",
        "exact_match",
    }

    if metric not in supported_metrics:
        raise ValueError(
            f"Unsupported metric {metric!r}. "
            f"Choose one of: {sorted(supported_metrics)}"
        )

    def load_metric(file_path) -> float:
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(
                f"Metrics file not found: {file_path}"
            )

        with file_path.open("r", encoding="utf-8") as file:
            report = json.load(file)

        final_results = report.get("Final_results")

        if not isinstance(final_results, dict):
            raise ValueError(
                f"Missing 'Final_results' in: {file_path}"
            )

        if metric not in final_results:
            raise ValueError(
                f"Metric {metric!r} is missing in: {file_path}"
            )

        return float(final_results[metric])

    configurations = [
        (
            "Baseline",
            source["baseline"],
            "#2E8B57",
        ),
        (
            "Raw text\nTop-5",
            source["raw_text"][5],
            "#9AC9F5",
        ),
        (
            "Raw text\nTop-10",
            source["raw_text"][10],
            "#4A90E2",
        ),
        (
            "Raw text\nTop-20",
            source["raw_text"][20],
            "#145DA0",
        ),
        (
            "JSON\nTop-5",
            source["json"][5],
            "#FFD180",
        ),
        (
            "JSON\nTop-10",
            source["json"][10],
            "#FFAB40",
        ),
        (
            "JSON\nTop-20",
            source["json"][20],
            "#F57C00",
        ),
        (
            "JSON\nTop-30",
            source["json"][30],
            "#C44700",
        ),
    ]

    labels = []
    values = []
    colors = []

    for label, file_path, color in configurations:
        labels.append(label)
        values.append(load_metric(file_path))
        colors.append(color)

    fig, ax = plt.subplots(figsize=(12, 6))

    bars = ax.bar(
        labels,
        values,
        color=colors,
        width=0.7,
        edgecolor="black",
        linewidth=0.4,
    )

    ax.bar_label(
        bars,
        fmt="%.3f",
        padding=3,
        fontsize=9,
    )

    metric_display_name = metric.replace("_", " ").title()

    ax.set_ylabel(metric_display_name)
    ax.set_xlabel("Experiment configuration")
    ax.set_title(
        f"End-to-end RAG performance: {metric_display_name}"
    )

    ax.set_ylim(0, 1.05)
    ax.grid(
        axis="y",
        linestyle="--",
        alpha=0.35,
    )
    ax.set_axisbelow(True)

    fig.tight_layout()
    plt.show()


def plot_metric_rarity(
    metric: str,
    source: dict,
) -> None:
    """
    Plot an end-to-end RAG metric grouped by query rarity.

    The baseline source can be either:

    1. One overall baseline file:
        "baseline": Path(...)

    2. Separate rarity-specific baseline files:
        "baseline": {
            "common": Path(...),
            "rare": Path(...),
        }

    Raw-text and JSON sources must contain separate files for each rarity.
    """
    supported_metrics = {
        "precision",
        "recall",
        "f1",
        "exact_match",
    }

    if metric not in supported_metrics:
        raise ValueError(
            f"Unsupported metric {metric!r}. "
            f"Choose one of: {sorted(supported_metrics)}"
        )

    rarities = ["common", "rare"]

    def load_metric(file_path) -> float:
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(
                f"Metrics file not found: {file_path}"
            )

        with file_path.open("r", encoding="utf-8") as file:
            report = json.load(file)

        final_results = report.get("Final_results")

        if not isinstance(final_results, dict):
            raise ValueError(
                f"Missing 'Final_results' in: {file_path}"
            )

        if metric not in final_results:
            raise ValueError(
                f"Metric {metric!r} is missing in: {file_path}"
            )

        return float(final_results[metric])

    baseline_source = source.get("baseline")
    raw_text_source = source.get("raw_text")
    json_source = source.get("json")

    if baseline_source is None:
        raise ValueError("source must contain 'baseline'.")

    if not isinstance(raw_text_source, dict):
        raise ValueError(
            "source['raw_text'] must contain common and rare files."
        )

    if not isinstance(json_source, dict):
        raise ValueError(
            "source['json'] must contain common and rare files."
        )

    baseline_values = []

    for rarity in rarities:
        if isinstance(baseline_source, dict):
            if rarity not in baseline_source:
                raise ValueError(
                    f"Missing baseline file for {rarity!r} queries."
                )

            baseline_file = baseline_source[rarity]
        else:
            # The same overall baseline is displayed for both groups.
            baseline_file = baseline_source

        baseline_values.append(load_metric(baseline_file))

    raw_text_values = []
    json_values = []

    for rarity in rarities:
        if rarity not in raw_text_source:
            raise ValueError(
                f"Missing raw-text file for {rarity!r} queries."
            )

        if rarity not in json_source:
            raise ValueError(
                f"Missing JSON file for {rarity!r} queries."
            )

        raw_text_values.append(
            load_metric(raw_text_source[rarity])
        )
        json_values.append(
            load_metric(json_source[rarity])
        )

    x = np.arange(len(rarities))
    width = 0.24

    fig, ax = plt.subplots(figsize=(9, 6))

    baseline_bars = ax.bar(
        x - width,
        baseline_values,
        width,
        label="Baseline",
        color="#2E8B57",
        edgecolor="black",
        linewidth=0.4,
    )

    raw_text_bars = ax.bar(
        x,
        raw_text_values,
        width,
        label="Raw text",
        color="#4A90E2",
        edgecolor="black",
        linewidth=0.4,
    )

    json_bars = ax.bar(
        x + width,
        json_values,
        width,
        label="JSON",
        color="#F57C00",
        edgecolor="black",
        linewidth=0.4,
    )

    for bars in [
        baseline_bars,
        raw_text_bars,
        json_bars,
    ]:
        ax.bar_label(
            bars,
            fmt="%.3f",
            padding=3,
            fontsize=9,
        )

    metric_display_name = metric.replace("_", " ").title()

    ax.set_ylabel(metric_display_name)
    ax.set_xlabel("Context rarity")
    ax.set_title(
        f"End-to-end RAG performance by context rarity: "
        f"{metric_display_name}"
    )

    ax.set_xticks(x)
    ax.set_xticklabels(["Common", "Rare"])
    ax.set_ylim(0, 1.05)

    ax.grid(
        axis="y",
        linestyle="--",
        alpha=0.35,
    )
    ax.set_axisbelow(True)
    ax.legend()

    fig.tight_layout()
    plt.show()


def plot_metric_query_type(
    metric: str,
    source: dict,
    query_type: str,
) -> None:
    """
    Plot one metric for baseline, raw-text, and JSON experiments.

    The source files must already contain the average metrics for the
    selected query type. The query_type parameter is used as a chart label.

    Expected source structure:
        {
            "baseline": path,
            "raw_text": path,
            "json": path,
        }
    """
    supported_metrics = {
        "precision",
        "recall",
        "f1",
        "exact_match",
    }

    if metric not in supported_metrics:
        raise ValueError(
            f"Unsupported metric {metric!r}. "
            f"Choose one of: {sorted(supported_metrics)}"
        )

    def load_metric(file_path) -> float:
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(
                f"Metrics file not found: {file_path}"
            )

        with file_path.open("r", encoding="utf-8") as file:
            report = json.load(file)

        final_results = report.get("Final_results")

        if not isinstance(final_results, dict):
            raise ValueError(
                f"Missing 'Final_results' in: {file_path}"
            )

        if metric not in final_results:
            raise ValueError(
                f"Metric {metric!r} is missing in: {file_path}"
            )

        return float(final_results[metric])

    configurations = [
        (
            "Baseline",
            source["baseline"],
            "#2E8B57",
        ),
        (
            "Raw text",
            source["raw_text"],
            "#4A90E2",
        ),
        (
            "JSON",
            source["json"],
            "#F57C00",
        ),
    ]

    labels = []
    values = []
    colors = []

    for label, file_path, color in configurations:
        labels.append(label)
        values.append(load_metric(file_path))
        colors.append(color)

    fig, ax = plt.subplots(figsize=(8, 5.5))

    bars = ax.bar(
        labels,
        values,
        color=colors,
        width=0.6,
        edgecolor="black",
        linewidth=0.4,
    )

    ax.bar_label(
        bars,
        fmt="%.3f",
        padding=3,
        fontsize=10,
    )

    metric_display_name = metric.replace("_", " ").title()

    ax.set_ylabel(metric_display_name)
    ax.set_xlabel("Experiment configuration")
    ax.set_title(
        f"End-to-end RAG performance: {metric_display_name}\n"
        f"Query type: {query_type}"
    )

    ax.set_ylim(0, 1.05)
    ax.grid(
        axis="y",
        linestyle="--",
        alpha=0.35,
    )
    ax.set_axisbelow(True)

    fig.tight_layout()
    plt.show()


def plot_json_error_count(source: dict) -> None:
    """
    Plot the relationship between JSON context size and run-error count.

    Expected source structure:
        {
            5: path,
            10: path,
            15: path,
            ...
        }

    Each file must contain:
        {
            "Final_results": {
                "run_error_count": ...
            }
        }
    """
    if not source:
        raise ValueError("source must contain at least one top-k file.")

    def load_error_count(file_path) -> int:
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(
                f"Metrics file not found: {file_path}"
            )

        with file_path.open("r", encoding="utf-8") as file:
            report = json.load(file)

        final_results = report.get("Final_results")

        if not isinstance(final_results, dict):
            raise ValueError(
                f"Missing 'Final_results' in: {file_path}"
            )

        if "run_error_count" not in final_results:
            raise ValueError(
                f"Missing 'run_error_count' in: {file_path}"
            )

        return int(final_results["run_error_count"])

    top_k_values = sorted(source.keys())

    if any(
        not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0
        for top_k in top_k_values
    ):
        raise ValueError(
            "All source keys must be positive integer top-k values."
        )

    error_counts = [
        load_error_count(source[top_k])
        for top_k in top_k_values
    ]

    fig, ax = plt.subplots(figsize=(9, 5.5))

    ax.plot(
        top_k_values,
        error_counts,
        color="#C44700",
        marker="o",
        markersize=7,
        linewidth=2.2,
        label="JSON run errors",
    )

    for top_k, error_count in zip(
        top_k_values,
        error_counts,
    ):
        ax.annotate(
            str(error_count),
            xy=(top_k, error_count),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=9,
        )

    ax.set_xlabel("Number of retrieved battles (top-k)")
    ax.set_ylabel("Run error count")
    ax.set_title(
        "JSON context size and generation error count"
    )

    ax.set_xticks(top_k_values)
    ax.set_ylim(
        bottom=0,
        top=max(error_counts) + max(1, int(max(error_counts) * 0.15)),
    )

    # Error counts should be displayed as whole numbers.
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    ax.grid(
        axis="both",
        linestyle="--",
        alpha=0.35,
    )
    ax.set_axisbelow(True)
    ax.legend()

    fig.tight_layout()
    plt.show()

###


if __name__ == "__main__":
    # Examples:
    # plot_metric_top_k("f1", top_k_plot_source)
    # plot_metric_rarity("f1", rarity_plot_source)
    # plot_metric_query_type(
    #     metric="f1",
    #     source=query_type_plot_source,
    #     query_type="Similarity",
    # )
    # plot_json_error_count(json_error_count_source)
    pass
