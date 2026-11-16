from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


RESULTS_DIR = Path("results/oracle_generation")
KB1_KB2_RESULTS_DIR = RESULTS_DIR / "kb1_vs_kb2"
COMMON_RARE_RESULTS_DIR = RESULTS_DIR / "common_vs_rare"
DIFFICULTY_RESULTS_DIR = RESULTS_DIR / "difficulty"
QUERY_TYPE_RESULTS_DIR = RESULTS_DIR / "query_type"


CONFIGURATIONS = [
    ("baseline", "Baseline", "green"),
    ("raw_text_augmented", "Raw text", "blue"),
    ("json_kb1", "JSON KB1", "orange"),
    ("json_kb2", "JSON KB2", "darkred"),
]


def load_json(file_path: Path) -> dict:
    if not file_path.exists():
        raise FileNotFoundError(
            f"Result file not found: {file_path.resolve()}"
        )

    with file_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def plot_metric_kb1_kb2(
    metric_name: str,
    file: Path = (
        KB1_KB2_RESULTS_DIR
        / "final_reports"
        / "comparison_final_report.json"
    ),
) -> None:
    """
    Plot one oracle-generation metric for:
    baseline, raw text, JSON KB1, and JSON KB2.
    """

    report = load_json(Path(file))
    report_configurations = report.get("configurations")

    if not isinstance(report_configurations, dict):
        raise ValueError(
            f"Missing 'configurations' object in report: {file}"
        )

    labels = []
    values = []
    colors = []

    for configuration_key, display_name, color in CONFIGURATIONS:
        if configuration_key not in report_configurations:
            raise ValueError(
                f"Missing configuration {configuration_key!r} "
                f"in report: {file}"
            )

        configuration_results = report_configurations[configuration_key]

        if "Final_results" in configuration_results:
            configuration_results = configuration_results["Final_results"]

        if metric_name not in configuration_results:
            raise ValueError(
                f"Metric {metric_name!r} is missing for "
                f"{configuration_key!r} in report: {file}"
            )

        labels.append(display_name)
        values.append(float(configuration_results[metric_name]))
        colors.append(color)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    bars = ax.bar(labels, values, color=colors, width=0.65)

    ax.bar_label(
        bars,
        fmt="%.3f",
        padding=3,
        fontsize=9,
    )

    ax.set_ylabel(metric_name)
    ax.set_title(f"Oracle-context generation: {metric_name}")

    if metric_name.startswith("exact"):
        ax.set_ylim(0.3, 0.9)
    else:
        ax.set_ylim(0.4, 1.05)

    ax.grid(axis="y", linestyle="--", alpha=0.35)

    fig.tight_layout()
    plt.show()


def plot_metric_common_rare(
    metric_name: str = "f1",
    common_file: Path = (
        COMMON_RARE_RESULTS_DIR
        / "oracle_generation_average_metrics_common.json"
    ),
    rare_file: Path = (
        COMMON_RARE_RESULTS_DIR
        / "oracle_generation_average_metrics_rare.json"
    ),
) -> None:
    """
    Plot one metric for common and rare oracle-generation queries.
    """

    common_results = load_json(Path(common_file))
    rare_results = load_json(Path(rare_file))

    labels = ["Common", "Rare"]
    reports = [common_results, rare_results]
    report_files = [Path(common_file), Path(rare_file)]

    x = np.arange(len(labels))
    width = 0.15

    fig, ax = plt.subplots(figsize=(9, 5.5))

    for i, (configuration_key, display_name, color) in enumerate(CONFIGURATIONS):
        values = []

        for report, report_file in zip(reports, report_files):
            report_configurations = report.get("configurations")

            if not isinstance(report_configurations, dict):
                raise ValueError(
                    f"Missing 'configurations' object in report: "
                    f"{report_file}"
                )

            if configuration_key not in report_configurations:
                raise ValueError(
                    f"Missing configuration {configuration_key!r} "
                    f"in report: {report_file}"
                )

            configuration_results = report_configurations[configuration_key]

            if metric_name not in configuration_results:
                raise ValueError(
                    f"Metric {metric_name!r} is missing for "
                    f"{configuration_key!r} in report: {report_file}"
                )

            values.append(float(configuration_results[metric_name]))

        offset = (i - (len(CONFIGURATIONS) - 1) / 2) * width

        bars = ax.bar(
            x + offset,
            values,
            width,
            label=display_name,
            color=color,
        )

        ax.bar_label(
            bars,
            fmt="%.3f",
            padding=3,
            fontsize=8,
        )

    ax.set_ylabel(metric_name)
    ax.set_title(f"Common vs rare queries: {metric_name}")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)

    if metric_name.startswith("exact"):
        ax.set_ylim(0.2, 0.8)
    else:
        ax.set_ylim(0.4, 1.05)

    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend()

    fig.tight_layout()
    plt.show()


def plot_metric_difficulty(
    metric_name: str = "f1",
    easy_file: Path = (
        DIFFICULTY_RESULTS_DIR
        / "oracle_generation_average_metrics_easy.json"
    ),
    medium_file: Path = (
        DIFFICULTY_RESULTS_DIR
        / "oracle_generation_average_metrics_medium.json"
    ),
    hard_file: Path = (
        DIFFICULTY_RESULTS_DIR
        / "oracle_generation_average_metrics_hard.json"
    ),
) -> None:
    """
    Plot one metric for easy, medium, and hard queries.
    """

    report_files = [
        Path(easy_file),
        Path(medium_file),
        Path(hard_file),
    ]

    reports = [
        load_json(report_file)
        for report_file in report_files
    ]

    labels = ["Easy", "Medium", "Hard"]

    x = np.arange(len(labels))
    width = 0.18

    fig, ax = plt.subplots(figsize=(10, 5.5))

    for i, (configuration_key, display_name, color) in enumerate(CONFIGURATIONS):
        values = []

        for report, report_file in zip(reports, report_files):
            report_configurations = report.get("configurations")

            if not isinstance(report_configurations, dict):
                raise ValueError(
                    f"Missing 'configurations' object in report: "
                    f"{report_file}"
                )

            if configuration_key not in report_configurations:
                raise ValueError(
                    f"Missing configuration {configuration_key!r} "
                    f"in report: {report_file}"
                )

            configuration_results = report_configurations[configuration_key]

            if metric_name not in configuration_results:
                raise ValueError(
                    f"Metric {metric_name!r} is missing for "
                    f"{configuration_key!r} in report: {report_file}"
                )

            values.append(float(configuration_results[metric_name]))

        offset = (i - (len(CONFIGURATIONS) - 1) / 2) * width

        bars = ax.bar(
            x + offset,
            values,
            width,
            label=display_name,
            color=color,
        )

        ax.bar_label(
            bars,
            fmt="%.3f",
            padding=3,
            fontsize=8,
        )

    ax.set_ylabel(metric_name)
    ax.set_title(
        f"Performance by query difficulty: {metric_name}"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.3, 1.05)

    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend()

    fig.tight_layout()
    plt.show()


def query_type_slug(query_type: str) -> str:
    slug = (
        query_type
        .strip()
        .lower()
        .replace(" / ", "_")
        .replace("/", "_")
        .replace(" ", "_")
        .replace("-", "_")
    )

    while "__" in slug:
        slug = slug.replace("__", "_")

    return slug


def plot_metric_query_type(
    metric_name: str = "f1",
    labels: list[str] | None = None,
    query_type_results_dir: Path = QUERY_TYPE_RESULTS_DIR,
) -> None:
    """
    Plot one metric for configurable query types.
    """

    if labels is None:
        labels = [
            "Participant-related",
            "Similarity",
            "Narrative / Tactical",
            "Temporal",
            "Spatial / Terrain",
            "Outcome-related",
            "Weapon / Unit-related",
        ]

    labels = list(dict.fromkeys(labels))

    if not labels:
        raise ValueError(
            "At least one query-type label must be provided."
        )

    query_type_results_dir = Path(query_type_results_dir)

    reports = []
    report_files = []

    for query_type in labels:
        slug = query_type_slug(query_type)

        report_file = (
            query_type_results_dir
            / f"oracle_generation_average_metrics_{slug}.json"
        )

        reports.append(load_json(report_file))
        report_files.append(report_file)

    x = np.arange(len(labels))
    width = 0.18
    figure_width = max(8, len(labels) * 2.2)

    fig, ax = plt.subplots(
        figsize=(figure_width, 5.5)
    )

    for i, (configuration_key, display_name, color) in enumerate(CONFIGURATIONS):
        values = []

        for report, report_file in zip(reports, report_files):
            report_configurations = report.get("configurations")

            if not isinstance(report_configurations, dict):
                raise ValueError(
                    f"Missing 'configurations' object in report: "
                    f"{report_file}"
                )

            if configuration_key not in report_configurations:
                raise ValueError(
                    f"Missing configuration {configuration_key!r} "
                    f"in report: {report_file}"
                )

            configuration_results = report_configurations[configuration_key]

            if metric_name not in configuration_results:
                raise ValueError(
                    f"Metric {metric_name!r} is missing for "
                    f"{configuration_key!r} in report: {report_file}"
                )

            values.append(float(configuration_results[metric_name]))

        offset = (i - (len(CONFIGURATIONS) - 1) / 2) * width

        bars = ax.bar(
            x + offset,
            values,
            width,
            label=display_name,
            color=color,
        )

        ax.bar_label(
            bars,
            fmt="%.3f",
            padding=3,
            fontsize=8,
        )

    ax.set_ylabel(metric_name)
    ax.set_title(
        f"Performance by query type: {metric_name}"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(
        labels,
        rotation=15 if len(labels) > 3 else 0,
        ha="right" if len(labels) > 3 else "center",
    )

    if metric_name.startswith("exact"):
        ax.set_ylim(0.1, 1.05)
    else:
        ax.set_ylim(0.5, 1.05)

    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend()

    fig.tight_layout()
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot oracle-generation experiment results."
    )

    parser.add_argument(
        "--plot",
        choices=[
            "kb",
            "rarity",
            "difficulty",
            "query-type",
        ],
        required=True,
        help="Diagram to generate.",
    )

    parser.add_argument(
        "--metric",
        default="f1",
        choices=[
            "precision",
            "recall",
            "f1",
            "exact_match",
        ],
        help="Metric to plot.",
    )

    parser.add_argument(
        "--types",
        nargs="*",
        default=None,
        help=(
            "Optional query-type labels for --plot query-type. "
            "If omitted, all query types are used."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.plot == "kb":
        plot_metric_kb1_kb2(args.metric)
    elif args.plot == "rarity":
        plot_metric_common_rare(args.metric)
    elif args.plot == "difficulty":
        plot_metric_difficulty(args.metric)
    elif args.plot == "query-type":
        plot_metric_query_type(
            metric_name=args.metric,
            labels=args.types,
        )


if __name__ == "__main__":
    main()
