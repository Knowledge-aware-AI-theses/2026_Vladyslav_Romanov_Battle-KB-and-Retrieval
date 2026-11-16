import argparse
import matplotlib.pyplot as plt
import numpy as np

DATA = {
    "Precision": {"Common": [0.665, 0.527, 0.537], "Rare": [0.137, 0.413, 0.416]},
    "Recall": {"Common": [0.629, 0.605, 0.663], "Rare": [0.145, 0.561, 0.583]},
    "F1": {"Common": [0.632, 0.541, 0.571], "Rare": [0.140, 0.440, 0.452]},
    "Exact Match": {"Common": [0.448, 0.350, 0.390], "Rare": [0.129, 0.333, 0.312]},
}

METHODS = ("Baseline", "Raw text", "JSON")
COLORS = ("#2e8b57", "#4a90e2", "#ff8200")


def plot_metric(metric: str, font_size: int = 16) -> None:
    """Create a separate bar chart for one metric (Common vs Rare)."""
    if metric not in DATA:
        raise ValueError(f"Unknown metric: {metric}. Choose from {', '.join(DATA)}")

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(2)
    width = 0.24

    for i, (method, color) in enumerate(zip(METHODS, COLORS)):
        values = [DATA[metric][group][i] for group in ("Common", "Rare")]
        bars = ax.bar(x + (i - 1) * width, values, width, label=method, color=color)
        ax.bar_label(bars, labels=[f"{v:.3f}" for v in values], padding=4, fontsize=max(9, font_size - 2))

    ax.set_xticks(x)
    ax.set_xticklabels(("Common", "Rare"), fontsize=font_size)
    ax.set_ylabel(metric, fontsize=font_size)
    ax.set_xlabel("Context rarity", fontsize=font_size)
    ax.set_title(f"End-to-end RAG performance by context rarity: {metric}", fontsize=font_size + 1)
    ax.set_ylim(0, 1.0)
    ax.tick_params(axis="y", labelsize=font_size - 1)
    ax.legend(loc="upper right", fontsize=max(9, font_size - 2))
    ax.grid(axis="y", linestyle="--", alpha=0.2)
    ax.set_axisbelow(True)
    fig.tight_layout()


def main() -> None:
    parser = argparse.ArgumentParser(description="Show individual end-to-end RAG rarity diagrams.")
    parser.add_argument("--font-size", type=int, default=22, help="Base plot font size (default: 24)")
    args = parser.parse_args()

    plot_metric("Precision", font_size=args.font_size)
    plot_metric("Recall", font_size=args.font_size)
    plot_metric("F1", font_size=args.font_size)
    plot_metric("Exact Match", font_size=args.font_size)
    plt.show()


if __name__ == "__main__":
    main()