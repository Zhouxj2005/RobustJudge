from __future__ import annotations

import matplotlib.pyplot as plt

from .config import FIGURE_DIR


def configure_matplotlib() -> None:
    plt.rcParams["font.sans-serif"] = ["SimHei", "PingFang SC", "Microsoft YaHei", "Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False


def plot_figure(
    x,
    y,
    title,
    num=None,
    xlabel="Number of Sampling Iterations (n)",
    ylabel="Average Standard Error of the Mean(SEM)",
):
    plt.figure(figsize=(8, 5))
    plt.title(title, fontsize=14)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.7)

    judge_models = y.keys()
    for judge_model in judge_models:
        plt.plot(x, y[judge_model], marker="o", linestyle="-", linewidth=2, label=judge_model)
        for i, value in enumerate(y[judge_model]):
            plt.annotate(f"{value:.2f}", (x[i], y[judge_model][i]), textcoords="offset points", xytext=(0, 10), ha="center")

    for judge_model in judge_models:
        label = judge_model if num is None else judge_model + f"({num[judge_model].shape[0]})"
        plt.annotate(label, (x[0], y[judge_model][0]), textcoords="offset points", xytext=(10, 0), ha="left", fontsize=9)

    plt.legend(title="Judge Models", fontsize=10)
    plt.savefig(FIGURE_DIR / f"{title}.png", dpi=300, bbox_inches="tight")
    plt.show()
