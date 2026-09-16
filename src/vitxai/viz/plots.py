from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def _category_bands(ax, categories: list[str]) -> None:
    """Horizontal lines between category blocks, labels on the right."""
    start = 0
    for i in range(1, len(categories) + 1):
        if i == len(categories) or categories[i] != categories[start]:
            if i < len(categories):
                ax.axhline(i - 0.5, color="white", lw=0.8)
            ax.text(1.01, 1 - (start + i) / 2 / len(categories), categories[start],
                    transform=ax.transAxes, va="center", fontsize=7)
            start = i


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_sample_images(images: list[np.ndarray], titles: list[str], categories: list[str], path: Path) -> None:
    n = len(images)
    cols = min(3, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 5 * rows), squeeze=False)
    for ax, img, title in zip(axes.flat, images, titles):
        im = ax.imshow(img, aspect="auto", cmap="coolwarm", vmin=-3, vmax=3)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("day (oldest -> newest)")
        ax.set_ylabel("indicator row")
        _category_bands(ax, categories)
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    fig.colorbar(im, ax=axes, shrink=0.6, label="standardized value")
    _save(fig, path)


def plot_confusion_matrix(cm: np.ndarray, class_names: list[str], title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(4, 3.6))
    ax.imshow(cm, cmap="Blues")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=9)
    ax.set_xticks(range(len(class_names)), class_names)
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title, fontsize=10)
    _save(fig, path)


def plot_class_heatmaps(mean_maps: dict[str, np.ndarray], names: list[str], categories: list[str],
                        title: str, path: Path) -> None:
    classes = list(mean_maps)
    fig, axes = plt.subplots(1, len(classes), figsize=(5.5 * len(classes), 16), squeeze=False)
    for ax, cls in zip(axes[0], classes):
        m = mean_maps[cls]
        im = ax.imshow(m, aspect="auto", cmap="viridis")
        ax.set_title(f"{cls}", fontsize=11)
        ax.set_yticks(range(len(names)), names, fontsize=5)
        w = m.shape[1]
        ticks = list(range(0, w, 13)) + [w - 1]
        ax.set_xticks(ticks, [f"t-{w - 1 - t}" if t != w - 1 else "t" for t in ticks], fontsize=7)
        ax.set_xlabel("day")
        _category_bands(ax, categories)
        fig.colorbar(im, ax=ax, shrink=0.3)
    fig.suptitle(title, fontsize=12)
    _save(fig, path)


def plot_top_indicators(rankings, class_names: list[str], k: int, title: str, path: Path) -> None:
    fig, axes = plt.subplots(1, len(class_names), figsize=(4.5 * len(class_names), 0.28 * k + 1.5), squeeze=False)
    for ax, cls in zip(axes[0], class_names):
        col = f"importance_{cls}"
        if col not in rankings:
            ax.axis("off")
            continue
        top = rankings.nlargest(k, col)[::-1]
        ax.barh(top["indicator"], top[col], color="#4C72B0")
        ax.set_title(cls, fontsize=10)
        ax.tick_params(axis="y", labelsize=7)
    fig.suptitle(title, fontsize=11)
    _save(fig, path)


def plot_recency(profiles: dict[str, np.ndarray], title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 3.5))
    for cls, prof in profiles.items():
        w = len(prof)
        ax.plot(np.arange(-(w - 1), 1), prof, label=cls)
    ax.set_xlabel("day relative to t")
    ax.set_ylabel("share of attribution")
    ax.set_title(title, fontsize=10)
    ax.legend()
    _save(fig, path)
