"""Phase 4: build one fold's images, run shape / NaN checks and plot Buy / Hold / Sell examples.

Output: results/<run_id>/figures/sample_images_fold<k>.png
"""
import argparse

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

from vitxai.config import add_common_args, load_config, run_dir
from vitxai.data import load_universe
from vitxai.features.indicators import indicator_categories
from vitxai.images.dataset import build_fold_datasets, labels_path, load_indicator_store
from vitxai.labels.folds import folds_from_cfg
from vitxai.viz.plots import plot_sample_images


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_common_args(ap)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--per-class", type=int, default=2)
    args = ap.parse_args()
    cfg = load_config(args.set, args.config_dir)

    fold = folds_from_cfg(cfg)[args.fold]
    store = load_indicator_store(cfg, load_universe(cfg))
    labels = pd.read_parquet(labels_path(cfg))
    datasets, stats = build_fold_datasets(cfg, fold, store, labels)

    expected = (1, cfg["model"]["img_size"][0], cfg["image"]["window"])
    for split, ds in datasets.items():
        x, y = ds[0]
        assert tuple(x.shape) == expected, f"{split}: shape {tuple(x.shape)} != {expected}"
        assert np.isfinite(x.numpy()).all()
        print(f"  {split:5s} n={len(ds):6d}  image {tuple(x.shape)}  "
              f"dates {ds.samples['date'].min().date()} .. {ds.samples['date'].max().date()}")
    print(f"  norm stats fitted on {stats['fit_start']} .. {stats['fit_end']} ({int(stats['n_rows'])} rows)")

    ds = datasets["train"]
    rng = np.random.default_rng(cfg["seed"])
    images, titles = [], []
    for c, name in enumerate(cfg["labels"]["names"]):
        idx = np.flatnonzero(ds.samples["label"].to_numpy() == c)
        for i in rng.choice(idx, size=min(args.per_class, len(idx)), replace=False):
            s = ds.samples.iloc[i]
            images.append(ds.image(i)[0])
            titles.append(f"{name}: {s['ticker']} {s['date'].date()} (r={s['ret_next']:+.3f})")
    out = run_dir(cfg) / "figures" / f"sample_images_fold{args.fold}.png"
    plot_sample_images(images, titles, indicator_categories(cfg), out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
