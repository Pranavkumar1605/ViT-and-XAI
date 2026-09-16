"""Phase 5: train the model on each walk-forward fold (and each seed in train.seeds).

A fold/seed with an existing predictions.parquet is skipped (use --force to retrain), so the
script can be re-run after a Colab disconnect.

Output per fold/seed: results/<run_id>/fold_XX/seed_S/{checkpoint.pt, history.csv, predictions.parquet, summary.json}
"""
import argparse

import _bootstrap  # noqa: F401
import pandas as pd

from vitxai.config import add_common_args, load_config, run_dir, save_config
from vitxai.data import load_universe
from vitxai.images.dataset import build_fold_datasets, labels_path, load_indicator_store
from vitxai.images.normalize import save_norm_stats
from vitxai.labels.folds import folds_from_cfg
from vitxai.train.train import fold_dir, get_device, train_fold


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_common_args(ap)
    ap.add_argument("--fold", type=int, nargs="*", help="fold ids (default: all)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.set, args.config_dir)

    rdir = run_dir(cfg)
    save_config(cfg, rdir / "config.yaml")
    folds = folds_from_cfg(cfg)
    if args.fold:
        folds = [f for f in folds if f["fold"] in args.fold]

    print(f"device: {get_device()}")
    store = load_indicator_store(cfg, load_universe(cfg))
    labels = pd.read_parquet(labels_path(cfg))

    for fold in folds:
        todo = [s for s in cfg["train"]["seeds"]
                if args.force or not (fold_dir(rdir, fold["fold"], s) / "predictions.parquet").exists()]
        if not todo:
            print(f"fold {fold['fold']} ({fold['test_year']}): done, skipping")
            continue
        datasets, stats = build_fold_datasets(cfg, fold, store, labels)
        save_norm_stats(stats, rdir / f"fold_{fold['fold']:02d}" / "norm_stats.npz")
        print(f"fold {fold['fold']} test {fold['test_year']}: "
              + ", ".join(f"{k}={len(v)}" for k, v in datasets.items()))
        for seed in todo:
            summary = train_fold(cfg, fold, datasets, fold_dir(rdir, fold["fold"], seed), seed)
            print(f"  seed {seed}: best_epoch={summary['best_epoch']} test_acc={summary['test_acc']:.4f}")


if __name__ == "__main__":
    main()
