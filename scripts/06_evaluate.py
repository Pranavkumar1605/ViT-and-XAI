"""Phase 6: metrics from saved predictions only.

Outputs in results/<run_id>/evaluation/:
  metrics_per_fold.csv   one row per fold x seed
  metrics_pooled.csv     all test years pooled, per seed
  metrics_summary.csv    mean / std across folds (per seed)
  confusion_matrices/    per fold and pooled
"""
import argparse

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

from vitxai.config import add_common_args, load_config, run_dir
from vitxai.eval.metrics import classification_metrics
from vitxai.viz.plots import plot_confusion_matrix


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_common_args(ap)
    args = ap.parse_args()
    cfg = load_config(args.set, args.config_dir)
    names = cfg["labels"]["names"]
    rdir = run_dir(cfg)
    out = rdir / "evaluation"
    (out / "confusion_matrices").mkdir(parents=True, exist_ok=True)

    pred_files = sorted(rdir.glob("fold_*/seed_*/predictions.parquet"))
    if not pred_files:
        raise SystemExit(f"no predictions found under {rdir}")

    rows, pooled = [], {}
    for pf in pred_files:
        fold = int(pf.parent.parent.name.split("_")[1])
        seed = int(pf.parent.name.split("_")[1])
        p = pd.read_parquet(pf)
        test_year = int(pd.to_datetime(p["date"]).dt.year.mode()[0])
        m = classification_metrics(p["y_true"], p["y_pred"], names)
        cm = m.pop("confusion_matrix")
        rows.append({"fold": fold, "test_year": test_year, "seed": seed, **m})
        plot_confusion_matrix(cm, names, f"fold {fold} ({test_year}) seed {seed}",
                              out / "confusion_matrices" / f"cm_fold{fold:02d}_seed{seed}.png")
        pooled.setdefault(seed, []).append(p)

    per_fold = pd.DataFrame(rows).sort_values(["seed", "fold"])
    per_fold.to_csv(out / "metrics_per_fold.csv", index=False)

    pooled_rows = []
    for seed, frames in pooled.items():
        p = pd.concat(frames)
        m = classification_metrics(p["y_true"], p["y_pred"], names)
        cm = m.pop("confusion_matrix")
        pooled_rows.append({"seed": seed, **m})
        plot_confusion_matrix(cm, names, f"pooled seed {seed}", out / "confusion_matrices" / f"cm_pooled_seed{seed}.png")
        pd.DataFrame(cm, index=[f"true_{n}" for n in names], columns=[f"pred_{n}" for n in names]) \
            .to_csv(out / f"confusion_pooled_seed{seed}.csv")
    pd.DataFrame(pooled_rows).to_csv(out / "metrics_pooled.csv", index=False)

    metric_cols = [c for c in per_fold.columns if c not in ("fold", "test_year", "seed")
                   and not c.startswith(("support_", "predicted_")) and c != "n"]
    summary = per_fold.groupby("seed")[metric_cols].agg(["mean", "std"])
    summary.to_csv(out / "metrics_summary.csv")

    show = ["fold", "test_year", "seed", "n", "accuracy", "majority_class_accuracy", "macro_f1"] + \
           [f"recall_{n}" for n in names]
    print(per_fold[show].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\nPooled:")
    print(pd.DataFrame(pooled_rows)[["seed", "n", "accuracy", "majority_class_accuracy", "macro_f1"]
                                    + [f"f1_{n}" for n in names]].to_string(index=False))
    print(f"\nsaved to {out}")


if __name__ == "__main__":
    main()
