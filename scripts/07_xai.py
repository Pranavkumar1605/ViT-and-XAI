"""Phase 7: explain the trained ViT with the methods in configs/xai.yaml.

Step 1 (per fold): sample up to `samples_per_class` test images per predicted class, run every
method, save raw maps -> results/<run_id>/xai/fold_XX/attributions_<method>.npz
Step 2 (all folds): normalize maps, average per predicted class (all / correct / incorrect),
rank indicators, recency profiles, method agreement, figures -> results/<run_id>/xai/summary/

Maps are computed with seed `--seed` (default: first seed in train.seeds).
"""
import argparse
import time

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd
import torch

from vitxai.config import add_common_args, load_config, run_dir
from vitxai.data import load_universe
from vitxai.features.indicators import indicator_categories, indicator_names
from vitxai.images.dataset import build_fold_datasets, labels_path, load_indicator_store
from vitxai.labels.folds import folds_from_cfg
from vitxai.train.train import fold_dir, get_device, load_trained_model
from vitxai.viz.plots import plot_class_heatmaps, plot_recency, plot_top_indicators
from vitxai.xai import build_explainer
from vitxai.xai.aggregate import (category_importance, class_mean_maps, method_agreement, normalize_maps,
                                  rankings_table, recency_profile)


def sample_indices(y_pred: np.ndarray, n_classes: int, per_class: int, rng) -> np.ndarray:
    idx = []
    for c in range(n_classes):
        cand = np.flatnonzero(y_pred == c)
        if len(cand):
            idx.append(rng.choice(cand, size=min(per_class, len(cand)), replace=False))
    return np.sort(np.concatenate(idx)) if idx else np.array([], dtype=int)


def compute_fold(cfg, fold, seed, store, labels, methods, xai_root, force):
    out_dir = xai_root / f"fold_{fold['fold']:02d}"
    todo = [m for m in methods if force or not (out_dir / f"attributions_{m}.npz").exists()]
    ckpt = fold_dir(run_dir(cfg), fold["fold"], seed) / "checkpoint.pt"
    if not todo or not ckpt.exists():
        print(f"fold {fold['fold']}: {'done' if ckpt.exists() else 'no checkpoint'}, skipping")
        return
    device = get_device()
    model, ckpt_data = load_trained_model(ckpt, device)
    model_cfg = ckpt_data["model_cfg"]          # the architecture actually trained, not the current config
    datasets, _ = build_fold_datasets(cfg, fold, store, labels)
    test = datasets["test"]

    if len(test) == 0:
        print(f"fold {fold['fold']}: empty test set, skipping")
        return
    preds = pd.read_parquet(ckpt.parent / "predictions.parquet")
    same = (len(preds) == len(test)
            and (preds["ticker"].to_numpy() == test.samples["ticker"].to_numpy()).all()
            and (pd.to_datetime(preds["date"]).to_numpy() == test.samples["date"].to_numpy()).all())
    if not same:
        raise RuntimeError(f"fold {fold['fold']}: test samples differ from predictions.parquet "
                           "(data or config changed after training); retrain with 05_train.py --force")
    x0, _ = test[0]
    if list(x0.shape[1:]) != list(model_cfg["img_size"]):
        raise RuntimeError(f"image shape {tuple(x0.shape)} does not match trained img_size {model_cfg['img_size']}")

    rng = np.random.default_rng(cfg["seed"] + fold["fold"])
    y_pred = preds["y_pred"].to_numpy()
    y_true = preds["y_true"].to_numpy()
    target_kind = str(cfg["xai"].get("target", "predicted"))
    if target_kind not in ("predicted", "actual"):
        raise ValueError("xai.target must be predicted or actual")
    target = y_pred if target_kind == "predicted" else y_true
    idx = sample_indices(y_pred, model_cfg["num_classes"], cfg["xai"]["samples_per_class"], rng)
    x_all = torch.from_numpy(np.stack([test.image(i) for i in idx]))
    target_all = torch.from_numpy(target[idx])
    meta = {
        "sample_idx": idx,
        "date": preds["date"].to_numpy()[idx].astype("datetime64[D]").astype(str),
        "ticker": preds["ticker"].to_numpy()[idx].astype(str),
        "y_true": y_true[idx],
        "y_pred": y_pred[idx],
        "target": target[idx],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    bs = cfg["xai"]["batch_size"]
    for name in todo:
        t0 = time.time()
        explainer = build_explainer(name, model, model_cfg, cfg["xai"])
        maps, deltas = [], []
        for s in range(0, len(idx), bs):
            x = x_all[s:s + bs].to(device)
            maps.append(explainer.explain(x, target_all[s:s + bs].to(device)).astype(np.float32))
            if getattr(explainer, "last_delta", None) is not None:
                deltas.append(explainer.last_delta)
        extra = {"ig_delta": np.concatenate(deltas)} if deltas else {}
        np.savez_compressed(out_dir / f"attributions_{name}.npz", maps=np.concatenate(maps), **meta, **extra)
        msg = f"  fold {fold['fold']} {name:22s} {len(idx)} samples in {time.time() - t0:.1f}s"
        if deltas:
            msg += f"  |IG delta| median={np.median(np.abs(extra['ig_delta'])):.2e}"
        print(msg)


def aggregate(cfg, methods, xai_root):
    names, categories = indicator_names(cfg), indicator_categories(cfg)
    class_names = cfg["labels"]["names"]
    out = xai_root / "summary"
    (out / "figures").mkdir(parents=True, exist_ok=True)
    rankings_by_method = {}
    for name in methods:
        files = sorted(xai_root.glob(f"fold_*/attributions_{name}.npz"))
        if not files:
            continue
        maps, signed, y_pred, y_true = [], [], [], []
        for f in files:
            with np.load(f) as d:
                maps.append(normalize_maps(d["maps"]))
                signed.append(normalize_maps(d["maps"], signed=True))
                y_pred.append(d["y_pred"])
                y_true.append(d["y_true"])
        maps, signed = np.concatenate(maps), np.concatenate(signed)
        y_pred, y_true = np.concatenate(y_pred), np.concatenate(y_true)
        correct = y_pred == y_true

        for subset, mask in (("all", None), ("correct", correct), ("incorrect", ~correct)):
            means = class_mean_maps(maps, y_pred, class_names, mask)
            if not means:
                continue
            signed_means = class_mean_maps(signed, y_pred, class_names, mask)
            rk = rankings_table(means, names, categories, signed_means)
            rk.to_csv(out / f"rankings_{name}_{subset}.csv", index=False)
            category_importance(rk).to_csv(out / f"category_importance_{name}_{subset}.csv")
            recency = {c: recency_profile(m) for c, m in means.items()}
            pd.DataFrame(recency).to_csv(out / f"recency_{name}_{subset}.csv", index_label="column")
            if subset == "all":
                rankings_by_method[name] = rk
                tag = f"{name} ({len(maps)} samples, {len(files)} folds)"
                plot_class_heatmaps(means, names, categories, f"Mean attribution by predicted class: {tag}",
                                    out / "figures" / f"heatmap_{name}.png")
                plot_top_indicators(rk, class_names, cfg["xai"]["top_k_plot"], f"Top indicators: {tag}",
                                    out / "figures" / f"top_indicators_{name}.png")
                plot_recency(recency, f"Recency profile: {tag}", out / "figures" / f"recency_{name}.png")
        print(f"  {name}: {len(maps)} samples from {len(files)} folds")

    if len(rankings_by_method) > 1:
        agree = method_agreement(rankings_by_method, class_names)
        agree.to_csv(out / "method_agreement_spearman.csv", index=False)
        print(agree.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"saved to {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_common_args(ap)
    ap.add_argument("--fold", type=int, nargs="*", help="fold ids (default: all)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--methods", nargs="*", default=None)
    ap.add_argument("--aggregate-only", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.set, args.config_dir)
    methods = args.methods or cfg["xai"]["methods"]
    seed = args.seed if args.seed is not None else cfg["train"]["seeds"][0]
    xai_root = run_dir(cfg) / "xai" / f"seed_{seed}"

    if not args.aggregate_only:
        folds = folds_from_cfg(cfg)
        if args.fold:
            folds = [f for f in folds if f["fold"] in args.fold]
        store = load_indicator_store(cfg, load_universe(cfg))
        labels = pd.read_parquet(labels_path(cfg))
        for fold in folds:
            compute_fold(cfg, fold, seed, store, labels, methods, xai_root, args.force)
    aggregate(cfg, methods, xai_root)


if __name__ == "__main__":
    main()
