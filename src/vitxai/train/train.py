"""Phase 5: train one model on one fold, early-stop on validation loss, save test predictions."""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from vitxai.models import build_model
from vitxai.seed import seed_everything

PRED_COLUMNS = ["date", "ticker", "y_true", "y_pred", "p_hold", "p_buy", "p_sell"]


def fold_dir(run_dir: Path, fold: int, seed: int) -> Path:
    return run_dir / f"fold_{fold:02d}" / f"seed_{seed}"


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _loader(ds, train_cfg: dict, shuffle: bool, seed: int) -> DataLoader:
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(ds, batch_size=train_cfg["batch_size"], shuffle=shuffle, generator=g,
                      num_workers=train_cfg.get("num_workers", 0), pin_memory=torch.cuda.is_available())


def _class_weights(labels: np.ndarray, num_classes: int) -> torch.Tensor:
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    w = counts.sum() / (num_classes * np.maximum(counts, 1))
    return torch.tensor(w, dtype=torch.float32)


def _lr_lambda(train_cfg: dict, steps_per_epoch: int):
    total = train_cfg["max_epochs"] * steps_per_epoch
    warmup = train_cfg.get("warmup_epochs", 0) * steps_per_epoch

    def f(step):
        if warmup and step < warmup:
            return (step + 1) / warmup
        if train_cfg.get("scheduler", "none") == "cosine":
            progress = (step - warmup) / max(1, total - warmup)
            return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))
        return 1.0
    return f


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """Returns (probs N x C, labels N) in dataset order."""
    model.eval()
    probs, ys = [], []
    for x, y in loader:
        logits = model(x.to(device, non_blocking=True))
        probs.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(probs), np.concatenate(ys)


def _per_class_recall(y: np.ndarray, pred: np.ndarray, num_classes: int) -> list[float]:
    return [float((pred[y == c] == c).mean()) if (y == c).any() else float("nan") for c in range(num_classes)]


def train_fold(cfg: dict, fold: dict, datasets: dict, out_dir: Path, seed: int) -> dict:
    seed_everything(seed)
    device = get_device()
    tcfg, mcfg = cfg["train"], cfg["model"]
    n_cls = mcfg["num_classes"]
    for split in ("train", "val", "test"):
        if len(datasets[split]) == 0:
            raise ValueError(f"fold {fold['fold']}: empty {split} split")

    model = build_model(mcfg).to(device)
    train_loader = _loader(datasets["train"], tcfg, shuffle=True, seed=seed)
    val_loader = _loader(datasets["val"], tcfg, shuffle=False, seed=seed)
    test_loader = _loader(datasets["test"], tcfg, shuffle=False, seed=seed)

    weight = _class_weights(datasets["train"].samples["label"].to_numpy(), n_cls) if tcfg.get("class_weights") else None
    criterion = nn.CrossEntropyLoss(weight=weight.to(device) if weight is not None else None)
    optimizers = {"adamw": torch.optim.AdamW, "adam": torch.optim.Adam}
    if tcfg.get("optimizer", "adamw") not in optimizers:
        raise ValueError(f"train.optimizer must be one of {list(optimizers)}")
    optimizer = optimizers[tcfg.get("optimizer", "adamw")](model.parameters(), lr=tcfg["lr"],
                                                          weight_decay=tcfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda(tcfg, len(train_loader)))
    use_amp = bool(tcfg.get("amp")) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "checkpoint.pt"
    ckpt_path.unlink(missing_ok=True)       # never reuse a checkpoint from an interrupted earlier run
    best_loss, best_epoch, bad_epochs = float("inf"), -1, 0
    history = []
    t0 = time.time()
    for epoch in range(tcfg["max_epochs"]):
        model.train()
        total, n = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total += loss.item() * len(y)
            n += len(y)

        val_probs, val_y = predict(model, val_loader, device)
        val_loss = float(nn.functional.cross_entropy(
            torch.log(torch.from_numpy(val_probs).clamp_min(1e-12)), torch.from_numpy(val_y),
            weight=weight).item())
        val_pred = val_probs.argmax(1)
        rec = _per_class_recall(val_y, val_pred, n_cls)
        history.append({"epoch": epoch, "train_loss": total / n, "val_loss": val_loss,
                        "val_acc": float((val_pred == val_y).mean()),
                        **{f"val_recall_{c}": r for c, r in enumerate(rec)},
                        "lr": optimizer.param_groups[0]["lr"], "elapsed_s": time.time() - t0})
        print(f"    epoch {epoch:3d} train_loss={total / n:.4f} val_loss={val_loss:.4f} "
              f"val_acc={history[-1]['val_acc']:.3f} val_recall={np.round(rec, 3).tolist()}")

        if math.isfinite(val_loss) and val_loss < best_loss - 1e-6:
            best_loss, best_epoch, bad_epochs = val_loss, epoch, 0
            torch.save({"model_state": model.state_dict(), "model_cfg": mcfg, "fold": fold,
                        "seed": seed, "epoch": epoch}, ckpt_path)
        else:
            bad_epochs += 1
            if bad_epochs >= tcfg["patience"]:
                break

    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)
    if not ckpt_path.exists():
        raise RuntimeError(f"fold {fold['fold']} seed {seed}: validation loss never finite; see history.csv")

    # Test predictions from the best checkpoint.
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    probs, y = predict(model, test_loader, device)
    meta = datasets["test"].samples
    preds = pd.DataFrame({
        "date": meta["date"].to_numpy(), "ticker": meta["ticker"].to_numpy(),
        "y_true": y, "y_pred": probs.argmax(1),
        "p_hold": probs[:, 0], "p_buy": probs[:, 1], "p_sell": probs[:, 2],
    })[PRED_COLUMNS]
    preds.to_parquet(out_dir / "predictions.parquet", index=False)

    summary = {
        "fold": fold["fold"], "test_year": fold["test_year"], "seed": seed, "device": str(device),
        "best_epoch": best_epoch, "best_val_loss": best_loss, "epochs_run": len(history),
        "n_train": len(datasets["train"]), "n_val": len(datasets["val"]), "n_test": len(datasets["test"]),
        "test_acc": float((preds["y_true"] == preds["y_pred"]).mean()),
        "train_time_s": time.time() - t0, "torch": torch.__version__,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def load_trained_model(ckpt_path: Path, device: torch.device | str = "cpu") -> tuple[nn.Module, dict]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_model(ckpt["model_cfg"])
    model.load_state_dict(ckpt["model_state"])
    return model.to(device).eval(), ckpt
