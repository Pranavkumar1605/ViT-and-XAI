"""Config loading.

`load_config()` reads configs/base.yaml and attaches the model, train, xai and indicator
files it names under cfg["model"], cfg["train"], cfg["xai"], cfg["indicators"].
Overrides use dotted keys, e.g. ["train.max_epochs=1", "data.max_tickers=3"].
"""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"

_SUBCONFIGS = {
    "model": "model_config",
    "train": "train_config",
    "xai": "xai_config",
    "indicators": "indicators_config",
}


def _read_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _set_dotted(cfg: dict, key: str, value) -> None:
    node = cfg
    parts = key.split(".")
    for p in parts[:-1]:
        node = node.setdefault(p, {})
    node[parts[-1]] = value


def apply_overrides(cfg: dict, overrides: list[str] | None) -> dict:
    for item in overrides or []:
        key, _, raw = item.partition("=")
        _set_dotted(cfg, key.strip(), yaml.safe_load(raw))
    return cfg


def load_config(overrides: list[str] | None = None, config_dir: Path | str = CONFIG_DIR) -> dict:
    config_dir = Path(config_dir)
    cfg = _read_yaml(config_dir / "base.yaml")
    # Overrides first, so e.g. `model_config=...` can swap a sub-config file.
    apply_overrides(cfg, overrides)
    for section, file_key in _SUBCONFIGS.items():
        cfg[section] = _read_yaml(config_dir / cfg[file_key])
    apply_overrides(cfg, overrides)
    return cfg


def resolve(path: str | Path) -> Path:
    """Relative paths in configs are relative to the project root."""
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def data_dir(cfg: dict) -> Path:
    return resolve(cfg["paths"]["data_dir"])


def run_dir(cfg: dict) -> Path:
    return resolve(cfg["paths"]["results_dir"]) / cfg["run_id"]


def save_config(cfg: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(copy.deepcopy(cfg), f, sort_keys=False)


def add_common_args(parser) -> None:
    parser.add_argument("--config-dir", default=str(CONFIG_DIR))
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                        help="config overrides, e.g. train.max_epochs=1 data.max_tickers=3")
