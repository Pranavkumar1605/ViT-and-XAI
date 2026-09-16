"""The base-paper replication config (configs/paper_etf) loads and produces valid 65 x 65 images."""
import numpy as np
import pytest
import torch

from conftest import synthetic_prices
from vitxai.config import CONFIG_DIR, load_config
from vitxai.features.indicators import compute_indicators, drop_warmup, indicator_specs
from vitxai.models import MODEL_REGISTRY

PAPER_ORDER = ["statistics", "price_transform", "volatility", "volume", "momentum", "overlap"]


@pytest.fixture(scope="module")
def paper_cfg():
    return load_config(config_dir=CONFIG_DIR / "paper_etf")


def test_paper_config_has_65_indicators_in_table2_order(paper_cfg):
    specs = indicator_specs(paper_cfg)
    assert len(specs) == 65 == paper_cfg["model"]["img_size"][0] == paper_cfg["image"]["window"]
    cats = [s["category"] for s in specs]
    assert [c for i, c in enumerate(cats) if i == 0 or c != cats[i - 1]] == PAPER_ORDER
    assert {c: cats.count(c) for c in PAPER_ORDER} == {"statistics": 9, "price_transform": 4, "volatility": 1,
                                                        "volume": 3, "momentum": 34, "overlap": 14}
    assert paper_cfg["train"]["seeds"] and paper_cfg["xai"]["methods"]   # shared sub-configs resolved


def test_paper_indicators_finite_causal_and_distinct(paper_cfg):
    p, idx = synthetic_prices()
    specs = indicator_specs(paper_cfg)
    full = compute_indicators(p, idx, specs)
    ind = drop_warmup(full)
    assert ind.shape[1] == 65 and len(ind) > 200 and np.isfinite(ind.to_numpy()).all()
    t = 300
    cut = compute_indicators(p.iloc[: t + 1], idx, specs)
    np.testing.assert_allclose(full.iloc[: t + 1].to_numpy(), cut.to_numpy(), rtol=1e-9, atol=1e-12, equal_nan=True)
    # no two rows are exact copies (ROC family differs by affine maps, which is allowed)
    assert len({tuple(np.round(ind[c].to_numpy(), 12)) for c in ind.columns}) == 65


def test_paper_vit_shape(paper_cfg):
    model_cfg = dict(paper_cfg["model"], embed_dim=48, depth=1, num_heads=4)
    model = MODEL_REGISTRY[model_cfg["name"]](model_cfg)
    assert model.patch_embed.num_patches == 25
    assert model(torch.zeros(2, 1, 65, 65)).shape == (2, 3)
