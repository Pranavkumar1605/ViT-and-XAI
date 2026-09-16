import numpy as np
import pytest
import torch
import torch.nn as nn

from vitxai.models.vit import build_vit
from vitxai.models.vit_lrp import lrp_from_timm
from vitxai.xai import XAI_REGISTRY, build_explainer
from vitxai.xai.aggregate import indicator_importance, normalize_maps, rankings_table, recency_profile
from vitxai.xai.chefer import patches_to_grid
from vitxai.xai.row_occlusion import RowOcclusion

SMALL = {"name": "vit", "img_size": [90, 65], "patch_size": [10, 13], "in_chans": 1, "num_classes": 3,
         "embed_dim": 48, "depth": 2, "num_heads": 4, "mlp_ratio": 2.0, "qkv_bias": True, "drop_rate": 0.1}


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    m = build_vit(SMALL)
    with torch.no_grad():                      # move away from init so attention is not uniform
        for p in m.parameters():
            p.add_(0.05 * torch.randn_like(p))
    return m.eval()


def test_lrp_port_matches_timm(model):
    x = torch.randn(3, 1, 90, 65)
    lrp = lrp_from_timm(SMALL, model)
    with torch.no_grad():
        torch.testing.assert_close(lrp(x), model(x), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("name", list(XAI_REGISTRY))
def test_explainers_shape_and_finite(model, name):
    x = torch.randn(2, 1, 90, 65)
    target = torch.tensor([0, 2])
    xai_cfg = {"integrated_gradients": {"n_steps": 16, "internal_batch_size": 16}}
    maps = build_explainer(name, model, SMALL, xai_cfg).explain(x, target)
    assert maps.shape == (2, 90, 65)
    assert np.isfinite(maps).all()
    assert np.abs(maps).sum() > 0


class _IgnoresFirstRow(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(89 * 65, 3)

    def forward(self, x):
        return self.fc(x[:, :, 1:, :].flatten(1))


def test_row_occlusion_zero_for_ignored_row():
    torch.manual_seed(0)
    x = torch.randn(4, 1, 90, 65)
    scores = RowOcclusion(_IgnoresFirstRow().eval(), SMALL, {}).row_scores(x, torch.tensor([0, 1, 2, 0]))
    assert scores.shape == (4, 90)
    np.testing.assert_allclose(scores[:, 0], 0.0, atol=1e-7)
    assert np.abs(scores[:, 1:]).sum() > 0


def test_integrated_gradients_completeness(model):
    x = torch.randn(2, 1, 90, 65)
    ig = build_explainer("integrated_gradients", model, SMALL, {"integrated_gradients": {"n_steps": 200}})
    maps = ig.explain(x, torch.tensor([1, 1]))
    with torch.no_grad():
        gap = (model(x)[:, 1] - model(torch.zeros_like(x))[:, 1]).numpy()
    np.testing.assert_allclose(maps.sum(axis=(1, 2)), gap, rtol=0.05, atol=1e-3)


def test_patch_grid_mapping():
    scores = np.arange(45, dtype=float)[None]           # patch k = row k//5, col k%5
    grid = patches_to_grid(scores, SMALL)
    assert grid.shape == (1, 90, 65)
    assert grid[0, 0, 0] == 0 and grid[0, 0, 13] == 1 and grid[0, 10, 0] == 5 and grid[0, 89, 64] == 44


def test_dropout_reaches_all_standard_positions():
    m = build_vit(SMALL)
    assert m.pos_drop.p == m.blocks[0].attn.proj_drop.p == m.blocks[0].mlp.drop1.p == m.head_drop.p == 0.1


def test_rankings_ties_and_signed_columns():
    names, cats = [f"i{k}" for k in range(4)], ["a"] * 4
    mean = np.array([[0.2, 0.1], [0.15, 0.15], [0.1, 0.1], [0.1, 0.1]])  # rows 0,1 tie; rows 2,3 tie
    signed = {"Buy": -mean}
    rk = rankings_table({"Buy": mean}, names, cats, signed)
    assert rk["rank_Buy"].tolist() == [1, 1, 3, 3]
    assert (rk["signed_Buy"] < 0).all()


def test_signed_normalization_keeps_direction():
    maps = np.array([[[1.0, -3.0]]])
    np.testing.assert_allclose(normalize_maps(maps), [[[0.25, 0.75]]])
    np.testing.assert_allclose(normalize_maps(maps, signed=True), [[[0.25, -0.75]]])


def test_aggregation_sums():
    maps = np.random.default_rng(0).normal(size=(5, 90, 65))
    norm = normalize_maps(maps)
    np.testing.assert_allclose(norm.sum(axis=(1, 2)), 1.0)
    m = norm.mean(axis=0)
    assert indicator_importance(m).shape == (90,)
    assert recency_profile(m).shape == (65,)
    assert np.isclose(indicator_importance(m).sum(), 1.0)
