"""Chefer et al. (2021) transformer relevance (method `transformer_attribution`).

The trained timm ViT weights are copied into the LRP-enabled port (models/vit_lrp.py); patch
relevance (9 x 5) is expanded to the 90 x 65 grid by repeating each value over its 10 x 13 patch."""
from __future__ import annotations

import numpy as np
import torch

from vitxai.models.vit import patch_grid
from vitxai.models.vit_lrp import generate_relevance, lrp_from_timm


def patches_to_grid(patch_scores: np.ndarray, model_cfg: dict) -> np.ndarray:
    """(N, num_patches) -> (N, H, W), row-major patch order as in timm's PatchEmbed."""
    gh, gw = patch_grid(model_cfg)
    ph, pw = model_cfg["patch_size"]
    grid = patch_scores.reshape(-1, gh, gw)
    return np.stack([np.kron(g, np.ones((ph, pw), dtype=grid.dtype)) for g in grid])


class CheferRelevance:
    def __init__(self, model, model_cfg: dict, cfg: dict):
        device = next(model.parameters()).device
        self.lrp = lrp_from_timm(model_cfg, model.cpu()).to(device)
        model.to(device)
        self.model_cfg = model_cfg
        self.start_layer = cfg.get("start_layer", 0)

    def explain(self, x: torch.Tensor, target: torch.Tensor) -> np.ndarray:
        rel = [generate_relevance(self.lrp, x[i:i + 1], index=int(target[i]), start_layer=self.start_layer)
               for i in range(len(x))]
        patch_scores = torch.stack(rel).cpu().numpy()
        return patches_to_grid(patch_scores, self.model_cfg)
