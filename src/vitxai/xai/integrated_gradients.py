"""Integrated Gradients (Sundararajan et al., 2017) via Captum.

Baseline = all-zero image, which equals the training mean after standardization."""
from __future__ import annotations

import numpy as np
import torch
from captum.attr import IntegratedGradients


class IntegratedGradientsExplainer:
    def __init__(self, model, model_cfg: dict, cfg: dict):
        self.model = model.eval()
        self.ig = IntegratedGradients(self.model)
        self.n_steps = cfg.get("n_steps", 50)
        self.internal_batch_size = cfg.get("internal_batch_size", 64)
        self.last_delta: np.ndarray | None = None

    def explain(self, x: torch.Tensor, target: torch.Tensor) -> np.ndarray:
        x = x.clone().requires_grad_(True)
        attr, delta = self.ig.attribute(
            x, baselines=torch.zeros_like(x), target=target, n_steps=self.n_steps,
            internal_batch_size=self.internal_batch_size, return_convergence_delta=True,
        )
        self.last_delta = delta.detach().cpu().numpy()   # completeness check: should be ~0
        return attr.detach().cpu().numpy()[:, 0]
