"""Row occlusion: replace one indicator row at a time with `fill_value` (0 = training mean)
and measure the drop in the target-class probability.

importance[row] = p_target(original) - p_target(row occluded)
Positive = the row supported the prediction. The score is spread evenly over the row's columns so
the map sums per row back to the row score and fits the shared (N, H, W) aggregation."""
from __future__ import annotations

import numpy as np
import torch


class RowOcclusion:
    def __init__(self, model, model_cfg: dict, cfg: dict):
        self.model = model.eval()
        self.fill_value = cfg.get("fill_value", 0.0)

    @torch.no_grad()
    def row_scores(self, x: torch.Tensor, target: torch.Tensor) -> np.ndarray:
        """x: (N, 1, H, W) -> (N, H)."""
        n, _, h, w = x.shape
        scores = np.zeros((n, h), dtype=np.float32)
        eye = torch.eye(h, dtype=torch.bool, device=x.device)            # row r of the batch occludes row r
        for i in range(n):
            base = x[i:i + 1]
            occluded = base.repeat(h, 1, 1, 1)
            occluded[eye[:, None, :, None].expand(h, 1, h, w)] = self.fill_value
            probs = torch.softmax(self.model(torch.cat([base, occluded])).float(), dim=1)[:, int(target[i])]
            scores[i] = (probs[0] - probs[1:]).cpu().numpy()
        return scores

    def explain(self, x: torch.Tensor, target: torch.Tensor) -> np.ndarray:
        scores = self.row_scores(x, target)
        w = x.shape[-1]
        return np.repeat(scores[:, :, None], w, axis=2) / w
