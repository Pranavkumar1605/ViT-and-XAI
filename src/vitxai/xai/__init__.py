"""XAI method registry.

Every explainer is built as `cls(model, model_cfg, method_cfg)` and exposes
`explain(x: Tensor[N, 1, H, W], target: Tensor[N]) -> ndarray[N, H, W]` (signed or unsigned scores
on the image grid). A new method = a new module + one entry here + its name in configs/xai.yaml.
"""
from vitxai.xai.chefer import CheferRelevance
from vitxai.xai.integrated_gradients import IntegratedGradientsExplainer
from vitxai.xai.row_occlusion import RowOcclusion

XAI_REGISTRY = {
    "integrated_gradients": IntegratedGradientsExplainer,
    "row_occlusion": RowOcclusion,
    "chefer": CheferRelevance,
}


def build_explainer(name: str, model, model_cfg: dict, xai_cfg: dict):
    return XAI_REGISTRY[name](model, model_cfg, xai_cfg.get(name, {}) or {})
