"""Model registry. A new model = a new module with build_<name>(model_cfg) + one entry here."""
from vitxai.models.vit import build_vit

MODEL_REGISTRY = {
    "vit": build_vit,
}


def build_model(model_cfg: dict):
    return MODEL_REGISTRY[model_cfg["name"]](model_cfg)
