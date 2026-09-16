from timm.models.vision_transformer import VisionTransformer


def build_vit(model_cfg: dict) -> VisionTransformer:
    # In timm, `drop_rate` alone only drops before the head; the standard ViT dropout positions
    # (after the position embedding, attention projection and MLP) have their own arguments.
    drop = model_cfg.get("drop_rate", 0.0)
    return VisionTransformer(
        img_size=tuple(model_cfg["img_size"]),
        patch_size=tuple(model_cfg["patch_size"]),
        in_chans=model_cfg["in_chans"],
        num_classes=model_cfg["num_classes"],
        embed_dim=model_cfg["embed_dim"],
        depth=model_cfg["depth"],
        num_heads=model_cfg["num_heads"],
        mlp_ratio=model_cfg.get("mlp_ratio", 4.0),
        qkv_bias=model_cfg.get("qkv_bias", True),
        drop_rate=drop,
        pos_drop_rate=drop,
        proj_drop_rate=drop,
        attn_drop_rate=model_cfg.get("attn_drop_rate", 0.0),
        global_pool="token",
        class_token=True,
    )


def patch_grid(model_cfg: dict) -> tuple[int, int]:
    (h, w), (ph, pw) = model_cfg["img_size"], model_cfg["patch_size"]
    assert h % ph == 0 and w % pw == 0, "image must divide evenly into patches"
    return h // ph, w // pw
