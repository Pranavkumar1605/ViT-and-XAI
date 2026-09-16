"""LRP-enabled Vision Transformer for Chefer et al. (2021) relevance.

Ported from hila-chefer/Transformer-Explainability (MIT License):
    baselines/ViT/layers_ours.py, baselines/ViT/ViT_LRP.py, baselines/ViT/ViT_explanation_generator.py
H. Chefer, S. Gur, L. Wolf, "Transformer Interpretability Beyond Attention Visualization", CVPR 2021.

Changes from the original:
  * non-square images and patches (img_size=(90, 65), patch_size=(10, 13)) and in_chans=1
  * no einops dependency; batch explanations are generated one sample at a time
  * the patch embedding is a plain Conv2d, since `transformer_attribution` stops at the blocks
Parameter names match timm's VisionTransformer, so a timm checkpoint loads with strict=True.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================== LRP layers (layers_ours.py)
def safe_divide(a, b):
    den = b.clamp(min=1e-9) + b.clamp(max=1e-9)
    den = den + den.eq(0).type(den.type()) * 1e-9
    return a / den * b.ne(0).type(b.type())


def forward_hook(self, input, output):
    if type(input[0]) in (list, tuple):
        self.X = []
        for i in input[0]:
            x = i.detach()
            x.requires_grad = True
            self.X.append(x)
    else:
        self.X = input[0].detach()
        self.X.requires_grad = True
    self.Y = output


class RelProp(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_forward_hook(forward_hook)

    def gradprop(self, Z, X, S):
        return torch.autograd.grad(Z, X, S, retain_graph=True)

    def relprop(self, R, alpha):
        return R


class RelPropSimple(RelProp):
    def relprop(self, R, alpha):
        Z = self.forward(self.X)
        S = safe_divide(R, Z)
        C = self.gradprop(Z, self.X, S)
        if not torch.is_tensor(self.X):
            return [self.X[0] * C[0], self.X[1] * C[1]]
        return self.X * C[0]


class GELU(nn.GELU, RelProp):
    pass


class Softmax(nn.Softmax, RelProp):
    pass


class LayerNorm(nn.LayerNorm, RelProp):
    pass


class Dropout(nn.Dropout, RelProp):
    pass


class Add(RelPropSimple):
    def forward(self, inputs):
        return torch.add(*inputs)

    def relprop(self, R, alpha):
        Z = self.forward(self.X)
        S = safe_divide(R, Z)
        C = self.gradprop(Z, self.X, S)

        a = self.X[0] * C[0]
        b = self.X[1] * C[1]

        a_sum = a.sum()
        b_sum = b.sum()

        a_fact = safe_divide(a_sum.abs(), a_sum.abs() + b_sum.abs()) * R.sum()
        b_fact = safe_divide(b_sum.abs(), a_sum.abs() + b_sum.abs()) * R.sum()

        a = a * safe_divide(a_fact, a.sum())
        b = b * safe_divide(b_fact, b.sum())
        return [a, b]


class Einsum(RelPropSimple):
    def __init__(self, equation):
        super().__init__()
        self.equation = equation

    def forward(self, operands):
        return torch.einsum(self.equation, *operands)


class IndexSelect(RelProp):
    def forward(self, inputs, dim, indices):
        self.dim = dim
        self.indices = indices
        return torch.index_select(inputs, dim, indices)

    def relprop(self, R, alpha):
        Z = self.forward(self.X, self.dim, self.indices)
        S = safe_divide(R, Z)
        C = self.gradprop(Z, self.X, S)
        return self.X * C[0]


class Clone(RelProp):
    def forward(self, input, num):
        self.num = num
        return [input for _ in range(num)]

    def relprop(self, R, alpha):
        Z = [self.X for _ in range(self.num)]
        S = [safe_divide(r, z) for r, z in zip(R, Z)]
        C = self.gradprop(Z, self.X, S)[0]
        return self.X * C


class Linear(nn.Linear, RelProp):
    def relprop(self, R, alpha):
        beta = alpha - 1
        pw = torch.clamp(self.weight, min=0)
        nw = torch.clamp(self.weight, max=0)
        px = torch.clamp(self.X, min=0)
        nx = torch.clamp(self.X, max=0)

        def f(w1, w2, x1, x2):
            Z1 = F.linear(x1, w1)
            Z2 = F.linear(x2, w2)
            S1 = safe_divide(R, Z1 + Z2)
            S2 = safe_divide(R, Z1 + Z2)
            C1 = x1 * torch.autograd.grad(Z1, x1, S1)[0]
            C2 = x2 * torch.autograd.grad(Z2, x2, S2)[0]
            return C1 + C2

        activator_relevances = f(pw, nw, px, nx)
        inhibitor_relevances = f(nw, pw, px, nx)
        return alpha * activator_relevances - beta * inhibitor_relevances


# =============================================================== ViT (ViT_LRP.py)
def compute_rollout_attention(all_layer_matrices, start_layer=0):
    num_tokens = all_layer_matrices[0].shape[1]
    batch_size = all_layer_matrices[0].shape[0]
    eye = torch.eye(num_tokens).expand(batch_size, num_tokens, num_tokens).to(all_layer_matrices[0].device)
    all_layer_matrices = [m + eye for m in all_layer_matrices]
    joint_attention = all_layer_matrices[start_layer]
    for i in range(start_layer + 1, len(all_layer_matrices)):
        joint_attention = all_layer_matrices[i].bmm(joint_attention)
    return joint_attention


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features, drop=0.0):
        super().__init__()
        self.fc1 = Linear(in_features, hidden_features)
        self.act = GELU()
        self.fc2 = Linear(hidden_features, in_features)
        self.drop = Dropout(drop)

    def forward(self, x):
        x = self.drop(self.act(self.fc1(x)))
        return self.drop(self.fc2(x))

    def relprop(self, cam, **kwargs):
        cam = self.drop.relprop(cam, **kwargs)
        cam = self.fc2.relprop(cam, **kwargs)
        cam = self.act.relprop(cam, **kwargs)
        return self.fc1.relprop(cam, **kwargs)


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.matmul1 = Einsum("bhid,bhjd->bhij")   # A = Q * K^T
        self.matmul2 = Einsum("bhij,bhjd->bhid")   # attn = A * V
        self.qkv = Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = Dropout(attn_drop)
        self.proj = Linear(dim, dim)
        self.proj_drop = Dropout(proj_drop)
        self.softmax = Softmax(dim=-1)
        self.attn_cam = None
        self.attn_gradients = None

    def save_attn_cam(self, cam):
        self.attn_cam = cam

    def save_attn_gradients(self, grad):
        self.attn_gradients = grad

    def forward(self, x):
        b, n, c = x.shape
        h = self.num_heads
        qkv = self.qkv(x).reshape(b, n, 3, h, c // h).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        dot = self.matmul1([q, k]) * self.scale
        attn = self.softmax(dot)
        attn = self.attn_drop(attn)
        if attn.requires_grad:
            attn.register_hook(self.save_attn_gradients)

        out = self.matmul2([attn, v])
        out = out.permute(0, 2, 1, 3).reshape(b, n, c)
        return self.proj_drop(self.proj(out))

    def relprop(self, cam, **kwargs):
        cam = self.proj_drop.relprop(cam, **kwargs)
        cam = self.proj.relprop(cam, **kwargs)
        b, n, c = cam.shape
        h = self.num_heads
        cam = cam.reshape(b, n, h, c // h).permute(0, 2, 1, 3)       # b h n d

        (cam1, cam_v) = self.matmul2.relprop(cam, **kwargs)
        cam1 /= 2
        cam_v /= 2
        self.save_attn_cam(cam1)

        cam1 = self.attn_drop.relprop(cam1, **kwargs)
        cam1 = self.softmax.relprop(cam1, **kwargs)

        (cam_q, cam_k) = self.matmul1.relprop(cam1, **kwargs)
        cam_q /= 2
        cam_k /= 2

        cam_qkv = torch.stack([cam_q, cam_k, cam_v], dim=0)              # 3 b h n d
        cam_qkv = cam_qkv.permute(1, 3, 0, 2, 4).reshape(b, n, c * 3)    # b n 3c
        return self.qkv.relprop(cam_qkv, **kwargs)


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False, drop=0.0, attn_drop=0.0):
        super().__init__()
        self.norm1 = LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), drop=drop)
        self.add1 = Add()
        self.add2 = Add()
        self.clone1 = Clone()
        self.clone2 = Clone()

    def forward(self, x):
        x1, x2 = self.clone1(x, 2)
        x = self.add1([x1, self.attn(self.norm1(x2))])
        x1, x2 = self.clone2(x, 2)
        return self.add2([x1, self.mlp(self.norm2(x2))])

    def relprop(self, cam, **kwargs):
        (cam1, cam2) = self.add2.relprop(cam, **kwargs)
        cam2 = self.mlp.relprop(cam2, **kwargs)
        cam2 = self.norm2.relprop(cam2, **kwargs)
        cam = self.clone2.relprop((cam1, cam2), **kwargs)

        (cam1, cam2) = self.add1.relprop(cam, **kwargs)
        cam2 = self.attn.relprop(cam2, **kwargs)
        cam2 = self.norm1.relprop(cam2, **kwargs)
        return self.clone1.relprop((cam1, cam2), **kwargs)


class PatchEmbed(nn.Module):
    def __init__(self, img_size, patch_size, in_chans, embed_dim):
        super().__init__()
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)   # row-major over the patch grid


class VisionTransformerLRP(nn.Module):
    def __init__(self, img_size=(90, 65), patch_size=(10, 13), in_chans=1, num_classes=3, embed_dim=192,
                 depth=6, num_heads=6, mlp_ratio=4.0, qkv_bias=True, drop_rate=0.0, attn_drop_rate=0.0):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(drop_rate)
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias, drop_rate, attn_drop_rate) for _ in range(depth)
        ])
        self.norm = LayerNorm(embed_dim, eps=1e-6)
        self.head = Linear(embed_dim, num_classes)
        self.pool = IndexSelect()
        self.add = Add()

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.add([x, self.pos_embed.expand(B, -1, -1)])
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        x = self.pool(x, dim=1, indices=torch.tensor(0, device=x.device))
        x = x.squeeze(1)
        return self.head(x)

    def relprop(self, cam, start_layer=0, **kwargs):
        cam = self.head.relprop(cam, **kwargs)
        cam = cam.unsqueeze(1)
        cam = self.pool.relprop(cam, **kwargs)
        cam = self.norm.relprop(cam, **kwargs)
        for blk in reversed(self.blocks):
            cam = blk.relprop(cam, **kwargs)

        # method == "transformer_attribution"
        cams = []
        for blk in self.blocks:
            grad = blk.attn.attn_gradients
            cam = blk.attn.attn_cam
            cam = cam[0].reshape(-1, cam.shape[-1], cam.shape[-1])
            grad = grad[0].reshape(-1, grad.shape[-1], grad.shape[-1])
            cam = grad * cam
            cam = cam.clamp(min=0).mean(dim=0)
            cams.append(cam.unsqueeze(0))
        rollout = compute_rollout_attention(cams, start_layer=start_layer)
        return rollout[:, 0, 1:]                      # CLS row over the patch tokens


def build_vit_lrp(model_cfg: dict) -> VisionTransformerLRP:
    return VisionTransformerLRP(
        img_size=tuple(model_cfg["img_size"]),
        patch_size=tuple(model_cfg["patch_size"]),
        in_chans=model_cfg["in_chans"],
        num_classes=model_cfg["num_classes"],
        embed_dim=model_cfg["embed_dim"],
        depth=model_cfg["depth"],
        num_heads=model_cfg["num_heads"],
        mlp_ratio=model_cfg.get("mlp_ratio", 4.0),
        qkv_bias=model_cfg.get("qkv_bias", True),
        drop_rate=model_cfg.get("drop_rate", 0.0),
        attn_drop_rate=model_cfg.get("attn_drop_rate", 0.0),
    )


def lrp_from_timm(model_cfg: dict, timm_model: nn.Module) -> VisionTransformerLRP:
    lrp = build_vit_lrp(model_cfg)
    lrp.load_state_dict(timm_model.state_dict(), strict=True)
    return lrp.eval()


def generate_relevance(model: VisionTransformerLRP, x: torch.Tensor, index: int | None = None,
                       start_layer: int = 0) -> torch.Tensor:
    """x: (1, C, H, W). Returns patch relevance (num_patches,) for class `index`
    (predicted class if None). Mirrors LRP.generate_LRP in ViT_explanation_generator.py."""
    with torch.enable_grad():
        output = model(x)
        if index is None:
            index = int(output.argmax(dim=-1)[0])
        one_hot_vector = torch.zeros_like(output)
        one_hot_vector[0, index] = 1.0
        one_hot = torch.sum(one_hot_vector * output)
        model.zero_grad()
        one_hot.backward(retain_graph=True)
        cam = model.relprop(one_hot_vector, start_layer=start_layer, alpha=1)
    return cam[0].detach()
