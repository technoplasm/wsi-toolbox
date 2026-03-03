"""CONCH v1.5 vision tower: timm ViT-L + AttentionalPooler."""

import timm
import torch
import torch.nn as nn
from einops import rearrange, repeat
from huggingface_hub import hf_hub_download
from timm.layers import resample_abs_pos_embed
from torch import einsum


class AttentionalPooler(nn.Module):
    def __init__(
        self,
        d_model: int,
        context_dim: int,
        n_head: int = 8,
        n_queries: int = 256,
        norm_layer: type = nn.LayerNorm,
    ):
        super().__init__()
        self.query = nn.Parameter(torch.randn(n_queries, d_model))
        dim_head = d_model // n_head
        self.scale = dim_head**-0.5
        self.heads = n_head
        inner_dim = dim_head * n_head
        self.ln_k = norm_layer(context_dim)
        self.ln_q = norm_layer(d_model)
        self.to_q = nn.Linear(d_model, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            x = rearrange(x, "b n d -> b 1 n d")
        q = repeat(self.query, "n d -> b m n d", b=x.shape[0], m=x.shape[1])
        x = self.ln_k(x)
        q = self.ln_q(q)
        h = self.heads
        q = self.to_q(q)
        k, v = self.to_kv(x).chunk(2, dim=-1)
        q, k, v = (rearrange(t, "b t n (h d) -> b h t n d", h=h) for t in (q, k, v))
        q = q * self.scale
        sim = einsum("... i d, ... j d -> ... i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)
        out = einsum("... i j, ... j d -> ... i d", attn, v)
        out = rearrange(out, "b h t n d -> b t n (h d)", h=h)
        return self.to_out(out).squeeze(dim=1)


class CONCHVisionTower(nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk = timm.create_model(
            "vit_large_patch16_224",
            pretrained=False,
            num_classes=0,
            init_values=1.0,
            dynamic_img_size=True,
        )
        self.attn_pool_contrast = AttentionalPooler(d_model=768, context_dim=1024, n_head=8, n_queries=1)
        self.ln_contrast = nn.LayerNorm(768)

    @property
    def num_features(self) -> int:
        return self.trunk.num_features

    @property
    def patch_embed(self):
        return self.trunk.patch_embed

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.trunk.forward_features(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.attn_pool_contrast(x)[:, 0]
        x = self.ln_contrast(x)
        return x


def create_conch_model() -> CONCHVisionTower:
    """Create CONCH v1.5 model with pretrained weights from HuggingFace."""
    model = CONCHVisionTower()

    checkpoint_path = hf_hub_download(
        "MahmoodLab/conchv1_5",
        filename="pytorch_model_vision.bin",
    )

    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    if model.trunk.pos_embed.shape != state_dict["trunk.pos_embed"].shape:
        state_dict["trunk.pos_embed"] = resample_abs_pos_embed(
            state_dict["trunk.pos_embed"],
            new_size=model.trunk.patch_embed.grid_size,
            num_prefix_tokens=model.trunk.num_prefix_tokens,
        )

    model.load_state_dict(state_dict, strict=True)
    return model
