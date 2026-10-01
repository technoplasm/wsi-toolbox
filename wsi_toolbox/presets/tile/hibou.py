"""Hibou-B / Hibou-L (HistAI): DINOv2 ViT/14 with 4 registers, SwiGLU FFN and LayerScale.

The HuggingFace repos ship transformers weights for a custom ``Dinov2ModelWithRegisters`` that is only
loadable with ``trust_remote_code=True`` (code written for transformers 4.35). The architecture is a plain
DINOv2-reg ViT, so it is rebuilt here as a timm ``VisionTransformer`` and the weights are renamed and
strict-loaded; nothing but ``model.safetensors`` is read from the hub, so it works with ``HF_HUB_OFFLINE=1``.

Embedding: the CLS token after the final LayerNorm (= ``pooler_output`` of the HF model, ``head(x_norm_clstoken)``
of the HistAI repo). Token order is ``[CLS, 4 registers, patches]``, so the patch tokens stay at the end.
The position embedding is resampled like the HistAI repo (bicubic, antialias, no offset), which is what timm does.
"""

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from timm.layers import SwiGLUPacked
from timm.models.vision_transformer import VisionTransformer

# variant -> (HF repo, embed_dim, depth, num_heads)
HIBOU_VARIANTS: dict[str, tuple[str, int, int, int]] = {
    "b": ("histai/hibou-b", 768, 12, 12),
    "l": ("histai/hibou-L", 1024, 24, 16),
}

_NUM_REGISTERS = 4


def _swiglu_hidden(embed_dim: int) -> int:
    """Hidden width of the DINOv2 SwiGLU FFN (mlp_ratio 4): 768 -> 2048, 1024 -> 2736."""
    return (int(embed_dim * 4 * 2 / 3) + 7) // 8 * 8


def build_hibou_vit(embed_dim: int, depth: int, num_heads: int) -> VisionTransformer:
    """The Hibou architecture without weights."""
    return VisionTransformer(
        img_size=224,
        patch_size=14,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        # SwiGLUPacked: fc1 -> 2 * hidden (silu(x1) * x2), fc2 <- hidden (DINOv2 weights_in / weights_out)
        mlp_ratio=2 * _swiglu_hidden(embed_dim) / embed_dim,
        mlp_layer=SwiGLUPacked,
        act_layer=nn.SiLU,
        init_values=1e-5,  # LayerScale
        qkv_bias=True,
        num_classes=0,
        global_pool="token",
        class_token=True,
        reg_tokens=_NUM_REGISTERS,
        no_embed_class=True,  # pos_embed covers only the patches; the CLS position is folded into cls_token
        dynamic_img_size=True,
        dynamic_img_pad=True,
    )


def convert_hibou_state_dict(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Rename a ``Dinov2ModelWithRegisters`` state dict to timm ``VisionTransformer`` keys.

    The CLS position embedding is a constant added to the CLS token, so it is folded into ``cls_token``
    (timm with ``no_embed_class=True`` adds ``pos_embed`` to the patches only). ``mask_token`` is dropped.
    """
    out: dict[str, torch.Tensor] = {}
    pos = sd["embeddings.position_embeddings"]
    out["cls_token"] = sd["embeddings.cls_token"] + pos[:, :1]
    out["pos_embed"] = pos[:, 1:]
    out["reg_token"] = sd["embeddings.register_tokens"]
    out["patch_embed.proj.weight"] = sd["embeddings.patch_embeddings.projection.weight"]
    out["patch_embed.proj.bias"] = sd["embeddings.patch_embeddings.projection.bias"]
    out["norm.weight"] = sd["layernorm.weight"]
    out["norm.bias"] = sd["layernorm.bias"]

    depth = 1 + max(int(k.split(".")[2]) for k in sd if k.startswith("encoder.layer."))
    for i in range(depth):
        src = f"encoder.layer.{i}."
        dst = f"blocks.{i}."
        att = src + "attention.attention."
        for p in ("weight", "bias"):
            out[dst + f"norm1.{p}"] = sd[src + f"norm1.{p}"]
            out[dst + f"norm2.{p}"] = sd[src + f"norm2.{p}"]
            out[dst + f"attn.qkv.{p}"] = torch.cat(
                [sd[att + f"query.{p}"], sd[att + f"key.{p}"], sd[att + f"value.{p}"]]
            )
            out[dst + f"attn.proj.{p}"] = sd[src + f"attention.output.dense.{p}"]
            out[dst + f"mlp.fc1.{p}"] = sd[src + f"mlp.weights_in.{p}"]
            out[dst + f"mlp.fc2.{p}"] = sd[src + f"mlp.weights_out.{p}"]
        out[dst + "ls1.gamma"] = sd[src + "layer_scale1.lambda1"]
        out[dst + "ls2.gamma"] = sd[src + "layer_scale2.lambda1"]
    return out


def create_hibou_model(variant: str) -> VisionTransformer:
    """Create Hibou-B (``"b"``) or Hibou-L (``"l"``) with pretrained weights from HuggingFace."""
    repo, embed_dim, depth, num_heads = HIBOU_VARIANTS[variant]
    model = build_hibou_vit(embed_dim, depth, num_heads)
    sd = load_file(hf_hub_download(repo, filename="model.safetensors"))
    model.load_state_dict(convert_hibou_state_dict(sd))
    return model
