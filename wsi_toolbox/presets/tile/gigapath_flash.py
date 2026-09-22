"""GigaPath-Flash: ViT-S/16 (DINOv2-small, SwiGLU FFN, LayerScale) distilled from Prov-GigaPath.

384-dim CLS embedding, ~22M params, ImageNet normalization, Apache-2.0.
The weights are timm-style (config.json + pytorch_model.bin) but the architecture name
``gigapath_tile_enc_dinov2s`` is registered only in the prov-gigapath repo, so the
VisionTransformer is built here directly (mirrors ``gigapath/tile_encoder.py``).
"""

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from timm.layers import SwiGLUPacked
from timm.models.vision_transformer import VisionTransformer

HF_REPO = "prov-gigapath/prov-gigapath-flash"


def create_gigapath_flash_model() -> VisionTransformer:
    """Create the GigaPath-Flash tile encoder with pretrained weights from HuggingFace."""
    model = VisionTransformer(
        img_size=224,
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=2048 / 384.0,  # SwiGLU: fc1 -> 2048, fc2 <- 1024 (DINOv2 w12/w3)
        mlp_layer=SwiGLUPacked,
        act_layer=nn.SiLU,
        init_values=1e-5,  # LayerScale
        num_classes=0,
        global_pool="token",
        class_token=True,
        reg_tokens=0,
        dynamic_img_size=True,
        dynamic_img_pad=True,
    )

    checkpoint_path = hf_hub_download(HF_REPO, filename="pytorch_model.bin")
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if "patch_embed.proj.weight" not in state_dict and "model" in state_dict:
        state_dict = state_dict["model"]
    model.load_state_dict(state_dict)

    return model
