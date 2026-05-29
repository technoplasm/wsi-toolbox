"""OpenMidnight: DINOv2 ViT-G/14 with registers, retrained on pathology data."""

import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download


class MidnightModel(nn.Module):
    """Wrapper around DINOv2 ViT-G/14-reg for pipeline compatibility."""

    def __init__(self):
        super().__init__()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="xFormers is not available")
            self.trunk = torch.hub.load("facebookresearch/dinov2", "dinov2_vitg14_reg", pretrained=False)

    @property
    def num_features(self) -> int:
        return self.trunk.embed_dim

    @property
    def patch_embed(self):
        return self.trunk.patch_embed

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        # Resize to nearest multiple of patch_size (14) if needed
        _, _, h, w = x.shape
        patch_size = self.trunk.patch_embed.patch_size[0]
        if h % patch_size != 0 or w % patch_size != 0:
            new_h = round(h / patch_size) * patch_size
            new_w = round(w / patch_size) * patch_size
            x = F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False)

        out = self.trunk.forward_features(x)
        cls_token = out["x_norm_clstoken"].unsqueeze(1)
        patch_tokens = out["x_norm_patchtokens"]
        return torch.cat([cls_token, patch_tokens], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_features(x)


def create_midnight_model() -> MidnightModel:
    """Create OpenMidnight model with pretrained weights from HuggingFace."""
    model = MidnightModel()

    checkpoint_path = hf_hub_download(
        "SophontAI/OpenMidnight",
        filename="teacher_checkpoint_load.pt",
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    # pos_embed shape differs (DINOv2 default=392, OpenMidnight=224 based)
    model.trunk.pos_embed = nn.Parameter(checkpoint["pos_embed"])
    model.trunk.load_state_dict(checkpoint)

    return model
