"""Phikon-v2: DINOv2 ViT-L/16 for pathology."""

import torch
import torch.nn as nn
from transformers import AutoModel


class _PatchEmbed:
    """Minimal patch embed proxy for pipeline compatibility."""

    def __init__(self, proj: nn.Module):
        self.proj = proj


class PhikonModel(nn.Module):
    """Wrapper around Phikon-v2 for pipeline compatibility."""

    def __init__(self):
        super().__init__()
        self.trunk = AutoModel.from_pretrained("owkin/phikon-v2")

    @property
    def num_features(self) -> int:
        return self.trunk.config.hidden_size

    @property
    def patch_embed(self):
        return _PatchEmbed(self.trunk.embeddings.patch_embeddings.projection)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.trunk(pixel_values=x)
        return outputs.last_hidden_state  # [B, 1+N, D]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_features(x)


def create_phikon_model() -> PhikonModel:
    """Create Phikon-v2 model with pretrained weights from HuggingFace."""
    return PhikonModel()
