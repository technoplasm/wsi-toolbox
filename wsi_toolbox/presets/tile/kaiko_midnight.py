"""Midnight-12k (kaiko.ai): DINOv2 ViT-G/14 (no registers, SwiGLU FFN) trained on TCGA, MIT license.

Not the same model as the ``midnight`` preset (SophontAI/OpenMidnight, a re-training with registers).
The checkpoint is a native transformers ``Dinov2Model`` (no remote code), so it loads from the HF cache with
``HF_HUB_OFFLINE=1``. Its position embedding is for 518 px (37x37) and is interpolated to the tile grid.

Embedding: the model card's classification embedding, CLS token concatenated with the mean of the patch
tokens (2 x 1536 = 3072), returned by ``forward``. It is not a single token, so the preset's ``extract_fn``
calls ``model(x)`` (no latent output).
"""

import torch
import torch.nn as nn
from transformers import Dinov2Model

HF_REPO = "kaiko-ai/midnight"


class _PatchEmbed:
    """Minimal patch embed proxy for pipeline compatibility."""

    def __init__(self, proj: nn.Module):
        self.proj = proj


class KaikoMidnightModel(nn.Module):
    """Wrapper around the transformers Dinov2Model of Midnight-12k."""

    def __init__(self):
        super().__init__()
        self.trunk = Dinov2Model.from_pretrained(HF_REPO)

    @property
    def num_features(self) -> int:
        return 2 * self.trunk.config.hidden_size

    @property
    def patch_embed(self):
        return _PatchEmbed(self.trunk.embeddings.patch_embeddings.projection)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.trunk(pixel_values=x).last_hidden_state  # [B, 1+N, D]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """concat(CLS, mean of patch tokens) -> [B, 2D] (the model card's classification embedding)."""
        tokens = self.forward_features(x)
        return torch.cat([tokens[:, 0], tokens[:, 1:].mean(dim=1)], dim=-1)


def create_kaiko_midnight_model() -> KaikoMidnightModel:
    """Create Midnight-12k with pretrained weights from HuggingFace."""
    return KaikoMidnightModel()
