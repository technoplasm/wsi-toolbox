"""Phikon (v1, iBOT ViT-B/16) and Phikon-v2 (DINOv2 ViT-L/16) by Owkin.

Both are native transformers models (``ViTModel`` / ``Dinov2Model``, no remote code), so they load from the
HF cache with ``HF_HUB_OFFLINE=1``. Embedding: the CLS token of ``last_hidden_state`` (after the final LayerNorm).
"""

import torch
import torch.nn as nn
from transformers import AutoModel, ViTModel

PHIKON_REPOS = {
    "v1": "owkin/phikon",
    "v2": "owkin/phikon-v2",
}


class _PatchEmbed:
    """Minimal patch embed proxy for pipeline compatibility."""

    def __init__(self, proj: nn.Module):
        self.proj = proj


class PhikonModel(nn.Module):
    """Wrapper around Phikon / Phikon-v2 for pipeline compatibility."""

    def __init__(self, version: str = "v2"):
        super().__init__()
        repo = PHIKON_REPOS[version]
        if version == "v1":
            # Plain ViT: the 224 px position embedding is interpolated for other tile sizes only when asked
            # (Dinov2Model always does). The pooler is loaded (the checkpoint has it) but not used.
            self.trunk = ViTModel.from_pretrained(repo)
            self._forward_kwargs = {"interpolate_pos_encoding": True}
        else:
            self.trunk = AutoModel.from_pretrained(repo)
            self._forward_kwargs = {}

    @property
    def num_features(self) -> int:
        return self.trunk.config.hidden_size

    @property
    def patch_embed(self):
        return _PatchEmbed(self.trunk.embeddings.patch_embeddings.projection)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.trunk(pixel_values=x, **self._forward_kwargs)
        return outputs.last_hidden_state  # [B, 1+N, D]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_features(x)


def create_phikon_model(version: str = "v2") -> PhikonModel:
    """Create Phikon (``"v1"``) or Phikon-v2 (``"v2"``) with pretrained weights from HuggingFace."""
    return PhikonModel(version)
