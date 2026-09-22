"""Shared fixtures: a tiny CPU model preset, a small PNG "slide", and an event-collecting sink.

No GPU, no model downloads.
"""

import numpy as np
import pytest
from PIL import Image

import wsi_toolbox as wt
from wsi_toolbox.presets.tile import TilePreset
from wsi_toolbox.progress import ProgressEvent

IMAGE_SIZE = 128
PATCH_SIZE = 32
GRID = IMAGE_SIZE // PATCH_SIZE  # 4x4 = 16 patches
FEATURE_DIM = 8


def _make_tiny_model():
    import torch  # noqa: PLC0415
    from torch import nn  # noqa: PLC0415

    class TinyViT(nn.Module):
        """ViT-shaped stub: forward_features returns (B, 1 + tokens, dim) with a CLS token first."""

        def __init__(self, dim: int = FEATURE_DIM, kernel: int = 16):
            super().__init__()
            self.patch_embed = nn.Module()
            self.patch_embed.proj = nn.Conv2d(3, dim, kernel_size=kernel, stride=kernel)
            self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))

        def forward_features(self, x):
            tokens = self.patch_embed.proj(x).flatten(2).transpose(1, 2)  # (B, N, dim)
            cls = self.cls_token.expand(tokens.shape[0], -1, -1) + tokens.mean(dim=1, keepdim=True)
            return torch.cat([cls, tokens], dim=1)

    return TinyViT()


class CollectSink:
    """Sink that records every ProgressEvent."""

    def __init__(self):
        self.events: list[ProgressEvent] = []

    def __call__(self, event: ProgressEvent) -> None:
        self.events.append(event)

    @property
    def phases(self) -> list[str]:
        """Distinct phase names in order of first appearance (excluding the done event)."""
        seen: list[str] = []
        for e in self.events:
            if e.done:
                continue
            if not seen or seen[-1] != e.phase:
                seen.append(e.phase)
        return seen


@pytest.fixture(autouse=True)
def quiet_defaults():
    """Keep tests silent and independent of process defaults."""
    saved = wt.defaults.model_copy()
    wt.set_default_progress(None)
    wt.defaults.preset = None
    wt.set_default_device("cpu")
    yield
    wt.defaults.preset = saved.preset
    wt.defaults.device = saved.device
    wt.defaults.progress = saved.progress


@pytest.fixture
def tiny_preset() -> TilePreset:
    return TilePreset(name="tiny", create_model=_make_tiny_model, norm_mean=(0.5, 0.5, 0.5), norm_std=(0.5, 0.5, 0.5))


@pytest.fixture
def png_path(tmp_path) -> str:
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 200, size=(IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    path = tmp_path / "slide.png"
    Image.fromarray(arr).save(path)
    return str(path)


@pytest.fixture
def collect() -> CollectSink:
    return CollectSink()
