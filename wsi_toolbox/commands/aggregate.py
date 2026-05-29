"""Slide-level aggregation command.

Loads tile features+coordinates from a tile-preset group and runs a slide-level
aggregator (TITAN, etc.) to produce one vector per WSI. Writes the result to:

    {tile_model_name}/aggregates/{slide_preset}/feature
"""

import logging

import h5py
import numpy as np
import torch
from pydantic import BaseModel

from ..presets.slide import SLIDE_PRESET_NAMES, create_slide_preset_model
from ..utils import safe_del
from . import _get

logger = logging.getLogger(__name__)


class AggregateResult(BaseModel):
    slide_preset: str
    tile_model_name: str
    target_path: str
    feature_dim: int
    n_patches: int
    skipped: bool = False


class AggregateCommand:
    """Run a slide-level aggregator on tile features in an HDF5 file."""

    def __init__(
        self,
        slide_preset: str,
        tile_model_name: str,
        device: str | None = None,
        overwrite: bool = False,
    ):
        """
        Args:
            slide_preset: Slide aggregator preset (e.g., "titan")
            tile_model_name: HDF5 storage key for the tile features to aggregate.
                Resolve compatibility/availability at the call site (see
                ``wsi_toolbox.presets.slide.resolve_tile_model_name``).
            device: Inference device. None uses global default.
            overwrite: Replace existing slide feature if present.
        """
        if slide_preset not in SLIDE_PRESET_NAMES:
            raise ValueError(f"Unknown slide preset: {slide_preset}. Must be one of {SLIDE_PRESET_NAMES}")
        self.slide_preset = slide_preset
        self.tile_model_name = tile_model_name
        self.device = _get("device", device)
        self.overwrite = overwrite

    def __call__(self, hdf5_path: str) -> AggregateResult:
        target_path = f"{self.tile_model_name}/aggregates/{self.slide_preset}/feature"

        # 1. Load inputs and check skip
        with h5py.File(hdf5_path, "r") as f:
            tile_grp = f.get(self.tile_model_name)
            if tile_grp is None or "features" not in tile_grp:
                raise RuntimeError(
                    f"Tile model_name '{self.tile_model_name}' not found or has no features in {hdf5_path}"
                )

            if not self.overwrite and target_path in f:
                existing = f[target_path]
                logger.info(f"Aggregate already exists at {target_path}, skipping")
                return AggregateResult(
                    slide_preset=self.slide_preset,
                    tile_model_name=self.tile_model_name,
                    target_path=target_path,
                    feature_dim=int(existing.shape[-1]),
                    n_patches=int(tile_grp["features"].shape[0]),
                    skipped=True,
                )

            features = tile_grp["features"][:]
            coords = tile_grp["coordinates"][:]
            patch_size_lv0 = int(tile_grp.attrs["patch_size"])

        logger.info(
            f"Aggregating {features.shape[0]} patches (D={features.shape[1]}) "
            f"with slide preset '{self.slide_preset}', patch_size_lv0={patch_size_lv0}"
        )

        # 2. Load slide aggregator
        device_spec = self.device or "auto"
        if device_spec == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = device_spec
        logger.info(f"Using device: {device}")

        model = create_slide_preset_model(self.slide_preset).to(device).eval()

        # 3. Run aggregation
        feat_t = torch.from_numpy(features).float().unsqueeze(0).to(device)
        coord_t = torch.from_numpy(coords).long().unsqueeze(0).to(device)

        with torch.inference_mode():
            slide_emb = model.encode_slide_from_patch_features(feat_t, coord_t, patch_size_lv0)
        slide_emb_np = slide_emb.squeeze(0).cpu().numpy().astype(np.float32)

        # 4. Cleanup
        del model
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

        # 5. Write
        with h5py.File(hdf5_path, "a") as f:
            if self.overwrite:
                safe_del(f, target_path)
            f.create_dataset(target_path, data=slide_emb_np)

        logger.info(f"Wrote {target_path} (D={slide_emb_np.shape[-1]})")

        return AggregateResult(
            slide_preset=self.slide_preset,
            tile_model_name=self.tile_model_name,
            target_path=target_path,
            feature_dim=int(slide_emb_np.shape[-1]),
            n_patches=int(features.shape[0]),
        )
