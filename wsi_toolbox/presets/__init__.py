"""Foundation model presets, organized by role.

Tile presets (per-patch encoders) live under ``presets.tile``.
Slide presets (slide-level aggregators) live under ``presets.slide``.
"""

from .slide import (
    SLIDE_PRESET_NAMES,
    SLIDE_PRESET_TILE_SOURCES,
    aggregate_slide,
    create_slide_preset_model,
)
from .tile import (
    PRESET_ALIASES,
    PRESET_EXTRACT_FN,
    PRESET_NAMES,
    PRESET_NORMALIZATION,
    TileHead,
    TilePreset,
    create_preset_model,
    get_tile_preset,
)

__all__ = [
    # Tile
    "PRESET_NAMES",
    "PRESET_ALIASES",
    "TileHead",
    "TilePreset",
    "get_tile_preset",
    "PRESET_NORMALIZATION",
    "PRESET_EXTRACT_FN",
    "create_preset_model",
    # Slide
    "SLIDE_PRESET_NAMES",
    "SLIDE_PRESET_TILE_SOURCES",
    "create_slide_preset_model",
    "aggregate_slide",
]
