"""Foundation model presets, organized by role.

Tile presets (per-patch encoders) live under ``presets.tile``.
Slide presets (slide-level aggregators) live under ``presets.slide``.
"""

from .slide import (
    SLIDE_PRESET_NAMES,
    SLIDE_PRESET_TILE_SOURCES,
    create_slide_preset_model,
)
from .tile import (
    PRESET_EXTRACT_FN,
    PRESET_NAMES,
    PRESET_NORMALIZATION,
    create_preset_model,
)

__all__ = [
    # Tile
    "PRESET_NAMES",
    "PRESET_NORMALIZATION",
    "PRESET_EXTRACT_FN",
    "create_preset_model",
    # Slide
    "SLIDE_PRESET_NAMES",
    "SLIDE_PRESET_TILE_SOURCES",
    "create_slide_preset_model",
]
