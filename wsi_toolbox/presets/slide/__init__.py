"""Slide-level aggregation presets (TITAN, GigaPath LongNet, etc.).

These take per-tile features+coordinates and produce a single slide-level vector.
"""

import logging

import h5py

logger = logging.getLogger(__name__)


SLIDE_PRESET_NAMES = [
    "titan",
]

# Tile-preset compatibility: which tile feature spaces each slide preset accepts.
SLIDE_PRESET_TILE_SOURCES: dict[str, tuple[str, ...]] = {
    "titan": ("conch15_768",),
}


def create_slide_preset_model(preset: str):
    """Instantiate a slide-level aggregator by preset name.

    Returns the bare model (caller is responsible for .to(device).eval()).
    """
    if preset == "titan":
        from .titan import create_titan_model  # noqa: PLC0415

        return create_titan_model()

    raise ValueError(f"Invalid slide preset: {preset}. Must be one of {SLIDE_PRESET_NAMES}")


def resolve_tile_model(
    hdf5_path: str,
    slide_preset: str,
    explicit: str | None = None,
) -> str:
    """Pick the tile model to feed into a slide preset.

    Args:
        hdf5_path: Path to the HDF5 file to scan.
        slide_preset: Name of the slide aggregator (e.g., "titan").
        explicit: When set, return as-is (just warn if attrs/preset mismatches).
            When None, scan top-level groups in the file for one whose
            attrs["preset"] is in the compatible-source list for slide_preset.

    Returns:
        Tile model (the h5 top-level storage key).

    Raises:
        RuntimeError: 0 compatible groups (need to extract first), or
            multiple groups (user must specify --model).
    """
    if slide_preset not in SLIDE_PRESET_NAMES:
        raise ValueError(f"Unknown slide preset: {slide_preset}. Must be one of {SLIDE_PRESET_NAMES}")

    allowed = SLIDE_PRESET_TILE_SOURCES.get(slide_preset, ())
    if not allowed:
        raise RuntimeError(f"Slide preset '{slide_preset}' has no registered tile sources")

    with h5py.File(hdf5_path, "r") as f:
        if explicit is not None:
            grp = f.get(explicit)
            if grp is None or not isinstance(grp, h5py.Group) or "features" not in grp:
                raise RuntimeError(f"Tile model '{explicit}' not found or has no features in {hdf5_path}")
            preset_attr = grp.attrs.get("preset", "")
            if preset_attr not in allowed:
                logger.warning(
                    f"Tile group '{explicit}' has preset='{preset_attr}', "
                    f"but slide preset '{slide_preset}' expects {allowed}. Proceeding anyway."
                )
            return explicit

        # Auto-resolve: scan top-level groups for compatible preset
        candidates = []
        for k in f.keys():
            obj = f[k]
            if not isinstance(obj, h5py.Group):
                continue
            if "features" not in obj:
                continue
            if obj.attrs.get("preset", "") in allowed:
                candidates.append(k)

        if not candidates:
            raise RuntimeError(
                f"No tile features compatible with slide preset '{slide_preset}' "
                f"(required preset: {allowed}). Run 'wsi-toolbox extract --preset {allowed[0]}' first."
            )
        if len(candidates) > 1:
            raise RuntimeError(
                f"Multiple compatible tile features found in {hdf5_path}: {sorted(candidates)}. "
                f"Specify --model explicitly."
            )

        logger.info(f"Auto-selected tile model: '{candidates[0]}' for slide preset '{slide_preset}'")
        return candidates[0]
