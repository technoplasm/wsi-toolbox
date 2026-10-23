"""Slide-level aggregation presets (TITAN, GigaPath LongNet, etc.).

These take per-tile features+coordinates and produce a single slide-level vector.
"""

import logging

import h5py

from ..tile import canonical_tile_source

logger = logging.getLogger(__name__)


SLIDE_PRESET_NAMES = [
    "titan",
]

# Tile-preset compatibility: which tile feature spaces each slide preset accepts, as (tile preset, output).
# A group written before the ``output`` attr existed is mapped with ``canonical_tile_source`` first, so the
# old ``preset="conch15_768"`` groups count as ("conch15", "proj").
SLIDE_PRESET_TILE_SOURCES: dict[str, tuple[tuple[str, str], ...]] = {
    "titan": (("conch15", "proj"),),
}


def _group_source(grp: h5py.Group) -> tuple[str, str]:
    """``(preset, output)`` of a tile feature group from its attrs (old groups have no ``output``)."""
    preset = grp.attrs.get("preset", "")
    output = grp.attrs.get("output", "")
    return canonical_tile_source(str(preset), str(output) if output else None)


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
            (attrs["preset"], attrs["output"]) is in the compatible-source list for slide_preset
            (old groups without ``output``: ``preset="conch15_768"`` counts as ``("conch15", "proj")``).

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
            source = _group_source(grp)
            if source not in allowed:
                logger.warning(
                    f"Tile group '{explicit}' has (preset, output)={source}, "
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
            if _group_source(obj) in allowed:
                candidates.append(k)

        if not candidates:
            preset, output = allowed[0]
            raise RuntimeError(
                f"No tile features compatible with slide preset '{slide_preset}' "
                f"(required (preset, output): {allowed}). "
                f"Run 'wsi-toolbox extract --preset {preset} --outputs {output}' first."
            )
        if len(candidates) > 1:
            raise RuntimeError(
                f"Multiple compatible tile features found in {hdf5_path}: {sorted(candidates)}. "
                f"Specify --model explicitly."
            )

        logger.info(f"Auto-selected tile model: '{candidates[0]}' for slide preset '{slide_preset}'")
        return candidates[0]
