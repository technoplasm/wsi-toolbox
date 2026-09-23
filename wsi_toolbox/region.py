"""
Region reads at an arbitrary resolution (µm/px), for exports and figures.

``read_region_at_mpp`` renders a level-0 box of a WSI at ``target_mpp``::

    wsi = create_wsi_file("slide.pyramid.tif")
    img = read_region_at_mpp(wsi, x=12000, y=8000, w=4000, h=3000, target_mpp=0.5)   # RGB uint8 (1500 x 2000 for 0.25 µm/px)

How it reads (the same level choice as ``wsi_toolbox.dzi``):

1. Pick the coarsest native level whose downsample does not exceed the output's
   (level-0 px per output px), with a 1 % tolerance because vendors round level sizes
   (SVS reports 4.0001). So a level a hair coarser than the output is used rather than
   one a whole power of two finer.
2. Render the output in ``tile`` px squares. For each, read the square's footprint on that
   native level plus the Lanczos-3 support as a margin, and resize the sub-pixel box with
   PIL's Lanczos (no seams between squares).

Outputs finer than level 0 are upsampled from level 0 (not refused). Memory is the output
array plus one source square.
"""

import math
from typing import Protocol

import numpy as np
from PIL import Image

# relative tolerance for "same downsample" (vendors round level sizes: 4.0001, 8.0031, ...)
SCALE_TOLERANCE = 0.01


class _NativeLevel(Protocol):
    width: int
    height: int
    downsample: float


def pick_native_level(levels: list, target_downsample: float) -> int:
    """Index of the coarsest native level not coarser than ``target_downsample`` (1 % tolerance).

    ``levels`` are ``NativeLevel``-like (``downsample`` relative to level 0), level 0 first.
    Among levels with the same downsample the first wins.
    """
    limit = target_downsample * (1 + SCALE_TOLERANCE)
    best = 0
    for i, lv in enumerate(levels):
        if lv.downsample <= limit and lv.downsample > levels[best].downsample:
            best = i
    return best


def read_region_at_mpp(
    wsi,
    x: int,
    y: int,
    w: int,
    h: int,
    target_mpp: float,
    *,
    mpp: float | None = None,
    tile: int = 2048,
) -> np.ndarray:
    """The level-0 box ``(x, y, w, h)`` of ``wsi`` rendered at ``target_mpp`` µm/px.

    Args:
        wsi: any opened ``WSIFile`` (pyramidal readers use their native levels; a reader
            without them is read at level 0 via ``read_region``).
        x, y, w, h: box in level-0 px. Must lie inside the slide (clamp it first).
        target_mpp: output µm/px.
        mpp: level-0 µm/px; ``None`` → ``wsi.get_mpp()`` (pass it when the caller knows
            better, e.g. a value stored in a database).
        tile: output px rendered per read (bounds memory; other sizes change pixels by
            at most ±1 from PIL's rounding, never seams).

    Returns:
        RGB uint8 ``(out_h, out_w, 3)`` with ``out_w = max(1, round(w * mpp / target_mpp))``
        (``out_h`` likewise).

    Raises:
        ValueError: ``mpp`` / ``target_mpp`` not positive, or an empty box.
    """
    if mpp is None:
        mpp = wsi.get_mpp()
    if not (mpp and mpp > 0 and math.isfinite(mpp)) or not (target_mpp > 0 and math.isfinite(target_mpp)):
        raise ValueError(f"mpp and target_mpp must be positive (mpp={mpp}, target_mpp={target_mpp})")
    if w <= 0 or h <= 0:
        raise ValueError(f"empty region: {w} x {h}")
    level0_per_out = target_mpp / mpp
    out_w = max(1, round(w / level0_per_out))
    out_h = max(1, round(h / level0_per_out))

    get_levels = getattr(wsi, "_get_native_levels", None)
    if get_levels is not None:
        levels = get_levels()
        idx = pick_native_level(levels, level0_per_out)
        down = levels[idx].downsample
        lw, lh = levels[idx].width, levels[idx].height

        def read(lx: int, ly: int, rw: int, rh: int) -> np.ndarray:
            return wsi._read_native_region(idx, lx, ly, rw, rh)
    else:
        down = 1.0
        lw, lh = wsi.get_original_size()

        def read(lx: int, ly: int, rw: int, rh: int) -> np.ndarray:
            return wsi.read_region((lx, ly, rw, rh))

    out = np.empty((out_h, out_w, 3), dtype=np.uint8)
    src_per_out = level0_per_out / down  # native px per output px
    margin = math.ceil(3 * max(1.0, src_per_out)) + 2  # Lanczos-3 support in native px, + 2
    for oy in range(0, out_h, tile):
        th = min(tile, out_h - oy)
        for ox in range(0, out_w, tile):
            tw = min(tile, out_w - ox)
            # the output square's footprint in native px (float)
            fx0 = (x + ox * level0_per_out) / down
            fy0 = (y + oy * level0_per_out) / down
            fx1 = (x + (ox + tw) * level0_per_out) / down
            fy1 = (y + (oy + th) * level0_per_out) / down
            lx0 = max(0, math.floor(fx0) - margin)
            ly0 = max(0, math.floor(fy0) - margin)
            lx1 = min(lw, math.ceil(fx1) + margin)
            ly1 = min(lh, math.ceil(fy1) + margin)
            src = np.ascontiguousarray(read(lx0, ly0, lx1 - lx0, ly1 - ly0)[..., :3])
            box = (fx0 - lx0, fy0 - ly0, min(fx1, lx1) - lx0, min(fy1, ly1) - ly0)
            square = Image.fromarray(src).resize((tw, th), Image.Resampling.LANCZOS, box=box)
            out[oy : oy + th, ox : ox + tw] = np.asarray(square)
    return out
