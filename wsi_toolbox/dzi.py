"""
Deep Zoom (DZI) serving: descriptor XML and tiles for any WSI toolbox can open.

``DziLayout`` is the pure geometry (levels, grid, tile rectangles); ``DziGenerator``
reads tiles for one opened WSI. Used by ``DziCommand`` (export to files) and by tile
servers (e.g. vision's compute-tiles) that answer ``{name}.dzi`` and
``{name}_files/{level}/{col}_{row}.{format}`` on demand::

    wsi = create_wsi_file("slide.pyramid.tif")      # a pyramid.tif (PyramidCommand) is fastest
    dzi = DziGenerator(wsi, tile_size=256, overlap=0)
    xml = dzi.xml()
    tile = dzi.tile(level, col, row)                 # RGB uint8 (H, W, 3)
    jpeg = encode_tile(tile, quality=90)

Geometry follows the Deep Zoom spec: level ``max_level = ceil(log2(max(W, H)))`` is full
resolution, each level below halves it (``ceil``), tiles are ``tile_size`` px plus
``overlap`` px on every side that has a neighbour. **Every tile has exactly the size the
spec gives** (``tile_rect``), whatever the source's native level sizes are.

How a tile is read (``DziGenerator.tile``):

1. Pick the coarsest native level whose downsample does not exceed the DZI level's
   (never upsample; 1 % tolerance because vendors round level sizes, e.g. SVS 4.0001).
2. If that native level *is* the DZI level (downsample within 1 %), copy the tile's pixels
   directly (no resampling). The native level may be a pixel narrower than the DZI level
   (``floor`` vs ``ceil`` when halving an odd size); missing edge pixels repeat the last
   row / column.
3. Otherwise read the tile's level-0 box from the finer native level and resize it to
   the tile size with Lanczos.

Thread safety: a ``DziGenerator`` holds no state besides the WSI; it is as thread-safe as
the WSI object (toolbox readers are one-thread-at-a-time; use one per thread).
"""

import math
from dataclasses import dataclass
from typing import Protocol

import imagecodecs
import numpy as np
from PIL import Image

# relative tolerance for "same downsample" (vendors round level sizes: 4.0001, 8.0031, ...)
_SCALE_TOLERANCE = 0.01


class DziTileNotFound(LookupError):
    """Level / column / row outside the DZI pyramid (a tile server answers 404)."""


class _NativeLevel(Protocol):
    width: int
    height: int
    downsample: float


class NativeReader(Protocol):
    """What ``DziGenerator`` needs from a WSI (``PyramidalWSIFile`` and ``StandardImage`` qualify)."""

    def get_original_size(self) -> tuple[int, int]: ...

    def _get_native_levels(self) -> list: ...

    def _read_native_region(self, level_idx: int, x: int, y: int, w: int, h: int) -> np.ndarray: ...


@dataclass(frozen=True)
class DziLayout:
    """Deep Zoom geometry of a ``width`` x ``height`` image."""

    width: int
    height: int
    tile_size: int = 256
    overlap: int = 0

    def __post_init__(self):
        if self.width < 1 or self.height < 1:
            raise ValueError(f"invalid image size {self.width}x{self.height}")
        if self.tile_size < 1 or self.overlap < 0:
            raise ValueError(f"invalid tile_size={self.tile_size} / overlap={self.overlap}")

    @property
    def max_level(self) -> int:
        """Full-resolution level (level 0 is 1x1)."""
        return math.ceil(math.log2(max(self.width, self.height)))

    @property
    def level_count(self) -> int:
        return self.max_level + 1

    def downsample(self, level: int) -> int:
        """Level-0 pixels per level pixel."""
        return 2 ** (self.max_level - level)

    def level_size(self, level: int) -> tuple[int, int]:
        """(width, height) of ``level``."""
        self._check_level(level)
        ds = self.downsample(level)
        return -(-self.width // ds), -(-self.height // ds)

    def grid(self, level: int) -> tuple[int, int]:
        """(cols, rows) of ``level``."""
        lw, lh = self.level_size(level)
        return -(-lw // self.tile_size), -(-lh // self.tile_size)

    def contains(self, level: int, col: int, row: int) -> bool:
        if not 0 <= level <= self.max_level:
            return False
        cols, rows = self.grid(level)
        return 0 <= col < cols and 0 <= row < rows

    def tile_rect(self, level: int, col: int, row: int) -> tuple[int, int, int, int]:
        """(x, y, w, h) of the tile in ``level`` pixels, overlap included."""
        if not self.contains(level, col, row):
            raise DziTileNotFound(f"no tile {col}_{row} at level {level}")
        lw, lh = self.level_size(level)
        ts, ov = self.tile_size, self.overlap
        x0 = col * ts - (ov if col > 0 else 0)
        y0 = row * ts - (ov if row > 0 else 0)
        x1 = min((col + 1) * ts + ov, lw)
        y1 = min((row + 1) * ts + ov, lh)
        return x0, y0, x1 - x0, y1 - y0

    def xml(self, format: str = "jpeg") -> str:
        """The ``.dzi`` descriptor."""
        return f'''<?xml version="1.0" encoding="utf-8"?>
<Image xmlns="http://schemas.microsoft.com/deepzoom/2008"
       Format="{format}"
       Overlap="{self.overlap}"
       TileSize="{self.tile_size}">
  <Size Width="{self.width}" Height="{self.height}"/>
</Image>'''

    def _check_level(self, level: int) -> None:
        if not 0 <= level <= self.max_level:
            raise DziTileNotFound(f"no level {level} (0..{self.max_level})")


class DziGenerator:
    """DZI descriptor and tiles of one opened WSI."""

    def __init__(self, wsi: NativeReader, tile_size: int = 256, overlap: int = 0, format: str = "jpeg"):
        self.wsi = wsi
        self.format = format
        width, height = wsi.get_original_size()
        self.layout = DziLayout(int(width), int(height), tile_size, overlap)

    @property
    def max_level(self) -> int:
        return self.layout.max_level

    def xml(self) -> str:
        return self.layout.xml(self.format)

    def tile(self, level: int, col: int, row: int) -> np.ndarray:
        """Tile ``(level, col, row)`` as RGB uint8 ``(h, w, 3)``; size is exactly ``layout.tile_rect``.

        Raises ``DziTileNotFound`` outside the pyramid.
        """
        x, y, w, h = self.layout.tile_rect(level, col, row)
        ds = self.layout.downsample(level)
        levels = self.wsi._get_native_levels()
        idx = _pick_native_level(levels, ds)
        native = levels[idx]

        if native.downsample >= ds / (1 + _SCALE_TOLERANCE):
            return self._copy_same_scale(idx, native, x, y, w, h)
        return self._resample(idx, native, ds, x, y, w, h)

    def _copy_same_scale(self, idx: int, native: _NativeLevel, x: int, y: int, w: int, h: int) -> np.ndarray:
        """The native level is this DZI level: copy pixels, repeat the last row / column if it is short."""
        rx0, ry0 = min(x, native.width - 1), min(y, native.height - 1)
        rx1, ry1 = max(min(x + w, native.width), rx0 + 1), max(min(y + h, native.height), ry0 + 1)
        region = self.wsi._read_native_region(idx, rx0, ry0, rx1 - rx0, ry1 - ry0)
        if region.shape[0] == h and region.shape[1] == w and rx0 == x and ry0 == y:
            return region
        rows = np.clip(np.arange(y, y + h), 0, region.shape[0] + ry0 - 1) - ry0
        cols = np.clip(np.arange(x, x + w), 0, region.shape[1] + rx0 - 1) - rx0
        return np.ascontiguousarray(region[np.ix_(rows, cols)])

    def _resample(self, idx: int, native: _NativeLevel, ds: int, x: int, y: int, w: int, h: int) -> np.ndarray:
        """Resize the tile's exact level-0 box, read from a finer native level, to ``(w, h)``.

        The box is passed to Lanczos with sub-pixel precision, with a margin of native pixels
        around it (the filter's support, so neighbouring tiles join without seams). Where the
        box runs past the image (the last DZI pixel may cover less than ``ds`` source pixels)
        the source is extended by repeating its edge.
        """
        sx, sy = _native_scale(native, self.layout.width, self.layout.height)
        fx0, fx1 = x * ds / sx, (x + w) * ds / sx  # tile box in native pixels
        fy0, fy1 = y * ds / sy, (y + h) * ds / sy
        mx = math.ceil(3 * ds / sx) + 1  # Lanczos-3 support at this reduction, + 1
        my = math.ceil(3 * ds / sy) + 1
        nx0, ny0 = max(0, math.floor(fx0) - mx), max(0, math.floor(fy0) - my)
        nx1 = min(native.width, math.ceil(fx1) + mx)
        ny1 = min(native.height, math.ceil(fy1) + my)
        region = self.wsi._read_native_region(idx, nx0, ny0, nx1 - nx0, ny1 - ny0)
        pad_x, pad_y = max(0, math.ceil(fx1) - nx1), max(0, math.ceil(fy1) - ny1)
        if pad_x or pad_y:
            region = np.pad(region, ((0, pad_y), (0, pad_x), (0, 0)), mode="edge")
        box = (fx0 - nx0, fy0 - ny0, fx1 - nx0, fy1 - ny0)
        return np.asarray(Image.fromarray(region).resize((w, h), Image.Resampling.LANCZOS, box=box))

    def iter_tiles(self):
        """Yield ``(level, col, row, tile)`` for every tile, full resolution first."""
        for level in range(self.max_level, -1, -1):
            cols, rows = self.layout.grid(level)
            for row in range(rows):
                for col in range(cols):
                    yield level, col, row, self.tile(level, col, row)


def _native_scale(native: _NativeLevel, width: int, height: int) -> tuple[float, float]:
    """Level-0 pixels per native pixel on each axis.

    A level whose downsample is within 1 % of a power of two is taken to be exactly that
    (vendors and libvips round odd sizes down: SVS reports 4.0001, a 1001 px image halves to
    500 px); anything else uses the measured per-axis ratio.
    """
    nominal = 2 ** round(math.log2(native.downsample)) if native.downsample > 0 else 1
    if abs(native.downsample - nominal) <= nominal * _SCALE_TOLERANCE:
        return float(nominal), float(nominal)
    return width / native.width, height / native.height


def _pick_native_level(levels: list, target_downsample: float) -> int:
    """Coarsest native level that is not coarser than ``target_downsample`` (never upsample)."""
    limit = target_downsample * (1 + _SCALE_TOLERANCE)
    best = 0
    for i, lv in enumerate(levels):
        if lv.downsample <= limit and lv.downsample > levels[best].downsample:
            best = i
    return best


def encode_tile(tile: np.ndarray, format: str = "jpeg", quality: int = 90) -> bytes:
    """Encode an RGB tile as JPEG (libjpeg-turbo via imagecodecs) or PNG."""
    tile = np.ascontiguousarray(tile)
    if format == "png":
        return bytes(imagecodecs.png_encode(tile))
    if format in ("jpeg", "jpg"):
        return bytes(imagecodecs.jpeg8_encode(tile, level=quality))
    raise ValueError(f"unsupported tile format {format!r} (jpeg or png)")
