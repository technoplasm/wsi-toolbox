"""wsi_toolbox.dzi: Deep Zoom geometry and tiles."""

import math

import numpy as np
import pytest
from PIL import Image

from tests.conftest import _smooth_rgb
from wsi_toolbox.dzi import DziGenerator, DziLayout, DziTileNotFound, encode_tile
from wsi_toolbox.wsi_files import create_wsi_file

LEGACY_XML = """<?xml version="1.0" encoding="utf-8"?>
<Image xmlns="http://schemas.microsoft.com/deepzoom/2008"
       Format="jpeg"
       Overlap="0"
       TileSize="256">
  <Size Width="1000" Height="700"/>
</Image>"""


def spec_tile_size(width, height, tile_size, overlap, level, col, row):
    """(w, h) of a tile straight from the Deep Zoom spec."""
    max_level = math.ceil(math.log2(max(width, height)))
    scale = 2 ** (max_level - level)
    lw, lh = math.ceil(width / scale), math.ceil(height / scale)
    x0 = col * tile_size - (overlap if col else 0)
    y0 = row * tile_size - (overlap if row else 0)
    return min((col + 1) * tile_size + overlap, lw) - x0, min((row + 1) * tile_size + overlap, lh) - y0


# --- geometry --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "size, max_level", [((1, 1), 0), ((2, 1), 1), ((1000, 700), 10), ((1024, 1024), 10), ((1025, 3), 11)]
)
def test_max_level(size, max_level):
    assert DziLayout(*size).max_level == max_level


def test_level_sizes_and_grid():
    layout = DziLayout(15374, 17497, 256, 0)  # odd sizes (a real JP2K SVS)
    assert layout.max_level == 15
    assert layout.level_size(15) == (15374, 17497)
    assert layout.level_size(14) == (7687, 8749)  # ceil
    assert layout.level_size(0) == (1, 1)
    assert layout.grid(15) == (61, 69)
    assert layout.grid(0) == (1, 1)


@pytest.mark.parametrize("overlap", [0, 1, 2])
def test_tile_rect_matches_spec(overlap):
    layout = DziLayout(1001, 699, 254, overlap)
    for level in range(layout.max_level + 1):
        cols, rows = layout.grid(level)
        for col in range(cols):
            for row in range(rows):
                _, _, w, h = layout.tile_rect(level, col, row)
                assert (w, h) == spec_tile_size(1001, 699, 254, overlap, level, col, row)


@pytest.mark.parametrize("level, col, row", [(-1, 0, 0), (11, 0, 0), (10, 4, 0), (10, 0, 3), (10, -1, 0)])
def test_out_of_range_raises(pyramid_tiff, level, col, row):
    dzi = DziGenerator(create_wsi_file(pyramid_tiff))
    with pytest.raises(DziTileNotFound):
        dzi.tile(level, col, row)
    assert not dzi.layout.contains(level, col, row)


def test_xml_is_unchanged(pyramid_tiff):
    wsi = create_wsi_file(pyramid_tiff)
    assert DziGenerator(wsi).xml() == LEGACY_XML
    assert wsi.get_dzi_xml(256, 0, "jpeg") == LEGACY_XML


# --- tiles -----------------------------------------------------------------------------------


@pytest.mark.parametrize("fixture", ["pyramid_tiff", "odd_pyramid_tiff"])
@pytest.mark.parametrize("tile_size, overlap", [(256, 0), (254, 1)])
def test_every_tile_has_spec_size(request, fixture, tile_size, overlap):
    wsi = create_wsi_file(request.getfixturevalue(fixture))
    width, height = wsi.get_original_size()
    dzi = DziGenerator(wsi, tile_size, overlap)
    n = 0
    for level, col, row, tile in dzi.iter_tiles():
        assert tile.dtype == np.uint8 and tile.ndim == 3 and tile.shape[2] == 3
        assert tile.shape[1::-1] == spec_tile_size(width, height, tile_size, overlap, level, col, row)
        n += 1
    assert n == sum(math.prod(dzi.layout.grid(lv)) for lv in range(dzi.max_level + 1))


def _reference_tile(base: np.ndarray, layout: DziLayout, level: int, col: int, row: int) -> np.ndarray:
    """Tile cut from the full-resolution image scaled by exactly 1/downsample (no pyramid involved).

    The source is edge-padded to ``level_size * downsample`` first, so a level pixel always
    covers ``downsample`` source pixels, as in the Deep Zoom spec.
    """
    lw, lh = layout.level_size(level)
    ds = layout.downsample(level)
    padded = np.pad(base, ((0, lh * ds - base.shape[0]), (0, lw * ds - base.shape[1]), (0, 0)), mode="edge")
    scaled = padded if ds == 1 else np.asarray(Image.fromarray(padded).resize((lw, lh), Image.Resampling.LANCZOS))
    x, y, w, h = layout.tile_rect(level, col, row)
    return scaled[y : y + h, x : x + w]


@pytest.mark.parametrize("fixture, size", [("pyramid_tiff", (1000, 700)), ("odd_pyramid_tiff", (1001, 699))])
def test_tile_pixels_match_reference(request, fixture, size):
    """Tiles from the JPEG pyramid equal a direct downscale of the source within JPEG tolerance."""
    base = _smooth_rgb(*size)
    dzi = DziGenerator(create_wsi_file(request.getfixturevalue(fixture)), 256, 0)
    for level in range(dzi.max_level, dzi.max_level - 6, -1):
        cols, rows = dzi.layout.grid(level)
        for col, row in {(0, 0), (cols - 1, rows - 1), (cols // 2, rows // 2)}:
            got = dzi.tile(level, col, row).astype(np.int16)
            want = _reference_tile(base, dzi.layout, level, col, row).astype(np.int16)
            assert got.shape == want.shape
            assert np.abs(got - want).mean() < 3.5, (level, col, row)


def test_same_scale_tiles_are_exact_copies(pyramid_tiff):
    """Where a native level is the DZI level, tiles are the native pixels (no resampling)."""
    wsi = create_wsi_file(pyramid_tiff)
    dzi = DziGenerator(wsi)
    for native_idx, level in enumerate((10, 9, 8)):
        lw, lh = dzi.layout.level_size(level)
        cols, rows = dzi.layout.grid(level)
        for col in range(cols):
            for row in range(rows):
                x, y, w, h = dzi.layout.tile_rect(level, col, row)
                assert np.array_equal(dzi.tile(level, col, row), wsi._read_native_region(native_idx, x, y, w, h))


def test_short_native_level_repeats_edge(odd_pyramid_tiff):
    """1001 x 699: DZI level 9 is 501 x 350, the native 2x level 500 x 349."""
    wsi = create_wsi_file(odd_pyramid_tiff)
    dzi = DziGenerator(wsi)
    assert dzi.layout.level_size(9) == (501, 350)
    assert (wsi._levels[1].width, wsi._levels[1].height) == (500, 349)
    tile = dzi.tile(9, 1, 1)  # bottom-right: x 256..501, y 256..350
    assert tile.shape == (94, 245, 3)
    assert np.array_equal(tile[:, -1], tile[:, -2])  # repeated last column
    assert np.array_equal(tile[-1], tile[-2])  # repeated last row
    assert np.array_equal(tile[:-1, :-1], wsi._read_native_region(1, 256, 256, 244, 93))


def test_legacy_wrappers_delegate(pyramid_tiff):
    wsi = create_wsi_file(pyramid_tiff)
    dzi = DziGenerator(wsi, 256, 0)
    assert wsi.get_dzi_max_level() == dzi.max_level
    assert wsi.get_dzi_level_info(9, 256) == (500, 350, 2, 2)
    assert np.array_equal(wsi.get_dzi_tile(9, 1, 1, 256, 0), dzi.tile(9, 1, 1))


def test_encode_tile_roundtrip():
    tile = _smooth_rgb(200, 100)
    jpeg = encode_tile(tile, quality=90)
    assert jpeg[:2] == b"\xff\xd8"
    png = encode_tile(tile, format="png")
    assert png[:4] == b"\x89PNG"
    with pytest.raises(ValueError):
        encode_tile(tile, format="webp")
