"""PyramidalTiffFile: the direct tile reader returns exactly what tifffile's zarr store returns."""

import random

import numpy as np
import zarr

from wsi_toolbox.wsi_files import PyramidalTiffFile, create_wsi_file


def _zarr_read(wsi: PyramidalTiffFile, level_idx: int, x: int, y: int, w: int, h: int) -> np.ndarray:
    lv = wsi._levels[level_idx]
    z = zarr.open(wsi.tif.pages[lv.index].aszarr(), mode="r")
    x = max(0, min(x, lv.width - 1))
    y = max(0, min(y, lv.height - 1))
    w = min(w, lv.width - x)
    h = min(h, lv.height - y)
    return wsi._normalize_color(z[y : y + h, x : x + w])


def test_opens_as_pyramidal_tiff(pyramid_tiff):
    wsi = create_wsi_file(pyramid_tiff)
    assert isinstance(wsi, PyramidalTiffFile)
    assert [lv.downsample for lv in wsi._get_native_levels()] == [1.0, 2.0, 4.0]
    assert wsi.get_mpp() == 0.5


def test_direct_tile_read_matches_zarr(pyramid_tiff):
    wsi = PyramidalTiffFile(pyramid_tiff)
    rng = random.Random(0)
    for li, lv in enumerate(wsi._levels):
        cases = [
            (0, 0, lv.width, lv.height),  # whole level
            (lv.width - 5, lv.height - 7, 64, 64),  # clipped at the far edge
            (127, 127, 2, 2),  # straddles four tiles
            (-10, -10, 50, 50),  # negative origin is clamped
        ]
        cases += [
            (rng.randrange(lv.width), rng.randrange(lv.height), rng.randint(1, 400), rng.randint(1, 400))
            for _ in range(40)
        ]
        for x, y, w, h in cases:
            got = wsi._read_native_region(li, x, y, w, h)
            want = _zarr_read(wsi, li, x, y, w, h)
            assert got.dtype == np.uint8
            assert got.shape == want.shape, (li, x, y, w, h)
            assert np.array_equal(got, want), (li, x, y, w, h)


def test_read_region_matches_zarr(pyramid_tiff):
    wsi = PyramidalTiffFile(pyramid_tiff)
    for x, y, w, h in [(0, 0, 300, 200), (900, 650, 300, 300), (130, 5, 1, 1)]:
        assert np.array_equal(wsi.read_region((x, y, w, h)), _zarr_read(wsi, 0, x, y, w, h))
