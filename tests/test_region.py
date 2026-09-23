"""wsi_toolbox.region: reads at a given µm/px."""

from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

import wsi_toolbox as wt
from tests.conftest import _smooth_rgb
from wsi_toolbox.region import pick_native_level
from wsi_toolbox.wsi_files import create_wsi_file


def _levels(*downsamples):
    return [SimpleNamespace(width=1, height=1, downsample=d) for d in downsamples]


@pytest.mark.parametrize(
    "downsamples, target, idx",
    [
        ((1, 2, 4), 1.0, 0),
        ((1, 2, 4), 1.9, 0),
        ((1, 2, 4), 2.0, 1),
        ((1, 2, 4), 3.9, 1),
        ((1, 4.0001, 16.002), 4.0, 1),  # vendor rounding within 1 %
        ((1, 4.05, 16), 4.0, 0),  # beyond 1 %: never upsample from a coarser level
        ((1, 2, 4), 100.0, 2),
        ((1, 2, 2), 2.0, 1),  # equal downsamples: first wins
    ],
)
def test_pick_native_level(downsamples, target, idx):
    assert pick_native_level(_levels(*downsamples), target) == idx


def test_output_size_and_native_copy(pyramid_tiff):
    wsi = create_wsi_file(pyramid_tiff)  # 1000 x 700, 0.5 µm/px
    img = wt.read_region_at_mpp(wsi, 100, 50, 400, 300, 0.5)
    assert img.shape == (300, 400, 3) and img.dtype == np.uint8
    ref = wsi._read_native_region(0, 100, 50, 400, 300)
    assert np.abs(img.astype(int) - ref.astype(int)).max() <= 2  # 1:1 Lanczos is (nearly) a copy


@pytest.mark.parametrize("target_mpp, size", [(1.0, (200, 150)), (1.3, (154, 115)), (0.25, (800, 600))])
def test_matches_reference_resize(pyramid_tiff, target_mpp, size):
    wsi = create_wsi_file(pyramid_tiff)
    x, y, w, h = 200, 100, 400, 300
    img = wt.read_region_at_mpp(wsi, x, y, w, h, target_mpp)
    assert img.shape == (size[1], size[0], 3)
    ref = np.asarray(
        Image.fromarray(_smooth_rgb(1000, 700)[y : y + h, x : x + w]).resize(size, Image.Resampling.LANCZOS)
    )
    assert np.abs(img.astype(float) - ref.astype(float)).mean() < 3  # JPEG + level choice


def test_tile_size_only_rounds(pyramid_tiff):
    """No seams: squares of another size differ at most by PIL's rounding (±1)."""
    wsi = create_wsi_file(pyramid_tiff)
    a = wt.read_region_at_mpp(wsi, 13, 7, 900, 650, 0.8)
    b = wt.read_region_at_mpp(wsi, 13, 7, 900, 650, 0.8, tile=97)
    assert np.abs(a.astype(int) - b.astype(int)).max() <= 1


def test_mpp_override_and_standard_image(png_path):
    wsi = create_wsi_file(png_path, mpp=0.5)
    img = wt.read_region_at_mpp(wsi, 0, 0, 64, 64, 1.0, mpp=0.25)  # the caller's mpp wins: 4x reduction
    assert img.shape == (16, 16, 3)


@pytest.mark.parametrize("kwargs", [dict(target_mpp=0), dict(target_mpp=1.0, mpp=0), dict(target_mpp=1.0, w=0)])
def test_rejects_bad_input(pyramid_tiff, kwargs):
    wsi = create_wsi_file(pyramid_tiff)
    args = dict(x=0, y=0, w=10, h=10) | kwargs
    with pytest.raises(ValueError):
        wt.read_region_at_mpp(wsi, **args)
