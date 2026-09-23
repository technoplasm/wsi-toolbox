"""The ptp white check keeps exactly the old keep/drop decisions (np.ptp reference)."""

import os

import numpy as np
import pytest

from wsi_toolbox.patch_reader import WSIPatchReader
from wsi_toolbox.utils.white import _count_low_range_pixels, create_white_detector, is_white_patch_ptp
from wsi_toolbox.wsi_files import create_wsi_file

S = 64


def reference_ptp(patch, white_ratio_threshold=0.9, rgb_range_threshold=20):
    """The implementation before 0.6 (np.ptp over the channel axis)."""
    rgb_range = np.ptp(patch, axis=2)
    white_pixels = np.sum(rgb_range < rgb_range_threshold)
    return white_pixels / (patch.shape[0] * patch.shape[1]) > white_ratio_threshold


def _patch_with_white_count(n_white: int, low: int, high: int, rng) -> np.ndarray:
    """S x S patch where exactly ``n_white`` pixels have channel range ``low`` and the rest ``high``."""
    base = rng.integers(0, 256 - max(low, high), size=(S * S, 1), dtype=np.int32)
    rng_col = np.where(np.arange(S * S)[:, None] < n_white, low, high)
    px = np.concatenate([base, base + rng_col, base + rng_col // 2], axis=1)
    rng.shuffle(px)
    return px.reshape(S, S, 3).astype(np.uint8)


def _synthetic_patches():
    rng = np.random.default_rng(0)
    total = S * S
    patches = []
    # around the ratio threshold (0.9 of the pixels) and the range threshold (19 / 20 / 21)
    for n_white in (int(total * 0.9) - 1, int(total * 0.9), int(total * 0.9) + 1, 0, total):
        for low in (0, 19, 20, 21):
            for high in (20, 21, 255):
                patches.append(_patch_with_white_count(n_white, low, high, rng))
    # extremes and uniform noise
    patches += [np.zeros((S, S, 3), np.uint8), np.full((S, S, 3), 255, np.uint8)]
    patches += [rng.integers(0, 256, (S, S, 3), dtype=np.uint8) for _ in range(50)]
    patches += [rng.integers(230, 256, (S, S, 3), dtype=np.uint8) for _ in range(50)]  # near white
    # strided views, as WSIPatchReader passes slices of a row strip
    strip = rng.integers(200, 256, (S, S * 8, 3), dtype=np.uint8)
    patches += [strip[:, i * S : (i + 1) * S, :] for i in range(8)]
    return patches


@pytest.mark.parametrize("threshold", [0.5, 0.75, 0.8, 0.9, 0.95, 0.999])
def test_ptp_matches_reference_on_synthetic_patches(threshold):
    for patch in _synthetic_patches():
        assert is_white_patch_ptp(patch, threshold) == reference_ptp(patch, threshold)


@pytest.mark.parametrize("rgb_range_threshold", [0, 1, 20, 20.5, 255, 256])
def test_pixel_count_matches_ptp(rgb_range_threshold):
    for patch in _synthetic_patches():
        expected = int(np.sum(np.ptp(patch, axis=2) < rgb_range_threshold))
        assert _count_low_range_pixels(patch, rgb_range_threshold) == expected


def test_non_uint8_falls_back_to_ptp():
    rng = np.random.default_rng(1)
    for dtype in (np.float32, np.uint16):
        patch = (rng.random((S, S, 3)) * 40).astype(dtype)
        assert is_white_patch_ptp(patch) == reference_ptp(patch)
    rgba = rng.integers(0, 256, (S, S, 4), dtype=np.uint8)
    assert is_white_patch_ptp(rgba) == reference_ptp(rgba)


@pytest.mark.skipif(not os.environ.get("WT_TEST_WSI"), reason="set WT_TEST_WSI=/path/to/slide to compare on a real WSI")
def test_ptp_matches_reference_on_real_slide():
    """Every patch of the first rows of a real slide gets the same keep/drop decision."""
    wsi = create_wsi_file(os.environ["WT_TEST_WSI"])
    reader = WSIPatchReader(wsi, patch_size=256, target_mpp=0.5)
    detector = create_white_detector("ptp")
    checked = 0
    for patches, _coords, _ in reader.iter_rows(rows_per_read=4):
        for patch in patches:
            assert detector(patch) == reference_ptp(patch)
            checked += 1
        if checked > 4000:
            break
    assert checked > 0
