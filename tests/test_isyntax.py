"""ISyntaxFile (Philips iSyntax via pyisyntax).

The reader tests need the sample slide ``data/isyntax/testslide.isyntax`` (Zenodo 5037046, MIT,
495 MB; ``data/`` is not in git) or ``WT_ISYNTAX_SAMPLE=<path>``, and are skipped without it.
"""

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from wsi_toolbox.patch_reader import WSIPatchReader
from wsi_toolbox.region import read_region_at_mpp
from wsi_toolbox.wsi_files import WSI_EXTENSIONS, ISyntaxFile, _rgba_to_rgb, create_wsi_file, find_wsi_for_h5

DEFAULT_SAMPLE = Path(__file__).resolve().parents[1] / "data" / "isyntax" / "testslide.isyntax"
DEFAULT_SAMPLE_BYTES = 495125898  # Zenodo 5037046 testslide.isyntax; smaller = download still running
SAMPLE = Path(os.environ.get("WT_ISYNTAX_SAMPLE", DEFAULT_SAMPLE))


def _sample_missing() -> str | None:
    if not SAMPLE.exists():
        return f"no iSyntax sample at {SAMPLE}"
    if SAMPLE == DEFAULT_SAMPLE and SAMPLE.stat().st_size != DEFAULT_SAMPLE_BYTES:
        return f"{SAMPLE} is incomplete ({SAMPLE.stat().st_size} of {DEFAULT_SAMPLE_BYTES} bytes)"
    return None


needs_sample = pytest.mark.skipif(_sample_missing() is not None, reason=_sample_missing() or "")


@pytest.fixture(scope="module")
def slide():
    wsi = create_wsi_file(str(SAMPLE))
    yield wsi
    wsi.close()


def test_rgba_to_rgb():
    rgba = np.array([[[10, 20, 30, 255], [10, 20, 30, 0], [0, 0, 0, 128]]], dtype=np.uint8)
    rgb = _rgba_to_rgb(rgba)
    assert rgb.shape == (1, 3, 3) and rgb.dtype == np.uint8
    assert rgb[0, 0].tolist() == [10, 20, 30]  # opaque: unchanged
    assert rgb[0, 1].tolist() == [255, 255, 255]  # transparent: white
    assert rgb[0, 2].tolist() == [127, 127, 127]  # half: blended onto white


def test_isyntax_is_a_wsi_extension(tmp_path):
    assert ".isyntax" in WSI_EXTENSIONS
    (tmp_path / "s.isyntax").touch()
    assert find_wsi_for_h5(str(tmp_path / "s.h5")) == str(tmp_path / "s.isyntax")


@needs_sample
def test_opens_with_auto_and_explicit_engine(slide):
    assert isinstance(slide, ISyntaxFile)
    explicit = create_wsi_file(str(SAMPLE), engine="isyntax")
    try:
        assert isinstance(explicit, ISyntaxFile)
        assert explicit.get_original_size() == slide.get_original_size()
    finally:
        explicit.close()


@needs_sample
def test_levels_and_mpp(slide):
    w, h = slide.get_original_size()
    levels = slide._get_native_levels()
    assert len(levels) > 1
    for i, lv in enumerate(levels):
        assert lv.index == i
        assert lv.downsample == 2.0**i
        assert (lv.width, lv.height) == (w >> i, h >> i)
    assert 0.1 < slide.get_mpp() < 1.0


@needs_sample
def test_read_region_is_rgb_and_clamped(slide):
    w, h = slide.get_original_size()
    region = slide.read_region((1000, 2000, 300, 200))
    assert region.shape == (200, 300, 3) and region.dtype == np.uint8
    # Past the far edge: clamped, so the region comes back smaller
    edge = slide.read_region((w - 50, h - 30, 256, 256))
    assert edge.shape == (30, 50, 3)
    lv = slide._get_native_levels()[-1]
    assert slide._read_native_region(lv.index, 0, 0, lv.width + 10, lv.height + 10).shape == (lv.height, lv.width, 3)


@needs_sample
def test_levels_line_up(slide):
    """A level-1 region matches the level-0 region it covers, 2x box-downscaled (no shift between levels)."""
    lv0 = slide._get_native_levels()[0]
    x, y, s = lv0.width // 2 // 2 * 2, lv0.height // 2 // 2 * 2, 512
    fine = slide._read_native_region(0, x, y, s, s)
    coarse = slide._read_native_region(1, x // 2, y // 2, s // 2, s // 2).astype(np.float32)
    down = np.asarray(Image.fromarray(fine).resize((s // 2, s // 2), Image.Resampling.BOX)).astype(np.float32)
    assert np.abs(coarse - down).mean() < 8
    # shifting the coarse region by one pixel makes it worse (so the match is not just flat background)
    shifted = slide._read_native_region(1, x // 2 + 1, y // 2, s // 2, s // 2).astype(np.float32)
    assert np.abs(shifted - down).mean() > np.abs(coarse - down).mean()


@needs_sample
def test_reopened_handles_read_the_same_pixels_in_parallel(slide):
    lv0 = slide._get_native_levels()[0]
    boxes = [(lv0.width // 2 + 300 * i, lv0.height // 2, 256, 256) for i in range(4)]
    expected = [slide.read_region(b) for b in boxes]
    handles = [slide.reopen() for _ in boxes]
    try:
        with ThreadPoolExecutor(len(boxes)) as pool:
            got = list(pool.map(lambda hb: hb[0].read_region(hb[1]), zip(handles, boxes)))
    finally:
        for hnd in handles:
            hnd.close()
    for e, g in zip(expected, got):
        assert np.array_equal(e, g)


@needs_sample
def test_dzi_tile_thumbnail_and_region_at_mpp(slide):
    max_level = slide.get_dzi_max_level()
    tile = slide.get_dzi_tile(max_level - 3, 0, 0)
    assert tile.shape == (256, 256, 3) and tile.dtype == np.uint8
    thumb = slide.generate_thumbnail(width=200)
    w, h = slide.get_original_size()
    assert thumb.shape == (round(200 * h / w), 200, 3) or thumb.shape == (int(200 / (w / h)), 200, 3)
    img = read_region_at_mpp(slide, 8000, 16000, 4000, 2000, target_mpp=2.0)
    assert img.shape == (round(2000 * slide.get_mpp() / 2.0), round(4000 * slide.get_mpp() / 2.0), 3)


@needs_sample
def test_patch_reader_workers_match_single_thread(slide):
    """read_workers > 1 (each worker reopens the slide) yields exactly the single-threaded patches."""
    readers = [WSIPatchReader(slide, patch_size=256, target_mpp=4.0, read_workers=n) for n in (1, 3)]
    assert readers[1].read_workers == 3  # ISyntaxFile can be reopened
    runs = [list(r.iter_batches(64)) for r in readers]
    assert len(runs[0]) == len(runs[1])
    for (b1, c1, _), (b3, c3, _) in zip(*runs):
        assert c1 == c3
        assert np.array_equal(b1, b3)
