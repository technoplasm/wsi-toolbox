"""PyramidCommand (streamed pyramid TIFF writer) and DZI tiles from its output vs the original."""

import os
import threading
from pathlib import Path

import numpy as np
import openslide
import pytest
import tifffile
from PIL import Image

import wsi_toolbox as wt
from wsi_toolbox.commands import pyramid as pyramid_mod
from wsi_toolbox.commands.pyramid import (
    PyramidCommand,
    PyramidError,
    VipsError,
    pyramid_sizes,
    read_pyramid_info,
    tmp_files_for,
    tmp_path_for,
)
from wsi_toolbox.dzi import DziGenerator
from wsi_toolbox.wsi_files import PyramidalTiffFile, create_wsi_file

from .conftest import _smooth_rgb


def test_level_sizes_halve_with_floor_until_one_tile():
    assert pyramid_sizes(36864, 35840, 512)[-1] == (288, 280)
    assert len(pyramid_sizes(36864, 35840, 512)) == 8
    assert pyramid_sizes(1001, 699, 128) == [(1001, 699), (500, 349), (250, 174), (125, 87)]
    assert pyramid_sizes(300, 200, 512) == [(300, 200)]


def test_default_output_matches_serving_settings(tmp_path, pyramid_tiff):
    """512 px JPEG tiles (Q85, 4:2:0, shared JPEGTables), BigTIFF, reduced-resolution subfiles."""
    dst = tmp_path / "pyramid.tif"
    PyramidCommand()(pyramid_tiff, dst, on_progress=None)
    with tifffile.TiffFile(dst) as tif:
        assert tif.is_bigtiff
        assert [p.shape[:2] for p in tif.pages] == [(700, 1000), (350, 500)]
        for i, page in enumerate(tif.pages):
            assert (page.tilewidth, page.tilelength) == (512, 512)
            assert page.compression == tifffile.COMPRESSION.JPEG
            assert page.photometric == tifffile.PHOTOMETRIC.YCBCR
            assert page.tags["YCbCrSubSampling"].value == (2, 2)
            assert "JPEGTables" in page.tags
            assert page.subfiletype == (1 if i else 0)
            assert page.tags["ResolutionUnit"].value == tifffile.RESUNIT.INCH


def test_quality_90_keeps_full_chroma(tmp_path, pyramid_tiff):
    dst = tmp_path / "pyramid.tif"
    PyramidCommand(tile_size=128, quality=90)(pyramid_tiff, dst, on_progress=None)
    with tifffile.TiffFile(dst) as tif:
        assert tif.pages[0].tags["YCbCrSubSampling"].value == (1, 1)


def test_vips_error_is_the_old_name_of_pyramid_error():
    assert VipsError is PyramidError
    assert wt.vips_available()


def test_tmp_name_is_unique_per_run(tmp_path):
    dst = tmp_path / "pyramid.tif"
    a, b = tmp_path_for(dst), tmp_path_for(dst)
    assert a != b
    assert a.parent == dst.parent and a.name.startswith(".pyramid.tif.") and a.suffix == ".tmp"
    assert tmp_path_for(dst, "x") == tmp_path / ".pyramid.tif.x.tmp"


def test_stale_tmp_is_removed_fresh_is_kept(tmp_path, monkeypatch):
    dst = tmp_path / "pyramid.tif"
    old, fresh = tmp_path_for(dst, "old"), tmp_path_for(dst, "fresh")
    old.write_bytes(b"x")
    fresh.write_bytes(b"x")
    past = old.stat().st_mtime - pyramid_mod.STALE_TMP_SECONDS - 10
    os.utime(old, (past, past))
    pyramid_mod._remove_stale_tmp(dst)
    assert tmp_files_for(dst) == [fresh]


def test_builds_pyramid_atomically_with_progress(tmp_path, pyramid_tiff, collect):
    dst = tmp_path / "derived" / "pyramid.tif"
    result = PyramidCommand(tile_size=128)(pyramid_tiff, dst, on_progress=collect)

    assert dst.is_file() and not tmp_files_for(dst)
    assert (result.width, result.height, result.tile_size) == (1000, 700, 128)
    assert result.levels >= 3
    assert result.bytes == dst.stat().st_size
    assert result == wt.PyramidResult(path=str(dst), elapsed=result.elapsed, **read_pyramid_info(dst).model_dump())

    assert collect.phases == ["Building pyramid"]
    steps = [e for e in collect.events if not e.done]
    assert steps[0].n == 0 and steps[0].total == 100
    assert all(a.n <= b.n for a, b in zip(steps, steps[1:]))
    assert collect.events[-1].done

    assert isinstance(create_wsi_file(str(dst)), PyramidalTiffFile)


def test_openslide_reads_the_output_like_tifffile(tmp_path, pyramid_tiff):
    """Abbreviated JPEG tiles + JPEGTables decode the same in openslide (libjpeg) and tifffile."""
    dst = tmp_path / "pyramid.tif"
    PyramidCommand(tile_size=128)(pyramid_tiff, dst, on_progress=None)
    slide = openslide.OpenSlide(str(dst))
    assert slide.level_count == read_pyramid_info(dst).levels
    with tifffile.TiffFile(dst) as tif:
        for level in (0, 1):
            expected = tif.pages[level].asarray()
            got = np.asarray(slide.read_region((0, 0), level, expected.shape[1::-1]))[..., :3]
            assert np.abs(got.astype(np.int16) - expected.astype(np.int16)).max() <= 2
    slide.close()


def test_failure_leaves_no_output(tmp_path):
    dst = tmp_path / "out.tif"
    with pytest.raises(PyramidError, match="cannot open"):
        PyramidCommand()(tmp_path / "does-not-exist.ndpi", dst, on_progress=None)
    assert not dst.exists() and not tmp_files_for(dst)


def test_cancel_cleans_up(tmp_path, pyramid_tiff):
    dst = tmp_path / "out.tif"
    with pytest.raises(wt.Cancelled):
        PyramidCommand()(pyramid_tiff, dst, on_progress=None, should_cancel=lambda: True)
    assert not dst.exists() and not tmp_files_for(dst)


def test_concurrent_runs_on_same_output_do_not_collide(tmp_path, pyramid_tiff):
    """Two runs writing the same output at once (vision 2026-09-23: forced regeneration twice)
    used to share ``.pyramid.tif.tmp``: one renamed it away under the other (FileNotFoundError)
    or one read the other's half-written file. Each run now has its own temporary file."""
    dst = tmp_path / "derived" / "pyramid.tif"
    results, errors = [], []

    def run():
        try:
            results.append(PyramidCommand(tile_size=128)(pyramid_tiff, dst, on_progress=None))
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=run) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert len(results) == 3
    assert all((r.width, r.height, r.tile_size) == (1000, 700, 128) for r in results)
    assert read_pyramid_info(dst).width == 1000
    assert not tmp_files_for(dst)


@pytest.mark.parametrize("fixture", ["pyramid_tiff", "odd_pyramid_tiff"])
def test_dzi_from_pyramid_matches_original(request, tmp_path, fixture):
    """Same DZI geometry tile for tile; pixels equal within tolerance.

    The two pyramids are built independently (the fixture's BOX levels vs the 2x2 mean cascade)
    and the output is JPEG Q85 again, hence mean abs error < 4 rather than exact equality.
    """
    src = request.getfixturevalue(fixture)
    dst = Path(tmp_path) / "pyramid.tif"
    PyramidCommand(tile_size=128)(src, dst, on_progress=None)

    orig = DziGenerator(create_wsi_file(src))
    pyr = DziGenerator(create_wsi_file(str(dst)))
    assert pyr.layout == orig.layout
    assert pyr.xml() == orig.xml()

    for (lo, co, ro, a), (lp, cp, rp, b) in zip(orig.iter_tiles(), pyr.iter_tiles(), strict=True):
        assert (lo, co, ro) == (lp, cp, rp)
        assert a.shape == b.shape, (lo, co, ro)
        assert np.abs(a.astype(np.int16) - b.astype(np.int16)).mean() < 4.0, (lo, co, ro)


class FakeSlide:
    """What ``_SlideStream`` uses of ``openslide.OpenSlide``: a smooth image with a transparent band."""

    def __init__(self, width: int = 2100, height: int = 1300, mpp: float = 0.25, background: str = "ffffff"):
        self.dimensions = (width, height)
        self.properties = {
            openslide.PROPERTY_NAME_MPP_X: str(mpp),
            openslide.PROPERTY_NAME_MPP_Y: str(mpp),
            openslide.PROPERTY_NAME_BACKGROUND_COLOR: background,
        }
        self.rgba = np.dstack([_smooth_rgb(width, height), np.full((height, width), 255, np.uint8)])
        self.rgba[:, :100, 3] = 0  # outside the scanned area
        self.reads = 0
        self.closed = False

    def read_region(self, location, level, size):
        assert level == 0
        (x, y), (w, h) = location, size
        self.reads += 1
        return Image.fromarray(self.rgba[y : y + h, x : x + w], "RGBA")

    def close(self):
        self.closed = True


@pytest.fixture
def fake_slide(monkeypatch):
    """Route PyramidCommand's openslide path to a FakeSlide (any path counts as a vendor slide)."""
    slide = FakeSlide()
    monkeypatch.setattr(openslide.OpenSlide, "detect_format", staticmethod(lambda _path: "fake-vendor"))
    monkeypatch.setattr(pyramid_mod.openslide, "OpenSlide", _fake_openslide_class(slide))
    return slide


def _fake_openslide_class(slide):
    class _OpenSlide:
        detect_format = staticmethod(lambda _path: "fake-vendor")

        def __new__(cls, _path):
            return slide

    return _OpenSlide


def test_openslide_slide_transparency_and_resolution(tmp_path, fake_slide, collect):
    """Level 0 is read from openslide in bands; transparent pixels become the background colour;
    the resolution is written in pixels per inch."""
    dst = tmp_path / "pyramid.tif"
    result = PyramidCommand(tile_size=256)(tmp_path / "slide.fake", dst, on_progress=collect)

    assert (result.width, result.height, result.tile_size) == (2100, 1300, 256)
    assert result.levels >= 4
    assert fake_slide.reads > 0 and fake_slide.closed
    assert collect.events[-1].done

    with tifffile.TiffFile(dst) as tif:
        page = tif.pages[0]
        assert page.tags["ResolutionUnit"].value == tifffile.RESUNIT.INCH
        num, den = page.tags["XResolution"].value
        assert abs(25400 / (num / den) - 0.25) < 1e-4  # mpp
        level0 = page.asarray()
    expected = fake_slide.rgba[..., :3].copy()
    expected[:, :100] = 255
    assert np.abs(level0.astype(np.int16) - expected.astype(np.int16)).mean() < 2.0
    assert level0[:, 8:92].min() >= 250  # background, not black


def test_openslide_read_error_is_reported(tmp_path, fake_slide):
    def broken(*_args):
        raise openslide.OpenSlideError("corrupt tile")

    fake_slide.read_region = broken
    dst = tmp_path / "out.tif"
    with pytest.raises(PyramidError, match="corrupt tile"):
        PyramidCommand()(tmp_path / "slide.fake", dst, on_progress=None)
    assert not dst.exists() and not tmp_files_for(dst)
    assert fake_slide.closed


def test_cancel_while_reading_an_openslide_slide(tmp_path, fake_slide):
    calls = []

    def should_cancel():
        calls.append(1)
        return len(calls) > 1  # let the phase start, cancel at the first poll

    dst = tmp_path / "out.tif"
    with pytest.raises(wt.Cancelled):
        PyramidCommand()(tmp_path / "slide.fake", dst, on_progress=None, should_cancel=should_cancel)
    assert not dst.exists() and not tmp_files_for(dst)
    assert fake_slide.closed
