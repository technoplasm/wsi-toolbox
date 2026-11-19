"""PyramidCommand (streamed pyramid TIFF writer) and DZI tiles from its output vs the original."""

import json
import os
import threading
import xml.etree.ElementTree as ET
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
    auto_bigtiff,
    pyramid_sizes,
    read_pyramid_info,
    tmp_files_for,
    tmp_path_for,
)
from wsi_toolbox.dzi import DziGenerator
from wsi_toolbox.wsi_files import PyramidalTiffFile, create_wsi_file

from .conftest import _smooth_rgb, write_pyramid_tiff


def test_level_sizes_halve_with_floor_until_one_tile():
    assert pyramid_sizes(36864, 35840, 512)[-1] == (288, 280)
    assert len(pyramid_sizes(36864, 35840, 512)) == 8
    assert pyramid_sizes(1001, 699, 128) == [(1001, 699), (500, 349), (250, 174), (125, 87)]
    assert pyramid_sizes(300, 200, 512) == [(300, 200)]


def test_default_output_matches_serving_settings(tmp_path, pyramid_tiff):
    """512 px JPEG tiles (Q85, 4:2:0, shared JPEGTables), reduced-resolution subfiles, in a classic
    TIFF (a slide below 2**31 pixels)."""
    dst = tmp_path / "pyramid.tif"
    PyramidCommand()(pyramid_tiff, dst, on_progress=None)
    with tifffile.TiffFile(dst) as tif:
        assert not tif.is_bigtiff
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


def _description(path) -> str:
    with tifffile.TiffFile(path) as tif:
        return tif.pages[0].description


def _native_sizes(path) -> list[tuple[int, int]]:
    """Level sizes as toolbox's own reader sees the file."""
    wsi = create_wsi_file(str(path))
    assert isinstance(wsi, PyramidalTiffFile)
    return [(level.width, level.height) for level in wsi._get_native_levels()]


def _openslide_sizes(path) -> list[tuple[int, int]]:
    assert openslide.OpenSlide.detect_format(str(path)) == "generic-tiff"
    slide = openslide.OpenSlide(str(path))
    try:
        return list(slide.level_dimensions)
    finally:
        slide.close()


def test_default_description_is_provenance_json(tmp_path, pyramid_tiff):
    """Page 0 says where the file comes from; openslide and toolbox still read every level."""
    dst = tmp_path / "out.pyramid.tif"
    PyramidCommand(tile_size=256)(pyramid_tiff, dst, on_progress=None)
    sizes = pyramid_sizes(1000, 700, 256)
    assert len(sizes) == 3

    with tifffile.TiffFile(dst) as tif:
        assert not tif.is_ome and not tif.is_shaped and not tif.is_imagej and not tif.is_svs
        assert [p.shape[1::-1] for p in tif.pages] == sizes
        assert all("ImageDescription" not in p.tags for p in tif.pages[1:])
        assert all(not p.subifds for p in tif.pages)
        text = tif.pages[0].description
    assert text.isascii()
    doc = json.loads(text)
    assert list(doc) == ["wsi_toolbox"]  # one top-level key: not tifffile's {"shape": ...}
    info = doc["wsi_toolbox"]
    # the source is a generic TIFF: openslide names the vendor, there is no objective power
    assert set(info) == {"version", "source", "mpp_x", "mpp_y", "tile_size", "quality", "vendor"}
    assert info["version"] == wt.__version__
    assert info["source"] == "pyramid.tif"
    assert info["mpp_x"] == pytest.approx(0.5) and info["mpp_y"] == pytest.approx(0.5)
    assert (info["tile_size"], info["quality"], info["vendor"]) == (256, 85, "generic-tiff")

    assert _openslide_sizes(dst) == sizes
    assert _native_sizes(dst) == sizes


def test_ome_output_reads_like_the_default_one(tmp_path, pyramid_tiff):
    """``ome=True``: OME-XML + the lower levels as SubIFDs of page 0, and the same top-level pages
    (same tiles) as the default output, so openslide and toolbox see no difference."""
    plain, ome = tmp_path / "plain.pyramid.tif", tmp_path / "out.ome.tif"
    PyramidCommand(tile_size=256)(pyramid_tiff, plain, on_progress=None)
    result = PyramidCommand(tile_size=256, ome=True)(pyramid_tiff, ome, on_progress=None)
    sizes = pyramid_sizes(1000, 700, 256)
    assert result.levels == len(sizes) == 3
    assert not tmp_files_for(ome)

    # b: the top-level pages as a plain TIFF reader sees them; c: the OME series
    with tifffile.TiffFile(plain) as a, tifffile.TiffFile(ome, is_ome=False) as b, tifffile.TiffFile(ome) as c:
        assert c.is_ome and not c.is_bigtiff and not a.is_ome
        assert len(b.pages) == len(a.pages) == len(sizes)
        levels = c.series[0].levels
        assert len(c.series) == 1 and len(levels) == len(c.pages) == len(sizes)
        # the series' lower levels are the SubIFDs of page 0, not the top-level pages
        assert len(b.pages[0].subifds) == len(sizes) - 1
        assert [level.pages[0].offset for level in levels[1:]] == list(b.pages[0].subifds)
        assert all(not p.subifds for p in b.pages[1:])
        assert all("ImageDescription" not in p.tags for p in b.pages[1:])
        for pa, pb, level in zip(a.pages, b.pages, levels, strict=True):
            expected = pa.asarray()
            assert np.array_equal(pb.asarray(), expected)
            assert np.array_equal(level.asarray(), expected)
            sub = level.pages[0]
            assert (sub.tilewidth, sub.subfiletype, sub.compression) == (pb.tilewidth, pb.subfiletype, pb.compression)
            assert pb.tags["XResolution"].value == pa.tags["XResolution"].value
    # twice the lower levels: a quarter to a third more for a full pyramid, never double
    assert plain.stat().st_size < ome.stat().st_size < 2 * plain.stat().st_size

    assert _openslide_sizes(ome) == _openslide_sizes(plain) == sizes
    assert _native_sizes(ome) == _native_sizes(plain) == sizes
    # toolbox's reader and DZI serving: the same tiles
    dzi_a, dzi_b = DziGenerator(create_wsi_file(str(plain))), DziGenerator(create_wsi_file(str(ome)))
    assert dzi_b.xml() == dzi_a.xml()
    for (la, ca, ra, ta), (lb, cb, rb, tb) in zip(dzi_a.iter_tiles(), dzi_b.iter_tiles(), strict=True):
        assert (la, ca, ra) == (lb, cb, rb)
        assert np.array_equal(ta, tb), (la, ca, ra)
    a, b = openslide.OpenSlide(str(plain)), openslide.OpenSlide(str(ome))
    for level, size in enumerate(sizes):
        assert np.array_equal(
            np.asarray(a.read_region((0, 0), level, size)), np.asarray(b.read_region((0, 0), level, size))
        )
    a.close()
    b.close()


def _is_bigtiff(path) -> bool:
    with tifffile.TiffFile(path) as tif:
        return tif.is_bigtiff


def _pages(path) -> list[tuple]:
    """Every top-level page and SubIFD: shape, tag values and the tiles' bytes. Leaves out what
    depends on the container or the run: the offsets, and the description (the OME-XML has a UUID)."""
    skip = {"TileOffsets", "SubIFDs", "ImageDescription"}
    out = []
    with tifffile.TiffFile(path, is_ome=False) as tif:
        fh = tif.filehandle
        for top in tif.pages:
            for page in [top, *(top.pages or [])]:
                tags = {tag.name: tag.value for tag in page.tags if tag.name not in skip}
                tiles = []
                for offset, count in zip(page.dataoffsets, page.databytecounts, strict=True):
                    fh.seek(offset)
                    tiles.append(fh.read(count))
                out.append((page.shape, tags, tiles))
    return out


def test_auto_bigtiff_is_from_2_to_the_31_pixels_on():
    assert pyramid_mod.BIGTIFF_MIN_PIXELS == 2**31
    assert not auto_bigtiff(1000, 700)
    assert not auto_bigtiff(36864, 35840)
    assert not auto_bigtiff(46340, 46340)  # 2**31 - 88048
    assert auto_bigtiff(46341, 46341)
    assert auto_bigtiff(65536, 32768) and not auto_bigtiff(65536, 32767)
    assert auto_bigtiff(108000, 63912)


@pytest.mark.parametrize("ome", [False, True])
def test_container_follows_the_slide_size_unless_given(tmp_path, pyramid_tiff, monkeypatch, ome):
    """Classic TIFF below the threshold, BigTIFF from it on; ``bigtiff=`` overrides. The pages
    are the same in both, and openslide and toolbox read both alike."""
    auto, classic, big = tmp_path / "auto.tif", tmp_path / "classic.tif", tmp_path / "big.tif"
    PyramidCommand(tile_size=256, ome=ome)(pyramid_tiff, auto, on_progress=None)
    PyramidCommand(tile_size=256, ome=ome, bigtiff=False)(pyramid_tiff, classic, on_progress=None)
    PyramidCommand(tile_size=256, ome=ome, bigtiff=True)(pyramid_tiff, big, on_progress=None)
    assert [_is_bigtiff(p) for p in (auto, classic, big)] == [False, False, True]
    assert len(_pages(big)) == (5 if ome else 3)  # with ome the two lower levels are SubIFDs too
    assert _pages(auto) == _pages(classic) == _pages(big)
    if not ome:
        assert _description(auto) == _description(big)
    assert auto.stat().st_size == classic.stat().st_size <= big.stat().st_size

    sizes = pyramid_sizes(1000, 700, 256)
    assert _openslide_sizes(auto) == _openslide_sizes(big) == sizes
    assert _native_sizes(auto) == _native_sizes(big) == sizes
    # toolbox's reader and DZI serving: the same tiles from both containers
    dzi_a, dzi_b = DziGenerator(create_wsi_file(str(auto))), DziGenerator(create_wsi_file(str(big)))
    assert dzi_b.xml() == dzi_a.xml()
    for (la, ca, ra, ta), (lb, cb, rb, tb) in zip(dzi_a.iter_tiles(), dzi_b.iter_tiles(), strict=True):
        assert (la, ca, ra) == (lb, cb, rb)
        assert np.array_equal(ta, tb), (la, ca, ra)
    a, b = openslide.OpenSlide(str(auto)), openslide.OpenSlide(str(big))
    for level, size in enumerate(sizes):
        assert np.array_equal(
            np.asarray(a.read_region((0, 0), level, size)), np.asarray(b.read_region((0, 0), level, size))
        )
    a.close()
    b.close()
    with tifffile.TiffFile(auto) as tif:
        assert tif.is_ome == ome and len(tif.series[0].levels) == 3

    # the same slide counted as a large one
    monkeypatch.setattr(pyramid_mod, "BIGTIFF_MIN_PIXELS", 1000 * 700)
    PyramidCommand(tile_size=256, ome=ome)(pyramid_tiff, auto, on_progress=None)
    assert _is_bigtiff(auto) and _pages(auto) == _pages(big)
    PyramidCommand(tile_size=256, ome=ome, bigtiff=False)(pyramid_tiff, classic, on_progress=None)
    assert not _is_bigtiff(classic)
    monkeypatch.setattr(pyramid_mod, "BIGTIFF_MIN_PIXELS", 1000 * 700 + 1)
    PyramidCommand(tile_size=256, ome=ome)(pyramid_tiff, auto, on_progress=None)
    assert not _is_bigtiff(auto)


@pytest.mark.parametrize("ome", [False, True])
def test_output_too_large_for_a_classic_tiff(tmp_path, pyramid_tiff, monkeypatch, collect, ome):
    """A classic TIFF ends at 4 GB (here: at the size the file would have). Chosen automatically it
    is given up and the pyramid built again as BigTIFF; asked for with ``bigtiff=False`` it is an
    error. Neither leaves a temporary file."""
    classic, big = tmp_path / "classic.tif", tmp_path / "big.tif"
    PyramidCommand(tile_size=128, ome=ome, bigtiff=False)(pyramid_tiff, classic, on_progress=None)
    PyramidCommand(tile_size=128, ome=ome, bigtiff=True)(pyramid_tiff, big, on_progress=None)
    monkeypatch.setattr(pyramid_mod, "CLASSIC_TIFF_MAX_BYTES", classic.stat().st_size)

    dst = tmp_path / "derived" / "out.tif"
    result = PyramidCommand(tile_size=128, ome=ome)(pyramid_tiff, dst, on_progress=collect)
    assert _is_bigtiff(dst) and _pages(dst) == _pages(big)
    assert result.bytes == dst.stat().st_size and result.levels == 4
    assert [p.name for p in dst.parent.iterdir()] == ["out.tif"]
    # the phase starts over for the second build
    steps = [e.n for e in collect.events if not e.done]
    assert collect.phases == ["Building pyramid"] and steps.count(0) == 2
    assert collect.events[-1].done

    with pytest.raises(PyramidError, match="classic TIFF"):
        PyramidCommand(tile_size=128, ome=ome, bigtiff=False)(pyramid_tiff, dst, on_progress=None)
    assert [p.name for p in dst.parent.iterdir()] == ["out.tif"]  # the earlier output, untouched
    assert _is_bigtiff(dst)

    # with room for it the file stays classic
    monkeypatch.setattr(pyramid_mod, "CLASSIC_TIFF_MAX_BYTES", 2 * classic.stat().st_size)
    PyramidCommand(tile_size=128, ome=ome)(pyramid_tiff, dst, on_progress=None)
    assert not _is_bigtiff(dst) and _pages(dst) == _pages(classic)


def test_ome_output_of_a_single_level(tmp_path):
    """An image that fits in one tile has no lower levels: one page, no SubIFDs, still OME."""
    src = write_pyramid_tiff(tmp_path / "small.tif", 300, 200, levels=1)
    dst = tmp_path / "small.ome.tif"
    result = PyramidCommand(ome=True)(src, dst, on_progress=None)
    assert (result.width, result.height, result.levels) == (300, 200, 1)
    with tifffile.TiffFile(dst) as tif:
        assert tif.is_ome and len(tif.pages) == 1 and not tif.pages[0].subifds
        assert len(tif.series[0].levels) == 1
        assert tif.series[0].asarray().shape == (200, 300, 3)
    assert _openslide_sizes(dst) == [(300, 200)]


def test_description_of_an_input_openslide_does_not_open(tmp_path, png_path):
    """``create_wsi_file`` path (PNG here, iSyntax in practice): no objective power, no vendor."""
    assert openslide.OpenSlide.detect_format(png_path) is None
    dst = tmp_path / "out.pyramid.tif"
    PyramidCommand(tile_size=256)(png_path, dst, on_progress=None)
    info = json.loads(_description(dst))["wsi_toolbox"]
    assert set(info) == {"version", "source", "mpp_x", "mpp_y", "tile_size", "quality"}
    assert info["source"] == Path(png_path).name

    PyramidCommand(tile_size=256, ome=True)(png_path, dst, on_progress=None)
    text = _description(dst)
    assert "<Instrument" not in text and "<Image " in text
    with tifffile.TiffFile(dst) as tif:
        assert tif.is_ome and len(tif.series[0].levels) == len(tif.pages)


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


# A source name that spells the markers readers sniff for ("imagej" / "<OME ": QuPath hands the
# file to Bio-Formats), with non-ASCII and XML / JSON special characters.
AWKWARD_NAME = "imagej<OME 標本 & \"x\" 'y'.fake"
OME_NS = {"ome": "http://www.openmicroscopy.org/Schemas/OME/2016-06"}


def test_description_json_objective_power_and_awkward_name(tmp_path, fake_slide):
    dst = tmp_path / "pyramid.tif"
    PyramidCommand(tile_size=256)(tmp_path / AWKWARD_NAME, dst, on_progress=None)
    info = json.loads(_description(dst))["wsi_toolbox"]
    # the slide gives neither an objective power nor a vendor: not guessed (0.25 um/px is not "40x")
    assert "objective_power" not in info and "vendor" not in info
    assert (info["mpp_x"], info["mpp_y"]) == (0.25, 0.25)

    fake_slide.properties[openslide.PROPERTY_NAME_OBJECTIVE_POWER] = "40"
    fake_slide.properties[openslide.PROPERTY_NAME_VENDOR] = "fake-vendor"
    del fake_slide.properties[openslide.PROPERTY_NAME_MPP_X], fake_slide.properties[openslide.PROPERTY_NAME_MPP_Y]
    PyramidCommand(tile_size=256)(tmp_path / AWKWARD_NAME, dst, on_progress=None)
    text = _description(dst)
    assert text.isascii() and text.startswith('{"wsi_toolbox"')
    assert "imagej" not in text and "<OME " not in text
    info = json.loads(text)["wsi_toolbox"]
    assert info["source"] == AWKWARD_NAME
    assert (info["objective_power"], info["vendor"]) == (40.0, "fake-vendor")
    assert info["mpp_x"] is None and info["mpp_y"] is None  # unknown is null, not 0
    with tifffile.TiffFile(dst) as tif:
        assert not tif.is_ome and not tif.is_imagej and not tif.is_svs


def test_ome_xml(tmp_path, fake_slide):
    dst = tmp_path / "out.ome.tif"
    fake_slide.properties[openslide.PROPERTY_NAME_OBJECTIVE_POWER] = "40"
    PyramidCommand(tile_size=256, ome=True)(tmp_path / AWKWARD_NAME, dst, on_progress=None)
    text = _description(dst)
    assert text.isascii() and text.startswith("<?xml") and "imagej" not in text
    root = ET.fromstring(text)
    assert root.tag == "{%s}OME" % OME_NS["ome"]
    assert root.get("UUID").startswith("urn:uuid:")  # Bio-Formats: no UUID, no OME-TIFF
    assert root.get("Creator") == f"wsi-toolbox {wt.__version__}"
    assert root.find("ome:Instrument/ome:Objective", OME_NS).get("NominalMagnification") == "40.0"
    image = root.find("ome:Image", OME_NS)
    assert image.get("Name") == AWKWARD_NAME
    assert [child.tag.split("}")[1] for child in image] == ["InstrumentRef", "ObjectiveSettings", "Pixels"]
    pixels = image.find("ome:Pixels", OME_NS)
    assert (pixels.get("SizeX"), pixels.get("SizeY"), pixels.get("SizeC")) == ("2100", "1300", "3")
    assert (pixels.get("PhysicalSizeX"), pixels.get("PhysicalSizeY")) == ("0.25", "0.25")
    assert pixels.find("ome:TiffData", OME_NS).attrib == {"IFD": "0", "PlaneCount": "1"}
    with tifffile.TiffFile(dst) as tif:
        assert tif.is_ome and len(tif.series[0].levels) == len(tif.pages) == 5

    # nothing known about the optics: no Instrument, no physical size
    del fake_slide.properties[openslide.PROPERTY_NAME_OBJECTIVE_POWER]
    del fake_slide.properties[openslide.PROPERTY_NAME_MPP_X], fake_slide.properties[openslide.PROPERTY_NAME_MPP_Y]
    PyramidCommand(tile_size=256, ome=True)(tmp_path / AWKWARD_NAME, dst, on_progress=None)
    root = ET.fromstring(_description(dst))
    assert root.find("ome:Instrument", OME_NS) is None
    image = root.find("ome:Image", OME_NS)
    assert [child.tag.split("}")[1] for child in image] == ["Pixels"]
    assert "PhysicalSizeX" not in image.find("ome:Pixels", OME_NS).attrib
    with tifffile.TiffFile(dst) as tif:
        assert tif.is_ome and len(tif.series[0].levels) == len(tif.pages)
