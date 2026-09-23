"""PyramidCommand (vips tiffsave subprocess) and DZI tiles from its output vs the original."""

from pathlib import Path

import numpy as np
import pytest

import wsi_toolbox as wt
from wsi_toolbox.commands import pyramid as pyramid_mod
from wsi_toolbox.commands.pyramid import PyramidCommand, VipsError, read_pyramid_info, tmp_path_for, vips_available
from wsi_toolbox.dzi import DziGenerator
from wsi_toolbox.wsi_files import PyramidalTiffFile, create_wsi_file

requires_vips = pytest.mark.skipif(not vips_available(), reason="libvips CLI (vips) not installed")


def test_default_args_match_serving_settings():
    """512 px tiles, pyramid, JPEG Q85, BigTIFF (the settings vision's pyramid job used)."""
    args = PyramidCommand().build_args("in.ndpi", "out.tif")
    assert args == [
        "vips",
        "tiffsave",
        "in.ndpi",
        "out.tif",
        "--tile",
        "--tile-width",
        "512",
        "--tile-height",
        "512",
        "--pyramid",
        "--compression",
        "jpeg",
        "--Q",
        "85",
        "--bigtiff",
    ]


def test_missing_vips_is_a_clear_error(tmp_path, monkeypatch, pyramid_tiff):
    monkeypatch.setattr(pyramid_mod, "VIPS_BIN", "vips-does-not-exist")
    with pytest.raises(VipsError, match="not found"):
        PyramidCommand()(pyramid_tiff, tmp_path / "out.tif", on_progress=None)


@requires_vips
def test_builds_pyramid_atomically_with_progress(tmp_path, pyramid_tiff, collect):
    dst = tmp_path / "derived" / "pyramid.tif"
    result = PyramidCommand(tile_size=128)(pyramid_tiff, dst, on_progress=collect)

    assert dst.is_file() and not tmp_path_for(dst).exists()
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


@requires_vips
def test_failure_leaves_no_output(tmp_path):
    dst = tmp_path / "out.tif"
    with pytest.raises(VipsError):
        PyramidCommand()(tmp_path / "does-not-exist.ndpi", dst, on_progress=None)
    assert not dst.exists() and not tmp_path_for(dst).exists()


@requires_vips
def test_cancel_kills_vips_and_cleans_up(tmp_path, pyramid_tiff):
    dst = tmp_path / "out.tif"
    with pytest.raises(wt.Cancelled):
        PyramidCommand()(pyramid_tiff, dst, on_progress=None, should_cancel=lambda: True)
    assert not dst.exists() and not tmp_path_for(dst).exists()


@requires_vips
@pytest.mark.parametrize("fixture", ["pyramid_tiff", "odd_pyramid_tiff"])
def test_dzi_from_pyramid_matches_original(request, tmp_path, fixture):
    """Same DZI geometry tile for tile; pixels equal within tolerance.

    The two pyramids are built independently (the fixture's BOX levels vs libvips' own reduction)
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
