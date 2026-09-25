"""PyramidCommand (vips tiffsave subprocess) and DZI tiles from its output vs the original."""

import os
import threading
from pathlib import Path

import numpy as np
import pytest

import wsi_toolbox as wt
from wsi_toolbox.commands import pyramid as pyramid_mod
from wsi_toolbox.commands.pyramid import (
    PyramidCommand,
    VipsError,
    read_pyramid_info,
    tmp_files_for,
    tmp_path_for,
    vips_available,
)
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


def test_missing_vips_is_a_clear_error(tmp_path, monkeypatch, pyramid_tiff):
    monkeypatch.setattr(pyramid_mod, "VIPS_BIN", "vips-does-not-exist")
    with pytest.raises(VipsError, match="not found"):
        PyramidCommand()(pyramid_tiff, tmp_path / "out.tif", on_progress=None)


@requires_vips
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


@requires_vips
def test_failure_leaves_no_output(tmp_path):
    dst = tmp_path / "out.tif"
    with pytest.raises(VipsError):
        PyramidCommand()(tmp_path / "does-not-exist.ndpi", dst, on_progress=None)
    assert not dst.exists() and not tmp_files_for(dst)


@requires_vips
def test_cancel_kills_vips_and_cleans_up(tmp_path, pyramid_tiff):
    dst = tmp_path / "out.tif"
    with pytest.raises(wt.Cancelled):
        PyramidCommand()(pyramid_tiff, dst, on_progress=None, should_cancel=lambda: True)
    assert not dst.exists() and not tmp_files_for(dst)


@requires_vips
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
