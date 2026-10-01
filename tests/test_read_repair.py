"""OpenSlideFile read repair: a block openslide cannot decode is read from level 0, or dropped when level 0 fails too.

Uses a fake ``OpenSlide`` (two levels, injected decode errors, sticky error state like openslide), no real slide.
"""

import json

import h5py
import numpy as np
import openslide
import pytest
from PIL import Image

import wsi_toolbox as wt
from wsi_toolbox import wsi_files
from wsi_toolbox.patch_reader import WSIPatchReader
from wsi_toolbox.wsi_files import OpenSlideFile

LEVEL0 = 256
PATCH = 32  # at level 1 (128 px): a 4x4 grid


def _texture(size: int) -> np.ndarray:
    """Non-white, varied image so the ptp white detector keeps every patch."""
    rng = np.random.default_rng(0)
    return rng.integers(30, 220, (size, size, 3), dtype=np.uint8)


class FakeOpenSlide:
    """The part of ``openslide.OpenSlide`` OpenSlideFile uses. ``bad`` maps level -> level-pixel boxes
    (x, y, w, h) whose decode fails; after a failure every read fails, as with openslide."""

    image0 = _texture(LEVEL0)
    bad: dict[int, list[tuple[int, int, int, int]]] = {}
    opened = 0

    def __init__(self, _path):
        type(self).opened += 1
        self.levels = [self.image0, np.array(Image.fromarray(self.image0).resize((LEVEL0 // 2,) * 2, Image.BOX))]
        self.properties = {"openslide.mpp-x": "0.25"}
        self.level_dimensions = [(a.shape[1], a.shape[0]) for a in self.levels]
        self.level_downsamples = [1.0, 2.0]
        self.broken = False

    def read_region(self, location, level, size):
        ds = self.level_downsamples[level]
        x, y = int(location[0] / ds), int(location[1] / ds)
        w, h = size
        if self.broken:
            raise openslide.OpenSlideError("handle is in an error state")
        for bx, by, bw, bh in self.bad.get(level, []):
            if x < bx + bw and bx < x + w and y < by + bh and by < y + h:
                self.broken = True
                raise openslide.OpenSlideError("Corrupt JPEG data: premature end of data segment")
        rgb = self.levels[level][y : y + h, x : x + w]
        return Image.fromarray(np.dstack([rgb, np.full(rgb.shape[:2], 255, np.uint8)]), "RGBA")

    def close(self):
        pass


@pytest.fixture
def fake_openslide(monkeypatch):
    monkeypatch.setattr(wsi_files, "OpenSlide", FakeOpenSlide)
    monkeypatch.setattr(FakeOpenSlide, "bad", {})
    return FakeOpenSlide


def _read_all(read_workers: int, white_detector=None):
    wsi = OpenSlideFile("slide.ndpi")
    reader = WSIPatchReader(
        wsi, patch_size=PATCH, target_mpp=0.5, white_detector=white_detector, read_workers=read_workers
    )
    assert reader.level.index == 1
    patches, coords = [], []
    for batch, c, _ in reader.iter_batches(8):
        patches.extend(batch)
        coords.extend(c)
    return reader, dict(zip(coords, patches))


def _level1_patch(x: int, y: int) -> np.ndarray:
    return FakeOpenSlide(None).levels[1][y : y + PATCH, x : x + PATCH]


@pytest.mark.parametrize("read_workers", [1, 2])
def test_readable_slide_is_unchanged(fake_openslide, read_workers):
    reader, got = _read_all(read_workers)
    assert len(got) == 16
    for (x, y), patch in got.items():
        assert np.array_equal(patch, _level1_patch(x, y))
    assert json.loads(reader.metadata["level0_fallback_tiles"]) == []
    assert json.loads(reader.metadata["unreadable_tiles"]) == []


@pytest.mark.parametrize("read_workers", [1, 2])
def test_undecodable_block_is_read_from_level0(fake_openslide, read_workers):
    fake_openslide.bad = {1: [(64, 32, PATCH, PATCH)]}
    reader, got = _read_all(read_workers)

    assert len(got) == 16  # nothing dropped
    # Level 1 of the fake is level 0 BOX-downscaled, so the substitute is exact
    for (x, y), patch in got.items():
        assert np.array_equal(patch, _level1_patch(x, y)), (x, y)
    fallback = json.loads(reader.metadata["level0_fallback_tiles"])
    assert [(e["level"], e["x"], e["y"], e["w"], e["h"]) for e in fallback] == [(1, 64, 32, PATCH, PATCH)]
    assert "Corrupt JPEG" in fallback[0]["error"]
    assert json.loads(reader.metadata["unreadable_tiles"]) == []


@pytest.mark.parametrize("read_workers", [1, 2])
def test_block_unreadable_at_level0_too_is_dropped(fake_openslide, read_workers):
    fake_openslide.bad = {1: [(0, 96, PATCH, PATCH), (96, 0, PATCH, PATCH)], 0: [(0, 192, 2 * PATCH, 2 * PATCH)]}
    # No white detector: the white fill must still not become a patch
    reader, got = _read_all(read_workers, white_detector=None)

    assert (0, 96) not in got and len(got) == 15
    assert np.array_equal(got[(96, 0)], _level1_patch(96, 0))  # the other one came from level 0
    unreadable = json.loads(reader.metadata["unreadable_tiles"])
    assert [(e["x"], e["y"]) for e in unreadable] == [(0, 96)]
    assert "error_level0" in unreadable[0]
    assert [(e["x"], e["y"]) for e in json.loads(reader.metadata["level0_fallback_tiles"])] == [(96, 0)]


def test_extraction_records_repairs_and_warns(tmp_path, fake_openslide, tiny_preset, collect):
    fake_openslide.bad = {1: [(32, 32, PATCH, PATCH), (64, 64, PATCH, PATCH)], 0: [(128, 128, 2 * PATCH, 2 * PATCH)]}
    h5 = str(tmp_path / "out.h5")
    result = wt.FeatureExtractionCommand(
        model="tiny", preset=tiny_preset, device="cpu", batch_size=8, patch_size=PATCH, target_mpp=0.5
    )(h5, wsi_path="slide.ndpi", on_progress=collect)

    assert result.patch_count == 15
    with h5py.File(h5, "r") as f:
        coords = {tuple(c) for c in f["tiny/coordinates"][:]}
        grp = f["tiny"].attrs
        assert [(e["x"], e["y"]) for e in json.loads(grp["level0_fallback_tiles"])] == [(32, 32)]
        assert [(e["x"], e["y"]) for e in json.loads(grp["unreadable_tiles"])] == [(64, 64)]
    assert (64, 64) not in coords and (32, 32) in coords
    writing = [e for e in collect.events if e.phase == "Writing"]
    assert "1 patch(es) read from level 0" in writing[0].message
    assert "1 unreadable patch(es) dropped" in writing[0].message
