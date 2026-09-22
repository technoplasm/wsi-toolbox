import os

import h5py
import pytest

import wsi_toolbox as wt
from wsi_toolbox.progress import Cancelled

from .conftest import FEATURE_DIM, GRID, PATCH_SIZE

N_PATCHES = GRID * GRID


def _extract_cmd(tiny_preset, **kw):
    return wt.FeatureExtractionCommand(
        model="tiny",
        preset=tiny_preset,
        device="cpu",
        batch_size=kw.pop("batch_size", 8),
        patch_size=PATCH_SIZE,
        target_mpp=0.5,
        white_detector=lambda patch: False,
        **kw,
    )


def test_extract_from_png_writes_features(tmp_path, png_path, tiny_preset, collect):
    h5 = str(tmp_path / "out.h5")
    result = _extract_cmd(tiny_preset)(h5, wsi_path=png_path, on_progress=collect)

    assert not result.skipped
    assert result.patch_count == N_PATCHES
    assert result.feature_dim == FEATURE_DIM
    assert result.total_batches == 2
    assert result.model == "tiny"

    with h5py.File(h5, "r") as f:
        assert f["tiny/features"].shape == (N_PATCHES, FEATURE_DIM)
        assert f["tiny/coordinates"].shape == (N_PATCHES, 2)
        assert not f["tiny/features"].attrs["writing"]
        assert f["tiny"].attrs["preset"] == "tiny"
        assert int(f["tiny"].attrs["patch_count"]) == N_PATCHES
        assert int(f["tiny"].attrs["patch_size"]) == PATCH_SIZE
        assert "tiny/latent_features" not in f

    assert collect.phases == ["Initializing model", "Processing patches", "Writing"]
    proc = [e for e in collect.events if e.phase == "Processing patches"]
    assert proc[0].n == 0 and proc[-1].n == proc[-1].total == 2
    assert proc[-1].message  # reader desc is passed as message
    assert collect.events[-1].done


def test_extract_skips_when_present(tmp_path, png_path, tiny_preset):
    h5 = str(tmp_path / "out.h5")
    cmd = _extract_cmd(tiny_preset)
    cmd(h5, wsi_path=png_path, on_progress=None)
    result = cmd(h5, wsi_path=png_path, on_progress=None)
    assert result.skipped


def test_cancel_mid_run_leaves_no_partial_dataset(tmp_path, png_path, tiny_preset, collect):
    h5 = str(tmp_path / "out.h5")
    seen_batches = {"n": 0}

    def should_cancel():
        # Called on every advance; cancel after the first batch was processed.
        return seen_batches["n"] > 0

    def sink(event):
        collect(event)
        if event.phase == "Processing patches" and event.n > 0:
            seen_batches["n"] = event.n

    cmd = _extract_cmd(tiny_preset, batch_size=4)  # 4 batches so cancel hits mid-way
    with pytest.raises(Cancelled):
        cmd(h5, wsi_path=png_path, on_progress=sink, should_cancel=should_cancel)

    assert collect.phases == ["Initializing model", "Processing patches"]
    assert not any(e.done for e in collect.events)
    if os.path.exists(h5):
        with h5py.File(h5, "r") as f:
            assert "tiny/features" not in f
            assert "tiny/coordinates" not in f


def test_preset_none_uses_defaults_and_errors_when_unset(tmp_path, png_path, tiny_preset):
    h5 = str(tmp_path / "out.h5")
    cmd = wt.FeatureExtractionCommand(model="tiny", batch_size=8, patch_size=PATCH_SIZE, white_detector=lambda p: False)
    with pytest.raises(ValueError):
        cmd(h5, wsi_path=png_path, on_progress=None)

    wt.set_default_preset(tiny_preset)
    result = cmd(h5, wsi_path=png_path, on_progress=None)
    assert result.patch_count == N_PATCHES
    with h5py.File(h5, "r") as f:
        assert f["tiny"].attrs["preset"] == "tiny"


def test_unspecified_on_progress_uses_default_sink(tmp_path, png_path, tiny_preset, collect):
    wt.set_default_progress(collect)
    h5 = str(tmp_path / "out.h5")
    _extract_cmd(tiny_preset)(h5, wsi_path=png_path)
    assert "Processing patches" in collect.phases
