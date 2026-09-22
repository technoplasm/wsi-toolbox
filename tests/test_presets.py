import pytest

import wsi_toolbox as wt
from wsi_toolbox.presets.tile import PRESET_EXTRACT_FN, PRESET_NAMES, PRESET_NORMALIZATION, TilePreset, get_tile_preset


@pytest.mark.parametrize("name", PRESET_NAMES)
def test_get_tile_preset_returns_preset_without_building_model(name):
    preset = get_tile_preset(name)
    assert isinstance(preset, TilePreset)
    assert preset.name == name
    assert len(preset.norm_mean) == 3 and len(preset.norm_std) == 3
    assert callable(preset.create_model)


def test_unknown_preset_raises_with_names():
    with pytest.raises(ValueError) as exc:
        get_tile_preset("does-not-exist")
    assert "uni2" in str(exc.value)


def test_compat_tables_derive_from_presets():
    assert set(PRESET_NORMALIZATION) == set(PRESET_NAMES)
    assert PRESET_NORMALIZATION["h-optimus-0"][0] == get_tile_preset("h-optimus-0").norm_mean
    assert set(PRESET_EXTRACT_FN) == {"conch15_768"}
    assert get_tile_preset("conch15_768").extract_fn is not None
    assert get_tile_preset("uni").extract_fn is None


def test_set_default_preset_accepts_name_and_tilepreset(tiny_preset):
    wt.set_default_preset("uni")
    assert wt.defaults.preset == "uni"
    assert wt.resolve_preset(None).name == "uni"

    wt.set_default_preset(tiny_preset)
    assert wt.defaults.preset is tiny_preset
    assert wt.resolve_preset(None) is tiny_preset

    with pytest.raises(ValueError):
        wt.set_default_preset("nope")


def test_resolve_preset_without_default_raises():
    wt.defaults.preset = None
    with pytest.raises(ValueError):
        wt.resolve_preset(None)
    assert wt.resolve_preset("uni2").name == "uni2"
