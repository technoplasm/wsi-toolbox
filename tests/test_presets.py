from types import SimpleNamespace

import pytest
import torch

import wsi_toolbox as wt
from wsi_toolbox.presets.tile import PRESET_EXTRACT_FN, PRESET_NAMES, PRESET_NORMALIZATION, TilePreset, get_tile_preset
from wsi_toolbox.presets.tile.hibou import _swiglu_hidden, build_hibou_vit, convert_hibou_state_dict


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
    assert set(PRESET_NORMALIZATION) == set(PRESET_NAMES) | {"conch15_768"}
    assert PRESET_NORMALIZATION["h-optimus-0"][0] == get_tile_preset("h-optimus-0").norm_mean
    # presets whose default features are not forward_features(x)[:, 0]
    assert set(PRESET_EXTRACT_FN) == {"conch15_768", "kaiko-midnight"}
    assert get_tile_preset("kaiko-midnight").extract_fn is not None
    assert get_tile_preset("uni").extract_fn is None


def test_preset_output_names():
    assert get_tile_preset("uni").output_names == ("cls", "latent")
    assert get_tile_preset("conch15").output_names == ("cls", "latent", "proj")
    assert get_tile_preset("kaiko-midnight").output_names == ("cls",)
    assert get_tile_preset("conch15").key_suffix("proj") == "_768"
    assert "conch15_768" not in PRESET_NAMES


@pytest.mark.parametrize(("name", "prefix"), [("virchow", 1), ("virchow2", 5)])
def test_virchow_concat_head(name, prefix):
    preset = get_tile_preset(name)
    assert preset.output_names == ("cls", "latent", "concat")
    assert preset.default_outputs == ("cls",)  # unchanged: virchow/features stays the 1280-dim CLS
    assert preset.key_suffix("concat") == "_2560"
    cmd = wt.FeatureExtractionCommand(model=name, preset=name, outputs=("cls", "concat"))
    assert cmd._groups(*cmd._resolve()) == {"cls": name, "concat": f"{name}_2560"}

    # concat(CLS, mean of the patch tokens); registers (Virchow2) are left out of the mean
    model = SimpleNamespace(num_prefix_tokens=prefix)
    tokens = torch.randn(2, prefix + 6, 4)
    out = preset.heads["concat"].fn(model, tokens)
    assert out.shape == (2, 8)
    assert torch.equal(out, torch.cat([tokens[:, 0], tokens[:, prefix:].mean(dim=1)], dim=-1))


def test_conch15_768_alias_warns_and_maps_to_conch15_proj():
    with pytest.warns(DeprecationWarning):
        preset = get_tile_preset("conch15_768")
    assert preset.name == "conch15"
    assert preset.default_outputs == ("proj",)
    assert preset.key_suffix("proj") == ""  # stays at {model}/features, not {model}_768
    with pytest.raises(ValueError):
        preset.resolve_outputs(("cls", "proj"))  # both would land in {model}/features
    wt.set_default_preset("conch15_768")  # accepted as a name


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


def _fake_hibou_state_dict(dim: int, depth: int) -> dict[str, torch.Tensor]:
    """Random tensors under the transformers Dinov2ModelWithRegisters keys (as in histai/hibou-*)."""
    hidden = _swiglu_hidden(dim)
    sd = {
        "embeddings.cls_token": torch.randn(1, 1, dim),
        "embeddings.mask_token": torch.randn(1, dim),
        "embeddings.position_embeddings": torch.randn(1, 1 + 16 * 16, dim),
        "embeddings.register_tokens": torch.randn(1, 4, dim),
        "embeddings.patch_embeddings.projection.weight": torch.randn(dim, 3, 14, 14),
        "embeddings.patch_embeddings.projection.bias": torch.randn(dim),
        "layernorm.weight": torch.randn(dim),
        "layernorm.bias": torch.randn(dim),
    }
    shapes = {
        "norm1": (dim,),
        "norm2": (dim,),
        "attention.attention.query": (dim, dim),
        "attention.attention.key": (dim, dim),
        "attention.attention.value": (dim, dim),
        "attention.output.dense": (dim, dim),
        "mlp.weights_in": (2 * hidden, dim),
        "mlp.weights_out": (dim, hidden),
    }
    for i in range(depth):
        for name, shape in shapes.items():
            sd[f"encoder.layer.{i}.{name}.weight"] = torch.randn(*shape)
            sd[f"encoder.layer.{i}.{name}.bias"] = torch.randn(shape[0])
        sd[f"encoder.layer.{i}.layer_scale1.lambda1"] = torch.randn(dim)
        sd[f"encoder.layer.{i}.layer_scale2.lambda1"] = torch.randn(dim)
    return sd


def test_hibou_state_dict_converts_to_timm_strictly():
    sd = _fake_hibou_state_dict(dim=32, depth=2)
    model = build_hibou_vit(embed_dim=32, depth=2, num_heads=2)
    converted = convert_hibou_state_dict(sd)
    model.load_state_dict(converted)  # strict: every timm parameter is covered, nothing extra

    # CLS position folded into the token; registers sit between CLS and the patches
    pos = sd["embeddings.position_embeddings"]
    assert torch.equal(model.cls_token, sd["embeddings.cls_token"] + pos[:, :1])
    assert torch.equal(model.pos_embed, pos[:, 1:])
    q, k, v = (sd[f"encoder.layer.1.attention.attention.{n}.weight"] for n in ("query", "key", "value"))
    assert torch.equal(model.blocks[1].attn.qkv.weight, torch.cat([q, k, v]))

    model.eval()
    with torch.no_grad():
        tokens = model.forward_features(torch.randn(1, 3, 224, 224))
    assert tokens.shape == (1, 1 + 4 + 16 * 16, 32)
    assert model.patch_embed.proj.kernel_size == (14, 14)
