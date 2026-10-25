"""Tile-level foundation model presets.

A ``TilePreset`` bundles everything ``FeatureExtractionCommand`` needs to run a per-patch encoder: the model
factory (``create_model()``), the input normalization and the outputs it can produce (``preset(model, x)`` for
the default features, ``preset.outputs(model, x, names)`` for several from one forward pass). Built-in presets
are subclasses registered by name and looked up with ``get_tile_preset(name)``; custom models are plain
``TilePreset(...)`` instances.

Building a preset is cheap: timm and the model code are imported only inside ``create_model()`` (torch is
imported for the Virchow ``concat`` head, but the package imports it anyway).
"""

import logging
import warnings
from collections.abc import Callable
from typing import Any

from ._base import BASE_OUTPUTS, IMAGENET_MEAN, IMAGENET_STD, TileHead, TilePreset
from .virchow import virchow_concat

logger = logging.getLogger(__name__)

# H-optimus-0 (also used by its distillation H0-mini)
_HOPTIMUS_MEAN = (0.707223, 0.578729, 0.703617)
_HOPTIMUS_STD = (0.211883, 0.230117, 0.177517)
# Hibou-B / Hibou-L (preprocessor_config.json)
_HIBOU_MEAN = (0.7068, 0.5755, 0.722)
_HIBOU_STD = (0.195, 0.2316, 0.1816)
_HALF = (0.5, 0.5, 0.5)


# Suppress noisy logs from huggingface_hub and timm
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("timm").setLevel(logging.WARNING)


# --- built-in preset classes ------------------------------------------------------------------


class TimmHubPreset(TilePreset):
    """A timm model loaded from the HF hub (``timm.create_model(hub_id, pretrained=True, **kwargs)``).

    ``swiglu=True`` adds ``mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU`` (imported lazily). ``heads`` are extra
    outputs from the same ``forward_features`` (see ``TileHead``).
    """

    def __init__(
        self,
        name: str,
        hub_id: str,
        *,
        norm=(IMAGENET_MEAN, IMAGENET_STD),
        swiglu=False,
        heads: dict[str, TileHead] | None = None,
        **timm_kwargs,
    ):
        super().__init__(name, norm_mean=norm[0], norm_std=norm[1], heads=heads)
        self.hub_id = hub_id
        self.swiglu = swiglu
        self.timm_kwargs = timm_kwargs

    def create_model(self):
        import timm  # noqa: PLC0415

        kwargs = dict(self.timm_kwargs)
        if self.swiglu:
            import torch  # noqa: PLC0415
            from timm.layers import SwiGLUPacked  # noqa: PLC0415

            kwargs.update(mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU)
        return timm.create_model(self.hub_id, pretrained=True, **kwargs)


def _conch_proj(model, tokens):
    # Same ops, same order as CONCHVisionTower.forward: attn_pool_contrast -> [:, 0] -> ln_contrast
    return model.project(tokens)


class Conch15Preset(TilePreset):
    """CONCH v1.5: ``cls`` = trunk CLS (1024), ``proj`` = attn_pool_contrast -> ln_contrast (768, TITAN input)."""

    def __init__(self, name: str = "conch15", *, proj_suffix: str = "_768", default_outputs=("cls",)):
        super().__init__(
            name,
            heads={"proj": TileHead(key_suffix=proj_suffix, fn=_conch_proj)},
            default_outputs=default_outputs,
        )

    def create_model(self):
        from .conch import create_conch_model  # noqa: PLC0415

        return create_conch_model()


class MidnightPreset(TilePreset):
    """OpenMidnight (SophontAI)."""

    def __init__(self):
        super().__init__("midnight", norm_mean=_HALF, norm_std=_HALF)

    def create_model(self):
        from .midnight import create_midnight_model  # noqa: PLC0415

        return create_midnight_model()


class GigapathFlashPreset(TilePreset):
    def __init__(self):
        super().__init__("gigapath-flash")

    def create_model(self):
        from .gigapath_flash import create_gigapath_flash_model  # noqa: PLC0415

        return create_gigapath_flash_model()


class PhikonPreset(TilePreset):
    def __init__(self, name: str, version: str):
        super().__init__(name)
        self.version = version

    def create_model(self):
        from .phikon import create_phikon_model  # noqa: PLC0415

        return create_phikon_model(self.version)


class HibouPreset(TilePreset):
    def __init__(self, variant: str):
        super().__init__(f"hibou-{variant}", norm_mean=_HIBOU_MEAN, norm_std=_HIBOU_STD)
        self.variant = variant

    def create_model(self):
        from .hibou import create_hibou_model  # noqa: PLC0415

        return create_hibou_model(self.variant)


def _kaiko_extract(model, x):
    # concat(CLS, mean of patch tokens), the model card's classification embedding
    return model(x)


class KaikoMidnightPreset(TilePreset):
    """Midnight-12k (kaiko.ai); the only output is ``cls`` = concat(CLS, mean of patch tokens) (3072)."""

    def __init__(self):
        super().__init__("kaiko-midnight", norm_mean=_HALF, norm_std=_HALF, extract_fn=_kaiko_extract)

    def create_model(self):
        from .kaiko_midnight import create_kaiko_midnight_model  # noqa: PLC0415

        return create_kaiko_midnight_model()


_DYN = {"dynamic_img_size": True, "dynamic_img_pad": True}

# Virchow / Virchow2: concat(CLS, mean of the patch tokens), 2 x 1280 (the model cards' embedding, PRISM's input)
_VIRCHOW_HEADS = {"concat": TileHead(key_suffix="_2560", fn=virchow_concat)}

_BUILTIN: list[TilePreset] = [
    TimmHubPreset("uni", "hf-hub:MahmoodLab/uni", dynamic_img_size=True, init_values=1e-5),
    TimmHubPreset(
        "uni2",
        "hf-hub:MahmoodLab/UNI2-h",
        swiglu=True,
        img_size=224,
        patch_size=14,
        depth=24,
        num_heads=24,
        init_values=1e-5,
        embed_dim=1536,
        mlp_ratio=2.66667 * 2,
        num_classes=0,
        no_embed_class=True,
        reg_tokens=8,
        **_DYN,
    ),
    TimmHubPreset("gigapath", "hf_hub:prov-gigapath/prov-gigapath", **_DYN),
    GigapathFlashPreset(),
    TimmHubPreset("virchow", "hf-hub:paige-ai/Virchow", swiglu=True, heads=_VIRCHOW_HEADS, **_DYN),
    TimmHubPreset("virchow2", "hf-hub:paige-ai/Virchow2", swiglu=True, heads=_VIRCHOW_HEADS, **_DYN),
    TimmHubPreset(
        "h-optimus-0", "hf-hub:bioptimus/H-optimus-0", norm=(_HOPTIMUS_MEAN, _HOPTIMUS_STD), init_values=1e-5, **_DYN
    ),
    Conch15Preset(),
    MidnightPreset(),
    PhikonPreset("phikon2", "v2"),
    HibouPreset("b"),
    HibouPreset("l"),
    # Distilled from H-optimus-0; timm config + weights on the hub, SwiGLU / SiLU as in the model card
    TimmHubPreset("h0-mini", "hf-hub:bioptimus/H0-mini", norm=(_HOPTIMUS_MEAN, _HOPTIMUS_STD), swiglu=True, **_DYN),
    PhikonPreset("phikon", "v1"),
    KaikoMidnightPreset(),
]

_REGISTRY: dict[str, TilePreset] = {p.name: p for p in _BUILTIN}

PRESET_NAMES: list[str] = list(_REGISTRY)

# Deprecated names kept for old scripts / HDF5 files: alias -> (preset name, outputs it stands for).
# 'conch15_768' writes the proj output under the storage key itself ({model}/features, no '_768' suffix),
# so `-M` omitted still lands in conch15_768/features as before.
PRESET_ALIASES: dict[str, tuple[str, tuple[str, ...]]] = {
    "conch15_768": ("conch15", ("proj",)),
}

_ALIAS_PRESETS: dict[str, TilePreset] = {
    "conch15_768": Conch15Preset(proj_suffix="", default_outputs=("proj",)),
}


def get_tile_preset(name: str) -> TilePreset:
    """Look up a built-in tile preset by name. Does not instantiate the model.

    A deprecated alias (``PRESET_ALIASES``) warns and returns the preset it stands for, with its outputs as
    ``default_outputs`` (e.g. ``'conch15_768'`` -> ``conch15`` writing ``proj`` to ``{model}/features``).

    Raises:
        ValueError: unknown name (message lists PRESET_NAMES).
    """
    if name in _REGISTRY:
        return _REGISTRY[name]
    if name in _ALIAS_PRESETS:
        target, outputs = PRESET_ALIASES[name]
        msg = (
            f"preset {name!r} is deprecated: use preset={target!r} with outputs={outputs!r} "
            f"(written as preset={target!r}, output={outputs[0]!r})"
        )
        warnings.warn(msg, DeprecationWarning, stacklevel=2)
        logger.warning(msg)
        return _ALIAS_PRESETS[name]
    raise ValueError(f"Invalid preset: {name}. Must be one of {PRESET_NAMES}")


def is_tile_preset_name(name: str) -> bool:
    """True for a built-in preset name or a deprecated alias."""
    return name in _REGISTRY or name in _ALIAS_PRESETS


def canonical_tile_source(preset: str, output: str | None = None) -> tuple[str, str]:
    """``(preset, output)`` attrs of an HDF5 feature group -> canonical ``(preset, output)``.

    Groups written before ``output`` existed have no ``output`` attr: an alias preset name maps to what it
    stands for (``'conch15_768'`` -> ``('conch15', 'proj')``), anything else to ``(preset, 'cls')``.
    """
    if output:
        return preset, output
    if preset in PRESET_ALIASES:
        target, outputs = PRESET_ALIASES[preset]
        return target, outputs[0]
    return preset, "cls"


# --- thin compatibility layer derived from TilePreset -------------------------------


def _all_presets() -> dict[str, TilePreset]:
    return {**_REGISTRY, **_ALIAS_PRESETS}


def create_preset_model(preset: str):
    """Create a tile-level foundation model instance by preset name (compat for ``get_tile_preset(name).create_model()``)."""
    return get_tile_preset(preset).create_model()


PRESET_NORMALIZATION: dict[str, tuple[tuple[float, ...], tuple[float, ...]]] = {
    name: (p.norm_mean, p.norm_std) for name, p in _all_presets().items()
}

# Presets whose default features are not the CLS token of forward_features: name -> fn(model, x)
PRESET_EXTRACT_FN: dict[str, Callable[[Any, Any], Any]] = {
    name: p for name, p in _all_presets().items() if p.extract_fn is not None or p.default_outputs != ("cls",)
}


__all__ = [
    "BASE_OUTPUTS",
    "PRESET_NAMES",
    "PRESET_ALIASES",
    "TileHead",
    "TilePreset",
    "TimmHubPreset",
    "Conch15Preset",
    "get_tile_preset",
    "is_tile_preset_name",
    "canonical_tile_source",
    "create_preset_model",
    "PRESET_NORMALIZATION",
    "PRESET_EXTRACT_FN",
]
