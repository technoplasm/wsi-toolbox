"""``TilePreset`` base class and ``TileHead``.

Kept free of torch / timm imports: presets are looked up (and validated) long before any model is built.
"""

import logging
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# ImageNet defaults
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Outputs every preset without a custom extract_fn can produce from forward_features
BASE_OUTPUTS = ("cls", "latent")


@dataclass(frozen=True)
class TileHead:
    """An extra output computed from the same forward pass as the CLS token.

    Attributes:
        key_suffix: Appended to the storage key (``{model}{key_suffix}/features``). Must not be empty
            for a head that can be selected together with ``cls``.
        fn: ``fn(model, tokens) -> (B, D)``; ``tokens`` is ``model.forward_features(x)`` (CLS first).
    """

    key_suffix: str
    fn: Callable[[Any, Any], Any]


class TilePreset:
    """Everything needed to run a tile encoder; built-in presets are subclasses.

    Use: ``preset.create_model()`` builds the module, ``preset(model, x)`` returns the default ``(B, D)``
    features, and ``preset.outputs(model, x, names)`` returns several outputs from one forward pass.

    Args:
        name: Preset name; written to the HDF5 group attrs as ``preset``.
        create_model: Factory returning a fresh ``torch.nn.Module``. Must NOT move it to a device or call
            ``.eval()``; the encoder does that. Subclasses override the ``create_model`` method instead.
        norm_mean / norm_std: Per-channel input normalization (RGB, 0-1 scale).
        extract_fn: ``fn(model, x) -> features``. When None the features are the CLS token of
            ``model.forward_features(x)`` (``[:, 0]``), and the model must expose
            ``patch_embed.proj.kernel_size`` for the ``latent`` output. With an ``extract_fn`` the only output
            is ``cls`` (= what ``extract_fn`` returns).
        heads: Extra outputs by name (see ``TileHead``); only without ``extract_fn``.
        default_outputs: Outputs written when the caller does not choose (default ``("cls",)``).
    """

    def __init__(
        self,
        name: str,
        create_model: Callable[[], Any] | None = None,
        norm_mean: tuple[float, float, float] = IMAGENET_MEAN,
        norm_std: tuple[float, float, float] = IMAGENET_STD,
        extract_fn: Callable | None = None,
        heads: Mapping[str, TileHead] | None = None,
        default_outputs: Iterable[str] = ("cls",),
    ):
        heads = dict(heads or {})
        if extract_fn is not None and heads:
            raise ValueError("extract_fn と heads は同時に指定できません: heads need forward_features tokens")
        for head_name in heads:
            if head_name in BASE_OUTPUTS:
                raise ValueError(f"head 名 {head_name!r} は予約済みです: reserved output names are {BASE_OUTPUTS}")
        self.name = name
        self._create_model = create_model
        self.norm_mean = tuple(norm_mean)
        self.norm_std = tuple(norm_std)
        self.extract_fn = extract_fn
        # plain dict (not a MappingProxyType): preset.outputs() runs inside torch.compile
        self.heads: dict[str, TileHead] = heads
        self.default_outputs: tuple[str, ...] = ()
        self.default_outputs = self.resolve_outputs(tuple(default_outputs))

    # --- model ---------------------------------------------------------------------------

    def create_model(self) -> Any:
        """Instantiate the model (not moved to a device, not in eval mode)."""
        if self._create_model is None:
            raise NotImplementedError(f"preset {self.name!r} has no create_model")
        return self._create_model()

    # --- outputs -------------------------------------------------------------------------

    @property
    def output_names(self) -> tuple[str, ...]:
        """Outputs this preset can produce."""
        if self.extract_fn is not None:
            return ("cls",)
        return BASE_OUTPUTS + tuple(self.heads)

    def resolve_outputs(self, outputs: Iterable[str] | None = None, *, with_latent: bool = False) -> tuple[str, ...]:
        """Validate an ``outputs`` selection; returns it deduplicated in ``output_names`` order.

        None -> ``default_outputs``. ``with_latent=True`` (deprecated) adds ``latent`` when the selection has
        ``cls`` and the preset supports it, otherwise it is ignored with a warning (the old behaviour).

        Raises:
            ValueError: empty selection, unknown output, ``latent`` without ``cls``, or two outputs that would
                be stored under the same key.
        """
        if outputs is None:
            selected = list(self.default_outputs)
        elif isinstance(outputs, str):
            selected = [outputs]
        else:
            selected = list(outputs)
        if not selected:
            raise ValueError("outputs が空です: choose at least one output")
        unknown = [o for o in selected if o not in self.output_names]
        if unknown:
            raise ValueError(
                f"preset {self.name!r} は output {unknown} に対応していません: available outputs {list(self.output_names)}"
            )
        if with_latent:
            logger.warning("with_latent is deprecated: use outputs=('cls', 'latent')")
            if "cls" in selected and "latent" in self.output_names:
                selected.append("latent")
            else:
                logger.warning(f"with_latent is not supported with preset {self.name!r} outputs {selected}; ignored")
        if "latent" in selected and "cls" not in selected:
            raise ValueError("output 'latent' は 'cls' と一緒のときだけ指定できます: add 'cls' to outputs")
        resolved = tuple(o for o in self.output_names if o in selected)
        keys = [self.key_suffix(o) for o in resolved if o != "latent"]
        if len(keys) != len(set(keys)):
            raise ValueError(f"outputs {list(resolved)} が同じ保存先になります: they share a storage key")
        return resolved

    def key_suffix(self, output: str) -> str:
        """Storage-key suffix of an output: ``""`` for ``cls`` / ``latent``, the head's ``key_suffix`` otherwise."""
        if output in BASE_OUTPUTS:
            return ""
        return self.heads[output].key_suffix

    def __call__(self, model: Any, x: Any) -> Any:
        """Default features ``(B, D)``: the first of ``default_outputs`` (normally ``cls``: ``extract_fn(model, x)``,
        or the CLS token of ``forward_features``)."""
        first = self.default_outputs[0]
        return self.outputs(model, x, (first,))[first]

    def outputs(self, model: Any, x: Any, names: Iterable[str]) -> dict[str, Any]:
        """Selected outputs from one forward pass, as device tensors.

        ``names`` must already be validated (``resolve_outputs``). ``cls`` / heads are ``(B, D)``,
        ``latent`` is the patch tokens ``(B, L, D)``.
        """
        names = tuple(names)
        if self.extract_fn is not None:
            return {"cls": self.extract_fn(model, x)}
        tokens = model.forward_features(x)
        out = {}
        for name in names:
            if name == "cls":
                out[name] = tokens[:, 0]
            elif name == "latent":
                latent_size = model.patch_embed.proj.kernel_size[0]
                out[name] = tokens[:, tokens.shape[1] - latent_size**2 :]
            else:
                out[name] = self.heads[name].fn(model, tokens)
        return out

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r}, outputs={list(self.output_names)})"
