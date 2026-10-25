"""PRISM slide-level aggregator (paige-ai/Prism): the Perceiver resampler only, rebuilt in plain torch.

PRISM is a vision-language model; its slide embedding comes from the ``image_resampler`` (a Perceiver over the
Virchow v1 tile embeddings, 2560 dims each), the BioGPT text decoder is not needed for it. The HF repo ships
remote code that fails to build under transformers 5.x (the BioGPT part), so the resampler is reimplemented here
(same modules, same parameter names, same ops in the same order as the repo's ``perceiver.py``, without its
``environs`` / xformers / attention-score branches) and its tensors are read from ``model.safetensors``; nothing
else is read from the hub, so it works with ``HF_HUB_OFFLINE=1`` once the weights are cached.

Output: ``model(tile_embeddings)`` returns ``{"image_embedding": (B, 1280), "image_latents": (B, 512, 1280)}``
like ``Prism.slide_representations``. PRISM takes no coordinates.

The Perceiver shares modules across layers (the transformer block from ``share_tf_start_layer`` on, the
cross-attention block from ``share_xattn_start_layer`` on); the checkpoint stores each shared tensor once, under
its first name, so a plain ``load_state_dict`` reports the later names as missing. ``load_resampler_state_dict``
checks that those are exactly the aliases and nothing else is missing or extra.
"""

import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from torch import Tensor, nn
from torch.nn.attention import SDPBackend, sdpa_kernel

_HUB_ID = "paige-ai/Prism"
_PREFIX = "image_resampler."

# perceiver_config of paige-ai/Prism config.json
PRISM_PERCEIVER_CONFIG: dict[str, int] = {
    "latent_seq": 512,
    "latent_dim": 1280,
    "context_dim": 2560,
    "mhsa_heads": 8,
    "perceiver_depth": 8,
    "transformer_depth": 6,
    "share_xattn_start_layer": 1,
    "share_tf_start_layer": 0,
    "xattn_heads": 1,
    "mlp_mult": 1,
}


class CrossAttention(nn.Module):
    """Latent queries attend to the tile context (no context LayerNorm: PRISM builds it with ``c_norm=False``)."""

    def __init__(self, *, query_dim: int, context_dim: int, head_dim: int, heads: int):
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        self.inner_dim = head_dim * heads
        self.x_norm = nn.LayerNorm(query_dim)
        self.to_q = nn.Linear(query_dim, self.inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, self.inner_dim * 2, bias=False)
        self.to_out = nn.Linear(self.inner_dim, query_dim, bias=False)

    def forward(
        self,
        x: Tensor,
        c: Tensor | None = None,
        kvt: tuple[Tensor, Tensor] | None = None,
        attn_mask: Tensor | None = None,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        """Queries ``x`` over the context ``c``, or over a key-value cache ``kvt`` from the previous layer
        (exactly one of the two). Returns the output and the key-value pair."""
        Bx, Nx, _ = x.shape
        x = self.x_norm(x)
        q = self.to_q(x).reshape(Bx, Nx, self.heads, self.head_dim)

        if c is not None and kvt is None:
            Bc, Nc, _ = c.shape
            kv = self.to_kv(c).reshape(Bc, Nc, 2, self.heads, self.head_dim)
            k, v = kv.unbind(2)
            kvt = (k, v)
        elif kvt is not None and c is None:
            k, v = kvt
            Bc, Nc = k.shape[:2]
        else:
            raise ValueError("CrossAttention takes exactly one of c (context) and kvt (key-value cache)")

        if attn_mask is not None:
            attn_mask = attn_mask.reshape(Bc, 1, Nx, Nc).expand(-1, self.heads, -1, -1)

        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        with sdpa_kernel(SDPBackend.MATH):
            a = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        a = a.transpose(1, 2).reshape(Bx, Nx, self.inner_dim)
        return self.to_out(a), kvt


class MHSA(nn.Module):
    """Pre-norm multi-head self-attention over the latents."""

    def __init__(self, *, dim: int, num_heads: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.mha = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=0.0,
            bias=False,
            add_bias_kv=False,
            add_zero_attn=False,
            kdim=dim,
            vdim=dim,
            batch_first=True,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm(x)
        with sdpa_kernel([SDPBackend.MATH]):
            x, _ = self.mha(
                x,
                x,
                x,
                key_padding_mask=None,
                need_weights=False,
                attn_mask=None,
                average_attn_weights=False,
                is_causal=False,
            )
        return x


class GEGLU(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


class FeedForward(nn.Module):
    """Pre-norm GEGLU feed-forward."""

    def __init__(self, *, dim: int, mult: int = 1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Linear(dim * mult, dim),
            nn.Dropout(0.0),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(self.norm(x))


class Perceiver(nn.Module):
    """Latent queries alternating cross-attention to the context and a transformer over the latents.

    Layer ``i`` builds its own cross-attention block while ``i <= share_xattn_start_layer`` and its own transformer
    while ``i <= share_tf_start_layer``; later layers reuse the previous layer's module (the same object). A layer
    whose cross-attention is shared reuses the key-value cache of the previous layer instead of the context.
    """

    def __init__(
        self,
        *,
        latent_seq: int,
        latent_dim: int,
        context_dim: int,
        mhsa_heads: int,
        perceiver_depth: int,
        transformer_depth: int,
        share_xattn_start_layer: int,
        share_tf_start_layer: int,
        xattn_heads: int,
        mlp_mult: int,
    ):
        super().__init__()
        if perceiver_depth <= 0 or share_xattn_start_layer < 0 or share_tf_start_layer < 0:
            raise ValueError("perceiver_depth must be > 0 and the share_*_start_layer >= 0")
        self.share_xattn_start_layer = share_xattn_start_layer
        self.latent_seq = latent_seq
        self.latents = nn.Parameter(torch.randn(latent_seq, latent_dim))

        def xattn_block() -> nn.ModuleDict:
            return nn.ModuleDict(
                {
                    "xattn": CrossAttention(
                        query_dim=latent_dim,
                        context_dim=context_dim,
                        head_dim=latent_dim // xattn_heads,
                        heads=xattn_heads,
                    ),
                    "ff": FeedForward(dim=latent_dim, mult=mlp_mult),
                }
            )

        def transformer() -> nn.ModuleList:
            return nn.ModuleList(
                [
                    nn.ModuleDict(
                        {
                            "mhsa": MHSA(dim=latent_dim, num_heads=mhsa_heads),
                            "ff": FeedForward(dim=latent_dim, mult=mlp_mult),
                        }
                    )
                    for _ in range(transformer_depth)
                ]
            )

        layers: list[nn.ModuleDict] = []
        for i in range(perceiver_depth):
            layers.append(
                nn.ModuleDict(
                    {
                        "xattn": xattn_block() if i <= share_xattn_start_layer else layers[-1]["xattn"],
                        "tf": transformer() if i <= share_tf_start_layer else layers[-1]["tf"],
                    }
                )
            )
        self.layers = nn.ModuleList(layers)

    def forward(self, context: Tensor, attn_mask: Tensor) -> Tensor:
        """``context`` (B, N, context_dim), ``attn_mask`` (B, N) bool (True = a real tile) -> latents
        (B, latent_seq, latent_dim)."""
        B, N, _ = context.shape
        if attn_mask.ndim != 2 or attn_mask.shape[0] != B:
            raise ValueError(f"attn_mask must be (B, N) = ({B}, {N}), got {tuple(attn_mask.shape)}")
        attn_mask = attn_mask.reshape(B, 1, N).expand(-1, self.latent_seq, -1)

        x = self.latents.unsqueeze(0).expand(B, -1, -1)
        kvt = None
        for i, layer in enumerate(self.layers):
            xattn = layer["xattn"]
            if i <= self.share_xattn_start_layer:
                xattn_out, kvt = xattn["xattn"](x, c=context, kvt=None, attn_mask=attn_mask)
            else:
                # shares W_k / W_v with the previous layer: reuse its key-value cache
                xattn_out, kvt = xattn["xattn"](x, c=None, kvt=kvt, attn_mask=attn_mask)
            x = xattn_out + x
            x = xattn["ff"](x) + x
            for block in layer["tf"]:
                x = block["mhsa"](x) + x
                x = block["ff"](x) + x
        return x


class PerceiverResampler(nn.Module):
    """PRISM's ``image_resampler``: tile embeddings -> slide embedding (CLS latent) and image latents.

    Keyword arguments are the ``perceiver_config`` (defaults: ``PRISM_PERCEIVER_CONFIG``). One extra latent query is
    prepended for the slide embedding (``latent_seq + 1`` queries in all).
    """

    def __init__(self, **config: int):
        super().__init__()
        cfg = {**PRISM_PERCEIVER_CONFIG, **config}
        self.query_dim = cfg["latent_dim"]
        self.query_seq = cfg["latent_seq"]
        self.cls_norm = nn.LayerNorm(self.query_dim)
        self.img_norm = nn.LayerNorm(self.query_dim)
        self.perceiver = Perceiver(**{**cfg, "latent_seq": 1 + cfg["latent_seq"]})

    def forward(self, tile_embeddings: Tensor, tile_mask: Tensor | None = None) -> dict[str, Tensor]:
        """
        Args:
            tile_embeddings: (B, N, context_dim) tile embeddings per slide (Virchow v1 concat, 2560).
            tile_mask: (B, N) bool, True for real tiles; required when B > 1 (padded batches).

        Returns:
            ``{"image_embedding": (B, latent_dim), "image_latents": (B, latent_seq, latent_dim)}``
        """
        if tile_mask is None:
            if len(tile_embeddings) > 1:
                raise ValueError("tile_mask is required when the batch has more than one slide")
            tile_mask = torch.ones(tile_embeddings.shape[:2], device=tile_embeddings.device, dtype=torch.bool)
        x = self.perceiver(context=tile_embeddings, attn_mask=tile_mask)
        return {"image_embedding": self.cls_norm(x[:, 0]), "image_latents": self.img_norm(x[:, 1:])}


def load_resampler_state_dict(model: nn.Module, state_dict: dict[str, Tensor]) -> None:
    """Load a state dict that holds each shared tensor once, under its first name; strict otherwise.

    ``state_dict`` must have exactly the canonical names (the first name of every distinct parameter / buffer in
    ``model.state_dict()`` order). The names left over are aliases of shared modules (the same tensors), so they
    are filled by loading their canonical name.

    Raises:
        RuntimeError: a canonical name is absent, a name is unknown, or an alias is given instead of / besides
            its canonical name.
    """
    canonical: dict[int, str] = {}
    alias_of: dict[str, str] = {}
    for name, tensor in model.state_dict(keep_vars=True).items():
        first = canonical.setdefault(id(tensor), name)
        if first != name:
            alias_of[name] = first
    canonical_names = set(canonical.values())
    given = set(state_dict)
    absent = sorted(canonical_names - given)
    unknown = sorted(given - canonical_names - set(alias_of))
    aliases = sorted(given & set(alias_of))
    if absent or unknown or aliases:
        raise RuntimeError(
            f"state dict does not match the PRISM resampler: missing {absent[:5]} ({len(absent)}), "
            f"unexpected {unknown[:5]} ({len(unknown)}), aliases of shared modules {aliases[:5]} ({len(aliases)})"
        )
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if set(missing) != set(alias_of) or unexpected:
        raise RuntimeError(f"unexpected load result: missing {sorted(missing)[:5]}, unexpected {unexpected[:5]}")


def read_resampler_state_dict(path: str) -> dict[str, Tensor]:
    """The ``image_resampler.*`` tensors of a PRISM ``model.safetensors``, prefix removed (the rest is not read)."""
    with safe_open(path, framework="pt") as f:
        return {k[len(_PREFIX) :]: f.get_tensor(k) for k in f.keys() if k.startswith(_PREFIX)}


def create_prism_model() -> PerceiverResampler:
    """PRISM's resampler with the pretrained weights. Returns the bare model (no device move, no eval)."""
    model = PerceiverResampler()
    load_resampler_state_dict(model, read_resampler_state_dict(hf_hub_download(_HUB_ID, filename="model.safetensors")))
    return model


def aggregate(
    model: PerceiverResampler,
    features: Tensor,
    coords: Tensor,
    patch_size_lv0: int,
) -> Tensor:
    """Run PRISM's slide-level aggregation.

    Args:
        model: PRISM resampler (already on the target device, in eval mode)
        features: (N, D) or (1, N, D) tile features (Virchow v1 concat, D=2560)
        coords: ignored (PRISM takes no coordinates); same signature as ``titan.aggregate``
        patch_size_lv0: ignored

    Returns:
        (1, 1280) slide embedding (``image_embedding``)
    """
    if features.ndim == 2:
        features = features.unsqueeze(0)
    with torch.inference_mode():
        return model(features)["image_embedding"]
