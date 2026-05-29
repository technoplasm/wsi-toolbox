"""
TITAN slide-level aggregator (MahmoodLab/TITAN).

Wraps the HuggingFace remote-code model with transformers 5.x compatibility
patches that are only applied when this module is imported.
"""

import logging

import torch
from transformers import AutoModel, modeling_utils

logger = logging.getLogger(__name__)

_HUB_ID = "MahmoodLab/TITAN"
_compat_patched = False


def _apply_compat_patches() -> None:
    """Apply transformers 5.x compatibility shims for TITAN's remote code.

    The shims are idempotent and broadly benign:
    - Tensor.item() returns 0.0 on meta tensors instead of raising. TITAN's
      VisionTransformer.__init__ calls .item() on torch.linspace results while
      transformers' from_pretrained runs the constructor under meta context.
    - PreTrainedModel.all_tied_weights_keys defaults to {}; transformers 5.x
      reads this attribute during finalize but TITAN's 4.x-style code never
      defines it.
    """
    global _compat_patched
    if _compat_patched:
        return

    _orig_item = torch.Tensor.item

    def _safe_item(self):
        if self.device.type == "meta":
            return 0.0
        return _orig_item(self)

    torch.Tensor.item = _safe_item

    cls = modeling_utils.PreTrainedModel
    cls.mark_tied_weights_as_initialized = lambda *a, **kw: None
    cls.all_tied_weights_keys = {}

    _compat_patched = True
    logger.debug("Applied transformers 5.x compat patches for TITAN")


def create_titan_model():
    """Load TITAN from the HuggingFace hub. Returns the bare model (no device move, no eval)."""
    _apply_compat_patches()
    return AutoModel.from_pretrained(_HUB_ID, trust_remote_code=True, low_cpu_mem_usage=False)


def aggregate(
    model,
    features: torch.Tensor,
    coords: torch.Tensor,
    patch_size_lv0: int,
) -> torch.Tensor:
    """Run TITAN's slide-level aggregation.

    Args:
        model: TITAN model (already on the target device, in eval mode)
        features: (N, D) or (B, N, D) tile features (CONCH 1.5, D=768)
        coords:   (N, 2) or (B, N, 2) level-0 (x, y) pixel coordinates
        patch_size_lv0: patch side length in level-0 pixels

    Returns:
        (B, D) slide embedding (B = 1 when features had no batch dim)
    """
    if features.ndim == 2:
        features = features.unsqueeze(0)
    if coords.ndim == 2:
        coords = coords.unsqueeze(0)
    with torch.inference_mode():
        return model.encode_slide_from_patch_features(features, coords, patch_size_lv0)
