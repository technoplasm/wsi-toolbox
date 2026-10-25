"""Virchow / Virchow2 (Paige): the model cards' tile embedding.

Both model cards define the tile embedding as concat(CLS token, mean of the patch tokens) = 2 x 1280 = 2560 dims,
computed from the token outputs of the timm model. Virchow2 has 4 register tokens after the CLS token; they are
left out of the mean (``tokens[:, 5:]``), as in its model card. The Virchow (v1) 2560-dim embedding is the input of
PRISM (``presets.slide.prism``).
"""

import torch


def virchow_concat(model, tokens: torch.Tensor) -> torch.Tensor:
    """``(B, 2D)`` = concat(CLS, mean of the patch tokens) from ``tokens = model.forward_features(x)``.

    The patch tokens follow the CLS token and the registers (``model.num_prefix_tokens``: 1 for Virchow, 5 for
    Virchow2).
    """
    patch_tokens = tokens[:, model.num_prefix_tokens :]
    return torch.cat([tokens[:, 0], patch_tokens.mean(dim=1)], dim=-1)
