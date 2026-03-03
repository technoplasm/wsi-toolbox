import logging

MODEL_NAMES = ["uni", "uni2", "gigapath", "virchow2", "h-optimus-0"]

# ImageNet defaults
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

MODEL_NORMALIZATION: dict[str, tuple[tuple[float, ...], tuple[float, ...]]] = {
    "uni": (_IMAGENET_MEAN, _IMAGENET_STD),
    "uni2": (_IMAGENET_MEAN, _IMAGENET_STD),
    "gigapath": (_IMAGENET_MEAN, _IMAGENET_STD),
    "virchow2": (_IMAGENET_MEAN, _IMAGENET_STD),
    "h-optimus-0": ((0.707223, 0.578729, 0.703617), (0.211883, 0.230117, 0.177517)),
}

# Suppress noisy logs from huggingface_hub and timm
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("timm").setLevel(logging.WARNING)


def create_foundation_model(model_name: str):
    """
    Create a foundation model instance by preset name.

    Args:
        model_name: One of 'uni', 'gigapath', 'virchow2'

    Returns:
        torch.nn.Module: Model instance (not moved to device, not in eval mode)
    """
    # Lazy import: timm/torch are slow to load (~2s), defer until model creation
    import timm  # noqa: PLC0415
    import torch  # noqa: PLC0415
    from timm.layers import SwiGLUPacked  # noqa: PLC0415

    if model_name == "uni":
        return timm.create_model("hf-hub:MahmoodLab/uni", pretrained=True, dynamic_img_size=True, init_values=1e-5)

    if model_name == "uni2":
        return timm.create_model(
            "hf-hub:MahmoodLab/UNI2-h",
            pretrained=True,
            **{
                "img_size": 224,
                "patch_size": 14,
                "depth": 24,
                "num_heads": 24,
                "init_values": 1e-5,
                "embed_dim": 1536,
                "mlp_ratio": 2.66667 * 2,
                "num_classes": 0,
                "no_embed_class": True,
                "mlp_layer": timm.layers.SwiGLUPacked,
                "act_layer": torch.nn.SiLU,
                "reg_tokens": 8,
                "dynamic_img_size": True,
            },
        )

    if model_name == "gigapath":
        return timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True, dynamic_img_size=True)

    if model_name == "h-optimus-0":
        return timm.create_model(
            "hf-hub:bioptimus/H-optimus-0", pretrained=True, init_values=1e-5, dynamic_img_size=True
        )

    if model_name == "virchow2":
        return timm.create_model(
            "hf-hub:paige-ai/Virchow2", pretrained=True, mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU
        )

    raise ValueError(f"Invalid model_name: {model_name}. Must be one of {MODEL_NAMES}")
