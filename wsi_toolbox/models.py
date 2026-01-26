import logging

MODEL_NAMES = ["uni", "gigapath", "virchow2"]

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

    if model_name == "gigapath":
        return timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True, dynamic_img_size=True)

    if model_name == "virchow2":
        return timm.create_model(
            "hf-hub:paige-ai/Virchow2", pretrained=True, mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU
        )

    raise ValueError(f"Invalid model_name: {model_name}. Must be one of {MODEL_NAMES}")
