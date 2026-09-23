"""
White patch detection methods for WSI processing
"""

from typing import Callable

import cv2
import numpy as np


def _count_low_range_pixels(patch: np.ndarray, rgb_range_threshold) -> int:
    """Number of pixels whose channel range (max - min) is below ``rgb_range_threshold``.

    Same count as ``np.sum(np.ptp(patch, axis=2) < rgb_range_threshold)``, but for the usual
    uint8 RGB patch it takes the max / min over three channel views with element-wise
    ``np.maximum`` / ``np.minimum`` instead of a reduction over a length-3 axis, which is
    several times faster (the ptp check used to cap patch splitting). ``max - min`` cannot
    wrap in uint8 because ``max >= min``.
    """
    if patch.dtype == np.uint8 and patch.ndim == 3 and patch.shape[2] == 3:
        r, g, b = patch[:, :, 0], patch[:, :, 1], patch[:, :, 2]
        hi = np.maximum(np.maximum(r, g), b)
        lo = np.minimum(np.minimum(r, g), b)
        np.subtract(hi, lo, out=hi)
        return int(np.count_nonzero(hi < rgb_range_threshold))
    return int(np.sum(np.ptp(patch, axis=2) < rgb_range_threshold))


def is_white_patch_ptp(patch, white_ratio_threshold=0.9, rgb_range_threshold=20):
    """
    Check if a patch is mostly white/blank using PTP (peak-to-peak) method

    Args:
        patch: RGB patch (H, W, 3)
        white_ratio_threshold: Ratio threshold for white pixels (0-1)
        rgb_range_threshold: Threshold for RGB range (max-min, 0-255)

    Returns:
        bool: True if patch is considered white/blank
    """
    # white: RGB range (max-min) < rgb_range_threshold
    white_pixels = _count_low_range_pixels(patch, rgb_range_threshold)
    total_pixels = patch.shape[0] * patch.shape[1]
    white_ratio_calculated = white_pixels / total_pixels
    return white_ratio_calculated > white_ratio_threshold


def is_white_patch_otsu(patch, white_ratio_threshold=0.8):
    """
    Check if a patch is mostly white/blank using Otsu method

    Args:
        patch: RGB patch (H, W, 3)
        white_ratio_threshold: Ratio threshold for white pixels (0-1)

    Returns:
        bool: True if patch is considered white/blank
    """
    # Convert to grayscale and apply Otsu thresholding
    gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    white_pixels = np.sum(binary == 255)
    total_pixels = patch.shape[0] * patch.shape[1]
    white_ratio_calculated = white_pixels / total_pixels
    return white_ratio_calculated > white_ratio_threshold


def is_white_patch_std(patch, white_ratio_threshold=0.75, rgb_std_threshold=7.0):
    """
    Check if a patch is mostly white/blank using STD method

    Args:
        patch: RGB patch (H, W, 3)
        white_ratio_threshold: Ratio threshold for white pixels (0-1)
        rgb_std_threshold: Threshold for RGB standard deviation (0-255)

    Returns:
        bool: True if patch is considered white/blank
    """
    # white: RGB std < rgb_std_threshold
    rgb_std_pixels = np.std(patch, axis=2) < rgb_std_threshold
    white_pixels = np.sum(rgb_std_pixels)
    total_pixels = patch.shape[0] * patch.shape[1]
    white_ratio_calculated = white_pixels / total_pixels
    return white_ratio_calculated > white_ratio_threshold


def is_white_patch_green(patch, green_threshold=0.9):
    """
    Check if a patch is mostly white/blank using green channel method

    Args:
        patch: RGB patch (H, W, 3)
        green_threshold: Threshold for green channel mean

    Returns:
        bool: True if patch is considered white/blank
    """
    # Extract green channel and normalize
    green = patch[:, :, 1] / 255.0
    green_mean = np.mean(green)
    return green_mean > green_threshold


def create_white_detector(method: str, threshold: float | None = None) -> Callable[[np.ndarray], bool]:
    """
    Create white detection function from method name and threshold

    Args:
        method: Detection method ('ptp', 'otsu', 'std', 'green')
        threshold: Threshold value (None for method-specific default)

    Returns:
        Function that takes (H, W, 3) numpy array and returns bool

    Raises:
        ValueError: If method is invalid

    Example:
        >>> detector = create_white_detector('ptp', 0.85)
        >>> is_white = detector(patch)  # patch is (256, 256, 3) array
    """
    # Map method names to functions and default thresholds
    method_map = {
        "ptp": (is_white_patch_ptp, 0.9),
        "otsu": (is_white_patch_otsu, 0.8),
        "std": (is_white_patch_std, 0.75),
        "green": (is_white_patch_green, 0.9),
    }

    if method not in method_map:
        raise ValueError(f"Invalid method '{method}'. Must be one of {list(method_map.keys())}")

    func, default_threshold = method_map[method]
    actual_threshold = threshold if threshold is not None else default_threshold

    # Return curried function: patch -> bool
    return lambda patch: func(patch, actual_threshold)
