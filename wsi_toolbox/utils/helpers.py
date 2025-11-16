"""
Helper utility functions for WSI processing
"""

import numpy as np
import h5py


def is_white_patch(patch, rgb_std_threshold=7.0, white_ratio=0.7):
    """
    Check if a patch is mostly white/blank

    Args:
        patch: RGB patch (H, W, 3)
        rgb_std_threshold: Threshold for RGB standard deviation
        white_ratio: Ratio threshold for white pixels

    Returns:
        bool: True if patch is considered white/blank
    """
    # white: RGB std < 7.0
    rgb_std_pixels = np.std(patch, axis=2) < rgb_std_threshold
    white_pixels = np.sum(rgb_std_pixels)
    total_pixels = patch.shape[0] * patch.shape[1]
    white_ratio_calculated = white_pixels / total_pixels
    # print('whi' if white_ratio_calculated > white_ratio else 'use',
    #       'std{:.3f}'.format(np.sum(rgb_std_pixels)/total_pixels)
    #      )
    return white_ratio_calculated > white_ratio


def cosine_distance(x, y):
    """
    Calculate cosine distance with exponential weighting

    Args:
        x: First vector
        y: Second vector

    Returns:
        tuple: (distance, weight)
    """
    distance = np.linalg.norm(x - y)
    weight = np.exp(-distance / distance.mean())
    return distance, weight


def safe_del(hdf_file, key_path):
    """
    Safely delete a dataset from HDF5 file if it exists

    Args:
        hdf_file: h5py.File object
        key_path: Dataset path to delete
    """
    if key_path in hdf_file:
        del hdf_file[key_path]
