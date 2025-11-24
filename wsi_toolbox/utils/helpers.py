"""
Helper utility functions for WSI processing
"""

import numpy as np


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
