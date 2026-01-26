"""
Page components for the Streamlit app
"""

from .hdf5 import render_mode_hdf5
from .wsi import render_mode_wsi

__all__ = ["render_mode_wsi", "render_mode_hdf5"]
