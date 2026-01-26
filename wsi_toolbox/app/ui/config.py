"""
Application configuration and constants
"""

import os

# Environment-based configuration
BASE_DIR = os.getenv("BASE_DIR", "data")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "uni")

# Model configuration
MODEL_LABELS = {
    "uni": "UNI",
    "gigapath": "Prov-Gigapath",
    "virchow2": "Virchow2",
}
MODEL_NAMES_BY_LABEL = {v: k for k, v in MODEL_LABELS.items()}
MODEL_NAMES = list(MODEL_LABELS.keys())

# Processing constants
BATCH_SIZE = 256
PATCH_SIZE = 256
THUMBNAIL_SIZE = 64

# Clustering constants
DEFAULT_CLUSTER_RESOLUTION = 1.0
MAX_CLUSTER_RESOLUTION = 3.0
MIN_CLUSTER_RESOLUTION = 0.0
CLUSTER_RESOLUTION_STEP = 0.1
