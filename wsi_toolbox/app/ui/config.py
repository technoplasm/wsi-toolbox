"""
Application configuration and constants
"""

import os

# Environment-based configuration
BASE_DIR = os.getenv("WT_BASE_DIR", "data")
MODEL = os.getenv("WT_MODEL", "uni2")
DEVICE = os.getenv("WT_DEVICE", "auto")
PREFETCH = int(os.getenv("WT_PREFETCH", "2"))

# Model configuration
MODEL_LABELS = {
    "uni2": "UNI2-h",
    "uni": "UNI",
    "gigapath": "Prov-GigaPath",
    "conch15": "Conch V1.5",
    "h-optimus-0": "H-optimus-0",
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
