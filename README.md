# WSI Toolbox

A comprehensive toolkit for Whole Slide Image (WSI) processing, feature extraction, and clustering analysis.

## Installation

### From PyPI

```bash
pip install wsi-toolbox
```

### For development

```bash
# Clone repository
git clone https://github.com/endaaman/WSI-toolbox.git
cd WSI-toolbox

# Install dependencies
uv sync
```

**Note**: For gigapath slide-level encoder (CLI only), install manually:
```bash
pip install git+https://github.com/prov-gigapath/prov-gigapath.git@5d77be0
pip install flash-attn einops fairscale
```

## Quick Start

### As a Python Library

```python
import wsi_toolbox as wt

# Basic workflow
wt.set_default_model('uni')
cmd = wt.Wsi2HDF5Command(patch_size=256)
result = cmd('input.ndpi', 'output.h5')
```

**See [README_API.md](README_API.md) for comprehensive API documentation** (detailed examples, command patterns, utilities, etc.)

### As a CLI Tool

```bash
# Convert WSI to HDF5
wsi-toolbox wsi2h5 --in input.ndpi --out output.h5 --patch-size 256

# Extract features
wsi-toolbox embed --in output.h5 --model uni

# Clustering
wsi-toolbox cluster --in output.h5 --resolution 1.0

# For all commands
wsi-toolbox --help
```

### Streamlit Web Application

```bash
uv run task app
```

## HDF5 File Structure

WSI-toolbox stores all data in a single HDF5 file:

```python
# Core data
'patches'                      # Patch images: [N, H, W, 3], e.g., [3237, 256, 256, 3]
'coordinates'                  # Patch pixel coordinates: [N, 2]

# Metadata
'metadata/original_mpp'        # Original microns per pixel
'metadata/original_width'      # Original image width (level=0)
'metadata/original_height'     # Original image height (level=0)
'metadata/image_level'         # Image level used (typically 0)
'metadata/mpp'                 # Output patch MPP
'metadata/scale'               # Scale factor
'metadata/patch_size'          # Patch size (e.g., 256)
'metadata/patch_count'         # Total patch count
'metadata/cols'                # Grid columns
'metadata/rows'                # Grid rows

# Model features (per model: uni, gigapath, virchow2)
'{model}/features'             # Patch features: [N, D]
                               #   uni: [N, 1024]
                               #   gigapath: [N, 1536]
                               #   virchow2: [N, 2560]
'{model}/latent_features'      # Latent features (optional): [N, K, K, D]
'{model}/clusters'             # Cluster labels: [N]

# Gigapath slide-level (CLI only)
'gigapath/slide_feature'       # Slide-level features: [768]
```

## Features

- WSI processing (.ndpi, .svs, .tiff → HDF5)
- Feature extraction (UNI, Gigapath, Virchow2)
- Leiden clustering with UMAP visualization
- Preview generation (cluster overlays, latent PCA)
- Type-safe command pattern with Pydantic results
- CLI, Python API, and Streamlit GUI

## Documentation

- [API Guide](README_API.md) - Comprehensive Python API documentation (日本語)
- [CLAUDE.md](CLAUDE.md) - Development guidelines

## Development

### Setup Development Environment

```bash
# Clone repository
git clone https://github.com/endaaman/wsi-toolbox.git
cd wsi-toolbox

# Install all dependencies
uv sync

# Install with optional gigapath support
uv sync --extra gigapath

# Install build tools
uv sync --group build
```

### Run Tests and Development Tools

```bash
# Run CLI
uv run wsi-toolbox --help

# Run Streamlit app
uv run task app

# Run watcher
uv run task watcher
```

### Build and Deploy

#### Build Package

```bash
# Clean previous builds
uv run task clean

# Build package
uv run task build
# or
python -m build

# Check package integrity
uv run task check
# or
python -m twine check dist/*
```

#### Deploy to PyPI

**Prerequisites**: Install build tools first
```bash
uv sync --group build
```

**Deploy**:
```bash
# Using deploy script (recommended)
./deploy.sh

# Or manually
python -m build
python -m twine check dist/*
python -m twine upload dist/*
```

**Note**: Configure your PyPI credentials before deploying:
```bash
# Create ~/.pypirc with your API token
# Or use environment variables
export TWINE_USERNAME=__token__
export TWINE_PASSWORD=<your-pypi-token>
```

## License

MIT
