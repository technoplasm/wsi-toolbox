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

**Note**: For gigapath slide-level encoder (CLI only):
```bash
# For flash-attn (requires CUDA, takes time to build)
uv sync --extra build
uv sync --extra build --extra compile

# Then install gigapath (uv-only dependency group, not available via PyPI)
uv sync --group gigapath
```

## Quick Start

### As a Python Library

```python
import wsi_toolbox as wt

# Basic workflow
wt.set_default_model_preset('uni')
cmd = wt.Wsi2HDF5Command(patch_size=256)
result = cmd('input.ndpi', 'output.h5')
```

**See [README_API.md](README_API.md) for comprehensive API documentation** (detailed examples, command patterns, utilities, etc.)

### As a CLI Tool

After `pip install wsi-toolbox`, the CLI is available as `wsi-toolbox` or `wt`.
For development, use `uv run wt`.

```bash
# Extract tile patches from WSI into HDF5
wt wsi2h5 -i input.ndpi -o output.h5

# Extract patch embeddings using foundation model
wt embed -i output.h5

# Run Leiden clustering on embeddings
wt cluster -i output.h5

# Compute UMAP projection
wt umap -i output.h5

# Compute PCA projection
wt pca -i output.h5

# Generate cluster overlay preview image
wt preview -i output.h5

# Generate PCA score heatmap preview
wt preview-pca -i output.h5 -n pca1

# Show HDF5 file structure
wt show -i output.h5

# Export WSI to Deep Zoom Image format
wt dzi -i input.ndpi -o ./output

# Generate thumbnail from WSI
wt thumb -i input.ndpi
```

Each subcommand has detailed help: `wt <subcommand> --help`

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

- [API Guide](README_API.md) - Python API documentation
- [CLAUDE.md](CLAUDE.md) - Development guidelines

## Development

### Setup Development Environment

```bash
# Clone repository
git clone https://github.com/endaaman/wsi-toolbox.git
cd wsi-toolbox

# Install all dependencies
uv sync

# Install with optional gigapath support (uv-only dependency group)
uv sync --group gigapath

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

### Code Quality (Linting)

This project uses [Ruff](https://docs.astral.sh/ruff/) for linting and code formatting.

```bash
# Install dev dependencies (includes ruff)
uv sync --group dev

# Check code quality
uv run ruff check wsi_toolbox/

# Auto-fix issues where possible
uv run ruff check wsi_toolbox/ --fix

# Format code
uv run ruff format wsi_toolbox/

# Check formatting without modifying files
uv run ruff format --check wsi_toolbox/
```

**Note**: Linting runs automatically on every push/PR via GitHub Actions.

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
