# WSI Toolbox

> **Note**: This package is currently unstable. API may change without notice.

A comprehensive toolkit for Whole Slide Image (WSI) processing, feature extraction, and clustering analysis.

## Installation

```bash
# From PyPI
pip install wsi-toolbox

# From GitHub (latest)
pip install git+https://github.com/technoplasm/wsi-toolbox.git
```

## Presets

### Tile presets (per-patch feature extractors)

| Preset | Arch | Params | Dim | HuggingFace |
|-------|------|--------|-----|-------------|
| `uni` | ViT-L/16 | 300M | 1024 | [MahmoodLab/UNI](https://huggingface.co/MahmoodLab/UNI) |
| `uni2` (default) | ViT-H/14 | 681M | 1536 | [MahmoodLab/UNI2-h](https://huggingface.co/MahmoodLab/UNI2-h) |
| `gigapath` | ViT-g/14 | 1.1B | 1536 | [prov-gigapath/prov-gigapath](https://huggingface.co/prov-gigapath/prov-gigapath) |
| `virchow` | ViT-H/14 | 632M | 1280 | [paige-ai/Virchow](https://huggingface.co/paige-ai/Virchow) |
| `virchow2` | ViT-H/14 | 632M | 1280 | [paige-ai/Virchow2](https://huggingface.co/paige-ai/Virchow2) |
| `h-optimus-0` | ViT-g/14 | 1.1B | 1536 | [bioptimus/H-optimus-0](https://huggingface.co/bioptimus/H-optimus-0) |
| `conch15` | ViT-L/16 | 300M | 1024 | [MahmoodLab/conchv1_5](https://huggingface.co/MahmoodLab/conchv1_5) |
| `conch15_768` | ViT-L/16 | 300M | 768 | [MahmoodLab/conchv1_5](https://huggingface.co/MahmoodLab/conchv1_5) |
| `midnight` | ViT-g/14 | 1.1B | 1536 | [SophontAI/OpenMidnight](https://huggingface.co/SophontAI/OpenMidnight) |
| `phikon2` | ViT-L/16 | 300M | 1024 | [owkin/phikon-v2](https://huggingface.co/owkin/phikon-v2) |

`conch15_768` outputs FC-projected features (not cls_token), intended for [TITAN](https://huggingface.co/MahmoodLab/TITAN) input.

### Slide presets (slide-level aggregators)

| Preset | Tile source | Dim | HuggingFace |
|-------|-------------|-----|-------------|
| `titan` | `conch15_768` | 768 | [MahmoodLab/TITAN](https://huggingface.co/MahmoodLab/TITAN) |

### Preset vs model

- `--preset` selects which foundation model to load (e.g. `uni`, `gigapath`). Only `preset` is a session-level default.
- `-M` / `--model` is the **HDF5 storage key** under which results are written. Defaults to `--preset`. Use a distinct value to keep multiple extractions of the same preset separate (e.g. different `--patch-size`).

```bash
wt extract -i sample.ndpi --preset uni                          # → uni/features
wt extract -i sample.ndpi --preset uni -M uni_224 -S 224        # → uni_224/features (same preset, 224 px)
```

**Setup**: These models require HuggingFace authentication. Accept the license on each model page, then:

```bash
huggingface-cli login
```

## GPU Configuration

Device selection is controlled by `--device` / `-D` (CLI) or `set_default_device()` (Python). Default is `auto`.

| Value | Behavior |
|-------|----------|
| `auto` (default) | Detect all GPUs. Multiple GPUs → parallel inference. Single GPU → `cuda:0`. No GPU → `cpu` (with warning) |
| `cuda:0` | Use GPU 0 only. Falls back to `cpu` if unavailable (with warning) |
| `cuda:1` | Use GPU 1 only |
| `cuda:0,1,3` | Use specified GPUs in parallel |
| `cpu` | CPU only |

```bash
wt extract -i sample.ndpi -D auto           # Auto-detect (default)
wt extract -i sample.ndpi -D cuda:0         # Single GPU
wt extract -i sample.ndpi -D cuda:0,1       # 2 GPUs in parallel
```

```python
wt.set_default_device('cuda:0,1')  # Use GPU 0 and 1
```

For the Streamlit app, set via environment variable:

```bash
WT_DEVICE=cuda:0 uv run task app
```

## Quick Start

```bash
# 1. Extract features from WSI
wt extract -i sample.ndpi -o sample.h5

# 2. Run clustering
wt cluster -i sample.h5

# 3. Generate preview image (requires sample.ndpi in same directory)
wt preview -i sample.h5
```

```python
import wsi_toolbox as wt

wt.set_default_preset('uni2')
wt.set_default_device('auto')

# 1. Extract
cmd = wt.FeatureExtractionCommand(model='uni2', preset='uni2', batch_size=256)
cmd('sample.h5', wsi_path='sample.ndpi')

# 2. Cluster
cluster_cmd = wt.ClusteringCommand(model='uni2', resolution=1.0)
cluster_cmd(['sample.h5'])

# 3. Preview
preview_cmd = wt.PreviewClustersCommand(model='uni2')
img = preview_cmd('sample.h5')
img.save('sample_preview.jpg')
```

**Important**: `preview` / `preview-score` commands require the original WSI file with the same stem in the same directory (e.g., `sample.h5` needs `sample.ndpi`).

## Commands

CLI is available as `wsi-toolbox` or `wt`. Each command has `--help`.

---

### extract

Extract patch embeddings from WSI using foundation models.

| CLI | Python |
|-----|--------|
| `wt extract -i sample.ndpi -o sample.h5` | `FeatureExtractionCommand()(h5_path, wsi_path=...)` |

```bash
wt extract -i sample.ndpi -o sample.h5
wt extract -i sample.ndpi --preset gigapath        # Use Gigapath
wt extract -i sample.ndpi --preset virchow2        # Use Virchow2
wt extract -i sample.ndpi --preset conch15_768     # CONCH v1.5 (768D, TITAN-ready)
wt extract -i sample.ndpi --preset midnight        # OpenMidnight
wt extract -i sample.ndpi -L                       # Include latent features
wt extract -i sample.ndpi -D cuda:0,1              # Multi-GPU parallel
```

```python
cmd = wt.FeatureExtractionCommand(batch_size=256, with_latent=True)
result = cmd('sample.h5', wsi_path='sample.ndpi')
# result.feature_dim, result.patch_count
```

---

### aggregate

Run a slide-level aggregator (e.g. TITAN) on tile features to produce a single slide embedding.

| CLI | Python |
|-----|--------|
| `wt aggregate -i sample.h5` | `AggregateCommand(slide_preset='titan', tile_model='conch15_768')('sample.h5')` |

```bash
# Auto-resolve: scans the h5 for a tile preset compatible with titan (= conch15_768)
wt aggregate -i sample.h5

# Explicit storage key (multiple compatible groups → required)
wt aggregate -i sample.h5 -M conch15_768
```

```python
cmd = wt.AggregateCommand(slide_preset='titan', tile_model='conch15_768')
result = cmd('sample.h5')
# → conch15_768/aggregates/titan/feature  (D=768)
```

Requires `conch15_768/features` to exist (run `wt extract --preset conch15_768 -S 512` first).

---

### cluster

Run Leiden clustering on embeddings.

| CLI | Python |
|-----|--------|
| `wt cluster -i sample.h5` | `ClusteringCommand()(['sample.h5'])` |

```bash
wt cluster -i sample.h5
wt cluster -i sample.h5 --resolution 0.5   # Fewer clusters
```

```python
cmd = wt.ClusteringCommand(resolution=1.0)
result = cmd(['sample.h5'])
# result.cluster_count, result.target_path
```

See [Advanced Usage](#advanced-usage) for multi-file clustering and sub-clustering.

---

### preview

Generate cluster overlay image. **Requires WSI with same stem**.

| CLI | Python |
|-----|--------|
| `wt preview -i sample.h5` | `PreviewClustersCommand()('sample.h5')` |

```bash
wt preview -i sample.h5
wt preview -i sample.h5 -f 1 2 3           # Filter to clusters 1,2,3
wt preview -i sample.h5 --size 32          # Smaller thumbnails
```

```python
cmd = wt.PreviewClustersCommand(size=64)
img = cmd('sample.h5', namespace='default')
img.save('preview.jpg')
```

---

### umap

Compute UMAP projection.

| CLI | Python |
|-----|--------|
| `wt umap -i sample.h5` | `UmapCommand()(['sample.h5'])` |

```bash
wt umap -i sample.h5
wt umap -i sample.h5 --show                # Display plot
wt umap -i sample.h5 --save                # Save plot
```

```python
cmd = wt.UmapCommand(n_neighbors=15, min_dist=0.1)
result = cmd(['sample.h5'])
# result.target_path → 'uni/default/umap'
```

---

### pca

Compute PCA projection.

| CLI | Python |
|-----|--------|
| `wt pca -i sample.h5` | `PCACommand()(['sample.h5'])` |

```bash
wt pca -i sample.h5
wt pca -i sample.h5 -n 2                   # 2 components
wt pca -i sample.h5 --show                 # Display plot
```

```python
cmd = wt.PCACommand(n_components=1, scaler='minmax')
result = cmd(['sample.h5'])
# result.target_path → 'uni/default/pca1'
```

---

### preview-score

Generate score heatmap overlay. **Requires WSI with same stem**.

| CLI | Python |
|-----|--------|
| `wt preview-score -i sample.h5 -n pca1` | `PreviewScoresCommand()('sample.h5', score_name='pca1')` |

```bash
wt preview-score -i sample.h5 -n pca1
wt preview-score -i sample.h5 -n pca1 --cmap viridis
wt preview-score -i sample.h5 -n pca1 --invert
```

```python
cmd = wt.PreviewScoresCommand(size=64)
img = cmd('sample.h5', score_name='pca1', cmap_name='jet')
img.save('pca_heatmap.jpg')
```

---

### show

Display HDF5 file structure.

| CLI | Python |
|-----|--------|
| `wt show -i sample.h5` | `ShowCommand()('sample.h5')` |

```bash
wt show -i sample.h5
wt show -i sample.h5 -v                    # Verbose
```

---

### thumb

Generate thumbnail from WSI.

| CLI | Python |
|-----|--------|
| `wt thumb -i sample.ndpi` | `wsi.generate_thumbnail()` |

```bash
wt thumb -i sample.ndpi
wt thumb -i sample.ndpi -w 1024            # Specify width
```

---

### dzi

Export WSI to Deep Zoom Image format (for OpenSeadragon).

| CLI | Python |
|-----|--------|
| `wt dzi -i sample.ndpi -o ./out` | `DziCommand()(wsi_path, output_dir, name)` |

```bash
wt dzi -i sample.ndpi -o ./output
wt dzi -i sample.ndpi -o ./output -t 512   # Tile size
```

---

### cache (optional)

Pre-cache patch images for repeated access:

```bash
wt cache -i sample.ndpi -o sample.h5
wt extract -i sample.h5   # Uses cache
wt preview -i sample.h5   # Uses cache
```

Structure:
```
cache/{patch_size}/
├── patches       # [N, H, W, 3] images
└── coordinates   # [N, 2] coords
```

---

### migrate

Migrate old HDF5 format to new format.

```bash
wt migrate -i sample.h5
wt migrate -i sample1.h5 sample2.h5      # Multiple files
```

---

## HDF5 File Structure

All data is stored in a single HDF5 file. Use `wt show -i sample.h5` to inspect.

### Root Attributes (Metadata)

```python
with h5py.File('sample.h5', 'r') as f:
    # WSI metadata
    f.attrs['original_mpp']      # Original microns per pixel
    f.attrs['original_width']    # Original width (px)
    f.attrs['original_height']   # Original height (px)

    # Default extraction grid (legacy/back-compat; per-preset values live on {model}/.attrs)
    f.attrs['mpp']
    f.attrs['patch_count']
    f.attrs['cols']
    f.attrs['rows']
```

### Tile features

Features are stored under `{model}/`. `model` (the storage key) defaults to the preset name (e.g. `uni`, `conch15_768`) but is a free string when `-M` is given.

```
{model}/                  attrs: preset, patch_size, target_mpp, mpp, cols, rows, patch_count
├── features                   # [N, D]
├── coordinates                # [N, 2] level-0 (x, y) in pixels
├── latent_features            # [N, L, D] optional (with -L flag)
├── aggregates/                # slide-level aggregator outputs
│   └── {slide_preset}/
│       └── feature            # [D_slide]
└── {namespace}/               # analysis results (see below)
```

Feature dim per tile preset: `uni: 1024`, `uni2: 1536`, `gigapath: 1536`, `virchow/2: 1280`, `h-optimus-0: 1536`, `conch15: 1024`, `conch15_768: 768`, `midnight: 1536`, `phikon2: 1024`.

```python
with h5py.File('sample.h5', 'r') as f:
    features = f['uni/features'][:]                              # (N, 1024)
    coords   = f['uni/coordinates'][:]                           # (N, 2)
    preset   = f['uni'].attrs['preset']                          # which foundation model
    slide    = f['conch15_768/aggregates/titan/feature'][:]      # (768,)
```

### Analysis Results (Hierarchical)

Results are stored under `{model}/{namespace}/`.

```
{model}/{namespace}/
├── clusters     # [N] cluster labels (int)
├── umap         # [N, 2] UMAP coordinates
└── pca1         # [N] PCA scores
```

**Namespace**:
- Single file: `default`
- Multi-file: `file1+file2+...` (auto-generated)

**Sub-clustering (filter hierarchy)**:

```
{model}/default/clusters                           # Base
{model}/default/filter/1+2+3/clusters              # Sub-cluster of 1,2,3
{model}/default/filter/1+2+3/filter/0+1/clusters   # Further nesting
```

See [Advanced Usage](#advanced-usage) for examples.

### Writing Status

Large datasets have a `writing` attribute (`True` during write, `False` when complete).

```python
if f['uni/features'].attrs.get('writing', False):
    raise RuntimeError('Dataset is incomplete')
```

## Advanced Usage

### Multi-file Joint Clustering

Cluster multiple WSIs together to find common patterns across samples.

```bash
# 1. Extract features from each WSI
wt extract -i sample1.ndpi -o sample1.h5
wt extract -i sample2.ndpi -o sample2.h5

# 2. Joint clustering (namespace auto-generated as "sample1+sample2")
wt cluster -i sample1.h5 sample2.h5

# 3. Analysis on joint clusters
wt pca -i sample1.h5 sample2.h5
wt umap -i sample1.h5 sample2.h5

# 4. Preview each file (uses shared cluster labels)
wt preview -i sample1.h5 -N sample1+sample2
wt preview -i sample2.h5 -N sample1+sample2
```

```python
# Joint clustering
cmd = wt.ClusteringCommand()
result = cmd(['sample1.h5', 'sample2.h5'])
# → namespace: 'sample1+sample2'
# → uni/sample1+sample2/clusters in both files
```

### Sub-clustering

Analyze a subset of clusters in more detail.

```bash
# Sub-cluster within clusters 1,2,3
wt cluster -i sample1.h5 sample2.h5 -f 1 2 3

# PCA/UMAP on filtered subset
wt pca -i sample1.h5 sample2.h5 -f 1 2 3
wt umap -i sample1.h5 sample2.h5 -f 1 2 3

# Preview filtered clusters
wt preview -i sample1.h5 -N sample1+sample2 -f 1 2 3
```

```python
# Sub-cluster
cmd = wt.ClusteringCommand(parent_filters=[[1, 2, 3]])
cmd(['sample1.h5', 'sample2.h5'])
# → uni/sample1+sample2/filter/1+2+3/clusters

# PCA on filtered subset
cmd = wt.PCACommand(parent_filters=[[1, 2, 3]])
cmd(['sample1.h5', 'sample2.h5'])
# → uni/sample1+sample2/filter/1+2+3/pca1
```

## Streamlit App

```bash
uv run task app

# Environment variables
WT_PRESET=gigapath WT_DEVICE=cuda:1 WT_PREFETCH=2 uv run task app
```

## Development

```bash
git clone https://github.com/technoplasm/wsi-toolbox.git
cd wsi-toolbox
uv sync

uv run wt --help
uv run task app
```

## License

MIT
