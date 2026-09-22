# WSI Toolbox

A comprehensive toolkit for Whole Slide Image (WSI) processing, feature extraction, and clustering analysis.

> **0.6**: progress, preset and device are now passed to commands as arguments (see [Python API](#python-api)).
> Upgrading from 0.5? Read [`_docs/migration-0.6.md`](_docs/migration-0.6.md). Changes are listed in [`CHANGELOG.md`](CHANGELOG.md).

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
| `gigapath-flash` | ViT-S/16 | 22M | 384 | [prov-gigapath/prov-gigapath-flash](https://huggingface.co/prov-gigapath/prov-gigapath-flash) |
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

Device selection is controlled by `--device` / `-D` (CLI) or the `device=` argument of a command (Python).
`set_default_device()` sets the process-wide default that is used when `device=` is not given. Default is `auto`.

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
cmd = wt.FeatureExtractionCommand(model='uni2', preset='uni2', device='cuda:0,1')  # Use GPU 0 and 1
wt.set_default_device('cuda:0,1')  # or: process-wide default for commands called without device=
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

# Process-wide defaults (optional). Commands read them when preset= / device= are not given.
wt.set_default_preset('uni2')
wt.set_default_device('auto')

# 1. Extract (progress goes to a tqdm bar by default; see "Python API" for sinks and cancellation)
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

## Python API

Every command follows the same pattern: configuration in `__init__`, execution in `__call__`, a Pydantic
result object back. In 0.6 the three things a caller may want to control at run time are all **arguments**:

| What | Where | Fallback when omitted |
|------|-------|-----------------------|
| Foundation model | `preset=` in `__init__` (name or `TilePreset`) | `wt.set_default_preset(...)`; `ValueError` if neither is set |
| Device | `device=` in `__init__` | `wt.set_default_device(...)` (default `auto`) |
| Progress display | `on_progress=` in `__call__` (keyword-only) | `wt.set_default_progress(...)` (default `tqdm`) |
| Cancellation | `should_cancel=` in `__call__` (keyword-only) | never cancelled |

The library never writes the defaults itself; only `wt.set_default_*` does. Notebooks can set them once, services
should pass everything explicitly. The full list of names is in [`README_API.md`](README_API.md).

### Progress sinks

Commands do not draw progress bars. They emit `wt.ProgressEvent` values and a *sink* — any callable taking one
event — decides what to do with them:

```python
@dataclass(frozen=True)
class ProgressEvent:
    phase: str          # "Initializing model", "Processing patches", "UMAP", ...
    n: int              # progress within the phase
    total: int | None   # phase size, None when unknown
    elapsed: float      # seconds since the command started
    message: str = ""   # extra text (batch description etc.)
    done: bool = False  # True only for the final event
    # .fraction -> n / total, or None
```

Built-in sinks (all importable from `wsi_toolbox`):

| Sink | Use |
|------|-----|
| `TqdmSink(**tqdm_kwargs)` | One tqdm bar per phase. **Default** when nothing is given |
| `RichSink(console=None)` | `rich.progress` display; what the CLI uses |
| `StreamlitSink(container=None)` | `st.progress` per phase; used by the Streamlit app |
| `LoggingSink(logger=None, every=5.0, level=INFO)` | `phase [n/total] message` to a logger, at most once per `every` seconds |
| `MultiSink(*sinks)` | Fan one stream out to several sinks |
| `NullSink()` | Silence. `on_progress=None` does the same |
| `resolve_sink("tqdm" \| "rich" \| "streamlit" \| "logging" \| "none")` | Name → fresh sink instance |

```python
import logging
import wsi_toolbox as wt

cmd = wt.FeatureExtractionCommand(model='uni2', preset='uni2', device='cuda:0')

cmd('sample.h5', wsi_path='sample.ndpi')                                # TqdmSink (default)
cmd('sample.h5', wsi_path='sample.ndpi', on_progress=wt.RichSink())     # rich
cmd('sample.h5', wsi_path='sample.ndpi', on_progress=wt.NullSink())     # silent (or on_progress=None)

# tqdm on the terminal + throttled lines in a service log
sink = wt.MultiSink(wt.TqdmSink(), wt.LoggingSink(logging.getLogger('myservice'), every=5.0))
cmd('sample.h5', wsi_path='sample.ndpi', on_progress=sink)

# Streamlit: draw inside a container
# cmd('sample.h5', on_progress=wt.StreamlitSink(st.container()))

# Any callable works as a sink
def print_progress(event: wt.ProgressEvent) -> None:
    if event.done:
        print(f"done in {event.elapsed:.1f}s")
    elif event.total:
        print(f"{event.phase}: {event.n}/{event.total} {event.message}")
    else:
        print(f"{event.phase} ...")

cmd('sample.h5', wsi_path='sample.ndpi', on_progress=print_progress)

# Process-wide default for commands called without on_progress= (a name, a callable, or None)
wt.set_default_progress('rich')
wt.set_default_progress(print_progress)
wt.set_default_progress(None)   # silent
```

Sinks are stateful (they hold the current bar), so create a new one per command call; `resolve_sink` and the
defaults machinery already do that for you.

### Cancellation

Pass `should_cancel`, a zero-argument callable returning `True` when the command should stop. It is polled at
every phase boundary and, in the long phases, after every batch / row strip / tile / patch. When it returns
`True` the command raises `wt.Cancelled` **after** cleaning up its partial output (an interrupted `extract`
leaves no half-written `features` dataset behind, an interrupted `cache` removes the partial cache).

```python
import threading
import wsi_toolbox as wt

stop = threading.Event()          # set from another thread, a signal handler, a UI button, ...

cmd = wt.FeatureExtractionCommand(model='uni2', preset='uni2', device='cuda:0')
try:
    cmd('sample.h5', wsi_path='sample.ndpi', should_cancel=stop.is_set)
except wt.Cancelled as e:
    print('cancelled:', e)         # "cancelled during 'Processing patches'"
```

`should_cancel` works with `on_progress=None` too; the two are independent.

### Custom models (`TilePreset`)

A built-in preset is just a `wt.TilePreset` looked up by name (`wt.get_tile_preset('uni2')`). To run your own
encoder, build one yourself and pass it wherever a preset name is accepted — `preset=` of
`FeatureExtractionCommand`, or `wt.set_default_preset(...)`:

```python
import wsi_toolbox as wt

def create_my_model():
    # Return a fresh torch.nn.Module. Do NOT move it to a device or call .eval(); the command does that.
    import timm
    return timm.create_model('hf-hub:MahmoodLab/uni', pretrained=True, dynamic_img_size=True, init_values=1e-5)

my_preset = wt.TilePreset(
    name='my-uni',                          # written to {model}/.attrs['preset'] in the HDF5 file
    create_model=create_my_model,
    norm_mean=(0.485, 0.456, 0.406),        # input normalization (RGB, 0-1 scale); ImageNet by default
    norm_std=(0.229, 0.224, 0.225),
    extract_fn=None,                        # None: model.forward_features(x)[:, 0] (CLS token)
)

cmd = wt.FeatureExtractionCommand(model='my-uni', preset=my_preset, device='cuda:0')
cmd('sample.h5', wsi_path='sample.ndpi')
```

With `extract_fn=None` the module must expose `forward_features(x)` returning `(B, 1 + tokens, dim)` with the CLS
token first, and `patch_embed.proj.kernel_size` (used for `with_latent=True`). For any other interface give
`extract_fn=lambda model, x: ...` returning the `(B, dim)` features; latent extraction is then skipped.

### Phase names

Sinks receive the phase names below, in this order, for each command. A phase with a known `total` advances step
by step; the others emit a single event at `n=0`. `ClusterWithUmapCommand` runs UMAP then clustering through one
reporter, so its stream is the two lists back to back, ending in a single `done` event.

| Command | Phases (`total`) |
|---------|------------------|
| `FeatureExtractionCommand` | `Initializing model` → `Processing patches` (batches; `message` = reader stats) → `Writing` |
| `CacheCommand` | `Caching patches` (row strips; `message` = reader stats) |
| `AggregateCommand` | `Loading features` → `Initializing model` → `Aggregating` → `Writing` |
| `ClusteringCommand` | `Loading features` → `PCA` → `KNN` → `Building graph` → `Leiden clustering` → `Finalizing` → `Sorting clusters` (only with `sort_clusters=True`) → `Writing` |
| `UmapCommand` | `Loading features` → `UMAP` → `Writing` |
| `PCACommand` | `Loading features` → `PCA` → `Writing` |
| `ClusterWithUmapCommand` | `UmapCommand` phases, then `ClusteringCommand` phases |
| `Preview*Command` | `Rendering patches` (patches) |
| `DziCommand` | `Generating tiles` (tiles; `message` = `Level L: row r/R`) |
| `ShowCommand` | none (no `on_progress`) |

When a command skips its work (output already present and `overwrite=False`) no phase is emitted, but the final
`done=True` event still arrives.

### CLI progress

The CLI uses the same machinery. `--progress` selects the sink for every subcommand:

```bash
wt extract -i sample.ndpi --progress rich    # default
wt extract -i sample.ndpi --progress tqdm
wt extract -i sample.ndpi --progress none    # no bars (logs only)
```

## Commands

CLI is available as `wsi-toolbox` or `wt`. Each command has `--help`. Options shared by all subcommands:
`--preset`, `-M/--model`, `-D/--device`, `--progress rich|tqdm|none`, `--seed`, `-v`.

---

### extract

Extract patch embeddings from WSI using foundation models.

| CLI | Python |
|-----|--------|
| `wt extract -i sample.ndpi -o sample.h5` | `FeatureExtractionCommand(model='uni2', preset='uni2')(h5_path, wsi_path=...)` |

```bash
wt extract -i sample.ndpi -o sample.h5
wt extract -i sample.ndpi --preset gigapath        # Use Gigapath
wt extract -i sample.ndpi --preset gigapath-flash  # GigaPath-Flash (ViT-S, ~50x cheaper)
wt extract -i sample.ndpi --preset virchow2        # Use Virchow2
wt extract -i sample.ndpi --preset conch15_768     # CONCH v1.5 (768D, TITAN-ready)
wt extract -i sample.ndpi --preset midnight        # OpenMidnight
wt extract -i sample.ndpi -L                       # Include latent features
wt extract -i sample.ndpi -D cuda:0,1              # Multi-GPU parallel
```

```python
cmd = wt.FeatureExtractionCommand(model='uni2', preset='uni2', batch_size=256, with_latent=True)
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
| `wt cluster -i sample.h5` | `ClusteringCommand(model='uni2')(['sample.h5'])` |

```bash
wt cluster -i sample.h5
wt cluster -i sample.h5 --resolution 0.5   # Fewer clusters
```

```python
cmd = wt.ClusteringCommand(model='uni2', resolution=1.0)
result = cmd(['sample.h5'])
# result.cluster_count, result.target_path
```

See [Advanced Usage](#advanced-usage) for multi-file clustering and sub-clustering.

---

### preview

Generate cluster overlay image. **Requires WSI with same stem**.

| CLI | Python |
|-----|--------|
| `wt preview -i sample.h5` | `PreviewClustersCommand(model='uni2')('sample.h5')` |

```bash
wt preview -i sample.h5
wt preview -i sample.h5 -f 1 2 3           # Filter to clusters 1,2,3
wt preview -i sample.h5 --size 32          # Smaller thumbnails
```

```python
cmd = wt.PreviewClustersCommand(model='uni2', size=64)
img = cmd('sample.h5', namespace='default')
img.save('preview.jpg')
```

---

### umap

Compute UMAP projection.

| CLI | Python |
|-----|--------|
| `wt umap -i sample.h5` | `UmapCommand(model='uni2')(['sample.h5'])` |

```bash
wt umap -i sample.h5
wt umap -i sample.h5 --show                # Display plot
wt umap -i sample.h5 --save                # Save plot
```

```python
cmd = wt.UmapCommand(model='uni2', n_neighbors=15, min_dist=0.1)
result = cmd(['sample.h5'])
# result.target_path → 'uni2/default/umap'
```

---

### pca

Compute PCA projection.

| CLI | Python |
|-----|--------|
| `wt pca -i sample.h5` | `PCACommand(model='uni2')(['sample.h5'])` |

```bash
wt pca -i sample.h5
wt pca -i sample.h5 -n 2                   # 2 components
wt pca -i sample.h5 --show                 # Display plot
```

```python
cmd = wt.PCACommand(model='uni2', n_components=1, scaler='minmax')
result = cmd(['sample.h5'])
# result.target_path → 'uni2/default/pca1'
```

---

### preview-score

Generate score heatmap overlay. **Requires WSI with same stem**.

| CLI | Python |
|-----|--------|
| `wt preview-score -i sample.h5 -n pca1` | `PreviewScoresCommand(model='uni2')('sample.h5', score_name='pca1')` |

```bash
wt preview-score -i sample.h5 -n pca1
wt preview-score -i sample.h5 -n pca1 --cmap viridis
wt preview-score -i sample.h5 -n pca1 --invert
```

```python
cmd = wt.PreviewScoresCommand(model='uni2', size=64)
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

Feature dim per tile preset: `uni: 1024`, `uni2: 1536`, `gigapath: 1536`, `gigapath-flash: 384`, `virchow/2: 1280`, `h-optimus-0: 1536`, `conch15: 1024`, `conch15_768: 768`, `midnight: 1536`, `phikon2: 1024`.

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
cmd = wt.ClusteringCommand(model='uni2')
result = cmd(['sample1.h5', 'sample2.h5'])
# → namespace: 'sample1+sample2'
# → uni2/sample1+sample2/clusters in both files
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
cmd = wt.ClusteringCommand(model='uni2', parent_filters=[[1, 2, 3]])
cmd(['sample1.h5', 'sample2.h5'])
# → uni2/sample1+sample2/filter/1+2+3/clusters

# PCA on filtered subset
cmd = wt.PCACommand(model='uni2', parent_filters=[[1, 2, 3]])
cmd(['sample1.h5', 'sample2.h5'])
# → uni2/sample1+sample2/filter/1+2+3/pca1
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
uv run task test     # pytest; CPU only, no model downloads
uv run task lint
```

## License

MIT
