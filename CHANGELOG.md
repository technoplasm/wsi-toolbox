# Changelog

## 0.7.0 (2026-10-23)

Breaking release: `TileEncoder.encode()` returns a dict (`enc.encode(b)['cls']` for the old features).

### Added

- `outputs=` on `FeatureExtractionCommand` / `TileEncoder` (CLI `wt extract --outputs cls,latent,proj`): pick the
  outputs saved from one forward pass. `cls` (default) -> `{model}/features`, `latent` -> `{model}/latent_features`
  (only together with `cls`), `proj` (`conch15`: CONCH v1.5 attn-pool + LayerNorm, 768 dims, TITAN input) ->
  `{model}_768/features`. `wt extract --preset conch15 --outputs cls,proj` writes the 1024- and 768-dim features
  in one run. Each group gets an `output` attr next to `preset`; `FeatureExtractResult` has `outputs` and
  `feature_dims`.
- `TilePreset` is a class: built-in presets are subclasses (`Conch15Preset`, ...) that own their model factory,
  normalization and heads. `preset(model, x)` gives the default output, `preset.outputs(model, x, names)` the
  selected ones as a dict. New: `TileHead`, `PRESET_ALIASES`. `TilePreset(name=..., create_model=...,
  extract_fn=...)` for custom models works as before.

### Changed

- `TileEncoder.encode()` returns `{"cls": ..., "latent": ..., "proj": ...}` (only the selected outputs) instead of
  `(features, latent | None)`.
- `TilePreset` is no longer a frozen dataclass (no `==` / `hash` by value).
- TITAN's tile source is `(preset, output) = ("conch15", "proj")` (`SLIDE_PRESET_TILE_SOURCES`); groups written
  with `preset="conch15_768"` are still picked up.
- With `accel="compile"` / `"graphs"` the compiled function now includes the CLS slice and the heads; eager
  features are bit-identical to 0.6.4.

### Deprecated

- `with_latent=True` / CLI `-L`: use `outputs=("cls", "latent")`.
- Preset `conch15_768`: an alias of `conch15` with `outputs=("proj",)` (still written to `conch15_768/features`);
  moved from `PRESET_NAMES` to `PRESET_ALIASES`.

## 0.6.4 (2026-10-20)

### Added

- Philips iSyntax (`.isyntax`) slides: `ISyntaxFile` (a `PyramidalWSIFile`, read with `pyisyntax` / libisyntax,
  new dependency; openslide does not read iSyntax). `create_wsi_file` picks it for `.isyntax` (or
  `engine="isyntax"`), so DZI tiles, `read_region_at_mpp`, thumbnails, `WSIPatchReader` (`reopen()` per read
  worker), `CacheCommand` and `PyramidCommand` take iSyntax like any other slide. Levels are the wavelet scales
  (downsample `2 ** scale`); regions are clamped to the level and returned as RGB (libisyntax's RGBA is opaque,
  missing background tiles come back white). One instance per thread, like `PyramidalTiffFile`; any thread may
  use one (pyisyntax 0.1.7 segfaults on threads other than the one that initialised libisyntax, so each thread
  is given libisyntax's scratch arena first and frees it on exit). `.isyntax` is in `WSI_EXTENSIONS` and the
  watcher's slide glob.

### Changed

- `PyramidCommand` over iSyntax: testslide.isyntax (37382 x 73222) 41 s -> 12 s, with the same bytes, and the
  same for every `concurrency` (openslide inputs are read as before; their output is byte-identical too). Inputs
  that go through `create_wsi_file` and can be `reopen`ed are read in `concurrency` contiguous column blocks,
  each with its own handle and thread, which run ahead into the next bands (before: one `read_region` of the
  whole band on one thread). `PyramidalWSIFile.reopen(read_width=)` tells the new handle how wide its reads are.
- `ISyntaxFile` assembles regions from whole libisyntax tiles through a small LRU of decoded tiles (exactly the
  pixels of libisyntax's `read_region`, which decoded again the tiles a band or DZI tile shares with the next
  one; DZI tiles ~30 % faster), and sizes libisyntax's coefficient cache for the width of its reads (20 tiles
  per tile column; with fewer, the ancestors are evicted before their next child and every tile repeats the
  inverse-DWT cascade: pyisyntax's fixed 2000 tiles thrashed on full-width reads of this slide).
- `PyramidCommand` default `concurrency` 8 -> 4 (also the `pyramid` CLI's `--concurrency`): on testslide.isyntax
  4 threads build the pyramid about as fast as 8 (12.5 s vs 11.4-12.0 s) for 70 s of CPU instead of 107-112 s.
  The output does not depend on `concurrency`.

### Fixed

- `create_wsi_file` opens vendor TIFFs that openslide recognises (`OpenSlide.detect_format` other than
  `generic-tiff`, e.g. Philips TIFF exported from iSyntax) with openslide instead of tifffile. Through tifffile a
  Philips TIFF had no mpp (`get_mpp` failed: it is in the XML ImageDescription), wrong level downsamples (the
  pages are padded to whole tiles: 3.96 / 7.74 / 14.8 instead of 4 / 8 / 16, and the levels stopped at 16x) and
  black squares for the tiles missing from sparse slides. Plain pyramidal TIFFs (`generic-tiff`, e.g.
  `PyramidCommand`'s pyramid.tif) and TIFFs openslide does not open stay on tifffile.
- `OpenSlideFile` composites transparent pixels onto `openslide.background-color` (white when absent) instead of
  dropping alpha, so missing tiles of sparse slides (Philips TIFF, MIRAX) and areas outside the slide read as
  background, not black.

## 0.6.3 (2026-10-02)

### Added

- Tile presets `hibou-b` / `hibou-l` (HistAI Hibou-B / Hibou-L, DINOv2 ViT-B/14 and ViT-L/14 with 4 registers,
  CLS token, Hibou normalization). The transformers weights are renamed into a timm `VisionTransformer` and
  strict-loaded, so neither `trust_remote_code` (whose code no longer imports on transformers 5) nor the network
  is needed. Both match the model card's `pooler_output` (cos 1.000000 at 224 px, > 0.99999 at 252 px).
- Tile preset `h0-mini` (Bioptimus H0-mini, ViT-B/14 distilled from H-optimus-0, timm hub config as in the model
  card, CLS token as recommended there, H-optimus-0 normalization = the hub config's `pretrained_cfg`).
  Identical to the model card's code at 224 / 252 px (256 px is padded to 266, as for `h-optimus-0`).
- Tile preset `phikon` (Owkin Phikon v1, iBOT ViT-B/16, transformers `ViTModel` with position-embedding
  interpolation, CLS token, ImageNet normalization).
- Tile preset `kaiko-midnight` (kaiko.ai Midnight-12k, DINOv2 ViT-g/14, transformers `Dinov2Model`,
  normalization 0.5 / 0.5). Outputs the model card's classification embedding, concat(CLS, mean of patch
  tokens) = 3072 dims, through `extract_fn` (no latent). Not the same model as `midnight` (OpenMidnight).
- `OpenSlideFile` repairs regions openslide cannot decode (e.g. `Corrupt JPEG data` in one tile of an NDPI
  level) instead of failing the whole read: it reopens the handle (openslide refuses every read after an
  error), reads the region again block by block, reads a block that still fails from level 0 and downscales
  it (`Image.BOX`), and fills a block that fails at level 0 too with white. `WSIPatchReader` sets the block to
  its patch size and drops unreadable patches regardless of the white detector. Every repair goes to a
  `ReadRepairLog` shared with the `reopen()`ed read-worker handles.
- `WSIPatchReader.metadata` over an `OpenSlideFile` has `level0_fallback_tiles` and `unreadable_tiles`
  (JSON lists of `{level, x, y, w, h, error}` in `level_used` pixels, `"[]"` when none), so
  `FeatureExtractionCommand` writes them to the model group's attrs. It also logs a warning and puts it in the
  `Writing` phase's progress message when any patch was repaired.

## 0.6.2 (2026-09-28)

### Changed

- `PyramidCommand` no longer needs libvips (neither the `vips` CLI nor pyvips). It builds the pyramid TIFF in the
  calling process, streaming: openslide reads level 0 in bands of `tile_size` rows (`concurrency` threads; inputs
  openslide does not open go through `create_wsi_file`), each level is the rounded 2x2 mean of the one above
  (OpenCV INTER_AREA on column chunks), imagecodecs encodes the JPEG tiles (`concurrency` threads, tables shared
  as JPEGTables) and tifffile writes the pages (lower levels are spooled to an unnamed temporary file while
  page 0 is written). Same pages, tile size, JPEG settings (Q85 4:2:0, full chroma from Q90), BigTIFF and
  resolution tags as `vips tiffsave --tile --pyramid` wrote; output size within 0.1 %, DZI tiles within
  0.06 mean abs error of the vips output; deterministic (same input and settings give the same bytes).
  Transparent pixels are composited over the slide's background colour. `_docs/benchmark-pyramid-dzi.md` §14.
- `should_cancel` is checked after every band (was polled every 0.2 s while vips ran).
- `concurrency` is now the reader / encoder thread count (was `VIPS_CONCURRENCY`).
- New dependency `openslide-bin` (openslide-python >= 1.4 loads it, so no system libopenslide is needed).

### Added

- `PyramidError` (exported as `wt.PyramidError`), `pyramid_sizes()`.

### Removed

- `PyramidCommand.build_args()` (there is no command line any more). `VipsError` stays as an alias of
  `PyramidError`, and `vips_available()` always returns True.

## 0.6.1 (2026-09-26)

### Added

- `TileEncoder` (`wsi_toolbox.encoder`, exported as `wt.TileEncoder`): the loaded tile model(s) plus the
  acceleration choice, built once and shared by many `FeatureExtractionCommand` calls (`encoder=`). One model
  copy per device, `encode()` splits a batch across GPUs and is thread-safe. `accel='none'` is the eager path
  of 0.6.0 (bit-identical features); `'compile'` is `torch.compile(dynamic=False)` and `'graphs'` adds CUDA
  graphs (`mode='reduce-overhead'`), both with batches padded to fixed buckets (64/128/256/512 by default,
  `buckets=`) so that every shape compiles once; `warmup()` compiles them up front. CUDA only (falls back to
  `'none'` with a warning on the CPU).
- `FeatureExtractionCommand(accel=..., encoder=...)`; `FeatureExtractResult.accel` and the H5 group attr
  `accel` record what ran. CLI `wt extract --accel none|compile|graphs`.
- `scripts/bench_pyramid.py extract --accel` for the `model` / `e2e` parts; `_docs/benchmark-pyramid-dzi.md`
  §12 (GB10: cu128 vs cu130, torch.compile / CUDA graphs, FP8 measured and rejected).

### Changed

- Linux installs get the `cu130` torch / torchvision wheels (both x86_64 and aarch64): the `cu128` wheels have no
  usable bf16 Tensor Core GEMM on Blackwell GB10 (sm_121), extract was 1.4x slower there.
- `[tool.uv] python-preference = "only-managed"`: `torch.compile` needs triton, which compiles against
  `Python.h`; the hosts' system Python 3.12 ships without headers.

### Fixed

- `fix_global_seed` (CLI) assigned `torch.use_deterministic_algorithms = True`, replacing the torch function
  with a bool; harmless in eager mode but `torch.compile` (dynamo) calls it. The assignment is removed.

### Removed

- `commands.feature_extraction._GPUWorker` (private) is replaced by `TileEncoder`.

## 0.6.0 (2026-09-25)

Breaking release: progress, preset and device become command arguments. Migration guide:
[`_docs/migration-0.6.md`](_docs/migration-0.6.md). Design: [`_docs/design-progress-events.md`](_docs/design-progress-events.md).

### Added

- `wsi_toolbox.progress` (public): `ProgressEvent`, `ProgressSink`, `Reporter`, `Cancelled`, `UNSET`, and the
  sinks `TqdmSink`, `RichSink`, `StreamlitSink`, `LoggingSink`, `MultiSink`, `NullSink`, plus `resolve_sink`.
- Every command's `__call__` takes keyword-only `on_progress=` (sink callable; not given → `defaults.progress`,
  `None` → silent) and `should_cancel=` (polled at phase boundaries and inside long loops; `True` → cleanup, then
  `Cancelled`).
- `presets.tile.TilePreset` and `get_tile_preset(name)`. Custom encoders are plain `TilePreset` instances passed
  as `preset=`.
- `common.Defaults` / `wt.defaults` / `get_defaults()`, `resolve_preset()`. `set_default_preset` accepts a
  `TilePreset`; `set_default_progress` accepts a sink callable or `None`.
- `FeatureExtractionCommand(preset=...)` is optional (falls back to `defaults.preset`, `ValueError` if unset) and
  accepts a `TilePreset`.
- `ClusterWithUmapCommand` feeds one `Reporter` to both children: one continuous phase stream, one `done`.
- `StandardImage` gains native-level readers so PNG/JPEG inputs work with the patch reader (used by the tests).
- `tests/` (pytest, CPU only, fake model): progress events, sinks, presets, extraction, cancellation, clustering,
  CLI `--help`. `uv run task test`.
- CLI `--progress none`.
- `PyramidCommand` (+ `PyramidResult`, `PyramidInfo`, `VipsError`, `vips_available`, `read_pyramid_info`) and CLI
  `pyramid`: convert a WSI into a DZI-optimised tiled pyramid TIFF (512 px JPEG Q85 tiles, BigTIFF) by running the
  `vips` CLI as a subprocess (optional feature; no Python dependency). Progress phase `Building pyramid`,
  cancellation kills vips, the output is written atomically (temp file + `os.replace`).
- `read_region_at_mpp(wsi, x, y, w, h, target_mpp, *, mpp=None)` (`wsi_toolbox.region`): a level-0 box
  rendered at a given µm/px from the best native level (same 1 % level tolerance as DZI; Lanczos, tiled, no
  seams). Replaces callers' use of the private `_get_native_levels` / `_read_native_region` for exports.
- `wsi_toolbox.dzi`: `DziLayout` (Deep Zoom geometry), `DziGenerator` (`.dzi` XML and on-demand tiles for any
  opened WSI), `DziTileNotFound`, `encode_tile`. `DziCommand` and the `get_dzi_*` methods use it.
- `scripts/bench_pyramid.py` and [`_docs/benchmark-pyramid-dzi.md`](_docs/benchmark-pyramid-dzi.md): pyramid TIFF vs
  original WSI benchmark (conversion, DZI tile latency per level, 1 vs N threads with a handle pool, patch reading
  with / without the white check, CPU vs I/O, cold / warm cache) with the 2026-09-24 results; moved from vision.
  `extract` / `extract-compare` modes: feature extraction split into reader / model / end-to-end, and feature
  similarity of two H5s at the same coordinates (original vs pyramid.tif).

### Changed

- Commands only *read* `defaults`; no code path in the library writes them. The CLI, Streamlit app and watcher pass
  preset / device / sink explicitly instead of calling `set_default_*`.
- `leiden_cluster(..., reporter: Reporter | None)` replaces `on_progress: Callable[[str], None]`; phases
  `PCA` / `KNN` / `Building graph` / `Leiden clustering` / `Finalizing`.
- `FeatureExtractionCommand` argument order: `model, preset, device, batch_size, ...` (`device` moved forward).
- `PRESET_NORMALIZATION`, `PRESET_EXTRACT_FN`, `create_preset_model` are now derived from `TilePreset`.
- `PyramidalTiffFile` reads plain tiled pages tile by tile (seek + read + `page.decode`, identical pixels) and caches
  pages / zarr arrays per level instead of rebuilding `page.aszarr()` + `zarr.open()` on every read. One instance
  per thread (documented).
- ptp white detection counts the channel range with `np.maximum` / `np.minimum` instead of `np.ptp(axis=2)`:
  about 9x faster, same decisions.
- `WSIPatchReader` aligns its row-strip reads to the level's native tile height (`align_reads=True`, new
  `native_tile_height(level)` on `PyramidalTiffFile` / `OpenSlideFile`) and keeps the last strip, so a 512 px tile
  of a pyramid.tif is decoded once instead of once per 256 px patch row. Same patches bit for bit (tests on the
  synthetic pyramid and, with `WT_TEST_WSI`, on real slides); a no-op where the tile height divides the patch size
  (NDPI 8 px rows, 256 px SVS tiles). Reader CPU -30 % on pyramid.tif.
- `FeatureExtractionCommand` uploads the uint8 batch and scales / normalises it on the device, and copies only the
  CLS token back (all tokens only with `with_latent`). Features are bit-identical; `infer` for GigaPath-Flash on an
  RTX 3090 goes from 1,370 to 2,370 patches/s (the CPU float conversion cost as much as the forward pass).
- `WSIPatchReader` reads row strips on a small pool of worker threads (`read_workers=`, default
  `min(4, CPUs // 2)`, `1` = the old single-threaded path), each with its own file handle (new
  `PyramidalWSIFile.reopen()` / `close()`); splitting, the white check and batch stacking run on the workers too.
  Rows still come out in order, at most `read_workers + 2` row groups in flight. Same patches, keep/drop decisions
  and features bit for bit. Extract on NDPI originals is no longer reader-bound: reader 1,370-1,570 -> 3,080-3,600
  grid patches/s, extract of a 55k-patch NDPI 37.8 s -> 16.6 s (GPU-bound). Also on `get_patch_reader`,
  `FeatureExtractionCommand` and CLI `extract --read-workers`.

### Fixed

- `PrefetchReader` stopped early (cancel, exception) now stops its producer thread and closes the inner reader
  instead of leaving it blocked on a full queue.

- DZI tiles always have the spec's size. Native levels whose downsample is only approximately 2^k (SVS 4.0001, odd
  sizes) gave 255 px and short edge tiles; overview levels read past the coarsest native level and came out
  partly black with openslide. Tiles at natively present 2x levels are unchanged.

- `PyramidCommand`: two runs writing the same output at once no longer collide. The temporary file was a fixed
  `.pyramid.tif.tmp`, so one run renamed or deleted it under the other (`FileNotFoundError` on `os.replace`) or
  read the other's half-written file. It is now unique per run (`.pyramid.tif.<pid>-<random>.tmp`, see
  `tmp_path_for` / `tmp_files_for`), the result shape is read from the run's own file before it is published,
  and leftovers of killed runs older than a day are removed by the next run.

### Removed

- `BaseProgress`, `register_progress`, `wsi_toolbox.utils.progress` (`TqdmProgress`, `RichProgress`,
  `StreamlitProgress`, `DummyProgress`, `_PROGRESS_REGISTRY`).
- `get_config`, `Config`, `common._get`, `common._progress`.
- `set_default_custom_preset`, `create_default_model`.
- `ClusteringCommand.__call__(..., progress=)`, `UmapCommand.__call__(..., progress=)`.
- Progress backend name `'dummy'` (use `'none'`).

## 0.5.1

- Add `gigapath-flash` tile preset.

## 0.5.0

- CLI split into modules with command descriptions; release docs.
