# Changelog

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
