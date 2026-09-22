# Migrating from wsi-toolbox 0.5 to 0.6

0.6 changes how a caller controls three things — **progress display**, **which foundation model**, and
**which device** — and how a run is **cancelled**. Everything else (command names, result types, HDF5 layout,
CLI subcommands and options) is unchanged. The design behind this is in [`design-progress-events.md`](design-progress-events.md).

In one sentence: *commands emit progress events instead of drawing bars, and take `preset=` / `device=` /
`on_progress=` / `should_cancel=` as arguments instead of reading a mutable global.*

## 1. Removed names and their replacements

| Removed in 0.6 | Use instead |
|----------------|-------------|
| `wt.BaseProgress`, `wt.register_progress`, `wsi_toolbox.utils.progress` (whole module, incl. `TqdmProgress` / `RichProgress` / `StreamlitProgress` / `DummyProgress`) | Pass a sink callable as `on_progress=`. Built-ins: `TqdmSink`, `RichSink`, `StreamlitSink`, `LoggingSink`, `MultiSink`, `NullSink` (see §3) |
| `wt.get_config()`, `wsi_toolbox.common.Config`, `common._config` | `wt.defaults` / `wt.get_defaults()` (a `Defaults` model with `preset`, `device`, `progress`, `cluster_cmap`, `verbose`) |
| `common._get(key, value)`, `common._progress(...)` | Gone. Commands resolve their own arguments (`resolve_preset`, `resolve_devices`, `resolve_sink`) |
| `wt.set_default_custom_preset(generator, norm_mean, norm_std, extract_fn)` | Build a `wt.TilePreset(...)` and pass it as `preset=` or to `wt.set_default_preset(...)` (§4) |
| `wt.create_default_model()` | `wt.resolve_preset(None).create_model()` for the default, or `wt.get_tile_preset(name).create_model()` / `wt.create_preset_model(name)` for a built-in |
| `Config.model_generator` / `norm_mean` / `norm_std` / `extract_fn` | The same fields on a `TilePreset` |
| `ClusteringCommand.__call__(paths, progress=BaseProgress)` | `ClusteringCommand.__call__(paths, *, on_progress=..., should_cancel=...)` |
| `UmapCommand.__call__(paths, progress=BaseProgress)` | `UmapCommand.__call__(paths, *, on_progress=..., should_cancel=...)` |
| `leiden_cluster(features, ..., on_progress: Callable[[str], None])` | `leiden_cluster(features, ..., reporter: Reporter | None)` |
| `set_default_progress('dummy')` | `set_default_progress('none')` or `set_default_progress(None)` |

Kept unchanged: `set_default_preset(name)`, `set_default_device(...)`, `set_default_progress('tqdm' | 'rich' | 'streamlit')`,
`set_default_cluster_cmap`, `set_verbose`, `resolve_devices`, `PRESET_NAMES`, `PRESET_NORMALIZATION`,
`PRESET_EXTRACT_FN`, `create_preset_model(name)`, all command / result class names, `Wsi2HDF5Command` alias.

## 2. Changed signatures

### `FeatureExtractionCommand`

```python
# 0.5
FeatureExtractionCommand(model: str, preset: str, batch_size=256, with_latent=False, overwrite=False,
                         device: str | None = None, patch_size=256, target_mpp=0.5, prefetch=1, white_detector=None)
cmd(hdf5_path, wsi_path=None)

# 0.6
FeatureExtractionCommand(model: str, preset: str | TilePreset | None = None, device: str | None = None,
                         batch_size=256, with_latent=False, overwrite=False, patch_size=256, target_mpp=0.5,
                         prefetch=1, white_detector=None)
cmd(hdf5_path, wsi_path=None, *, on_progress=UNSET, should_cancel=None)
```

- `preset` is now optional and also accepts a `TilePreset`. `None` falls back to `wt.defaults.preset`; if that is
  unset too the call raises `ValueError` (0.5 raised `RuntimeError` from `create_default_model`).
- `device` moved before `batch_size` in the positional order. If you passed arguments positionally, switch to
  keywords.

### Every other command

`__call__` gained the same two keyword-only arguments: `on_progress=UNSET` and `should_cancel=None`
(`CacheCommand`, `AggregateCommand`, `ClusteringCommand`, `ClusterWithUmapCommand`, `UmapCommand`, `PCACommand`,
`BasePreviewCommand` and subclasses, `DziCommand`). `ShowCommand` is unchanged. Constructor arguments are unchanged.

### `set_default_progress`

Accepts a sink name (`'tqdm'`, `'rich'`, `'streamlit'`, `'logging'`, `'none'`), a sink callable, or `None`.
The `'dummy'` name is gone (`'none'`).

### `set_default_preset`

Accepts a `TilePreset` as well as a name. It no longer instantiates anything — the model is created inside the
command, on each call.

## 3. Progress: from `BaseProgress` to events

0.5 asked you to *implement a tqdm-like object* and register it under a backend name. 0.6 *sends you events*.

```python
# 0.5
from wsi_toolbox import BaseProgress, register_progress

class MyProgress(BaseProgress):
    def __init__(self, iterable=None, total=None, desc="", **kw): ...
    def update(self, n=1): ...
    def set_description(self, desc, refresh=True): ...
    def set_postfix(self, ordered_dict=None, **kw): ...
    def close(self): ...

register_progress("mine", MyProgress)
wt.set_default_progress("mine")
cmd(hdf5_path)

# 0.6
def my_sink(event: wt.ProgressEvent) -> None:
    print(event.phase, event.n, event.total, event.message, event.done)

cmd(hdf5_path, on_progress=my_sink)          # per call
wt.set_default_progress(my_sink)             # or as the process default
```

`ProgressEvent` is a frozen dataclass:

| Field | Type | Meaning |
|-------|------|---------|
| `phase` | `str` | Stage name, e.g. `"Initializing model"`, `"Processing patches"`, `"PCA"`, `"Leiden clustering"`, `"Writing"` |
| `n` | `int` | Progress within the current phase (resets to 0 on a phase change) |
| `total` | `int \| None` | Size of the phase; `None` for phases without a known count |
| `elapsed` | `float` | Seconds since the command started |
| `message` | `str` | Supplementary text (what 0.5 put in `set_postfix`) |
| `done` | `bool` | `True` on the single final event of a command |
| `fraction` | property `float \| None` | `n / total` clamped to `[0, 1]`, `None` when `total` is `None` |

`dataclasses.asdict(event)` gives a JSON-ready dict (`fraction` is a property and is not included).

Which phases each command emits, in order, is tabulated in [README.md → Phase names](../README.md#phase-names).
Since each command's stream is fixed, a UI can map phase names to a step list. The stream ends with exactly one
`done=True` event, even when the command skipped its work (nothing to do → no phases, still one `done`).

**Semantics to be aware of:**

- `on_progress` *not given* → the process default (`resolve_sink(wt.defaults.progress)`, `tqdm` out of the box).
  `on_progress=None` → silent. The sentinel `wt.UNSET` distinguishes the two; you never need to pass it.
- Sinks are stateful (they own the current bar). `resolve_sink('tqdm')` returns a fresh instance each time and the
  command does that for you when you rely on the default; when you construct a sink yourself, make one per call
  (or write a stateless one).
- Bar-style sinks (`TqdmSink`, `RichSink`, `StreamlitSink`) close the previous bar when `phase` changes.
  Unbounded phases (`total=None`) show as an indeterminate bar.
- `LoggingSink(logger, every=5.0)` logs phase changes always and advances at most every `every` seconds.

## 4. Custom models: from `set_default_custom_preset` to `TilePreset`

```python
# 0.5
wt.set_default_custom_preset(
    generator=lambda: MyEncoder(),
    norm_mean=(0.5, 0.5, 0.5),
    norm_std=(0.5, 0.5, 0.5),
    extract_fn=lambda model, x: model(x),
)
cmd = wt.FeatureExtractionCommand(model='mine', preset='uni')   # preset was required but ignored for the model

# 0.6
my_preset = wt.TilePreset(
    name='mine',                                # stored in the HDF5 group attrs as 'preset'
    create_model=lambda: MyEncoder(),           # fresh module; the command moves it to the device and calls .eval()
    norm_mean=(0.5, 0.5, 0.5),
    norm_std=(0.5, 0.5, 0.5),
    extract_fn=lambda model, x: model(x),       # None -> forward_features(x)[:, 0]
)
cmd = wt.FeatureExtractionCommand(model='mine', preset=my_preset)
# or, notebook style:
wt.set_default_preset(my_preset)
cmd = wt.FeatureExtractionCommand(model='mine')
```

`TilePreset` is immutable and self-contained, so two commands with different custom models can coexist in one
process — the thing 0.5's single global generator could not do.

## 5. Cancellation

0.5 had no cancellation API; the only way was to raise from inside a progress callback. 0.6 makes it explicit:

```python
stop = threading.Event()

try:
    cmd(hdf5_path, should_cancel=stop.is_set)
except wt.Cancelled:
    ...
```

- `should_cancel` is polled at every phase boundary and, in the loops, after every batch (`extract`), row strip
  (`cache`), tile (`dzi`) or patch (`preview`).
- When it returns `True` the command **cleans up first** (deletes the partial `features` / `coordinates` / cache
  datasets, releases the GPU) and then raises `wt.Cancelled`. The message names the phase it was in.
- `Cancelled` is a plain `Exception` subclass, not `KeyboardInterrupt`, so it does not slip past `except Exception`.
- It is independent of `on_progress`: `should_cancel` works with `on_progress=None`.

## 6. Calling from a service (e.g. vision compute)

The 0.5 pattern — register a `callback` backend, smuggle the per-job callback through a contextvar, raise from
inside it to cancel, and rewrite `set_default_*` before each job — is replaced by plain arguments. Nothing is
process-global any more, so jobs can run concurrently in one process with different models and devices.

```python
import dataclasses
import json
import threading
import wsi_toolbox as wt


class NdjsonSink:
    """Write one JSON line per ProgressEvent to a stream (stdout, a pipe, a websocket adapter, ...)."""

    def __init__(self, stream):
        self._stream = stream

    def __call__(self, event: wt.ProgressEvent) -> None:
        self._stream.write(json.dumps(dataclasses.asdict(event)) + "\n")
        self._stream.flush()


def run_job(job, stream, cancel_flag: threading.Event):
    cmd = wt.FeatureExtractionCommand(
        model=job.model,
        preset=job.preset,        # explicit: never rely on wt.defaults in a service
        device=job.device,        # explicit: 'cuda:0', 'cuda:1', ...
        batch_size=job.batch_size,
    )
    try:
        return cmd(job.h5_path, wsi_path=job.wsi_path,
                   on_progress=NdjsonSink(stream),
                   should_cancel=cancel_flag.is_set)
    except wt.Cancelled:
        return None
```

Each line on the stream looks like:

```json
{"phase": "Processing patches", "n": 17, "total": 214, "elapsed": 42.8, "message": "256/256 (q=1/1 wait=0ms)", "done": false}
```

Throttling: `Reporter` emits on every advance by default. If a consumer cannot keep up, throttle inside the sink
(e.g. keep the last event per phase and flush on a timer), or wrap a `LoggingSink(every=...)` for logs. Phase
changes and the final step of a phase should never be dropped.

Checklist for a service moving from 0.5:

- [ ] Delete the `register_progress('callback', ...)` backend and the contextvar plumbing.
- [ ] Delete every `set_default_preset` / `set_default_device` / `set_default_progress` call in job code;
      pass `preset=` and `device=` to the command constructor instead.
- [ ] Pass `on_progress=` (your sink) and `should_cancel=` (your flag) to `__call__`.
- [ ] Replace "raise inside the callback" cancellation with `should_cancel` and catch `wt.Cancelled`.
- [ ] If you post-process progress, switch from tqdm-ish `(desc, n, total, postfix)` to `ProgressEvent` fields
      (`phase`, `n`, `total`, `message`, `done`).

## 7. CLI, Streamlit app, watcher

- CLI: `--progress` now accepts `rich` (default), `tqdm` or `none`. Other options are unchanged.
- The CLI, the Streamlit app and the watcher no longer mutate `wt.defaults` at import time; each passes its
  preset / device / sink to the commands it runs. If you imported `wsi_toolbox.cli` to get `rich` progress as a
  side effect, call `wt.set_default_progress('rich')` yourself.

## 8. Tests

`uv run task test` runs the pytest suite (CPU only, no model downloads). `tests/conftest.py` shows how to build a
fake `TilePreset` and a PNG "slide" — a useful template for testing your own integration without a GPU.
