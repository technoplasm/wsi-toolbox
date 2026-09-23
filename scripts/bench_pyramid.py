#!/usr/bin/env python3
"""
Benchmark a WSI against its DZI-optimised pyramid TIFF (``PyramidCommand``).

Measures, for each input slide ("orig") and its ``<stem>.pyramid.tif`` ("pyramid"):

- convert   ``PyramidCommand`` wall / CPU time and output size
- tiles     DZI tile latency p50/p95 and tiles/s per level (max = full resolution, mid = 1/8,
            low = 1/128), 1 thread: ``DziGenerator.tile`` + ``encode_tile`` (JPEG Q90)
- threads   full-resolution tiles with N threads: ``shared`` (one handle + lock) vs ``pool``
            (a simple per-thread handle pool, one handle lent to one thread at a time)
- patches   ``WSIPatchReader`` (256 px, 0.5 mpp) batches/s with the ptp white check
- patches-raw  the same without the white check (read + decode only)

Every case runs in its own child process (fresh openslide caches, rusage and
``/proc/self/io``) and records wall time, user+sys CPU, bytes read from the block layer
(``/proc/self/io`` read_bytes) and, on NFS, bytes the server sent (``/proc/self/mountstats``).
The page cache is prepared by the parent before each child:

- cold: ``fsync`` + ``posix_fadvise(POSIX_FADV_DONTNEED)`` on the file (no sudo; on NFS only
  the client cache is dropped, the server's stays warm; no effect on tmpfs)
- warm: the whole file is read once

No GPU, no model. Results are appended to a JSONL file and printed as Markdown tables.
Methodology and past results: ``_docs/benchmark-pyramid-dzi.md``.

Feature extraction (GPU, separate from ``run``; check ``nvidia-smi`` first -- nothing else should be
using the GPU) splits ``FeatureExtractionCommand`` into its two halves and the whole:

- extract --parts reader  ``get_patch_reader`` (prefetch thread + ptp white check) alone, patches/s,
                          with and without tile-aligned strip reads (``--align both``)
- extract --parts model   ``_GPUWorker.infer`` on synthetic uint8 batches (H2D + normalise + forward
                          + D2H) and the bare forward pass
- extract --parts e2e     ``FeatureExtractionCommand`` itself (cancelled after ``--budget`` s), with
                          the time spent inside ``infer`` ("GPU busy") vs the patch-processing phase
- extract-compare A.h5 B.h5   cosine similarity of the features of the same coordinates (e.g. the
                          original vs its pyramid.tif) and k-means(10) cluster agreement

Usage (repository root)::

    # OpenSlide public test slides -> data/bench/src (1.8 GB for all; --only to pick)
    uv run python scripts/bench_pyramid.py download
    uv run python scripts/bench_pyramid.py download --only Aperio_CMU-1.svs Hamamatsu_CMU-1.ndpi

    # everything on a folder of mixed formats (pyramids go to data/bench/pyramid, reused if present)
    uv run python scripts/bench_pyramid.py run data/bench/src --label ssd

    # pick modes / caps
    uv run python scripts/bench_pyramid.py run 'data/bench/src/*.ndpi' /mnt/nfs/x.svs \\
        --modes tiles,threads --levels max --caches cold --n 300 --tile-budget 15 --label nfs

    uv run python scripts/bench_pyramid.py info data/bench/src/Aperio_CMU-1.svs
    uv run python scripts/bench_pyramid.py report data/bench/results.jsonl [--run RUN_ID]

    # feature extraction: reader vs GPU vs end-to-end (writes data/bench/extract/<name>.h5)
    uv run python scripts/bench_pyramid.py extract slide.ndpi slide.pyramid.tif --parts reader,model,e2e
    uv run python scripts/bench_pyramid.py extract-compare data/bench/extract/slide.ndpi.h5 \\
        data/bench/extract/slide.pyramid.tif.h5
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import platform
import queue
import random
import resource
import statistics
import subprocess
import sys
import threading
import time
import urllib.request
import zipfile
from contextlib import contextmanager
from pathlib import Path

import h5py
import numpy as np
import openslide
import tifffile
import torch
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

import wsi_toolbox
from wsi_toolbox.commands import FeatureExtractionCommand
from wsi_toolbox.commands.feature_extraction import _GPUWorker
from wsi_toolbox.commands.pyramid import DEFAULT_CONCURRENCY, PyramidCommand, read_pyramid_info
from wsi_toolbox.dzi import DziGenerator, encode_tile
from wsi_toolbox.patch_reader import WSIPatchReader, get_patch_reader
from wsi_toolbox.presets.tile import get_tile_preset
from wsi_toolbox.progress import Cancelled
from wsi_toolbox.utils.white import create_white_detector
from wsi_toolbox.wsi_files import create_wsi_file

REPO = Path(__file__).resolve().parent.parent
BENCH_DIR = REPO / "data" / "bench"  # data/ is git-ignored

# DZI serving parameters (as a tile server such as vision's compute-tiles uses them)
DZI_TILE_SIZE = 256
DZI_OVERLAP = 0
TILE_JPEG_QUALITY = 90
# patch splitting (as `wt extract` reads)
PATCH_SIZE = 256
TARGET_MPP = 0.5
BATCH_SIZE = 256

LEVEL_OFFSETS = {"max": 0, "mid": 3, "low": 7}  # below the DZI max level
MODES = ("convert", "tiles", "threads", "patches", "patches-raw")
WSI_SUFFIXES = {".ndpi", ".svs", ".tif", ".tiff", ".mrxs", ".scn", ".bif", ".vms", ".svslide", ".qptiff"}
PYRAMID_SUFFIX = ".pyramid.tif"

TESTDATA_BASE = "https://openslide.cs.cmu.edu/download/openslide-testdata"
# (remote path, local name, bytes) -- the set measured in _docs/benchmark-pyramid-dzi.md
TESTDATA = [
    ("Aperio/CMU-1.svs", "Aperio_CMU-1.svs", 177_552_579),
    ("Aperio/JP2K-33003-1.svs", "Aperio_JP2K-33003-1.svs", 63_847_265),
    ("Hamamatsu/CMU-1.ndpi", "Hamamatsu_CMU-1.ndpi", 198_030_965),
    ("Generic-TIFF/CMU-1.tiff", "Generic-TIFF_CMU-1.tiff", 204_117_846),
    ("Philips-TIFF/Philips-1.tiff", "Philips-TIFF_Philips-1.tiff", 326_607_275),
    ("Mirax/CMU-1.zip", "Mirax_CMU-1.zip", 565_106_593),  # extracted to mirax/CMU-1.mrxs
    ("Ventana/Ventana-1.bif", "Ventana_Ventana-1.bif", 227_377_284),  # openslide 4.0.1 cannot open it
]


# ---------------------------------------------------------------------------
# measurement helpers
# ---------------------------------------------------------------------------


def read_proc_io() -> dict[str, int]:
    out: dict[str, int] = {}
    with open("/proc/self/io") as f:
        for line in f:
            k, _, v = line.partition(":")
            out[k.strip()] = int(v)
    return out


def nfs_server_read_bytes(path: Path) -> int | None:
    """serverreadbytes of the NFS mount holding ``path`` (None if not on NFS)."""
    real = os.path.realpath(path)
    best_mnt, best_val = "", None
    try:
        f = open("/proc/self/mountstats")
    except OSError:
        return None
    with f:
        mnt, fstype = None, None
        for line in f:
            if line.startswith("device "):
                parts = line.split()
                mnt = parts[4] if len(parts) > 4 else None
                fstype = parts[7] if len(parts) > 7 else ""
            elif line.strip().startswith("bytes:") and fstype and fstype.startswith("nfs") and mnt:
                if (real == mnt or real.startswith(mnt.rstrip("/") + "/")) and len(mnt) > len(best_mnt):
                    # normalR normalW directR directW serverR ...
                    best_mnt, best_val = mnt, int(line.split()[5])
    return best_val


class Meter:
    """wall / user+sys CPU (all threads) / block-layer read bytes / NFS server read bytes."""

    def __init__(self, path: Path):
        self.path = path

    def __enter__(self) -> Meter:
        self._ru = resource.getrusage(resource.RUSAGE_SELF)
        self._io = read_proc_io()
        self._nfs = nfs_server_read_bytes(self.path)
        self._t = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        self.wall = time.perf_counter() - self._t
        ru = resource.getrusage(resource.RUSAGE_SELF)
        io = read_proc_io()
        nfs = nfs_server_read_bytes(self.path)
        self.cpu = (ru.ru_utime - self._ru.ru_utime) + (ru.ru_stime - self._ru.ru_stime)
        self.read_bytes = io["read_bytes"] - self._io["read_bytes"]
        self.nfs_read_bytes = None if nfs is None or self._nfs is None else nfs - self._nfs

    def as_dict(self) -> dict:
        return {
            "wall_s": round(self.wall, 3),
            "cpu_s": round(self.cpu, 3),
            "cpu_ratio": round(self.cpu / self.wall, 3) if self.wall > 0 else None,
            "read_bytes": self.read_bytes,
            "nfs_read_bytes": self.nfs_read_bytes,
        }


def related_files(path: Path) -> list[Path]:
    """Files making up a slide (MIRAX: the .mrxs plus its data directory)."""
    files = [path]
    if path.suffix.lower() == ".mrxs":
        d = path.with_suffix("")
        if d.is_dir():
            files += sorted(p for p in d.iterdir() if p.is_file())
    return files


def prepare_cache(path: Path, cache: str) -> None:
    for p in related_files(path):
        if cache == "cold":
            fd = os.open(p, os.O_RDONLY)
            try:
                os.fsync(fd)  # dirty pages (a pyramid just written) cannot be dropped
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            finally:
                os.close(fd)
        else:
            with open(p, "rb", buffering=0) as f:
                while f.read(16 << 20):
                    pass


def pct(values: list[float], q: float) -> float:
    s = sorted(values)
    if not s:
        return float("nan")
    k = (len(s) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def run_text(cmd: list[str]) -> str | None:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def environment() -> dict:
    """Machine and library versions recorded with every run."""
    cpu = None
    try:
        with open("/proc/cpuinfo") as f:
            cpu = next((ln.split(":", 1)[1].strip() for ln in f if ln.startswith("model name")), None)
    except OSError:
        pass
    mem_gb = None
    try:
        with open("/proc/meminfo") as f:
            mem_gb = round(int(next(ln for ln in f if ln.startswith("MemTotal")).split()[1]) / 1024**2)
    except (OSError, StopIteration):
        pass
    pkg_dir = Path(wsi_toolbox.__file__).resolve().parent.parent
    commit = run_text(["git", "-C", str(pkg_dir), "rev-parse", "--short", "HEAD"])
    if commit and run_text(["git", "-C", str(pkg_dir), "status", "--porcelain", "--", "wsi_toolbox"]):
        commit += "-dirty"
    return {
        "cpu": cpu,
        "nproc": os.cpu_count(),
        "mem_gb": mem_gb,
        "kernel": platform.release(),
        "python": platform.python_version(),
        "wsi_toolbox": wsi_toolbox.__version__,
        "wsi_toolbox_commit": commit,
        "openslide": openslide.__library_version__,
        "tifffile": tifffile.__version__,
        "vips": run_text(["vips", "--version"]),
    }


# ---------------------------------------------------------------------------
# download
# ---------------------------------------------------------------------------


def cmd_download(args) -> None:
    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    for remote, name, size in TESTDATA:
        if args.only and name not in args.only:
            continue
        out = dest / name
        if out.exists() and out.stat().st_size == size:
            print(f"ok       {name} ({size / 1e6:.0f} MB)")
        else:
            url = f"{TESTDATA_BASE}/{remote}"
            print(f"download {url} ({size / 1e6:.0f} MB)", flush=True)
            part = out.with_name(out.name + ".part")
            urllib.request.urlretrieve(url, part)
            if part.stat().st_size != size:
                print(f"  warning: {part.stat().st_size} bytes, expected {size}")
            part.replace(out)
        if name.endswith(".zip"):
            target = dest / "mirax"
            if not (target / "CMU-1.mrxs").exists():
                print(f"extract  {name} -> {target}/")
                with zipfile.ZipFile(out) as z:
                    z.extractall(target)


# ---------------------------------------------------------------------------
# info
# ---------------------------------------------------------------------------


def describe(path: Path) -> dict:
    info: dict = {"path": str(path), "bytes": sum(p.stat().st_size for p in related_files(path))}
    try:
        wsi = create_wsi_file(str(path))
        info["toolbox_reader"] = type(wsi).__name__
        w, h = wsi.get_original_size()
        info["width"], info["height"] = int(w), int(h)
        try:
            info["mpp"] = round(float(wsi.get_mpp()), 4)
        except Exception as e:  # noqa: BLE001
            info["mpp"] = f"error: {e}"
        info["native_levels"] = [round(lv.downsample, 3) for lv in wsi._get_native_levels()]
    except Exception as e:  # noqa: BLE001
        info["toolbox_reader"] = f"error: {e}"
    try:
        osr = openslide.OpenSlide(str(path))
        info["openslide_vendor"] = osr.properties.get("openslide.vendor")
        info["openslide_downsamples"] = [round(d, 2) for d in osr.level_downsamples]
    except Exception as e:  # noqa: BLE001
        info["openslide_vendor"] = f"error: {e}"
    try:
        with tifffile.TiffFile(str(path)) as tif:
            p = tif.pages[0]
            info["tiff_page0"] = {
                "shape": list(p.shape),
                "tile": [p.tilewidth, p.tilelength] if p.is_tiled else None,
                "strips": None if p.is_tiled else len(p.dataoffsets),
                "compression": getattr(p.compression, "name", str(p.compression)),
                "pages": len(tif.pages),
            }
    except Exception as e:  # noqa: BLE001
        info["tiff_page0"] = f"error: {e}"
    return info


def cmd_info(args) -> None:
    for f in expand_inputs(args.inputs):
        print(json.dumps(describe(f), ensure_ascii=False))


# ---------------------------------------------------------------------------
# child cases: tiles / patches
# ---------------------------------------------------------------------------


class HandlePool:
    """Up to ``size`` WSI handles, each lent to one thread at a time (opened lazily)."""

    def __init__(self, opener, size: int, first=None):
        self._opener = opener
        self._size = size
        self._made = 0
        self._idle: queue.LifoQueue = queue.LifoQueue()
        self._lock = threading.Lock()
        if first is not None:  # an already opened handle
            self._made = 1
            self._idle.put(first)

    @contextmanager
    def get(self):
        try:
            h = self._idle.get_nowait()
        except queue.Empty:
            with self._lock:
                make = self._made < self._size
                if make:
                    self._made += 1
            h = self._opener() if make else self._idle.get()
        try:
            yield h
        finally:
            self._idle.put(h)


def cmd_tiles(args) -> None:
    path = Path(args.file)

    def open_gen() -> DziGenerator:
        return DziGenerator(create_wsi_file(str(path), engine=args.engine), DZI_TILE_SIZE, DZI_OVERLAP)

    t0 = time.perf_counter()
    gen = open_gen()
    open_ms = (time.perf_counter() - t0) * 1000
    level = max(0, gen.max_level - LEVEL_OFFSETS[args.level])
    cols, rows = gen.layout.grid(level)
    rng = random.Random(args.seed)
    todo = iter([(rng.randrange(cols), rng.randrange(rows)) for _ in range(args.n)])

    pool = HandlePool(open_gen, args.threads, first=gen) if args.mode == "pool" else None
    handle_lock = threading.Lock()  # shared: one handle, one thread at a time
    todo_lock = threading.Lock()
    lat: list[float] = []
    out_bytes = [0]
    deadline = time.perf_counter() + args.budget

    def one(g: DziGenerator, col: int, row: int) -> bytes:
        return encode_tile(g.tile(level, col, row), quality=TILE_JPEG_QUALITY)

    def worker() -> None:
        while time.perf_counter() < deadline:
            with todo_lock:
                nxt = next(todo, None)
            if nxt is None:
                return
            s = time.perf_counter()
            if pool is not None:
                with pool.get() as g:
                    jpeg = one(g, *nxt)
            else:
                with handle_lock:
                    jpeg = one(gen, *nxt)
            e = (time.perf_counter() - s) * 1000
            with todo_lock:
                lat.append(e)
                out_bytes[0] += len(jpeg)

    with Meter(path) as m:
        ths = [threading.Thread(target=worker) for _ in range(args.threads)]
        for t in ths:
            t.start()
        for t in ths:
            t.join()

    n = len(lat)
    print(
        json.dumps(
            {
                "kind": "tiles",
                "reader": type(gen.wsi).__name__,
                "level_name": args.level,
                "dzi_level": level,
                "dzi_max_level": gen.max_level,
                "threads": args.threads,
                "mode": args.mode,
                "open_ms": round(open_ms, 1),
                "tiles": n,
                "p50_ms": round(pct(lat, 0.5), 2),
                "p95_ms": round(pct(lat, 0.95), 2),
                "mean_ms": round(statistics.fmean(lat), 2) if lat else None,
                "tiles_per_s": round(n / m.wall, 1) if m.wall > 0 else None,
                "avg_jpeg_bytes": out_bytes[0] // n if n else None,
                **m.as_dict(),
            }
        )
    )


def cmd_patches(args) -> None:
    path = Path(args.file)
    wsi = create_wsi_file(str(path), engine=args.engine)
    reader = WSIPatchReader(
        wsi,
        patch_size=PATCH_SIZE,
        target_mpp=TARGET_MPP,
        white_detector=None if args.no_white else create_white_detector("ptp"),
    )
    rows_per_batch = max(1, BATCH_SIZE // max(1, reader.cols))
    deadline = time.perf_counter() + args.budget
    grid = kept = 0
    with Meter(path) as m:
        for _batch, coords, _desc in reader.iter_batches(BATCH_SIZE):
            grid += reader.cols * rows_per_batch
            kept += len(coords)
            if time.perf_counter() > deadline:
                break
    grid = min(grid, reader.total_patches)
    print(
        json.dumps(
            {
                "kind": "patches",
                "reader": type(wsi).__name__,
                "white_check": not args.no_white,
                "level_index": reader.level.index,
                "level_downsample": round(reader.level.downsample, 3),
                "actual_mpp": round(reader.actual_mpp, 4),
                "grid_total": reader.total_patches,
                "grid_done": grid,
                "kept": kept,
                "patches_per_s": round(grid / m.wall, 1) if m.wall > 0 else None,
                **m.as_dict(),
            }
        )
    )


# ---------------------------------------------------------------------------
# extract: reader vs GPU vs end-to-end (GPU; not part of `run`)
# ---------------------------------------------------------------------------


def _extract_reader(path: Path, align: bool, budget: float) -> dict:
    """The patch reader exactly as FeatureExtractionCommand builds it, consumed as fast as possible."""
    reader = get_patch_reader(
        h5_path=str(BENCH_DIR / "extract" / "none.h5"),  # never exists: forces the WSI reader
        wsi_path=str(path),
        patch_size=PATCH_SIZE,
        target_mpp=TARGET_MPP,
        white_detector=create_white_detector("ptp"),
        prefetch=1,
    )
    inner = reader.reader
    if not align:
        inner._align = 1
    rows_per_batch = max(1, BATCH_SIZE // inner.cols)
    deadline = time.perf_counter() + budget
    grid = kept = 0
    with Meter(path) as m:
        for _batch, coords, _desc in reader.iter_batches(BATCH_SIZE):
            grid += inner.cols * rows_per_batch
            kept += len(coords)
            if time.perf_counter() > deadline:
                break
    grid = min(grid, inner.total_patches)
    return {
        "part": "reader",
        "reader": type(inner.wsi).__name__,
        "tile_h": inner._native_tile_height(),
        "align": inner._align,
        "grid_done": grid,
        "grid_total": inner.total_patches,
        "kept": kept,
        "grid_per_s": round(grid / m.wall, 1),
        "kept_per_s": round(kept / m.wall, 1),
        **m.as_dict(),
    }


def _extract_model(preset: str, device: str, budget: float) -> list[dict]:
    tp = get_tile_preset(preset)
    mean = torch.tensor(tp.norm_mean).view(1, 3, 1, 1)
    std = torch.tensor(tp.norm_std).view(1, 3, 1, 1)
    worker = _GPUWorker(tp.create_model().eval(), device, mean, std, tp.extract_fn, False)
    rng = np.random.default_rng(0)
    out = []
    try:
        for bs in (128, BATCH_SIZE):
            batch = rng.integers(0, 256, (bs, PATCH_SIZE, PATCH_SIZE, 3), dtype=np.uint8)
            for _ in range(3):
                worker.infer(batch)
            n, t0 = 0, time.perf_counter()
            while time.perf_counter() - t0 < budget / 4:
                worker.infer(batch)
                n += bs
            infer_s = time.perf_counter() - t0
            x = torch.randn(bs, 3, PATCH_SIZE, PATCH_SIZE, device=device).contiguous(memory_format=torch.channels_last)
            with torch.inference_mode(), torch.autocast(device_type=worker.device_type, dtype=worker.autocast_dtype):
                fwd = (
                    (lambda: worker.model.forward_features(x))
                    if tp.extract_fn is None
                    else (lambda: tp.extract_fn(worker.model, x))
                )
                for _ in range(3):
                    fwd()
                torch.cuda.synchronize()
                k, t1 = 0, time.perf_counter()
                while time.perf_counter() - t1 < budget / 4:
                    fwd()
                    k += bs
                torch.cuda.synchronize()
                fwd_s = time.perf_counter() - t1
            out.append(
                {
                    "part": "model",
                    "preset": preset,
                    "device": torch.cuda.get_device_name(device) if device.startswith("cuda") else device,
                    "batch": bs,
                    "infer_per_s": round(n / infer_s, 1),
                    "forward_per_s": round(k / fwd_s, 1),
                }
            )
    finally:
        worker.cleanup()
    return out


def _extract_e2e(path: Path, preset: str, device: str, budget: float, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    h5 = out_dir / f"{path.name}.h5"
    marks: dict[str, float] = {}
    t0 = time.perf_counter()

    def sink(ev) -> None:
        marks.setdefault(ev.phase, time.perf_counter())

    cmd = FeatureExtractionCommand(model=preset, preset=preset, device=device, batch_size=BATCH_SIZE, overwrite=True)
    try:
        res = cmd(str(h5), str(path), on_progress=sink, should_cancel=lambda: time.perf_counter() - t0 > budget)
    except Cancelled:
        return {"part": "e2e", "file": str(path), "cancelled_after_s": budget}
    proc = marks["Writing"] - marks["Processing patches"]
    return {
        "part": "e2e",
        "h5": str(h5),
        "kept": res.patch_count,
        "grid": res.total_patches,
        "init_s": round(marks["Processing patches"] - marks.get("Initializing model", t0), 2),
        "processing_s": round(proc, 2),
        "gpu_busy_s": round(res.batch_time_mean * res.total_batches, 2),
        "kept_per_s": round(res.patch_count / proc, 1),
        "grid_per_s": round(res.total_patches / proc, 1),
    }


def cmd_extract(args) -> None:
    parts = set(args.parts.split(","))
    if "model" in parts:
        for rec in _extract_model(args.preset, args.device, args.budget):
            print(json.dumps(rec), flush=True)
    for f in args.files:
        path = Path(f)
        if "reader" in parts:
            for align in {"both": (False, True), "on": (True,), "off": (False,)}[args.align]:
                print(json.dumps({"file": path.name, **_extract_reader(path, align, args.budget)}), flush=True)
        if "e2e" in parts:
            print(
                json.dumps(
                    {"file": path.name, **_extract_e2e(path, args.preset, args.device, args.budget, Path(args.out))}
                ),
                flush=True,
            )


def cmd_extract_compare(args) -> None:
    feats, coords = [], []
    for f in (args.a, args.b):
        with h5py.File(f, "r") as h:
            feats.append(h[f"{args.model}/features"][:])
            coords.append([tuple(c) for c in h[f"{args.model}/coordinates"][:]])
    index_b = {c: i for i, c in enumerate(coords[1])}
    pairs = [(i, index_b[c]) for i, c in enumerate(coords[0]) if c in index_b]
    ia = np.array([i for i, _ in pairs])
    ib = np.array([j for _, j in pairs])
    fa, fb = feats[0][ia], feats[1][ib]
    na = fa / np.linalg.norm(fa, axis=1, keepdims=True)
    nb = fb / np.linalg.norm(fb, axis=1, keepdims=True)
    cos = (na * nb).sum(1)
    # scale: how close is the nearest *other* patch of A, and does B's feature still find its own patch in A
    rng = np.random.default_rng(0)
    sample = rng.choice(len(na), min(500, len(na)), replace=False)
    sim = na[sample] @ na.T
    sim[np.arange(len(sample)), sample] = -1
    self_nn = float((np.argmax(nb[sample] @ na.T, axis=1) == sample).mean())
    scaler = StandardScaler().fit(fa)
    km = KMeans(10, n_init=4, random_state=0).fit(scaler.transform(fa))
    agree = float((km.labels_ == km.predict(scaler.transform(fb))).mean())
    q = np.percentile(cos, [0, 1, 5, 50])
    print(
        json.dumps(
            {
                "kind": "extract-compare",
                "a": args.a,
                "b": args.b,
                "kept_a": len(coords[0]),
                "kept_b": len(coords[1]),
                "common": len(pairs),
                "cos_mean": round(float(cos.mean()), 4),
                "cos_min": round(float(q[0]), 4),
                "cos_p1": round(float(q[1]), 4),
                "cos_p5": round(float(q[2]), 4),
                "cos_median": round(float(q[3]), 4),
                "nearest_other_patch_cos_median": round(float(np.median(sim.max(1))), 4),
                "b_nearest_is_same_patch": round(self_nn, 4),
                "kmeans10_same_cluster": round(agree, 4),
            }
        )
    )


# ---------------------------------------------------------------------------
# run (parent): inputs -> pyramids -> cases -> JSONL + tables
# ---------------------------------------------------------------------------


def expand_inputs(inputs: list[str]) -> list[Path]:
    """Files, directories (non-recursive, known WSI suffixes) and glob patterns."""
    out: list[Path] = []
    for item in inputs:
        paths = [Path(p) for p in sorted(glob.glob(item))] if any(c in item for c in "*?[") else [Path(item)]
        for p in paths:
            if p.is_dir():
                cands = sorted(c for c in p.iterdir() if c.is_file() and c.suffix.lower() in WSI_SUFFIXES)
                # MIRAX slides live one level down (mirax/CMU-1.mrxs)
                cands += sorted(p.glob("*/*.mrxs"))
                out += [c for c in cands if not c.name.endswith(PYRAMID_SUFFIX)]
            elif p.exists():
                out.append(p)
            else:
                print(f"warning: {item} not found", file=sys.stderr)
    seen: set[Path] = set()
    return [p for p in out if not (p.resolve() in seen or seen.add(p.resolve()))]


def run_case(sub: list[str], file: Path, cache: str, engine: str) -> dict:
    prepare_cache(file, cache)
    cmd = [sys.executable, str(Path(__file__).resolve()), *sub, str(file), "--engine", engine]
    out = subprocess.run(cmd, capture_output=True, text=True)
    if out.returncode != 0:
        return {"kind": sub[0], "error": out.stderr.strip()[-600:]}
    return json.loads(out.stdout.strip().splitlines()[-1])


def convert(src: Path, dst: Path, concurrency: int) -> dict:
    dst.parent.mkdir(parents=True, exist_ok=True)
    ru0 = resource.getrusage(resource.RUSAGE_CHILDREN)
    t0 = time.perf_counter()
    try:
        PyramidCommand(concurrency=concurrency)(src, dst, on_progress=None)
    except Exception as e:  # noqa: BLE001
        return {"kind": "convert", "error": str(e)[-600:]}
    wall = time.perf_counter() - t0
    ru1 = resource.getrusage(resource.RUSAGE_CHILDREN)
    info = read_pyramid_info(dst)
    cpu = (ru1.ru_utime - ru0.ru_utime) + (ru1.ru_stime - ru0.ru_stime)
    return {
        "kind": "convert",
        "src_bytes": sum(p.stat().st_size for p in related_files(src)),
        "dst_bytes": info.bytes,
        "wall_s": round(wall, 2),
        "cpu_s": round(cpu, 2),
        "levels": info.levels,
        "vips_concurrency": concurrency,
    }


def cmd_run(args) -> None:
    modes = set(args.modes.split(","))
    unknown = modes - set(MODES)
    if unknown:
        sys.exit(f"unknown modes: {sorted(unknown)} (choose from {', '.join(MODES)})")
    files = expand_inputs(args.inputs)
    if not files:
        sys.exit("no input slides")
    levels = args.levels.split(",")
    caches = args.caches.split(",")
    pyr_dir = Path(args.pyramid_dir)
    run_id = time.strftime("%Y%m%d-%H%M%S")
    results = Path(args.results)
    results.parent.mkdir(parents=True, exist_ok=True)
    env = environment()
    base_all = {"run": run_id, "label": args.label}

    with open(results, "a") as fout:

        def emit(rec: dict) -> None:
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            brief = {
                k: rec[k]
                for k in (
                    "name",
                    "variant",
                    "kind",
                    "level_name",
                    "threads",
                    "mode",
                    "cache",
                    "white_check",
                    "p50_ms",
                    "p95_ms",
                    "tiles_per_s",
                    "patches_per_s",
                    "wall_s",
                    "cpu_s",
                    "read_bytes",
                    "error",
                )  # fmt: skip
                if rec.get(k) is not None
            }
            print(json.dumps(brief, ensure_ascii=False), flush=True)

        emit({**base_all, "kind": "env", **env})
        for src in files:
            name = src.stem
            pyr = pyr_dir / f"{name}{PYRAMID_SUFFIX}"
            variants = [("orig", src)]
            if "convert" in modes and (args.reconvert or not pyr.exists()):
                emit(
                    {
                        **base_all,
                        "name": name,
                        "src": str(src),
                        "dst": str(pyr),
                        **convert(src, pyr, args.vips_concurrency),
                    }
                )
            if pyr.exists():
                variants.append(("pyramid", pyr))
            else:
                print(f"{name}: no pyramid at {pyr} (add 'convert' to --modes)", flush=True)

            for variant, f in variants:
                engine = args.engine if variant == "orig" else "auto"
                base = {**base_all, "name": name, "variant": variant, "file": str(f)}
                budget = ["--budget", str(args.tile_budget), "--n", str(args.n)]
                if "tiles" in modes:
                    for level in levels:
                        for cache in caches:
                            sub = ["tiles", "--level", level, "--threads", "1", *budget]
                            emit({**base, "cache": cache, **run_case(sub, f, cache, engine)})
                if "threads" in modes:
                    for cache in caches:
                        for mode in ("shared", "pool"):
                            sub = ["tiles", "--level", "max", "--threads", str(args.threads), "--mode", mode, *budget]
                            emit({**base, "cache": cache, **run_case(sub, f, cache, engine)})
                for mode, extra in (("patches", []), ("patches-raw", ["--no-white"])):
                    if mode in modes:
                        for cache in caches:
                            sub = ["patches", "--budget", str(args.patch_budget), *extra]
                            emit({**base, "cache": cache, **run_case(sub, f, cache, engine)})

    print()
    print_report([json.loads(ln) for ln in open(results) if ln.strip()], run_id)


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


def mb(v) -> str:
    return "-" if v is None else f"{v / 1e6:.0f}"


def fmt(v, nd=2) -> str:
    return "-" if v is None else f"{v:.{nd}f}" if isinstance(v, float) else str(v)


def ratio(a, b) -> str:
    return f"{a / b:.1f}x" if a and b else "-"


def print_report(recs: list[dict], run_id: str | None = None) -> None:
    if run_id:
        recs = [r for r in recs if r.get("run") == run_id]
    envs = [r for r in recs if r.get("kind") == "env"]
    ok = [r for r in recs if "error" not in r]
    conv = [r for r in ok if r.get("kind") == "convert"]
    tiles1 = [r for r in ok if r.get("kind") == "tiles" and r.get("threads") == 1]
    tilesn = [r for r in ok if r.get("kind") == "tiles" and r.get("threads", 1) > 1]
    patches = [r for r in ok if r.get("kind") == "patches"]
    errors = [r for r in recs if "error" in r]

    for e in envs:
        print(
            f"run {e['run']} ({e.get('label')}): {e.get('cpu')} x{e.get('nproc')}, {e.get('mem_gb')} GB, "
            f"Linux {e.get('kernel')}; wsi-toolbox {e.get('wsi_toolbox')} ({e.get('wsi_toolbox_commit')}), "
            f"openslide {e.get('openslide')}, tifffile {e.get('tifffile')}, {e.get('vips')}\n"
        )

    if conv:
        print("### convert (PyramidCommand)\n")
        print("| name | orig MB | pyramid MB | size ratio | wall s | CPU s | levels |")
        print("|---|---:|---:|---:|---:|---:|---:|")
        for r in conv:
            print(
                f"| {r['name']} | {mb(r['src_bytes'])} | {mb(r['dst_bytes'])} | {r['dst_bytes'] / r['src_bytes']:.2f} "
                f"| {r['wall_s']} | {r['cpu_s']} | {r['levels']} |"
            )
        print()

    if tiles1:
        idx = {(r["label"], r["name"], r["variant"], r["level_name"], r["cache"]): r for r in tiles1}
        keys = sorted({(r["label"], r["name"], r["level_name"], r["cache"]) for r in tiles1}, key=str)
        print("### DZI tiles, 1 thread: orig vs pyramid (p50 / p95 ms, read MB)\n")
        print(
            "| label | name | level | cache | orig p50 / p95 | pyramid p50 / p95 | speed-up (p50) | orig read MB | pyramid read MB |"
        )
        print("|---|---|---|---|---|---|---:|---:|---:|")
        for label, name, level, cache in keys:
            o = idx.get((label, name, "orig", level, cache), {})
            p = idx.get((label, name, "pyramid", level, cache), {})
            print(
                f"| {label} | {name} | {level} | {cache} | {fmt(o.get('p50_ms'))} / {fmt(o.get('p95_ms'))} "
                f"| {fmt(p.get('p50_ms'))} / {fmt(p.get('p95_ms'))} | {ratio(o.get('p50_ms'), p.get('p50_ms'))} "
                f"| {mb(o.get('read_bytes'))} | {mb(p.get('read_bytes'))} |"
            )
        print()

    if tilesn:
        one = {(r["label"], r["name"], r["variant"], r["cache"]): r for r in tiles1 if r["level_name"] == "max"}
        idx = {(r["label"], r["name"], r["variant"], r["cache"], r["mode"]): r for r in tilesn}
        keys = sorted({(r["label"], r["name"], r["variant"], r["cache"], r["threads"]) for r in tilesn}, key=str)
        print("### DZI tiles, full resolution, N threads (tiles/s)\n")
        print(
            "| label | name | variant | cache | 1 thread | N shared (1 handle + lock) | N pool (handle per thread) | pool / 1 thread |"
        )
        print("|---|---|---|---|---:|---:|---:|---:|")
        for label, name, variant, cache, n in keys:
            t1 = one.get((label, name, variant, cache), {}).get("tiles_per_s")
            sh = idx.get((label, name, variant, cache, "shared"), {}).get("tiles_per_s")
            po = idx.get((label, name, variant, cache, "pool"), {}).get("tiles_per_s")
            print(
                f"| {label} | {name} | {variant} | {cache} | {fmt(t1, 0)} | {fmt(sh, 0)} (N={n}) | {fmt(po, 0)} | {ratio(po, t1)} |"
            )
        print()

    if patches:
        print("### patch splitting (WSIPatchReader 256 px @ 0.5 mpp, patches/s incl. white-skipped)\n")
        print(
            "| label | name | variant | white check | cache | level (down) | mpp | patches/s | CPU/wall | read MB | done/total |"
        )
        print("|---|---|---|---|---|---|---:|---:|---:|---:|---|")
        for r in patches:
            print(
                f"| {r['label']} | {r['name']} | {r['variant']} | {'ptp' if r['white_check'] else 'none'} | {r['cache']} "
                f"| {r['level_index']} ({r['level_downsample']}) | {r['actual_mpp']} | {r['patches_per_s']} "
                f"| {r['cpu_ratio']} | {mb(r['read_bytes'])} | {r['grid_done']}/{r['grid_total']} |"
            )
        print()

    if tiles1 or tilesn:
        print("<details><summary>all tile cases</summary>\n")
        print(
            "| label | name | variant | level | threads | mode | cache | p50 ms | p95 ms | tiles/s | CPU s | wall s | CPU/wall | read MB | NFS MB | reader |"
        )
        print("|---|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|")
        for r in tiles1 + tilesn:
            print(
                f"| {r['label']} | {r['name']} | {r['variant']} | {r['level_name']} | {r['threads']} | {r['mode']} "
                f"| {r['cache']} | {r['p50_ms']} | {r['p95_ms']} | {r['tiles_per_s']} | {r['cpu_s']} | {r['wall_s']} "
                f"| {r['cpu_ratio']} | {mb(r['read_bytes'])} | {mb(r['nfs_read_bytes'])} | {r['reader']} |"
            )
        print("\n</details>\n")

    if errors:
        print("### errors\n")
        for r in errors:
            print(f"- {r.get('name')} {r.get('variant', '')} {r.get('kind')}: {r['error'][-300:]}")


def cmd_report(args) -> None:
    print_report([json.loads(ln) for ln in open(args.results) if ln.strip()], args.run)


# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = p.add_subparsers(dest="cmd", required=True)

    a = sp.add_parser("download", help="fetch OpenSlide public test slides")
    a.add_argument("--dest", default=str(BENCH_DIR / "src"))
    a.add_argument("--only", nargs="*", help="local names, e.g. Aperio_CMU-1.svs")
    a.set_defaults(fn=cmd_download)

    a = sp.add_parser("info", help="size / mpp / levels / TIFF layout of slides")
    a.add_argument("inputs", nargs="+")
    a.set_defaults(fn=cmd_info)

    a = sp.add_parser("run", help="convert + benchmark slides, append JSONL, print tables")
    a.add_argument("inputs", nargs="+", help="files, directories or glob patterns")
    a.add_argument("--pyramid-dir", default=str(BENCH_DIR / "pyramid"), help="<stem>.pyramid.tif (reused if present)")
    a.add_argument("--results", default=str(BENCH_DIR / "results.jsonl"))
    a.add_argument("--label", default="local", help="storage label for the tables (ssd / hdd / nfs ...)")
    a.add_argument("--modes", default=",".join(MODES), help=f"comma list of {', '.join(MODES)}")
    a.add_argument("--levels", default="low,mid,max")
    a.add_argument("--caches", default="cold,warm")
    a.add_argument("--n", type=int, default=300, help="tiles per case (same random coordinates for orig / pyramid)")
    a.add_argument("--tile-budget", type=float, default=15.0, help="seconds per tile case")
    a.add_argument("--patch-budget", type=float, default=20.0, help="seconds per patch case")
    a.add_argument("--threads", type=int, default=4)
    a.add_argument("--engine", default="auto", help="reader for the originals (auto / openslide / tifffile)")
    a.add_argument("--reconvert", action="store_true")
    a.add_argument("--vips-concurrency", type=int, default=DEFAULT_CONCURRENCY)
    a.set_defaults(fn=cmd_run)

    a = sp.add_parser("report", help="Markdown tables from a results JSONL")
    a.add_argument("results")
    a.add_argument("--run", default=None, help="only this run id")
    a.set_defaults(fn=cmd_report)

    # child cases (spawned by `run`; can be called directly)
    a = sp.add_parser("tiles")
    a.add_argument("file")
    a.add_argument("--level", choices=list(LEVEL_OFFSETS), default="max")
    a.add_argument("--n", type=int, default=300)
    a.add_argument("--threads", type=int, default=1)
    a.add_argument("--mode", choices=["shared", "pool"], default="shared")
    a.add_argument("--seed", type=int, default=0)
    a.add_argument("--budget", type=float, default=15.0)
    a.add_argument("--engine", default="auto")
    a.set_defaults(fn=cmd_tiles)

    a = sp.add_parser("patches")
    a.add_argument("file")
    a.add_argument("--budget", type=float, default=20.0)
    a.add_argument("--no-white", action="store_true")
    a.add_argument("--engine", default="auto")
    a.set_defaults(fn=cmd_patches)

    a = sp.add_parser("extract", help="feature extraction: reader / model / end-to-end (GPU)")
    a.add_argument("files", nargs="*")
    a.add_argument("--parts", default="reader,model,e2e")
    a.add_argument(
        "--align", choices=["both", "on", "off"], default="both", help="tile-aligned strip reads (reader part)"
    )
    a.add_argument("--preset", default="gigapath-flash")
    a.add_argument("--device", default="cuda:0")
    a.add_argument("--budget", type=float, default=30.0, help="seconds per part and file")
    a.add_argument("--out", default=str(BENCH_DIR / "extract"), help="e2e writes <out>/<file name>.h5")
    a.set_defaults(fn=cmd_extract)

    a = sp.add_parser("extract-compare", help="feature similarity of two extract H5s at the same coordinates")
    a.add_argument("a")
    a.add_argument("b")
    a.add_argument("--model", default="gigapath-flash", help="H5 storage key")
    a.set_defaults(fn=cmd_extract_compare)

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
