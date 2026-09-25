"""Tile encoder: the loaded tile model(s) plus the acceleration choice, reusable across commands.

``FeatureExtractionCommand`` builds a ``TileEncoder`` per call unless one is passed in with
``encoder=``. Long-lived services build it once (``warmup()`` pays the ``torch.compile`` cost per
process, not per slide) and share it between commands; ``encode`` is thread-safe.

Acceleration (``accel``):

- ``"none"``: eager forward, batches as they come (bit-identical to wsi-toolbox <= 0.6.0)
- ``"compile"``: ``torch.compile(dynamic=False)`` + static batch buckets
- ``"graphs"``: ``torch.compile(mode="reduce-overhead", dynamic=False)`` (CUDA graphs) + static buckets

With ``dynamic=False`` every new input shape is a recompile, and the batches ``WSIPatchReader`` yields
vary in size (one slide row minus the white patches), so the compiled paths pad each batch up to a fixed
bucket by repeating its last patch and slice the padding off again (``_pad_to_bucket``). Features of the
real patches do not depend on the padding (ViT attention is per sample). Measured numbers:
``_docs/benchmark-pyramid-dzi.md`` §12.
"""

from __future__ import annotations

import copy
import gc
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np

from .common import resolve_devices, resolve_preset
from .presets.tile import TilePreset

logger = logging.getLogger(__name__)

ACCEL_NAMES = ("none", "compile", "graphs")
DEFAULT_BUCKETS = (64, 128, 256, 512)


def validate_accel(accel: str) -> str:
    """Check an ``accel`` name; returns it unchanged (ValueError otherwise)."""
    if accel not in ACCEL_NAMES:
        raise ValueError(f"accel が不正です: {accel!r}. Must be one of {list(ACCEL_NAMES)}")
    return accel


def _pad_to_bucket(batch: np.ndarray, buckets: tuple[int, ...]) -> tuple[np.ndarray, int]:
    """Pad ``batch`` (n patches on axis 0) so that the compiled model only ever sees bucket shapes.

    n <= max(buckets): padded to the smallest bucket >= n. n > max(buckets): whole chunks of
    max(buckets) plus the remainder padded to its bucket, so slicing the result into pieces of
    max(buckets) gives chunks that are all bucket sizes. Padding repeats the last patch.
    Returns ``(padded, n)``; ``padded[:n]`` is the original batch (a view when no padding was needed).
    """
    n = len(batch)
    if n == 0:
        raise ValueError("空のバッチはパディングできません: batch has no patches")
    buckets = tuple(sorted(buckets))
    largest = buckets[-1]
    full, rem = divmod(n, largest)
    if rem == 0:
        return batch, n
    target = full * largest + next(b for b in buckets if b >= rem)
    pad = np.repeat(batch[-1:], target - n, axis=0)
    return np.concatenate([batch, pad], axis=0), n


class _DeviceWorker:
    """One model copy on one device, with its forward callable (eager or compiled) and a lock."""

    def __init__(
        self,
        model: Any,
        device: str,
        preset: TilePreset,
        *,
        accel: str,
        buckets: tuple[int, ...],
        with_latent: bool,
    ):
        import torch  # noqa: PLC0415

        self.device = device
        self.preset = preset
        self.accel = accel
        self.buckets = buckets
        self.with_latent = with_latent
        self.lock = threading.Lock()
        self.model = model.to(device, memory_format=torch.channels_last)
        self.mean = torch.tensor(preset.norm_mean).view(1, 3, 1, 1).to(device)
        self.std = torch.tensor(preset.norm_std).view(1, 3, 1, 1).to(device)

        # Select best autocast dtype for this device
        if device.startswith("cuda"):
            self.autocast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            self.autocast_dtype = torch.bfloat16
        self.device_type = "cuda" if device.startswith("cuda") else "cpu"

        extract_fn = preset.extract_fn
        if extract_fn is None:
            self.latent_size = model.patch_embed.proj.kernel_size[0]
            forward = self.model.forward_features
        else:
            self.latent_size = 0
            model_ref = self.model

            def forward(x):
                return extract_fn(model_ref, x)

        if accel != "none":
            # One compiled graph per bucket shape (x patch size). The default recompile limit (8) is
            # tight when two patch sizes share a process; raise it so we never fall back to eager silently.
            torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 8 * len(buckets))
            mode = "reduce-overhead" if accel == "graphs" else "default"
            forward = torch.compile(forward, mode=mode, dynamic=False)
        self.forward = forward

        logger.debug(f"Worker {device}: autocast={self.autocast_dtype}, channels_last, accel={accel}")

    def _forward(self, batch: np.ndarray):
        """uint8 BHWC batch -> the model's output tensor (on the device, autocast dtype)."""
        import torch  # noqa: PLC0415

        # Upload uint8 and scale / normalise on the device: converting to float32 on the CPU first
        # cost as much as the forward pass of a small model (ViT-S) and moved 4x the bytes.
        x = torch.from_numpy(batch).to(self.device).permute(0, 3, 1, 2)  # BHWC->BCHW (a channels_last view)
        x = x.float().div_(255).contiguous(memory_format=torch.channels_last)
        x = (x - self.mean) / self.std
        with torch.inference_mode(), torch.autocast(device_type=self.device_type, dtype=self.autocast_dtype):
            return self.forward(x)

    def _split(self, out, n: int) -> tuple[np.ndarray, np.ndarray | None]:
        """Model output -> (features, latent | None) for the first ``n`` rows, copied to the host.

        Copies right away: with CUDA graphs the output buffers are overwritten by the next replay.
        """
        if self.preset.extract_fn is not None:
            return out[:n].float().cpu().numpy(), None
        # Only the CLS token leaves the device unless the latent tokens are wanted
        features = out[:n, 0, ...].float().cpu().numpy()
        latent = None
        if self.with_latent:
            latent_index = out.shape[1] - self.latent_size**2
            latent = out[:n, latent_index:, ...].float().cpu().numpy().astype(np.float16)
        return features, latent

    def infer(self, batch: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        """Run inference on a uint8 BHWC batch. Returns (features, latent_or_None)."""
        with self.lock:
            if self.accel == "none":
                return self._split(self._forward(batch), len(batch))

            padded, n = _pad_to_bucket(batch, self.buckets)
            largest = max(self.buckets)
            feats, lats = [], []
            for start in range(0, len(padded), largest):
                chunk = padded[start : start + largest]
                keep = min(len(chunk), n - start)  # real rows in this chunk
                f, latent = self._split(self._forward(chunk), keep)
                feats.append(f)
                if latent is not None:
                    lats.append(latent)
            return np.concatenate(feats, axis=0), (np.concatenate(lats, axis=0) if lats else None)

    def close(self) -> None:
        """Release the device resources held by this worker."""
        import torch  # noqa: PLC0415

        with self.lock:
            self.forward = None
            del self.model, self.mean, self.std
            if self.device.startswith("cuda"):
                torch.cuda.synchronize(self.device)
                torch.cuda.empty_cache()


class TileEncoder:
    """The loaded tile model(s) plus the acceleration choice, built once and reused across commands.

    Args:
        preset: Preset name or ``TilePreset``; None -> ``defaults.preset`` (ValueError if unset too).
        device: Device spec ('auto', 'cpu', 'cuda:0', 'cuda:0,1'); None -> ``defaults.device``.
            Several devices -> one model copy per device, ``encode`` splits each batch across them.
        accel: ``"none"`` (eager), ``"compile"`` or ``"graphs"`` (see the module docstring).
            Anything but ``"none"`` on the CPU logs a warning and runs eager.
        buckets: Batch sizes the compiled model is specialised for (ascending after sorting).
        with_latent: Also return the patch-token latents from ``encode`` (float16). Ignored (with a
            warning) for presets with a custom ``extract_fn``.

    Usage:
        enc = wt.TileEncoder('gigapath-flash', device='cuda', accel='graphs')
        enc.warmup()                        # compile every bucket shape now, not on the first slide
        cmd = wt.FeatureExtractionCommand(model='gigapath-flash', encoder=enc)
        cmd('a.h5', wsi_path='a.ndpi'); cmd('b.h5', wsi_path='b.ndpi')
        enc.close()
    """

    def __init__(
        self,
        preset: str | TilePreset | None = None,
        device: str | None = None,
        *,
        accel: str = "none",
        buckets: tuple[int, ...] = DEFAULT_BUCKETS,
        with_latent: bool = False,
    ):
        validate_accel(accel)
        buckets = tuple(sorted(int(b) for b in buckets))
        if not buckets or buckets[0] <= 0:
            raise ValueError(f"buckets が不正です: {buckets!r}. Must be a non-empty tuple of positive ints")

        self.preset: TilePreset = resolve_preset(preset)
        self.devices: list[str] = resolve_devices(device)
        self.buckets = buckets
        self.with_latent = with_latent

        if accel != "none" and not all(d.startswith("cuda") for d in self.devices):
            logger.warning(f"accel={accel!r} is only supported on CUDA devices ({self.devices}); running eager")
            accel = "none"
        self.accel: str = accel

        if with_latent and self.preset.extract_fn is not None:
            logger.warning("with_latent is not supported with custom extract_fn, skipping latent extraction")
            self.with_latent = False

        if len(self.devices) > 1:
            logger.info(f"Using {len(self.devices)} GPUs for parallel inference: {self.devices}")
        else:
            logger.info(f"Using device: {self.devices[0]}")

        base_model = self.preset.create_model().eval()
        self._workers: list[_DeviceWorker] = []
        for i, dev in enumerate(self.devices):
            model = base_model if i == 0 else copy.deepcopy(base_model)
            self._workers.append(
                _DeviceWorker(model, dev, self.preset, accel=self.accel, buckets=buckets, with_latent=self.with_latent)
            )
        self._executor = ThreadPoolExecutor(max_workers=len(self._workers)) if len(self._workers) > 1 else None
        self._closed = False

    @property
    def feature_dim(self) -> int | None:
        """Feature dimension once known (after the first ``encode`` / ``warmup``), else None."""
        return getattr(self, "_feature_dim", None)

    def warmup(self, patch_size: int = 256) -> None:
        """Run every bucket shape once per device so the compile cost is paid now (no-op for ``"none"``).

        ``patch_size`` must match the patches ``encode`` will see; a different size compiles again later.
        """
        if self.accel == "none":
            return
        # CUDA graph trees run a shape eagerly first and record the graph on a later call, so each
        # bucket is fed a few times; for plain compile the first call does all the work.
        repeats = 3 if self.accel == "graphs" else 1
        rng = np.random.default_rng(0)
        for worker in self._workers:
            for b in self.buckets:
                batch = rng.integers(0, 256, (b, patch_size, patch_size, 3), dtype=np.uint8)
                for _ in range(repeats):
                    features, _ = worker.infer(batch)
                self._feature_dim = int(features.shape[-1])
                logger.debug(f"warmup {worker.device}: bucket {b} x {patch_size}px done")

    def encode(self, batch: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        """uint8 BHWC patches -> (features float32 (n, dim), latent float16 (n, tokens, dim) | None).

        Row order is preserved. The batch is split across the devices when there are several.
        """
        if self._closed:
            raise RuntimeError("TileEncoder は close 済みです: encoder has been closed")
        if len(batch) == 0:
            raise ValueError("空のバッチです: batch has no patches")

        if self._executor is None:
            features, latent = self._workers[0].infer(batch)
        else:
            chunks = np.array_split(batch, len(self._workers))
            futures = [
                self._executor.submit(worker.infer, chunk)
                for worker, chunk in zip(self._workers, chunks)
                if len(chunk) > 0
            ]
            results = [f.result() for f in futures]
            features = np.concatenate([f for f, _ in results], axis=0)
            latents = [lat for _, lat in results if lat is not None]
            latent = np.concatenate(latents, axis=0) if latents else None
        self._feature_dim = int(features.shape[-1])
        return features, latent

    def close(self) -> None:
        """Release the models (idempotent)."""
        if self._closed:
            return
        self._closed = True
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None
        for worker in self._workers:
            worker.close()
        self._workers = []
        gc.collect()

    def __enter__(self) -> TileEncoder:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"TileEncoder(preset={self.preset.name!r}, devices={self.devices}, accel={self.accel!r})"


__all__ = ["ACCEL_NAMES", "DEFAULT_BUCKETS", "TileEncoder", "validate_accel"]
