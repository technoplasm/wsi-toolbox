"""Progress events, the Reporter that emits them, and the sinks that display them.

Commands never draw progress themselves. They call ``Reporter.phase`` / ``advance``
and the Reporter forwards ``ProgressEvent`` values to a sink (any callable). The
sinks in this module are stateful callables that turn the event stream into a
tqdm / rich / streamlit bar or log lines. Cancellation is checked on every
``advance`` and ``phase`` and surfaces as ``Cancelled``.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class ProgressEvent:
    """One snapshot of a command's progress.

    Attributes:
        phase: Stage name ("Initializing model", "Processing patches", "UMAP", ...).
        n: Progress count within the current phase.
        total: Total count of the current phase, or None when unknown.
        elapsed: Seconds since the command (Reporter) started.
        message: Supplementary text (batch description etc.); tqdm's postfix.
        done: True only for the final event of a command.
    """

    phase: str
    n: int
    total: int | None
    elapsed: float
    message: str = ""
    done: bool = False

    @property
    def fraction(self) -> float | None:
        """``n / total`` clamped to [0, 1], or None when total is unknown."""
        if not self.total:
            return None
        return min(self.n / self.total, 1.0)


ProgressSink = Callable[[ProgressEvent], None]


class Cancelled(Exception):
    """Raised by a command when ``should_cancel()`` returns True.

    Partial outputs are cleaned up by the command before this propagates.
    """


class _Unset:
    """Sentinel type distinguishing "argument not given" from an explicit ``None``."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "UNSET"


Unset = _Unset
UNSET = _Unset()


class Reporter:
    """Emits ProgressEvents to a sink and checks for cancellation.

    Args:
        on_progress: Sink callable, or None to emit nothing (cancellation still works).
        should_cancel: Callable returning True when the command should abort.
        min_interval: Minimum seconds between ``advance`` events within one phase.
            Phase changes, explicit messages, the last step of a phase and ``finish``
            are always emitted.
    """

    def __init__(
        self,
        on_progress: ProgressSink | None,
        should_cancel: Callable[[], bool] | None = None,
        *,
        min_interval: float = 0.0,
    ):
        self._sink = on_progress
        self._should_cancel = should_cancel
        self._min_interval = min_interval
        self._t0 = time.perf_counter()
        self._phase = ""
        self._n = 0
        self._total: int | None = None
        self._message = ""
        self._last_emit = -float("inf")

    # --- state -----------------------------------------------------------------

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self._t0

    @property
    def current_phase(self) -> str:
        return self._phase

    def _emit(self, done: bool = False) -> None:
        self._last_emit = time.perf_counter()
        if self._sink is None:
            return
        self._sink(
            ProgressEvent(
                phase=self._phase,
                n=self._n,
                total=self._total,
                elapsed=self.elapsed,
                message=self._message,
                done=done,
            )
        )

    # --- API used by commands ----------------------------------------------------

    def phase(self, name: str, total: int | None = None, message: str = "") -> None:
        """Switch to a new phase (n resets to 0) and emit its first event."""
        self._phase = name
        self._n = 0
        self._total = total
        self._message = message
        self.check_cancel()
        self._emit()

    def advance(self, n: int = 1, message: str | None = None) -> None:
        """Advance the current phase by ``n``, emit (throttled), then check cancellation."""
        self._n += n
        if message is not None:
            self._message = message
        finished = self._total is not None and self._n >= self._total
        if finished or time.perf_counter() - self._last_emit >= self._min_interval:
            self._emit()
        self.check_cancel()

    def set_message(self, message: str) -> None:
        self._message = message
        self._emit()

    def check_cancel(self) -> None:
        if self._should_cancel is not None and self._should_cancel():
            raise Cancelled(f"cancelled during '{self._phase}'")

    def iter(self, iterable: Iterable[T], *, total: int | None = None) -> Iterator[T]:
        """Iterate ``iterable`` advancing the current phase once per item (tqdm(iterable) equivalent)."""
        if total is None and hasattr(iterable, "__len__"):
            total = len(iterable)  # type: ignore[arg-type]
        if total is not None:
            self._total = total
        for item in iterable:
            yield item
            self.advance(1)

    def finish(self) -> None:
        """Emit the final ``done=True`` event."""
        self._emit(done=True)

    def __enter__(self) -> Reporter:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if exc_type is None:
            self.finish()


# --- sinks ---------------------------------------------------------------------------


class NullSink:
    """Sink that discards every event. Use ``on_progress=NullSink()`` for silence."""

    def __call__(self, event: ProgressEvent) -> None:
        pass


class MultiSink:
    """Fan one event stream out to several sinks."""

    def __init__(self, *sinks: ProgressSink):
        self._sinks = [s for s in sinks if s is not None]

    def __call__(self, event: ProgressEvent) -> None:
        for sink in self._sinks:
            sink(event)


class _PhasedSink:
    """Base for bar-style sinks: closes the previous bar when the phase changes."""

    def __init__(self):
        self._phase: str | None = None

    def __call__(self, event: ProgressEvent) -> None:
        if event.done:
            if self._phase is not None:
                self._close(event)
                self._phase = None
            self._finish(event)
            return
        if event.phase != self._phase:
            if self._phase is not None:
                self._close(event)
            self._phase = event.phase
            self._open(event)
        self._update(event)

    def _open(self, event: ProgressEvent) -> None:
        raise NotImplementedError

    def _update(self, event: ProgressEvent) -> None:
        raise NotImplementedError

    def _close(self, event: ProgressEvent) -> None:
        raise NotImplementedError

    def _finish(self, event: ProgressEvent) -> None:
        pass


class TqdmSink(_PhasedSink):
    """One tqdm bar per phase. ``message`` is shown as the postfix. Default sink."""

    def __init__(self, **tqdm_kwargs: Any):
        super().__init__()
        self._kwargs = tqdm_kwargs
        self._bar = None

    def _open(self, event: ProgressEvent) -> None:
        from tqdm import tqdm  # noqa: PLC0415

        kwargs = dict(self._kwargs)
        if event.total is None:
            # Unbounded phase (e.g. "Initializing model"): show only the name and elapsed time,
            # not tqdm's "0it [00:00, ?it/s]" counter.
            kwargs.setdefault("bar_format", "{desc} [{elapsed}]{postfix}")
        self._bar = tqdm(total=event.total, desc=event.phase, **kwargs)

    def _update(self, event: ProgressEvent) -> None:
        bar = self._bar
        if event.total != bar.total:
            bar.total = event.total
        if event.message:
            bar.set_postfix_str(event.message, refresh=False)
        delta = event.n - bar.n
        if delta > 0:
            bar.update(delta)
        else:
            bar.refresh()

    def _close(self, event: ProgressEvent) -> None:
        self._bar.close()
        self._bar = None


class RichSink(_PhasedSink):
    """rich.progress display; one task per phase. Used by the CLI."""

    def __init__(self, console: Any = None):
        super().__init__()
        self._console = console
        self._progress = None
        self._task = None

    def _ensure_progress(self):
        if self._progress is None:
            from rich.progress import (  # noqa: PLC0415
                BarColumn,
                MofNCompleteColumn,
                Progress,
                SpinnerColumn,
                TextColumn,
                TimeElapsedColumn,
                TimeRemainingColumn,
            )

            self._progress = Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.description}"),
                BarColumn(bar_width=40),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                TextColumn("•"),
                TimeRemainingColumn(),
                TextColumn("[cyan]{task.fields[message]}"),
                console=self._console,
                transient=False,
            )
            self._progress.start()
        return self._progress

    def _open(self, event: ProgressEvent) -> None:
        progress = self._ensure_progress()
        self._task = progress.add_task(event.phase, total=event.total, message=event.message)

    def _update(self, event: ProgressEvent) -> None:
        self._progress.update(self._task, completed=event.n, total=event.total, message=event.message)

    def _close(self, event: ProgressEvent) -> None:
        task = self._progress.tasks[self._task]
        if task.total is None:
            # Unbounded phase: mark as finished so the bar renders complete
            self._progress.update(self._task, total=max(task.completed, 1), completed=max(task.completed, 1))
        self._task = None

    def _finish(self, event: ProgressEvent) -> None:
        if self._progress is not None:
            self._progress.stop()
            self._progress = None


class StreamlitSink(_PhasedSink):
    """``st.progress`` per phase inside ``container`` (default: the main page)."""

    def __init__(self, container: Any = None):
        super().__init__()
        import streamlit as st  # noqa: PLC0415

        self._container = container if container is not None else st
        self._bar = None

    @staticmethod
    def _label(event: ProgressEvent) -> str:
        if event.total:
            label = f"{event.phase} [{event.n}/{event.total}]"
        else:
            label = event.phase
        if event.message:
            label = f"{label} — {event.message}"
        return label

    def _open(self, event: ProgressEvent) -> None:
        self._bar = self._container.progress(0, text=self._label(event))

    def _update(self, event: ProgressEvent) -> None:
        self._bar.progress(event.fraction or 0.0, text=self._label(event))

    def _close(self, event: ProgressEvent) -> None:
        self._bar.progress(1.0)
        self._bar = None


class LoggingSink:
    """Log ``phase [n/total] message`` at most once per ``every`` seconds (phase changes always)."""

    def __init__(self, logger: logging.Logger | None = None, every: float = 5.0, level: int = logging.INFO):
        self._logger = logger if logger is not None else logging.getLogger("wsi_toolbox.progress")
        self._every = every
        self._level = level
        self._phase: str | None = None
        self._last = -float("inf")

    def __call__(self, event: ProgressEvent) -> None:
        now = time.monotonic()
        if event.done:
            self._logger.log(self._level, f"Done ({event.elapsed:.1f}s)")
            self._phase = None
            return
        phase_changed = event.phase != self._phase
        if not phase_changed and now - self._last < self._every:
            return
        self._phase = event.phase
        self._last = now
        count = f"[{event.n}/{event.total}]" if event.total is not None else f"[{event.n}]"
        text = f"{event.phase} {count}"
        if event.message:
            text = f"{text} {event.message}"
        self._logger.log(self._level, text)


_SINK_FACTORIES: dict[str, Callable[[], ProgressSink | None]] = {
    "tqdm": TqdmSink,
    "rich": RichSink,
    "streamlit": StreamlitSink,
    "logging": LoggingSink,
    "none": lambda: None,
}

SINK_NAMES: tuple[str, ...] = tuple(_SINK_FACTORIES)


def resolve_sink(name_or_sink: str | ProgressSink | None) -> ProgressSink | None:
    """Turn a sink name ("tqdm" / "rich" / "streamlit" / "logging" / "none") or callable into a sink.

    A fresh sink instance is created for each call because sinks are stateful.
    """
    if name_or_sink is None:
        return None
    if isinstance(name_or_sink, str):
        try:
            factory = _SINK_FACTORIES[name_or_sink]
        except KeyError:
            raise ValueError(f"Unknown progress sink: {name_or_sink!r}. Available: {list(_SINK_FACTORIES)}")
        return factory()
    if callable(name_or_sink):
        return name_or_sink
    raise TypeError(f"on_progress must be a sink name, a callable or None, got {type(name_or_sink).__name__}")


__all__ = [
    "ProgressEvent",
    "ProgressSink",
    "Cancelled",
    "Reporter",
    "Unset",
    "UNSET",
    "NullSink",
    "MultiSink",
    "TqdmSink",
    "RichSink",
    "StreamlitSink",
    "LoggingSink",
    "SINK_NAMES",
    "resolve_sink",
]
