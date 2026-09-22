"""Shared helpers for commands: turning ``on_progress`` / ``should_cancel`` into a Reporter."""

from collections.abc import Callable

from ..common import defaults
from ..progress import UNSET, ProgressSink, Reporter, Unset, resolve_sink


def make_reporter(
    on_progress: ProgressSink | None | Unset,
    should_cancel: Callable[[], bool] | None,
) -> Reporter:
    """Build the Reporter for one command invocation.

    ``on_progress`` not given (UNSET) -> ``resolve_sink(defaults.progress)``; an explicit
    ``None`` means silence. A fresh sink is created per call because sinks are stateful.
    """
    if on_progress is UNSET:
        on_progress = resolve_sink(defaults.progress)
    return Reporter(on_progress, should_cancel)


__all__ = ["UNSET", "Unset", "make_reporter"]
