import logging
import time

import pytest

from wsi_toolbox.progress import (
    Cancelled,
    LoggingSink,
    MultiSink,
    NullSink,
    ProgressEvent,
    Reporter,
    RichSink,
    TqdmSink,
    resolve_sink,
)


def _sample_stream() -> list[ProgressEvent]:
    """A representative event sequence: bounded phase, unbounded phase, done."""
    events = []
    rep = Reporter(events.append)
    rep.phase("Initializing model")
    rep.phase("Processing patches", total=3)
    for i in range(3):
        rep.advance(1, message=f"batch {i}")
    rep.phase("Writing")
    rep.finish()
    return events


def test_reporter_event_sequence():
    events = _sample_stream()
    assert [e.phase for e in events] == [
        "Initializing model",
        "Processing patches",
        "Processing patches",
        "Processing patches",
        "Processing patches",
        "Writing",
        "Writing",
    ]
    proc = [e for e in events if e.phase == "Processing patches"]
    assert [e.n for e in proc] == [0, 1, 2, 3]
    assert all(e.total == 3 for e in proc)
    assert proc[-1].message == "batch 2"
    assert proc[-1].fraction == 1.0
    assert events[0].total is None and events[0].fraction is None
    assert events[-1].done and events[-1].phase == "Writing"
    assert not any(e.done for e in events[:-1])
    elapsed = [e.elapsed for e in events]
    assert elapsed == sorted(elapsed) and elapsed[0] >= 0


def test_set_message_emits_event():
    events = []
    rep = Reporter(events.append)
    rep.phase("Generating tiles", total=10)
    rep.set_message("Level 3: row 1/2")
    assert events[-1].message == "Level 3: row 1/2" and events[-1].n == 0


def test_min_interval_throttles_but_keeps_last():
    events = []
    rep = Reporter(events.append, min_interval=10.0)
    rep.phase("P", total=5)
    for _ in range(5):
        rep.advance(1)
    ns = [e.n for e in events]
    assert ns[0] == 0  # phase start always emitted
    assert ns[-1] == 5  # last step always emitted
    assert len(ns) < 6  # intermediate steps throttled


def test_min_interval_lets_events_through_after_interval():
    events = []
    rep = Reporter(events.append, min_interval=0.01)
    rep.phase("P", total=100)
    rep.advance(1)
    time.sleep(0.02)
    rep.advance(1)
    assert [e.n for e in events] == [0, 2]


def test_should_cancel_raises_cancelled_on_advance():
    calls = {"n": 0}

    def should_cancel():
        calls["n"] += 1
        return calls["n"] >= 2

    rep = Reporter(None, should_cancel)
    rep.phase("P", total=3)
    with pytest.raises(Cancelled):
        rep.advance(1)  # first check False
        rep.advance(1)  # second check True


def test_should_cancel_checked_at_phase_boundary():
    rep = Reporter(None, lambda: True)
    with pytest.raises(Cancelled):
        rep.phase("P")


def test_iter_advances_and_sets_total():
    events = []
    rep = Reporter(events.append)
    rep.phase("Rendering patches")
    assert list(rep.iter(range(4))) == [0, 1, 2, 3]
    assert events[-1].n == 4 and events[-1].total == 4


def test_context_manager_finishes_only_on_success():
    events = []
    with Reporter(events.append) as rep:
        rep.phase("P")
    assert events[-1].done

    events.clear()
    with pytest.raises(RuntimeError):
        with Reporter(events.append) as rep:
            rep.phase("P")
            raise RuntimeError("boom")
    assert not any(e.done for e in events)


def test_none_sink_still_cancels():
    rep = Reporter(None, lambda: True)
    with pytest.raises(Cancelled):
        rep.check_cancel()


@pytest.mark.parametrize("sink_factory", [TqdmSink, RichSink, LoggingSink, NullSink])
def test_sinks_accept_event_stream(sink_factory):
    sink = sink_factory()
    for event in _sample_stream():
        sink(event)


def test_tqdm_sink_kwargs():
    sink = TqdmSink(disable=True, leave=False)
    for event in _sample_stream():
        sink(event)


def test_logging_sink_logs_phases(caplog):
    logger = logging.getLogger("test.progress")
    sink = LoggingSink(logger=logger, every=100.0)
    with caplog.at_level(logging.INFO, logger="test.progress"):
        for event in _sample_stream():
            sink(event)
    messages = [r.getMessage() for r in caplog.records]
    assert any(m.startswith("Initializing model") for m in messages)
    assert any(m.startswith("Processing patches [0/3]") for m in messages)
    assert any(m.startswith("Writing") for m in messages)
    assert messages[-1].startswith("Done")
    # throttled: intermediate advances are not logged
    assert not any("[2/3]" in m for m in messages)


def test_streamlit_sink_accepts_event_stream():
    pytest.importorskip("streamlit")
    from wsi_toolbox.progress import StreamlitSink  # noqa: PLC0415

    sink = StreamlitSink()
    for event in _sample_stream():
        sink(event)


def test_multi_sink_fans_out():
    a, b = [], []
    sink = MultiSink(a.append, b.append)
    for event in _sample_stream():
        sink(event)
    assert len(a) == len(b) == 7


def test_resolve_sink():
    assert resolve_sink(None) is None
    assert resolve_sink("none") is None
    assert isinstance(resolve_sink("tqdm"), TqdmSink)
    assert isinstance(resolve_sink("rich"), RichSink)
    assert isinstance(resolve_sink("logging"), LoggingSink)
    assert resolve_sink("tqdm") is not resolve_sink("tqdm")  # stateful -> fresh instance each time
    fn = print
    assert resolve_sink(fn) is fn
    with pytest.raises(ValueError):
        resolve_sink("nope")
