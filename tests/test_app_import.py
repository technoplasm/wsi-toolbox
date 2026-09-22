"""Entry points (streamlit app pages, watcher) import cleanly and never write process-wide defaults."""

import importlib
import logging
import subprocess
import sys

import pytest

import wsi_toolbox as wt
from wsi_toolbox.progress import LoggingSink, MultiSink, TqdmSink

ENTRY_MODULES = [
    "wsi_toolbox.watcher",
    "wsi_toolbox.app.ui.pages.wsi",
    "wsi_toolbox.app.ui.pages.hdf5",
]


@pytest.mark.parametrize("module", ENTRY_MODULES)
def test_entry_module_imports(module):
    before = wt.defaults.model_copy()
    importlib.import_module(module)
    assert wt.defaults == before


def test_app_main_imports_in_subprocess():
    """app/main.py runs streamlit calls at import time; check it in a fresh process (bare mode)."""
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import wsi_toolbox as wt; before = wt.defaults.model_copy(); "
            "import wsi_toolbox.app.main; assert wt.defaults == before, (before, wt.defaults)",
        ],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert proc.returncode == 0, proc.stderr


def test_watcher_task_sink_is_tqdm_plus_logfile(tmp_path):
    from wsi_toolbox.watcher import Task  # noqa: PLC0415

    task = Task(tmp_path, "uni2, rotate")
    assert task.preset == "uni2"
    assert task.should_rotate is True

    sink = task.make_sink()
    assert isinstance(sink, MultiSink)
    kinds = [type(s) for s in sink._sinks]
    assert kinds == [TqdmSink, LoggingSink]

    # LoggingSink lines land in the task's log file, not in the root logging tree
    task.logger.log(logging.INFO, "Processing patches [3/10]")
    assert (tmp_path / Task.LOG_FILE).read_text() == "Processing patches [3/10]\n"
    assert task.logger.propagate is False
