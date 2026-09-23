import os
import subprocess
import sys

import pytest

import wsi_toolbox as wt
from wsi_toolbox.cli import CLI
from wsi_toolbox.cli._base import CommonArgs

_ENV = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}

SUBCOMMANDS = [
    "cache",
    "extract",
    "aggregate",
    "cluster",
    "umap",
    "pca",
    "preview",
    "preview-score",
    "show",
    "dzi",
    "pyramid",
    "thumb",
    "migrate",
]


def _run_cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "wsi_toolbox.cli", *args],
        capture_output=True,
        text=True,
        env=_ENV,
        timeout=120,
    )


def test_cli_help_exits_zero():
    proc = _run_cli("--help")
    assert proc.returncode == 0, proc.stderr
    assert "usage" in proc.stdout.lower()


def test_cli_help_lists_every_subcommand():
    proc = _run_cli("--help")
    assert proc.returncode == 0, proc.stderr
    for name in SUBCOMMANDS:
        assert name in proc.stdout, f"subcommand {name!r} missing from top-level --help"


@pytest.mark.parametrize("name", SUBCOMMANDS)
def test_subcommand_help_exits_zero(name):
    proc = _run_cli(name, "--help")
    assert proc.returncode == 0, proc.stderr
    assert "usage" in proc.stdout.lower()
    assert "--progress" in proc.stdout


def test_progress_rejects_unknown_choice():
    proc = _run_cli("show", "--progress", "bogus", "--in", "x.h5")
    assert proc.returncode != 0
    assert "invalid choice" in proc.stderr
    for choice in ("rich", "tqdm", "none"):
        assert choice in proc.stderr


@pytest.mark.parametrize(
    ("progress", "expected"),
    [("rich", wt.RichSink), ("tqdm", wt.TqdmSink), ("none", type(None))],
)
def test_prepare_resolves_sink(progress, expected):
    cli = CLI()
    cli.prepare(CommonArgs(progress=progress))
    assert isinstance(cli.sink, expected)


def test_prepare_holds_session_settings_without_touching_defaults():
    saved = wt.defaults.model_copy()
    cli = CLI()
    cli.prepare(CommonArgs(preset="gigapath", device="cpu", progress="none"))
    assert cli.preset == "gigapath"
    assert cli.device == "cpu"
    assert cli.cluster_cmap == "tab20"
    # The CLI must pass these to Commands explicitly, never via process-wide defaults
    assert wt.defaults.preset == saved.preset
    assert wt.defaults.device == saved.device
    assert wt.defaults.progress == saved.progress


def test_importing_cli_does_not_write_defaults():
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import wsi_toolbox as wt; before = wt.defaults.model_copy(); import wsi_toolbox.cli; "
            "assert wt.defaults == before, (before, wt.defaults)",
        ],
        capture_output=True,
        text=True,
        env=_ENV,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
