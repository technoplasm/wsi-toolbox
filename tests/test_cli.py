import os
import subprocess
import sys

import pytest

_ENV = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}


@pytest.mark.parametrize("args", [["--help"], ["extract", "--help"]])
def test_cli_help_exits_zero(args):
    proc = subprocess.run(
        [sys.executable, "-m", "wsi_toolbox.cli", *args],
        capture_output=True,
        text=True,
        env=_ENV,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    assert "usage" in proc.stdout.lower()
