"""Shared base for the CLI: CLIBase, CommonArgs, prepare(), and helpers."""

import logging
import os
import warnings
from pathlib import Path

from pydantic import BaseModel
from pydantic_autocli import AutoCLI, param

from ..progress import resolve_sink
from ..utils.seed import fix_global_seed, get_global_seed

logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=FutureWarning, message=".*force_all_finite.*")
warnings.filterwarnings(
    "ignore", category=FutureWarning, message="You are using `torch.load` with `weights_only=False`"
)
warnings.filterwarnings("ignore", category=UserWarning, message=".*cuda capability.*")

DEFAULT_PRESET = os.getenv("WT_PRESET", "uni2")


def build_output_path(input_path: str, namespace: str, filename: str) -> str:
    """
    Build output path based on namespace.

    - namespace="default": save in same directory as input file
    - otherwise: save in namespace subdirectory (created if needed)
    """
    p = Path(input_path)
    if namespace == "default":
        output_dir = p.parent
    else:
        output_dir = p.parent / namespace
        os.makedirs(output_dir, exist_ok=True)
    return str(output_dir / filename)


class CommonArgs(BaseModel):
    seed: int = get_global_seed()
    preset: str = param(
        DEFAULT_PRESET,
        l="--preset",
        description="Foundation model preset (uni, uni2, gigapath, ...)",
    )
    model: str = param(
        "",
        l="--model",
        s="-M",
        description="HDF5 storage key (free string; defaults to --preset)",
    )
    progress: str = param("rich", choices=["rich", "tqdm", "none"], description="Progress display")
    device: str = param("auto", s="-D", description="Device: auto, cpu, cuda:0, cuda:0,1")
    verbose: bool = param(False, s="-v")


class CLIBase(AutoCLI):
    """AutoCLI base that owns prepare() and shared helpers.

    Subcommands are added via mixins (see cli/pipeline.py, cli/analysis.py, cli/tools.py).
    """

    CommonArgs = CommonArgs

    def prepare(self, a: CommonArgs):
        fix_global_seed(a.seed)
        # Session-level settings are held here and passed explicitly to each
        # Command (preset= / device= / on_progress=); the CLI never writes the
        # process-wide defaults in wsi_toolbox.common.
        # The h5 storage key (a.model) is passed per-command.
        self.preset = a.preset
        self.device = a.device
        self.sink = resolve_sink(a.progress)
        self.cluster_cmap = "tab20"
        logging.basicConfig(
            format="[wsi-toolbox] %(levelname)s - %(message)s",
            level=logging.DEBUG if a.verbose else logging.INFO,
        )

    def _parse_white_detect(self, detect_white: list[str]) -> tuple[str, float | None]:
        """Parse white detection arguments. Returns (method, threshold)."""
        if not detect_white or len(detect_white) == 0:
            return ("ptp", None)

        method = detect_white[0]
        valid_methods = ("ptp", "otsu", "std", "green")
        if method not in valid_methods:
            raise ValueError(f"Invalid method '{method}'. Must be one of {valid_methods}")

        if len(detect_white) == 1:
            return (method, None)

        try:
            threshold = float(detect_white[1])
        except ValueError:
            raise ValueError(f"Invalid threshold value '{detect_white[1]}'. Must be a number.")

        return (method, threshold)
