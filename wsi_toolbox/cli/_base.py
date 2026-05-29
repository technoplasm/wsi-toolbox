"""Shared base for the CLI: CLIBase, CommonArgs, prepare(), and helpers."""

import logging
import os
import warnings
from pathlib import Path

from pydantic import BaseModel
from pydantic_autocli import AutoCLI, param

from .. import common
from ..utils.seed import fix_global_seed, get_global_seed

logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=FutureWarning, message=".*force_all_finite.*")
warnings.filterwarnings(
    "ignore", category=FutureWarning, message="You are using `torch.load` with `weights_only=False`"
)
warnings.filterwarnings("ignore", category=UserWarning, message=".*cuda capability.*")

DEFAULT_PRESET = os.getenv("WT_PRESET", "uni2")

# Module-import-time defaults (preserved from original cli.py).
common.set_default_progress("rich")
common.set_default_model_preset(DEFAULT_PRESET)
common.set_default_cluster_cmap("tab20")


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
        l="--model-name",
        s="-M",
        description="HDF5 storage key (free string; defaults to --preset)",
    )
    progress: str = param("rich", choices=["rich", "tqdm"])
    device: str = param("auto", s="-D", description="Device: auto, cpu, cuda:0, cuda:0,1")
    verbose: bool = param(False, s="-v")


class CLIBase(AutoCLI):
    """AutoCLI base that owns prepare() and shared helpers.

    Subcommands are added via mixins (see cli/pipeline.py, cli/analysis.py, cli/tools.py).
    """

    CommonArgs = CommonArgs

    def prepare(self, a: CommonArgs):
        fix_global_seed(a.seed)
        # Preset registers the foundation model. model_name (storage key) defaults to preset.
        common.set_default_model_preset(a.preset)
        common.set_default_model_name(a.model if a.model else a.preset)
        common.set_default_device(a.device)
        common.set_default_progress(a.progress)
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
