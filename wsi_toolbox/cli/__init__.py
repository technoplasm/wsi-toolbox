"""CLI entry point — composes all subcommand mixins onto CLIBase."""

from ._base import DEFAULT_PRESET, CLIBase, CommonArgs, build_output_path
from .analysis import (
    ClusterMixin,
    PcaMixin,
    PreviewMixin,
    PreviewScoreMixin,
    UmapMixin,
)
from .pipeline import (
    AggregateMixin,
    CacheMixin,
    ExtractMixin,
)
from .tools import (
    DziMixin,
    MigrateMixin,
    ShowMixin,
    ThumbMixin,
    migrate_h5,
)


class CLI(
    # pipeline
    CacheMixin,
    ExtractMixin,
    AggregateMixin,
    # analysis
    ClusterMixin,
    UmapMixin,
    PcaMixin,
    PreviewMixin,
    PreviewScoreMixin,
    # tools
    ShowMixin,
    DziMixin,
    ThumbMixin,
    MigrateMixin,
    # base last (most-derived methods like prepare() come from here)
    CLIBase,
):
    """All wsi-toolbox subcommands gathered via mixins; AutoCLI introspects MRO."""


def main():
    CLI().run()


__all__ = [
    "CLI",
    "CLIBase",
    "CommonArgs",
    "DEFAULT_PRESET",
    "build_output_path",
    "main",
    "migrate_h5",
]
