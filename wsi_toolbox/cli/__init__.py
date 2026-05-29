"""CLI entry point — composes pipeline / analysis / tools mixins onto CLIBase."""

from ._base import DEFAULT_PRESET, CLIBase, CommonArgs, build_output_path
from .analysis import AnalysisMixin
from .pipeline import PipelineMixin
from .tools import ToolsMixin, migrate_h5


class CLI(PipelineMixin, AnalysisMixin, ToolsMixin, CLIBase):
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
