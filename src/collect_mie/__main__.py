"""
collect-mie: run any plot from a single YAML config file.

    collect-mie path/to/run.yaml
    collect-mie --config path/to/run.yaml

The plot/command is read from run.command in the config. All other parameters
must appear in the YAML (mie:, ssc:, command sections, run.args:, etc.).
"""

from __future__ import annotations

import sys

from collect_mie.plot_title import COMMAND_ALIASES
from collect_mie.run_config import COMMAND_TARGETS, dispatch_config, resolve_config_path


def main(argv: list[str] | None = None) -> None:
    argv = argv if argv is not None else sys.argv[1:]
    if argv and argv[0] in ("-h", "--help") and len(argv) == 1:
        _print_help()
        return
    config_path = resolve_config_path(argv)
    dispatch_config(config_path)


def _print_help() -> None:
    commands = "\n".join(
        f"  {name:<32}  {COMMAND_ALIASES.get(name, name)}"
        for name in sorted(COMMAND_TARGETS)
    )
    print(
        "Usage: collect-mie CONFIG.yaml\n"
        "       collect-mie --config CONFIG.yaml\n\n"
        "All parameters are read from the YAML file. The config must include:\n\n"
        "  run:\n"
        "    command: <name>    # one of:\n"
        f"{commands}\n\n"
        "See examples/*_run.example.yaml for templates."
    )


if __name__ == "__main__":
    main()
