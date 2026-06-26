#!/usr/bin/env python3
"""Run one example YAML config via collect-mie.
CONFIG_PATH below or pass a config path on the command line.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = Path("examples/plot_diameter_run.example.yaml")
# CONFIG_PATH = Path("examples/plot_diameter_fsc_rect_mask_run.example.yaml")


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    if len(argv) > 1:
        print("Usage: run_one_example_config.py [CONFIG_PATH]", file=sys.stderr)
        return 2

    config_path = Path(argv[0]) if argv else CONFIG_PATH
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path

    if not config_path.exists():
        print(f"Config file not found: {config_path}", file=sys.stderr)
        return 1

    print(f"Running config: {config_path}")
    subprocess.run(
        [sys.executable, "-m", "collect_mie", str(config_path)],
        cwd=REPO_ROOT,
        check=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
