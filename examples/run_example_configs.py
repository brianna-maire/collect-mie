#!/usr/bin/env python3
"""
Run all example YAML configs via collect-mie.

Each `*_run.example.yaml` must include `run.command`.
Writes figures and run records under `examples/output/`.

Skips `compare-fcs` when `examples/compare_manifest.txt` is missing.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from collect_mie.run_config import COMMAND_TARGETS, load_run_command


def _output_path(config_path: Path, repo_root: Path) -> Path | None:
    import yaml

    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    run = data.get("run") if isinstance(data, dict) else None
    if isinstance(run, dict) and isinstance(run.get("args"), dict):
        out = run["args"].get("output")
        if isinstance(out, str) and out.strip():
            path = Path(out.strip())
            if not path.is_absolute():
                path = repo_root / path
            return path
    return None


def _manifest_missing(config_path: Path, repo_root: Path) -> bool:
    import yaml

    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    manifest = None
    if isinstance(data, dict) and isinstance(data.get("compare_fcs"), dict):
        manifest = data["compare_fcs"].get("manifest")
    if manifest is None and isinstance(data.get("run"), dict):
        run_args = data["run"].get("args")
        if isinstance(run_args, dict):
            manifest = run_args.get("manifest")
    if manifest is None:
        manifest = "examples/compare_manifest.txt"
    path = Path(manifest)
    if not path.is_absolute():
        path = repo_root / path
    return not path.is_file()


def main() -> int:
    examples_dir = Path(__file__).resolve().parent
    repo_root = examples_dir.parent

    configs = sorted(examples_dir.glob("*_run.example.yaml"))
    if not configs:
        print("No *_run.example.yaml files found.", file=sys.stderr)
        return 1

    exe = sys.executable
    ran = 0
    skipped: list[str] = []

    for config_path in configs:
        rel_config = config_path.relative_to(repo_root)
        try:
            command = load_run_command(str(rel_config))
        except SystemExit as err:
            print(f"Skip {config_path.name}: {err}", file=sys.stderr)
            skipped.append(config_path.name)
            continue

        if command not in COMMAND_TARGETS:
            print(
                f"Skip {config_path.name}: unknown run.command {command!r}",
                file=sys.stderr,
            )
            skipped.append(config_path.name)
            continue

        if command == "compare-fcs" and _manifest_missing(config_path, repo_root):
            print(
                "Skipping compare-fcs: add examples/compare_manifest.txt "
                "(see compare_manifest.example.txt)."
            )
            skipped.append(config_path.name)
            continue

        print(f"Running {command} <- {rel_config}")
        subprocess.run(
            [exe, "-m", "collect_mie", str(rel_config)],
            cwd=repo_root,
            check=True,
        )
        out_path = _output_path(config_path, repo_root)
        if out_path is not None:
            print(f"  -> {out_path}")
        ran += 1

    print(f"Completed {ran} example config run(s).")
    if skipped:
        print(f"Skipped {len(skipped)}: {', '.join(skipped)}")
    return 0 if ran > 0 or not configs else 1


if __name__ == "__main__":
    raise SystemExit(main())
