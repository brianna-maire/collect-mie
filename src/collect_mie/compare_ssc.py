"""Compare experimental SSC medians to Mie SSC model curves at manifest diameters."""

from __future__ import annotations

from collect_mie.compare_channel import _main_for_command
from collect_mie.config_schema import CompareSscConfig


def main(argv: list[str] | None = None, *, config_path: str | None = None) -> None:
    _main_for_command(
        "compare-ssc",
        "ssc",
        CompareSscConfig,
        config_path=config_path,
        argv=argv,
    )


if __name__ == "__main__":
    main()
