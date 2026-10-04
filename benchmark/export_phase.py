"""Export one complete cumulative campaign phase without changing its frozen inputs."""

from __future__ import annotations

import argparse
from pathlib import Path

from shapiq_benchmark.campaign import assemble_campaign
from shapiq_benchmark.campaign_replacements import replace_methods
from shapiq_benchmark.report import write_report


def main() -> None:
    """Authenticate every batch before computing a global public comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--through-phase", type=int, required=True)
    parser.add_argument("--supplement", type=Path, action="append", default=[])
    parser.add_argument(
        "--replacements", type=Path, help="Manifest of complete corrected method runs"
    )
    args = parser.parse_args()
    data = assemble_campaign(args.campaign, args.through_phase, supplements=tuple(args.supplement))
    if args.replacements:
        data = replace_methods(data, args.replacements)
    write_report(data, args.output, public=True, compact=True)


if __name__ == "__main__":
    main()
