"""Export one complete cumulative campaign phase without changing its frozen inputs."""

from __future__ import annotations

import argparse
from pathlib import Path

from shapiq_benchmark.campaign import assemble_campaign
from shapiq_benchmark.report import write_report


def main() -> None:
    """Authenticate every batch before computing a global public comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--through-phase", type=int, required=True)
    parser.add_argument("--supplement", type=Path, action="append", default=[])
    parser.add_argument(
        "--cache-dir", type=Path, help="Reuse verified batch normalization in a private lab cache"
    )
    parser.add_argument(
        "--replacements", type=Path, help="Manifest of complete corrected method runs"
    )
    parser.add_argument(
        "--backend-supersession",
        type=Path,
        help="Authenticated failed-GPU receipt and separately qualified CPU recovery campaign",
    )
    args = parser.parse_args()
    data = assemble_campaign(
        args.campaign,
        args.through_phase,
        supplements=tuple(args.supplement),
        replacements=args.replacements,
        backend_supersession=args.backend_supersession,
        cache_dir=args.cache_dir,
    )
    write_report(data, args.output, public=True, compact=True)


if __name__ == "__main__":
    main()
