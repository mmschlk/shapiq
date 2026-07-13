"""Entry point for running the leaderboard UI on HuggingFace Spaces."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

# Build and install C extensions from local source
subprocess.check_call( # noqa: S603
    [sys.executable, "-m", "pip", "install", "-e", ".", "--no-deps", "--ignore-requires-python"]
)

# Add src to path for leaderboard package
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

# Now launch the actual app
from leaderboard.ui.ui import main # noqa: E402

main()
