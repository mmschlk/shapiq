import subprocess
import sys
import os

# Build and install C extensions from local source
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-e", ".",
    "--no-deps", "--ignore-requires-python"
])

# Add src to path for leaderboard package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# Now launch the actual app
from leaderboard.ui.ui import main
main()