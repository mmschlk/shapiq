"""Copy notebooks from examples/api_examples to docs/source/api_examples."""

import os
import shutil

# Source: project/examples/api_examples/
src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "examples", "api_examples"))

# Destination: project/docs/source/api_examples/
dst_dir = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "docs", "source", "api_examples")
)

# Clear and re-copy
if os.path.exists(dst_dir):
    shutil.rmtree(dst_dir)
shutil.copytree(src_dir, dst_dir)
