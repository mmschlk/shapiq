"""Environment information retrieval for the leaderboard runner."""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from pathlib import Path


def get_hardware_info() -> dict:
    """Retrieve runtime hardware environment and Python engine specifications.

    Returns:
        A dictionary containing the parsed CPU processor name, total system
        RAM formatted in gigabytes (GB), and the clean Python runtime version.
    """
    # 1. Parse the specific processor name
    cpu_name = platform.processor() or platform.machine()

    # Get specific CPU brand name on macOS
    if platform.system().lower() == "darwin":
        try:
            result = subprocess.run(
                ["/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                check=True,
            )
            if result.stdout.strip():
                cpu_name = result.stdout.strip()
        except (subprocess.SubprocessError, FileNotFoundError):
            pass

    # 2. Dynamically capture total system memory (alternative to psutil)
    ram_gb = None
    sys_type = platform.system().lower()

    try:
        if sys_type == "darwin":
            # macOS: Use sysctl to get total memory in bytes
            result = subprocess.run(
                ["/usr/sbin/sysctl", "-n", "hw.memsize"], capture_output=True, text=True, check=True
            )
            bytes_val = int(result.stdout.strip())
            ram_gb = round(bytes_val / (1024**3), 2)

        elif sys_type == "linux":
            # Linux: Read MemTotal from /proc/meminfo
            meminfo_path = Path("/proc/meminfo")
            if meminfo_path.exists():
                with meminfo_path.open() as f:
                    for line in f:
                        if line.startswith("MemTotal:"):
                            # Format is usually: MemTotal:       16345300 kB
                            kb_val = int(line.split()[1])
                            ram_gb = round(kb_val / (1024**2), 2)
                            break

        elif sys_type == "windows":
            system_root = os.environ.get("SYSTEMROOT", "C:\\Windows")

            wmic_path = Path(system_root) / "System32" / "wbem" / "wmic.exe"

            cmd_executable = str(wmic_path) if wmic_path.exists() else "wmic"

            result = subprocess.run(  # noqa: S603
                [cmd_executable, "computersystem", "get", "totalphysicalmemory"],
                capture_output=True,
                text=True,
                check=True,
            )

            lines = result.stdout.strip().split("\n")

            if len(lines) > 1:
                bytes_val = int(lines[1].strip())

                ram_gb = round(bytes_val / (1024**3), 2)

    except (subprocess.SubprocessError, ValueError, IndexError, OSError):
        ram_gb = None

    return {
        "cpu": cpu_name,
        "ram_gb": ram_gb,
        "python_version": sys.version.split()[0],
    }
