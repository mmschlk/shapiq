"""Environment information retrieval for the leaderboard runner."""

from __future__ import annotations

import os
import platform
import subprocess
import sys


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
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                check=True,
            )
            if result.stdout.strip():
                cpu_name = result.stdout.strip()
        except Exception:
            pass

    # 2. Dynamically capture total system memory (alternative to psutil)
    ram_gb = None
    sys_type = platform.system().lower()

    try:
        if sys_type == "darwin":
            # macOS: Use sysctl to get total memory in bytes
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, check=True
            )
            bytes_val = int(result.stdout.strip())
            ram_gb = round(bytes_val / (1024**3), 2)

        elif sys_type == "linux":
            # Linux: Read MemTotal from /proc/meminfo
            if os.path.exists("/proc/meminfo"):
                with open("/proc/meminfo", "r") as f:
                    for line in f:
                        if line.startswith("MemTotal:"):
                            # Format is usually: MemTotal:       16345300 kB
                            kb_val = int(line.split()[1])
                            ram_gb = round(kb_val / (1024**2), 2)
                            break

        elif sys_type == "windows":
            # Windows: Use wmic to get total physical memory in bytes
            result = subprocess.run(
                ["wmic", "computersystem", "get", "totalphysicalmemory"],
                capture_output=True,
                text=True,
                check=True,
            )
            # Output looks like: TotalPhysicalMemory\n 17179869184
            lines = result.stdout.strip().split("\n")
            if len(lines) > 1:
                bytes_val = int(lines[1].strip())
                ram_gb = round(bytes_val / (1024**3), 2)
    except Exception:
        ram_gb = None

    return {
        "cpu": cpu_name,
        "ram_gb": ram_gb,
        "python_version": sys.version.split()[0],
    }
