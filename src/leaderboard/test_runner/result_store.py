import json
import hashlib
import datetime
import platform
import psutil
import subprocess

# Gesamter RAM in GB (gerundet)
total_ram = round(psutil.virtual_memory().total / (1024**3), 2)


def get_cpu_name():
    # Versucht den exakten Namen unter Windows auszulesen
    try:
        return subprocess.check_output(["wmic", "cpu", "get", "name"]).decode().split('\n')[1].strip()
    except:
        return platform.processor()


def make_run_id(game_name, approximator_name, budget, seed, index):
    params = {
        "game": game_name,
        "approximator": approximator_name,
        "budget": budget,
        "seed": seed,
        "index": index,
    }
    return hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()


def write_record(path: str, record: dict):
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def make_record(
    game_name, game_params, n_players,
    approximator_name, approximator_params,
    shapiq_version, index, max_order, budget, seed,
    ground_truth_method, metrics,
    runtime_seconds, run_failed=False, error_message=None
) -> dict:
    return {
        "run_id": make_run_id(game_name, approximator_name, budget, seed, index),
        "game_name": game_name,
        "game_params": game_params,
        "n_players": n_players,
        "approximator_name": approximator_name,
        "approximator_params": approximator_params,
        "shapiq_version": shapiq_version,
        "index": index,
        "max_order": max_order,
        "budget": budget,
        "seed": seed,
        "ground_truth_method": ground_truth_method,
        "metrics": metrics,
        "runtime_seconds": runtime_seconds,
        "run_failed": run_failed,
        "error_message": error_message,
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "hardware": {
            "cpu": get_cpu_name(),
            "ram_gb": total_ram,
            "python_version": platform.python_version(),
        },
        "notes": "",
    }
