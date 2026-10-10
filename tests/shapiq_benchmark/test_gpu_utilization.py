"""GPU utilization admission and termination without requiring GPU hardware."""

from __future__ import annotations

import argparse
import importlib.util
import json
import signal
import subprocess
import sys
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "gpu_utilization", Path(__file__).parents[2] / "benchmark" / "gpu_utilization.py"
)
guard = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(guard)


@pytest.mark.parametrize(
    "output",
    [
        "GPU-a, N/A, Disabled",
        "GPU-a, nan, Disabled",
        "GPU-a, 101, Disabled",
        "GPU-a, 85, Enabled",
        "GPU-other, 95, Disabled",
        "GPU-a, 95, Disabled\nGPU-a, 95, Disabled",
        "",
    ],
)
def test_invalid_telemetry_is_not_a_pass(monkeypatch, output):
    monkeypatch.setattr(guard.shutil, "which", lambda _: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(guard.subprocess, "check_output", lambda *a, **kw: output)
    with pytest.raises(ValueError):
        guard.sample(["GPU-a"])


def test_sampling_is_scoped_to_allocated_uuids(monkeypatch):
    monkeypatch.setattr(guard.shutil, "which", lambda _: "/usr/bin/nvidia-smi")
    commands = []

    def query(command, **kwargs):
        commands.append(command)
        return "GPU-a, 81, Disabled\nGPU-b, 93, [N/A]"

    monkeypatch.setattr(guard.subprocess, "check_output", query)
    assert guard.sample(["GPU-a", "GPU-b"]) == {"GPU-a": 81, "GPU-b": 93}
    assert "--id=GPU-a,GPU-b" in commands[0]


@pytest.mark.parametrize("visibility", ["", "-1", "MIG-example", "0,", "all"])
def test_visibility_must_be_explicit_whole_devices(monkeypatch, visibility):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visibility)
    with pytest.raises(ValueError):
        guard.visible_uuids()


def test_cuda_driver_resolves_remapped_ordinal(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")

    class Driver:
        def cuInit(self, flags):
            return 0

        def cuDeviceGetCount(self, count):
            count._obj.value = 1
            return 0

        def cuDeviceGet(self, device, ordinal):
            assert ordinal == 0
            device._obj.value = 17
            return 0

        def cuDeviceGetUuid_v2(self, raw, device):
            assert device.value == 17
            for index in range(16):
                raw._obj[index] = index
            return 0

    monkeypatch.setattr(guard.ctypes, "CDLL", lambda _: Driver())
    assert guard.visible_uuids() == ["GPU-00010203-0405-0607-0809-0a0b0c0d0e0f"]


def simulate(monkeypatch, tmp_path, utilization, duration=10, returncode=0):
    """Drive a real monitor loop with deterministic clock, child and telemetry."""
    clock = [0.0]
    stopped = []

    class Child:
        pid = 123456
        returncode = None

        def poll(self):
            if clock[0] >= duration:
                self.returncode = returncode
            return self.returncode

    child = Child()
    monkeypatch.setattr(guard.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(guard.time, "sleep", lambda delay: clock.__setitem__(0, clock[0] + delay))
    monkeypatch.setattr(guard.subprocess, "check_output", lambda *a, **kw: '["GPU-a", "GPU-b"]')
    monkeypatch.setattr(guard.subprocess, "Popen", lambda *a, **kw: child)
    monkeypatch.setattr(guard, "stop_owned_group", stopped.append)

    def sample(identifiers):
        if utilization is None:
            guard.fail("missing metric")
        return utilization(clock[0]) if callable(utilization) else utilization

    monkeypatch.setattr(guard, "sample", sample)
    receipt = tmp_path / "receipt.json"
    args = argparse.Namespace(
        command=["prepare"], receipt=receipt, interval=1, warmup=1, window=3, threshold=80
    )
    code = guard.run(args)
    return code, json.loads(receipt.read_text()), stopped


def test_one_idle_gpu_stops_command_and_never_claims_completion(monkeypatch, tmp_path):
    code, receipt, stopped = simulate(monkeypatch, tmp_path, {"GPU-a": 100, "GPU-b": 60})
    assert code == guard.LOW_UTILIZATION
    assert receipt["reason"] == "sustained_low_utilization"
    assert not receipt["command_completed"]
    assert not receipt["utilization_verified"]
    assert receipt["elapsed_seconds"] == 4
    assert len(stopped) == 1


def test_success_requires_real_command_completion_and_bounded_samples(monkeypatch, tmp_path):
    code, receipt, _ = simulate(monkeypatch, tmp_path, {"GPU-a": 80, "GPU-b": 95}, duration=50)
    assert code == 0
    assert receipt["command_completed"] and receipt["utilization_verified"]
    assert receipt["samples_observed"] == 49
    assert len(receipt["recent_samples"]) == 4


@pytest.mark.parametrize("returncode", [1, 9, -15])
def test_busy_failed_command_is_still_failure(monkeypatch, tmp_path, returncode):
    code, receipt, _ = simulate(
        monkeypatch, tmp_path, {"GPU-a": 95, "GPU-b": 95}, returncode=returncode
    )
    assert code != 0
    assert receipt["reason"] == "command_failed"
    assert not receipt["utilization_verified"]


def test_short_command_is_not_false_verified(monkeypatch, tmp_path):
    code, receipt, _ = simulate(monkeypatch, tmp_path, {"GPU-a": 95, "GPU-b": 95}, duration=2)
    assert code == guard.INSUFFICIENT_OBSERVATION
    assert receipt["command_completed"]
    assert not receipt["utilization_verified"]


def test_missing_telemetry_prevents_launch(monkeypatch, tmp_path):
    code, receipt, stopped = simulate(monkeypatch, tmp_path, None)
    assert code == guard.MONITOR_FAILURE
    assert receipt["reason"] == "telemetry_failure"
    assert not stopped


def test_later_low_utilization_invalidates_earlier_good_window(monkeypatch, tmp_path):
    code, receipt, stopped = simulate(
        monkeypatch,
        tmp_path,
        lambda now: {"GPU-a": 100 if now < 6 else 0, "GPU-b": 95},
        duration=20,
    )
    assert code == guard.LOW_UTILIZATION
    assert not receipt["utilization_verified"]
    assert len(stopped) == 1


def test_real_owned_child_is_killed_after_ignoring_graceful_stop():
    code = (
        "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        "print('ready', flush=True); time.sleep(30)"
    )
    with subprocess.Popen(
        [sys.executable, "-c", code],
        start_new_session=True,
        stdout=subprocess.PIPE,
        text=True,
    ) as child:
        try:
            assert child.stdout.readline().strip() == "ready"
            guard.stop_owned_group(child, grace=0.1)
            assert child.returncode == -signal.SIGKILL
        finally:
            if child.poll() is None:
                child.kill()
