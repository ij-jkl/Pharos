"""Profiler GPU tests: NVML success path + graceful degradation to available=False. Checkpoint 2."""

from __future__ import annotations

import types

import pytest

import pharos.profiler.gpu as gpu_mod
from pharos.profiler.gpu import probe_gpu


def test_gpu_unavailable_when_nvml_fails_and_no_smi(monkeypatch: pytest.MonkeyPatch) -> None:
    import pynvml

    def _raise() -> None:
        raise RuntimeError("NVML unavailable")

    monkeypatch.setattr(pynvml, "nvmlInit", _raise)
    monkeypatch.setattr(gpu_mod.shutil, "which", lambda _name: None)

    info = probe_gpu()
    assert info.available is False
    assert info.detail is not None


def test_gpu_available_via_nvml(monkeypatch: pytest.MonkeyPatch) -> None:
    import pynvml

    mem = types.SimpleNamespace(total=12 * 1024**3, used=4 * 1024**3, free=8 * 1024**3)
    monkeypatch.setattr(pynvml, "nvmlInit", lambda: None)
    monkeypatch.setattr(pynvml, "nvmlShutdown", lambda: None)
    monkeypatch.setattr(pynvml, "nvmlDeviceGetHandleByIndex", lambda _index: "handle")
    monkeypatch.setattr(pynvml, "nvmlDeviceGetMemoryInfo", lambda _handle: mem)
    monkeypatch.setattr(pynvml, "nvmlDeviceGetName", lambda _handle: "NVIDIA GeForce RTX 3060")

    info = probe_gpu()
    assert info.available is True
    assert info.name == "NVIDIA GeForce RTX 3060"
    assert info.total_mib == 12 * 1024
    assert info.free_mib == 8 * 1024
    assert info.source == "nvml"
