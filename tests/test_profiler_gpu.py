"""GPU detection: NVML, the nvidia-smi fallback under it, and giving up cleanly.

The fallback is the interesting half. It runs only when NVML has already failed -- a driver
mismatch, a container without the bindings, a machine where the library is present and the
device is not -- so it runs on machines where something is already wrong, and it parses text
that a tool prints for humans. Every way that parse can fail has to end at available=False
with a reason, because the alternative is a crash on the one path that exists for a machine
in trouble.
"""

from __future__ import annotations

import subprocess
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


# --- the nvidia-smi fallback, which runs when NVML has already failed ----------------------------


def _nvml_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """Put NVML out of action so probe_gpu reaches the fallback."""
    import pynvml

    def _raise() -> None:
        raise RuntimeError("NVML unavailable")

    monkeypatch.setattr(pynvml, "nvmlInit", _raise)


def _smi(monkeypatch: pytest.MonkeyPatch, stdout: str, *, exc: Exception | None = None) -> None:
    """Pretend nvidia-smi is on PATH and answers with ``stdout`` (or raises)."""
    monkeypatch.setattr(gpu_mod.shutil, "which", lambda _name: "/usr/bin/nvidia-smi")

    def _run(*_args: object, **_kwargs: object) -> types.SimpleNamespace:
        if exc is not None:
            raise exc
        return types.SimpleNamespace(stdout=stdout, returncode=0)

    monkeypatch.setattr(gpu_mod.subprocess, "run", _run)


def test_nvidia_smi_answers_when_nvml_cannot(monkeypatch: pytest.MonkeyPatch) -> None:
    """The whole point of the fallback: a real answer from a machine NVML would not serve."""
    _nvml_fails(monkeypatch)
    _smi(monkeypatch, "NVIDIA GeForce RTX 3060, 12288, 9000, 3288\n")

    info = probe_gpu()

    assert info.available is True
    assert info.source == "nvidia-smi"
    assert (info.name, info.total_mib, info.used_mib, info.free_mib) == (
        "NVIDIA GeForce RTX 3060",
        12288,
        9000,
        3288,
    )


@pytest.mark.parametrize(
    ("case", "stdout", "exc"),
    [
        ("it timed out", "", subprocess.TimeoutExpired(cmd="nvidia-smi", timeout=5)),
        ("it could not be executed", "", OSError("exec failed")),
        ("it exited non-zero", "", subprocess.CalledProcessError(1, "nvidia-smi")),
        ("it printed nothing", "\n", None),
        ("it printed too few fields", "RTX 3060, 12288\n", None),
        ("it printed too many fields", "RTX 3060, 1, 2, 3, 4\n", None),
        ("the memory is not a number", "RTX 3060, N/A, N/A, N/A\n", None),
    ],
)
def test_a_fallback_that_cannot_answer_degrades_rather_than_raising(
    monkeypatch: pytest.MonkeyPatch, case: str, stdout: str, exc: Exception | None
) -> None:
    """Every failure of a tool meant for humans ends at available=False with a reason.

    This path exists for a machine that is already misconfigured; a traceback here would
    replace the diagnosis with a second problem.
    """
    _nvml_fails(monkeypatch)
    _smi(monkeypatch, stdout, exc=exc)

    info = probe_gpu()

    assert info.available is False, case
    assert info.detail, "an unavailable GPU must say why"


def test_no_nvidia_smi_at_all_is_the_documented_end_of_the_line(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Neither NVML nor the binary: the message names both, so the user knows what to install."""
    _nvml_fails(monkeypatch)
    monkeypatch.setattr(gpu_mod.shutil, "which", lambda _name: None)

    info = probe_gpu()

    assert info.available is False
    assert info.detail is not None
    assert "NVML" in info.detail and "nvidia-smi" in info.detail
