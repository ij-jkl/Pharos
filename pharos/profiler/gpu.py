"""GPU detection via nvidia-ml-py (imported as `pynvml`), with an `nvidia-smi` CSV fallback.

Must never crash on a machine without an NVIDIA GPU: on NVML init failure it falls back to
nvidia-smi, and if that is unavailable too it returns GpuInfo(available=False) so the VRAM gauge
can show N/A.
"""

from __future__ import annotations

import shutil
import subprocess

from pharos.profiler.types import GpuInfo

_BYTES_PER_MIB = 1024 * 1024


def probe_gpu() -> GpuInfo:
    """Detect the primary NVIDIA GPU, degrading to GpuInfo(available=False) when none is found."""
    via_nvml = _probe_nvml()
    if via_nvml is not None:
        return via_nvml
    via_smi = _probe_nvidia_smi()
    if via_smi is not None:
        return via_smi
    return GpuInfo(
        available=False,
        detail="no NVIDIA GPU detected (NVML unavailable and nvidia-smi not found)",
    )


def _probe_nvml() -> GpuInfo | None:
    """Query NVML. Returns None on any failure so the caller can fall back."""
    try:
        import pynvml  # nvidia-ml-py; the import itself succeeds even without a driver

        pynvml.nvmlInit()
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            raw_name = pynvml.nvmlDeviceGetName(handle)
            name = raw_name.decode() if isinstance(raw_name, bytes) else str(raw_name)
            return GpuInfo(
                available=True,
                name=name,
                total_mib=int(mem.total) // _BYTES_PER_MIB,
                used_mib=int(mem.used) // _BYTES_PER_MIB,
                free_mib=int(mem.free) // _BYTES_PER_MIB,
                source="nvml",
            )
        finally:
            pynvml.nvmlShutdown()
    except Exception:
        # Any NVML problem (no driver, no GPU, binding error) -> degrade, never crash.
        return None


def _probe_nvidia_smi() -> GpuInfo | None:
    """Fall back to parsing `nvidia-smi` CSV output. Returns None when unavailable."""
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return None
    try:
        result = subprocess.run(
            [
                exe,
                "--query-gpu=name,memory.total,memory.used,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None

    lines = result.stdout.strip().splitlines()
    if not lines:
        return None
    parts = [p.strip() for p in lines[0].split(",")]
    if len(parts) != 4:
        return None
    name, total, used, free = parts
    try:
        return GpuInfo(
            available=True,
            name=name,
            total_mib=int(total),
            used_mib=int(used),
            free_mib=int(free),
            source="nvidia-smi",
        )
    except ValueError:
        return None
