"""Profiler backend tests: unreachable degradation + happy-path mismatch detection. Checkpoint 2.

The happy path uses respx-mocked Ollama payloads with a DELIBERATE advertised-vs-loaded context
mismatch (262144 advertised, 32768 loaded) so the headline feature and the budget math are both
unit-tested offline here, not deferred to the desktop.
"""

from __future__ import annotations

import httpx
import pytest

from pharos.accountant import Accountant
from pharos.config import PharosConfig
from pharos.profiler.backend import probe_backend
from pharos.profiler.profiler import build_profile, detect_ctx_mismatch
from pharos.profiler.types import GpuInfo

BASE = "http://localhost:11434"

TAGS = {
    "models": [
        {
            "name": "qwen3.5:9b",
            "model": "qwen3.5:9b",
            "size": 5_200_000_000,
            "details": {
                "format": "gguf",
                "family": "qwen3",
                "parameter_size": "9.0B",
                "quantization_level": "Q4_K_M",
            },
        }
    ]
}

PS = {
    "models": [
        {
            "name": "qwen3.5:9b",
            "model": "qwen3.5:9b",
            "size": 7_000_000_000,
            "size_vram": 7_000_000_000,
            "details": {"parameter_size": "9.0B", "quantization_level": "Q4_K_M"},
            "context_length": 32768,  # the ACTUAL loaded window
        }
    ]
}

SHOW = {
    "details": {"parameter_size": "9.0B", "quantization_level": "Q4_K_M", "format": "gguf"},
    "model_info": {
        "general.architecture": "qwen3",
        "qwen3.context_length": 262144,  # advertised GGUF max
        "qwen3.embedding_length": 4096,
    },
}


async def test_backend_unreachable(respx_mock) -> None:
    respx_mock.get(f"{BASE}/api/ps").mock(side_effect=httpx.ConnectError("connection refused"))
    info = await probe_backend(PharosConfig(backend_url=BASE))
    assert info.reachable is False
    assert info.loaded_ctx is None
    assert info.detail is not None


@pytest.mark.respx(assert_all_called=False)
async def test_backend_happy_path_detects_mismatch(respx_mock) -> None:
    respx_mock.get(f"{BASE}/api/tags").respond(json=TAGS)
    respx_mock.get(f"{BASE}/api/ps").respond(json=PS)
    respx_mock.post(f"{BASE}/api/show").respond(json=SHOW)

    info = await probe_backend(PharosConfig(backend_url=BASE))
    assert info.reachable is True
    assert info.model == "qwen3.5:9b"
    assert info.architecture == "qwen3"
    assert info.quantization == "Q4_K_M"
    assert info.advertised_max_ctx == 262144
    assert info.loaded_ctx == 32768
    assert info.size_vram_bytes == 7_000_000_000

    mismatch, ratio = detect_ctx_mismatch(info)
    assert mismatch is True
    assert ratio == pytest.approx(32768 / 262144)

    budget = Accountant(PharosConfig(backend_url=BASE, response_reserve=1024)).report(
        loaded_ctx=info.loaded_ctx, gpu=GpuInfo(available=False)
    )
    assert budget.usable_budget == 31744  # 32768 - 1024


@pytest.mark.respx(assert_all_called=False)
async def test_build_profile_composes_mismatch_and_budget(
    respx_mock, monkeypatch: pytest.MonkeyPatch
) -> None:
    respx_mock.get(f"{BASE}/api/tags").respond(json=TAGS)
    respx_mock.get(f"{BASE}/api/ps").respond(json=PS)
    respx_mock.post(f"{BASE}/api/show").respond(json=SHOW)
    monkeypatch.setattr(
        "pharos.profiler.profiler.probe_gpu",
        lambda: GpuInfo(available=False, detail="no GPU (test)"),
    )

    profile = await build_profile(PharosConfig(backend_url=BASE, response_reserve=1024))
    assert profile.ctx_mismatch is True
    assert profile.ctx_mismatch_ratio == pytest.approx(32768 / 262144)
    assert profile.budget.usable_budget == 31744
    assert profile.gpu.available is False
