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
from pharos.naming import normalize_model_name
from pharos.profiler.backend import _extract_loaded_ctx, probe_backend
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


# --- model-name tag matching ---------------------------------------------------------------
#
# Mirrors the real desktop payload: Ollama reports "<name>:latest" from /api/ps even though it
# accepts the untagged name on every endpoint. An untagged pharos.toml `model` must still match,
# or loaded_ctx comes back None and the mismatch banner silently never fires.

PS_LATEST = {
    "models": [
        {
            "name": "qwen3.5-9b-heretic:latest",
            "model": "qwen3.5-9b-heretic:latest",
            "size": 9_688_129_207,
            "size_vram": 9_688_129_207,
            "details": {"parameter_size": "9.0B", "quantization_level": "Q8_0"},
            "context_length": 32768,
        }
    ]
}

SHOW_QWEN35 = {
    "details": {"parameter_size": "9.0B", "quantization_level": "Q8_0", "format": "gguf"},
    "model_info": {
        "general.architecture": "qwen35",
        "qwen35.context_length": 262144,
    },
}


@pytest.mark.respx(assert_all_called=False)
async def test_untagged_config_matches_latest_ps_entry(respx_mock) -> None:
    """The common case: user writes the untagged name, /api/ps reports ":latest"."""
    respx_mock.get(f"{BASE}/api/ps").respond(json=PS_LATEST)
    respx_mock.post(f"{BASE}/api/show").respond(json=SHOW_QWEN35)

    info = await probe_backend(
        PharosConfig(backend_url=BASE, model="qwen3.5-9b-heretic")
    )
    assert info.loaded_ctx == 32768
    assert info.size_vram_bytes == 9_688_129_207
    assert info.advertised_max_ctx == 262144
    assert info.architecture == "qwen35"

    mismatch, ratio = detect_ctx_mismatch(info)
    assert mismatch is True
    assert ratio == pytest.approx(32768 / 262144)


@pytest.mark.respx(assert_all_called=False)
async def test_tagged_config_matches_tagged_ps_entry(respx_mock) -> None:
    respx_mock.get(f"{BASE}/api/ps").respond(json=PS_LATEST)
    respx_mock.post(f"{BASE}/api/show").respond(json=SHOW_QWEN35)

    info = await probe_backend(
        PharosConfig(backend_url=BASE, model="qwen3.5-9b-heretic:latest")
    )
    assert info.loaded_ctx == 32768
    assert detect_ctx_mismatch(info)[0] is True


@pytest.mark.respx(assert_all_called=False)
async def test_explicit_tag_does_not_match_a_different_tag(respx_mock) -> None:
    """Normalization defaults a missing tag; it must not make distinct tags equal."""
    respx_mock.get(f"{BASE}/api/tags").respond(json=TAGS)
    respx_mock.get(f"{BASE}/api/ps").respond(json=PS_LATEST)
    respx_mock.post(f"{BASE}/api/show").respond(json=SHOW_QWEN35)

    info = await probe_backend(
        PharosConfig(backend_url=BASE, model="qwen3.5-9b-heretic:q4")
    )
    assert info.loaded_ctx is None
    assert info.size_vram_bytes is None


@pytest.mark.respx(assert_all_called=False)
async def test_absent_model_still_reports_none(respx_mock) -> None:
    """A model that genuinely is not loaded yields None, not a false match."""
    respx_mock.get(f"{BASE}/api/tags").respond(json=TAGS)
    respx_mock.get(f"{BASE}/api/ps").respond(json=PS_LATEST)
    respx_mock.post(f"{BASE}/api/show").respond(json=SHOW_QWEN35)

    info = await probe_backend(PharosConfig(backend_url=BASE, model="not-installed"))
    assert info.model == "not-installed"
    assert info.loaded_ctx is None
    assert detect_ctx_mismatch(info) == (False, None)


def test_normalize_model_name_tag_handling() -> None:
    assert normalize_model_name("qwen3.5-9b-heretic") == "qwen3.5-9b-heretic:latest"
    assert normalize_model_name("qwen3.5-9b-heretic:latest") == "qwen3.5-9b-heretic:latest"
    assert normalize_model_name("qwen3.5:9b") == "qwen3.5:9b"
    assert normalize_model_name("hf.co/user/repo-GGUF:Q4_K_M") == "hf.co/user/repo-GGUF:Q4_K_M"
    # A registry port is not a tag: the last colon precedes a "/".
    assert normalize_model_name("localhost:5000/mymodel") == "localhost:5000/mymodel:latest"


def test_loaded_ctx_ignores_details_context_length() -> None:
    """/api/tags puts the ADVERTISED max under details.context_length; never read it as loaded."""
    entry = {"name": "m:latest", "details": {"context_length": 262144}}
    assert _extract_loaded_ctx(entry) is None


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
