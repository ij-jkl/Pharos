"""Config loading / validation tests (pharos.toml -> PharosConfig). Checkpoint 2."""

from __future__ import annotations

from pathlib import Path

import pytest

from pharos.config import ConfigError, load_config


def test_defaults_when_no_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)  # no pharos.toml in this dir -> all defaults
    cfg = load_config()
    assert cfg.backend_url == "http://localhost:11434"
    assert cfg.response_reserve == 1024
    assert cfg.warn_threshold == pytest.approx(0.80)
    assert cfg.alert_threshold == pytest.approx(0.90)
    assert cfg.proxy_port == 11435
    assert cfg.model is None
    assert cfg.log_file == "pharos.log"


def test_load_overrides(tmp_path: Path) -> None:
    path = tmp_path / "pharos.toml"
    path.write_text(
        'backend_url = "http://desktop:11434"\n'
        'model = "qwen3.5-9b-heretic"\n'
        "response_reserve = 2048\n"
        "warn_threshold = 0.7\n"
        "alert_threshold = 0.85\n",
        encoding="utf-8",
    )
    cfg = load_config(path)
    assert cfg.backend_url == "http://desktop:11434"
    assert cfg.model == "qwen3.5-9b-heretic"
    assert cfg.response_reserve == 2048
    assert cfg.warn_threshold == pytest.approx(0.7)
    assert cfg.alert_threshold == pytest.approx(0.85)


def test_threshold_order_rejected(tmp_path: Path) -> None:
    path = tmp_path / "pharos.toml"
    path.write_text("warn_threshold = 0.95\nalert_threshold = 0.80\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(path)


def test_unknown_key_rejected(tmp_path: Path) -> None:
    path = tmp_path / "pharos.toml"
    path.write_text("bogus_key = 1\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(path)


def test_out_of_range_threshold_rejected(tmp_path: Path) -> None:
    path = tmp_path / "pharos.toml"
    path.write_text("warn_threshold = 1.5\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(path)


def test_missing_explicit_path_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        load_config(tmp_path / "does-not-exist.toml")
