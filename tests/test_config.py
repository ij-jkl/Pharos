"""Config loading / validation tests (pharos.toml -> PharosConfig). Checkpoint 2."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

import pharos
from pharos.config import ConfigError, PharosConfig, load_config

_REPO_ROOT = Path(__file__).resolve().parents[1]


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


def test_shipped_example_is_a_valid_config() -> None:
    """`pharos.toml.example` is copied verbatim to `pharos.toml` on a fresh install, and the
    model forbids unknown keys — so a key documented in the example but never added to
    PharosConfig (or renamed out of it) breaks setup for every new user at step one."""
    cfg = load_config(_REPO_ROOT / "pharos.toml.example")
    assert cfg.handoff_reserve == 500  # the value the example documents


def test_version_matches_pyproject() -> None:
    """Two hand-edited places hold the version; a release that bumps one and forgets the
    other ships a package whose metadata disagrees with the module it installs."""
    declared = tomllib.loads((_REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert pharos.__version__ == declared["project"]["version"]


def test_readme_status_line_matches_the_version() -> None:
    """The README is the third hand-edited place the version lives, and the one people read.

    It drifted once already: a v0.4 section was added while the status line above it still
    said v0.3, and the check above passed the whole time because it only compares the module
    to the packaging metadata. A repo whose own front page disagrees with what it installs is
    exactly the kind of unlabelled inconsistency this project is about.
    """
    readme = (_REPO_ROOT / "README.md").read_text(encoding="utf-8")
    status = re.search(r"\*\*Status: v(\d+\.\d+)", readme)
    assert status is not None, "README has no **Status: vX.Y** line"

    major_minor = ".".join(pharos.__version__.split(".")[:2])
    assert status.group(1) == major_minor

    # Every tier heading must be at or below the declared version; a section for a tier that
    # has not shipped reads as a promise rather than a description.
    released = tuple(int(part) for part in major_minor.split("."))
    headings = set(re.findall(r"^## .*\(v(\d+\.\d+) \"", readme, re.MULTILINE))
    ahead = [v for v in headings if tuple(int(p) for p in v.split(".")) > released]
    assert not ahead, f"README documents tiers not yet released: {sorted(ahead)}"


def test_the_files_per_part_default_is_the_measured_one() -> None:
    """Two, not four. On a 13-file task against qwen2.5-coder:14b a part completes about 1.5
    to 2.0 files whatever it is given; at four per part that task covered 46/62/46%, at two it
    covered 92%. The number is load-bearing, so a silent change should fail here first."""
    assert PharosConfig().max_files_per_part == 2
