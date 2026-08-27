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


def test_readme_describes_one_version() -> None:
    """The README documents what Pharos IS, not the order its tiers arrived in.

    It drifted the other way once already: a section for one tier was added while the status
    line above it still named the previous one, and nothing caught it because the only version
    check compared the module to the packaging metadata. The fix was to stop numbering the
    prose at all — `CHANGELOG.md` is the release history — so this guards that instead. A
    heading or status line that pins the front page to a release is how the drift comes back.
    """
    readme = (_REPO_ROOT / "README.md").read_text(encoding="utf-8")

    assert re.search(r"\*\*Status: v\d", readme) is None, (
        "README carries a status line pinned to a release; the CHANGELOG holds the history"
    )
    tiered = re.findall(r"^#{2,4} .*\(v\d+\.\d+", readme, re.MULTILINE)
    assert not tiered, f"README headings are pinned to releases: {tiered}"

    # Prose that dates a feature to a release is the same drift in a sentence.
    dated = re.findall(r"\b(?:from|since|until|new in|before) v\d+\.\d+", readme, re.IGNORECASE)
    assert not dated, f"README dates features to releases: {sorted(set(dated))}"


def test_readme_documents_every_run_flag() -> None:
    """A flag the CLI accepts and the README never mentions is a feature nobody finds.

    `--exclude` shipped and went undocumented for exactly that reason: it was added to both
    parsers and to neither page.
    """
    readme = (_REPO_ROOT / "README.md").read_text(encoding="utf-8")
    parser_source = (_REPO_ROOT / "pharos" / "agent" / "cli.py").read_text(encoding="utf-8")
    flags = set(re.findall(r'"(--[a-z][a-z-]+)"', parser_source))
    missing = sorted(flag for flag in flags if flag not in readme)
    assert not missing, f"`pharos run` flags missing from the README: {missing}"


def test_the_files_per_part_default_is_the_measured_one() -> None:
    """Two, not four. On a 13-file task against qwen2.5-coder:14b a part completes about 1.5
    to 2.0 files whatever it is given; at four per part that task covered 46/62/46%, at two it
    covered 92%. The number is load-bearing, so a silent change should fail here first."""
    assert PharosConfig().max_files_per_part == 2
