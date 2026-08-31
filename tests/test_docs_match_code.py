"""The documentation is checked against the code, because prose does not fail a build.

Every other claim this project makes is measured. These are the two that were not, and both
had already drifted: `pharos check` and `pharos split` grew four flags the README never
mentioned — `--target` among them, which is the flag that makes the README's own "runs with no
GPU and no backend" true — and there was nothing to notice.

The failure mode is specific to a public repository. A user reads the README, types a flag it
documents, and gets `unrecognized arguments`; or a flag exists, does something useful, and is
found only by people who run `--help`. Neither shows up in a test suite that only exercises the
code, and neither is caught by `ruff` or `mypy`, because prose is not compiled.

So the flags are read out of argparse with `ast` rather than by importing and running the
parser: the parser is built inside `main()` behind the config load, and a test that has to
reach it would be testing the wrong thing. The README is read as text. Both directions are
asserted, and the second matters more than the first — a documented flag that does not exist
is a promise the tool breaks the moment somebody believes it.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from pharos.config import PharosConfig

_ROOT = Path(__file__).resolve().parent.parent


def _argparse_flags(module: Path) -> set[str]:
    """Every long option the module registers with argparse."""
    found: set[str] = set()
    for node in ast.walk(ast.parse(module.read_text(encoding="utf-8"))):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
        ):
            for argument in node.args:
                if isinstance(argument, ast.Constant) and str(argument.value).startswith("--"):
                    found.add(str(argument.value))
    return found


def _readme_table(heading: str) -> set[str]:
    """The flags named in one of the README's option tables."""
    readme = (_ROOT / "README.md").read_text(encoding="utf-8")
    assert heading in readme, f"the README no longer has a {heading!r} table"
    block = readme.split(heading, 1)[1].split("\n\n", 1)[0]
    return set(re.findall(r"`(--[a-z-]+)", block))


@pytest.mark.parametrize(
    ("module", "heading"),
    [
        ("pharos/agent/cli.py", "| `pharos run` | |"),
        ("pharos/preflight/cli.py", "| `pharos check` / `pharos split` | |"),
    ],
)
def test_every_flag_is_documented_and_every_documented_flag_exists(
    module: str, heading: str
) -> None:
    """Both directions. The second is the one that breaks a user's day."""
    real = _argparse_flags(_ROOT / module)
    documented = _readme_table(heading)

    assert real, f"no flags were parsed out of {module} — the AST walk has stopped working"
    assert not (real - documented), (
        f"{module} accepts flags the README does not document: {sorted(real - documented)}"
    )
    assert not (documented - real), (
        f"the README documents flags {module} does not accept: {sorted(documented - real)}"
    )


def test_every_config_key_is_in_the_example_file() -> None:
    """`pharos.toml` is gitignored, so the example is the only documentation of a key.

    A field added to PharosConfig and left out of the example is invisible: nobody can set what
    they cannot find, and the example is what a first run copies.
    """
    example = (_ROOT / "pharos.toml.example").read_text(encoding="utf-8")
    documented = set(re.findall(r"^\s*#?\s*([a-z_][a-z0-9_]*)\s*=", example, re.MULTILINE))
    missing = sorted(set(PharosConfig.model_fields) - documented)

    assert not missing, f"pharos.toml.example does not mention: {missing}"


def test_the_example_file_only_names_real_keys() -> None:
    """The config forbids unknown keys, so a stale example is not a typo — it is a crash.

    Copying `pharos.toml.example` to `pharos.toml` is the documented first step, and
    `extra="forbid"` means one renamed key there stops Pharos from starting at all.
    """
    example = (_ROOT / "pharos.toml.example").read_text(encoding="utf-8")
    named = set(re.findall(r"^\s*#?\s*([a-z_][a-z0-9_]*)\s*=", example, re.MULTILINE))
    unknown = sorted(named - set(PharosConfig.model_fields))

    assert not unknown, f"pharos.toml.example names keys the config would reject: {unknown}"
