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
import sys
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


def test_the_package_and_the_project_agree_on_the_version() -> None:
    """`__version__` and pyproject's `version` are written out separately and must match.

    They are read by different things -- the CLI and an installed wheel's metadata -- so a
    mismatch ships a build that reports one number and identifies itself as another, and the
    first person to notice is someone comparing a bug report against a release.
    """
    import tomllib

    import pharos

    with (_ROOT / "pyproject.toml").open("rb") as handle:
        declared = tomllib.load(handle)["project"]["version"]

    assert pharos.__version__ == declared, (
        f"pharos.__version__ is {pharos.__version__} and pyproject says {declared}"
    )


def test_the_cli_can_say_its_own_version(capsys: pytest.CaptureFixture[str]) -> None:
    """The first line of any bug report. The number existed; nothing could print it."""
    import pharos
    from pharos.__main__ import main

    for flag in ("--version", "-V"):
        sys.argv = ["pharos", flag]
        main()  # must not raise SystemExit: asking the version is not an error
        assert capsys.readouterr().out.strip() == f"pharos {pharos.__version__}"


def test_the_cli_can_explain_itself(capsys: pytest.CaptureFixture[str]) -> None:
    """`--help` is the first thing anyone types, and it was the one thing that answered wrong.

    There is no top-level argparse parser to supply this -- dispatch is by hand so that
    `pharos check` does not pay to import the TUI -- so `--help` fell through to the
    unknown-argument branch: it printed to stderr, called the argument unknown, and exited 1.
    Every one of those is wrong for a request the tool is happy to answer, and a nonzero exit
    on `--help` is the kind of thing a packaging script notices before a person does.
    """
    from pharos.__main__ import main

    for flag in ("--help", "-h", "help"):
        sys.argv = ["pharos", flag]
        main()  # must not raise SystemExit: asking what a tool does is not an error
        printed = capsys.readouterr()
        assert printed.err == "", f"`pharos {flag}` wrote to stderr"
        assert printed.out.startswith("pharos —"), f"`pharos {flag}` printed no usage"


def test_the_usage_block_names_every_subcommand_the_dispatcher_accepts() -> None:
    """The usage text is hand-written, so it can drift from the dispatch it describes.

    This is the same failure this module exists for, one level up: a subcommand that works and
    is documented nowhere is found only by reading the source, and a usage block naming one
    that no longer dispatches is a promise the tool breaks on the next line the user types.
    Read out of `main()` with `ast` rather than by calling it, for the reason given at the top.
    """
    from pharos.__main__ import _USAGE

    source = ast.parse((_ROOT / "pharos" / "__main__.py").read_text(encoding="utf-8"))
    main_def = next(
        node
        for node in ast.walk(source)
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )

    dispatched: set[str] = set()
    for node in ast.walk(main_def):
        # `argv[0] == "check"` and `argv[0] in ("--version", "-V")` alike.
        if isinstance(node, ast.Compare) and isinstance(node.ops[0], ast.Eq | ast.In):
            for operand in node.comparators:
                for constant in ast.walk(operand):
                    if isinstance(constant, ast.Constant) and isinstance(constant.value, str):
                        dispatched.add(constant.value)

    # The help aliases describe themselves; everything else has to be in the block.
    for name in sorted(dispatched - {"--help", "-h", "help", "-V"}):
        assert name in _USAGE, (
            f"`pharos {name}` dispatches and the usage block does not mention it, so the only "
            f"way to find it is to read __main__.py"
        )


def test_the_readme_does_not_overstate_the_test_count(request: pytest.FixtureRequest) -> None:
    """The README's headline count had gone stale by 25 while every flag beside it was checked.

    Overstating is the half that matters: a README claiming more tests than the suite contains
    is a false claim about the one thing a reader cannot verify without cloning. Understating
    is merely stale, so it is allowed a little room rather than forcing a README edit with
    every test added -- but not unbounded room, which is how 840 survived to 865.

    Counted from the session rather than from `def test_`, because parametrisation is most of
    the difference (736 functions, 865 cases), and skipped when the whole suite was not
    collected so that running this file alone does not fail on a subset.
    """
    collected = request.session.items
    files_run = {item.path for item in collected}
    files_on_disk = set(Path(__file__).parent.glob("test_*.py"))
    if files_run < files_on_disk:
        pytest.skip("only part of the suite was collected; the total would be a subset")

    readme = (_ROOT / "README.md").read_text(encoding="utf-8")
    match = re.search(r"\*\*([\d,]+) tests\*\*", readme)
    assert match, "the README no longer states a test count"
    claimed = int(match.group(1).replace(",", ""))
    actual = len(collected)

    assert claimed <= actual, (
        f"the README claims {claimed} tests and the suite collects {actual} -- it is not true"
    )
    assert claimed >= actual * 0.95, (
        f"the README claims {claimed} tests and the suite collects {actual}; the number has "
        f"drifted far enough to be worth updating"
    )
