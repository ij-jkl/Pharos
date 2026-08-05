"""Countable-text reading: what a file contributes to a prompt, versus what is on disk."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pharos.preflight.content import BinaryFile, read_countable


def test_plain_text_is_itself_and_carries_no_note(tmp_path: Path) -> None:
    path = tmp_path / "a.py"
    # Written as bytes: the reader must NOT normalise newlines, because the tokenizer counts
    # the bytes that reach the model, and CRLF is not free.
    path.write_bytes(b"x = 1\r\n")
    result = read_countable(path)
    assert result.text == "x = 1\r\n"
    assert result.note is None  # no note means "read as written", and must stay that way


def test_notebook_is_its_cell_sources_without_outputs(tmp_path: Path) -> None:
    path = tmp_path / "n.ipynb"
    path.write_text(
        json.dumps({
            "cells": [
                {"cell_type": "markdown", "source": ["# Title\n"]},
                {
                    "cell_type": "code",
                    "source": ["import os\n", "print(os.getcwd())\n"],
                    "outputs": [{"data": {"image/png": "A" * 50000}}],
                    "execution_count": 3,
                },
            ],
            "metadata": {"kernelspec": {"name": "python3"}},
        }),
        encoding="utf-8",
    )
    result = read_countable(path)
    assert result.text == "# Title\n\nimport os\nprint(os.getcwd())\n"
    assert result.note == "notebook — 2 cells, outputs excluded"
    assert "A" * 100 not in result.text  # the base64 blob is the whole point of not counting it


def test_notebook_accepts_a_string_source(tmp_path: Path) -> None:
    path = tmp_path / "n.ipynb"
    path.write_text(json.dumps({"cells": [{"source": "x = 1\n"}]}), encoding="utf-8")
    assert read_countable(path).text == "x = 1\n"


@pytest.mark.parametrize(
    "payload",
    ["{not json", json.dumps({"nope": []}), json.dumps({"cells": "not a list"})],
)
def test_unparseable_notebook_falls_back_and_admits_it(tmp_path: Path, payload: str) -> None:
    """Better a labeled over-count than a confident number from a file we did not understand."""
    path = tmp_path / "n.ipynb"
    path.write_text(payload, encoding="utf-8")
    result = read_countable(path)
    assert result.text == payload
    assert result.note is not None and "over-count" in result.note


def test_binary_raises_rather_than_being_given_a_count(tmp_path: Path) -> None:
    path = tmp_path / "logo.png"
    path.write_bytes(b"\x89PNG\r\n\x1a\n\x00\xff\xfe")
    with pytest.raises(BinaryFile):
        read_countable(path)


def test_missing_file_raises_oserror_for_the_caller_to_report(tmp_path: Path) -> None:
    with pytest.raises(OSError):
        read_countable(tmp_path / "absent.py")
