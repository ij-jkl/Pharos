"""Every legal state a report can be in still renders.

The renderers were the least-covered code in the project — `pharos run`'s report at 36%, the
profiler panel at 20% — and they are also the last thing that runs, after a job that may have
taken twenty minutes. A `TypeError` formatting `None` there loses the whole measurement, and
it loses it at the one moment the user has nothing else to look at.

The risk is specific and it is not logic: these functions read dataclasses full of `int | None`
fields and possibly-empty lists, and a format spec like `f"{x:,}"` raises on None. `mypy` does
not check format specs, so nothing in CI was watching. The exposure is not "is the arithmetic
right" — that is measured everywhere else — but "can this combination of legal values be
printed at all".

So these tests do not assert on wording. They build states out of the real types, drive the
real renderers, and assert only that nothing raises. Wording is asserted where it means
something, in the tests that own each number; duplicating it here would make this file break
every time a sentence is reworded, which is how a suite stops being run.

Every state is REACHABLE by construction: the budgets come from the real `Accountant` rather
than being written by hand, so a combination that the accountant cannot produce is not tested
and a combination it can produce cannot be missed.
"""

from __future__ import annotations

import contextlib
import io
import itertools

import pytest
from rich.console import Console

from pharos.accountant import Accountant
from pharos.agent.audit import PartAudit, TreeDiff
from pharos.agent.cli import _render
from pharos.agent.review import Finding, Review
from pharos.agent.runner import RunOutcome
from pharos.agent.session import PartResult
from pharos.agent.verify import CheckOutcome, Damage, Verification
from pharos.config import PharosConfig
from pharos.profiler.cli import _render as _render_profile
from pharos.profiler.types import BackendInfo, EnvironmentProfile, GpuInfo


def _console() -> Console:
    return Console(file=io.StringIO(), width=100, force_terminal=False)


# --- the environment profile ---------------------------------------------------------------------

_GPUS = {
    "none detected": GpuInfo(available=False, detail="no NVIDIA GPU detected"),
    "present": GpuInfo(
        available=True, name="RTX 3060", total_mib=12288, used_mib=9000,
        free_mib=3288, source="nvml",
    ),
    "nothing free": GpuInfo(
        available=True, name="RTX 3060", total_mib=12288, used_mib=12288,
        free_mib=0, source="nvml",
    ),
    # NVML answered, and answered with nothing. Every figure on the VRAM row is None at once.
    "present but blank": GpuInfo(available=True, source="nvml"),
}

_BACKENDS = {
    "unreachable": BackendInfo(reachable=False, base_url="http://x", detail="unreachable"),
    "no model resident": BackendInfo(reachable=True, base_url="http://x"),
    "fully described": BackendInfo(
        reachable=True, base_url="http://x", model="m:1", architecture="qwen35",
        quantization="Q8_0", parameter_size="9B", advertised_max_ctx=262144, loaded_ctx=32768,
    ),
    "loaded, nothing advertised": BackendInfo(
        reachable=True, base_url="http://x", model="m:1", loaded_ctx=8192
    ),
    "advertised, nothing loaded": BackendInfo(
        reachable=True, base_url="http://x", model="m:1", advertised_max_ctx=262144
    ),
    # A zero window is the shape that makes every derived figure zero or undefined at once.
    "zero window": BackendInfo(
        reachable=True, base_url="http://x", model="m:1",
        advertised_max_ctx=262144, loaded_ctx=0,
    ),
}


@pytest.mark.parametrize("gpu_name", sorted(_GPUS))
@pytest.mark.parametrize("backend_name", sorted(_BACKENDS))
@pytest.mark.parametrize("kv_bytes", [None, 147_456])
def test_the_environment_panel_renders_in_every_state(
    gpu_name: str, backend_name: str, kv_bytes: int | None
) -> None:
    """No GPU, no backend, no model, no metadata — the panel still prints, N/A and all.

    A profiler that only runs on a fully configured machine cannot be used to find out why a
    machine is not configured, which is the one job it has.
    """
    gpu, backend = _GPUS[gpu_name], _BACKENDS[backend_name]
    budget = Accountant(PharosConfig()).report(
        loaded_ctx=backend.loaded_ctx, gpu=gpu, kv_bytes_per_token=kv_bytes
    )
    advertised, loaded = backend.advertised_max_ctx, backend.loaded_ctx
    mismatch = bool(advertised and loaded and loaded < advertised)
    profile = EnvironmentProfile(
        gpu=gpu,
        backend=backend,
        budget=budget,
        ctx_mismatch=mismatch,
        ctx_mismatch_ratio=(loaded / advertised) if mismatch and advertised else None,
    )

    _render_profile(_console(), profile)


# --- the run report ------------------------------------------------------------------------------


def _part(**overrides: object) -> PartResult:
    base: dict[str, object] = {
        "text": "done, changed a.py",
        "steps": 1,
        "files_written": ["a.py"],
        "peak_tokens": 500,
        "reported_tokens": 480,
        "stopped_early": False,
        "ceiling": 1000,
        "scoped": ["a.py"],
        "drift_samples": [(500, 520)],
    }
    base.update(overrides)
    return PartResult(**base)  # type: ignore[arg-type]


_PARTS = {
    "none at all": [],
    "one clean": [_part()],
    # A failed part reports no ceiling fraction and no drift, so every derived figure is absent.
    "one failed": [_part(error="the backend went away", files_written=[], drift_samples=[])],
    "wrote nothing": [_part(files_written=[], text="")],
    # The part whose body would not fit: nothing ran, so ceiling is 0 and peak_fraction has no
    # denominator. This is the state that would divide by zero if anything did.
    "never started": [_part(ceiling=0, peak_tokens=0, drift_samples=[], stopped_early=True)],
    "hit the ceiling": [
        _part(stopped_early=True, nudges=2, room_refusals=3, compacted_tokens=900, compactions=1)
    ],
    "nothing measured": [_part(drift_samples=[], reported_tokens=None)],
    "context dropped": [_part(truncated=True, exposed_requests=2)],
}

_VERIFICATIONS = {
    "absent": None,
    "ran nothing": Verification(),
    "newly broken": Verification(
        checks=(CheckOutcome(name="ruff check .", ok=False, detail="E501"),)
    ),
    "already red": Verification(
        checks=(CheckOutcome(name="ruff check .", ok=False, detail="x", pre_existing=True),)
    ),
    "skipped": Verification(
        checks=(CheckOutcome(name="pytest -q", ok=True, skipped="not on PATH"),)
    ),
    "no baseline": Verification(checks=(CheckOutcome(name="mypy", ok=False, unattributable=True),)),
}

_DAMAGE = {
    "none": [],
    "outstanding": [Damage(label="part 1", path="a.py", error="SyntaxError: bad")],
    "repaired later": [Damage(label="part 1", path="a.py", error="x", repaired_by="part 2")],
    # Damage with no path at all: a project check broke rather than a file.
    "a check broke": [Damage(label="part 1", path="", error="failed", check="ruff check .")],
}

_REVIEWS = {
    "absent": None,
    "nothing to read": Review(note="the run changed no files"),
    "with findings": Review(
        findings=[Finding(file="a.py", line=3, severity="bug", note="off by one")],
        reviewed=["a.py"],
        discarded=2,
    ),
}

_AUDITS = {
    "none": [],
    "clean": [
        PartAudit(
            part="part 1", changed=TreeDiff(modified=("a.py",)), claimed=("a.py",),
            unattributed=(), absent=(), out_of_scope=(),
        )
    ],
    # Every audit finding at once, including a truncated index.
    "every finding": [
        PartAudit(
            part="part 1", changed=TreeDiff(modified=("b.py",)), claimed=("a.py",),
            unattributed=("b.py",), absent=("a.py",), out_of_scope=("b.py",),
            by_checks=TreeDiff(modified=("c.py",)), truncated=True,
        )
    ],
}


@pytest.mark.parametrize("parts_name", sorted(_PARTS))
@pytest.mark.parametrize("as_json", [False, True])
def test_the_run_report_renders_in_every_state(parts_name: str, as_json: bool) -> None:
    """The full cross-product of verification, damage, review and audit, for each shape of run.

    Driven rather than asserted: what each number MEANS is pinned in test_scorecard,
    test_verify, test_audit and test_review. What is pinned here is that the report survives
    being asked to print them together, in whatever combination a real run produced.
    """
    for verification, damage, review, audits in itertools.product(
        _VERIFICATIONS.values(), _DAMAGE.values(), _REVIEWS.values(), _AUDITS.values()
    ):
        outcome = RunOutcome(
            parts=list(_PARTS[parts_name]),
            verification=verification,
            damage=list(damage),
            review=review,
            audits=list(audits),
            planned_files=["a.py", "b.py"],
            handoff_reserve=500,
            files_changed=["a.py"],
            branch="pharos-run/2026-01-01-000000",
            base_branch="main",
        )
        # stdout carries the --json payload; the human half goes to the console above it.
        with contextlib.redirect_stdout(io.StringIO()):
            _render(_console(), outcome, dry_run=False, as_json=as_json, root="/w")


@pytest.mark.parametrize("parts_name", sorted(_PARTS))
def test_a_dry_run_renders_in_every_state(parts_name: str) -> None:
    """--dry-run prints the plan and returns before the scorecard, which is its own path."""
    outcome = RunOutcome(
        parts=list(_PARTS[parts_name]), planned_files=["a.py"], handoff_reserve=500
    )
    for as_json in (False, True):
        with contextlib.redirect_stdout(io.StringIO()):
            assert _render(_console(), outcome, dry_run=True, as_json=as_json, root="/w") == 0


def test_a_run_with_no_plan_and_no_parts_still_renders() -> None:
    """The emptiest legal outcome there is: nothing ran, nothing was planned, nothing changed."""
    with contextlib.redirect_stdout(io.StringIO()):
        _render(_console(), RunOutcome(), dry_run=False, as_json=False, root="/w")
        _render(_console(), RunOutcome(), dry_run=False, as_json=True, root="/w")
