"""`pharos run` — drive a coding model through a task that does not fit in one window.

This is the first part of Pharos that WRITES. Everything before it observes or advises, so
the boundaries are worth stating once, here, rather than rediscovering them per module.

What it does: pre-flight the task exactly as ``pharos check`` does, divide it exactly as
``pharos split`` does, then execute each part as its own agent conversation — fresh context,
seeded with the previous part's hand-off. The division is the v0.3 splitter unchanged: file
packing by scope and position. No model is ever asked what the task means.

Three properties hold by construction, and they are the reason this can exist alongside the
observe-only proxy:

* **The proxy is untouched.** The agent is a CLIENT of the proxy, the same as Continue or
  Cursor. Its traffic is observed on the way past, never rewritten. `pharos run` lives outside
  the request path in exactly the sense ``check`` and ``split`` do.
* **Overhead is exact, not learned.** Pharos wrote this client, so it can count its own system
  prompt and tool catalogue instead of estimating them from observed traffic. The floor a run
  plans against has no calibration guess in it.
* **Nothing is trimmed.** A conversation that would exceed the window is stopped and handed
  off, never silently compacted. History compaction is out of scope here for the same reason
  it is out of scope in the proxy: a number you cannot explain is worse than a refusal.

The scope rule is enforced in the tool layer, not merely requested in the prompt. A part that
is told "open only these three files" cannot open a fourth — see ``pharos.agent.tools``. That
is what turns the splitter's projection from a hope into a bound.
"""

from __future__ import annotations

from pharos.agent.runner import RunOutcome, run_task
from pharos.agent.workspace import GitGuardError, Workspace, WorkspaceError

__all__ = ["GitGuardError", "RunOutcome", "Workspace", "WorkspaceError", "run_task"]
