"""Pharos — a transparent, context-aware orchestration proxy and live terminal dashboard
for local LLM backends.

The proxy tier ("Observe") is observe-only and a pure passthrough: it never mutates a
request. v0.2 adds "Warn" — a standalone pre-flight check (`pharos check`) that runs BEFORE
a prompt reaches a coding agent, plus passive calibration of client overhead from observed
traffic. v0.3 adds "Divide" — `pharos split`, which cuts a prompt the check rejects into
parts that each fit. Both are advisory and run outside the request path, which is unchanged.

v0.4 adds "Do" — `pharos run`, which carries those parts out against the local model, one
conversation each. It is the first tier that WRITES, and it writes as an ordinary client of
the proxy rather than through it: the passthrough guarantee is about other people's traffic
and has not moved. It refuses rather than truncates, enforces a part's scope in the tool
layer rather than the prompt, and will not start without an undo.
"""

__version__ = "0.4.2"
