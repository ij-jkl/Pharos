"""Pharos — a transparent, context-aware orchestration proxy and live terminal dashboard
for local LLM backends.

The proxy tier ("Observe") is observe-only and a pure passthrough: it never mutates a
request. v0.2 adds "Warn" — a standalone pre-flight check (`pharos check`) that runs BEFORE
a prompt reaches a coding agent, plus passive calibration of client overhead from observed
traffic. v0.3 adds "Divide" — `pharos split`, which cuts a prompt the check rejects into
parts that each fit. Both are advisory and run outside the request path, which is unchanged.
"""

__version__ = "0.3.0"
