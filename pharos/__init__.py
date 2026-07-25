"""Pharos — a transparent, context-aware orchestration proxy and live terminal dashboard
for local LLM backends.

The proxy tier ("Observe") is observe-only and a pure passthrough: it never mutates a
request. v0.2 adds "Warn" — a standalone pre-flight check (`pharos check`) that runs BEFORE
a prompt reaches a coding agent, plus passive calibration of client overhead from observed
traffic. The request path itself is unchanged.
"""

__version__ = "0.2.0"
