"""Pharos — a transparent, context-aware orchestration proxy and live terminal dashboard
for local LLM backends.

The proxy is observe-only and a pure passthrough: it never mutates a request or a response.
Around it sit three tools that run OUTSIDE the request path — `pharos check` pre-flights a
prompt against the real context budget before it reaches a coding agent, `pharos split` cuts
one the check rejects into parts that each fit, and `pharos run` carries those parts out
against the local model, one conversation each.

`pharos run` is the only part that WRITES, and it writes as an ordinary client of the proxy
rather than through it: the passthrough guarantee is about other people's traffic. It refuses
rather than truncates, enforces a part's scope in the tool layer rather than the prompt, and
will not start without an undo.
"""

__version__ = "1.1.9"
