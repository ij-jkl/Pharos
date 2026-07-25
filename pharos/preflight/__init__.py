"""Pre-flight analyzer (v0.2 "Warn"): check an intended prompt against the context budget
BEFORE pasting it into a coding agent — the only point where an overflow can be prevented
rather than narrated. Standalone; nothing in the proxy request path imports this package."""
