"""Pre-flight analyzer: check an intended prompt against the context budget BEFORE pasting it
into a coding agent — the only point where an overflow can be prevented rather than narrated
(v0.2 "Warn") — and cut one that does not fit into parts that do (v0.3 "Divide"). Standalone;
nothing in the proxy request path imports this package."""
