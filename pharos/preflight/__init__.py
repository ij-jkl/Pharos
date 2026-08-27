"""Pre-flight analyzer: check an intended prompt against the context budget BEFORE pasting it
into a coding agent — the only point where an overflow can be prevented rather than narrated —
and cut one that does not fit into parts that do. Standalone; nothing in the proxy request
path imports this package."""
