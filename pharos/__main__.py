"""Entry point for `pharos` / `python -m pharos`.

Runs the Textual TUI in the foreground with the proxy as an asyncio task in the same
process. The two share nothing but the event bus: the proxy publishes, the TUI consumes.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import sys
from collections.abc import Iterator

import uvicorn

from pharos import __version__
from pharos.config import ConfigError, PharosConfig, load_config
from pharos.console import force_utf8
from pharos.events import EventBus
from pharos.log import configure_logging
from pharos.proxy.app import create_app
from pharos.tui.app import PharosApp

_SERVER_START_TIMEOUT_S = 5.0

_logger = logging.getLogger("pharos")


class _EmbeddedServer(uvicorn.Server):
    """Uvicorn embedded next to a Textual app: the TUI owns the terminal and the signals."""

    def install_signal_handlers(self) -> None:  # older uvicorn hook
        return

    @contextlib.contextmanager
    def capture_signals(self) -> Iterator[None]:  # newer uvicorn hook
        yield


async def _run(config: PharosConfig) -> None:
    # All logging (pharos, uvicorn, httpx) goes to the rotating file; the TUI owns stdout.
    log_path = configure_logging(config.log_file)
    _logger.info(
        "starting: proxy %s:%d -> backend %s (log: %s)",
        config.proxy_host,
        config.proxy_port,
        config.backend_url,
        log_path,
    )
    bus = EventBus()
    proxy_app = create_app(config, bus)
    server = _EmbeddedServer(
        uvicorn.Config(
            proxy_app,
            host=config.proxy_host,
            port=config.proxy_port,
            log_config=None,
            access_log=False,
        )
    )
    server_task = asyncio.create_task(server.serve())

    # A proxy that failed to bind must be loud, not a silently dead dashboard.
    deadline = asyncio.get_running_loop().time() + _SERVER_START_TIMEOUT_S
    while not server.started and not server_task.done():
        if asyncio.get_running_loop().time() > deadline:
            break
        await asyncio.sleep(0.02)
    if server_task.done() or not server.started:
        exc = server_task.exception() if server_task.done() else None
        detail = f" ({exc})" if exc else ""
        _logger.error(
            "proxy failed to start on %s:%d%s", config.proxy_host, config.proxy_port, detail
        )
        raise SystemExit(
            f"pharos: proxy failed to start on "
            f"{config.proxy_host}:{config.proxy_port}{detail} — is the port in use?"
        )
    _logger.info("proxy listening on %s:%d", config.proxy_host, config.proxy_port)

    tui = PharosApp(config=config, bus=bus)
    try:
        await tui.run_async()
    finally:
        server.should_exit = True
        with contextlib.suppress(Exception, asyncio.TimeoutError):
            await asyncio.wait_for(server_task, timeout=5.0)
        _logger.info("shutdown complete")


def main() -> None:
    """Console-script entry point: bare `pharos` runs the TUI; `pharos check` pre-flights.

    Subcommand dispatch happens before any TUI import cost is paid, and leaves the bare
    invocation untouched: pointing a coding agent at the proxy is unaffected by any of it.
    """
    # The unknown-argument message below carries an em dash, so even the error path needs a
    # stream that can encode it.
    force_utf8(sys.stdout, sys.stderr)

    argv = sys.argv[1:]
    # Before the subcommands, because it is what a bug report opens with. The number already
    # existed and CI already asserts the wheel prints it; nothing but the CLI could say it.
    if argv and argv[0] in ("--version", "-V"):
        print(f"pharos {__version__}")
        return
    if argv and argv[0] == "check":
        from pharos.preflight.cli import main as check_main

        raise SystemExit(check_main(argv[1:]))
    if argv and argv[0] == "split":
        from pharos.preflight.cli import split_main

        raise SystemExit(split_main(argv[1:]))
    if argv and argv[0] == "run":
        from pharos.agent.cli import main as run_main

        raise SystemExit(run_main(argv[1:]))
    if argv:
        raise SystemExit(
            f"pharos: unknown arguments {argv!r} — run `pharos` for the dashboard, "
            f"`pharos check --help` for the pre-flight analyzer, `pharos split --help` "
            f"to cut an oversized prompt into parts that fit, `pharos run --help` to "
            f"carry the task out, or `pharos --version`"
        )
    try:
        config = load_config()
    except ConfigError as exc:
        raise SystemExit(f"pharos: {exc}") from exc
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(_run(config))


if __name__ == "__main__":
    main()
