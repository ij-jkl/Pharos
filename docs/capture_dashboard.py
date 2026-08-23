"""Capture the dashboard for the README, with real traffic through the real proxy."""
from __future__ import annotations

import asyncio
import contextlib
import sys
from pathlib import Path

import httpx
import uvicorn

from pharos.config import load_config
from pharos.events import EventBus
from pharos.log import configure_logging
from pharos.proxy.app import create_app
from pharos.tui.app import PharosApp


class _Server(uvicorn.Server):
    def install_signal_handlers(self) -> None:
        return

    @contextlib.contextmanager
    def capture_signals(self):
        yield


PROMPTS = [
    "Reply with the single word: ok",
    "In one short sentence, what does a KV cache store?" + (" context line for sizing." * 40),
    "Name three Python built-ins. One line." + (" more context to widen the request." * 120),
    "Summarise this in one clause:" + (" a paragraph of filler to vary token counts." * 260),
    "Say done.",
]


async def main() -> None:
    config = load_config()
    configure_logging(config.log_file)
    bus = EventBus()
    server = _Server(
        uvicorn.Config(create_app(config, bus), host=config.proxy_host,
                       port=config.proxy_port, log_config=None, access_log=False)
    )
    task = asyncio.create_task(server.serve())
    deadline = asyncio.get_running_loop().time() + 10.0
    while not server.started and asyncio.get_running_loop().time() < deadline:  # noqa: ASYNC110
        await asyncio.sleep(0.05)
    if not server.started:
        raise SystemExit(f"proxy failed to start on {config.proxy_host}:{config.proxy_port}")

    app = PharosApp(config=config, bus=bus)
    out = Path("docs/pharos-dashboard.svg")
    async with app.run_test(size=(104, 30)) as pilot:
        await pilot.pause()
        await asyncio.sleep(2.5)  # let the first profile probe land
        base = f"http://{config.proxy_host}:{config.proxy_port}"
        async with httpx.AsyncClient(timeout=180.0) as client:
            for i, prompt in enumerate(PROMPTS, 1):
                try:
                    await client.post(
                        f"{base}/api/chat",
                        json={"model": config.model, "stream": False,
                              "messages": [{"role": "user", "content": prompt}],
                              "options": {"num_predict": 24}},
                    )
                    print(f"  request {i}/{len(PROMPTS)} ok", file=sys.stderr)
                except Exception as exc:  # noqa: BLE001
                    print(f"  request {i} failed: {exc}", file=sys.stderr)
                await pilot.pause()
                await asyncio.sleep(0.4)
        await asyncio.sleep(1.5)
        await pilot.pause()
        app.save_screenshot(str(out))
    server.should_exit = True
    with contextlib.suppress(Exception, asyncio.TimeoutError):
        await asyncio.wait_for(task, timeout=5.0)
    print(f"wrote {out}", file=sys.stderr)


asyncio.run(main())
