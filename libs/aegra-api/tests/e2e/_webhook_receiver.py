"""In-process HTTP webhook receiver for E2E tests.

Starts a lightweight asyncio TCP server that captures incoming webhook POST
requests so E2E tests can assert on what Aegra actually sent.

Usage::

    async with WebhookReceiver() as receiver:
        # receiver.url  →  "http://host.docker.internal:<port>/webhook"
        await client.runs.create(..., webhook=receiver.url)
        call = await receiver.wait_for_call(timeout=30)
        assert call["body"]["status"] == "success"

The receiver host is configurable via ``WEBHOOK_RECEIVER_HOST``.  On macOS
Docker Desktop the default ``host.docker.internal`` works out of the box.  On
Linux Docker you need to add ``extra_hosts: ["host.docker.internal:host-gateway"]``
to the container (already done in docker-compose.yml).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager


class WebhookReceiver:
    """Minimal async HTTP server that captures a single webhook POST body.

    Attributes:
        url: Public URL that Aegra (inside Docker) can reach to send the hook.
        port: Local TCP port the server is bound to.
    """

    def __init__(self) -> None:
        self._queue: asyncio.Queue[dict] = asyncio.Queue()
        self._server: asyncio.Server | None = None
        self.port: int = 0
        self.url: str = ""

    async def start(self) -> None:
        """Bind to a random free port and start accepting connections."""
        self._server = await asyncio.start_server(
            self._handle,
            host="0.0.0.0",
            port=0,  # OS assigns a free port
        )
        self.port = self._server.sockets[0].getsockname()[1]
        receiver_host = os.environ.get("WEBHOOK_RECEIVER_HOST", "host.docker.internal")
        self.url = f"http://{receiver_host}:{self.port}/webhook"
        await self._server.start_serving()

    async def stop(self) -> None:
        """Shutdown the server and wait for active connections to close."""
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()

    async def wait_for_call(self, timeout: float = 30.0) -> dict:
        """Block until a webhook POST arrives or *timeout* seconds elapse.

        Returns:
            A dict with keys:
            - ``body``: parsed JSON payload (dict)
            - ``headers``: dict of request headers (lowercase keys)
            - ``raw``: raw request body bytes

        Raises:
            TimeoutError: if no call arrives within *timeout* seconds.
        """
        try:
            return await asyncio.wait_for(self._queue.get(), timeout=timeout)
        except TimeoutError:
            raise TimeoutError(f"No webhook received within {timeout}s on port {self.port}")

    def received_count(self) -> int:
        """Return the number of webhook calls currently queued (not yet consumed)."""
        return self._queue.qsize()

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        """Parse a minimal HTTP/1.x request and enqueue the body."""
        try:
            # Read headers line-by-line until blank line
            headers: dict[str, str] = {}
            content_length = 0

            while True:
                line = await reader.readline()
                decoded = line.decode("utf-8", errors="replace").strip()
                if not decoded:
                    break
                if ":" in decoded and not decoded.startswith(("GET", "POST", "PUT")):
                    key, _, value = decoded.partition(":")
                    headers[key.strip().lower()] = value.strip()
                    if key.strip().lower() == "content-length":
                        content_length = int(value.strip())

            body_bytes = await reader.readexactly(content_length) if content_length else b""

            try:
                body = json.loads(body_bytes)
            except json.JSONDecodeError:
                body = {}

            await self._queue.put({"body": body, "headers": headers, "raw": body_bytes})

            # Send minimal HTTP 200 OK
            writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
            await writer.drain()
        except Exception:
            pass
        finally:
            writer.close()
            with contextlib.suppress(Exception):
                await writer.wait_closed()

    async def __aenter__(self) -> WebhookReceiver:
        await self.start()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.stop()


@asynccontextmanager
async def webhook_receiver() -> AsyncGenerator[WebhookReceiver, None]:
    """Async context manager that starts a ``WebhookReceiver`` and tears it down."""
    receiver = WebhookReceiver()
    await receiver.start()
    try:
        yield receiver
    finally:
        await receiver.stop()
