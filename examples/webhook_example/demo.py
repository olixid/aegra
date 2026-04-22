#!/usr/bin/env python3
"""Demo script: create a run with a webhook and receive the callback.

Demonstrates the full webhook flow end-to-end using the ``webhook_echo`` graph
(which requires no LLM — it runs instantly and always succeeds).

Prerequisites:
  1. Aegra server running: ``uv run aegra dev``  or  ``docker compose up``
  2. This script runs on the **host machine** (outside Docker).

The script:
  1. Starts an in-process webhook receiver on a random free port.
  2. Creates a thread and a background run with ``webhook=<receiver URL>``.
  3. Blocks until the webhook arrives (or times out after 30 s).
  4. Pretty-prints the received payload and verifies the HMAC signature
     if ``WEBHOOK_SECRET`` is set.

Usage::

    # Default: server at http://localhost:2026
    python examples/webhook_example/demo.py

    # Custom server URL
    SERVER_URL=http://myserver:8080 python examples/webhook_example/demo.py

    # With HMAC verification
    WEBHOOK_SECRET=mysecret python examples/webhook_example/demo.py

Environment variables:
  SERVER_URL           Aegra server base URL (default: http://localhost:2026)
  WEBHOOK_SECRET       Optional — if set, the script verifies the X-Aegra-Signature
  WEBHOOK_HOST         Host Aegra uses to call back (default: host.docker.internal
                       on Docker; use 127.0.0.1 for aegra dev)
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import hmac
import json
import os
import sys
from datetime import UTC, datetime

try:
    from langgraph_sdk import get_client
except ImportError:
    print("ERROR: langgraph-sdk is not installed. Run: pip install langgraph-sdk", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SERVER_URL = os.environ.get("SERVER_URL", "http://localhost:2026")
WEBHOOK_SECRET = os.environ.get("WEBHOOK_SECRET", "")
GRAPH_ID = "webhook_echo"
TIMEOUT = 30.0


def _get_webhook_host() -> str:
    """Determine the webhook host based on how Aegra is running."""
    # If explicitly set, use that
    if host := os.environ.get("WEBHOOK_HOST"):
        return host
    # When using 'aegra dev' (local process), Aegra reaches the host directly
    # When using Docker, it must go through host.docker.internal
    return "host.docker.internal"


# ---------------------------------------------------------------------------
# Minimal in-process webhook receiver
# ---------------------------------------------------------------------------


class _Receiver:
    """Minimal asyncio HTTP receiver that captures one POST body."""

    def __init__(self) -> None:
        self._server: asyncio.Server | None = None
        self._queue: asyncio.Queue[dict] = asyncio.Queue()
        self.port = 0
        self.url = ""

    async def start(self) -> None:
        self._server = await asyncio.start_server(self._handle, "0.0.0.0", 0)  # nosec B104
        self.port = self._server.sockets[0].getsockname()[1]
        host = _get_webhook_host()
        self.url = f"http://{host}:{self.port}/webhook"
        await self._server.start_serving()
        print(f"[receiver] Listening on http://0.0.0.0:{self.port}")
        print(f"[receiver] Aegra will call back at: {self.url}")

    async def stop(self) -> None:
        if self._server:
            self._server.close()
            await self._server.wait_closed()

    async def wait(self, timeout: float = TIMEOUT) -> dict:
        try:
            return await asyncio.wait_for(self._queue.get(), timeout=timeout)
        except TimeoutError:
            raise TimeoutError(f"No webhook received within {timeout}s")

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            await reader.readline()  # request line
            headers: dict[str, str] = {}
            content_length = 0
            while True:
                line = await reader.readline()
                decoded = line.decode("utf-8", errors="replace").strip()
                if not decoded:
                    break
                if ":" in decoded and not decoded.startswith(("GET", "POST", "PUT")):
                    k, _, v = decoded.partition(":")
                    headers[k.strip().lower()] = v.strip()
                    if k.strip().lower() == "content-length":
                        content_length = int(v.strip())
            body = await reader.readexactly(content_length) if content_length else b""
            await self._queue.put({"headers": headers, "raw": body, "body": json.loads(body) if body else {}})
            writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
            with contextlib.suppress(Exception):
                await writer.drain()
        except Exception as exc:
            print(f"[receiver] Connection error: {exc}", file=sys.stderr)
        finally:
            writer.close()


# ---------------------------------------------------------------------------
# HMAC verification
# ---------------------------------------------------------------------------


def _verify_signature(secret: str, body: bytes, sig_header: str) -> bool:
    expected = "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, sig_header)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    client = get_client(url=SERVER_URL)
    receiver = _Receiver()
    await receiver.start()

    print(f"\n[demo] Connecting to Aegra at {SERVER_URL}")
    print(f"[demo] Using graph: {GRAPH_ID}\n")

    # Create assistant (idempotent)
    assistant = await client.assistants.create(
        graph_id=GRAPH_ID,
        config={"tags": ["webhook-demo"]},
        if_exists="do_nothing",
    )
    assistant_id = assistant["assistant_id"]
    print(f"[demo] Assistant: {assistant_id}")

    # Create thread and run
    thread = await client.threads.create()
    print(f"[demo] Thread: {thread['thread_id']}")

    run = await client.runs.create(
        thread_id=thread["thread_id"],
        assistant_id=assistant_id,
        input={"messages": [{"role": "user", "content": "hello from the demo!"}]},
        webhook=receiver.url,
    )
    print(f"[demo] Run started: {run['run_id']}")
    print(f"[demo] Waiting for webhook (timeout={TIMEOUT}s)...\n")

    try:
        call = await receiver.wait(timeout=TIMEOUT)
    finally:
        await receiver.stop()

    body = call["body"]
    now = datetime.now(UTC).isoformat()

    print(f"\n{'=' * 60}")
    print(f"[{now}] Webhook received!")
    print(f"  run_id     : {body.get('run_id')}")
    print(f"  thread_id  : {body.get('thread_id')}")
    print(f"  graph_id   : {body.get('graph_id')}")
    print(f"  status     : {body.get('status')}")
    print(f"  completed_at: {body.get('completed_at')}")
    if body.get("error"):
        print(f"  error      : {body['error']}")
    print(f"\n  Full payload:\n{json.dumps(body, indent=4, default=str)}")

    # HMAC verification
    sig = call["headers"].get("x-aegra-signature", "")
    if WEBHOOK_SECRET and sig:
        valid = _verify_signature(WEBHOOK_SECRET, call["raw"], sig)
        result = "✓ valid" if valid else "✗ INVALID"
        print(f"\n  Signature ({result}): {sig}")
    elif sig:
        print(f"\n  Signature (not verified — WEBHOOK_SECRET not set): {sig}")

    print(f"{'=' * 60}\n")

    assert body.get("run_id") == run["run_id"], "run_id mismatch"
    assert body.get("status") in {"success", "error", "interrupted"}, f"Unexpected status: {body.get('status')}"
    print("[demo] ✓ Webhook received and validated successfully!")


if __name__ == "__main__":
    asyncio.run(main())
