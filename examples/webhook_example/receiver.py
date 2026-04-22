#!/usr/bin/env python3
"""Standalone webhook receiver server.

Starts a simple HTTP server that listens for incoming webhook POST requests
from Aegra and prints the payload to stdout.

Usage::

    # In one terminal — start the receiver on port 8765
    python examples/webhook_example/receiver.py

    # In another terminal — run a demo that sends a run with a webhook
    python examples/webhook_example/demo.py

By default the receiver listens on 0.0.0.0:8765.  Set PORT to override::

    PORT=9000 python examples/webhook_example/receiver.py

When running Aegra in Docker, Aegra reaches this server via
``host.docker.internal``.  The demo.py script handles this automatically.
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


def _verify_signature(secret: str, body: bytes, header: str) -> bool:
    """Return True if the HMAC-SHA256 signature in *header* is valid."""
    expected = "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, header)


async def _handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    """Parse an HTTP POST and pretty-print the webhook payload."""
    try:
        request_line = await reader.readline()
        method = request_line.decode().split()[0] if request_line else "UNKNOWN"

        headers: dict[str, str] = {}
        content_length = 0
        while True:
            line = await reader.readline()
            decoded = line.decode("utf-8", errors="replace").strip()
            if not decoded:
                break
            if ":" in decoded and not decoded.startswith(("GET", "POST", "PUT")):
                key, _, value = decoded.partition(":")
                key_lower = key.strip().lower()
                headers[key_lower] = value.strip()
                if key_lower == "content-length":
                    content_length = int(value.strip())

        body_bytes = await reader.readexactly(content_length) if content_length else b""

        received_at = datetime.now(UTC).isoformat()
        print(f"\n{'=' * 60}", flush=True)
        print(f"[{received_at}] Received {method} webhook", flush=True)
        print(f"  Content-Type : {headers.get('content-type', '(none)')}", flush=True)
        print(f"  User-Agent   : {headers.get('user-agent', '(none)')}", flush=True)

        signature = headers.get("x-aegra-signature", "")
        secret = os.environ.get("WEBHOOK_SECRET", "")
        if signature:
            if secret:
                valid = _verify_signature(secret, body_bytes, signature)
                status = "✓ valid" if valid else "✗ INVALID"
                print(f"  Signature    : {signature} ({status})", flush=True)
            else:
                print(f"  Signature    : {signature} (WEBHOOK_SECRET not set — not verified)", flush=True)

        try:
            payload = json.loads(body_bytes)
            print(f"\n  Payload:\n{json.dumps(payload, indent=4, default=str)}", flush=True)
        except json.JSONDecodeError:
            print(f"  Raw body: {body_bytes!r}", flush=True)

        print(f"{'=' * 60}", flush=True)

        writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
        await writer.drain()
    except Exception as exc:
        print(f"[receiver] Error handling connection: {exc}", file=sys.stderr, flush=True)
    finally:
        writer.close()
        with contextlib.suppress(Exception):
            await writer.wait_closed()


async def main() -> None:
    port = int(os.environ.get("PORT", "8765"))
    server = await asyncio.start_server(_handle, host="0.0.0.0", port=port)  # nosec B104
    bound_port = server.sockets[0].getsockname()[1]

    print(f"Webhook receiver listening on http://0.0.0.0:{bound_port}/webhook")
    print("Waiting for incoming webhooks... (Ctrl+C to stop)\n")

    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nStopped.")
