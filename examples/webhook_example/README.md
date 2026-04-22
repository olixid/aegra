# Webhook Example

A complete, runnable webhook demo that shows how Aegra notifies external services when a run completes — with no LLM dependency.

## What it does

1. `graph.py` — A deterministic **echo graph** that instantly echoes the user's message back. No API keys required. Sends `status=error` if the input contains the special sentinel `__error__`.
2. `receiver.py` — A standalone webhook receiver server that listens for Aegra callbacks and pretty-prints each payload.
3. `demo.py` — An end-to-end demo script that starts an in-process receiver, creates a run with a webhook URL, and validates the callback.

## Quick start

### 1. Start Aegra

```bash
# Local dev (no Docker required)
uv run aegra dev

# Or with Docker
docker compose up
```

### 2. In one terminal — run the demo (handles receiver internally)

```bash
python examples/webhook_example/demo.py
```

Expected output:

```
[receiver] Listening on http://0.0.0.0:54321
[receiver] Aegra will call back at: http://host.docker.internal:54321/webhook

[demo] Connecting to Aegra at http://localhost:2026
[demo] Using graph: webhook_echo

[demo] Assistant: asst-xxxx
[demo] Thread: thread-xxxx
[demo] Run started: run-xxxx
[demo] Waiting for webhook (timeout=30.0s)...

============================================================
[2025-...Z] Webhook received!
  run_id     : run-xxxx
  thread_id  : thread-xxxx
  graph_id   : webhook_echo
  status     : success
  ...

[demo] ✓ Webhook received and validated successfully!
```

### 3. (Alternative) Run receiver separately

**Terminal 1 — receiver:**

```bash
python examples/webhook_example/receiver.py
```

**Terminal 2 — send a run:**

```python
import asyncio
from langgraph_sdk import get_client

async def main():
    client = get_client(url="http://localhost:2026")
    thread = await client.threads.create()
    run = await client.runs.create(
        thread_id=thread["thread_id"],
        assistant_id="webhook_echo",
        input={"messages": [{"role": "user", "content": "hello!"}]},
        webhook="http://localhost:8765/webhook",  # points to receiver
    )
    print(f"Run created: {run['run_id']}")

asyncio.run(main())
```

## Configuration

| Environment variable | Default | Description |
|---|---|---|
| `SERVER_URL` | `http://localhost:2026` | Aegra server URL (demo.py) |
| `WEBHOOK_SECRET` | _(empty)_ | When set, Aegra adds `X-Aegra-Signature` header; demo.py verifies it |
| `WEBHOOK_HOST` | `host.docker.internal` | Hostname Aegra uses to reach the receiver (set to `127.0.0.1` when using `aegra dev`) |
| `PORT` | `8765` | Receiver port (receiver.py only) |

## HMAC signature verification

When `WEBHOOK_SECRET` is set in Aegra's environment (`.env` file), every webhook
request includes an `X-Aegra-Signature: sha256=<hex>` header. Verify it in your
receiver like this:

```python
import hashlib
import hmac

def verify(secret: str, body: bytes, signature: str) -> bool:
    expected = "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature)
```

## Triggering a status=error webhook

Send a message with content `__error__` to make the graph raise a `RuntimeError`:

```python
run = await client.runs.create(
    thread_id=thread["thread_id"],
    assistant_id="webhook_echo",
    input={"messages": [{"role": "user", "content": "__error__"}]},
    webhook="http://localhost:8765/webhook",
)
# The webhook will arrive with status="error" and an error message.
```

## How it works

The `webhook` field is part of the LangGraph SDK's `RunCreate` schema and is
passed transparently by the `langgraph_sdk` client. Aegra picks it up, stores
it alongside the run's execution parameters, and dispatches the callback after
the run reaches any terminal state (success, error, or interrupted).

Webhook delivery is **best-effort**: Aegra logs a warning on failure but never
affects the run's recorded status.

See [docs/guides/webhooks.mdx](../../docs/guides/webhooks.mdx) for the full
documentation including payload format, retry behaviour, and HMAC signing.
