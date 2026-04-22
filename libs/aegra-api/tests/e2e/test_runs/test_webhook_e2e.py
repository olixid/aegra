"""E2E tests for webhook callbacks on run completion.

These tests start a real in-process HTTP server as the webhook receiver,
then create runs against a live Aegra server.  After each run finishes,
the tests assert that Aegra POSTed the expected payload to the receiver.

Requirements:
  - A running Aegra server (``uv run aegra dev`` or ``docker compose up``)
  - The ``agent`` graph deployed in the server's ``aegra.json``
  - On Linux Docker: ``extra_hosts: ["host.docker.internal:host-gateway"]``
    in the aegra service of docker-compose.yml (already added)
"""

from __future__ import annotations

import hashlib
import hmac
import os

import pytest
from httpx import AsyncClient

from aegra_api.settings import settings
from tests.e2e._utils import elog, get_e2e_client
from tests.e2e._webhook_receiver import WebhookReceiver, webhook_receiver

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SHORT_PROMPT = {"messages": [{"role": "user", "content": "Say exactly: done"}]}
_TIMEOUT = 60.0  # seconds to wait for a webhook call


async def _ensure_assistant(sdk, graph_id: str = "agent") -> str:
    """Create (or reuse) an assistant and return its ID."""
    assistant = await sdk.assistants.create(
        graph_id=graph_id,
        config={"tags": ["webhook-e2e"]},
        if_exists="do_nothing",
    )
    elog("Assistant.create", assistant)
    return assistant["assistant_id"]


def _verify_hmac(secret: str, body: bytes, header_value: str) -> bool:
    """Return True if the HMAC-SHA256 signature matches."""
    expected = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    # header_value is "sha256=<hex>"
    return hmac.compare_digest(f"sha256={expected}", header_value)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_webhook_fires_on_run_success() -> None:
    """Aegra POSTs a webhook after a run completes (any terminal state)."""
    sdk = get_e2e_client()
    assistant_id = await _ensure_assistant(sdk)

    async with webhook_receiver() as receiver:
        elog("WebhookReceiver URL", receiver.url)

        thread = await sdk.threads.create()
        run = await sdk.runs.create(
            thread_id=thread["thread_id"],
            assistant_id=assistant_id,
            input=_SHORT_PROMPT,
            webhook=receiver.url,
        )
        elog("Runs.create", run)

        call = await receiver.wait_for_call(timeout=_TIMEOUT)
        elog("Webhook call received", call["body"])

    body = call["body"]
    # The webhook MUST fire for any terminal state
    assert body["status"] in {"success", "error", "interrupted"}, f"Unexpected webhook status: {body['status']}"
    assert body["run_id"] == run["run_id"]
    assert body["thread_id"] == thread["thread_id"]

    if body["status"] == "error":
        error_msg = body.get("error", "")
        if "api_key" in error_msg.lower() or "unsupported_country" in error_msg.lower():
            pytest.skip(f"Webhook fired correctly but run failed (infra): {error_msg[:80]}")

    # Only assert error=None if run actually succeeded
    assert body["error"] is None


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_webhook_payload_has_required_fields() -> None:
    """Webhook payload mirrors the run response shape: run_id, thread_id, status, etc."""
    sdk = get_e2e_client()
    assistant_id = await _ensure_assistant(sdk)

    async with webhook_receiver() as receiver:
        thread = await sdk.threads.create()
        run = await sdk.runs.create(
            thread_id=thread["thread_id"],
            assistant_id=assistant_id,
            input=_SHORT_PROMPT,
            webhook=receiver.url,
        )

        call = await receiver.wait_for_call(timeout=_TIMEOUT)

    body = call["body"]
    required_fields = {
        "run_id",
        "thread_id",
        "graph_id",
        "status",
        "input",
        "output",
        "error",
        "metadata",
        "completed_at",
    }
    missing = required_fields - set(body.keys())
    assert not missing, f"Webhook payload missing fields: {missing}"

    assert isinstance(body["run_id"], str)
    assert isinstance(body["thread_id"], str)
    assert isinstance(body["graph_id"], str)
    assert isinstance(body["status"], str)
    assert isinstance(body["completed_at"], str)
    assert body["run_id"] == run["run_id"]


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_webhook_hmac_signature_is_valid() -> None:
    """When WEBHOOK_SECRET is set, Aegra signs the payload with HMAC-SHA256."""
    secret = os.environ.get("WEBHOOK_SECRET", "")
    if not secret:
        pytest.skip("WEBHOOK_SECRET not set — skipping HMAC signature test")

    sdk = get_e2e_client()
    assistant_id = await _ensure_assistant(sdk)

    async with webhook_receiver() as receiver:
        thread = await sdk.threads.create()
        await sdk.runs.create(
            thread_id=thread["thread_id"],
            assistant_id=assistant_id,
            input=_SHORT_PROMPT,
            webhook=receiver.url,
        )
        call = await receiver.wait_for_call(timeout=_TIMEOUT)

    signature = call["headers"].get("x-aegra-signature", "")
    assert signature, "Expected X-Aegra-Signature header when WEBHOOK_SECRET is set"
    assert _verify_hmac(secret, call["raw"], signature), "HMAC signature verification failed"


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_no_webhook_when_url_not_provided() -> None:
    """When no webhook URL is given, the receiver never gets a call."""
    sdk = get_e2e_client()
    assistant_id = await _ensure_assistant(sdk)

    receiver = WebhookReceiver()
    await receiver.start()
    try:
        thread = await sdk.threads.create()
        run = await sdk.runs.create(
            thread_id=thread["thread_id"],
            assistant_id=assistant_id,
            input=_SHORT_PROMPT,
            # No webhook= field
        )
        # Wait for the run to finish by joining it
        await sdk.runs.join(thread["thread_id"], run["run_id"])
        # Give extra time for any accidental webhook call to arrive
        import asyncio

        await asyncio.sleep(2)
        assert receiver.received_count() == 0, "Received unexpected webhook call"
    finally:
        await receiver.stop()


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_webhook_fires_on_stateless_run() -> None:
    """POST /runs/wait with a webhook field triggers a webhook on completion."""
    sdk = get_e2e_client()
    assistant_id = await _ensure_assistant(sdk)

    async with webhook_receiver() as receiver:
        elog("WebhookReceiver URL (stateless)", receiver.url)

        async with AsyncClient(base_url=settings.app.SERVER_URL, timeout=120.0) as http:
            resp = await http.post(
                "/runs/wait",
                json={
                    "assistant_id": assistant_id,
                    "input": _SHORT_PROMPT,
                    "webhook": receiver.url,
                },
            )
            elog("POST /runs/wait", {"status": resp.status_code})
            # 200 on success or 500 on LLM error — either fires the webhook
            assert resp.status_code in {200, 500}, f"Unexpected status: {resp.status_code}: {resp.text}"

        call = await receiver.wait_for_call(timeout=_TIMEOUT)
        elog("Stateless webhook call", call["body"])

    body = call["body"]
    assert body["status"] in {"success", "error", "interrupted"}
    assert "run_id" in body

    if body["status"] == "error":
        error_msg = body.get("error", "")
        if "api_key" in error_msg.lower() or "unsupported_country" in error_msg.lower():
            pytest.skip(f"Webhook fired correctly but run failed (infra): {error_msg[:80]}")


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_webhook_content_type_is_json() -> None:
    """Aegra sends Content-Type: application/json on webhook POST."""
    sdk = get_e2e_client()
    assistant_id = await _ensure_assistant(sdk)

    async with webhook_receiver() as receiver:
        thread = await sdk.threads.create()
        await sdk.runs.create(
            thread_id=thread["thread_id"],
            assistant_id=assistant_id,
            input=_SHORT_PROMPT,
            webhook=receiver.url,
        )
        call = await receiver.wait_for_call(timeout=_TIMEOUT)

    content_type = call["headers"].get("content-type", "")
    assert "application/json" in content_type, f"Expected application/json, got: {content_type}"


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_webhook_fires_once_per_run() -> None:
    """Each run fires exactly one webhook call, even for streaming runs."""
    sdk = get_e2e_client()
    assistant_id = await _ensure_assistant(sdk)

    import asyncio

    async with webhook_receiver() as receiver:
        thread = await sdk.threads.create()
        run = await sdk.runs.create(
            thread_id=thread["thread_id"],
            assistant_id=assistant_id,
            input=_SHORT_PROMPT,
            webhook=receiver.url,
        )

        # Wait for the first call
        await receiver.wait_for_call(timeout=_TIMEOUT)

        # Give extra time for any duplicate call to arrive
        await asyncio.sleep(3)

    # run should have fired exactly once
    # (count = 0 remaining since we consumed the only call above)
    assert receiver.received_count() == 0, "Webhook fired more than once for a single run"
    elog("Run that fired webhook", {"run_id": run["run_id"]})


# ---------------------------------------------------------------------------
# Tests using webhook_echo (no LLM — always succeeds, deterministic)
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_webhook_echo_graph_always_succeeds() -> None:
    """webhook_echo graph completes with status=success — no LLM API key needed."""
    sdk = get_e2e_client()
    assistant_id = await _ensure_assistant(sdk, graph_id="webhook_echo")

    async with webhook_receiver() as receiver:
        thread = await sdk.threads.create()
        run = await sdk.runs.create(
            thread_id=thread["thread_id"],
            assistant_id=assistant_id,
            input={"messages": [{"role": "user", "content": "hello webhook"}]},
            webhook=receiver.url,
        )
        elog("Echo run created", run)

        call = await receiver.wait_for_call(timeout=_TIMEOUT)
        elog("Echo webhook received", call["body"])

    body = call["body"]
    assert body["status"] == "success", f"Echo graph should always succeed, got: {body['status']}"
    assert body["run_id"] == run["run_id"]
    assert body["thread_id"] == thread["thread_id"]
    assert body["error"] is None


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_webhook_echo_graph_input_in_payload() -> None:
    """Input messages sent to the run appear verbatim in the webhook payload."""
    sdk = get_e2e_client()
    assistant_id = await _ensure_assistant(sdk, graph_id="webhook_echo")
    user_content = "test-input-verification-12345"

    async with webhook_receiver() as receiver:
        thread = await sdk.threads.create()
        await sdk.runs.create(
            thread_id=thread["thread_id"],
            assistant_id=assistant_id,
            input={"messages": [{"role": "user", "content": user_content}]},
            webhook=receiver.url,
        )

        call = await receiver.wait_for_call(timeout=_TIMEOUT)

    body = call["body"]
    assert body["status"] == "success"
    # The input passed to the run must appear in the webhook payload
    input_payload = body.get("input", {})
    messages = input_payload.get("messages", [])
    assert any(user_content in str(m.get("content", "")) for m in messages), (
        f"User content {user_content!r} not found in webhook input: {input_payload}"
    )


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_webhook_echo_graph_output_echoes_input() -> None:
    """Echo graph output contains the echoed assistant message."""
    sdk = get_e2e_client()
    assistant_id = await _ensure_assistant(sdk, graph_id="webhook_echo")
    user_content = "ping-from-e2e-test"

    async with webhook_receiver() as receiver:
        thread = await sdk.threads.create()
        await sdk.runs.create(
            thread_id=thread["thread_id"],
            assistant_id=assistant_id,
            input={"messages": [{"role": "user", "content": user_content}]},
            webhook=receiver.url,
        )

        call = await receiver.wait_for_call(timeout=_TIMEOUT)

    body = call["body"]
    assert body["status"] == "success"
    # Output contains the assistant's echo response — messages may be dicts or strings
    output = body.get("output", {})
    messages = output.get("messages", [])
    output_text = " ".join(str(m.get("content", "") if isinstance(m, dict) else m) for m in messages)
    assert f"Echo: {user_content}" in output_text, f"Expected 'Echo: {user_content}' in output messages: {messages}"


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_webhook_echo_graph_graph_id_in_payload() -> None:
    """Webhook payload graph_id matches the assistant's configured graph."""
    sdk = get_e2e_client()
    assistant_id = await _ensure_assistant(sdk, graph_id="webhook_echo")

    async with webhook_receiver() as receiver:
        thread = await sdk.threads.create()
        await sdk.runs.create(
            thread_id=thread["thread_id"],
            assistant_id=assistant_id,
            input={"messages": [{"role": "user", "content": "graph id check"}]},
            webhook=receiver.url,
        )

        call = await receiver.wait_for_call(timeout=_TIMEOUT)

    body = call["body"]
    assert body["graph_id"] == "webhook_echo", f"Expected graph_id='webhook_echo', got {body['graph_id']!r}"


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_webhook_echo_graph_thread_id_in_payload() -> None:
    """Webhook payload thread_id matches the thread the run was created on."""
    sdk = get_e2e_client()
    assistant_id = await _ensure_assistant(sdk, graph_id="webhook_echo")

    async with webhook_receiver() as receiver:
        thread = await sdk.threads.create()
        run = await sdk.runs.create(
            thread_id=thread["thread_id"],
            assistant_id=assistant_id,
            input={"messages": [{"role": "user", "content": "thread id check"}]},
            webhook=receiver.url,
        )
        elog("Run + thread IDs", {"run_id": run["run_id"], "thread_id": thread["thread_id"]})

        call = await receiver.wait_for_call(timeout=_TIMEOUT)

    body = call["body"]
    assert body["thread_id"] == thread["thread_id"], (
        f"Expected thread_id={thread['thread_id']!r}, got {body['thread_id']!r}"
    )
    assert body["run_id"] == run["run_id"]


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_webhook_echo_graph_error_sentinel_produces_error_status() -> None:
    """Sending __error__ as content makes the graph raise, firing a status=error webhook."""
    sdk = get_e2e_client()
    assistant_id = await _ensure_assistant(sdk, graph_id="webhook_echo")

    async with webhook_receiver() as receiver:
        thread = await sdk.threads.create()
        run = await sdk.runs.create(
            thread_id=thread["thread_id"],
            assistant_id=assistant_id,
            input={"messages": [{"role": "user", "content": "__error__"}]},
            webhook=receiver.url,
        )
        elog("Error-sentinel run", run)

        call = await receiver.wait_for_call(timeout=_TIMEOUT)
        elog("Error webhook received", call["body"])

    body = call["body"]
    assert body["status"] == "error", f"Expected status=error for __error__ sentinel, got: {body['status']}"
    assert body["error"], "Expected non-empty error field on error run"
    # The error message should come from the graph's RuntimeError
    assert "Deliberate error" in body["error"] or "error" in body["error"].lower(), (
        f"Unexpected error message: {body['error']!r}"
    )
    assert body["run_id"] == run["run_id"]


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_webhook_multiple_concurrent_runs_each_get_one_call() -> None:
    """Two concurrent runs on separate threads each fire exactly one webhook call to separate receivers."""
    import asyncio

    sdk = get_e2e_client()
    assistant_id = await _ensure_assistant(sdk, graph_id="webhook_echo")

    receiver_a = WebhookReceiver()
    receiver_b = WebhookReceiver()

    await asyncio.gather(receiver_a.start(), receiver_b.start())
    try:
        # Create two threads and runs concurrently
        thread_a, thread_b = await asyncio.gather(
            sdk.threads.create(),
            sdk.threads.create(),
        )
        run_a, run_b = await asyncio.gather(
            sdk.runs.create(
                thread_id=thread_a["thread_id"],
                assistant_id=assistant_id,
                input={"messages": [{"role": "user", "content": "run-a"}]},
                webhook=receiver_a.url,
            ),
            sdk.runs.create(
                thread_id=thread_b["thread_id"],
                assistant_id=assistant_id,
                input={"messages": [{"role": "user", "content": "run-b"}]},
                webhook=receiver_b.url,
            ),
        )
        elog("Concurrent runs", {"run_a": run_a["run_id"], "run_b": run_b["run_id"]})

        # Both webhooks should arrive
        call_a, call_b = await asyncio.gather(
            receiver_a.wait_for_call(timeout=_TIMEOUT),
            receiver_b.wait_for_call(timeout=_TIMEOUT),
        )
    finally:
        await asyncio.gather(receiver_a.stop(), receiver_b.stop())

    # Each receiver gets its own run's webhook
    assert call_a["body"]["run_id"] == run_a["run_id"], f"Receiver A got wrong run_id: {call_a['body']['run_id']!r}"
    assert call_b["body"]["run_id"] == run_b["run_id"], f"Receiver B got wrong run_id: {call_b['body']['run_id']!r}"
    assert call_a["body"]["status"] == "success"
    assert call_b["body"]["status"] == "success"

    # Each receiver received exactly one call
    assert receiver_a.received_count() == 0, "Receiver A got extra webhook calls"
    assert receiver_b.received_count() == 0, "Receiver B got extra webhook calls"


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_webhook_echo_completed_at_is_recent() -> None:
    """completed_at timestamp in the webhook payload is a recent ISO-8601 UTC datetime."""
    from datetime import UTC, datetime, timedelta

    sdk = get_e2e_client()
    assistant_id = await _ensure_assistant(sdk, graph_id="webhook_echo")

    before = datetime.now(UTC)

    async with webhook_receiver() as receiver:
        thread = await sdk.threads.create()
        await sdk.runs.create(
            thread_id=thread["thread_id"],
            assistant_id=assistant_id,
            input={"messages": [{"role": "user", "content": "timestamp check"}]},
            webhook=receiver.url,
        )
        call = await receiver.wait_for_call(timeout=_TIMEOUT)

    after = datetime.now(UTC)

    body = call["body"]
    completed_at_str = body.get("completed_at", "")
    assert completed_at_str, "completed_at must be non-empty"

    completed_at = datetime.fromisoformat(completed_at_str)
    if completed_at.tzinfo is None:
        completed_at = completed_at.replace(tzinfo=UTC)

    assert before - timedelta(seconds=5) <= completed_at <= after + timedelta(seconds=5), (
        f"completed_at={completed_at_str!r} is outside the expected range [{before}, {after}]"
    )
