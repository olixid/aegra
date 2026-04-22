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
