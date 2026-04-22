"""Outbound webhook dispatcher.

Fires a best-effort HTTP POST to a caller-supplied URL after a run reaches a
terminal state. The caller controls the URL; Aegra simply delivers the payload.

Signing (optional):
    When ``WEBHOOK_SECRET`` is set, each request includes an
    ``X-Aegra-Signature: sha256=<hex>`` header so recipients can verify the
    payload was not tampered with. Verification uses HMAC-SHA256 over the raw
    JSON body bytes.

Usage in run_executor:
    await _best_effort_signal(dispatch_webhook, webhook_url, payload)
"""

import hashlib
import hmac
import json
from typing import Any

import httpx
import structlog

from aegra_api.settings import settings

logger = structlog.getLogger(__name__)


def _compute_signature(secret: str, body: bytes) -> str:
    """Return ``sha256=<hex>`` HMAC digest of *body* using *secret*."""
    digest = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    return f"sha256={digest}"


async def dispatch_webhook(webhook_url: str, payload: dict[str, Any]) -> None:
    """POST *payload* as JSON to *webhook_url*.

    Retries up to ``settings.webhook.WEBHOOK_MAX_RETRIES`` times on transient
    errors (network failures, 5xx responses). All exceptions are logged and
    suppressed — a webhook failure must never affect the run's persisted state.

    Args:
        webhook_url: Destination URL supplied by the run creator.
        payload: Run result data to deliver.
    """
    body = json.dumps(payload, default=str).encode()

    headers: dict[str, str] = {"Content-Type": "application/json"}
    if settings.webhook.WEBHOOK_SECRET:
        headers["X-Aegra-Signature"] = _compute_signature(settings.webhook.WEBHOOK_SECRET, body)

    timeout = settings.webhook.WEBHOOK_TIMEOUT_SECONDS
    max_attempts = 1 + max(0, settings.webhook.WEBHOOK_MAX_RETRIES)

    async with httpx.AsyncClient(timeout=timeout) as client:
        for attempt in range(1, max_attempts + 1):
            try:
                response = await client.post(webhook_url, content=body, headers=headers)
                if response.is_success:
                    logger.info(
                        "Webhook delivered",
                        url=webhook_url,
                        status=response.status_code,
                        attempt=attempt,
                    )
                    return

                logger.warning(
                    "Webhook returned non-2xx status",
                    url=webhook_url,
                    status=response.status_code,
                    attempt=attempt,
                )
                if not response.is_server_error or attempt == max_attempts:
                    return

            except httpx.TransportError as exc:
                logger.warning(
                    "Webhook delivery failed (network error)",
                    url=webhook_url,
                    error=str(exc),
                    attempt=attempt,
                )
                if attempt == max_attempts:
                    return
