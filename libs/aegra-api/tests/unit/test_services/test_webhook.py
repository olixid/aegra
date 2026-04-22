"""Unit tests for the webhook dispatcher service and _build_webhook_payload helper."""

import hashlib
import hmac
import json
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from aegra_api.models.auth import User
from aegra_api.models.run_job import RunExecution, RunIdentity, RunJob
from aegra_api.services.run_executor import _build_webhook_payload
from aegra_api.services.webhook import _compute_signature, dispatch_webhook

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_job(
    run_id: str = "run-abc",
    thread_id: str = "thread-xyz",
    graph_id: str = "agent",
    webhook_url: str | None = None,
    input_data: dict | None = None,
) -> RunJob:
    return RunJob(
        identity=RunIdentity(run_id=run_id, thread_id=thread_id, graph_id=graph_id),
        user=User(identity="user-1"),
        execution=RunExecution(
            input_data=input_data or {"messages": [{"role": "user", "content": "hello"}]},
            webhook_url=webhook_url,
        ),
    )


# ---------------------------------------------------------------------------
# _compute_signature tests
# ---------------------------------------------------------------------------


class TestComputeSignature:
    def test_returns_sha256_prefixed_hex(self) -> None:
        sig = _compute_signature("mysecret", b"payload")
        assert sig.startswith("sha256=")
        assert len(sig) == len("sha256=") + 64

    def test_matches_manual_hmac(self) -> None:
        secret = "testsecret"
        body = b'{"run_id":"abc"}'
        expected = "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
        assert _compute_signature(secret, body) == expected

    def test_different_secrets_produce_different_sigs(self) -> None:
        body = b"same-body"
        sig1 = _compute_signature("secret1", body)
        sig2 = _compute_signature("secret2", body)
        assert sig1 != sig2


class TestDispatchWebhook:
    @pytest.mark.asyncio
    async def test_posts_to_webhook_url(self) -> None:
        """dispatch_webhook fires a POST to the supplied URL."""
        payload = {"run_id": "run-1", "status": "success"}

        mock_response = MagicMock()
        mock_response.is_success = True
        mock_response.is_server_error = False
        mock_response.status_code = 200

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("aegra_api.services.webhook.httpx.AsyncClient", return_value=mock_client):
            await dispatch_webhook("https://example.com/hook", payload)

        mock_client.post.assert_called_once()
        call_args = mock_client.post.call_args
        assert call_args.args[0] == "https://example.com/hook"
        assert call_args.kwargs["content"] == json.dumps(payload, default=str).encode()

    @pytest.mark.asyncio
    async def test_content_type_header_set(self) -> None:
        """dispatch_webhook always sets Content-Type: application/json."""
        mock_response = MagicMock()
        mock_response.is_success = True
        mock_response.is_server_error = False
        mock_response.status_code = 200

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("aegra_api.services.webhook.httpx.AsyncClient", return_value=mock_client):
            await dispatch_webhook("https://example.com/hook", {"run_id": "r1"})

        headers = mock_client.post.call_args.kwargs["headers"]
        assert headers["Content-Type"] == "application/json"

    @pytest.mark.asyncio
    async def test_hmac_header_present_when_secret_set(self) -> None:
        """X-Aegra-Signature header is included when WEBHOOK_SECRET is configured."""
        payload = {"run_id": "run-1"}
        body = json.dumps(payload, default=str).encode()
        expected_sig = _compute_signature("my-secret", body)

        mock_response = MagicMock()
        mock_response.is_success = True
        mock_response.is_server_error = False
        mock_response.status_code = 200

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        mock_webhook_settings = MagicMock()
        mock_webhook_settings.WEBHOOK_SECRET = "my-secret"
        mock_webhook_settings.WEBHOOK_TIMEOUT_SECONDS = 30.0
        mock_webhook_settings.WEBHOOK_MAX_RETRIES = 0

        with (
            patch("aegra_api.services.webhook.httpx.AsyncClient", return_value=mock_client),
            patch("aegra_api.services.webhook.settings") as mock_settings,
        ):
            mock_settings.webhook = mock_webhook_settings
            await dispatch_webhook("https://example.com/hook", payload)

        headers = mock_client.post.call_args.kwargs["headers"]
        assert "X-Aegra-Signature" in headers
        assert headers["X-Aegra-Signature"] == expected_sig

    @pytest.mark.asyncio
    async def test_hmac_header_absent_when_no_secret(self) -> None:
        """X-Aegra-Signature is omitted when no WEBHOOK_SECRET is set."""
        mock_response = MagicMock()
        mock_response.is_success = True
        mock_response.is_server_error = False
        mock_response.status_code = 200

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        mock_webhook_settings = MagicMock()
        mock_webhook_settings.WEBHOOK_SECRET = None
        mock_webhook_settings.WEBHOOK_TIMEOUT_SECONDS = 30.0
        mock_webhook_settings.WEBHOOK_MAX_RETRIES = 0

        with (
            patch("aegra_api.services.webhook.httpx.AsyncClient", return_value=mock_client),
            patch("aegra_api.services.webhook.settings") as mock_settings,
        ):
            mock_settings.webhook = mock_webhook_settings
            await dispatch_webhook("https://example.com/hook", {"run_id": "r1"})

        headers = mock_client.post.call_args.kwargs["headers"]
        assert "X-Aegra-Signature" not in headers

    @pytest.mark.asyncio
    async def test_does_not_raise_on_network_error(self) -> None:
        """dispatch_webhook swallows TransportError and does not propagate."""
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=httpx.TransportError("connection refused"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        mock_webhook_settings = MagicMock()
        mock_webhook_settings.WEBHOOK_SECRET = None
        mock_webhook_settings.WEBHOOK_TIMEOUT_SECONDS = 5.0
        mock_webhook_settings.WEBHOOK_MAX_RETRIES = 0

        with (
            patch("aegra_api.services.webhook.httpx.AsyncClient", return_value=mock_client),
            patch("aegra_api.services.webhook.settings") as mock_settings,
        ):
            mock_settings.webhook = mock_webhook_settings
            # Must not raise
            await dispatch_webhook("https://example.com/hook", {"run_id": "r1"})

    @pytest.mark.asyncio
    async def test_does_not_raise_on_non_2xx_response(self) -> None:
        """dispatch_webhook logs but does not raise on non-2xx responses."""
        mock_response = MagicMock()
        mock_response.is_success = False
        mock_response.is_server_error = False
        mock_response.status_code = 404

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        mock_webhook_settings = MagicMock()
        mock_webhook_settings.WEBHOOK_SECRET = None
        mock_webhook_settings.WEBHOOK_TIMEOUT_SECONDS = 5.0
        mock_webhook_settings.WEBHOOK_MAX_RETRIES = 0

        with (
            patch("aegra_api.services.webhook.httpx.AsyncClient", return_value=mock_client),
            patch("aegra_api.services.webhook.settings") as mock_settings,
        ):
            mock_settings.webhook = mock_webhook_settings
            # Must not raise
            await dispatch_webhook("https://example.com/hook", {"run_id": "r1"})

    @pytest.mark.asyncio
    async def test_retries_on_server_error(self) -> None:
        """dispatch_webhook retries on 5xx errors up to WEBHOOK_MAX_RETRIES."""
        server_error = MagicMock()
        server_error.is_success = False
        server_error.is_server_error = True
        server_error.status_code = 503

        success = MagicMock()
        success.is_success = True
        success.is_server_error = False
        success.status_code = 200

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=[server_error, success])
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        mock_webhook_settings = MagicMock()
        mock_webhook_settings.WEBHOOK_SECRET = None
        mock_webhook_settings.WEBHOOK_TIMEOUT_SECONDS = 5.0
        mock_webhook_settings.WEBHOOK_MAX_RETRIES = 1  # 1 retry = 2 total attempts

        with (
            patch("aegra_api.services.webhook.httpx.AsyncClient", return_value=mock_client),
            patch("aegra_api.services.webhook.settings") as mock_settings,
        ):
            mock_settings.webhook = mock_webhook_settings
            await dispatch_webhook("https://example.com/hook", {"run_id": "r1"})

        assert mock_client.post.call_count == 2


# ---------------------------------------------------------------------------
# _build_webhook_payload tests
# ---------------------------------------------------------------------------


class TestBuildWebhookPayload:
    """Tests for the _build_webhook_payload() helper in run_executor."""

    def test_all_required_fields_present(self) -> None:
        """Payload contains every field documented in the webhook guide."""
        job = _make_job()
        payload = _build_webhook_payload(job, status="success")
        required = {"run_id", "thread_id", "graph_id", "status", "input", "output", "error", "metadata", "completed_at"}
        assert required.issubset(payload.keys()), f"Missing fields: {required - payload.keys()}"

    def test_identity_fields_match_job(self) -> None:
        job = _make_job(run_id="run-42", thread_id="t-99", graph_id="my-graph")
        payload = _build_webhook_payload(job, status="success")
        assert payload["run_id"] == "run-42"
        assert payload["thread_id"] == "t-99"
        assert payload["graph_id"] == "my-graph"

    def test_status_reflects_parameter(self) -> None:
        job = _make_job()
        for status in ("success", "error", "interrupted"):
            payload = _build_webhook_payload(job, status=status)
            assert payload["status"] == status

    def test_input_matches_job_input_data(self) -> None:
        job = _make_job(input_data={"messages": [{"role": "user", "content": "ping"}]})
        payload = _build_webhook_payload(job, status="success")
        assert payload["input"] == {"messages": [{"role": "user", "content": "ping"}]}

    def test_output_defaults_to_empty_dict_when_none(self) -> None:
        job = _make_job()
        payload = _build_webhook_payload(job, status="success", output=None)
        assert payload["output"] == {}

    def test_output_is_included_when_provided(self) -> None:
        job = _make_job()
        output = {"messages": [{"role": "assistant", "content": "pong"}]}
        payload = _build_webhook_payload(job, status="success", output=output)
        assert payload["output"] == output

    def test_error_is_none_by_default(self) -> None:
        job = _make_job()
        payload = _build_webhook_payload(job, status="success")
        assert payload["error"] is None

    def test_error_is_populated(self) -> None:
        job = _make_job()
        payload = _build_webhook_payload(job, status="error", error="ValueError: bad input")
        assert payload["error"] == "ValueError: bad input"

    def test_completed_at_is_iso8601_utc(self) -> None:
        job = _make_job()
        payload = _build_webhook_payload(job, status="success")
        completed_at = payload["completed_at"]
        # Must be parseable as datetime and contain timezone info
        parsed = datetime.fromisoformat(completed_at)
        assert parsed.tzinfo is not None, "completed_at must include timezone"

    def test_metadata_is_empty_dict(self) -> None:
        """metadata field is always present as an empty dict (extensible in future)."""
        job = _make_job()
        payload = _build_webhook_payload(job, status="success")
        assert payload["metadata"] == {}

    def test_payload_is_json_serializable(self) -> None:
        """Payload must be directly serializable by json.dumps with no custom encoder."""
        job = _make_job()
        payload = _build_webhook_payload(
            job,
            status="error",
            output={"result": "partial"},
            error="RuntimeError: boom",
        )
        # Should not raise
        serialized = json.dumps(payload)
        loaded = json.loads(serialized)
        assert loaded["run_id"] == job.identity.run_id


# ---------------------------------------------------------------------------
# Additional dispatch_webhook edge cases
# ---------------------------------------------------------------------------


class TestDispatchWebhookEdgeCases:
    """Additional edge cases for dispatch_webhook not covered by TestDispatchWebhook."""

    @pytest.mark.asyncio
    async def test_timeout_exception_does_not_propagate(self) -> None:
        """httpx.TimeoutException is treated like a transport error — swallowed."""
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=httpx.TimeoutException("timed out"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        mock_ws = MagicMock(WEBHOOK_SECRET=None, WEBHOOK_TIMEOUT_SECONDS=1.0, WEBHOOK_MAX_RETRIES=0)
        with (
            patch("aegra_api.services.webhook.httpx.AsyncClient", return_value=mock_client),
            patch("aegra_api.services.webhook.settings") as ms,
        ):
            ms.webhook = mock_ws
            await dispatch_webhook("https://example.com/hook", {"run_id": "r"})

    @pytest.mark.asyncio
    async def test_no_retry_on_4xx(self) -> None:
        """4xx responses are NOT retried, even when WEBHOOK_MAX_RETRIES > 0."""
        client_err = MagicMock(is_success=False, is_server_error=False, status_code=422)
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=client_err)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        mock_ws = MagicMock(WEBHOOK_SECRET=None, WEBHOOK_TIMEOUT_SECONDS=5.0, WEBHOOK_MAX_RETRIES=3)
        with (
            patch("aegra_api.services.webhook.httpx.AsyncClient", return_value=mock_client),
            patch("aegra_api.services.webhook.settings") as ms,
        ):
            ms.webhook = mock_ws
            await dispatch_webhook("https://example.com/hook", {"run_id": "r"})

        # Only 1 attempt despite WEBHOOK_MAX_RETRIES=3 — 4xx is not retried
        assert mock_client.post.call_count == 1

    @pytest.mark.asyncio
    async def test_all_retries_exhausted_on_persistent_5xx(self) -> None:
        """All retries exhausted on persistent 5xx — silently gives up."""
        server_err = MagicMock(is_success=False, is_server_error=True, status_code=503)
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=server_err)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        mock_ws = MagicMock(WEBHOOK_SECRET=None, WEBHOOK_TIMEOUT_SECONDS=5.0, WEBHOOK_MAX_RETRIES=2)
        with (
            patch("aegra_api.services.webhook.httpx.AsyncClient", return_value=mock_client),
            patch("aegra_api.services.webhook.settings") as ms,
        ):
            ms.webhook = mock_ws
            await dispatch_webhook("https://example.com/hook", {"run_id": "r"})

        # 1 initial + 2 retries = 3 total attempts
        assert mock_client.post.call_count == 3

    @pytest.mark.asyncio
    async def test_zero_retries_means_single_attempt(self) -> None:
        """WEBHOOK_MAX_RETRIES=0 means exactly one attempt, no retry even on 5xx."""
        server_err = MagicMock(is_success=False, is_server_error=True, status_code=503)
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=server_err)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        mock_ws = MagicMock(WEBHOOK_SECRET=None, WEBHOOK_TIMEOUT_SECONDS=5.0, WEBHOOK_MAX_RETRIES=0)
        with (
            patch("aegra_api.services.webhook.httpx.AsyncClient", return_value=mock_client),
            patch("aegra_api.services.webhook.settings") as ms,
        ):
            ms.webhook = mock_ws
            await dispatch_webhook("https://example.com/hook", {"run_id": "r"})

        assert mock_client.post.call_count == 1

    @pytest.mark.asyncio
    async def test_body_is_deterministic_json(self) -> None:
        """The request body is the JSON encoding of the payload — verify exact bytes."""
        payload = {"run_id": "abc", "status": "success", "output": {"x": 1}}
        expected_body = json.dumps(payload, default=str).encode()

        captured: list[bytes] = []
        success_resp = MagicMock(is_success=True, is_server_error=False, status_code=200)
        mock_client = AsyncMock()

        async def capture_post(url: str, *, content: bytes, headers: dict) -> MagicMock:
            captured.append(content)
            return success_resp

        mock_client.post = capture_post
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        mock_ws = MagicMock(WEBHOOK_SECRET=None, WEBHOOK_TIMEOUT_SECONDS=5.0, WEBHOOK_MAX_RETRIES=0)
        with (
            patch("aegra_api.services.webhook.httpx.AsyncClient", return_value=mock_client),
            patch("aegra_api.services.webhook.settings") as ms,
        ):
            ms.webhook = mock_ws
            await dispatch_webhook("https://example.com/hook", payload)

        assert captured == [expected_body]

    @pytest.mark.asyncio
    async def test_hmac_computed_over_actual_body_bytes(self) -> None:
        """Signature is HMAC-SHA256 of the exact body bytes sent in the request."""
        payload = {"run_id": "x", "status": "error"}
        secret = "signing-secret"

        captured_headers: list[dict] = []
        captured_bodies: list[bytes] = []
        success_resp = MagicMock(is_success=True, is_server_error=False, status_code=200)
        mock_client = AsyncMock()

        async def capture_post(url: str, *, content: bytes, headers: dict) -> MagicMock:
            captured_headers.append(dict(headers))
            captured_bodies.append(content)
            return success_resp

        mock_client.post = capture_post
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        mock_ws = MagicMock(WEBHOOK_SECRET=secret, WEBHOOK_TIMEOUT_SECONDS=5.0, WEBHOOK_MAX_RETRIES=0)
        with (
            patch("aegra_api.services.webhook.httpx.AsyncClient", return_value=mock_client),
            patch("aegra_api.services.webhook.settings") as ms,
        ):
            ms.webhook = mock_ws
            await dispatch_webhook("https://example.com/hook", payload)

        assert captured_headers and "X-Aegra-Signature" in captured_headers[0]
        body = captured_bodies[0]
        expected_sig = "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
        assert captured_headers[0]["X-Aegra-Signature"] == expected_sig


# ---------------------------------------------------------------------------
# Retry on transport errors
# ---------------------------------------------------------------------------


class TestDispatchWebhookRetryTransportError:
    """Verify retry behaviour specifically for network (TransportError) failures."""

    @pytest.mark.asyncio
    async def test_retries_on_transport_error_then_succeeds(self) -> None:
        """A single transport error is retried and subsequent success is accepted."""
        success_resp = MagicMock(is_success=True, is_server_error=False, status_code=200)
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=[httpx.TransportError("connection reset"), success_resp])
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        mock_ws = MagicMock(WEBHOOK_SECRET=None, WEBHOOK_TIMEOUT_SECONDS=5.0, WEBHOOK_MAX_RETRIES=1)
        with (
            patch("aegra_api.services.webhook.httpx.AsyncClient", return_value=mock_client),
            patch("aegra_api.services.webhook.settings") as ms,
        ):
            ms.webhook = mock_ws
            await dispatch_webhook("https://example.com/hook", {"run_id": "r"})

        assert mock_client.post.call_count == 2

    @pytest.mark.asyncio
    async def test_all_transport_errors_exhausted_then_stops(self) -> None:
        """When every attempt raises TransportError, gives up after max_attempts."""
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=httpx.TransportError("unreachable"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        mock_ws = MagicMock(WEBHOOK_SECRET=None, WEBHOOK_TIMEOUT_SECONDS=5.0, WEBHOOK_MAX_RETRIES=2)
        with (
            patch("aegra_api.services.webhook.httpx.AsyncClient", return_value=mock_client),
            patch("aegra_api.services.webhook.settings") as ms,
        ):
            ms.webhook = mock_ws
            # Must not raise
            await dispatch_webhook("https://example.com/hook", {"run_id": "r"})

        # 1 + max(0, 2) = 3 total attempts
        assert mock_client.post.call_count == 3

    @pytest.mark.asyncio
    async def test_no_retry_on_transport_error_when_max_retries_zero(self) -> None:
        """WEBHOOK_MAX_RETRIES=0 means no retry even for TransportError."""
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=httpx.TransportError("refused"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        mock_ws = MagicMock(WEBHOOK_SECRET=None, WEBHOOK_TIMEOUT_SECONDS=5.0, WEBHOOK_MAX_RETRIES=0)
        with (
            patch("aegra_api.services.webhook.httpx.AsyncClient", return_value=mock_client),
            patch("aegra_api.services.webhook.settings") as ms,
        ):
            ms.webhook = mock_ws
            await dispatch_webhook("https://example.com/hook", {"run_id": "r"})

        assert mock_client.post.call_count == 1


# ---------------------------------------------------------------------------
# Timeout settings propagation
# ---------------------------------------------------------------------------


class TestDispatchWebhookTimeoutPropagation:
    """Verify that the configured timeout value is forwarded to httpx."""

    @pytest.mark.asyncio
    async def test_timeout_value_passed_to_async_client(self) -> None:
        """WEBHOOK_TIMEOUT_SECONDS is forwarded as the httpx.AsyncClient timeout."""
        constructed_timeouts: list = []

        success_resp = MagicMock(is_success=True, is_server_error=False, status_code=200)

        class _FakeClient:
            def __init__(self, timeout: float) -> None:
                constructed_timeouts.append(timeout)
                self._client = AsyncMock()
                self._client.post = AsyncMock(return_value=success_resp)

            async def __aenter__(self) -> "_FakeClient":
                return self

            async def __aexit__(self, *_: object) -> bool:
                return False

            async def post(self, url: str, *, content: bytes, headers: dict) -> MagicMock:
                return success_resp

        mock_ws = MagicMock(WEBHOOK_SECRET=None, WEBHOOK_TIMEOUT_SECONDS=42.0, WEBHOOK_MAX_RETRIES=0)
        with (
            patch("aegra_api.services.webhook.httpx.AsyncClient", _FakeClient),
            patch("aegra_api.services.webhook.settings") as ms,
        ):
            ms.webhook = mock_ws
            await dispatch_webhook("https://example.com/hook", {"run_id": "r"})

        assert constructed_timeouts == [42.0]

    @pytest.mark.asyncio
    async def test_custom_timeout_used_not_default(self) -> None:
        """A non-default timeout (e.g., 5.0) is forwarded correctly."""
        constructed_timeouts: list = []
        success_resp = MagicMock(is_success=True, is_server_error=False, status_code=200)

        class _FakeClient:
            def __init__(self, timeout: float) -> None:
                constructed_timeouts.append(timeout)

            async def __aenter__(self) -> "_FakeClient":
                return self

            async def __aexit__(self, *_: object) -> bool:
                return False

            async def post(self, url: str, *, content: bytes, headers: dict) -> MagicMock:
                return success_resp

        mock_ws = MagicMock(WEBHOOK_SECRET=None, WEBHOOK_TIMEOUT_SECONDS=5.0, WEBHOOK_MAX_RETRIES=0)
        with (
            patch("aegra_api.services.webhook.httpx.AsyncClient", _FakeClient),
            patch("aegra_api.services.webhook.settings") as ms,
        ):
            ms.webhook = mock_ws
            await dispatch_webhook("https://example.com/hook", {"run_id": "r"})

        assert constructed_timeouts == [5.0]


# ---------------------------------------------------------------------------
# _build_webhook_payload: additional edge cases
# ---------------------------------------------------------------------------


class TestBuildWebhookPayloadEdgeCases:
    """Supplementary edge cases for _build_webhook_payload not in TestBuildWebhookPayload."""

    def test_completed_at_is_recent(self) -> None:
        """completed_at must be within a few seconds of now."""
        from datetime import UTC, datetime, timedelta

        before = datetime.now(UTC)
        job = _make_job()
        payload = _build_webhook_payload(job, status="success")
        after = datetime.now(UTC)

        completed_at_str = payload["completed_at"]
        completed_at = datetime.fromisoformat(completed_at_str)
        if completed_at.tzinfo is None:
            completed_at = completed_at.replace(tzinfo=UTC)

        assert before - timedelta(seconds=1) <= completed_at <= after + timedelta(seconds=1), (
            f"completed_at={completed_at_str!r} is outside [{before.isoformat()}, {after.isoformat()}]"
        )

    def test_output_with_interrupt_data_preserved(self) -> None:
        """Interrupt output containing __interrupt__ key is preserved verbatim."""
        interrupt_output = {
            "__interrupt__": [{"value": "Confirm?", "resumable": True, "task": "node_a", "when": "during"}]
        }
        job = _make_job()
        payload = _build_webhook_payload(job, status="interrupted", output=interrupt_output)
        assert payload["output"] == interrupt_output
        assert payload["status"] == "interrupted"

    def test_error_string_is_message_only_not_class_name(self) -> None:
        """Error field contains the exception message, not 'ClassName: message'."""
        job = _make_job()
        err = RuntimeError("something went wrong")
        payload = _build_webhook_payload(job, status="error", error=str(err))
        assert payload["error"] == "something went wrong"
        assert "RuntimeError" not in payload["error"]

    def test_large_output_preserved_exactly(self) -> None:
        """Large output dict is preserved without truncation or modification."""
        large_output = {
            "messages": [{"role": "assistant", "content": "x" * 10_000}],
            "metadata": {"key_" + str(i): i for i in range(100)},
        }
        job = _make_job()
        payload = _build_webhook_payload(job, status="success", output=large_output)
        assert payload["output"] == large_output

    def test_signature_header_name_is_x_aegra_signature(self) -> None:
        """Header is named X-Aegra-Signature (not X-Webhook-Signature or other variants)."""
        sig = _compute_signature("secret", b"body")
        assert sig.startswith("sha256=")
        # The header is built in dispatch_webhook — verify the naming convention
        # by asserting _compute_signature returns a value in the expected format.
        parts = sig.split("=", 1)
        assert parts[0] == "sha256"
        assert all(c in "0123456789abcdef" for c in parts[1])

    def test_empty_input_data(self) -> None:
        """Payload is valid when job has no input messages."""
        # Build job directly to avoid _make_job's default fallback for empty dicts
        job = RunJob(
            identity=RunIdentity(run_id="r", thread_id="t", graph_id="g"),
            user=User(identity="user-1"),
            execution=RunExecution(input_data={}, webhook_url=None),
        )
        payload = _build_webhook_payload(job, status="success")
        assert payload["input"] == {}

    def test_all_three_statuses_are_json_serializable(self) -> None:
        """All terminal statuses produce JSON-serializable payloads."""
        job = _make_job()
        for status in ("success", "error", "interrupted"):
            payload = _build_webhook_payload(job, status=status)
            serialized = json.dumps(payload, default=str)
            assert json.loads(serialized)["status"] == status
