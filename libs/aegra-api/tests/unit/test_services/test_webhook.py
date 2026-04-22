"""Unit tests for the webhook dispatcher service."""

import hashlib
import hmac
import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from aegra_api.services.webhook import _compute_signature, dispatch_webhook


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
