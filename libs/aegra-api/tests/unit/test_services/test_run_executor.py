"""Unit tests for run_executor service."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aegra_api.models.auth import User
from aegra_api.models.run_job import RunExecution, RunIdentity, RunJob


async def _empty_async_gen():  # type: ignore[no-untyped-def]
    return
    yield  # noqa: RET504 — makes this an async generator


def _make_job(run_id: str = "run-1") -> RunJob:
    return RunJob(
        identity=RunIdentity(run_id=run_id, thread_id="thread-1", graph_id="graph-1"),
        user=User(identity="user-1"),
        execution=RunExecution(input_data={"msg": "hello"}),
    )


def _patch_execute_run_deps() -> dict[str, MagicMock | AsyncMock]:
    """Return a dict of patch targets and their mocks for execute_run tests."""
    return {}


class TestExecuteRunSuccess:
    @pytest.mark.asyncio
    async def test_success_path_updates_status_and_signals(self) -> None:
        """execute_run sets running -> success and signals end event."""
        mock_graph = MagicMock()
        mock_graph.__aenter__ = AsyncMock(return_value=mock_graph)
        mock_graph.__aexit__ = AsyncMock(return_value=False)

        mock_service = MagicMock()
        mock_service.get_graph = MagicMock(return_value=mock_graph)

        mock_update = AsyncMock()
        mock_finalize = AsyncMock()

        with (
            patch("aegra_api.services.run_executor.get_langgraph_service", return_value=mock_service),
            patch("aegra_api.services.run_executor.update_run_status", mock_update),
            patch("aegra_api.services.run_executor.finalize_run", mock_finalize),
            patch("aegra_api.services.run_executor.streaming_service") as mock_streaming,
            patch("aegra_api.services.run_executor.stream_graph_events", return_value=_empty_async_gen()),
            patch("aegra_api.services.run_executor._signal_end_event", new_callable=AsyncMock) as mock_signal_end,
            patch("aegra_api.services.run_executor._signal_run_done", new_callable=AsyncMock),
            patch("aegra_api.services.run_executor.with_auth_ctx") as mock_auth,
        ):
            mock_auth_ctx = AsyncMock()
            mock_auth_ctx.__aenter__ = AsyncMock(return_value=None)
            mock_auth_ctx.__aexit__ = AsyncMock(return_value=False)
            mock_auth.return_value = mock_auth_ctx
            mock_streaming.cleanup_run = AsyncMock()

            from aegra_api.services.run_executor import execute_run

            await execute_run(_make_job())

        # update_run_status called once for "running"
        assert mock_update.await_count == 1
        assert mock_update.await_args_list[0].args == ("run-1", "running")

        # finalize_run called once for success
        mock_finalize.assert_awaited_once()
        assert mock_finalize.await_args.kwargs["status"] == "success"

        mock_signal_end.assert_awaited_once_with("run-1", "success")


class TestExecuteRunCancelledError:
    @pytest.mark.asyncio
    async def test_cancelled_error_sets_interrupted_and_signals(self) -> None:
        mock_update = AsyncMock()
        mock_finalize = AsyncMock()

        with (
            patch("aegra_api.services.run_executor.update_run_status", mock_update),
            patch("aegra_api.services.run_executor.finalize_run", mock_finalize),
            patch("aegra_api.services.run_executor.streaming_service") as mock_streaming,
            patch("aegra_api.services.run_executor._signal_run_done", new_callable=AsyncMock),
            patch(
                "aegra_api.services.run_executor._stream_graph",
                new_callable=AsyncMock,
                side_effect=asyncio.CancelledError,
            ),
        ):
            mock_streaming.signal_run_cancelled = AsyncMock()
            mock_streaming.cleanup_run = AsyncMock()

            from aegra_api.services.run_executor import execute_run

            with pytest.raises(asyncio.CancelledError):
                await execute_run(_make_job())

        # update_run_status called once for "running"
        assert mock_update.await_count == 1
        assert mock_update.await_args_list[0].args == ("run-1", "running")
        # finalize_run called for "interrupted"
        mock_finalize.assert_awaited_once()
        assert mock_finalize.await_args.kwargs["status"] == "interrupted"
        mock_streaming.signal_run_cancelled.assert_awaited_once_with("run-1")


class TestExecuteRunException:
    @pytest.mark.asyncio
    async def test_exception_sets_error_and_signals(self) -> None:
        mock_update = AsyncMock()
        mock_finalize = AsyncMock()

        with (
            patch("aegra_api.services.run_executor.update_run_status", mock_update),
            patch("aegra_api.services.run_executor.finalize_run", mock_finalize),
            patch("aegra_api.services.run_executor.streaming_service") as mock_streaming,
            patch("aegra_api.services.run_executor._signal_run_done", new_callable=AsyncMock),
            patch(
                "aegra_api.services.run_executor._stream_graph",
                new_callable=AsyncMock,
                side_effect=RuntimeError("graph exploded"),
            ),
        ):
            mock_streaming.signal_run_error = AsyncMock()
            mock_streaming.cleanup_run = AsyncMock()

            from aegra_api.services.run_executor import execute_run

            await execute_run(_make_job())

        # update_run_status called once for "running"
        assert mock_update.await_count == 1
        assert mock_update.await_args_list[0].args == ("run-1", "running")
        # finalize_run called for "error"
        mock_finalize.assert_awaited_once()
        assert mock_finalize.await_args.kwargs["status"] == "error"
        assert mock_finalize.await_args.kwargs["thread_status"] == "error"
        mock_streaming.signal_run_error.assert_awaited_once()
        # Verify sanitized message used (not raw exception)
        error_args = mock_streaming.signal_run_error.await_args
        assert "RuntimeError" in error_args.args[1]
        assert "execution failed" in error_args.args[1]


class TestSignalEndEvent:
    @pytest.mark.asyncio
    async def test_publishes_end_event(self) -> None:
        mock_broker = MagicMock()
        mock_broker.is_finished.return_value = False
        mock_broker.put = AsyncMock()

        with patch("aegra_api.services.run_executor.broker_manager") as mock_bm:
            mock_bm.get_broker.return_value = mock_broker
            mock_bm.allocate_event_id = AsyncMock(return_value="run-1_event_5")

            from aegra_api.services.run_executor import _signal_end_event

            await _signal_end_event("run-1", "success")

        mock_broker.put.assert_awaited_once_with("run-1_event_5", ("end", {"status": "success"}))

    @pytest.mark.asyncio
    async def test_noop_when_broker_is_none(self) -> None:
        with patch("aegra_api.services.run_executor.broker_manager") as mock_bm:
            mock_bm.get_broker.return_value = None

            from aegra_api.services.run_executor import _signal_end_event

            await _signal_end_event("run-1", "success")
            # No error, no put call

    @pytest.mark.asyncio
    async def test_noop_when_broker_is_finished(self) -> None:
        mock_broker = MagicMock()
        mock_broker.is_finished.return_value = True

        with patch("aegra_api.services.run_executor.broker_manager") as mock_bm:
            mock_bm.get_broker.return_value = mock_broker

            from aegra_api.services.run_executor import _signal_end_event

            await _signal_end_event("run-1", "success")


class TestSignalRunDone:
    @pytest.mark.asyncio
    async def test_sets_redis_key(self) -> None:
        mock_client = AsyncMock()

        with patch("aegra_api.services.run_executor.redis_manager") as mock_rm:
            mock_rm.get_client.return_value = mock_client

            from aegra_api.services.run_executor import _signal_run_done

            await _signal_run_done("run-1")

        mock_client.set.assert_awaited_once()
        call_args = mock_client.set.await_args
        assert "run-1" in call_args.args[0]
        assert call_args.args[1] == "1"

    @pytest.mark.asyncio
    async def test_uses_configured_channel_prefix(self) -> None:
        """Regression: done-key must derive from REDIS_CHANNEL_PREFIX, not a hardcoded string."""
        mock_client = AsyncMock()

        with (
            patch("aegra_api.services.run_executor.redis_manager") as mock_rm,
            patch("aegra_api.services.run_executor.settings") as mock_settings,
        ):
            mock_rm.get_client.return_value = mock_client
            mock_settings.redis.REDIS_CHANNEL_PREFIX = "aegra:agent-foo:run:"

            from aegra_api.services.run_executor import _signal_run_done

            await _signal_run_done("run-1")

        key = mock_client.set.await_args.args[0]
        assert key == "aegra:agent-foo:run:done:run-1"

    @pytest.mark.asyncio
    async def test_logs_debug_on_redis_failure(self) -> None:
        with patch("aegra_api.services.run_executor.redis_manager") as mock_rm:
            mock_rm.get_client.side_effect = Exception("connection refused")

            from aegra_api.services.run_executor import _signal_run_done

            # Should not raise
            await _signal_run_done("run-1")


class TestLeaseLossCancellation:
    @pytest.mark.asyncio
    async def test_lease_loss_cancel_skips_finalize_and_signal(self) -> None:
        """Regression: when cancellation is due to lease loss (not user action),
        execute_run must NOT finalize the run, send SSE events, signal done,
        or clean up the broker — another worker will re-execute it."""
        mock_update = AsyncMock()
        mock_finalize = AsyncMock()
        mock_signal_done = AsyncMock()

        with (
            patch("aegra_api.services.run_executor.update_run_status", mock_update),
            patch("aegra_api.services.run_executor.finalize_run", mock_finalize),
            patch("aegra_api.services.run_executor.streaming_service") as mock_streaming,
            patch("aegra_api.services.run_executor._signal_run_done", mock_signal_done),
            patch(
                "aegra_api.services.run_executor._stream_graph",
                new_callable=AsyncMock,
                side_effect=asyncio.CancelledError,
            ),
        ):
            mock_streaming.signal_run_cancelled = AsyncMock()
            mock_streaming.cleanup_run = AsyncMock()

            from aegra_api.services.run_executor import _lease_loss_cancellations, execute_run

            # Simulate heartbeat marking this as a lease-loss cancel
            _lease_loss_cancellations.add("run-1")
            try:
                with pytest.raises(asyncio.CancelledError):
                    await execute_run(_make_job())
            finally:
                _lease_loss_cancellations.discard("run-1")

        # finalize_run must NOT be called — the new worker owns this run
        mock_finalize.assert_not_awaited()
        # SSE cancel signal must NOT be sent — clients should stay connected
        mock_streaming.signal_run_cancelled.assert_not_awaited()
        # Done-key must NOT be set — would cause wait_for_completion to return early
        mock_signal_done.assert_not_awaited()
        # Broker must NOT be cleaned up — new worker needs it
        mock_streaming.cleanup_run.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_user_cancel_still_finalizes(self) -> None:
        """Normal (user-initiated) cancellation must still finalize and signal."""
        mock_update = AsyncMock()
        mock_finalize = AsyncMock()
        mock_signal_done = AsyncMock()

        with (
            patch("aegra_api.services.run_executor.update_run_status", mock_update),
            patch("aegra_api.services.run_executor.finalize_run", mock_finalize),
            patch("aegra_api.services.run_executor.streaming_service") as mock_streaming,
            patch("aegra_api.services.run_executor._signal_run_done", mock_signal_done),
            patch(
                "aegra_api.services.run_executor._stream_graph",
                new_callable=AsyncMock,
                side_effect=asyncio.CancelledError,
            ),
        ):
            mock_streaming.signal_run_cancelled = AsyncMock()
            mock_streaming.cleanup_run = AsyncMock()

            from aegra_api.services.run_executor import execute_run

            with pytest.raises(asyncio.CancelledError):
                await execute_run(_make_job())

        # Normal cancel: finalize and signal MUST happen
        mock_finalize.assert_awaited_once()
        assert mock_finalize.await_args.kwargs["status"] == "interrupted"
        mock_streaming.signal_run_cancelled.assert_awaited_once_with("run-1")
        # Done-key and cleanup MUST happen on normal cancel
        mock_signal_done.assert_awaited_once_with("run-1")
        mock_streaming.cleanup_run.assert_awaited_once_with("run-1")


# ---------------------------------------------------------------------------
# Webhook dispatch from execute_run
# ---------------------------------------------------------------------------


def _make_job_with_webhook(
    run_id: str = "run-1",
    webhook_url: str | None = "https://example.com/webhook",
) -> RunJob:
    return RunJob(
        identity=RunIdentity(run_id=run_id, thread_id="thread-1", graph_id="graph-1"),
        user=User(identity="user-1"),
        execution=RunExecution(input_data={"msg": "hello"}, webhook_url=webhook_url),
    )


def _base_executor_patches() -> dict:
    """Common patch targets for execute_run tests."""
    return {
        "aegra_api.services.run_executor.update_run_status": AsyncMock(),
        "aegra_api.services.run_executor.finalize_run": AsyncMock(),
        "aegra_api.services.run_executor._signal_run_done": AsyncMock(),
        "aegra_api.services.run_executor._signal_end_event": AsyncMock(),
    }


class TestExecuteRunWebhookDispatch:
    """Verify that execute_run dispatches webhooks for each terminal state."""

    @pytest.mark.asyncio
    async def test_webhook_dispatched_on_success(self) -> None:
        """execute_run fires a webhook when the run completes successfully."""
        mock_dispatch = AsyncMock()
        mocks = _base_executor_patches()

        with (
            patch(
                "aegra_api.services.run_executor.update_run_status",
                mocks["aegra_api.services.run_executor.update_run_status"],
            ),
            patch(
                "aegra_api.services.run_executor.finalize_run", mocks["aegra_api.services.run_executor.finalize_run"]
            ),
            patch(
                "aegra_api.services.run_executor._signal_run_done",
                mocks["aegra_api.services.run_executor._signal_run_done"],
            ),
            patch(
                "aegra_api.services.run_executor._signal_end_event",
                mocks["aegra_api.services.run_executor._signal_end_event"],
            ),
            patch("aegra_api.services.run_executor.streaming_service") as mock_streaming,
            patch("aegra_api.services.run_executor._stream_graph", new_callable=AsyncMock) as mock_stream,
            patch("aegra_api.services.run_executor.dispatch_webhook", mock_dispatch),
        ):
            mock_streaming.cleanup_run = AsyncMock()
            mock_output = MagicMock()
            mock_output.has_interrupt = False
            mock_output.data = {"messages": [{"role": "assistant", "content": "hi"}]}
            mock_stream.return_value = mock_output

            from aegra_api.services.run_executor import execute_run

            await execute_run(_make_job_with_webhook())

        mock_dispatch.assert_awaited_once()
        payload = mock_dispatch.await_args.args[1]
        assert payload["status"] == "success"
        assert payload["run_id"] == "run-1"

    @pytest.mark.asyncio
    async def test_webhook_dispatched_on_error(self) -> None:
        """execute_run fires a webhook when the graph raises an exception."""
        mock_dispatch = AsyncMock()
        mocks = _base_executor_patches()

        with (
            patch(
                "aegra_api.services.run_executor.update_run_status",
                mocks["aegra_api.services.run_executor.update_run_status"],
            ),
            patch(
                "aegra_api.services.run_executor.finalize_run", mocks["aegra_api.services.run_executor.finalize_run"]
            ),
            patch(
                "aegra_api.services.run_executor._signal_run_done",
                mocks["aegra_api.services.run_executor._signal_run_done"],
            ),
            patch(
                "aegra_api.services.run_executor._signal_end_event",
                mocks["aegra_api.services.run_executor._signal_end_event"],
            ),
            patch("aegra_api.services.run_executor.streaming_service") as mock_streaming,
            patch(
                "aegra_api.services.run_executor._stream_graph",
                new_callable=AsyncMock,
                side_effect=RuntimeError("graph boom"),
            ),
            patch("aegra_api.services.run_executor.dispatch_webhook", mock_dispatch),
        ):
            mock_streaming.cleanup_run = AsyncMock()
            mock_streaming.signal_run_error = AsyncMock()

            from aegra_api.services.run_executor import execute_run

            await execute_run(_make_job_with_webhook())

        mock_dispatch.assert_awaited_once()
        payload = mock_dispatch.await_args.args[1]
        assert payload["status"] == "error"
        assert "graph boom" in payload["error"]

    @pytest.mark.asyncio
    async def test_webhook_dispatched_on_cancelled_non_lease_loss(self) -> None:
        """execute_run fires a webhook when the run is cancelled by the user."""
        mock_dispatch = AsyncMock()
        mocks = _base_executor_patches()

        with (
            patch(
                "aegra_api.services.run_executor.update_run_status",
                mocks["aegra_api.services.run_executor.update_run_status"],
            ),
            patch(
                "aegra_api.services.run_executor.finalize_run", mocks["aegra_api.services.run_executor.finalize_run"]
            ),
            patch(
                "aegra_api.services.run_executor._signal_run_done",
                mocks["aegra_api.services.run_executor._signal_run_done"],
            ),
            patch(
                "aegra_api.services.run_executor._signal_end_event",
                mocks["aegra_api.services.run_executor._signal_end_event"],
            ),
            patch("aegra_api.services.run_executor.streaming_service") as mock_streaming,
            patch(
                "aegra_api.services.run_executor._stream_graph",
                new_callable=AsyncMock,
                side_effect=asyncio.CancelledError,
            ),
            patch("aegra_api.services.run_executor.dispatch_webhook", mock_dispatch),
        ):
            mock_streaming.cleanup_run = AsyncMock()
            mock_streaming.signal_run_cancelled = AsyncMock()

            from aegra_api.services.run_executor import execute_run

            with pytest.raises(asyncio.CancelledError):
                await execute_run(_make_job_with_webhook())

        mock_dispatch.assert_awaited_once()
        payload = mock_dispatch.await_args.args[1]
        assert payload["status"] == "interrupted"

    @pytest.mark.asyncio
    async def test_webhook_not_dispatched_on_lease_loss_cancel(self) -> None:
        """execute_run must NOT fire a webhook on lease-loss cancellation."""
        mock_dispatch = AsyncMock()
        mocks = _base_executor_patches()

        with (
            patch(
                "aegra_api.services.run_executor.update_run_status",
                mocks["aegra_api.services.run_executor.update_run_status"],
            ),
            patch(
                "aegra_api.services.run_executor.finalize_run", mocks["aegra_api.services.run_executor.finalize_run"]
            ),
            patch(
                "aegra_api.services.run_executor._signal_run_done",
                mocks["aegra_api.services.run_executor._signal_run_done"],
            ),
            patch(
                "aegra_api.services.run_executor._signal_end_event",
                mocks["aegra_api.services.run_executor._signal_end_event"],
            ),
            patch("aegra_api.services.run_executor.streaming_service") as mock_streaming,
            patch(
                "aegra_api.services.run_executor._stream_graph",
                new_callable=AsyncMock,
                side_effect=asyncio.CancelledError,
            ),
            patch("aegra_api.services.run_executor.dispatch_webhook", mock_dispatch),
        ):
            mock_streaming.cleanup_run = AsyncMock()
            mock_streaming.signal_run_cancelled = AsyncMock()

            from aegra_api.services.run_executor import _lease_loss_cancellations, execute_run

            _lease_loss_cancellations.add("run-1")
            try:
                with pytest.raises(asyncio.CancelledError):
                    await execute_run(_make_job_with_webhook())
            finally:
                _lease_loss_cancellations.discard("run-1")

        # Lease-loss cancellation MUST NOT dispatch webhook — run is re-enqueued
        mock_dispatch.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_webhook_not_dispatched_when_no_url(self) -> None:
        """When webhook_url is None, dispatch_webhook must never be called."""
        mock_dispatch = AsyncMock()
        mocks = _base_executor_patches()

        with (
            patch(
                "aegra_api.services.run_executor.update_run_status",
                mocks["aegra_api.services.run_executor.update_run_status"],
            ),
            patch(
                "aegra_api.services.run_executor.finalize_run", mocks["aegra_api.services.run_executor.finalize_run"]
            ),
            patch(
                "aegra_api.services.run_executor._signal_run_done",
                mocks["aegra_api.services.run_executor._signal_run_done"],
            ),
            patch(
                "aegra_api.services.run_executor._signal_end_event",
                mocks["aegra_api.services.run_executor._signal_end_event"],
            ),
            patch("aegra_api.services.run_executor.streaming_service") as mock_streaming,
            patch("aegra_api.services.run_executor._stream_graph", new_callable=AsyncMock) as mock_stream,
            patch("aegra_api.services.run_executor.dispatch_webhook", mock_dispatch),
        ):
            mock_streaming.cleanup_run = AsyncMock()
            mock_output = MagicMock(has_interrupt=False, data={})
            mock_stream.return_value = mock_output

            from aegra_api.services.run_executor import execute_run

            # webhook_url=None
            await execute_run(_make_job_with_webhook(webhook_url=None))

        mock_dispatch.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_webhook_dispatched_on_interrupt(self) -> None:
        """execute_run fires a webhook with status='interrupted' for human-in-the-loop."""
        mock_dispatch = AsyncMock()
        mocks = _base_executor_patches()

        with (
            patch(
                "aegra_api.services.run_executor.update_run_status",
                mocks["aegra_api.services.run_executor.update_run_status"],
            ),
            patch(
                "aegra_api.services.run_executor.finalize_run", mocks["aegra_api.services.run_executor.finalize_run"]
            ),
            patch(
                "aegra_api.services.run_executor._signal_run_done",
                mocks["aegra_api.services.run_executor._signal_run_done"],
            ),
            patch(
                "aegra_api.services.run_executor._signal_end_event",
                mocks["aegra_api.services.run_executor._signal_end_event"],
            ),
            patch("aegra_api.services.run_executor.streaming_service") as mock_streaming,
            patch("aegra_api.services.run_executor._stream_graph", new_callable=AsyncMock) as mock_stream,
            patch("aegra_api.services.run_executor.dispatch_webhook", mock_dispatch),
        ):
            mock_streaming.cleanup_run = AsyncMock()
            # Simulate interrupt: has_interrupt=True
            mock_output = MagicMock(has_interrupt=True, data={"__interrupt__": [{"value": "need input"}]})
            mock_stream.return_value = mock_output

            from aegra_api.services.run_executor import execute_run

            await execute_run(_make_job_with_webhook())

        mock_dispatch.assert_awaited_once()
        payload = mock_dispatch.await_args.args[1]
        assert payload["status"] == "interrupted"

    @pytest.mark.asyncio
    async def test_webhook_failure_does_not_affect_run_status(self) -> None:
        """A failing webhook call must not propagate — run finalize must still complete."""
        mock_finalize = AsyncMock()

        with (
            patch("aegra_api.services.run_executor.update_run_status", AsyncMock()),
            patch("aegra_api.services.run_executor.finalize_run", mock_finalize),
            patch("aegra_api.services.run_executor._signal_run_done", AsyncMock()),
            patch("aegra_api.services.run_executor._signal_end_event", AsyncMock()),
            patch("aegra_api.services.run_executor.streaming_service") as mock_streaming,
            patch("aegra_api.services.run_executor._stream_graph", new_callable=AsyncMock) as mock_stream,
            # dispatch_webhook raises — should be swallowed
            patch("aegra_api.services.run_executor.dispatch_webhook", AsyncMock(side_effect=Exception("network down"))),
        ):
            mock_streaming.cleanup_run = AsyncMock()
            mock_stream.return_value = MagicMock(has_interrupt=False, data={})

            from aegra_api.services.run_executor import execute_run

            # Must not raise despite webhook failure
            await execute_run(_make_job_with_webhook())

        # Run was still finalized successfully
        mock_finalize.assert_awaited_once()
        assert mock_finalize.await_args.kwargs["status"] == "success"

    @pytest.mark.asyncio
    async def test_webhook_url_passed_to_dispatch_correctly(self) -> None:
        """The correct URL from the job is passed as the first argument to dispatch_webhook."""
        mock_dispatch = AsyncMock()
        url = "https://my-service.example.com/hooks/abc123"

        with (
            patch("aegra_api.services.run_executor.update_run_status", AsyncMock()),
            patch("aegra_api.services.run_executor.finalize_run", AsyncMock()),
            patch("aegra_api.services.run_executor._signal_run_done", AsyncMock()),
            patch("aegra_api.services.run_executor._signal_end_event", AsyncMock()),
            patch("aegra_api.services.run_executor.streaming_service") as mock_streaming,
            patch("aegra_api.services.run_executor._stream_graph", new_callable=AsyncMock) as mock_stream,
            patch("aegra_api.services.run_executor.dispatch_webhook", mock_dispatch),
        ):
            mock_streaming.cleanup_run = AsyncMock()
            mock_stream.return_value = MagicMock(has_interrupt=False, data={})

            from aegra_api.services.run_executor import execute_run

            await execute_run(_make_job_with_webhook(webhook_url=url))

        dispatched_url = mock_dispatch.await_args.args[0]
        assert dispatched_url == url
