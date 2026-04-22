"""Tests for RunJob Pydantic model and serialization."""

import pytest
from pydantic import ValidationError

from aegra_api.models.auth import User
from aegra_api.models.run_job import RunBehavior, RunExecution, RunIdentity, RunJob


class TestRunIdentity:
    def test_frozen(self) -> None:
        identity = RunIdentity(run_id="r1", thread_id="t1", graph_id="g1")
        with pytest.raises(ValidationError):
            identity.run_id = "r2"  # type: ignore[misc]

    def test_fields(self) -> None:
        identity = RunIdentity(run_id="r1", thread_id="t1", graph_id="g1")
        assert identity.run_id == "r1"
        assert identity.thread_id == "t1"
        assert identity.graph_id == "g1"


class TestRunExecution:
    def test_defaults(self) -> None:
        execution = RunExecution()
        assert execution.input_data == {}
        assert execution.config == {}
        assert execution.context == {}
        assert execution.stream_mode is None
        assert execution.checkpoint is None
        assert execution.command is None

    def test_with_values(self) -> None:
        execution = RunExecution(
            input_data={"key": "value"},
            config={"configurable": {}},
            stream_mode=["values", "updates"],
            command={"resume": True},
        )
        assert execution.input_data == {"key": "value"}
        assert execution.stream_mode == ["values", "updates"]
        assert execution.command == {"resume": True}


class TestRunBehavior:
    def test_defaults(self) -> None:
        behavior = RunBehavior()
        assert behavior.interrupt_before is None
        assert behavior.interrupt_after is None
        assert behavior.multitask_strategy is None
        assert behavior.subgraphs is False


class TestRunJob:
    @pytest.fixture()
    def sample_job(self) -> RunJob:
        return RunJob(
            identity=RunIdentity(run_id="run-1", thread_id="thread-1", graph_id="graph-1"),
            user=User(identity="user-1", is_authenticated=True, permissions=["read"]),
            execution=RunExecution(
                input_data={"message": "hello"},
                config={"configurable": {"model": "gpt-4"}},
                context={"tenant": "acme"},
                stream_mode="values",
                checkpoint={"thread_ts": "123"},
                command=None,
            ),
            behavior=RunBehavior(
                interrupt_before=["review"],
                interrupt_after=None,
                subgraphs=True,
            ),
        )

    def test_roundtrip_model_dump_validate(self, sample_job: RunJob) -> None:
        """model_dump -> model_validate produces identical job."""
        data = sample_job.model_dump()
        restored = RunJob.model_validate(data)
        assert restored == sample_job

    def test_execution_params_roundtrip(self, sample_job: RunJob) -> None:
        """to_execution_params -> from_run_orm produces identical job."""

        class FakeORM:
            run_id = "run-1"
            thread_id = "thread-1"
            execution_params = sample_job.to_execution_params()

        restored = RunJob.from_run_orm(FakeORM())
        assert restored.identity == sample_job.identity
        assert restored.user.identity == sample_job.user.identity
        assert restored.execution == sample_job.execution
        assert restored.behavior == sample_job.behavior

    def test_execution_params_includes_graph_id(self, sample_job: RunJob) -> None:
        params = sample_job.to_execution_params()
        assert params["graph_id"] == "graph-1"
        assert "run_id" not in params
        assert "thread_id" not in params

    def test_extra_user_fields_preserved(self) -> None:
        """User model allows extra fields (ConfigDict extra='allow')."""
        job = RunJob(
            identity=RunIdentity(run_id="r1", thread_id="t1", graph_id="g1"),
            user=User(identity="u1", is_authenticated=True, permissions=[], team_id="team-42"),
        )
        params = job.to_execution_params()
        assert params["user"]["team_id"] == "team-42"

        class FakeORM:
            run_id = "r1"
            thread_id = "t1"
            execution_params = params

        restored = RunJob.from_run_orm(FakeORM())
        assert restored.user.model_extra.get("team_id") == "team-42"

    def test_defaults_when_optional_fields_missing(self) -> None:
        """RunJob with minimal required fields."""
        job = RunJob(
            identity=RunIdentity(run_id="r1", thread_id="t1", graph_id="g1"),
            user=User(identity="u1"),
        )
        assert job.execution.input_data == {}
        assert job.behavior.subgraphs is False


class TestRunExecutionWebhookUrl:
    """Tests that webhook_url is properly handled by RunExecution and RunJob serialization."""

    def test_webhook_url_defaults_to_none(self) -> None:
        execution = RunExecution()
        assert execution.webhook_url is None

    def test_webhook_url_accepted_as_string(self) -> None:
        execution = RunExecution(webhook_url="https://example.com/hook")
        assert execution.webhook_url == "https://example.com/hook"

    def test_webhook_url_serialized_in_to_execution_params(self) -> None:
        """webhook_url appears in execution_params so workers can retrieve it."""
        job = RunJob(
            identity=RunIdentity(run_id="r1", thread_id="t1", graph_id="g1"),
            user=User(identity="u1"),
            execution=RunExecution(webhook_url="https://example.com/hook"),
        )
        params = job.to_execution_params()
        assert params["execution"]["webhook_url"] == "https://example.com/hook"

    def test_webhook_url_none_serialized_as_none(self) -> None:
        """webhook_url=None is explicitly stored (Pydantic model_dump behaviour)."""
        job = RunJob(
            identity=RunIdentity(run_id="r1", thread_id="t1", graph_id="g1"),
            user=User(identity="u1"),
            execution=RunExecution(webhook_url=None),
        )
        params = job.to_execution_params()
        assert "webhook_url" in params["execution"]
        assert params["execution"]["webhook_url"] is None

    def test_webhook_url_round_trips_through_from_run_orm(self) -> None:
        """webhook_url survives to_execution_params → from_run_orm round trip."""
        job = RunJob(
            identity=RunIdentity(run_id="r1", thread_id="t1", graph_id="g1"),
            user=User(identity="u1"),
            execution=RunExecution(
                input_data={"msg": "hello"},
                webhook_url="https://hooks.example.com/run-done",
            ),
        )
        _params = job.to_execution_params()

        class FakeORM:
            run_id = "r1"
            thread_id = "t1"
            execution_params = _params

        restored = RunJob.from_run_orm(FakeORM())
        assert restored.execution.webhook_url == "https://hooks.example.com/run-done"

    def test_webhook_url_none_round_trips_through_from_run_orm(self) -> None:
        """webhook_url=None survives the serialization round trip."""
        job = RunJob(
            identity=RunIdentity(run_id="r2", thread_id="t2", graph_id="g2"),
            user=User(identity="u2"),
            execution=RunExecution(webhook_url=None),
        )
        params = job.to_execution_params()

        class FakeORM:
            run_id = "r2"
            thread_id = "t2"
            execution_params = params

        restored = RunJob.from_run_orm(FakeORM())
        assert restored.execution.webhook_url is None

    def test_old_execution_params_without_webhook_url_loads_as_none(self) -> None:
        """Execution params stored before webhook support (no webhook_url key) load cleanly."""

        class FakeORM:
            run_id = "r3"
            thread_id = "t3"
            execution_params = {
                "graph_id": "g3",
                "user": {"identity": "u3"},
                "execution": {
                    "input_data": {"msg": "legacy"},
                    "config": {},
                    "context": {},
                    "stream_mode": None,
                    "checkpoint": None,
                    "command": None,
                    # no webhook_url — old record
                },
                "behavior": {
                    "interrupt_before": None,
                    "interrupt_after": None,
                    "multitask_strategy": None,
                    "subgraphs": False,
                },
            }

        restored = RunJob.from_run_orm(FakeORM())
        assert restored.execution.webhook_url is None
