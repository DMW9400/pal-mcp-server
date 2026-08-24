"""Tests for the clink async contract: job store, background runs and clink_poll."""

import asyncio
import json
from datetime import datetime, timedelta, timezone

import pytest

from clink import jobs
from clink.agents import AgentOutput
from clink.agents.base import BaseCLIAgent
from clink.parsers.base import ParsedCLIResponse
from tools.clink import CLinkTool
from tools.clink_poll import MAX_WAIT_SECONDS, CLinkPollTool


@pytest.fixture(autouse=True)
def job_store(tmp_path, monkeypatch):
    """Keep every job record inside the test's tmp dir."""
    root = tmp_path / "clink_results"
    jobs.set_store_root(root)
    monkeypatch.setattr(jobs, "_CLAIMS", {}, raising=False)
    yield root
    jobs.set_store_root(None)


def _agent_output(content: str = "done", *, duration: float = 0.1) -> AgentOutput:
    return AgentOutput(
        parsed=ParsedCLIResponse(content=content, metadata={"model_used": "gpt-5.6-sol"}),
        sanitized_command=["codex"],
        returncode=0,
        stdout="{}",
        stderr="",
        duration_seconds=duration,
        parser_name="codex_jsonl",
        output_file_content=None,
    )


def _install_agent(monkeypatch, agent):
    monkeypatch.setattr("tools.clink.create_agent", lambda client: agent)


class _DummyAgent:
    def __init__(self, output=None, error=None):
        self._output = output if output is not None else _agent_output()
        self._error = error

    async def run(self, **kwargs):
        if self._error is not None:
            raise self._error
        return self._output


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


def test_clink_background_defaults_to_false_and_is_exposed():
    tool = CLinkTool()
    schema = tool.get_input_schema()

    assert "background" in schema["properties"]
    assert schema["properties"]["background"]["type"] == "boolean"
    assert "background" not in schema["required"]
    assert "clink_poll" in schema["properties"]["background"]["description"]

    request = tool.get_request_model()(prompt="hi")
    assert bool(request.background) is False


def test_clink_poll_schema_rejects_conversation_fields():
    schema = CLinkPollTool().get_input_schema()

    assert set(schema["properties"]) == {"run_id", "wait_seconds"}
    assert "prompt" not in schema["properties"]
    assert "continuation_id" not in schema["properties"]
    assert schema["additionalProperties"] is False
    assert schema["required"] == ["run_id"]
    assert schema["properties"]["wait_seconds"]["maximum"] == MAX_WAIT_SECONDS


@pytest.mark.asyncio
async def test_clink_poll_rejects_path_like_run_id():
    tool = CLinkPollTool()

    for bad in ["../../etc/passwd", "not-a-uuid", "9f6c0f1e-0000", ""]:
        result = await tool.execute({"run_id": bad})
        payload = json.loads(result[0].text)
        assert payload["status"] == "error"
        assert "run_id" in payload["content"]


# ---------------------------------------------------------------------------
# Job store
# ---------------------------------------------------------------------------


def test_job_store_create_read_and_terminal_immutability(job_store):
    run_id = jobs.new_run_id()
    jobs.create(run_id=run_id, continuation_id="thread-1", cli_name="codex", role="default")

    record = jobs.read(run_id)
    assert record["status"] == jobs.STATUS_QUEUED
    assert record["schema_version"] == jobs.SCHEMA_VERSION
    assert record["attachment"] == jobs.ATTACHMENT_ATTACHED
    assert (job_store / f"{run_id}.json").exists()

    jobs.mark_running(run_id)
    assert jobs.read(run_id)["status"] == jobs.STATUS_RUNNING

    # Attachment changes are allowed while the run is live.
    jobs.set_attachment(run_id, jobs.ATTACHMENT_DETACHED)
    assert jobs.read(run_id)["attachment"] == jobs.ATTACHMENT_DETACHED
    assert jobs.read(run_id)["status"] == jobs.STATUS_RUNNING

    jobs.finalize(run_id, status=jobs.STATUS_COMPLETED, result='{"content": "first"}', duration_seconds=1.0)
    first = jobs.read(run_id)
    assert first["status"] == jobs.STATUS_COMPLETED
    assert first["output_sha256"]


def test_terminal_records_are_byte_for_byte_immutable(job_store):
    run_id = jobs.new_run_id()
    jobs.create(run_id=run_id, continuation_id=None, cli_name="codex", role="default")
    jobs.mark_running(run_id)
    jobs.finalize(run_id, status=jobs.STATUS_COMPLETED, result='{"content": "first"}', duration_seconds=1.0)

    path = job_store / f"{run_id}.json"
    before = path.read_bytes()

    jobs.mark_running(run_id)
    jobs.touch(run_id)
    jobs.set_attachment(run_id, jobs.ATTACHMENT_DETACHED)
    jobs.finalize(run_id, status=jobs.STATUS_FAILED, result='{"content": "second"}')

    assert path.read_bytes() == before


@pytest.mark.asyncio
async def test_late_cancellation_cannot_rewrite_a_finished_record(job_store):
    tool = CLinkTool()
    run_id = jobs.new_run_id()
    jobs.create(run_id=run_id, continuation_id="thread-1", cli_name="codex", role="default")
    jobs.mark_running(run_id)
    jobs.finalize(run_id, status=jobs.STATUS_COMPLETED, result='{"content": "done"}', duration_seconds=0.5)

    path = job_store / f"{run_id}.json"
    before = path.read_bytes()

    # Cancellation arriving after completion must not rewrite the audit record.
    tool._detach_cancelled_run("codex", "thread-1", run_id)

    assert path.read_bytes() == before
    assert jobs.read(run_id)["attachment"] == jobs.ATTACHMENT_ATTACHED


def test_job_store_validates_run_ids():
    with pytest.raises(jobs.JobStoreError):
        jobs.validate_run_id("../secrets")
    with pytest.raises(jobs.JobStoreError):
        jobs.validate_run_id("9f6c0f1e")
    with pytest.raises(jobs.JobStoreError):
        jobs.validate_run_id(None)

    run_id = jobs.new_run_id()
    assert jobs.validate_run_id(run_id) == run_id


def test_job_store_sweep_only_removes_old_terminal_records(job_store):
    old_terminal = jobs.new_run_id()
    fresh_terminal = jobs.new_run_id()
    old_running = jobs.new_run_id()

    for run_id in (old_terminal, fresh_terminal, old_running):
        jobs.create(run_id=run_id, continuation_id=None, cli_name="codex", role="default")

    jobs.finalize(old_terminal, status=jobs.STATUS_COMPLETED, result="{}")
    jobs.finalize(fresh_terminal, status=jobs.STATUS_COMPLETED, result="{}")
    jobs.mark_running(old_running)

    stale = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
    for run_id in (old_terminal, old_running):
        path = job_store / f"{run_id}.json"
        record = json.loads(path.read_text())
        record["finished_at"] = stale
        record["updated_at"] = stale
        path.write_text(json.dumps(record))

    removed = jobs.sweep()

    assert removed == 1
    assert jobs.read(old_terminal) is None
    assert jobs.read(fresh_terminal) is not None
    assert jobs.read(old_running) is not None


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sync_success_preserves_payload_and_writes_completed_record(monkeypatch, job_store):
    tool = CLinkTool()
    _install_agent(monkeypatch, _DummyAgent(_agent_output("Sync answer")))

    result = await tool.execute({"prompt": "Ping", "cli_name": "codex", "absolute_file_paths": [], "images": []})
    payload = json.loads(result[0].text)

    assert payload["status"] == "continuation_available"
    assert payload["content"] == "Sync answer"
    assert payload["continuation_offer"]["continuation_id"]
    assert payload["metadata"]["cli_name"] == "codex"

    records = [json.loads(path.read_text()) for path in job_store.glob("*.json")]
    assert len(records) == 1
    record = records[0]
    assert record["status"] == jobs.STATUS_COMPLETED
    assert record["attachment"] == jobs.ATTACHMENT_ATTACHED
    assert record["error"] is None
    # The stored result is byte-identical to what the synchronous caller received.
    assert record["result"] == result[0].text


@pytest.mark.asyncio
async def test_background_returns_identifiers_before_the_run_finishes(monkeypatch, job_store):
    tool = CLinkTool()

    started = asyncio.Event()
    release = asyncio.Event()

    class SlowAgent:
        async def run(self, **kwargs):
            started.set()
            await release.wait()
            return _agent_output("Late answer")

    _install_agent(monkeypatch, SlowAgent())

    result = await tool.execute(
        {
            "prompt": "Long question",
            "cli_name": "codex",
            "background": True,
            "absolute_file_paths": [],
            "images": [],
        }
    )
    payload = json.loads(result[0].text)
    assert payload["status"] == "clink_background_started"

    body = json.loads(payload["content"])
    run_id = body["run_id"]
    assert body["continuation_id"]
    assert body["poll_with"]["tool"] == "clink_poll"
    assert payload["continuation_offer"]["continuation_id"] == body["continuation_id"]

    await started.wait()
    assert jobs.read(run_id)["status"] == jobs.STATUS_RUNNING

    poll = CLinkPollTool()
    running = json.loads(json.loads((await poll.execute({"run_id": run_id}))[0].text)["content"])
    assert running["status"] == jobs.STATUS_RUNNING
    assert running["stale_heartbeat"] is False

    release.set()
    for task in list(tool._background_tasks):
        await task

    finished = json.loads(json.loads((await poll.execute({"run_id": run_id}))[0].text)["content"])
    assert finished["status"] == jobs.STATUS_COMPLETED
    assert json.loads(finished["result"])["content"] == "Late answer"

    # Polling is idempotent and consumes no turns.
    again = json.loads(json.loads((await poll.execute({"run_id": run_id}))[0].text)["content"])
    assert again["result"] == finished["result"]


@pytest.mark.asyncio
async def test_cancellation_detaches_but_the_pipeline_completes(monkeypatch, job_store, caplog):
    import utils.conversation_memory as conversation_memory

    tool = CLinkTool()

    started = asyncio.Event()
    release = asyncio.Event()

    class SlowAgent:
        async def run(self, **kwargs):
            started.set()
            await release.wait()
            return _agent_output("Detached answer")

    _install_agent(monkeypatch, SlowAgent())

    caplog.set_level("INFO")
    exec_task = asyncio.create_task(
        tool.execute({"prompt": "Long question", "cli_name": "codex", "absolute_file_paths": [], "images": []})
    )
    await started.wait()
    exec_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await exec_task

    run_id = next(iter(job_store.glob("*.json"))).stem
    detached = jobs.read(run_id)
    assert detached["attachment"] == jobs.ATTACHMENT_DETACHED
    assert detached["status"] == jobs.STATUS_RUNNING

    release.set()
    for task in list(tool._background_tasks):
        await task

    record = jobs.read(run_id)
    assert record["status"] == jobs.STATUS_COMPLETED
    assert record["attachment"] == jobs.ATTACHMENT_DETACHED
    assert json.loads(record["result"])["content"] == "Detached answer"

    context = conversation_memory.get_thread(record["continuation_id"])
    assert [turn.role for turn in context.turns] == ["user", "assistant"]

    assert "CLINK_CANCELLED" in caplog.text
    assert "CLINK_SALVAGED" in caplog.text


@pytest.mark.asyncio
async def test_agent_failure_records_bounded_error(monkeypatch, job_store):
    from clink.agents import CLIAgentError
    from tools.shared.exceptions import ToolExecutionError

    tool = CLinkTool()
    _install_agent(monkeypatch, _DummyAgent(error=CLIAgentError("boom " * 2000)))

    with pytest.raises(ToolExecutionError):
        await tool.execute({"prompt": "Ping", "cli_name": "codex", "absolute_file_paths": [], "images": []})

    record = json.loads(next(iter(job_store.glob("*.json"))).read_text())
    assert record["status"] == jobs.STATUS_FAILED
    assert record["error"]["category"] == "cli_error"
    assert len(record["error"]["message"]) <= jobs.MAX_ERROR_MESSAGE_CHARS
    assert json.loads(record["result"])["status"] == "error"


@pytest.mark.asyncio
async def test_conversation_write_failure_still_finalizes_the_record(monkeypatch, job_store):
    tool = CLinkTool()
    _install_agent(monkeypatch, _DummyAgent(_agent_output("Answer despite storage failure")))

    def exploding_record(*args, **kwargs):
        raise RuntimeError("conversation storage unavailable")

    monkeypatch.setattr(CLinkTool, "_record_assistant_turn", exploding_record)

    result = await tool.execute({"prompt": "Ping", "cli_name": "codex", "absolute_file_paths": [], "images": []})
    payload = json.loads(result[0].text)
    assert payload["content"] == "Answer despite storage failure"

    record = json.loads(next(iter(job_store.glob("*.json"))).read_text())
    assert record["status"] == jobs.STATUS_COMPLETED
    assert record["error"]["category"] == "conversation_write"


@pytest.mark.asyncio
async def test_same_continuation_contention_is_rejected_without_launching_a_cli(monkeypatch, job_store):
    tool = CLinkTool()

    started = asyncio.Event()
    release = asyncio.Event()
    launches = {"count": 0}

    class SlowAgent:
        async def run(self, **kwargs):
            launches["count"] += 1
            started.set()
            await release.wait()
            return _agent_output("First answer")

    _install_agent(monkeypatch, SlowAgent())

    first = await tool.execute(
        {
            "prompt": "First",
            "cli_name": "codex",
            "background": True,
            "absolute_file_paths": [],
            "images": [],
        }
    )
    body = json.loads(json.loads(first[0].text)["content"])
    continuation_id = body["continuation_id"]
    await started.wait()

    second = await tool.execute(
        {
            "prompt": "Second",
            "cli_name": "codex",
            "continuation_id": continuation_id,
            "absolute_file_paths": [],
            "images": [],
        }
    )
    payload = json.loads(second[0].text)
    assert payload["status"] == "error"
    assert body["run_id"] in payload["content"]
    assert payload["metadata"]["active_run_id"] == body["run_id"]
    assert launches["count"] == 1

    release.set()
    for task in list(tool._background_tasks):
        await task


@pytest.mark.asyncio
async def test_poll_reports_unknown_run_and_stale_run(job_store):
    poll = CLinkPollTool()

    missing = json.loads((await poll.execute({"run_id": jobs.new_run_id()}))[0].text)
    assert missing["status"] == "error"
    assert "retention" in missing["content"]

    run_id = jobs.new_run_id()
    jobs.create(run_id=run_id, continuation_id=None, cli_name="codex", role="default")
    jobs.mark_running(run_id)

    path = job_store / f"{run_id}.json"
    record = json.loads(path.read_text())
    record["updated_at"] = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    path.write_text(json.dumps(record))

    payload = json.loads(json.loads((await poll.execute({"run_id": run_id}))[0].text)["content"])
    assert payload["stale_heartbeat"] is True
    assert payload["reported_status"] == jobs.STATUS_INTERRUPTED
    # Classification is read-only: the stored record is untouched.
    assert jobs.read(run_id)["status"] == jobs.STATUS_RUNNING


# ---------------------------------------------------------------------------
# Durable-store invariants
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_background_refuses_to_launch_without_a_durable_record(monkeypatch, job_store):
    from tools.shared.exceptions import ToolExecutionError

    tool = CLinkTool()
    launches = {"count": 0}

    class CountingAgent:
        async def run(self, **kwargs):  # pragma: no cover - must never run
            launches["count"] += 1
            return _agent_output()

    _install_agent(monkeypatch, CountingAgent())

    def boom(**kwargs):
        raise OSError("read-only file system")

    monkeypatch.setattr(jobs, "create", boom)

    continuation_id = "11111111-2222-3333-4444-555555555555"
    with pytest.raises(ToolExecutionError) as exc_info:
        await tool.execute(
            {
                "prompt": "Long question",
                "cli_name": "codex",
                "background": True,
                "continuation_id": continuation_id,
                "absolute_file_paths": [],
                "images": [],
            }
        )

    payload = json.loads(str(exc_info.value))
    assert payload["status"] == "error"
    assert payload["metadata"]["error_category"] == "durable_store"
    assert launches["count"] == 0

    # The claim was released, so the continuation is not blocked for the process lifetime.
    follow_up = jobs.new_run_id()
    assert jobs.claim(continuation_id, follow_up) == follow_up


@pytest.mark.asyncio
async def test_sync_proceeds_without_a_durable_record_but_says_so(monkeypatch, job_store, caplog):
    tool = CLinkTool()
    _install_agent(monkeypatch, _DummyAgent(_agent_output("In-band answer")))

    def boom(**kwargs):
        raise OSError("read-only file system")

    monkeypatch.setattr(jobs, "create", boom)
    caplog.set_level("INFO")

    result = await tool.execute({"prompt": "Ping", "cli_name": "codex", "absolute_file_paths": [], "images": []})
    payload = json.loads(result[0].text)

    assert payload["content"] == "In-band answer"
    assert payload["metadata"]["durable_record"] == "unavailable"
    assert list(job_store.glob("*.json")) == []
    assert "durable=false" in caplog.text


@pytest.mark.asyncio
async def test_mark_running_failure_also_refuses_background(monkeypatch, job_store):
    from tools.shared.exceptions import ToolExecutionError

    tool = CLinkTool()
    _install_agent(monkeypatch, _DummyAgent())
    monkeypatch.setattr(jobs, "mark_running", lambda run_id: None)

    with pytest.raises(ToolExecutionError) as exc_info:
        await tool.execute(
            {
                "prompt": "Long question",
                "cli_name": "codex",
                "background": True,
                "absolute_file_paths": [],
                "images": [],
            }
        )

    assert json.loads(str(exc_info.value))["metadata"]["error_category"] == "durable_store"


@pytest.mark.asyncio
async def test_formatter_exception_terminalizes_the_record(monkeypatch, job_store):
    from tools.shared.exceptions import ToolExecutionError

    tool = CLinkTool()
    _install_agent(monkeypatch, _DummyAgent(_agent_output("Answer")))

    def exploding_limit(self, *args, **kwargs):
        raise RuntimeError("formatter blew up")

    monkeypatch.setattr(CLinkTool, "_apply_output_limit", exploding_limit)

    with pytest.raises(ToolExecutionError):
        await tool.execute({"prompt": "Ping", "cli_name": "codex", "absolute_file_paths": [], "images": []})

    record = json.loads(next(iter(job_store.glob("*.json"))).read_text())
    assert record["status"] == jobs.STATUS_FAILED
    assert record["error"]["category"] == "internal_error"


@pytest.mark.asyncio
async def test_finalize_failure_still_returns_the_answer_and_logs_durable_false(monkeypatch, job_store, caplog):
    tool = CLinkTool()
    _install_agent(monkeypatch, _DummyAgent(_agent_output("Answer anyway")))

    def boom(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(jobs, "finalize", boom)
    caplog.set_level("INFO")

    result = await tool.execute({"prompt": "Ping", "cli_name": "codex", "absolute_file_paths": [], "images": []})

    assert json.loads(result[0].text)["content"] == "Answer anyway"
    completed = [line for line in caplog.text.splitlines() if "CLINK_COMPLETED" in line]
    assert completed and all("durable=false" in line for line in completed)


@pytest.mark.asyncio
async def test_identity_notification_cancellation_does_not_kill_the_run(monkeypatch, job_store):
    import sys
    import types

    tool = CLinkTool()
    _install_agent(monkeypatch, _DummyAgent(_agent_output("Survived")))

    class FakeSession:
        async def send_progress_notification(self, **kwargs):
            raise asyncio.CancelledError()

    class FakeMeta:
        progressToken = "tok-cancel"

    class FakeRequestContext:
        meta = FakeMeta()
        session = FakeSession()

    class FakeServer:
        request_context = FakeRequestContext()

    fake_server_module = types.ModuleType("server")
    fake_server_module.server = FakeServer()
    monkeypatch.setitem(sys.modules, "server", fake_server_module)

    result = await tool.execute({"prompt": "Ping", "cli_name": "codex", "absolute_file_paths": [], "images": []})
    payload = json.loads(result[0].text)
    assert payload["content"] == "Survived"

    record = json.loads(next(iter(job_store.glob("*.json"))).read_text())
    assert record["status"] == jobs.STATUS_COMPLETED
    # The claim was released by the pipeline's finally.
    follow_up = jobs.new_run_id()
    assert jobs.claim(record["continuation_id"], follow_up) == follow_up


# ---------------------------------------------------------------------------
# Durable payload sanitization and golden sync payload
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sidecar_redacts_diagnostic_streams(monkeypatch, job_store):
    import hashlib

    tool = CLinkTool()

    noisy_stderr = "SECRET-TOKEN " * 5000
    noisy_file = "raw output file blob " * 5000
    output = AgentOutput(
        parsed=ParsedCLIResponse(content="Clean answer", metadata={"model_used": "gpt-5.6-sol"}),
        sanitized_command=["codex"],
        returncode=0,
        stdout="{}",
        stderr=noisy_stderr,
        duration_seconds=0.1,
        parser_name="codex_jsonl",
        output_file_content=noisy_file,
    )
    _install_agent(monkeypatch, _DummyAgent(output))

    result = await tool.execute({"prompt": "Ping", "cli_name": "codex", "absolute_file_paths": [], "images": []})
    payload = json.loads(result[0].text)

    # The in-band response is unchanged: it still carries the diagnostics.
    assert payload["metadata"]["stderr"].startswith("SECRET-TOKEN")
    assert payload["metadata"]["raw_output_file"].startswith("raw output file blob")

    record = json.loads(next(iter(job_store.glob("*.json"))).read_text())
    stored = json.loads(record["result"])
    assert stored["metadata"]["stderr"] == "<omitted from durable record>"
    assert stored["metadata"]["raw_output_file"] == "<omitted from durable record>"
    assert "SECRET-TOKEN" not in record["result"]
    assert stored["content"] == "Clean answer"
    # The hash describes exactly what a poll replays.
    assert record["output_sha256"] == hashlib.sha256(record["result"].encode("utf-8")).hexdigest()


@pytest.mark.asyncio
async def test_sidecar_redacts_raw_parser_payloads(monkeypatch, job_store):
    """claude/gemini parsers put whole CLI responses under metadata.raw/raw_events."""
    tool = CLinkTool()

    raw_blob = {"result": "SECRET-RAW " * 5000, "session_id": "abc"}
    raw_events = ["SECRET-EVENT " * 2000, "SECRET-EVENT " * 2000]
    output = AgentOutput(
        parsed=ParsedCLIResponse(
            content="Clean answer",
            metadata={"model_used": "claude-fable-5", "raw": raw_blob, "raw_events": raw_events},
        ),
        sanitized_command=["claude"],
        returncode=0,
        stdout="{}",
        stderr="",
        duration_seconds=0.1,
        parser_name="claude_json",
        output_file_content=None,
    )
    _install_agent(monkeypatch, _DummyAgent(output))

    result = await tool.execute({"prompt": "Ping", "cli_name": "claude", "absolute_file_paths": [], "images": []})
    payload = json.loads(result[0].text)

    # In-band callers still receive the raw parser payloads unchanged.
    assert payload["metadata"]["raw"] == raw_blob
    assert payload["metadata"]["raw_events"] == raw_events

    record = json.loads(next(iter(job_store.glob("*.json"))).read_text())
    stored = json.loads(record["result"])
    assert stored["metadata"]["raw"] == "<omitted from durable record>"
    assert stored["metadata"]["raw_events"] == "<omitted from durable record>"
    assert "SECRET" not in record["result"]
    assert stored["content"] == "Clean answer"
    assert stored["metadata"]["model_used"] == "claude-fable-5"


def test_sidecar_denylist_covers_every_raw_payload_key():
    from tools.clink import SIDECAR_REDACTED_METADATA_KEYS

    assert {"stdout", "stderr", "output_file_content", "raw_output_file", "raw", "raw_events", "events"} <= set(
        SIDECAR_REDACTED_METADATA_KEYS
    )


@pytest.mark.asyncio
async def test_failure_sidecar_redacts_diagnostic_streams(monkeypatch, job_store):
    from clink.agents import CLIAgentError
    from tools.shared.exceptions import ToolExecutionError

    tool = CLinkTool()
    error = CLIAgentError("exited with status 1", returncode=1, stdout="STDOUT-SECRET " * 2000, stderr="STDERR-SECRET")
    _install_agent(monkeypatch, _DummyAgent(error=error))

    with pytest.raises(ToolExecutionError) as exc_info:
        await tool.execute({"prompt": "Ping", "cli_name": "codex", "absolute_file_paths": [], "images": []})

    in_band = json.loads(str(exc_info.value))
    assert in_band["metadata"]["stdout"].startswith("STDOUT-SECRET")

    record = json.loads(next(iter(job_store.glob("*.json"))).read_text())
    stored = json.loads(record["result"])
    assert stored["metadata"]["stdout"] == "<omitted from durable record>"
    assert stored["metadata"]["stderr"] == "<omitted from durable record>"
    assert "SECRET" not in record["result"]


@pytest.mark.asyncio
async def test_sync_payload_matches_the_pre_patch_golden(monkeypatch, job_store):
    """The in-band response must be exactly what the pre-refactor tool returned."""
    from utils.conversation_memory import MAX_CONVERSATION_TURNS

    tool = CLinkTool()
    # Neutralise the host's local CLI config so the golden is host-independent.
    monkeypatch.setattr(CLinkTool, "_configured_partner_model", staticmethod(lambda client: None))

    output = AgentOutput(
        parsed=ParsedCLIResponse(content="Golden answer", metadata={"model_used": "gpt-5.6-sol"}),
        sanitized_command=["codex"],
        returncode=0,
        stdout="{}",
        stderr="warning: rate limited",
        duration_seconds=0.42,
        parser_name="codex_jsonl",
        output_file_content=None,
    )
    _install_agent(monkeypatch, _DummyAgent(output))

    result = await tool.execute({"prompt": "Ping", "cli_name": "codex", "absolute_file_paths": [], "images": []})
    payload = json.loads(result[0].text)

    remaining_turns = MAX_CONVERSATION_TURNS - 2
    expected = {
        "status": "continuation_available",
        "content": "Golden answer",
        "content_type": "text",
        "metadata": {
            "tool_name": "clink",
            "conversation_ready": True,
            "model_used": "gpt-5.6-sol",
            "provider_used": "codex",
            "cli_name": "codex",
            "role": "default",
            "command": ["codex"],
            "duration_seconds": 0.42,
            "parser": "codex_jsonl",
            "return_code": 0,
            "stderr": "warning: rate limited",
        },
        "continuation_offer": {
            "continuation_id": payload["continuation_offer"]["continuation_id"],
            "note": f"You can continue this conversation for {remaining_turns} more exchanges.",
            "remaining_turns": remaining_turns,
        },
    }
    assert payload == expected


# ---------------------------------------------------------------------------
# Poll details
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_poll_wait_expiry_returns_running(job_store):
    poll = CLinkPollTool()
    run_id = jobs.new_run_id()
    jobs.create(run_id=run_id, continuation_id=None, cli_name="codex", role="default")
    jobs.mark_running(run_id)

    result = await poll.execute({"run_id": run_id, "wait_seconds": 1})
    envelope = json.loads(result[0].text)
    payload = json.loads(envelope["content"])

    assert payload["status"] == jobs.STATUS_RUNNING
    assert envelope["metadata"]["run_status"] == jobs.STATUS_RUNNING
    assert envelope["metadata"]["stored_status"] == jobs.STATUS_RUNNING


@pytest.mark.asyncio
async def test_poll_reports_interrupted_status_in_metadata_and_logs(job_store, caplog):
    from pydantic import ValidationError

    from tools.clink_poll import CLinkPollRequest

    poll = CLinkPollTool()
    run_id = jobs.new_run_id()
    jobs.create(run_id=run_id, continuation_id=None, cli_name="codex", role="default")
    jobs.mark_running(run_id)

    path = job_store / f"{run_id}.json"
    record = json.loads(path.read_text())
    record["updated_at"] = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    path.write_text(json.dumps(record))

    caplog.set_level("INFO")
    envelope = json.loads((await poll.execute({"run_id": run_id}))[0].text)

    assert envelope["metadata"]["run_status"] == jobs.STATUS_INTERRUPTED
    assert envelope["metadata"]["stored_status"] == jobs.STATUS_RUNNING
    assert "CLINK_INTERRUPTED (observed)" in caplog.text

    # Defense in depth: conversation fields are rejected even on a direct call.
    with pytest.raises(ValidationError):
        CLinkPollRequest(run_id=run_id, continuation_id="abc")
    with pytest.raises(ValidationError):
        CLinkPollRequest(run_id=run_id, prompt="hi")


# ---------------------------------------------------------------------------
# Silence heartbeat
# ---------------------------------------------------------------------------


class _SilentReader:
    """Emits one line only after a delay, then EOF."""

    def __init__(self, delay: float, payload: bytes):
        self._delay = delay
        self._lines = [payload]

    async def readline(self) -> bytes:
        await asyncio.sleep(self._delay)
        if not self._lines:
            return b""
        return self._lines.pop(0)

    async def read(self, _n: int = -1) -> bytes:  # pragma: no cover - fallback path
        return b""


class _ChattyReader:
    def __init__(self, lines: list[bytes]):
        self._lines = list(lines)

    async def readline(self) -> bytes:
        await asyncio.sleep(0)
        if not self._lines:
            return b""
        return self._lines.pop(0)

    async def read(self, _n: int = -1) -> bytes:  # pragma: no cover - fallback path
        return b""


class _EmptyReader(_ChattyReader):
    def __init__(self):
        super().__init__([])


class _DummyStdin:
    def write(self, data: bytes) -> None:
        self.data = data

    async def drain(self) -> None:
        return None

    def close(self) -> None:
        return None


class _DummyProcess:
    def __init__(self, stdout, stderr):
        self.stdout = stdout
        self.stderr = stderr
        self.stdin = _DummyStdin()
        self.returncode = 0

    async def wait(self) -> int:
        return self.returncode


def _agent_for_heartbeat(interval: float) -> BaseCLIAgent:
    from pathlib import Path

    from clink.models import ResolvedCLIClient, ResolvedCLIRole

    role = ResolvedCLIRole(
        name="default",
        prompt_path=Path("systemprompts/clink/default.txt").resolve(),
        role_args=[],
    )
    client = ResolvedCLIClient(
        name="codex",
        executable=["codex"],
        internal_args=["exec"],
        config_args=["--json"],
        env={},
        timeout_seconds=30,
        parser="codex_jsonl",
        roles={"default": role},
        output_to_file=None,
        working_dir=None,
    )
    agent = BaseCLIAgent(client)
    agent.heartbeat_interval_seconds = interval
    return agent


@pytest.mark.asyncio
async def test_silent_run_emits_heartbeats_and_chatty_run_does_not():
    events: list[tuple[str, str]] = []

    async def on_event(kind, text):
        events.append((kind, text))

    silent_agent = _agent_for_heartbeat(0.05)
    process = _DummyProcess(_SilentReader(0.3, b"finally\n"), _EmptyReader())
    stdout, stderr = await silent_agent._stream_subprocess(process, b"ping", on_event)

    heartbeats = [event for event in events if event[0] == "heartbeat"]
    assert heartbeats, "a silent run must produce heartbeat events"
    assert all("still running" in text for _kind, text in heartbeats)
    # Heartbeats never pollute the captured CLI output.
    assert stdout == "finally\n"
    assert stderr == ""

    events.clear()
    chatty_agent = _agent_for_heartbeat(1.0)
    chatty_process = _DummyProcess(_ChattyReader([b"a\n", b"b\n", b"c\n"]), _EmptyReader())
    await chatty_agent._stream_subprocess(chatty_process, b"ping", on_event)

    assert [kind for kind, _text in events] == ["stdout", "stdout", "stdout"]

    # No heartbeat task outlives the run.
    pending = [task for task in asyncio.all_tasks() if task is not asyncio.current_task()]
    assert pending == []
