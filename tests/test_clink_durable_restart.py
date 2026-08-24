import asyncio
import json

import pytest

import server
from clink.agents import AgentOutput
from clink.parsers.base import ParsedCLIResponse
from utils.conversation_memory import get_thread
from utils.sqlite_conversation_storage import get_default_storage, reset_default_storage_for_tests


class _Agent:
    calls = 0

    async def run(self, **kwargs):
        self.__class__.calls += 1
        return AgentOutput(
            parsed=ParsedCLIResponse(
                content=f"durable answer {self.calls}",
                metadata={"model_used": "gpt-5.6-sol", "reasoning_effort_used": "high"},
            ),
            sanitized_command=[
                "codex",
                "exec",
                "--model",
                "gpt-5.6-sol",
                "-c",
                'model_reasoning_effort="high"',
            ],
            returncode=0,
            stdout="{}",
            stderr="",
            duration_seconds=0.01,
            parser_name="codex_jsonl",
            output_file_content=None,
        )


@pytest.mark.asyncio
async def test_completed_clink_thread_survives_pal_store_reopen(tmp_path, monkeypatch):
    monkeypatch.setenv("PAL_CONVERSATION_BACKEND", "sqlite")
    monkeypatch.setenv("PAL_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("PAL_STATE_KEY_FILE", str(tmp_path / "keys" / "state.key"))
    monkeypatch.setattr("tools.clink.create_agent", lambda client: _Agent())
    reset_default_storage_for_tests()
    _Agent.calls = 0

    first = await server.handle_call_tool(
        "clink",
        {
            "prompt": "first durable prompt",
            "cli_name": "codex",
            "role": "default",
            "absolute_file_paths": [],
            "images": [],
            "idempotency_key": "restart-test-request-1",
        },
    )
    first_payload = json.loads(first[0].text)
    continuation_id = first_payload["continuation_offer"]["continuation_id"]

    replay = await server.handle_call_tool(
        "clink",
        {
            "prompt": "first durable prompt",
            "cli_name": "codex",
            "role": "default",
            "absolute_file_paths": [],
            "images": [],
            "idempotency_key": "restart-test-request-1",
        },
    )
    assert _Agent.calls == 1
    assert json.loads(replay[0].text)["continuation_offer"]["continuation_id"] == continuation_id

    # Simulate the persistence boundary of a fresh PAL process.
    reset_default_storage_for_tests()

    second = await server.handle_call_tool(
        "clink",
        {
            "prompt": "second durable prompt",
            "cli_name": "codex",
            "role": "default",
            "absolute_file_paths": [],
            "images": [],
            "continuation_id": continuation_id,
            "idempotency_key": "restart-test-request-2",
        },
    )
    assert json.loads(second[0].text)["status"] == "continuation_available"

    second_replay = await server.handle_call_tool(
        "clink",
        {
            "prompt": "second durable prompt",
            "cli_name": "codex",
            "role": "default",
            "absolute_file_paths": [],
            "images": [],
            "continuation_id": continuation_id,
            "idempotency_key": "restart-test-request-2",
        },
    )
    assert json.loads(second_replay[0].text)["status"] == "continuation_available"
    assert _Agent.calls == 2

    reset_default_storage_for_tests()
    thread = get_thread(continuation_id)
    assert thread is not None
    assert [turn.content for turn in thread.turns] == [
        "first durable prompt",
        "durable answer 1",
        "second durable prompt",
        "durable answer 2",
    ]
    raw = (tmp_path / "state" / "pal_state.sqlite3").read_bytes()
    assert b"first durable prompt" not in raw
    assert b"durable answer 2" not in raw


@pytest.mark.asyncio
async def test_server_cancellation_detaches_without_failing_exchange(tmp_path, monkeypatch):
    monkeypatch.setenv("PAL_CONVERSATION_BACKEND", "sqlite")
    monkeypatch.setenv("PAL_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("PAL_STATE_KEY_FILE", str(tmp_path / "keys" / "state.key"))
    reset_default_storage_for_tests()
    _Agent.calls = 0
    started = asyncio.Event()
    release = asyncio.Event()

    class SlowAgent(_Agent):
        async def run(self, **kwargs):
            started.set()
            await release.wait()
            return await super().run(**kwargs)

    monkeypatch.setattr("tools.clink.create_agent", lambda client: SlowAgent())
    task = asyncio.create_task(
        server.handle_call_tool(
            "clink",
            {
                "prompt": "survive caller detach",
                "cli_name": "codex",
                "role": "default",
                "absolute_file_paths": [],
                "images": [],
                "idempotency_key": "detach-request-1",
            },
        )
    )
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    store = get_default_storage()
    exchange = store.connection().execute("SELECT status FROM conversation_exchanges").fetchone()
    assert exchange["status"] in {"admitted", "running"}
    release.set()
    clink_tool = server.TOOLS["clink"]
    for background_task in list(clink_tool._background_tasks):
        await background_task

    thread_id = store.connection().execute("SELECT thread_id FROM conversation_threads").fetchone()[0]
    assert [turn["content"] for turn in store.load_thread(thread_id)["turns"]] == [
        "survive caller detach",
        "durable answer 1",
    ]
    assert store.connection().execute("SELECT status FROM clink_runs").fetchone()[0] == "completed"
