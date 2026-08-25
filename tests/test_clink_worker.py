import asyncio
import json
import os
import shutil
import sys
import time
import uuid
from pathlib import Path

import pytest

from clink.agents import AgentOutput
from clink.parsers.base import ParsedCLIResponse
from clink.policy import attest_client
from clink.worker import ClinkWorker
from tools.clink import CLinkRequest, CLinkTool
from tools.shared.exceptions import ToolExecutionError
from utils.sqlite_conversation_storage import get_default_storage, reset_default_storage_for_tests


class _FakeAgent:
    calls = 0

    async def run(self, **kwargs):
        self.__class__.calls += 1
        return AgentOutput(
            parsed=ParsedCLIResponse(
                content="worker answer",
                metadata={"model_used": "gpt-5.6-sol", "reasoning_effort_used": "high"},
            ),
            sanitized_command=["codex", "--model", "gpt-5.6-sol"],
            returncode=0,
            stdout="{}",
            stderr="",
            duration_seconds=0.01,
            parser_name="codex_jsonl",
            output_file_content=None,
        )


def _configure(tmp_path, monkeypatch):
    monkeypatch.setenv("PAL_CONVERSATION_BACKEND", "sqlite")
    monkeypatch.setenv("PAL_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("PAL_STATE_KEY_FILE", str(tmp_path / "keys" / "state.key"))
    reset_default_storage_for_tests()
    return get_default_storage()


@pytest.mark.asyncio
async def test_worker_completes_exchange_and_run_atomically(tmp_path, monkeypatch):
    store = _configure(tmp_path, monkeypatch)
    worker = ClinkWorker()
    client = worker.registry.get_client("codex")
    role = client.get_role("default")
    capability = attest_client(client, role)
    store.register_process(worker.instance_id, "clink_worker")
    store.publish_capability(
        cli_name="codex",
        role="default",
        config_digest=capability.config_digest,
        executable_identity=capability.executable_identity,
        model=capability.model,
        reasoning_effort=capability.reasoning_effort,
        owner_instance_id=worker.instance_id,
        owner_mode="clink_worker",
    )
    thread_id = store.create_thread("clink", {})
    admission = store.begin_exchange(
        thread_id,
        "clink",
        {"role": "user", "content": "worker question", "timestamp": "2026-08-24T00:00:00+00:00"},
        capability_digest=capability.config_digest,
    )
    run_id = str(uuid.uuid4())
    request = CLinkRequest(prompt="worker question", cli_name="codex", role="default")
    envelope = {
        "schema_version": 1,
        "run_id": run_id,
        "thread_id": thread_id,
        "continuation_id": thread_id,
        "exchange_id": admission.exchange_id,
        "cli_name": "codex",
        "role": "default",
        "capability_digest": capability.config_digest,
        "request": request.model_dump(mode="json"),
        "prompt_text": "prepared prompt",
        "system_prompt_text": "",
        "absolute_file_paths": [],
        "images": [],
    }
    store.create_queued_worker_run(
        run_id=run_id,
        continuation_id=thread_id,
        exchange_id=admission.exchange_id,
        cli_name="codex",
        role="default",
        envelope=envelope,
        assigned_worker_instance_id=worker.instance_id,
        capability_digest=capability.config_digest,
    )
    claim = store.claim_next_run(worker.instance_id)
    monkeypatch.setattr("clink.worker.create_agent", lambda client: _FakeAgent())
    _FakeAgent.calls = 0

    await worker._execute(claim)

    assert _FakeAgent.calls == 1
    assert store.read_run(run_id)["status"] == "completed"
    loaded = store.load_thread(thread_id)
    rows = [
        dict(row)
        for row in store.connection().execute(
            "SELECT ordinal,exchange_id,role FROM conversation_turns WHERE thread_id=? ORDER BY ordinal",
            (thread_id,),
        )
    ]
    assert [turn["content"] for turn in loaded["turns"]] == [
        "worker question",
        "worker answer",
    ], (loaded, rows)
    assert store.connection().execute("SELECT COUNT(*) FROM clink_run_queue").fetchone()[0] == 0


def test_expired_claim_is_interrupted_not_reexecuted(tmp_path, monkeypatch):
    store = _configure(tmp_path, monkeypatch)
    store.register_process("dead-worker", "clink_worker")
    store.publish_capability(
        cli_name="codex",
        role="default",
        config_digest="test-digest",
        executable_identity="test-executable",
        model="gpt-5.6-sol",
        reasoning_effort="high",
        owner_instance_id="dead-worker",
        owner_mode="clink_worker",
    )
    run_id = str(uuid.uuid4())
    store.create_queued_worker_run(
        run_id=run_id,
        continuation_id=None,
        exchange_id=None,
        cli_name="codex",
        role="default",
        envelope={
            "secret": "never replay",
            "cli_name": "codex",
            "role": "default",
            "capability_digest": "test-digest",
        },
        assigned_worker_instance_id="dead-worker",
        capability_digest="test-digest",
    )
    assert store.claim_next_run("dead-worker", lease_seconds=1)["run_id"] == run_id
    with store._write() as connection:
        connection.execute("UPDATE clink_run_queue SET lease_expires_at_us=0 WHERE run_id=?", (run_id,))

    assert store.claim_next_run("replacement-worker") is None
    assert store.interrupt_stale_claims() == 1
    assert store.read_run(run_id)["status"] == "interrupted"


@pytest.mark.asyncio
async def test_preflight_selects_only_exact_fresh_worker_capability(tmp_path, monkeypatch):
    store = _configure(tmp_path, monkeypatch)
    tool = CLinkTool()
    client = tool._registry.get_client("codex")
    role = client.get_role("default")
    capability = attest_client(client, role)
    store.register_process("worker", "clink_worker")
    store.publish_capability(
        cli_name="codex",
        role="default",
        config_digest=capability.config_digest,
        executable_identity=capability.executable_identity,
        model=capability.model,
        reasoning_effort=capability.reasoning_effort,
        owner_instance_id="worker",
        owner_mode="clink_worker",
    )
    arguments = await tool.preflight_continuation(
        {"prompt": "queued", "cli_name": "codex", "role": "default", "background": True}
    )
    assert arguments["_execution_owner"] == "worker"
    assert arguments["_capability_digest"] == capability.config_digest


@pytest.mark.asyncio
async def test_protected_preflight_requires_supervised_worker(tmp_path, monkeypatch):
    _configure(tmp_path, monkeypatch)
    monkeypatch.delenv("PAL_CLINK_ALLOW_UNSUPERVISED", raising=False)
    tool = CLinkTool()
    with pytest.raises(ToolExecutionError, match="independently supervised worker"):
        await tool.preflight_continuation({"prompt": "must not launch locally", "cli_name": "codex", "role": "default"})


def test_new_durable_thread_failure_propagates_before_execution(tmp_path, monkeypatch):
    _configure(tmp_path, monkeypatch)
    tool = CLinkTool()

    def fail_create_thread(*args, **kwargs):
        raise RuntimeError("disk unavailable")

    monkeypatch.setattr("utils.conversation_memory.create_thread", fail_create_thread)
    request = CLinkRequest(prompt="must remain recoverable", cli_name="codex", role="default")
    with pytest.raises(RuntimeError, match="disk unavailable"):
        tool._create_recovery_thread(request)


@pytest.mark.asyncio
async def test_separate_worker_survives_producer_store_reopen_without_model_tokens(tmp_path, monkeypatch):
    state = tmp_path / "state"
    key = tmp_path / "keys" / "state.key"
    config = tmp_path / "codex.json"
    fixture_cli = Path(__file__).parent / "fixtures" / "fake_codex_cli.py"
    fake_home = tmp_path / "home"
    fake_bin = fake_home / ".local" / "bin"
    fake_bin.mkdir(parents=True)
    shutil.copy(fixture_cli, fake_bin / "codex")
    (fake_bin / "codex").chmod(0o700)
    config.write_text(
        json.dumps(
            {
                "name": "codex",
                "command": "codex",
                "additional_args": [
                    "--json",
                    "--model",
                    "gpt-5.6-sol",
                    "-c",
                    'model_reasoning_effort="high"',
                ],
                "roles": {"default": {"prompt_path": "systemprompts/clink/default.txt"}},
            }
        )
    )
    env = os.environ.copy()
    env.update(
        {
            "PAL_CONVERSATION_BACKEND": "sqlite",
            "PAL_STATE_DIR": str(state),
            "PAL_STATE_KEY_FILE": str(key),
            "CLI_CLIENTS_CONFIG_PATH": str(config),
            "PYTHONPATH": str(Path(__file__).parent.parent),
            "PYTHONDONTWRITEBYTECODE": "1",
            "HOME": str(fake_home),
        }
    )
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "clink.worker",
        "serve",
        cwd=str(Path(__file__).parent.parent),
        env=env,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    monkeypatch.setenv("PAL_CONVERSATION_BACKEND", "sqlite")
    monkeypatch.setenv("PAL_STATE_DIR", str(state))
    monkeypatch.setenv("PAL_STATE_KEY_FILE", str(key))
    reset_default_storage_for_tests()
    store = get_default_storage()
    try:
        deadline = time.monotonic() + 10
        capability = None
        while time.monotonic() < deadline:
            capability = store.get_fresh_worker_capability("codex", "default")
            if capability:
                break
            await asyncio.sleep(0.05)
        assert capability is not None

        thread_id = store.create_thread("clink", {})
        admission = store.begin_exchange(
            thread_id,
            "clink",
            {"role": "user", "content": "fixture question", "timestamp": "2026-08-24T00:00:00+00:00"},
            capability_digest=capability["config_digest"],
        )
        run_id = str(uuid.uuid4())
        request = CLinkRequest(prompt="fixture question", cli_name="codex", role="default")
        store.create_queued_worker_run(
            run_id=run_id,
            continuation_id=thread_id,
            exchange_id=admission.exchange_id,
            cli_name="codex",
            role="default",
            envelope={
                "schema_version": 1,
                "run_id": run_id,
                "thread_id": thread_id,
                "continuation_id": thread_id,
                "exchange_id": admission.exchange_id,
                "cli_name": "codex",
                "role": "default",
                "capability_digest": capability["config_digest"],
                "request": request.model_dump(mode="json"),
                "prompt_text": "prepared fixture prompt",
                "system_prompt_text": "",
                "absolute_file_paths": [],
                "images": [],
            },
            assigned_worker_instance_id=capability["owner_instance_id"],
            capability_digest=capability["config_digest"],
        )

        # Dropping and reopening the producer's store simulates the PAL-side
        # persistence boundary while the independently owned CLI remains active.
        await asyncio.sleep(0.2)
        reset_default_storage_for_tests()
        store = get_default_storage()
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and store.read_run(run_id)["status"] not in {
            "completed",
            "failed",
            "interrupted",
        }:
            await asyncio.sleep(0.05)
        assert store.read_run(run_id)["status"] == "completed"
        assert [turn["content"] for turn in store.load_thread(thread_id)["turns"]] == [
            "fixture question",
            "fixture answer",
        ]
    finally:
        process.terminate()
        await asyncio.wait_for(process.wait(), timeout=5)
