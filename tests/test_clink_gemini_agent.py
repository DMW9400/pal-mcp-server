import asyncio
import shutil
from pathlib import Path

import pytest

from clink.agents.base import CLIAgentError
from clink.agents.gemini import GeminiAgent
from clink.models import ResolvedCLIClient, ResolvedCLIRole


def _make_stream_reader(payload: bytes) -> asyncio.StreamReader:
    reader = asyncio.StreamReader()
    if payload:
        reader.feed_data(payload)
    reader.feed_eof()
    return reader


class _DummyStdin:
    def __init__(self) -> None:
        self.data: bytes = b""
        self.closed = False

    def write(self, data: bytes) -> None:
        self.data += data

    async def drain(self) -> None:
        return None

    def close(self) -> None:
        self.closed = True


class DummyProcess:
    def __init__(self, *, stdout: bytes = b"", stderr: bytes = b"", returncode: int = 0):
        self.stdout = _make_stream_reader(stdout)
        self.stderr = _make_stream_reader(stderr)
        self.stdin = _DummyStdin()
        self.returncode = returncode

    async def wait(self) -> int:
        return self.returncode

    def kill(self) -> None:
        pass


@pytest.fixture()
def gemini_agent():
    prompt_path = Path("systemprompts/clink/gemini_default.txt").resolve()
    role = ResolvedCLIRole(name="default", prompt_path=prompt_path, role_args=[])
    client = ResolvedCLIClient(
        name="gemini",
        executable=["gemini"],
        internal_args=[],
        config_args=[],
        env={},
        timeout_seconds=30,
        parser="gemini_json",
        roles={"default": role},
        output_to_file=None,
        working_dir=None,
    )
    return GeminiAgent(client), role


async def _run_agent_with_process(monkeypatch, agent, role, process):
    async def fake_create_subprocess_exec(*_args, **_kwargs):
        return process

    def fake_which(executable_name):
        return f"/usr/bin/{executable_name}"

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)
    monkeypatch.setattr(shutil, "which", fake_which)
    return await agent.run(role=role, prompt="do something", files=[], images=[])


@pytest.mark.asyncio
async def test_gemini_agent_recovers_tool_error(monkeypatch, gemini_agent):
    agent, role = gemini_agent
    error_json = """{
  "error": {
    "type": "FatalToolExecutionError",
    "message": "Error executing tool replace: Failed to edit",
    "code": "edit_expected_occurrence_mismatch"
  }
}"""
    stderr = ("Error: Failed to edit, expected 1 occurrence but found 2.\n" + error_json).encode()
    process = DummyProcess(stderr=stderr, returncode=54)

    result = await _run_agent_with_process(monkeypatch, agent, role, process)

    assert result.returncode == 54
    assert result.parsed.metadata["cli_error_recovered"] is True
    assert result.parsed.metadata["cli_error_code"] == "edit_expected_occurrence_mismatch"
    assert "Gemini CLI reported a tool failure" in result.parsed.content


@pytest.mark.asyncio
async def test_gemini_agent_propagates_unrecoverable_error(monkeypatch, gemini_agent):
    agent, role = gemini_agent
    stderr = b"Plain failure without structured payload"
    process = DummyProcess(stderr=stderr, returncode=54)

    with pytest.raises(CLIAgentError):
        await _run_agent_with_process(monkeypatch, agent, role, process)
