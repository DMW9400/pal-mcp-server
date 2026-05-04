"""Tests for BaseCLIAgent's streaming subprocess reader and progress callback wiring."""

import asyncio
import shutil
from pathlib import Path

import pytest

from clink.agents.base import BaseCLIAgent, CLIAgentError
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


class _DelayedStreamReader:
    """A StreamReader-like object whose lines arrive on a delay between each line.

    Used to verify that on_event fires per line (i.e. progress is reported as the
    subprocess produces output, not in one batch at the end).
    """

    def __init__(self, lines: list[bytes], per_line_delay: float = 0.0) -> None:
        self._lines = list(lines)
        self._delay = per_line_delay

    async def readline(self) -> bytes:
        if self._delay:
            await asyncio.sleep(self._delay)
        if not self._lines:
            return b""
        return self._lines.pop(0)

    async def read(self, _n: int = -1) -> bytes:  # pragma: no cover - fallback path
        rest = b"".join(self._lines)
        self._lines.clear()
        return rest


class DummyProcess:
    def __init__(
        self,
        *,
        stdout: bytes | _DelayedStreamReader = b"",
        stderr: bytes | _DelayedStreamReader = b"",
        returncode: int = 0,
    ):
        self.stdout = stdout if isinstance(stdout, _DelayedStreamReader) else _make_stream_reader(stdout)
        self.stderr = stderr if isinstance(stderr, _DelayedStreamReader) else _make_stream_reader(stderr)
        self.stdin = _DummyStdin()
        self.returncode = returncode
        self.killed = False

    async def wait(self) -> int:
        return self.returncode

    def kill(self) -> None:
        self.killed = True


@pytest.fixture()
def base_agent():
    prompt_path = Path("systemprompts/clink/default.txt").resolve()
    role = ResolvedCLIRole(name="default", prompt_path=prompt_path, role_args=[])
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
    return BaseCLIAgent(client), role


async def _run(monkeypatch, agent, role, process, *, on_event=None):
    async def fake_create_subprocess_exec(*_args, **_kwargs):
        return process

    def fake_which(name):
        return f"/usr/bin/{name}"

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)
    monkeypatch.setattr(shutil, "which", fake_which)

    return await agent.run(
        role=role,
        prompt="ping",
        files=[],
        images=[],
        on_event=on_event,
    )


@pytest.mark.asyncio
async def test_streaming_reader_collects_full_stdout(monkeypatch, base_agent):
    """The streaming reader should reassemble the same final stdout as communicate()."""
    agent, role = base_agent
    payload = (
        b'{"type":"item.completed","item":{"id":"i0","type":"agent_message","text":"line one"}}\n'
        b'{"type":"item.completed","item":{"id":"i1","type":"agent_message","text":"line two"}}\n'
        b'{"type":"turn.completed","usage":{"input_tokens":1,"output_tokens":2}}\n'
    )
    process = DummyProcess(stdout=payload, returncode=0)

    result = await _run(monkeypatch, agent, role, process)

    assert result.stdout == payload.decode("utf-8")
    assert result.returncode == 0
    assert "line one" in result.parsed.content
    assert "line two" in result.parsed.content


@pytest.mark.asyncio
async def test_on_event_fires_per_line(monkeypatch, base_agent):
    """Each stdout/stderr line should trigger a separate on_event call."""
    agent, role = base_agent
    stdout_payload = b"alpha\nbeta\ngamma\n"
    stderr_payload = b"warn-1\nwarn-2\n"
    process = DummyProcess(stdout=stdout_payload, stderr=stderr_payload, returncode=1)

    events: list[tuple[str, str]] = []

    async def on_event(kind, text):
        events.append((kind, text))

    # Codex parser will fail on plain text and the agent will raise — we only care
    # about whether on_event was called per line before the failure.
    with pytest.raises(CLIAgentError):
        await _run(monkeypatch, agent, role, process, on_event=on_event)

    stdout_events = [e for e in events if e[0] == "stdout"]
    stderr_events = [e for e in events if e[0] == "stderr"]
    assert [e[1] for e in stdout_events] == ["alpha", "beta", "gamma"]
    assert [e[1] for e in stderr_events] == ["warn-1", "warn-2"]


@pytest.mark.asyncio
async def test_on_event_callback_failure_does_not_kill_run(monkeypatch, base_agent):
    """A misbehaving on_event must never abort the underlying CLI run."""
    agent, role = base_agent
    payload = (
        b'{"type":"item.completed","item":{"id":"i0","type":"agent_message","text":"hi"}}\n'
        b'{"type":"turn.completed","usage":{"input_tokens":1,"output_tokens":1}}\n'
    )
    process = DummyProcess(stdout=payload, returncode=0)

    async def on_event(_kind, _text):
        raise RuntimeError("progress dispatch failed")

    result = await _run(monkeypatch, agent, role, process, on_event=on_event)
    assert result.returncode == 0
    assert "hi" in result.parsed.content


@pytest.mark.asyncio
async def test_streaming_reader_passes_prompt_via_stdin(monkeypatch, base_agent):
    agent, role = base_agent
    payload = (
        b'{"type":"item.completed","item":{"id":"i0","type":"agent_message","text":"ok"}}\n'
        b'{"type":"turn.completed","usage":{"input_tokens":1,"output_tokens":1}}\n'
    )
    process = DummyProcess(stdout=payload, returncode=0)

    await _run(monkeypatch, agent, role, process)

    assert process.stdin.data == b"ping"
    assert process.stdin.closed is True


@pytest.mark.asyncio
async def test_timeout_kills_process_and_raises(monkeypatch, base_agent):
    """Timeout enforcement should still work with the new streaming reader."""
    agent, role = base_agent
    # Force the run to time out by giving it a delayed-stream that never EOFs in time.
    # We achieve this by overriding _stream_subprocess to sleep past the configured timeout.
    agent.client.__dict__["timeout_seconds"] = 0  # tight timeout via dataclass __dict__

    process = DummyProcess(stdout=b"", returncode=0)

    async def slow_stream(*_args, **_kwargs):
        await asyncio.sleep(5)
        return ("", "")

    monkeypatch.setattr(agent, "_stream_subprocess", slow_stream)

    with pytest.raises(CLIAgentError) as excinfo:
        await _run(monkeypatch, agent, role, process)

    assert "timed out" in str(excinfo.value)
    assert process.killed is True
