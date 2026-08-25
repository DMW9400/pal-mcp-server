import asyncio
import json
import os
import shutil
import sys
from pathlib import Path

import pytest

from clink.agents.base import CLIAgentError
from clink.agents.claude import ClaudeAgent
from clink.models import NestedAgentPreference, ResolvedCLIClient, ResolvedCLIRole
from clink.policy import ClientCapability


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
        self.pid = os.getpid()

    async def wait(self) -> int:
        return self.returncode

    def kill(self) -> None:
        pass

    @property
    def stdin_data(self) -> bytes:
        return self.stdin.data


class FakeBoundaryAuthority:
    socket_path = Path("/private/tmp/fake-pal-boundary.sock")

    def issue_pending(self, expected_executable):
        self.expected_executable = expected_executable
        return "test-boundary", "test-nonce"

    def activate(self, _boundary_id, _nonce, _pid):
        return None

    def cleanup(self, _boundary_id):
        return None


@pytest.fixture()
def claude_agent(tmp_path, monkeypatch):
    monkeypatch.setenv("PAL_STATE_DIR", str(tmp_path / "state"))
    prompt_path = Path("systemprompts/clink/default.txt").resolve()
    role = ResolvedCLIRole(name="default", prompt_path=prompt_path, role_args=[])
    client = ResolvedCLIClient(
        name="claude",
        executable=["claude"],
        internal_args=["--print", "--output-format", "json"],
        config_args=[
            "--permission-mode",
            "acceptEdits",
            "--model",
            "fable",
            "--effort",
            "xhigh",
        ],
        env={},
        timeout_seconds=30,
        parser="claude_json",
        runner="claude",
        roles={"default": role},
        output_to_file=None,
        working_dir=None,
        nested_agent_preference=NestedAgentPreference(
            substantive_model="opus", substantive_effort="high", enforcement="none"
        ),
    )
    return ClaudeAgent(client, partner_boundary_authority=FakeBoundaryAuthority()), role


async def _run_agent_with_process(
    monkeypatch,
    agent,
    role,
    process,
    *,
    system_prompt="System prompt",
    files=(),
):
    async def fake_create_subprocess_exec(*_args, **_kwargs):
        return process

    def fake_which(_executable_name, path=None):
        return sys.executable

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)
    monkeypatch.setattr(shutil, "which", fake_which)

    return await agent.run(
        role=role,
        prompt="Respond with 42",
        system_prompt=system_prompt,
        files=files,
        images=[],
    )


@pytest.mark.asyncio
async def test_claude_agent_injects_system_prompt(monkeypatch, claude_agent):
    agent, role = claude_agent
    stdout_payload = json.dumps(
        {
            "type": "result",
            "subtype": "success",
            "is_error": False,
            "result": "42",
        }
    ).encode()
    process = DummyProcess(stdout=stdout_payload)

    result = await _run_agent_with_process(monkeypatch, agent, role, process)

    assert "--append-system-prompt" in result.sanitized_command
    idx = result.sanitized_command.index("--append-system-prompt")
    assert result.sanitized_command[idx + 1].startswith("System prompt\n\n")
    assert "prefer opus at high effort for substantive delegated investigation" in result.sanitized_command[idx + 1]
    assert "advisory only; never an admission or result gate" in result.sanitized_command[idx + 1]
    assert process.stdin_data.decode().startswith("Respond with 42")
    assert result.parsed.metadata["model_used"] == "fable"
    assert result.parsed.metadata["reasoning_effort_used"] == "xhigh"
    assert result.parsed.metadata["policy_observation_source"] == "attested_command"
    assert result.parsed.metadata["nested_agent_preference"] == {
        "substantive_model": "opus",
        "substantive_effort": "high",
        "enforcement": "none",
    }


@pytest.mark.asyncio
async def test_claude_agent_grants_only_attached_file_parents(monkeypatch, claude_agent, tmp_path):
    agent, role = claude_agent
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    first_dir = tmp_path / "one"
    second_dir = tmp_path / "two"
    first_dir.mkdir()
    second_dir.mkdir()
    first = first_dir / "brief.md"
    second = second_dir / "contract.json"
    first.write_text("brief", encoding="utf-8")
    second.write_text("contract", encoding="utf-8")
    process = DummyProcess(stdout=b'{"type":"result","result":"ok"}')

    result = await _run_agent_with_process(
        monkeypatch,
        agent,
        role,
        process,
        files=[str(first), str(first), str(second)],
    )

    grants = [
        result.sanitized_command[index + 1]
        for index, value in enumerate(result.sanitized_command[:-1])
        if value == "--add-dir"
    ]
    assert grants == [str(first_dir), str(second_dir)]


def test_claude_agent_refuses_broad_add_dir_grants(claude_agent):
    agent, role = claude_agent

    with pytest.raises(CLIAgentError, match="broad Claude --add-dir"):
        agent._build_command(role=role, system_prompt=None, files=[str(Path.home())])


def test_claude_agent_refuses_add_dir_outside_user_home(claude_agent):
    agent, role = claude_agent

    with pytest.raises(CLIAgentError, match="outside the user home"):
        agent._build_command(role=role, system_prompt=None, files=["/etc/hosts"])


def test_claude_agent_preserves_inherited_nested_routing(monkeypatch, claude_agent):
    agent, _role = claude_agent
    monkeypatch.setenv("ANTHROPIC_MODEL", "forced")
    monkeypatch.setenv("CLAUDE_CODE_SUBAGENT_MODEL", "forced-child")
    monkeypatch.setenv("MAX_THINKING_TOKENS", "1")
    monkeypatch.setenv("UNRELATED_ENV", "preserved")

    env = agent._build_environment(ClientCapability(
        cli_name="claude",
        executable="/opt/example/claude",
        executable_identity="1:2:3:4",
        model="fable",
        reasoning_effort="xhigh",
        config_digest="test",
    ))

    assert env["ANTHROPIC_MODEL"] == "forced"
    assert env["CLAUDE_CODE_SUBAGENT_MODEL"] == "forced-child"
    assert env["MAX_THINKING_TOKENS"] == "1"
    assert env["UNRELATED_ENV"] == "preserved"


@pytest.mark.asyncio
async def test_claude_agent_allows_nested_model_usage(monkeypatch, claude_agent):
    agent, role = claude_agent
    stdout_payload = json.dumps(
        {
            "type": "result",
            "subtype": "success",
            "is_error": False,
            "result": "review complete",
            "modelUsage": {
                "claude-fable-5": {"inputTokens": 10, "outputTokens": 5},
                "claude-sonnet-4-5": {"inputTokens": 20, "outputTokens": 8},
            },
        }
    ).encode()
    process = DummyProcess(stdout=stdout_payload)

    result = await _run_agent_with_process(monkeypatch, agent, role, process)

    assert result.parsed.content == "review complete"
    assert result.parsed.metadata["model_used"] == "fable"
    assert result.parsed.metadata["reasoning_effort_used"] == "xhigh"
    assert result.parsed.metadata["models_used"] == ["claude-fable-5", "claude-sonnet-4-5"]
    assert result.parsed.metadata["policy_observation_source"] == "attested_command"
    assert result.parsed.metadata["post_admission_model_policy"] == "observe_only"


@pytest.mark.asyncio
async def test_claude_agent_ignores_terminal_effort_for_policy(monkeypatch, claude_agent):
    agent, role = claude_agent
    process = DummyProcess(
        stdout=b'{"type":"result","result":"ok","effort":"low","modelUsage":{"claude-haiku":{}}}'
    )

    result = await _run_agent_with_process(monkeypatch, agent, role, process)

    assert result.parsed.metadata["reasoning_efforts_used"] == ["low"]
    assert result.parsed.metadata["reasoning_effort_used"] == "xhigh"
    assert result.parsed.metadata["models_used"] == ["claude-haiku"]
    assert result.parsed.metadata["post_admission_model_policy"] == "observe_only"


@pytest.mark.asyncio
async def test_claude_agent_recovers_error_payload(monkeypatch, claude_agent):
    agent, role = claude_agent
    stdout_payload = json.dumps(
        {
            "type": "result",
            "subtype": "success",
            "is_error": True,
            "result": "API Error",
        }
    ).encode()
    process = DummyProcess(stdout=stdout_payload, returncode=2)

    result = await _run_agent_with_process(monkeypatch, agent, role, process)

    assert result.returncode == 2
    assert result.parsed.content == "API Error"
    assert result.parsed.metadata["is_error"] is True


@pytest.mark.asyncio
async def test_claude_agent_propagates_unparseable_output(monkeypatch, claude_agent):
    agent, role = claude_agent
    process = DummyProcess(stdout=b"", returncode=1)

    with pytest.raises(CLIAgentError):
        await _run_agent_with_process(monkeypatch, agent, role, process)
