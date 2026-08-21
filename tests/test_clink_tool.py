import json

import pytest

from clink import get_registry
from clink.agents import AgentOutput
from clink.parsers.base import ParsedCLIResponse
from tools.clink import MAX_RESPONSE_CHARS, CLinkRequest, CLinkTool


@pytest.mark.asyncio
async def test_clink_tool_execute(monkeypatch):
    tool = CLinkTool()

    async def fake_run(**kwargs):
        return AgentOutput(
            parsed=ParsedCLIResponse(content="Hello from Gemini", metadata={"model_used": "gemini-2.5-pro"}),
            sanitized_command=["gemini", "-o", "json"],
            returncode=0,
            stdout='{"response": "Hello from Gemini"}',
            stderr="",
            duration_seconds=0.42,
            parser_name="gemini_json",
            output_file_content=None,
        )

    class DummyAgent:
        async def run(self, **kwargs):
            return await fake_run(**kwargs)

    def fake_create_agent(client):
        return DummyAgent()

    monkeypatch.setattr("tools.clink.create_agent", fake_create_agent)

    arguments = {
        "prompt": "Summarize the project",
        "cli_name": "gemini",
        "role": "default",
        "absolute_file_paths": [],
        "images": [],
    }

    results = await tool.execute(arguments)
    assert len(results) == 1

    payload = json.loads(results[0].text)
    assert payload["status"] in {"success", "continuation_available"}
    assert "Hello from Gemini" in payload["content"]
    metadata = payload.get("metadata", {})
    assert metadata.get("cli_name") == "gemini"
    assert metadata.get("command") == ["gemini", "-o", "json"]


def test_registry_lists_roles():
    registry = get_registry()
    clients = registry.list_clients()
    assert {"claude", "codex", "gemini"}.issubset(set(clients))
    roles = registry.list_roles("gemini")
    assert "default" in roles
    assert "default" in registry.list_roles("codex")
    assert {"default", "planner", "codereviewer", "sparring"}.issubset(set(registry.list_roles("claude")))
    claude_client = registry.get_client("claude")
    model_arg_index = claude_client.config_args.index("--model")
    assert claude_client.config_args[model_arg_index + 1] == "fable"
    assert "sonnet" not in claude_client.config_args
    assert claude_client.get_role("planner").prompt_path.name == "claude_planner.txt"
    assert claude_client.get_role("codereviewer").prompt_path.name == "claude_codereviewer.txt"
    assert claude_client.get_role("sparring").prompt_path.name == "claude_sparring.txt"
    codex_client = registry.get_client("codex")
    # Verify codex uses --enable web_search_request (not --search which is unsupported by exec)
    assert codex_client.config_args == [
        "--json",
        "--dangerously-bypass-approvals-and-sandbox",
        "--enable",
        "web_search_request",
    ]


def test_clink_schema_identifies_claude_fable_as_partner_model():
    schema = CLinkTool().get_input_schema()
    assert "claude (partner model: fable)" in schema["properties"]["cli_name"]["description"]


@pytest.mark.asyncio
async def test_clink_tool_defaults_to_first_cli(monkeypatch):
    tool = CLinkTool()

    async def fake_run(**kwargs):
        return AgentOutput(
            parsed=ParsedCLIResponse(content="Default CLI response", metadata={"events": ["foo"]}),
            sanitized_command=["gemini"],
            returncode=0,
            stdout='{"response": "Default CLI response"}',
            stderr="",
            duration_seconds=0.1,
            parser_name="gemini_json",
            output_file_content=None,
        )

    class DummyAgent:
        async def run(self, **kwargs):
            return await fake_run(**kwargs)

    monkeypatch.setattr("tools.clink.create_agent", lambda client: DummyAgent())

    arguments = {
        "prompt": "Hello",
        "absolute_file_paths": [],
        "images": [],
    }

    result = await tool.execute(arguments)
    payload = json.loads(result[0].text)
    metadata = payload.get("metadata", {})
    assert metadata.get("cli_name") == tool._default_cli_name
    assert metadata.get("events_removed_for_normal") is True


@pytest.mark.asyncio
async def test_clink_tool_truncates_large_output(monkeypatch):
    tool = CLinkTool()

    summary_section = "<SUMMARY>This is the condensed summary.</SUMMARY>"
    long_text = "A" * (MAX_RESPONSE_CHARS + 500) + summary_section

    async def fake_run(**kwargs):
        return AgentOutput(
            parsed=ParsedCLIResponse(content=long_text, metadata={"events": ["event1", "event2"]}),
            sanitized_command=["codex"],
            returncode=0,
            stdout="{}",
            stderr="",
            duration_seconds=0.2,
            parser_name="codex_jsonl",
            output_file_content=None,
        )

    class DummyAgent:
        async def run(self, **kwargs):
            return await fake_run(**kwargs)

    monkeypatch.setattr("tools.clink.create_agent", lambda client: DummyAgent())

    arguments = {
        "prompt": "Summarize",
        "cli_name": tool._default_cli_name,
        "absolute_file_paths": [],
        "images": [],
    }

    result = await tool.execute(arguments)
    payload = json.loads(result[0].text)
    assert payload["status"] in {"success", "continuation_available"}
    assert payload["content"].strip() == "This is the condensed summary."
    metadata = payload.get("metadata", {})
    assert metadata.get("output_summarized") is True
    assert metadata.get("events_removed_for_normal") is True
    assert metadata.get("output_original_length") == len(long_text)


@pytest.mark.asyncio
async def test_clink_tool_forwards_progress_notifications(monkeypatch):
    """Progress callback should fire send_progress_notification for each agent event."""
    tool = CLinkTool()

    captured_kwargs: dict = {}

    async def fake_run(**kwargs):
        captured_kwargs.update(kwargs)
        on_event = kwargs.get("on_event")
        # Simulate the agent emitting two stdout lines and one stderr line.
        if on_event:
            await on_event("stdout", "first line")
            await on_event("stderr", "warn")
            await on_event("stdout", "second line")
        return AgentOutput(
            parsed=ParsedCLIResponse(content="ok", metadata={"model_used": "gemini-2.5-pro"}),
            sanitized_command=["gemini"],
            returncode=0,
            stdout="ok",
            stderr="",
            duration_seconds=0.1,
            parser_name="gemini_json",
            output_file_content=None,
        )

    class DummyAgent:
        async def run(self, **kwargs):
            return await fake_run(**kwargs)

    monkeypatch.setattr("tools.clink.create_agent", lambda client: DummyAgent())

    progress_calls: list[dict] = []

    class FakeSession:
        async def send_progress_notification(self, **kwargs):
            progress_calls.append(kwargs)

    class FakeMeta:
        progressToken = "tok-123"

    class FakeRequestContext:
        meta = FakeMeta()
        session = FakeSession()

    class FakeServer:
        request_context = FakeRequestContext()

    # Inject a fake server module so _build_progress_callback resolves a token.
    import sys
    import types

    fake_server_module = types.ModuleType("server")
    fake_server_module.server = FakeServer()
    monkeypatch.setitem(sys.modules, "server", fake_server_module)

    arguments = {
        "prompt": "Hi",
        "cli_name": "gemini",
        "absolute_file_paths": [],
        "images": [],
    }

    await tool.execute(arguments)

    assert "on_event" in captured_kwargs and captured_kwargs["on_event"] is not None
    assert len(progress_calls) == 3
    assert progress_calls[0]["progress_token"] == "tok-123"
    assert progress_calls[0]["progress"] == 1.0
    assert "[gemini:stdout]" in progress_calls[0]["message"]
    assert "[gemini:stderr]" in progress_calls[1]["message"]
    assert progress_calls[2]["progress"] == 3.0


@pytest.mark.asyncio
async def test_clink_tool_skips_progress_when_no_token(monkeypatch):
    """Without a progressToken in request meta, on_event should be None."""
    tool = CLinkTool()

    captured_kwargs: dict = {}

    async def fake_run(**kwargs):
        captured_kwargs.update(kwargs)
        return AgentOutput(
            parsed=ParsedCLIResponse(content="ok", metadata={}),
            sanitized_command=["gemini"],
            returncode=0,
            stdout="ok",
            stderr="",
            duration_seconds=0.1,
            parser_name="gemini_json",
            output_file_content=None,
        )

    class DummyAgent:
        async def run(self, **kwargs):
            return await fake_run(**kwargs)

    monkeypatch.setattr("tools.clink.create_agent", lambda client: DummyAgent())

    class FakeMeta:
        progressToken = None

    class FakeRequestContext:
        meta = FakeMeta()
        session = object()

    class FakeServer:
        request_context = FakeRequestContext()

    import sys
    import types

    fake_server_module = types.ModuleType("server")
    fake_server_module.server = FakeServer()
    monkeypatch.setitem(sys.modules, "server", fake_server_module)

    await tool.execute({"prompt": "Hi", "cli_name": "gemini", "absolute_file_paths": [], "images": []})

    assert captured_kwargs.get("on_event") is None


@pytest.mark.asyncio
async def test_clink_tool_executes_claude_sparring_with_external_system_prompt(monkeypatch):
    tool = CLinkTool()
    captured_kwargs: dict = {}

    async def fake_run(**kwargs):
        captured_kwargs.update(kwargs)
        return AgentOutput(
            parsed=ParsedCLIResponse(content="Proceed with changes.", metadata={"model_used": "claude-fable-5"}),
            sanitized_command=["claude", "--print", "--output-format", "json", "--model", "fable"],
            returncode=0,
            stdout='{"result": "Proceed with changes."}',
            stderr="",
            duration_seconds=0.1,
            parser_name="claude_json",
            output_file_content=None,
        )

    class DummyAgent:
        async def run(self, **kwargs):
            return await fake_run(**kwargs)

    monkeypatch.setattr("tools.clink.create_agent", lambda client: DummyAgent())

    result = await tool.execute(
        {
            "prompt": "Spar with Codex on whether this clink integration is complete.",
            "cli_name": "claude",
            "role": "sparring",
            "absolute_file_paths": [],
            "images": [],
        }
    )

    payload = json.loads(result[0].text)
    assert payload["status"] in {"success", "continuation_available"}
    metadata = payload.get("metadata", {})
    assert metadata.get("cli_name") == "claude"
    assert metadata.get("partner_model") == "fable"
    assert metadata.get("role") == "sparring"

    assert captured_kwargs["role"].name == "sparring"
    assert captured_kwargs["role"].prompt_path.name == "claude_sparring.txt"
    assert captured_kwargs["system_prompt"].startswith("You are Claude Code sparring with Codex")
    assert "Claude Code CLI agent" in captured_kwargs["prompt"]
    assert "Claude Fable Clink partner model" in captured_kwargs["prompt"]
    assert "Gemini CLI agent" not in captured_kwargs["prompt"]
    assert "You are Claude Code sparring with Codex" not in captured_kwargs["prompt"]


@pytest.mark.asyncio
async def test_clink_prepare_prompt_defaults_to_configured_cli():
    tool = CLinkTool()
    request = CLinkRequest(
        prompt="Summarize the repository.",
        cli_name=None,
        role="default",
        absolute_file_paths=[],
        images=[],
    )

    prompt = await tool.prepare_prompt(request)

    assert "Summarize the repository." in prompt
    assert f"{tool._default_cli_name.capitalize()} CLI agent" in prompt


@pytest.mark.asyncio
async def test_clink_tool_truncates_without_summary(monkeypatch):
    tool = CLinkTool()

    long_text = "B" * (MAX_RESPONSE_CHARS + 1000)

    async def fake_run(**kwargs):
        return AgentOutput(
            parsed=ParsedCLIResponse(content=long_text, metadata={"events": ["event"]}),
            sanitized_command=["codex"],
            returncode=0,
            stdout="{}",
            stderr="",
            duration_seconds=0.2,
            parser_name="codex_jsonl",
            output_file_content=None,
        )

    class DummyAgent:
        async def run(self, **kwargs):
            return await fake_run(**kwargs)

    monkeypatch.setattr("tools.clink.create_agent", lambda client: DummyAgent())

    arguments = {
        "prompt": "Summarize",
        "cli_name": tool._default_cli_name,
        "absolute_file_paths": [],
        "images": [],
    }

    result = await tool.execute(arguments)
    payload = json.loads(result[0].text)
    assert payload["status"] in {"success", "continuation_available"}
    assert "exceeding the configured clink limit" in payload["content"]
    metadata = payload.get("metadata", {})
    assert metadata.get("output_truncated") is True
    assert metadata.get("events_removed_for_normal") is True
    assert metadata.get("output_original_length") == len(long_text)
