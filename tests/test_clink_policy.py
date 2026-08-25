import json
from pathlib import Path

import pytest

from clink.models import NestedAgentPreference, ResolvedCLIClient, ResolvedCLIRole
from clink.parsers.claude import ClaudeJSONParser
from clink.parsers.codex import CodexJSONLParser
from clink.policy import (
    PartnerModelPolicyError,
    attest_client,
    validate_command_policy,
    validate_request_policy,
    verify_observed_policy,
)
from clink.registry import ClinkRegistry


def _client(name: str, args: list[str], executable: str | None = None) -> ResolvedCLIClient:
    role = ResolvedCLIRole(name="default", prompt_path=Path(__file__), role_args=[])
    return ResolvedCLIClient(
        name=name,
        executable=[executable or name],
        working_dir=None,
        internal_args=[],
        config_args=args,
        timeout_seconds=30,
        parser="codex_jsonl" if name == "codex" else "claude_json",
        roles={"default": role},
    )


@pytest.mark.parametrize(
    ("name", "model", "effort"),
    [("claude", "fable", "xhigh"), ("codex", "gpt-5.6-sol", "high")],
)
def test_exact_policy_assertions_are_accepted(name, model, effort):
    validate_request_policy(name, None, None)
    validate_request_policy(name, model, effort)


@pytest.mark.parametrize(
    ("name", "model", "effort"),
    [
        ("claude", "sonnet", "xhigh"),
        ("claude", "fable", "high"),
        ("codex", "gpt-5.6-terra", "high"),
        ("codex", "gpt-5.6-sol", "xhigh"),
    ],
)
def test_conflicting_policy_assertions_fail_closed(name, model, effort):
    with pytest.raises(PartnerModelPolicyError):
        validate_request_policy(name, model, effort)


def test_command_requires_exact_single_claude_pin():
    good = ["claude", "--model", "fable", "--effort", "xhigh"]
    validate_command_policy("claude", good)
    with pytest.raises(PartnerModelPolicyError):
        validate_command_policy("claude", good + ["--model", "sonnet"])
    with pytest.raises(PartnerModelPolicyError):
        validate_command_policy("claude", good + ["--model=sonnet"])
    with pytest.raises(PartnerModelPolicyError):
        validate_command_policy("claude", good + ["--effort=low"])
    with pytest.raises(PartnerModelPolicyError, match="direct-model fallback"):
        validate_command_policy("claude", good + ["--fallback-model", "sonnet"])
    with pytest.raises(PartnerModelPolicyError, match="direct-model fallback"):
        validate_command_policy("claude", good + ["--fallback-model=sonnet"])


def test_claude_client_does_not_override_nested_agent_models():
    config_path = Path(__file__).parents[1] / "conf" / "cli_clients" / "claude.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))

    assert "ANTHROPIC_SMALL_FAST_MODEL" not in config.get("env", {})
    assert config["nested_agent_preference"] == {
        "substantive_model": "opus",
        "substantive_effort": "high",
        "enforcement": "none",
    }


def test_registry_accepts_claude_without_nested_agent_override(tmp_path, monkeypatch):
    override = tmp_path / "claude.json"
    override.write_text(
        json.dumps(
            {
                "name": "claude",
                "command": "claude",
                "additional_args": ["--model", "fable", "--effort", "xhigh"],
                "roles": {"default": {"prompt_path": "systemprompts/clink/default.txt"}},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("CLI_CLIENTS_CONFIG_PATH", str(override))

    registry = ClinkRegistry()
    assert registry.get_client("claude").name == "claude"


def test_command_requires_exact_single_codex_pin():
    good = ["codex", "--model", "gpt-5.6-sol", "-c", 'model_reasoning_effort="high"']
    validate_command_policy("codex", good)
    with pytest.raises(PartnerModelPolicyError):
        validate_command_policy("codex", ["codex", "--model", "gpt-5.6-sol"])
    with pytest.raises(PartnerModelPolicyError):
        validate_command_policy("codex", good + ["--config", 'model="gpt-5.6-terra"'])
    with pytest.raises(PartnerModelPolicyError):
        validate_command_policy("codex", good + ["--config=model_reasoning_effort=low"])
    with pytest.raises(PartnerModelPolicyError):
        validate_command_policy("codex", good + ["--effort", "low"])
    with pytest.raises(PartnerModelPolicyError):
        validate_command_policy("codex", ["wrapper", *good[1:]])
    with pytest.raises(PartnerModelPolicyError):
        validate_command_policy("codex", ["codex", "--", *good[1:]])


def test_capability_rejects_configured_wrapper():
    client = _client(
        "codex",
        ["--model", "gpt-5.6-sol", "-c", 'model_reasoning_effort="high"'],
        executable="python3",
    )
    with pytest.raises(PartnerModelPolicyError, match="canonical"):
        attest_client(client, client.get_role("default"))


def test_capability_binds_executable_and_policy():
    client = _client(
        "codex",
        ["--model", "gpt-5.6-sol", "-c", 'model_reasoning_effort="high"'],
    )
    capability = attest_client(client, client.get_role("default"))
    assert Path(capability.executable).is_absolute()
    assert capability.model == "gpt-5.6-sol"
    assert capability.reasoning_effort == "high"
    assert len(capability.config_digest) == 64


def test_nested_preference_never_changes_admission_digest():
    plain = _client("claude", ["--model", "fable", "--effort", "xhigh"])
    preferred = plain.model_copy(
        update={
            "nested_agent_preference": NestedAgentPreference(
                substantive_model="opus",
                substantive_effort="high",
                enforcement="none",
            )
        }
    )
    plain_capability = attest_client(plain, plain.get_role("default"))
    preferred_capability = attest_client(preferred, preferred.get_role("default"))
    assert preferred_capability.config_digest == plain_capability.config_digest


def test_claude_terminal_metadata_is_never_a_post_admission_gate():
    verify_observed_policy(
        "claude",
        {"model_used": "claude-sonnet-4", "reasoning_effort_used": "low"},
    )


def test_codex_terminal_metadata_is_also_never_a_post_admission_gate():
    verify_observed_policy(
        "codex",
        {"model_used": "gpt-5.6-terra", "reasoning_effort_used": "medium"},
    )


def test_codex_parser_captures_observed_model_and_effort():
    stdout = "\n".join(
        [
            '{"type":"turn_context","payload":{"model":"gpt-5.6-sol","effort":"high"}}',
            '{"type":"item.completed","item":{"type":"agent_message","text":"ok"}}',
        ]
    )
    parsed = CodexJSONLParser().parse(stdout, "")
    assert parsed.metadata["models_used"] == ["gpt-5.6-sol"]
    assert parsed.metadata["reasoning_efforts_used"] == ["high"]


def test_codex_parser_preserves_mixed_nested_telemetry():
    stdout = "\n".join(
        [
            '{"type":"turn_context","payload":{"model":"gpt-5.6-sol","effort":"high"}}',
            '{"type":"turn_context","payload":{"model":"gpt-5.6-terra","effort":"medium"}}',
            '{"type":"item.completed","item":{"type":"agent_message","text":"ok"}}',
        ]
    )
    parsed = CodexJSONLParser().parse(stdout, "")
    assert parsed.metadata["models_used"] == ["gpt-5.6-sol", "gpt-5.6-terra"]
    assert parsed.metadata["reasoning_efforts_used"] == ["high", "medium"]


def test_claude_parser_preserves_multiple_observed_models():
    stdout = '{"result":"ok","modelUsage":{"claude-fable-5":{},"claude-sonnet-4":{}}}'
    parsed = ClaudeJSONParser().parse(stdout, "")
    assert parsed.metadata["models_used"] == ["claude-fable-5", "claude-sonnet-4"]
    assert "model_used" not in parsed.metadata
