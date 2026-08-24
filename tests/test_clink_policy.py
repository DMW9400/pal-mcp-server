from pathlib import Path

import pytest

from clink.models import ResolvedCLIClient, ResolvedCLIRole
from clink.parsers.claude import ClaudeJSONParser
from clink.parsers.codex import CodexJSONLParser
from clink.policy import (
    PartnerModelPolicyError,
    attest_client,
    validate_command_policy,
    validate_request_policy,
    verify_observed_policy,
)


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


def test_observed_positive_mismatch_fails_closed():
    with pytest.raises(PartnerModelPolicyError):
        verify_observed_policy("claude", {"model_used": "claude-sonnet-4"})
    verify_observed_policy("claude", {"model_used": "claude-fable-5"})


def test_codex_parser_captures_observed_model_and_effort():
    stdout = "\n".join(
        [
            '{"type":"turn_context","payload":{"model":"gpt-5.6-sol","effort":"high"}}',
            '{"type":"item.completed","item":{"type":"agent_message","text":"ok"}}',
        ]
    )
    parsed = CodexJSONLParser().parse(stdout, "")
    assert parsed.metadata["model_used"] == "gpt-5.6-sol"
    assert parsed.metadata["reasoning_effort_used"] == "high"


def test_claude_parser_rejects_multiple_observed_models():
    stdout = '{"result":"ok","modelUsage":{"claude-fable-5":{},"claude-sonnet-4":{}}}'
    with pytest.raises(Exception, match="multiple models"):
        ClaudeJSONParser().parse(stdout, "")
