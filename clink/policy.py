"""Fail-closed partner-model policy for PAL clink collaboration."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path


class PartnerModelPolicyError(RuntimeError):
    """Raised before admission when an absolute partner-model invariant fails."""


@dataclass(frozen=True)
class PartnerModelPolicy:
    cli_name: str
    model: str
    reasoning_effort: str
    model_args: tuple[str, ...]
    effort_args: tuple[str, ...]
    observed_model_aliases: frozenset[str]


POLICIES: dict[str, PartnerModelPolicy] = {
    "claude": PartnerModelPolicy(
        cli_name="claude",
        model="fable",
        reasoning_effort="xhigh",
        model_args=("--model", "fable"),
        effort_args=("--effort", "xhigh"),
        observed_model_aliases=frozenset({"fable", "claude-fable-5"}),
    ),
    "codex": PartnerModelPolicy(
        cli_name="codex",
        model="gpt-5.6-sol",
        reasoning_effort="high",
        model_args=("--model", "gpt-5.6-sol"),
        effort_args=("-c", 'model_reasoning_effort="high"'),
        observed_model_aliases=frozenset({"gpt-5.6-sol"}),
    ),
}

@dataclass(frozen=True)
class ClientCapability:
    cli_name: str
    executable: str
    executable_identity: str
    model: str | None
    reasoning_effort: str | None
    config_digest: str


def get_policy(cli_name: str) -> PartnerModelPolicy | None:
    return POLICIES.get(cli_name.strip().lower())


def validate_request_policy(cli_name: str, model: str | None, reasoning_effort: str | None) -> None:
    """Reject conflicting overrides; exact redundant values remain compatible."""
    policy = get_policy(cli_name)
    if policy is None:
        return
    if model is not None and model.strip() != policy.model:
        raise PartnerModelPolicyError(
            f"CLI '{cli_name}' is fixed to model '{policy.model}'; per-call model overrides are forbidden"
        )
    if reasoning_effort is not None and reasoning_effort.strip() != policy.reasoning_effort:
        raise PartnerModelPolicyError(
            f"CLI '{cli_name}' is fixed to reasoning effort '{policy.reasoning_effort}'; "
            "per-call effort overrides are forbidden"
        )


def _count_pair(argv: Sequence[str], pair: tuple[str, ...]) -> int:
    width = len(pair)
    return sum(1 for index in range(len(argv) - width + 1) if tuple(argv[index : index + width]) == pair)


def validate_command_policy(cli_name: str, argv: Sequence[str]) -> None:
    policy = get_policy(cli_name)
    if policy is None:
        return
    if not argv or Path(argv[0]).name != policy.cli_name or "--" in argv:
        raise PartnerModelPolicyError(
            f"CLI '{cli_name}' must use the canonical {policy.cli_name} executable with effective pins"
        )
    if _count_pair(argv, policy.model_args) != 1 or _count_pair(argv, policy.effort_args) != 1:
        raise PartnerModelPolicyError(
            f"CLI '{cli_name}' command must contain exactly one immutable "
            f"{policy.model}/{policy.reasoning_effort} policy pin"
        )

    forbidden_model_flags = {"--model", "-m"}
    forbidden_effort_flags = {"--effort"}
    model_positions = [index for index, value in enumerate(argv) if value in forbidden_model_flags]
    effort_positions = [index for index, value in enumerate(argv) if value in forbidden_effort_flags]
    inline_model_flags = [value for value in argv if value.startswith(("--model=", "-m="))]
    inline_effort_flags = [value for value in argv if value.startswith("--effort=")]
    if cli_name.lower() == "claude" and any(
        value == "--fallback-model" or value.startswith("--fallback-model=") for value in argv
    ):
        raise PartnerModelPolicyError("CLI 'claude' command contains a direct-model fallback")
    if len(model_positions) != 1 or inline_model_flags:
        raise PartnerModelPolicyError(f"CLI '{cli_name}' command contains a conflicting model flag")
    if cli_name.lower() == "claude" and (len(effort_positions) != 1 or inline_effort_flags):
        raise PartnerModelPolicyError("CLI 'claude' command contains a conflicting effort flag")
    if cli_name.lower() == "codex":
        if effort_positions or inline_effort_flags:
            raise PartnerModelPolicyError("CLI 'codex' command contains a conflicting effort flag")
        config_values = [argv[index + 1] for index, value in enumerate(argv[:-1]) if value in {"-c", "--config"}]
        config_values.extend(value.split("=", 1)[1] for value in argv if value.startswith("--config="))
        model_configs = [value for value in config_values if str(value).split("=", 1)[0].strip() == "model"]
        if model_configs:
            raise PartnerModelPolicyError("CLI 'codex' command contains a conflicting model config")
        effort_configs = [
            value for value in config_values if str(value).split("=", 1)[0].strip() == "model_reasoning_effort"
        ]
        if effort_configs != [policy.effort_args[1]]:
            raise PartnerModelPolicyError("CLI 'codex' command contains a conflicting effort config")


def attest_client(client, role=None, *, path: str | None = None) -> ClientCapability:
    """Resolve and attest a launch capability before any thread/run mutation."""
    command = [*client.executable, *client.internal_args, *client.config_args]
    if role is not None:
        command.extend(role.role_args)
    validate_command_policy(client.name, command)
    policy = get_policy(client.name)
    if policy and (len(client.executable) != 1 or client.executable[0] != client.name.lower()):
        raise PartnerModelPolicyError(
            f"CLI '{client.name}' cannot replace its canonical executable with a wrapper or compound command"
        )
    if path is None and policy:
        search_dirs = [
            str(candidate)
            for candidate in (
                Path.home() / ".local" / "bin",
                Path("/opt/homebrew/bin"),
                Path("/usr/local/bin"),
            )
            if candidate.is_dir()
        ]
        nvm_root = Path.home() / ".nvm" / "versions" / "node"
        if nvm_root.is_dir():
            search_dirs.extend(str(candidate / "bin") for candidate in sorted(nvm_root.iterdir(), reverse=True))
        path = os.pathsep.join(dict.fromkeys(search_dirs))
    executable = shutil.which(command[0], path=path) if path is not None else shutil.which(command[0])
    if executable is None:
        raise PartnerModelPolicyError(
            f"Executable '{command[0]}' is unavailable for CLI '{client.name}'; no work was admitted"
        )
    resolved = Path(executable).resolve(strict=True)
    info = resolved.stat()
    if not stat.S_ISREG(info.st_mode) or not os.access(resolved, os.X_OK):
        raise PartnerModelPolicyError(f"Executable for CLI '{client.name}' is not a runnable regular file")
    material = {
        "cli_name": client.name.lower(),
        "executable": str(resolved),
        "executable_device": info.st_dev,
        "executable_inode": info.st_ino,
        "executable_size": info.st_size,
        "executable_mtime_ns": info.st_mtime_ns,
        "command": [str(resolved), *command[1:]],
        "configured_env": dict(sorted(client.env.items())),
        "role": role.name if role is not None else None,
        "role_args": list(role.role_args) if role is not None else [],
        "model": policy.model if policy else None,
        "reasoning_effort": policy.reasoning_effort if policy else None,
    }
    digest = hashlib.sha256(json.dumps(material, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return ClientCapability(
        cli_name=client.name.lower(),
        executable=str(resolved),
        executable_identity=f"{info.st_dev}:{info.st_ino}:{info.st_size}:{info.st_mtime_ns}",
        model=policy.model if policy else None,
        reasoning_effort=policy.reasoning_effort if policy else None,
        config_digest=digest,
    )


def verify_observed_policy(cli_name: str, metadata: dict) -> None:
    """Retained compatibility hook; protected partners are attested pre-admission.

    Terminal CLI metadata can aggregate child agents. Once an exact protected
    launch is admitted it is greenlit, so no post-run model or effort value may
    invalidate it.
    """
    policy = get_policy(cli_name)
    if policy is None:
        return
    _ = metadata
