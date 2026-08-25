"""Pydantic models for clink configuration and runtime structures."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, PositiveInt, field_validator


class OutputCaptureConfig(BaseModel):
    """Optional configuration for CLIs that write output to disk."""

    flag_template: str = Field(..., description="Template used to inject the output path, e.g. '--output {path}'.")
    cleanup: bool = Field(
        default=True,
        description="Whether the temporary file should be removed after reading.",
    )


class CLIRoleConfig(BaseModel):
    """Role-specific configuration loaded from JSON manifests."""

    prompt_path: str | None = Field(
        default=None,
        description="Path to the prompt file that seeds this role.",
    )
    role_args: list[str] = Field(default_factory=list)
    description: str | None = Field(default=None)

    @field_validator("role_args", mode="before")
    @classmethod
    def _ensure_list(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [str(item) for item in value]
        if isinstance(value, str):
            return [value]
        raise TypeError("role_args must be a list of strings or a single string")


class NestedAgentPreference(BaseModel):
    """Advisory partner-owned delegation preference; never an admission gate."""

    substantive_model: str
    substantive_effort: str
    enforcement: Literal["none"] = "none"


class CLIClientConfig(BaseModel):
    """Raw CLI client configuration before internal defaults are applied."""

    name: str
    command: str | None = None
    working_dir: str | None = None
    additional_args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    timeout_seconds: PositiveInt | None = Field(default=None)
    roles: dict[str, CLIRoleConfig] = Field(default_factory=dict)
    output_to_file: OutputCaptureConfig | None = None
    model_arg_template: list[str] | None = Field(
        default=None,
        description="Argv template for a per-call model override, e.g. ['--model', '{model}']. Overrides the CLI's internal default.",
    )
    reasoning_effort_arg_template: list[str] | None = Field(
        default=None,
        description="Argv template for a per-call reasoning-effort override, e.g. ['-c', 'model_reasoning_effort=\"{effort}\"'].",
    )
    nested_agent_preference: NestedAgentPreference | None = None

    @field_validator("model_arg_template", "reasoning_effort_arg_template", mode="before")
    @classmethod
    def _ensure_template_list(cls, value: Any) -> list[str] | None:
        if value is None:
            return None
        if isinstance(value, list):
            return [str(item) for item in value]
        if isinstance(value, str):
            return [value]
        raise TypeError("arg templates must be a list of strings or a single string")

    @field_validator("additional_args", mode="before")
    @classmethod
    def _ensure_args_list(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [str(item) for item in value]
        if isinstance(value, str):
            return [value]
        raise TypeError("additional_args must be a list of strings or a single string")


class ResolvedCLIRole(BaseModel):
    """Runtime representation of a CLI role with resolved prompt path."""

    name: str
    prompt_path: Path
    role_args: list[str] = Field(default_factory=list)
    description: str | None = None


class ResolvedCLIClient(BaseModel):
    """Runtime configuration after merging defaults and validating paths."""

    name: str
    executable: list[str]
    working_dir: Path | None
    internal_args: list[str] = Field(default_factory=list)
    config_args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    timeout_seconds: int
    parser: str
    runner: str | None = None
    roles: dict[str, ResolvedCLIRole]
    output_to_file: OutputCaptureConfig | None = None
    model_arg_template: list[str] = Field(default_factory=list)
    reasoning_effort_arg_template: list[str] = Field(default_factory=list)
    nested_agent_preference: NestedAgentPreference | None = None

    def supports_model_override(self) -> bool:
        return bool(self.model_arg_template)

    def supports_reasoning_effort_override(self) -> bool:
        return bool(self.reasoning_effort_arg_template)

    def list_roles(self) -> list[str]:
        return list(self.roles.keys())

    def get_role(self, role_name: str | None) -> ResolvedCLIRole:
        key = role_name or "default"
        if key not in self.roles:
            available = ", ".join(sorted(self.roles.keys()))
            raise KeyError(f"Role '{role_name}' not configured for CLI '{self.name}'. Available roles: {available}")
        return self.roles[key]
