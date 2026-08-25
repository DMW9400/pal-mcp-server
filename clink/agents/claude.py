"""Claude-specific CLI agent hooks."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from clink.agents.base import CLIAgentError
from clink.models import ResolvedCLIRole
from clink.parsers.base import ParserError
from clink.policy import ClientCapability

from .base import AgentOutput, BaseCLIAgent


class ClaudeAgent(BaseCLIAgent):
    """Claude CLI agent with system-prompt injection support."""

    def _build_environment(self, capability: ClientCapability | None = None) -> dict[str, str]:
        env = super()._build_environment(capability)
        authority = self._partner_boundary_authority
        if authority is None:
            raise CLIAgentError("Claude direct partner launch requires the durable worker boundary authority")
        if capability is None or capability.model != "fable" or capability.reasoning_effort != "xhigh":
            raise CLIAgentError("Claude direct partner launch requires an attested fable/xhigh capability")
        self._partner_boundary_id, nonce = authority.issue_pending(capability.executable)
        self._partner_boundary_nonce = nonce
        env["PAL_CLINK_PARTNER_BOUNDARY_SOCKET"] = str(authority.socket_path)
        env["PAL_CLINK_PARTNER_BOUNDARY_ID"] = self._partner_boundary_id
        env["PAL_CLINK_PARTNER_BOUNDARY_NONCE"] = nonce
        return env

    def _activate_launch_context(self, process) -> None:
        self._partner_boundary_authority.activate(
            self._partner_boundary_id, self._partner_boundary_nonce, process.pid
        )

    def _cleanup_launch_context(self) -> None:
        authority = self._partner_boundary_authority
        if authority is not None:
            authority.cleanup(getattr(self, "_partner_boundary_id", None))
        self._partner_boundary_id = None
        self._partner_boundary_nonce = None

    def _build_command(
        self,
        *,
        role: ResolvedCLIRole,
        system_prompt: str | None,
        files: Sequence[str] = (),
    ) -> list[str]:
        command = list(self.client.executable)
        command.extend(self.client.internal_args)
        command.extend(self.client.config_args)

        preference = self.client.nested_agent_preference
        if preference is not None:
            advisory = (
                "PAL clink nested-agent preference (advisory only; never an admission or result gate): "
                f"prefer {preference.substantive_model} at {preference.substantive_effort} effort for substantive "
                "delegated investigation and review; "
                "use lightweight agents only for narrow mechanical retrieval. If that model is unavailable or "
                "you have a better task-specific choice, continue with your own routing decision."
            )
            system_prompt = f"{system_prompt.rstrip()}\n\n{advisory}" if system_prompt else advisory

        if system_prompt and "--append-system-prompt" not in self.client.config_args:
            command.extend(["--append-system-prompt", system_prompt])

        home = Path.home().resolve()
        add_directories: list[Path] = []
        seen: set[Path] = set()
        for raw_path in files:
            resolved = Path(raw_path).expanduser().resolve(strict=True)
            directory = resolved if resolved.is_dir() else resolved.parent
            try:
                directory.relative_to(home)
            except ValueError as exc:
                raise CLIAgentError(
                    f"Refusing Claude --add-dir grant outside the user home for attached path {raw_path!r}"
                ) from exc
            if directory == home:
                raise CLIAgentError(
                    f"Refusing broad Claude --add-dir grant for attached path {raw_path!r}"
                )
            if directory not in seen:
                seen.add(directory)
                add_directories.append(directory)
        for directory in add_directories:
            command.extend(["--add-dir", str(directory)])

        command.extend(role.role_args)
        return command

    def _recover_from_error(
        self,
        *,
        returncode: int,
        stdout: str,
        stderr: str,
        sanitized_command: list[str],
        duration_seconds: float,
        output_file_content: str | None,
    ) -> AgentOutput | None:
        try:
            parsed = self._parser.parse(stdout, stderr)
        except ParserError:
            return None

        return AgentOutput(
            parsed=parsed,
            sanitized_command=sanitized_command,
            returncode=returncode,
            stdout=stdout,
            stderr=stderr,
            duration_seconds=duration_seconds,
            parser_name=self._parser.name,
            output_file_content=output_file_content,
        )
