"""Execute configured CLI agents for the clink tool and parse output."""

from __future__ import annotations

import asyncio
import logging
import os
import re
import shlex
import signal
import tempfile
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from clink.constants import DEFAULT_STREAM_LIMIT
from clink.models import ResolvedCLIClient, ResolvedCLIRole
from clink.parsers import BaseParser, ParsedCLIResponse, ParserError, get_parser
from clink.policy import (
    PartnerModelPolicyError,
    attest_client,
    get_policy,
    validate_command_policy,
    validate_request_policy,
    verify_observed_policy,
)

logger = logging.getLogger("clink.agent")

# Seconds of complete stdio silence before a synthetic heartbeat event is emitted.
HEARTBEAT_INTERVAL_SECONDS = 20.0

_PLACEHOLDER_RE = re.compile(r"\{(\w+)\}")


def _template_part_regex(part: str) -> re.Pattern[str]:
    """Compile an argv template part into a matcher, with placeholders wild.

    'model_reasoning_effort="{effort}"' matches any effort value but never an
    unrelated ``-c`` payload, so stripping a pinned override cannot clobber
    other config arguments.
    """

    pieces: list[str] = []
    pos = 0
    for match in _PLACEHOLDER_RE.finditer(part):
        pieces.append(re.escape(part[pos : match.start()]))
        pieces.append(".*")
        pos = match.end()
    pieces.append(re.escape(part[pos:]))
    return re.compile("".join(pieces))


def render_arg_template(template: Sequence[str], values: dict[str, str]) -> list[str]:
    """Substitute {placeholders} in an argv template."""

    rendered: list[str] = []
    for part in template:
        for key, value in values.items():
            part = part.replace("{" + key + "}", value)
        rendered.append(part)
    return rendered


def strip_templated_args(args: Sequence[str], template: Sequence[str]) -> list[str]:
    """Remove arguments already matching ``template`` so an override replaces
    the configured pin instead of appending a second, conflicting flag."""

    if not template:
        return list(args)

    if len(template) == 1:
        pattern = _template_part_regex(template[0])
        return [arg for arg in args if not pattern.fullmatch(arg)]

    flag = template[0]
    value_pattern = _template_part_regex(template[1])
    out: list[str] = []
    index = 0
    while index < len(args):
        if args[index] == flag and index + 1 < len(args) and value_pattern.fullmatch(args[index + 1]):
            index += 2
            continue
        out.append(args[index])
        index += 1
    return out


# Async callback invoked for each subprocess output event so callers (e.g. the clink
# tool) can forward MCP progress notifications. Receives (kind, text) where kind is
# "stdout" or "stderr" and text is a single decoded line without trailing newline.
EventCallback = Callable[[str, str], Awaitable[None]]


@dataclass
class AgentOutput:
    """Container returned by CLI agents after successful execution."""

    parsed: ParsedCLIResponse
    sanitized_command: list[str]
    returncode: int
    stdout: str
    stderr: str
    duration_seconds: float
    parser_name: str
    output_file_content: str | None = None


class CLIAgentError(RuntimeError):
    """Raised when a CLI agent fails (non-zero exit, timeout, parse errors)."""

    def __init__(self, message: str, *, returncode: int | None = None, stdout: str = "", stderr: str = "") -> None:
        super().__init__(message)
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class BaseCLIAgent:
    """Execute a configured CLI command and parse its output."""

    #: Overridable so tests can exercise the heartbeat without waiting 20 seconds.
    heartbeat_interval_seconds: float = HEARTBEAT_INTERVAL_SECONDS

    def __init__(self, client: ResolvedCLIClient, *, partner_boundary_authority=None):
        self.client = client
        self._partner_boundary_authority = partner_boundary_authority
        self._parser: BaseParser = get_parser(client.parser)
        self._logger = logging.getLogger(f"clink.runner.{client.name}")

    async def run(
        self,
        *,
        role: ResolvedCLIRole,
        prompt: str,
        system_prompt: str | None = None,
        files: Sequence[str],
        images: Sequence[str],
        on_event: EventCallback | None = None,
        model: str | None = None,
        reasoning_effort: str | None = None,
    ) -> AgentOutput:
        # Files and images are already embedded into the prompt by the tool; they are
        # accepted here only to keep parity with SimpleTool callers.
        _ = (files, images)
        # The runner simply executes the configured CLI command for the selected role.
        command = self._build_command(role=role, system_prompt=system_prompt, files=files)
        command = self._apply_runtime_overrides(
            command,
            model=model,
            reasoning_effort=reasoning_effort,
        )
        try:
            validate_command_policy(self.client.name, command)
            capability = attest_client(self.client, role)
        except PartnerModelPolicyError as exc:
            raise CLIAgentError(str(exc)) from exc
        env = self._build_environment(capability)

        # Resolve executable path for cross-platform compatibility (especially Windows)
        command[0] = capability.executable

        sanitized_command = list(command)

        cwd = str(self.client.working_dir) if self.client.working_dir else None
        limit = DEFAULT_STREAM_LIMIT

        stdout_text = ""
        stderr_text = ""
        output_file_content: str | None = None
        start_time = time.monotonic()

        output_file_path: Path | None = None
        command_with_output_flag = list(command)

        if self.client.output_to_file:
            fd, tmp_path = tempfile.mkstemp(prefix="clink-", suffix=".json")
            os.close(fd)
            output_file_path = Path(tmp_path)
            flag_template = self.client.output_to_file.flag_template
            try:
                rendered_flag = flag_template.format(path=str(output_file_path))
            except KeyError as exc:  # pragma: no cover - defensive
                raise CLIAgentError(f"Invalid output flag template '{flag_template}': missing placeholder {exc}")
            command_with_output_flag.extend(shlex.split(rendered_flag))
            sanitized_command = list(command_with_output_flag)

        self._logger.debug("Executing CLI command: %s", " ".join(sanitized_command))
        if cwd:
            self._logger.debug("Working directory: %s", cwd)

        try:
            process = await asyncio.create_subprocess_exec(
                *command_with_output_flag,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
                limit=limit,
                env=env,
                start_new_session=os.name == "posix",
            )
        except OSError as exc:
            self._cleanup_launch_context()
            raise CLIAgentError(f"Could not start executable for CLI '{self.client.name}': {exc}") from exc

        try:
            self._activate_launch_context(process)
        except Exception as exc:
            await self._kill_subprocess(process)
            self._cleanup_launch_context()
            raise CLIAgentError(f"Could not activate CLI launch boundary: {exc}") from exc

        try:
            stdout_text, stderr_text = await asyncio.wait_for(
                self._stream_subprocess(process, prompt.encode("utf-8"), on_event),
                timeout=self.client.timeout_seconds,
            )
        except asyncio.CancelledError:
            await self._kill_subprocess(process)
            self._cleanup_launch_context()
            raise
        except asyncio.TimeoutError as exc:
            await self._kill_subprocess(process)
            self._cleanup_launch_context()
            raise CLIAgentError(
                f"CLI '{self.client.name}' timed out after {self.client.timeout_seconds} seconds",
                returncode=None,
            ) from exc

        duration = time.monotonic() - start_time
        return_code = process.returncode

        if output_file_path and output_file_path.exists():
            output_file_content = output_file_path.read_text(encoding="utf-8", errors="replace")
            if self.client.output_to_file and self.client.output_to_file.cleanup:
                try:
                    output_file_path.unlink()
                except OSError:  # pragma: no cover - best effort cleanup
                    pass

            if output_file_content and not stdout_text.strip():
                stdout_text = output_file_content

        if return_code != 0:
            recovered = self._recover_from_error(
                returncode=return_code,
                stdout=stdout_text,
                stderr=stderr_text,
                sanitized_command=sanitized_command,
                duration_seconds=duration,
                output_file_content=output_file_content,
            )
            if recovered is not None:
                self._verify_effective_policy(recovered.parsed, capability, return_code, stdout_text, stderr_text)
                self._cleanup_launch_context()
                return recovered

        if return_code != 0:
            self._cleanup_launch_context()
            raise CLIAgentError(
                f"CLI '{self.client.name}' exited with status {return_code}",
                returncode=return_code,
                stdout=stdout_text,
                stderr=stderr_text,
            )

        try:
            parsed = self._parser.parse(stdout_text, stderr_text)
        except ParserError as exc:
            self._cleanup_launch_context()
            raise CLIAgentError(
                f"Failed to parse output from CLI '{self.client.name}': {exc}",
                returncode=return_code,
                stdout=stdout_text,
                stderr=stderr_text,
            ) from exc

        self._verify_effective_policy(parsed, capability, return_code, stdout_text, stderr_text)
        self._cleanup_launch_context()

        return AgentOutput(
            parsed=parsed,
            sanitized_command=sanitized_command,
            returncode=return_code,
            stdout=stdout_text,
            stderr=stderr_text,
            duration_seconds=duration,
            parser_name=self._parser.name,
            output_file_content=output_file_content,
        )

    def _verify_effective_policy(
        self,
        parsed: ParsedCLIResponse,
        capability,
        return_code: int,
        stdout: str,
        stderr: str,
    ) -> None:
        """Bind direct-partner metadata to the already-attested exact command.

        A CLI's aggregate usage may include agents launched by the addressed
        partner. Those nested models are telemetry, not PAL policy subjects.
        """
        policy = get_policy(self.client.name)
        if policy is not None:
            cli_reported_model = parsed.metadata.pop("model_used", None)
            if cli_reported_model is not None:
                parsed.metadata.setdefault("models_used", [cli_reported_model])
            cli_reported_effort = parsed.metadata.pop("reasoning_effort_used", None)
            if cli_reported_effort is not None:
                parsed.metadata.setdefault("reasoning_efforts_used", [cli_reported_effort])
            parsed.metadata["model_used"] = capability.model
            parsed.metadata["reasoning_effort_used"] = capability.reasoning_effort
            parsed.metadata["policy_observation_source"] = "attested_command"
            parsed.metadata["post_admission_model_policy"] = "observe_only"
            if self.client.nested_agent_preference is not None:
                parsed.metadata["nested_agent_preference"] = self.client.nested_agent_preference.model_dump()
            return
        try:
            verify_observed_policy(self.client.name, parsed.metadata)
        except PartnerModelPolicyError as exc:
            raise CLIAgentError(
                str(exc),
                returncode=return_code,
                stdout=stdout,
                stderr=stderr,
            ) from exc

    async def _kill_subprocess(self, process) -> None:
        """Kill the CLI process group and reap it before propagating cancellation."""
        try:
            pid = getattr(process, "pid", None)
            if os.name == "posix" and pid:
                os.killpg(pid, signal.SIGKILL)
            else:
                process.kill()
        except (OSError, ProcessLookupError, AttributeError):
            try:
                process.kill()
            except (OSError, ProcessLookupError, AttributeError):
                pass
        try:
            await asyncio.wait_for(asyncio.shield(process.wait()), timeout=5)
        except asyncio.TimeoutError:  # pragma: no cover - defensive
            pass

    async def _stream_subprocess(
        self,
        process: asyncio.subprocess.Process,
        stdin_bytes: bytes,
        on_event: EventCallback | None,
    ) -> tuple[str, str]:
        """Feed stdin and concurrently drain stdout/stderr line-by-line.

        Each decoded line is appended to the in-memory buffer AND forwarded to
        on_event (if provided). This keeps continuous stdio activity flowing
        through the MCP boundary so subagent watchdogs don't kill long calls
        during quiet stretches of the wrapped CLI subprocess.

        A companion coroutine emits a synthetic "heartbeat" event after each
        interval of complete silence. Heartbeats are never appended to the
        captured stdout/stderr buffers.
        """
        stdout_chunks: list[str] = []
        stderr_chunks: list[str] = []
        activity = asyncio.Event()
        finished = asyncio.Event()

        async def feed_stdin() -> None:
            if process.stdin is None:  # pragma: no cover - defensive
                return
            try:
                process.stdin.write(stdin_bytes)
                await process.stdin.drain()
            except (BrokenPipeError, ConnectionResetError):
                # The child may exit before consuming stdin; that's fine.
                pass
            finally:
                try:
                    process.stdin.close()
                except Exception:  # pragma: no cover - defensive
                    pass

        async def drain_stream(
            reader: asyncio.StreamReader | None,
            chunks: list[str],
            kind: str,
        ) -> None:
            if reader is None:  # pragma: no cover - defensive
                return
            while True:
                try:
                    line = await reader.readline()
                except ValueError:
                    # Line exceeded the configured buffer limit. Fall back to
                    # bounded reads so we still consume the remaining output.
                    line = await reader.read(DEFAULT_STREAM_LIMIT)
                if not line:
                    break
                decoded = line.decode("utf-8", errors="replace")
                chunks.append(decoded)
                activity.set()
                if on_event is not None:
                    try:
                        await on_event(kind, decoded.rstrip("\r\n"))
                    except Exception:
                        # Never let progress-callback failures kill the run.
                        self._logger.debug("on_event callback raised", exc_info=True)

        async def emit_heartbeats() -> None:
            if on_event is None:
                return
            interval = self.heartbeat_interval_seconds
            if not interval or interval <= 0:
                return
            start = time.monotonic()
            while not finished.is_set():
                activity.clear()
                waiters = [
                    asyncio.ensure_future(activity.wait()),
                    asyncio.ensure_future(finished.wait()),
                ]
                try:
                    done, _pending = await asyncio.wait(
                        waiters,
                        timeout=interval,
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                finally:
                    for waiter in waiters:
                        waiter.cancel()
                    await asyncio.gather(*waiters, return_exceptions=True)
                if done:
                    # Real output (or completion) resets the silence window.
                    continue
                elapsed = int(time.monotonic() - start)
                try:
                    await on_event("heartbeat", f"{self.client.name} still running; {elapsed}s elapsed")
                except Exception:
                    self._logger.debug("on_event heartbeat raised", exc_info=True)

        heartbeat_task = asyncio.ensure_future(emit_heartbeats())
        try:
            await asyncio.gather(
                feed_stdin(),
                drain_stream(process.stdout, stdout_chunks, "stdout"),
                drain_stream(process.stderr, stderr_chunks, "stderr"),
            )
        finally:
            finished.set()
            heartbeat_task.cancel()
            try:
                await heartbeat_task
            except asyncio.CancelledError:
                pass
            except Exception:  # pragma: no cover - defensive
                self._logger.debug("heartbeat task failed", exc_info=True)
        await process.wait()
        return "".join(stdout_chunks), "".join(stderr_chunks)

    def _apply_runtime_overrides(
        self,
        command: Sequence[str],
        *,
        model: str | None,
        reasoning_effort: str | None,
    ) -> list[str]:
        """Apply per-call model / reasoning-effort overrides to a built command.

        Runs after ``_build_command`` so subclass command builders inherit it.
        The prompt is delivered on stdin, so appending flags is safe. Raises
        rather than silently ignoring an unsupported override — a silent
        fallback to the configured pin is the exact failure this guards.
        """

        try:
            validate_request_policy(self.client.name, model, reasoning_effort)
        except PartnerModelPolicyError as exc:
            raise CLIAgentError(str(exc)) from exc

        # Claude and Codex pins are immutable. Exact redundant values are
        # accepted for compatibility but never rewrite the attested command.
        if self.client.name.lower() in {"claude", "codex"}:
            return list(command)

        result = list(command)
        overrides: list[str] = []
        if model:
            template = self.client.model_arg_template
            if not template:
                raise CLIAgentError(f"CLI '{self.client.name}' does not support a per-call model override")
            result = strip_templated_args(result, template)
            overrides.extend(render_arg_template(template, {"model": model}))
        if reasoning_effort:
            template = self.client.reasoning_effort_arg_template
            if not template:
                raise CLIAgentError(f"CLI '{self.client.name}' does not support a per-call reasoning-effort override")
            result = strip_templated_args(result, template)
            overrides.extend(render_arg_template(template, {"effort": reasoning_effort}))
        result.extend(overrides)
        return result

    def _build_command(
        self,
        *,
        role: ResolvedCLIRole,
        system_prompt: str | None,
        files: Sequence[str] = (),
    ) -> list[str]:
        _ = files
        base = list(self.client.executable)
        base.extend(self.client.internal_args)
        base.extend(self.client.config_args)
        base.extend(role.role_args)

        return base

    def _build_environment(self, capability=None) -> dict[str, str]:
        _ = capability
        env = os.environ.copy()
        env.update(self.client.env)
        return env

    def _activate_launch_context(self, process) -> None:
        _ = process

    def _cleanup_launch_context(self) -> None:
        return None

    # ------------------------------------------------------------------
    # Error recovery hooks
    # ------------------------------------------------------------------

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
        """Hook for subclasses to convert CLI errors into successful outputs.

        Return an AgentOutput to treat the failure as success, or None to signal
        that normal error handling should proceed.
        """

        return None
