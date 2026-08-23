"""clink tool - bridge PAL MCP requests to external AI CLIs."""

from __future__ import annotations

import asyncio
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mcp.types import TextContent
from pydantic import BaseModel, Field

from clink import get_registry
from clink.agents import AgentOutput, CLIAgentError, create_agent
from clink.models import ResolvedCLIClient, ResolvedCLIRole
from config import TEMPERATURE_BALANCED
from tools.models import ToolModelCategory, ToolOutput
from tools.shared.base_models import COMMON_FIELD_DESCRIPTIONS
from tools.shared.exceptions import ToolExecutionError
from tools.simple.base import SchemaBuilder, SimpleTool

logger = logging.getLogger(__name__)

MAX_RESPONSE_CHARS = 20_000
SUMMARY_PATTERN = re.compile(r"<SUMMARY>(.*?)</SUMMARY>", re.IGNORECASE | re.DOTALL)


class CLinkRequest(BaseModel):
    """Request model for clink tool."""

    prompt: str = Field(..., description="Prompt forwarded to the target CLI.")
    cli_name: str | None = Field(
        default=None,
        description="Configured CLI client name to invoke. Defaults to the first configured CLI if omitted.",
    )
    role: str | None = Field(
        default=None,
        description="Optional role preset defined in the CLI configuration (defaults to 'default').",
    )
    absolute_file_paths: list[str] = Field(
        default_factory=list,
        description=COMMON_FIELD_DESCRIPTIONS["absolute_file_paths"],
    )
    images: list[str] = Field(
        default_factory=list,
        description=COMMON_FIELD_DESCRIPTIONS["images"],
    )
    continuation_id: str | None = Field(
        default=None,
        description=COMMON_FIELD_DESCRIPTIONS["continuation_id"],
    )
    model: str | None = Field(
        default=None,
        description="Override the CLI's configured model for this call (e.g. 'gpt-5.6-terra').",
    )
    reasoning_effort: str | None = Field(
        default=None,
        description="Override the CLI's configured reasoning effort for this call (e.g. 'low', 'medium', 'high').",
    )


class CLinkTool(SimpleTool):
    """Bridge MCP requests to configured CLI agents.

    Schema metadata is cached at construction time and execution relies on the shared
    SimpleTool hooks for conversation memory. Prompt preparation is customised so we
    pass instructions and file references suitable for another CLI agent.
    """

    def __init__(self) -> None:
        # Cache registry metadata so the schema surfaces concrete enum values.
        self._registry = get_registry()
        self._cli_names = self._registry.list_clients()
        self._role_map: dict[str, list[str]] = {name: self._registry.list_roles(name) for name in self._cli_names}
        self._all_roles: list[str] = sorted({role for roles in self._role_map.values() for role in roles})
        if "gemini" in self._cli_names:
            self._default_cli_name = "gemini"
        else:
            self._default_cli_name = self._cli_names[0] if self._cli_names else None
        self._active_system_prompt: str = ""
        self._background_tasks: set[asyncio.Task] = set()
        super().__init__()

    def get_name(self) -> str:
        return "clink"

    def get_description(self) -> str:
        return (
            "Link a request to an external AI CLI (Claude Code, Codex CLI, Gemini CLI, etc.) through PAL MCP "
            "to reuse their capabilities inside existing workflows."
        )

    def get_annotations(self) -> dict[str, Any]:
        return {"readOnlyHint": True}

    def requires_model(self) -> bool:
        return False

    def get_model_category(self) -> ToolModelCategory:
        return ToolModelCategory.BALANCED

    def get_default_temperature(self) -> float:
        return TEMPERATURE_BALANCED

    def get_system_prompt(self) -> str:
        return self._active_system_prompt or ""

    def get_request_model(self):
        return CLinkRequest

    def get_input_schema(self) -> dict[str, Any]:
        # Surface configured CLI names and roles directly in the schema so MCP clients
        # (and downstream agents) can discover available options without consulting
        # a separate registry call.
        role_descriptions = []
        for name in self._cli_names:
            roles = ", ".join(sorted(self._role_map.get(name, ["default"]))) or "default"
            role_descriptions.append(f"{name}: {roles}")

        if role_descriptions:
            cli_labels = []
            for name in self._cli_names:
                client = self._registry.get_client(name)
                partner_model = self._configured_partner_model(client)
                cli_labels.append(f"{name} (partner model: {partner_model})" if partner_model else name)
            cli_available = ", ".join(cli_labels) if cli_labels else "(none configured)"
            default_text = (
                f" Default: {self._default_cli_name}." if self._default_cli_name and len(self._cli_names) <= 1 else ""
            )
            cli_description = (
                "Configured CLI client name (from conf/cli_clients). Available: " + cli_available + default_text
            )
            role_description = (
                "Optional role preset defined for the selected CLI (defaults to 'default'). Roles per CLI: "
                + "; ".join(role_descriptions)
            )
        else:
            cli_description = "Configured CLI client name (from conf/cli_clients)."
            role_description = "Optional role preset defined for the selected CLI (defaults to 'default')."

        properties = {
            "prompt": {
                "type": "string",
                "description": "User request forwarded to the CLI (conversation context is pre-applied).",
            },
            "cli_name": {
                "type": "string",
                "enum": self._cli_names,
                "description": cli_description,
            },
            "role": {
                "type": "string",
                "enum": self._all_roles or ["default"],
                "description": role_description,
            },
            "absolute_file_paths": SchemaBuilder.SIMPLE_FIELD_SCHEMAS["absolute_file_paths"],
            "images": SchemaBuilder.COMMON_FIELD_SCHEMAS["images"],
            "continuation_id": SchemaBuilder.COMMON_FIELD_SCHEMAS["continuation_id"],
            "model": {
                "type": "string",
                "description": (
                    "Optional per-call model override for the selected CLI, replacing its configured "
                    "pin for this call only (e.g. 'gpt-5.6-terra'). Omit to use the configured model."
                ),
            },
            "reasoning_effort": {
                "type": "string",
                "description": (
                    "Optional per-call reasoning-effort override for the selected CLI, replacing its "
                    "configured pin for this call only (commonly 'low', 'medium', or 'high'). "
                    "Omit to use the configured effort."
                ),
            },
        }

        schema = {
            "type": "object",
            "properties": properties,
            "required": ["prompt"],
            "additionalProperties": False,
        }

        if len(self._cli_names) > 1:
            schema["required"].append("cli_name")

        return schema

    def get_tool_fields(self) -> dict[str, dict[str, Any]]:
        """Unused by clink because we override the schema end-to-end."""
        return {}

    async def execute(self, arguments: dict[str, Any]) -> list[TextContent]:
        self._current_arguments = arguments
        request = self.get_request_model()(**arguments)

        path_error = self._validate_file_paths(request)
        if path_error:
            self._raise_tool_error(path_error)

        selected_cli = request.cli_name or self._default_cli_name
        if not selected_cli:
            self._raise_tool_error("No CLI clients are configured for clink.")

        try:
            client_config = self._registry.get_client(selected_cli)
        except KeyError as exc:
            self._raise_tool_error(str(exc))

        try:
            role_config = client_config.get_role(request.role)
        except KeyError as exc:
            self._raise_tool_error(str(exc))

        absolute_file_paths = self.get_request_files(request)
        images = self.get_request_images(request)
        continuation_id = self.get_request_continuation_id(request)

        self._model_context = arguments.get("_model_context")

        system_prompt_text = role_config.prompt_path.read_text(encoding="utf-8")
        include_system_prompt = not self._use_external_system_prompt(client_config)

        try:
            prompt_text = await self._prepare_prompt_for_role(
                request,
                client_config,
                role_config,
                system_prompt=system_prompt_text,
                include_system_prompt=include_system_prompt,
            )
        except Exception as exc:
            logger.exception("Failed to prepare clink prompt")
            self._raise_tool_error(f"Failed to prepare prompt: {exc}")

        agent = create_agent(client_config)
        on_event = self._build_progress_callback(client_config.name)

        # Ensure the conversation thread exists BEFORE the CLI launches. MCP clients
        # can time out and cancel long-running calls; with the thread (and its ID in
        # the activity log) created up front, the salvage path below can attach the
        # CLI's eventual result to it, making a timed-out call recoverable via a
        # follow-up request with this continuation_id.
        thread_id = continuation_id or self._create_recovery_thread(request)
        self._log_run_started(client_config.name, thread_id)

        run_task = asyncio.create_task(
            agent.run(
                role=role_config,
                prompt=prompt_text,
                system_prompt=system_prompt_text if system_prompt_text.strip() else None,
                files=absolute_file_paths,
                images=images,
                on_event=on_event,
                model=request.model,
                reasoning_effort=request.reasoning_effort,
            )
        )
        try:
            result = await asyncio.shield(run_task)
        except asyncio.CancelledError:
            self._salvage_cancelled_run(run_task, client_config, request, thread_id)
            raise
        except CLIAgentError as exc:
            metadata = self._build_error_metadata(client_config, exc)
            if thread_id:
                metadata["continuation_id"] = thread_id
            self._raise_tool_error(
                f"CLI '{client_config.name}' execution failed: {exc}",
                metadata=metadata,
            )

        metadata = self._build_success_metadata(client_config, role_config, result)
        metadata = self._prune_metadata(metadata, client_config, reason="normal")

        content, metadata = self._apply_output_limit(
            client_config,
            result.parsed.content,
            metadata,
        )

        model_info = {
            "provider": client_config.name,
            "model_name": result.parsed.metadata.get("model_used"),
        }

        if continuation_id:
            try:
                self._record_assistant_turn(continuation_id, content, request, model_info)
            except Exception:
                logger.debug("Failed to record assistant turn for continuation %s", continuation_id, exc_info=True)

        continuation_offer = self._continuation_offer_for_thread(thread_id)
        if continuation_offer:
            tool_output = self._create_continuation_offer_response(
                content,
                continuation_offer,
                request,
                model_info,
            )
            tool_output.metadata = self._merge_metadata(tool_output.metadata, metadata)
        else:
            tool_output = ToolOutput(
                status="success",
                content=content,
                content_type="text",
                metadata=metadata,
            )

        return [TextContent(type="text", text=tool_output.model_dump_json())]

    def _create_recovery_thread(self, request: CLinkRequest) -> str | None:
        """Create the conversation thread (with the user's turn) ahead of the CLI run.

        Historically the thread was created only after the CLI returned, so a call
        cancelled by the MCP client left no continuation to resume. Creating it up
        front means the ID is known (and logged) for the whole lifetime of the run.
        """
        try:
            from utils.conversation_memory import add_turn, create_thread

            thread_id = create_thread(tool_name=self.get_name(), initial_request=self.get_request_as_dict(request))
            add_turn(
                thread_id,
                "user",
                self.get_request_prompt(request),
                files=self.get_request_files(request),
                images=self.get_request_images(request),
                tool_name=self.get_name(),
            )
            return thread_id
        except Exception:
            logger.warning("Failed to create clink conversation thread ahead of CLI launch", exc_info=True)
            return None

    def _continuation_offer_for_thread(self, thread_id: str | None) -> dict[str, Any] | None:
        """Build a continuation offer from the already-existing thread."""
        if not thread_id:
            return None
        try:
            from utils.conversation_memory import MAX_CONVERSATION_TURNS, get_thread

            context = get_thread(thread_id)
            if context is None:
                return None
            turn_count = len(context.turns)
            if turn_count >= MAX_CONVERSATION_TURNS - 1:
                return None
            remaining_turns = MAX_CONVERSATION_TURNS - turn_count - 1
            return {
                "continuation_id": thread_id,
                "remaining_turns": remaining_turns,
                "note": f"You can continue this conversation for {remaining_turns} more exchanges.",
            }
        except Exception:
            return None

    def _log_run_started(self, cli_name: str, thread_id: str | None) -> None:
        if not thread_id:
            return
        message = f"CLINK_RUN_STARTED: cli={cli_name} continuation_id={thread_id}"
        logger.info(message)
        try:
            logging.getLogger("mcp_activity").info(message)
        except Exception:  # pragma: no cover - logging must never break execution
            pass

    def _salvage_cancelled_run(
        self,
        run_task: asyncio.Task,
        client: ResolvedCLIClient,
        request: CLinkRequest,
        thread_id: str | None,
    ) -> None:
        """Supervise a CLI run whose MCP request was cancelled (e.g. client timeout).

        The run task is shielded from the request's cancellation, so the CLI child
        keeps running here under supervision: the agent's own wait_for still kills
        the process at the configured timeout, and if the run completes its result
        is recorded on the pre-created continuation thread so a follow-up call with
        that continuation_id retrieves the work instead of losing it.

        Must stay synchronous: it runs inside an already-cancelled scope where any
        await would immediately re-raise CancelledError.
        """
        cli_name = client.name
        notice = (
            f"CLINK_CANCELLED: cli={cli_name} continuation_id={thread_id or 'unavailable'} - "
            "letting the CLI finish in the background; the result will be stored on the continuation thread"
        )
        logger.warning(notice)
        try:
            logging.getLogger("mcp_activity").info(notice)
        except Exception:  # pragma: no cover - logging must never break execution
            pass

        async def _salvage() -> None:
            try:
                result = await run_task
            except asyncio.CancelledError:
                return
            except CLIAgentError as exc:
                logger.warning("clink CLI '%s' failed after client cancellation: %s", cli_name, exc)
                return
            except Exception:
                logger.warning("clink CLI '%s' raised after client cancellation", cli_name, exc_info=True)
                return

            if not thread_id:
                logger.warning(
                    "clink CLI '%s' finished after client cancellation but no continuation thread exists; "
                    "its result was dropped",
                    cli_name,
                )
                return

            model_info = {
                "provider": cli_name,
                "model_name": result.parsed.metadata.get("model_used"),
            }
            try:
                self._record_assistant_turn(thread_id, result.parsed.content, request, model_info)
            except Exception:
                logger.warning("Failed to store salvaged clink result for continuation %s", thread_id, exc_info=True)
                return

            salvaged = (
                f"CLINK_SALVAGED: cli={cli_name} continuation_id={thread_id} "
                f"duration={result.duration_seconds:.1f}s - call clink with this continuation_id to retrieve the result"
            )
            logger.info(salvaged)
            try:
                logging.getLogger("mcp_activity").info(salvaged)
            except Exception:  # pragma: no cover - logging must never break execution
                pass

        task = asyncio.create_task(_salvage())
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    async def prepare_prompt(self, request) -> str:
        selected_cli = request.cli_name or self._default_cli_name
        client_config = self._registry.get_client(selected_cli)
        role_config = client_config.get_role(request.role)
        system_prompt_text = role_config.prompt_path.read_text(encoding="utf-8")
        include_system_prompt = not self._use_external_system_prompt(client_config)
        return await self._prepare_prompt_for_role(
            request,
            client_config,
            role_config,
            system_prompt=system_prompt_text,
            include_system_prompt=include_system_prompt,
        )

    async def _prepare_prompt_for_role(
        self,
        request: CLinkRequest,
        client: ResolvedCLIClient,
        role: ResolvedCLIRole,
        *,
        system_prompt: str,
        include_system_prompt: bool,
    ) -> str:
        """Load the role prompt and assemble the final user message."""
        self._active_system_prompt = system_prompt
        try:
            user_content = self.handle_prompt_file_with_fallback(request).strip()
            guidance = self._agent_capabilities_guidance(client)
            file_section = self._format_file_references(self.get_request_files(request))

            sections: list[str] = []
            active_prompt = self.get_system_prompt().strip()
            if include_system_prompt and active_prompt:
                sections.append(active_prompt)
            sections.append(guidance)
            sections.append("=== USER REQUEST ===\n" + user_content)
            if file_section:
                sections.append("=== FILE REFERENCES ===\n" + file_section)
            sections.append("Provide your response below using your own CLI tools as needed:")
            return "\n\n".join(sections)
        finally:
            self._active_system_prompt = ""

    def _use_external_system_prompt(self, client: ResolvedCLIClient) -> bool:
        runner_name = (client.runner or client.name).lower()
        return runner_name == "claude"

    def _build_success_metadata(
        self,
        client: ResolvedCLIClient,
        role: ResolvedCLIRole,
        result: AgentOutput,
    ) -> dict[str, Any]:
        """Capture execution metadata for successful CLI calls."""
        metadata: dict[str, Any] = {
            "cli_name": client.name,
            "role": role.name,
            "command": result.sanitized_command,
            "duration_seconds": round(result.duration_seconds, 3),
            "parser": result.parser_name,
            "return_code": result.returncode,
        }
        partner_model = self._configured_partner_model(client)
        if partner_model:
            metadata["partner_model"] = partner_model
        metadata.update(result.parsed.metadata)

        if result.stderr.strip():
            metadata.setdefault("stderr", result.stderr.strip())
        if result.output_file_content and "raw" not in metadata:
            metadata["raw_output_file"] = result.output_file_content
        return metadata

    def _merge_metadata(self, base: dict[str, Any] | None, extra: dict[str, Any]) -> dict[str, Any]:
        merged = dict(base or {})
        merged.update(extra)
        return merged

    def _apply_output_limit(
        self,
        client: ResolvedCLIClient,
        content: str,
        metadata: dict[str, Any],
    ) -> tuple[str, dict[str, Any]]:
        if len(content) <= MAX_RESPONSE_CHARS:
            return content, metadata

        summary = self._extract_summary(content)
        if summary:
            summary_text = summary
            if len(summary_text) > MAX_RESPONSE_CHARS:
                logger.debug(
                    "Clink summary from %s exceeded %d chars; truncating summary to fit.",
                    client.name,
                    MAX_RESPONSE_CHARS,
                )
                summary_text = summary_text[:MAX_RESPONSE_CHARS]
            summary_metadata = self._prune_metadata(metadata, client, reason="summary")
            summary_metadata.update(
                {
                    "output_summarized": True,
                    "output_original_length": len(content),
                    "output_summary_length": len(summary_text),
                    "output_limit": MAX_RESPONSE_CHARS,
                }
            )
            logger.info(
                "Clink compressed %s output via <SUMMARY>: original=%d chars, summary=%d chars",
                client.name,
                len(content),
                len(summary_text),
            )
            return summary_text, summary_metadata

        truncated_metadata = self._prune_metadata(metadata, client, reason="truncated")
        truncated_metadata.update(
            {
                "output_truncated": True,
                "output_original_length": len(content),
                "output_limit": MAX_RESPONSE_CHARS,
            }
        )

        excerpt_limit = min(4000, MAX_RESPONSE_CHARS // 2)
        excerpt = content[:excerpt_limit]
        truncated_metadata["output_excerpt_length"] = len(excerpt)

        logger.warning(
            "Clink truncated %s output: original=%d chars exceeds limit=%d; excerpt_length=%d",
            client.name,
            len(content),
            MAX_RESPONSE_CHARS,
            len(excerpt),
        )

        message = (
            f"CLI '{client.name}' produced {len(content)} characters, exceeding the configured clink limit "
            f"({MAX_RESPONSE_CHARS} characters). The full output was suppressed to stay within MCP response caps. "
            "Please narrow the request (review fewer files, summarize results) or run the CLI directly for the full log.\n\n"
            f"--- Begin excerpt ({len(excerpt)} of {len(content)} chars) ---\n{excerpt}\n--- End excerpt ---"
        )

        return message, truncated_metadata

    def _extract_summary(self, content: str) -> str | None:
        match = SUMMARY_PATTERN.search(content)
        if not match:
            return None
        summary = match.group(1).strip()
        return summary or None

    def _prune_metadata(
        self,
        metadata: dict[str, Any],
        client: ResolvedCLIClient,
        *,
        reason: str,
    ) -> dict[str, Any]:
        cleaned = dict(metadata)
        events = cleaned.pop("events", None)
        if events is not None:
            cleaned[f"events_removed_for_{reason}"] = True
            logger.debug(
                "Clink dropped %s events metadata for %s response (%s)",
                client.name,
                reason,
                type(events).__name__,
            )
        return cleaned

    def _build_error_metadata(self, client: ResolvedCLIClient, exc: CLIAgentError) -> dict[str, Any]:
        """Assemble metadata for failed CLI calls."""
        metadata: dict[str, Any] = {
            "cli_name": client.name,
            "return_code": exc.returncode,
        }
        partner_model = self._configured_partner_model(client)
        if partner_model:
            metadata["partner_model"] = partner_model
        if exc.stdout:
            metadata["stdout"] = exc.stdout.strip()
        if exc.stderr:
            metadata["stderr"] = exc.stderr.strip()
        return metadata

    def _raise_tool_error(self, message: str, metadata: dict[str, Any] | None = None) -> None:
        error_output = ToolOutput(status="error", content=message, content_type="text", metadata=metadata)
        raise ToolExecutionError(error_output.model_dump_json())

    def _agent_capabilities_guidance(self, client: ResolvedCLIClient) -> str:
        runner_name = (client.runner or client.name).lower()
        partner_model = self._configured_partner_model(client)
        display_names = {
            "claude": "Claude Code CLI agent",
            "codex": "Codex CLI agent",
            "gemini": "Gemini CLI agent",
        }
        display_name = display_names.get(runner_name, f"{client.name} CLI agent")

        capability_notes = {
            "claude": (
                "Use Claude Code's repository tools to inspect files, run focused commands, reason about plans and "
                "reviews, and make edits only when the role prompt or user request explicitly calls for implementation."
            ),
            "codex": (
                "Use Codex CLI's repository tools to inspect files, run focused commands, review code, and make edits "
                "only when the role prompt or user request explicitly calls for implementation."
            ),
            "gemini": (
                "Use Gemini CLI's available tools, including file access, shell commands, and web/search capabilities "
                "when they are available and relevant."
            ),
        }
        capability_note = capability_notes.get(
            runner_name,
            "Use your available CLI tools to inspect files, run focused commands, and gather the context needed.",
        )

        partner_identity = (
            f" You are the Claude {partner_model.title()} Clink partner model."
            if runner_name == "claude" and partner_model
            else ""
        )

        return (
            f"You are operating through the {display_name}.{partner_identity} {capability_note} "
            "You are collaborating with the calling PAL MCP host agent; gather needed context yourself and deliver "
            "the final answer directly without asking the host to perform searches or file reads."
        )

    @staticmethod
    def _configured_partner_model(client: ResolvedCLIClient) -> str | None:
        """Return the one explicit client-level model selection, if valid."""
        model_positions = [index for index, arg in enumerate(client.config_args) if arg == "--model"]
        if len(model_positions) != 1:
            return None
        model_index = model_positions[0] + 1
        if model_index >= len(client.config_args):
            return None
        model_name = client.config_args[model_index].strip()
        if not model_name or model_name.startswith("-"):
            return None
        return model_name

    def _build_progress_callback(self, cli_name: str):
        """Return an async callback that forwards CLI stdio lines as MCP progress.

        Streaming progress notifications keep the wrapper subagent's stream watchdog
        alive during long codex/gemini runs: without them, ~600s of stdio silence on
        the subagent process would cause the harness to kill the subagent before the
        wrapped CLI finishes.

        Returns None when no progress token was supplied by the client (older clients,
        or when the tool is invoked outside an MCP request scope, e.g. in unit tests).
        Errors during context lookup or notification dispatch are swallowed so they
        never abort the underlying CLI run.
        """
        try:
            from server import server as mcp_server
        except Exception:
            return None

        try:
            request_context = mcp_server.request_context
        except (AttributeError, LookupError):
            return None

        meta = getattr(request_context, "meta", None)
        progress_token = getattr(meta, "progressToken", None) if meta else None
        session = getattr(request_context, "session", None)
        if progress_token is None or session is None:
            return None

        send_progress = getattr(session, "send_progress_notification", None)
        if send_progress is None:
            return None

        progress_counter = 0
        max_message_chars = 240

        async def on_event(kind: str, text: str) -> None:
            nonlocal progress_counter
            progress_counter += 1
            stripped = text.strip()
            if not stripped:
                return
            if len(stripped) > max_message_chars:
                stripped = stripped[:max_message_chars] + "…"
            message = f"[{cli_name}:{kind}] {stripped}"
            try:
                await send_progress(
                    progress_token=progress_token,
                    progress=float(progress_counter),
                    message=message,
                )
            except Exception:
                logger.debug("Failed to send progress notification", exc_info=True)

        return on_event

    def _format_file_references(self, files: list[str]) -> str:
        if not files:
            return ""

        references: list[str] = []
        for file_path in files:
            try:
                path = Path(file_path)
                stat = path.stat()
                modified = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()
                size = stat.st_size
                references.append(f"- {file_path} (last modified {modified}, {size} bytes)")
            except OSError:
                references.append(f"- {file_path} (unavailable)")
        return "\n".join(references)
