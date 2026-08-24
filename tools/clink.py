"""clink tool - bridge PAL MCP requests to external AI CLIs."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mcp.types import TextContent
from pydantic import BaseModel, Field

from clink import get_registry, jobs
from clink.agents import AgentOutput, CLIAgentError, create_agent
from clink.models import ResolvedCLIClient, ResolvedCLIRole
from clink.policy import PartnerModelPolicyError, attest_client, get_policy, validate_request_policy
from config import TEMPERATURE_BALANCED
from tools.models import ContinuationOffer, ToolModelCategory, ToolOutput
from tools.shared.base_models import COMMON_FIELD_DESCRIPTIONS
from tools.shared.exceptions import ToolExecutionError
from tools.simple.base import SchemaBuilder, SimpleTool

logger = logging.getLogger(__name__)

MAX_RESPONSE_CHARS = 20_000
SUMMARY_PATTERN = re.compile(r"<SUMMARY>(.*?)</SUMMARY>", re.IGNORECASE | re.DOTALL)

# Metadata blobs that are echoed in-band but never kept in the durable record:
# unbounded raw CLI payloads and diagnostic streams. ``raw_output_file`` is where
# _build_success_metadata stores result.output_file_content; ``raw``/``raw_events``
# are whole CLI responses from the gemini and claude parsers; ``events`` is the
# codex parser's event list (normally pruned in-band, listed here as a backstop).
SIDECAR_REDACTED_METADATA_KEYS = frozenset(
    {"stdout", "stderr", "output_file_content", "raw_output_file", "raw", "raw_events", "events"}
)
SIDECAR_REDACTION_PLACEHOLDER = "<omitted from durable record>"

BACKGROUND_FIELD_DESCRIPTION = (
    "Return immediately with a run_id and continuation_id instead of waiting for the CLI to finish, "
    "then poll the run with the clink_poll tool. Use this for work expected to take longer than the "
    "MCP client's request timeout."
)


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
        description=(
            "Optional exact-policy assertion. Claude accepts only 'fable'; Codex accepts only "
            "'gpt-5.6-sol'. Conflicting overrides are rejected before admission."
        ),
    )
    reasoning_effort: str | None = Field(
        default=None,
        description=(
            "Optional exact-policy assertion. Claude accepts only 'xhigh'; Codex accepts only 'high'. "
            "Conflicting overrides are rejected before admission."
        ),
    )
    background: bool | None = Field(
        default=False,
        description=BACKGROUND_FIELD_DESCRIPTION,
    )
    idempotency_key: str | None = Field(
        default=None,
        min_length=8,
        max_length=256,
        description="Stable caller key for retry-safe durable exchange admission.",
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

    async def preflight_continuation(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Attest clink readiness before server middleware mutates a thread."""
        request = CLinkRequest(**arguments)
        cli_name = request.cli_name or self._default_cli_name
        if not cli_name:
            raise ToolExecutionError("No clink CLI client is configured")
        client = self._registry.get_client(cli_name)
        role = client.get_role(request.role)
        try:
            validate_request_policy(client.name, request.model, request.reasoning_effort)
            capability = attest_client(client, role)
        except PartnerModelPolicyError as exc:
            self._raise_tool_error(f"Clink execution owner is not ready: {exc}")
        if capability:
            try:
                from utils.conversation_memory import get_storage

                storage = get_storage()
                from utils.sqlite_conversation_storage import SQLiteConversationStorage

                if isinstance(storage, SQLiteConversationStorage):
                    worker = storage.get_fresh_worker_capability(capability.cli_name, role.name)
                    worker_matches = bool(
                        worker
                        and worker["config_digest"] == capability.config_digest
                        and worker["executable_identity"] == capability.executable_identity
                        and worker["model"] == capability.model
                        and worker["reasoning_effort"] == capability.reasoning_effort
                    )
                    if worker_matches:
                        arguments["_execution_owner"] = "worker"
                        arguments["_capability_digest"] = capability.config_digest
                        return arguments
                    storage.publish_capability(
                        cli_name=capability.cli_name,
                        role=role.name,
                        config_digest=capability.config_digest,
                        executable_identity=capability.executable_identity,
                        model=capability.model,
                        reasoning_effort=capability.reasoning_effort,
                        owner_instance_id=jobs.INSTANCE_ID,
                        owner_mode="pal",
                    )
                    arguments["_capability_digest"] = capability.config_digest
                    arguments["_execution_owner"] = "pal"
                    if request.background:
                        self._raise_tool_error(
                            "Background clink requires a fresh independently supervised worker; "
                            "no turn or model cost was consumed"
                        )
            except ToolExecutionError:
                raise
            except Exception as exc:
                self._raise_tool_error(f"Clink durable execution capability could not be published: {exc}")
        return arguments

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
                    "Optional exact-policy assertion. Claude accepts only fable; Codex accepts only "
                    "gpt-5.6-sol. Conflicting values are rejected before admission."
                ),
            },
            "reasoning_effort": {
                "type": "string",
                "description": (
                    "Optional exact-policy assertion. Claude accepts only xhigh; Codex accepts only high. "
                    "Conflicting values are rejected before admission."
                ),
            },
            "background": {
                "type": "boolean",
                "default": False,
                "description": BACKGROUND_FIELD_DESCRIPTION,
            },
            "idempotency_key": {
                "type": "string",
                "minLength": 8,
                "maxLength": 256,
                "description": "Stable caller key for retry-safe durable exchange admission.",
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
        execution_owner = arguments.get("_execution_owner", "pal")
        admitted_capability_digest = arguments.get("_capability_digest")
        request = self.get_request_model()(**arguments)

        if arguments.get("_idempotent_replay"):
            return await self._return_idempotent_replay(request)

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

        # Fail closed before creating a recovery thread, appending a turn, or
        # launching a paid CLI process. The capability digest binds the exact
        # executable identity, role/config and immutable partner-model policy.
        try:
            validate_request_policy(client_config.name, request.model, request.reasoning_effort)
            capability = (
                attest_client(client_config, role_config)
                if execution_owner == "worker" or get_policy(client_config.name)
                else None
            )
        except PartnerModelPolicyError as exc:
            self._raise_tool_error(f"Clink execution owner is not ready: {exc}")
        if execution_owner == "worker" and (
            capability is None or admitted_capability_digest != capability.config_digest
        ):
            self._raise_tool_error("Clink worker capability changed after admission; no process was launched")

        # Ensure the conversation thread and the run identity exist BEFORE the CLI
        # launches and before the progress callback closure is built. MCP clients can
        # time out and cancel long-running calls; with both IDs known up front the run
        # keeps going under supervision and stays retrievable through clink_poll.
        thread_id = continuation_id or self._create_recovery_thread(
            request, capability_digest=admitted_capability_digest
        )
        if arguments.get("_idempotent_replay"):
            return await self._return_idempotent_replay(request)
        run_id = jobs.new_run_id()
        background = bool(request.background)

        # No await may separate the claim from the task that owns its release: a
        # cancellation in that window would leak the claim for the process lifetime.
        if thread_id and execution_owner != "worker":
            active_run_id = jobs.claim(thread_id, run_id)
            if active_run_id != run_id:
                return self._active_run_conflict(client_config.name, thread_id, active_run_id)

        if execution_owner == "worker":
            try:
                from utils.conversation_memory import current_exchange_id, get_storage

                storage = get_storage()
                envelope = {
                    "schema_version": 1,
                    "run_id": run_id,
                    "thread_id": thread_id,
                    "continuation_id": continuation_id,
                    "exchange_id": current_exchange_id.get(),
                    "cli_name": client_config.name,
                    "role": role_config.name,
                    "capability_digest": capability.config_digest,
                    "request": request.model_dump(mode="json"),
                    "prompt_text": prompt_text,
                    "system_prompt_text": system_prompt_text,
                    "absolute_file_paths": absolute_file_paths,
                    "images": images,
                }
                storage.create_queued_worker_run(
                    run_id=run_id,
                    continuation_id=thread_id,
                    exchange_id=current_exchange_id.get(),
                    cli_name=client_config.name,
                    role=role_config.name,
                    envelope=envelope,
                )
            except Exception as exc:
                logger.warning("Failed to enqueue clink worker run %s", run_id, exc_info=True)
                self._raise_tool_error(f"Clink worker queue admission failed: {exc}")

            self._log_run_started(
                client_config.name,
                thread_id,
                run_id,
                background=background,
                has_progress_token=False,
                durable=True,
            )
            if background:
                output = self._background_started_output(client_config.name, thread_id, run_id)
                return [TextContent(type="text", text=output.model_dump_json())]
            try:
                tool_output = await self._await_worker_result(run_id)
            except asyncio.CancelledError:
                self._detach_cancelled_run(client_config.name, thread_id, run_id)
                raise
            return [TextContent(type="text", text=tool_output.model_dump_json())]

        agent = create_agent(client_config)
        durable = True
        try:
            jobs.create(
                run_id=run_id,
                continuation_id=thread_id,
                cli_name=client_config.name,
                role=role_config.name,
            )
            durable = jobs.mark_running(run_id) is not None
        except Exception:
            durable = False
            logger.warning("Failed to create clink job record %s", run_id, exc_info=True)

        if not durable:
            if background:
                # Background's whole contract is the sidecar; without it the caller
                # would be told to poll a record that does not exist.
                jobs.release(thread_id, run_id)
                logger.warning("Refusing clink background run %s: no durable record could be written", run_id)
                self._raise_durable_store_error(client_config.name, thread_id, run_id)
            logger.warning(
                "clink run %s has no durable record; the in-band response still applies but clink_poll "
                "will not find this run",
                run_id,
            )

        on_event = self._build_progress_callback(client_config.name, run_id=run_id, continuation_id=thread_id)
        self._log_run_started(
            client_config.name,
            thread_id,
            run_id,
            background=background,
            has_progress_token=on_event is not None,
            durable=durable,
        )

        task = asyncio.create_task(
            self._run_pipeline(
                agent=agent,
                client_config=client_config,
                role_config=role_config,
                request=request,
                prompt_text=prompt_text,
                system_prompt_text=system_prompt_text,
                absolute_file_paths=absolute_file_paths,
                images=images,
                on_event=on_event,
                thread_id=thread_id,
                run_id=run_id,
                continuation_id=continuation_id,
                durable=durable,
                capability_digest=capability.config_digest if capability else None,
                exchange_id=arguments.get("_exchange_id"),
            )
        )
        self._background_tasks.add(task)
        task.add_done_callback(self._on_pipeline_done)

        if background:
            output = self._background_started_output(client_config.name, thread_id, run_id)
            return [TextContent(type="text", text=output.model_dump_json())]

        try:
            tool_output = await asyncio.shield(task)
        except asyncio.CancelledError:
            self._detach_cancelled_run(client_config.name, thread_id, run_id)
            raise

        return [TextContent(type="text", text=tool_output.model_dump_json())]

    async def _await_worker_result(self, run_id: str) -> ToolOutput:
        """Wait without owning execution; cancellation only detaches this caller."""
        while True:
            record = jobs.read(run_id)
            if record is None:
                self._raise_tool_error(f"Durable worker run {run_id} disappeared")
            status = record.get("status")
            if status in jobs.TERMINAL_STATUSES:
                stored = record.get("result")
                if stored:
                    output = ToolOutput.model_validate_json(stored)
                    if status != jobs.STATUS_COMPLETED:
                        raise ToolExecutionError(output.model_dump_json())
                    return output
                error = record.get("error") or {}
                self._raise_tool_error(
                    f"Clink worker run {run_id} ended as {status}: {error.get('category', 'unknown')}"
                )
            await asyncio.sleep(0.1)

    async def _return_idempotent_replay(self, request: CLinkRequest) -> list[TextContent]:
        """Return or follow the one run already bound to an idempotency key."""
        run_id = self._current_arguments.get("_idempotent_run_id")
        exchange_id = self._current_arguments.get("_exchange_id")
        thread_id = self._current_arguments.get("_idempotent_thread_id") or request.continuation_id
        record = jobs.read(run_id) if run_id else None
        if record is None and exchange_id:
            from utils.conversation_memory import get_storage, is_durable_storage

            storage = get_storage()
            if is_durable_storage(storage):
                # The original dispatcher may be between exchange admission and
                # run insertion. Follow it briefly; never create a second run.
                deadline = asyncio.get_running_loop().time() + 5
                while record is None and asyncio.get_running_loop().time() < deadline:
                    record = storage.find_run_by_exchange(exchange_id)
                    if record is None:
                        await asyncio.sleep(0.05)
                if record:
                    run_id = record["run_id"]
        if record is None or not run_id:
            self._raise_tool_error(
                "The idempotent clink request was already admitted, but its original run is not yet retrievable. "
                "No second model call was launched; retry with the same idempotency_key or inspect PAL health."
            )
        cli_name = str(record.get("cli_name") or request.cli_name or self._default_cli_name or "unknown")
        thread_id = record.get("continuation_id") or thread_id
        if jobs.is_terminal(record) or not request.background:
            try:
                output = await self._await_worker_result(run_id)
            except asyncio.CancelledError:
                self._detach_cancelled_run(cli_name, thread_id, run_id)
                raise
            return [TextContent(type="text", text=output.model_dump_json())]
        output = self._background_started_output(cli_name, thread_id, run_id)
        return [TextContent(type="text", text=output.model_dump_json())]

    async def _run_pipeline(
        self,
        *,
        agent,
        client_config: ResolvedCLIClient,
        role_config: ResolvedCLIRole,
        request: CLinkRequest,
        prompt_text: str,
        system_prompt_text: str,
        absolute_file_paths: list[str],
        images: list[str],
        on_event,
        thread_id: str | None,
        run_id: str,
        continuation_id: str | None,
        durable: bool,
        capability_digest: str | None,
        exchange_id: str | None,
    ) -> ToolOutput:
        """Run the CLI and own the terminal outcome for both sync and background calls.

        The same coroutine formats the response, records the assistant turn and
        finalizes the durable job record, so a caller that detaches mid-run loses only
        the transport - never the work. Every exit path after this point is terminal:
        no outcome may leave the record in ``running``.
        """
        started = time.monotonic()
        heartbeat_task = asyncio.create_task(self._touch_job_record(run_id))
        try:
            if on_event is not None:
                await self._send_identity_notification(
                    on_event,
                    (
                        f"clink {client_config.name} run started; run_id={run_id} "
                        f"continuation_id={thread_id or 'unavailable'}; if this call times out the run continues "
                        "— poll with clink_poll"
                    ),
                    run_id,
                )
            try:
                result = await agent.run(
                    role=role_config,
                    prompt=prompt_text,
                    system_prompt=system_prompt_text if system_prompt_text.strip() else None,
                    files=absolute_file_paths,
                    images=images,
                    on_event=on_event,
                    model=request.model,
                    reasoning_effort=request.reasoning_effort,
                )
            except asyncio.CancelledError:
                raise
            except CLIAgentError as exc:
                metadata = self._build_error_metadata(client_config, exc)
                if thread_id:
                    metadata["continuation_id"] = thread_id
                if not durable:
                    metadata["durable_record"] = "unavailable"
                error_output = ToolOutput(
                    status="error",
                    content=f"CLI '{client_config.name}' execution failed: {exc}",
                    content_type="text",
                    metadata=metadata,
                )
                self._finalize_failure(
                    client_config.name,
                    thread_id,
                    run_id,
                    error_output,
                    category="cli_error",
                    message=str(exc),
                    duration_seconds=time.monotonic() - started,
                )
                raise ToolExecutionError(error_output.model_dump_json()) from exc
            except Exception as exc:
                error_output = ToolOutput(
                    status="error",
                    content=f"CLI '{client_config.name}' execution failed: {exc}",
                    content_type="text",
                    metadata={"cli_name": client_config.name, "continuation_id": thread_id},
                )
                self._finalize_failure(
                    client_config.name,
                    thread_id,
                    run_id,
                    error_output,
                    category="internal_error",
                    message=str(exc),
                    duration_seconds=time.monotonic() - started,
                )
                raise ToolExecutionError(error_output.model_dump_json()) from exc

            try:
                metadata = self._build_success_metadata(client_config, role_config, result)
                metadata = self._prune_metadata(metadata, client_config, reason="normal")

                content, metadata = self._apply_output_limit(
                    client_config,
                    result.parsed.content,
                    metadata,
                )
                if not durable:
                    metadata["durable_record"] = "unavailable"

                model_info = {
                    "provider": client_config.name,
                    "model_name": result.parsed.metadata.get("model_used"),
                }

                conversation_error: str | None = None
                if continuation_id:
                    try:
                        if not exchange_id:
                            self._record_assistant_turn(continuation_id, content, request, model_info)
                    except Exception as exc:
                        conversation_error = str(exc)
                        logger.debug(
                            "Failed to record assistant turn for continuation %s", continuation_id, exc_info=True
                        )

                continuation_offer = self._continuation_offer_for_thread(thread_id)
                if continuation_offer:
                    tool_output = self._create_continuation_offer_response(
                        content,
                        continuation_offer,
                        request,
                        model_info,
                    )
                    tool_output.metadata = self._merge_metadata(tool_output.metadata, metadata)
                    # The offer builder swallows conversation-storage failures and downgrades
                    # to a plain success response; that downgrade is our only signal.
                    if conversation_error is None and tool_output.status != "continuation_available":
                        conversation_error = "failed to record the assistant turn on the continuation thread"
                else:
                    tool_output = ToolOutput(
                        status="success",
                        content=content,
                        content_type="text",
                        metadata=metadata,
                    )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                # A formatter/serialization bug must terminalize the record, never
                # leave it running until a reader infers staleness.
                error_output = ToolOutput(
                    status="error",
                    content=f"clink failed to assemble the response from CLI '{client_config.name}': {exc}",
                    content_type="text",
                    metadata={"cli_name": client_config.name, "continuation_id": thread_id},
                )
                self._finalize_failure(
                    client_config.name,
                    thread_id,
                    run_id,
                    error_output,
                    category="internal_error",
                    message=str(exc),
                    duration_seconds=time.monotonic() - started,
                )
                raise ToolExecutionError(error_output.model_dump_json()) from exc

            self._finalize_success(
                client_config.name,
                thread_id,
                run_id,
                tool_output,
                metadata,
                duration_seconds=time.monotonic() - started,
                conversation_error=conversation_error,
                content=content,
                request=request,
                model_info=model_info,
                exchange_id=exchange_id,
            )
            return tool_output
        finally:
            # Release before awaiting: an await here re-raises when the pipeline itself
            # is cancelled, and the claim must not survive that.
            jobs.release(thread_id, run_id)
            heartbeat_task.cancel()
            try:
                await asyncio.gather(heartbeat_task, return_exceptions=True)
            except asyncio.CancelledError:
                pass

    async def _send_identity_notification(self, on_event, message: str, run_id: str) -> None:
        """Announce the run over progress; a broken transport must never kill the run.

        Only a genuine cancellation of this pipeline task propagates - a CancelledError
        raised by the progress transport itself is swallowed.
        """
        try:
            await on_event("identity", message)
        except asyncio.CancelledError:
            current = asyncio.current_task()
            cancelling = getattr(current, "cancelling", None)
            if current is not None and cancelling is not None and cancelling() > 0:
                raise
            logger.debug("clink identity notification cancelled for run %s", run_id, exc_info=True)
        except Exception:
            logger.debug("clink identity notification failed for run %s", run_id, exc_info=True)

    async def _touch_job_record(self, run_id: str) -> None:
        """Keep ``updated_at`` fresh so readers do not classify a live run as interrupted."""
        try:
            while True:
                await asyncio.sleep(jobs.HEARTBEAT_INTERVAL_SECONDS)
                jobs.touch(run_id)
        except asyncio.CancelledError:
            raise
        except Exception:  # pragma: no cover - heartbeat must never break a run
            logger.debug("clink job heartbeat failed for run %s", run_id, exc_info=True)

    def _on_pipeline_done(self, task: asyncio.Task) -> None:
        self._background_tasks.discard(task)
        if task.cancelled():
            return
        # Retrieve the exception so a detached (background) failure is not reported
        # by asyncio as an unhandled task exception; it is already logged/persisted.
        exc = task.exception()
        if exc is not None:
            logger.debug("clink pipeline finished with %s", type(exc).__name__)

    def _finalize_success(
        self,
        cli_name: str,
        thread_id: str | None,
        run_id: str,
        tool_output: ToolOutput,
        metadata: dict[str, Any],
        *,
        duration_seconds: float,
        conversation_error: str | None,
        content: str,
        request: CLinkRequest,
        model_info: dict[str, Any],
        exchange_id: str | None,
    ) -> None:
        error = None
        if conversation_error:
            error = {"category": "conversation_write", "message": conversation_error}
            logger.warning(
                "clink run %s completed but its conversation turn was not stored: %s",
                run_id,
                conversation_error,
            )
        record = None
        try:
            from utils.conversation_memory import complete_exchange

            terminal = {
                "run_id": run_id,
                "status": jobs.STATUS_COMPLETED,
                "result": self._sidecar_result(tool_output),
                "error": error,
                "duration_seconds": duration_seconds,
                "output_truncated": bool(metadata.get("output_truncated") or metadata.get("output_summarized")),
            }
            if exchange_id and os.environ.get("PAL_CONVERSATION_BACKEND", "sqlite").lower() != "memory":
                complete_exchange(
                    exchange_id,
                    content=content,
                    files=self.get_request_files(request),
                    images=self.get_request_images(request),
                    tool_name=self.get_name(),
                    model_provider=str(model_info.get("provider") or ""),
                    model_name=model_info.get("model_name"),
                    run_terminal=terminal,
                )
                record = jobs.read(run_id)
            else:
                record = jobs.finalize(run_id, **{k: v for k, v in terminal.items() if k != "run_id"})
        except Exception:
            logger.warning("Failed to finalize clink job record %s", run_id, exc_info=True)
            if os.environ.get("PAL_CONVERSATION_BACKEND", "sqlite").lower() != "memory":
                raise
        self._log_run_finished(
            "CLINK_COMPLETED",
            cli_name,
            thread_id,
            run_id,
            record,
            duration_seconds,
        )

    def _finalize_failure(
        self,
        cli_name: str,
        thread_id: str | None,
        run_id: str,
        error_output: ToolOutput,
        *,
        category: str,
        message: str,
        duration_seconds: float,
    ) -> None:
        try:
            from utils.conversation_memory import current_exchange_id, fail_exchange

            fail_exchange(current_exchange_id.get(), category)
        except Exception:
            logger.warning("Failed to terminalize clink exchange for run %s", run_id, exc_info=True)
        record = None
        try:
            record = jobs.finalize(
                run_id,
                status=jobs.STATUS_FAILED,
                result=self._sidecar_result(error_output),
                error={"category": category, "message": message},
                duration_seconds=duration_seconds,
            )
        except Exception:
            logger.warning("Failed to finalize clink job record %s", run_id, exc_info=True)
        self._log_run_finished("CLINK_FAILED", cli_name, thread_id, run_id, record, duration_seconds)

    @staticmethod
    def _sidecar_result(tool_output: ToolOutput) -> str:
        """Serialize a response for durable storage with raw payloads redacted.

        The in-band response keeps its full metadata; only the seven-day record drops
        the denylisted raw CLI payloads and diagnostic streams, which are unbounded by
        _apply_output_limit (it bounds content only) and can be sensitive.
        """
        metadata = tool_output.metadata
        if not metadata or SIDECAR_REDACTED_METADATA_KEYS.isdisjoint(metadata):
            return tool_output.model_dump_json()
        redacted = dict(metadata)
        for key in SIDECAR_REDACTED_METADATA_KEYS:
            if key in redacted:
                redacted[key] = SIDECAR_REDACTION_PLACEHOLDER
        return tool_output.model_copy(update={"metadata": redacted}).model_dump_json()

    def _background_started_output(self, cli_name: str, thread_id: str | None, run_id: str) -> ToolOutput:
        payload = {
            "run_id": run_id,
            "continuation_id": thread_id,
            "status": jobs.STATUS_RUNNING,
            "cli_name": cli_name,
            "poll_with": {
                "tool": "clink_poll",
                "arguments": {"run_id": run_id, "wait_seconds": 20},
            },
            "note": (
                "The CLI run continues in the background. Poll clink_poll with this run_id until status is "
                "completed, failed or interrupted; the terminal payload is exactly what a synchronous clink "
                "call would have returned."
            ),
        }
        offer = self._continuation_offer_for_thread(thread_id)
        return ToolOutput(
            status="clink_background_started",
            content=json.dumps(payload, indent=2),
            content_type="json",
            metadata={
                "tool_name": self.get_name(),
                "cli_name": cli_name,
                "run_id": run_id,
                "background": True,
            },
            continuation_offer=ContinuationOffer(**offer) if offer else None,
        )

    def _raise_durable_store_error(self, cli_name: str, thread_id: str | None, run_id: str) -> None:
        self._raise_tool_error(
            f"clink could not write a durable run record for run_id={run_id}, so background mode was refused: "
            "its result would not be retrievable with clink_poll. Check the PAL logs directory, or retry "
            "without background=true to receive the answer in-band.",
            metadata={
                "tool_name": self.get_name(),
                "cli_name": cli_name,
                "continuation_id": thread_id,
                "run_id": run_id,
                "error_category": "durable_store",
            },
        )

    def _active_run_conflict(self, cli_name: str, thread_id: str, active_run_id: str) -> list[TextContent]:
        message = (
            f"clink already has an active run on continuation_id={thread_id} (run_id={active_run_id}). "
            f"No CLI was launched and no turn was consumed. Wait for it to finish or fetch it with "
            f"clink_poll run_id={active_run_id}."
        )
        output = ToolOutput(
            status="error",
            content=message,
            content_type="text",
            metadata={
                "tool_name": self.get_name(),
                "cli_name": cli_name,
                "continuation_id": thread_id,
                "active_run_id": active_run_id,
            },
        )
        return [TextContent(type="text", text=output.model_dump_json())]

    def _create_recovery_thread(self, request: CLinkRequest, capability_digest: str | None = None) -> str | None:
        """Create the conversation thread (with the user's turn) ahead of the CLI run.

        Historically the thread was created only after the CLI returned, so a call
        cancelled by the MCP client left no continuation to resume. Creating it up
        front means the ID is known (and logged) for the whole lifetime of the run.
        """
        from utils.conversation_memory import (
            add_turn,
            begin_exchange,
            create_thread,
            current_exchange_id,
            get_storage,
            is_durable_storage,
        )

        durable = is_durable_storage(get_storage())
        try:
            thread_id = create_thread(
                tool_name=self.get_name(),
                initial_request=self.get_request_as_dict(request),
                initial_idempotency_key=request.idempotency_key,
            )
            if durable:
                admission = begin_exchange(
                    thread_id,
                    tool_name=self.get_name(),
                    content=self.get_request_prompt(request),
                    files=self.get_request_files(request),
                    images=self.get_request_images(request),
                    idempotency_key=request.idempotency_key,
                    capability_digest=capability_digest,
                )
                self._current_arguments["_exchange_id"] = admission["exchange_id"]
                if admission.get("idempotent"):
                    self._current_arguments["_idempotent_replay"] = True
                    self._current_arguments["_idempotent_run_id"] = admission.get("run_id")
                    self._current_arguments["_idempotent_thread_id"] = thread_id
                current_exchange_id.set(admission["exchange_id"])
            else:
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
            if durable:
                raise
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

    @staticmethod
    def _log_activity(message: str, *, warning: bool = False) -> None:
        if warning:
            logger.warning(message)
        else:
            logger.info(message)
        try:
            logging.getLogger("mcp_activity").info(message)
        except Exception:  # pragma: no cover - logging must never break execution
            pass

    def _log_run_started(
        self,
        cli_name: str,
        thread_id: str | None,
        run_id: str,
        *,
        background: bool,
        has_progress_token: bool,
        durable: bool,
    ) -> None:
        self._log_activity(
            f"CLINK_RUN_STARTED: cli={cli_name} run_id={run_id} "
            f"continuation_id={thread_id or 'unavailable'} pid={os.getpid()} "
            f"instance={jobs.INSTANCE_ID} mode={'background' if background else 'sync'} "
            f"progress_token={'present' if has_progress_token else 'absent'} "
            f"durable={'true' if durable else 'false'}"
        )

    def _log_run_finished(
        self,
        token: str,
        cli_name: str,
        thread_id: str | None,
        run_id: str,
        record: dict[str, Any] | None,
        duration_seconds: float,
    ) -> None:
        attachment = (record or {}).get("attachment", jobs.ATTACHMENT_ATTACHED)
        # ``record is None`` means finalize refused or raised: never claim a durable
        # terminal outcome the store does not actually hold.
        self._log_activity(
            f"{token}: cli={cli_name} run_id={run_id} continuation_id={thread_id or 'unavailable'} "
            f"pid={os.getpid()} instance={jobs.INSTANCE_ID} attachment={attachment} "
            f"durable={'true' if record is not None else 'false'} duration={duration_seconds:.1f}s"
        )
        if attachment == jobs.ATTACHMENT_DETACHED:
            self._log_activity(
                f"CLINK_SALVAGED: cli={cli_name} run_id={run_id} continuation_id={thread_id or 'unavailable'} "
                f"duration={duration_seconds:.1f}s - the detached run finished; "
                f"fetch it with clink_poll run_id={run_id}"
            )

    def _detach_cancelled_run(self, cli_name: str, thread_id: str | None, run_id: str) -> None:
        """Record that only the MCP request went away; the CLI run continues.

        Must stay synchronous: it runs inside an already-cancelled scope where any
        await would immediately re-raise CancelledError.
        """
        try:
            jobs.set_attachment(run_id, jobs.ATTACHMENT_DETACHED)
        except Exception:
            logger.debug("Failed to mark clink run %s detached", run_id, exc_info=True)
        self._log_activity(
            f"CLINK_CANCELLED: cli={cli_name} run_id={run_id} continuation_id={thread_id or 'unavailable'} - "
            f"only the MCP request detached; the CLI run continues under supervision. "
            f"Recover the result with clink_poll run_id={run_id}",
            warning=True,
        )

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

    def _build_progress_callback(self, cli_name: str, *, run_id: str | None = None, continuation_id: str | None = None):
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
        send_lock = asyncio.Lock()

        async def _send(progress: float, message: str) -> None:
            try:
                async with send_lock:
                    await send_progress(
                        progress_token=progress_token,
                        progress=progress,
                        message=message,
                    )
            except Exception:
                logger.debug(
                    "Failed to send progress notification (run_id=%s continuation_id=%s)",
                    run_id,
                    continuation_id,
                    exc_info=True,
                )

        async def on_event(kind: str, text: str) -> None:
            nonlocal progress_counter
            if kind == "identity":
                # Advisory discovery: emitted once, before any CLI output, so a caller
                # that later times out still knows which run to poll.
                stripped = text.strip()
                if stripped:
                    await _send(0.0, stripped)
                return
            progress_counter += 1
            stripped = text.strip()
            if not stripped:
                return
            if len(stripped) > max_message_chars:
                stripped = stripped[:max_message_chars] + "…"
            if kind == "heartbeat" and run_id:
                message = f"[{cli_name}:heartbeat] run_id={run_id} {stripped}"
            else:
                message = f"[{cli_name}:{kind}] {stripped}"
            await _send(float(progress_counter), message)

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
