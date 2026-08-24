"""clink_poll tool - read the durable record of a clink run.

Deliberately keyed by ``run_id`` only: accepting a ``continuation_id`` would make the
server's generic conversation middleware reconstruct and mutate the thread before the
tool runs. Polling therefore consumes no turns, launches no CLI, and is idempotent.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any

from mcp.types import TextContent
from pydantic import BaseModel, ConfigDict, Field

from clink import jobs
from config import TEMPERATURE_BALANCED
from tools.models import ToolModelCategory, ToolOutput
from tools.simple.base import SimpleTool

logger = logging.getLogger(__name__)

MAX_WAIT_SECONDS = 20
POLL_SLEEP_SECONDS = 0.5

RUN_ID_PATTERN = r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"


class CLinkPollRequest(BaseModel):
    """Request model for clink_poll."""

    # Defense in depth: a direct (non-MCP) call must not silently accept
    # conversation fields either.
    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(..., description="The run_id returned by a clink call (canonical UUID).")
    wait_seconds: int | None = Field(
        default=0,
        description=(
            f"Optionally wait up to this many seconds (max {MAX_WAIT_SECONDS}) for the run to reach a "
            "terminal state before returning."
        ),
    )


class CLinkPollTool(SimpleTool):
    """Return the status - and for terminal runs the exact result - of a clink run."""

    def get_name(self) -> str:
        return "clink_poll"

    def get_description(self) -> str:
        return (
            "Fetch the status and result of a clink run by run_id. Long clink calls survive an MCP client "
            "timeout; poll them here instead of re-running the CLI. Terminal runs return exactly the payload "
            "a synchronous clink call would have produced."
        )

    def get_annotations(self) -> dict[str, Any]:
        return {"readOnlyHint": True}

    def requires_model(self) -> bool:
        return False

    def get_model_category(self) -> ToolModelCategory:
        return ToolModelCategory.FAST_RESPONSE

    def get_default_temperature(self) -> float:
        return TEMPERATURE_BALANCED

    def get_system_prompt(self) -> str:
        return ""

    def get_request_model(self):
        return CLinkPollRequest

    def get_tool_fields(self) -> dict[str, dict[str, Any]]:
        """Unused by clink_poll because we override the schema end-to-end."""
        return {}

    def get_input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "run_id": {
                    "type": "string",
                    "pattern": RUN_ID_PATTERN,
                    "description": "The run_id reported when the clink run started.",
                },
                "wait_seconds": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": MAX_WAIT_SECONDS,
                    "default": 0,
                    "description": (
                        f"Wait up to this many seconds (max {MAX_WAIT_SECONDS}) for a terminal status "
                        "before returning."
                    ),
                },
            },
            "required": ["run_id"],
            "additionalProperties": False,
        }

    async def prepare_prompt(self, request) -> str:  # pragma: no cover - no model is involved
        return ""

    async def execute(self, arguments: dict[str, Any]) -> list[TextContent]:
        request = CLinkPollRequest(**arguments)

        try:
            run_id = jobs.validate_run_id(request.run_id)
        except jobs.JobStoreError as exc:
            return self._error(str(exc), run_id=None)

        wait_seconds = max(0, min(int(request.wait_seconds or 0), MAX_WAIT_SECONDS))
        loop = asyncio.get_running_loop()
        deadline = loop.time() + wait_seconds

        jobs.interrupt_stale_worker_queue()
        record = jobs.read(run_id)
        while record is not None and not jobs.is_terminal(record):
            if loop.time() >= deadline:
                break
            await asyncio.sleep(POLL_SLEEP_SECONDS)
            record = jobs.read(run_id)

        if record is None:
            return self._error(
                f"No clink run record found for run_id={run_id}. It may never have existed, it may have "
                f"passed the {jobs.RETENTION_SECONDS // 86400}-day retention window, or its record is "
                "unreadable (corrupt).",
                run_id=run_id,
            )

        stored_status = record.get("status")
        reported_status = stored_status
        payload = dict(record)
        payload["stale_heartbeat"] = False
        if jobs.is_stale(record):
            # Read-only classification: the owning PAL process died with its CLI child.
            reported_status = jobs.STATUS_INTERRUPTED
            payload["stale_heartbeat"] = True
            payload["reported_status"] = reported_status
            payload["note"] = (
                "This run has not reported progress recently; its PAL process most likely exited. "
                "Running work does not survive a restart - re-issue the request."
            )
            self._log_interrupted_observation(run_id, jobs.staleness_seconds(record))

        output = ToolOutput(
            status="success",
            content=json.dumps(payload, indent=2),
            content_type="json",
            metadata={
                "tool_name": self.get_name(),
                "run_id": run_id,
                "run_status": reported_status,
                "stored_status": stored_status,
            },
        )
        return [TextContent(type="text", text=output.model_dump_json())]

    @staticmethod
    def _log_interrupted_observation(run_id: str, stale_for: float | None) -> None:
        """Report the classification as an observation - the poller does not own the run."""
        elapsed = f"{stale_for:.0f}" if stale_for is not None and stale_for != float("inf") else "unknown"
        message = f"CLINK_INTERRUPTED (observed): run_id={run_id} stale_for={elapsed}s pid={os.getpid()}"
        logger.warning(message)
        try:
            logging.getLogger("mcp_activity").info(message)
        except Exception:  # pragma: no cover - logging must never break execution
            pass

    def _error(self, message: str, *, run_id: str | None) -> list[TextContent]:
        output = ToolOutput(
            status="error",
            content=message,
            content_type="text",
            metadata={"tool_name": self.get_name(), "run_id": run_id},
        )
        return [TextContent(type="text", text=output.model_dump_json())]
