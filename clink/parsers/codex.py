"""Parser for Codex CLI JSONL output."""

from __future__ import annotations

import json
from typing import Any

from .base import BaseParser, ParsedCLIResponse, ParserError


class CodexJSONLParser(BaseParser):
    """Parse stdout emitted by `codex exec --json`."""

    name = "codex_jsonl"

    def parse(self, stdout: str, stderr: str) -> ParsedCLIResponse:
        lines = [line.strip() for line in (stdout or "").splitlines() if line.strip()]
        events: list[dict[str, Any]] = []
        agent_messages: list[str] = []
        errors: list[str] = []
        usage: dict[str, Any] | None = None
        observed_models: set[str] = set()
        observed_efforts: set[str] = set()

        for line in lines:
            if not line.startswith("{"):
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue

            events.append(event)
            event_type = event.get("type")
            if event_type == "turn_context":
                payload = event.get("payload") or {}
                model = payload.get("model")
                effort = payload.get("effort")
                if isinstance(model, str) and model:
                    observed_models.add(model)
                if isinstance(effort, str) and effort:
                    observed_efforts.add(effort)
            if event_type == "item.completed":
                item = event.get("item") or {}
                if item.get("type") == "agent_message":
                    text = item.get("text")
                    if isinstance(text, str) and text.strip():
                        agent_messages.append(text.strip())
            elif event_type == "error":
                message = event.get("message")
                if isinstance(message, str) and message.strip():
                    errors.append(message.strip())
            elif event_type == "turn.completed":
                turn_usage = event.get("usage")
                if isinstance(turn_usage, dict):
                    usage = turn_usage

        if not agent_messages and errors:
            agent_messages.extend(errors)

        if not agent_messages:
            raise ParserError("Codex CLI JSONL output did not include an agent_message item")

        content = "\n\n".join(agent_messages).strip()
        metadata: dict[str, Any] = {"events": events}
        if errors:
            metadata["errors"] = errors
        if usage:
            metadata["usage"] = usage
        if len(observed_models) == 1:
            metadata["model_used"] = next(iter(observed_models))
        elif len(observed_models) > 1:
            raise ParserError("Codex CLI reported multiple models in one run")
        if len(observed_efforts) == 1:
            metadata["reasoning_effort_used"] = next(iter(observed_efforts))
        elif len(observed_efforts) > 1:
            raise ParserError("Codex CLI reported multiple reasoning efforts in one run")
        if stderr and stderr.strip():
            metadata["stderr"] = stderr.strip()

        return ParsedCLIResponse(content=content, metadata=metadata)
