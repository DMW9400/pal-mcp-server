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
        observed_models: list[str] = []
        observed_efforts: list[str] = []

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
                if isinstance(model, str) and model and model not in observed_models:
                    observed_models.append(model)
                if isinstance(effort, str) and effort and effort not in observed_efforts:
                    observed_efforts.append(effort)
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
        if observed_models:
            metadata["models_used"] = observed_models
        if observed_efforts:
            metadata["reasoning_efforts_used"] = observed_efforts
        if stderr and stderr.strip():
            metadata["stderr"] = stderr.strip()

        return ParsedCLIResponse(content=content, metadata=metadata)
