#!/usr/bin/env python3
"""No-network Codex-shaped CLI used only by durable-worker process tests."""

import json
import sys
import time

_prompt = sys.stdin.read()
time.sleep(1.0)
print(json.dumps({"type": "turn_context", "payload": {"model": "gpt-5.6-sol", "effort": "high"}}))
print(json.dumps({"type": "item.completed", "item": {"type": "agent_message", "text": "fixture answer"}}))
