"""Detect tool calls in the model's output.

A tool call is "a sequence of tokens produced by the LLM to request the use of a tool,
written in a recognisable format so the application can detect it, stop generation and
execute it". Qwen models emit::

    <tool_call>
    {"name": "get_weather", "arguments": {"location": "Paris, France"}}
    </tool_call>

possibly several in a row (parallel calls) and possibly after a ``<think>`` block, which
we keep: it is part of the trace we want to show. Malformed JSON is reported as an error
entry instead of being dropped silently, so the loop can tell the model what went wrong.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

TOOL_CALL_PATTERN = re.compile(r"<tool_call>\s*(?P<body>.*?)\s*</tool_call>", re.DOTALL)
THINK_PATTERN = re.compile(r"<think>(?P<think>.*?)</think>", re.DOTALL)


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    error: str | None = None  # set when the block could not be parsed


@dataclass
class ParsedOutput:
    thinking: str
    text: str  # what remains after removing think and tool_call blocks
    tool_calls: list[ToolCall]

    @property
    def has_tool_calls(self) -> bool:
        return any(call.error is None for call in self.tool_calls)


def parse_tool_calls(output: str) -> ParsedOutput:
    """Split a raw completion into thinking, plain text and tool calls."""
    think = THINK_PATTERN.search(output)
    thinking = think.group("think").strip() if think else ""

    calls: list[ToolCall] = []
    for match in TOOL_CALL_PATTERN.finditer(output):
        body = match.group("body")
        try:
            payload = json.loads(body)
        except json.JSONDecodeError as exc:
            calls.append(ToolCall(name="", error=f"tool_call is not valid JSON: {exc.msg}"))
            continue
        name = payload.get("name") if isinstance(payload, dict) else None
        if not isinstance(name, str) or not name:
            calls.append(ToolCall(name="", error="tool_call JSON has no 'name' field"))
            continue
        arguments = payload.get("arguments", payload.get("parameters", {}))
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {"input": arguments}
        if not isinstance(arguments, dict):
            calls.append(ToolCall(name=name, error="'arguments' must be a JSON object"))
            continue
        calls.append(ToolCall(name=name, arguments=arguments))

    text = TOOL_CALL_PATTERN.sub("", THINK_PATTERN.sub("", output)).strip()
    return ParsedOutput(thinking=thinking, text=text, tool_calls=calls)


def format_tool_result(name: str, result: Any) -> str:
    """How a tool result is written back into the conversation (the 'tool' role content)."""
    if isinstance(result, str):
        return result
    return json.dumps(result, ensure_ascii=False, default=str)
