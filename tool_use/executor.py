"""The tool-use workflow: the five steps from class, implemented by hand, with no framework.

    1. Get the user message together with the system prompt that carries the tool definitions.
    2. The model generates: text, or one or more <tool_call> blocks.
    3. The application executes the calls (``ToolRegistry.call``) and never trusts the
       arguments blindly: they are validated first.
    4. Each result goes back into the conversation as a message with role ``tool``.
    5. The model is prompted again; if it answers without calling tools, we are done.

Steps 2-4 repeat until the model answers or ``max_turns`` is reached (multi-step tool use).
Several <tool_call> blocks in one generation are executed in the same turn (parallel calls).

``run_tool_loop`` is what ``POST /tools`` calls. Until you implement it, the endpoint
answers 501.
"""

from __future__ import annotations

from api.schemas import ToolCallRecord, ToolsResponse
from tool_use.registry import ToolRegistry
from tool_use.tools import default_registry

SYSTEM_PROMPT_TEMPLATE = """You are an AI assistant designed to help users.
You will be given a task to solve. To help you, you have access to a set of tools.
Each tool is a function and has a description explaining what it does and the inputs it
expects:
---
{tools_description}
---
You should decide if you want to call a tool. If so, your call must be formatted as follows:
<tool_call>
{{"name": "tool name", "arguments": {{"argument": "value"}}}}
</tool_call>
You may emit several <tool_call> blocks in one message if the calls are independent.
After the tool calls are executed, you will receive their results and can continue.
When you have everything you need, answer the user directly without calling tools."""


def build_system_prompt(registry: ToolRegistry) -> str:
    """Step 1: the system prompt with the tool definitions."""
    return SYSTEM_PROMPT_TEMPLATE.format(tools_description=registry.render_for_prompt())


class ToolLoop:
    """Holds the model and the registry and runs the five-step workflow."""

    def __init__(self, registry: ToolRegistry | None = None, model_id: str | None = None):
        self.registry = registry or default_registry()
        self.model_id = model_id or "Qwen/Qwen3-1.7B"
        self.model = None
        self.tokenizer = None

    def load(self) -> None:
        """Tu turno: load tokenizer and model (bf16 on GPU).

        Tip: Qwen's chat template accepts ``tools=registry.definitions()`` directly in
        ``apply_chat_template``. Using it instead of pasting JSON into the system prompt is
        allowed, but then explain in your README what the template actually renders.
        """
        raise NotImplementedError

    def generate(self, messages: list[dict]) -> str:
        """Tu turno: step 2, one generation from the conversation so far."""
        raise NotImplementedError

    def run(self, query: str, max_turns: int = 5) -> ToolsResponse:
        """Tu turno: the loop.

        Sketch::

            messages = [system, user]
            records: list[ToolCallRecord] = []
            for turn in range(max_turns):
                raw = self.generate(messages)
                parsed = parse_tool_calls(raw)
                if not parsed.has_tool_calls:
                    return ToolsResponse(answer=parsed.text, tool_calls=records, turns=turn + 1)
                messages.append({"role": "assistant", "content": raw})
                for call in parsed.tool_calls:
                    record = self.registry.call(call.name, call.arguments)   # step 3
                    records.append(record)
                    payload = record.result if record.ok else record.error
                    content = format_tool_result(record.name, payload)
                    messages.append({"role": "tool", "name": record.name, "content": content})
            # out of turns: return what you have, flag it

        Decide what to do with parse errors (``call.error``) and with tools that have
        ``requires_confirmation``. Both are observations the model should see.
        """
        raise NotImplementedError


_loop: ToolLoop | None = None


def build_tool_loop() -> ToolLoop:
    """Create and load the loop once; the API keeps the instance."""
    global _loop
    if _loop is None:
        loop = ToolLoop()
        loop.load()  # raises NotImplementedError until phase 2 is done
        _loop = loop
    return _loop


def run_tool_loop(query: str, max_turns: int = 5) -> ToolsResponse:
    """Convenience entry point for scripts and the evaluation."""
    return build_tool_loop().run(query, max_turns=max_turns)


__all__ = ["ToolCallRecord", "ToolLoop", "build_system_prompt", "build_tool_loop", "run_tool_loop"]
