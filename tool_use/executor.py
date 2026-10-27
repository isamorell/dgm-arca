"""The tool-use workflow: the five steps from class, implemented by hand, with no framework.

    1. Get the user message together with the system prompt that carries the tool definitions.
    2. The model generates: text, or one or more <tool_call> blocks.
    3. The application executes the calls (``ToolRegistry.call``) and never trusts the
       arguments blindly: they are validated first.
    4. Each result goes back into the conversation as a message with role ``tool``.
    5. The model is prompted again; if it answers without calling tools, we are done.

Steps 2-4 repeat until the model answers or ``max_turns`` is reached (multi-step tool use).
Several <tool_call> blocks in one generation are executed in the same turn (parallel calls).

``run_tool_loop`` is what ``POST /tools`` calls.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from api.schemas import ToolCallRecord, ToolsResponse
from tool_use.parser import format_tool_result, parse_tool_calls
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

    def __init__(self, registry: ToolRegistry | None = None, model_id: str | None = None, allow_side_effects: bool | None = None, max_new_tokens: int = 1024):
        self.registry = registry or default_registry()

        # Without an explicit model, the adapter of the environment must be available (see load()).
        self._check_env_adapter = model_id is None

        self.model_id = model_id or os.environ.get("ARCA_RLM_BASE_MODEL")
        self.model = None
        self.tokenizer = None

        if allow_side_effects is None:
            allow_side_effects = os.environ.get("ARCA_TOOLS_ALLOW_SIDE_EFFECTS", "1") == "1"
        self.allow_side_effects = allow_side_effects
        self.max_new_tokens = max_new_tokens

    def _ensure_env_adapter(self) -> None:
        """Make sure the ``ARCA_RLM_ADAPTER`` folder exists, downloading it from the Hub if needed.

        Done in ``load()`` and not in ``__init__`` so that building a ``ToolLoop`` (tests, scripts
        that only use ``generate`` or the registry) does not need the adapter nor the network.
        If ``ARCA_RLM_ADAPTER`` is not defined there is nothing to ensure.
        """
        adapter = os.environ.get("ARCA_RLM_ADAPTER")
        if not adapter or Path(adapter).exists():
            return
        repo = os.environ.get("ARCA_RLM_ADAPTER_REPO")
        if not repo:
            raise FileNotFoundError(
                f"ARCA_RLM_ADAPTER points to a missing folder: {adapter} "
                "(define ARCA_RLM_ADAPTER_REPO to download it from Hub)"
            )
        from huggingface_hub import snapshot_download

        snapshot_download(repo, local_dir=adapter)

    def load(self) -> None:
        """Load the tokenizer and the model (bf16 on GPU).

        Only the base model is loaded. The tool definitions go in the system prompt
        (``build_system_prompt``), not through the chat template. The LoRA adapter of phase 1
        is not applied here: ``ARCA_RLM_ADAPTER`` is only checked (and downloaded from the Hub
        if it is missing). ``tool_use/evaluate.py --adapter`` applies it to evaluate it.
        """
        if self._check_env_adapter:
            self._ensure_env_adapter()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id, dtype=dtype, device_map=device
        ).eval()

    def generate(self, messages: list[dict]) -> str:
        """Step 2: one greedy generation from the conversation so far, with thinking disabled."""
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
        new_tokens = out[0, inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    def run(self, query: str, max_turns: int = 5) -> ToolsResponse:
        """Steps 2-5: generate, parse, execute and feed the results back until the model answers.

        Stops when a generation has no valid ``<tool_call>`` or after ``max_turns`` rounds.
        Parse errors (``call.error``), validation errors and tools that require confirmation
        (when ``allow_side_effects`` is false) are not raised: they are recorded in
        ``tool_calls`` and sent back to the model as ``tool`` messages, so it can correct itself.
        """
        messages: list[dict] = [
            {"role": "system", "content": build_system_prompt(self.registry)},
            {"role": "user", "content": query},
        ]
        records: list[ToolCallRecord] = []

        for turn in range(max_turns):
            raw = self.generate(messages)
            parsed = parse_tool_calls(raw)

            if not parsed.has_tool_calls:
                return ToolsResponse(
                    answer=parsed.text or raw,
                    tool_calls=records,
                    turns=turn + 1,
                    model=self.model_id or "",
                )
            messages.append({"role": "assistant", "content": raw})

            for call in parsed.tool_calls:
                tool = self.registry.tools.get(call.name)
                needs_confirmation = tool is not None and tool.requires_confirmation

                if call.error:
                    record = ToolCallRecord(name=call.name or "unknown", arguments=call.arguments, ok=False, error=f"invalid tool call: {call.error}")
                elif needs_confirmation and not self.allow_side_effects:
                    record = ToolCallRecord(
                        name=call.name,
                        arguments=call.arguments,
                        ok=False,
                        requires_confirmation=True,
                        error="this tool has side effects and requires user confirmation; "
                        "it was NOT executed. Ask the user to confirm.",
                    )
                else:
                    record = self.registry.call(call.name, call.arguments)
                    record.requires_confirmation = needs_confirmation
                records.append(record)

                payload = record.result if record.ok else record.error
                content = format_tool_result(record.name, payload)
                messages.append({"role": "tool", "name": record.name, "content": content})

        return ToolsResponse(
            answer=f"No pude completar la tarea en {max_turns} turnos.",
            tool_calls=records,
            turns=max_turns,
            model=self.model_id or "",
        )


_loop: ToolLoop | None = None


def build_tool_loop() -> ToolLoop:
    """Create and load the loop once; the API keeps the instance."""
    global _loop
    if _loop is None:
        loop = ToolLoop()
        loop.load()
        _loop = loop
    return _loop


def run_tool_loop(query: str, max_turns: int = 5) -> ToolsResponse:
    """Convenience entry point for scripts and the evaluation."""
    return build_tool_loop().run(query, max_turns=max_turns)


__all__ = ["ToolCallRecord", "ToolLoop", "build_system_prompt", "build_tool_loop", "run_tool_loop"]
