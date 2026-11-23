"""The ReAct agent: Thought -> Action -> Observation, in a loop, with every tool you built.

This is where phases 1 to 3 meet. The brain is a model (``brain="base"``, your phase-1
RLM with ``brain="rlm"``, or a thinking model with ``brain="thinking"``); the hands are the
phase-2 registry plus ``search_knowledge_base`` from phase 3; the loop is yours to write.

``ReActAgent.run`` is what ``POST /agent`` calls. Until it is implemented the endpoint
answers 501. Reuse what you already have: ``tool_use.parser.parse_tool_calls`` to read
the model's output, ``ToolRegistry.call`` to execute, ``ConversationMemory`` to keep the
history within budget.
"""

from __future__ import annotations

import json
import os
import threading
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import re

from agent.memory import ConversationMemory
from agent.prompts import REACT_SYSTEM_PROMPT, SUMMARY_PROMPT
from api.schemas import AgentAction, AgentResponse, AgentStep, ToolCallRecord
from tool_use.registry import ToolRegistry, tool
from tool_use.tools import default_registry
from tool_use.parser import THINK_PATTERN, ToolCall, parse_tool_calls
from rag.retriever import build_retrievers

RETRIEVER = "hybrid"
LAMBDA_BM25 = 0.4  # best MRR in the λ sweep with recursive chunking (EXPERIMENTS.md)
MAX_NEW_TOKENS = 1024
# Per tool call. Measured in reports/agent_tool_latency.json (see EXPERIMENTS.md, 2026-11-03):
# normal calls take at most ~1.1 s, the first search (builds the retriever) ~12 s, and CIMA
# gives up by itself after 15 s. 30 s leaves ~2.5x margin over the slowest of those.
TOOL_TIMEOUT_S = 30
SUMMARY_FALLBACK_CHARS = 1000  # used when the model gives no summary
MAX_REPEATED_CALLS = 3  # the same call (name + arguments) this many times stops the run

ACTION_PATTERN = re.compile(r"Action:\s*`?(?P<name>[A-Za-z_]\w*)`?")


def _action_line_fallback(raw: str, known_tools) -> ToolCall | None:
    """Read ``Action: tool_name`` (optionally followed by a JSON object) written without a
    ``<tool_call>`` block. Small models do this a lot (phase 4: ~half of the steps without a
    valid call). Only names that exist in the registry count, and only text outside ``<think>``.
    """
    visible = THINK_PATTERN.sub("", raw)
    visible = visible.rsplit("<think>", 1)[0] if "<think>" in visible else visible
    matches = [m for m in ACTION_PATTERN.finditer(visible) if m.group("name") in known_tools]
    if not matches:
        return None
    match = matches[-1]  # the last Action: is the decision the model ended up with
    arguments: dict = {}
    rest = visible[match.end():].lstrip()
    if rest.startswith("{"):
        try:
            payload, _ = json.JSONDecoder().raw_decode(rest)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict):
            arguments = payload.get("arguments", payload) if "name" in payload else payload
            if not isinstance(arguments, dict):
                arguments = {}
    return ToolCall(name=match.group("name"), arguments=arguments)

def _get_retriever():
    """Ingest -> chunk -> build_retrievers, one single time. Returns the chosen retriever."""
    global _retriever
    if _retriever is None:
        from core.models import get_embedder
        from rag.chunking import recursive_chunks
        from rag.generate import CORPUS_DIR, INDEX_DIR
        from rag.ingest import load_corpus

        chunks = recursive_chunks(load_corpus(CORPUS_DIR)) # decided in EXPERIMENTS.md. Works better than semantic
        retrievers = build_retrievers(
            chunks, get_embedder(), persist_dir=INDEX_DIR / "recursive", weight_bm25=LAMBDA_BM25
        )
        _retriever = retrievers[RETRIEVER]

    return _retriever

@tool
def final_answer(answer: str) -> str:
    """Return the final answer to the user and stop the loop.

    Args:
        answer: The complete answer to the task.
    """
    return answer


@tool
def search_knowledge_base(query: str, top_k: int = 5) -> str:
    """Search the domain knowledge base and return the most relevant passages with their ids.

    Args:
        query: What to look for, phrased as a question.
        top_k: How many passages to return.
    """
    # Call your phase-3 retriever and render hits as "[id] (source)\\ntext".
    hits = _get_retriever().search(query, top_k)
    if not hits:
        return "No se han encontrado pasajes relevantes."
    return "\n\n".join(f"[{h.chunk.id}] ({h.chunk.source})\n{h.chunk.text}" for h in hits)


BRAINS = {
    "base": lambda: os.environ.get("ARCA_RLM_BASE_MODEL", "Qwen/Qwen3-0.6B"),
    "rlm": lambda: os.environ.get("ARCA_RLM_ADAPTER", ""),
    "thinking": lambda: os.environ.get("ARCA_THINKING_MODEL", "Qwen/Qwen3-1.7B"),
}


def _resolve_adapter() -> str:
    """Folder of the trained LoRA adapter (brain="rlm"), downloaded from the Hub if missing.

    Same rule as ``rlm/inference.py``: ``ARCA_RLM_ADAPTER`` is the local folder and
    ``ARCA_RLM_ADAPTER_REPO`` the Hub repo it is downloaded from when the folder does not exist.
    """
    adapter = BRAINS["rlm"]()
    if not adapter:
        raise ValueError("brain='rlm' needs ARCA_RLM_ADAPTER to be set.")
    if not Path(adapter).exists():
        repo = os.environ.get("ARCA_RLM_ADAPTER_REPO")
        if not repo:
            raise FileNotFoundError(
                f"ARCA_RLM_ADAPTER points to a missing folder: {adapter} "
                "(define ARCA_RLM_ADAPTER_REPO to download it from the Hub)"
            )
        from huggingface_hub import snapshot_download

        snapshot_download(repo, local_dir=adapter)
    return adapter


@dataclass
class ReActAgent:
    registry: ToolRegistry = field(default_factory=default_registry)
    token_budget: int = 6000
    steps: list[AgentStep] = field(default_factory=list)
    _models: dict = field(default_factory=dict) # brain -> (tokenizer, model)
    _tokens_used: int = field(default=0, init=False, repr=False)  # of the run in progress

    @classmethod
    def from_env(cls) -> ReActAgent:
        agent = cls()
        agent.registry.register(final_answer, search_knowledge_base)
        agent.load()
        return agent

    def load(self) -> None:
        """Load the model(s) behind BRAINS. Loading lazily per brain is fine.
        Nothing to load at startup. Each brain is loaded on first use, in _get_brain()."""
        
    def _get_brain(self, brain: str):
        if brain not in BRAINS:
            raise ValueError(f"brain must be one of {sorted(BRAINS)}, got {brain!r}")
        if brain not in self._models:
            from core.models import get_llm

            adapter = _resolve_adapter() if brain == "rlm" else None
            model_id = BRAINS["thinking"]() if brain == "thinking" else BRAINS["base"]()
            if adapter:
                from peft import PeftConfig
                trained_on = PeftConfig.from_pretrained(adapter).base_model_name_or_path
                if trained_on and trained_on != model_id:
                    warnings.warn(f"adapter trained on {trained_on}, but the loaded model is {model_id}.", stacklevel=2)
            self._models[brain] = get_llm(model_id, adapter)
        return self._models[brain]

    def system_prompt(self, max_steps: int) -> str:
        return REACT_SYSTEM_PROMPT.format(
            max_steps=max_steps, tools_description=self.registry.render_for_prompt()
        )

    def generate(self, messages: list[dict], brain: str, thinking: bool | None = None) -> str:
        """One generation with the chosen brain.

        ``thinking`` forces the chat template flag; by default only ``base`` does not think.
        """
        import torch

        tokenizer, model = self._get_brain(brain)
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=(brain != "base") if thinking is None else thinking,
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                # stop as soon as the first tool call is closed: the model must not go on to
                # write the "Observation:" itself (it makes up tool results)
                stop_strings=["</tool_call>"],
                tokenizer=tokenizer,
            )
        self._tokens_used += int(out.shape[1])  # prompt + generated tokens of this call
        return tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    def _call_tool(self, name: str, arguments: dict) -> ToolCallRecord:
        """Run one tool call with a time limit. Never raises: a failure comes back as a record.

        Tools marked ``requires_confirmation`` are not run: nobody can confirm during a
        ``POST /agent`` call, so the model is told and can choose another way.
        """
        spec = self.registry.tools.get(name)
        if spec is not None and spec.requires_confirmation:
            return ToolCallRecord(
                name=name,
                arguments=arguments,
                ok=False,
                error=f"tool '{name}' requires user confirmation and cannot run here. "
                "Use another tool or explain what you would do in final_answer.",
                requires_confirmation=True,
            )
        box: list[ToolCallRecord] = []
        worker = threading.Thread(
            target=lambda: box.append(self.registry.call(name, arguments)), daemon=True
        )
        worker.start()
        worker.join(TOOL_TIMEOUT_S)
        if worker.is_alive():  # a thread cannot be killed: it is abandoned (daemon) and we move on
            return ToolCallRecord(
                name=name,
                arguments=arguments,
                ok=False,
                error=f"tool '{name}' timed out after {TOOL_TIMEOUT_S}s. Try another approach.",
                latency_ms=1000 * TOOL_TIMEOUT_S,
            )
        return box[0]

    def summarize(self, messages: list[dict], brain: str) -> str:
        """Summarise old messages with the brain itself, for ``ConversationMemory.compress``.

        Thinking is off: a summary does not need a reasoning block and it would eat tokens.
        If the model returns nothing usable (for example a ``<think>`` cut by the token limit),
        we fall back to the start of the history rather than lose it all.
        """
        history = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
        raw = self.generate(
            [{"role": "user", "content": SUMMARY_PROMPT.format(history=history)}],
            brain,
            thinking=False,
        )
        summary = raw.rsplit("</think>", 1)[-1].strip()
        if not summary or summary.startswith("<think>"):
            summary = history[:SUMMARY_FALLBACK_CHARS]
        return summary

    def _model_label(self, brain: str) -> str:
        """Which model answered, written like ``/rag`` does: base model (+ adapter)."""
        if brain == "thinking":
            return BRAINS["thinking"]()
        if brain == "rlm":
            return f"{BRAINS['base']()} + {BRAINS['rlm']()}"
        return BRAINS["base"]()

    def run(self, task: str, max_steps: int = 10, brain: str = "rlm") -> AgentResponse:
        """What ``POST /agent`` calls: runs the loop and fills in ``model`` and ``tokens_used``."""
        self._tokens_used = 0
        response = self._run_loop(task, max_steps, brain)
        response.model = self._model_label(brain)
        response.tokens_used = self._tokens_used
        return response

    def _run_loop(self, task: str, max_steps: int, brain: str) -> AgentResponse:
        """The ReAct loop.
        """
        self.steps = []
        tokenizer, _ = self._get_brain(brain)

        def count_tokens(text: str) -> int:
            return len(tokenizer.encode(text))

        memory = ConversationMemory(
            self.system_prompt(max_steps), count_tokens, self.token_budget, pin_first=True
        )
        memory.add("user", task)
        nudged = False  # True after we already asked the model once to fix its format
        seen: dict[tuple[str, str], int] = {}  # (tool, arguments) -> times called
        for step in range(max_steps):
            raw = self.generate(memory.render(), brain)
            if raw.count("<tool_call>") > raw.count("</tool_call>"):
                raw += "\n</tool_call>"  # small models sometimes stop before closing the tag
            parsed = parse_tool_calls(raw)
            if not parsed.has_tool_calls:
                fallback = _action_line_fallback(raw, self.registry.tools)
                if fallback is not None:
                    parsed.tool_calls = [fallback]
            thought = parsed.thinking or parsed.text
            problem = None  # why this step cannot be run, if it cannot
            if parsed.has_tool_calls:
                first = next(c for c in parsed.tool_calls if c.error is None)
                if first.name == "final_answer" and not str(first.arguments.get("answer", "")).strip():
                    problem = (
                        "final_answer was called without an answer. Call it again with the "
                        'complete answer: <tool_call>{"name": "final_answer", "arguments": '
                        '{"answer": "..."}}</tool_call>'
                    )
            if (not parsed.has_tool_calls or problem) and not nudged:
                # first format failure: tell the model what went wrong and let it try again
                nudged = True
                errors = [c.error for c in parsed.tool_calls if c.error]
                message = problem or (
                    (f"Invalid tool call: {errors[0]}. " if errors else "No tool call found. ")
                    + "Do not write 'Action: name'. Reply with one "
                    '<tool_call>{"name": "...", "arguments": {...}}</tool_call> block, '
                    "or call final_answer."
                )
                self.steps.append(AgentStep(thought=thought, observation=message))
                memory.add("assistant", raw)
                memory.add("user", message)
                continue
            if not parsed.has_tool_calls or problem:
                # second failure in a row: accept what the model wrote, but flag it
                self.steps.append(AgentStep(thought=thought, observation="(no tool call)"))
                return AgentResponse(
                    final_answer="" if problem else (parsed.text or thought),  # an empty final_answer is not an answer
                    steps = self.steps,
                    n_steps=step+1,
                    succeeded=False
                )
            nudged = False  # a valid call resets the counter
            call = parsed.tool_calls[0]
            action = AgentAction(name=call.name, arguments=call.arguments)
            if call.name == "final_answer":
                self.steps.append(AgentStep(thought=thought, action=action))
                return AgentResponse(
                    final_answer=str(call.arguments.get("answer", "")), 
                    steps=self.steps, 
                    n_steps=step + 1,
                )
            key = (call.name, json.dumps(call.arguments, sort_keys=True, default=str))
            seen[key] = seen.get(key, 0) + 1
            if seen[key] >= MAX_REPEATED_CALLS:
                # the model keeps asking for the same thing: stop instead of burning all the steps
                message = f"Loop detected: '{call.name}' was called {seen[key]} times with the same arguments."
                self.steps.append(AgentStep(thought=thought, action=action, observation=message))
                return AgentResponse(final_answer="", steps=self.steps, n_steps=step + 1, succeeded=False)
            record = self._call_tool(call.name, call.arguments)
            observation = str(record.result if record.ok else record.error)
            if seen[key] == 2:  # second identical call: run it, but warn the model
                observation += (
                    "\n[Warning: you already made this exact call and the result will not "
                    "change. Try different arguments or another tool, or call final_answer.]"
                )
            step_record = AgentStep(
                thought=thought, 
                action=AgentAction(
                    name=call.name, arguments=call.arguments
                ), 
                observation=observation)
            self.steps.append(step_record)
            memory.add("assistant", raw); memory.add("tool", observation, name=call.name)
            memory.compress(lambda old: self.summarize(old, brain))
        return AgentResponse(final_answer="", steps=self.steps, n_steps=max_steps,
                                succeeded=False)
