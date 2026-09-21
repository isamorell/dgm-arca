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

import os
from dataclasses import dataclass, field

from agent.prompts import REACT_SYSTEM_PROMPT
from api.schemas import AgentResponse, AgentStep
from tool_use.registry import ToolRegistry, tool
from tool_use.tools import default_registry


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
    # Tu turno: call your phase-3 retriever and render hits as "[id] (source)\\ntext".
    raise NotImplementedError("wire search_knowledge_base to rag/retriever.py")


BRAINS = {
    "base": lambda: os.environ.get("ARCA_RLM_BASE_MODEL", "Qwen/Qwen3-0.6B"),
    "rlm": lambda: os.environ.get("ARCA_RLM_ADAPTER", ""),
    "thinking": lambda: os.environ.get("ARCA_THINKING_MODEL", "Qwen/Qwen3-1.7B"),
}


@dataclass
class ReActAgent:
    registry: ToolRegistry = field(default_factory=default_registry)
    token_budget: int = 6000
    steps: list[AgentStep] = field(default_factory=list)

    @classmethod
    def from_env(cls) -> ReActAgent:
        agent = cls()
        agent.registry.register(final_answer, search_knowledge_base)
        agent.load()
        return agent

    def load(self) -> None:
        """Tu turno: load the model(s) behind BRAINS. Loading lazily per brain is fine."""
        raise NotImplementedError("Phase 4 is not implemented yet: see agent/README.md.")

    def system_prompt(self, max_steps: int) -> str:
        return REACT_SYSTEM_PROMPT.format(
            max_steps=max_steps, tools_description=self.registry.render_for_prompt()
        )

    def generate(self, messages: list[dict], brain: str) -> str:
        """Tu turno: one generation with the chosen brain."""
        raise NotImplementedError

    def run(self, task: str, max_steps: int = 10, brain: str = "rlm") -> AgentResponse:
        """Tu turno: the ReAct loop.

        Sketch::

            memory = ConversationMemory(self.system_prompt(max_steps), count_tokens, budget)
            memory.add("user", task)
            for step in range(max_steps):
                raw = self.generate(memory.render(), brain)
                parsed = parse_tool_calls(raw)
                thought = parsed.thinking or parsed.text
                if not parsed.has_tool_calls:
                    # the model answered without the final_answer tool: accept it, but flag it
                    ...
                call = parsed.tool_calls[0]
                if call.name == "final_answer":
                    return AgentResponse(final_answer=..., steps=..., n_steps=step + 1, ...)
                record = self.registry.call(call.name, call.arguments)
                observation = str(record.result if record.ok else record.error)
                step_record = AgentStep(thought=thought, action=AgentAction(...), observation=obs)
                self.steps.append(step_record)
                memory.add("assistant", raw); memory.add("tool", observation, name=call.name)
                memory.compress(self.summarize)
            return AgentResponse(final_answer="", steps=self.steps, n_steps=max_steps,
                                 succeeded=False)

        Things to decide and document: timeouts per tool, what happens on repeated identical
        calls (loop detection), and how tools with ``requires_confirmation`` are handled.
        """
        raise NotImplementedError
