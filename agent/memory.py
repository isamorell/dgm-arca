"""Conversation memory with a token budget.

The context window is the agent's working memory, and we saw in class that stuffing it
does not work: things in the middle get lost. ``ConversationMemory`` keeps the message
list, counts tokens, and when the budget is exceeded it asks you (``summarize``) to
compress the oldest turns into a summary that stays at the top of the conversation.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass
class ConversationMemory:
    system_prompt: str
    count_tokens: Callable[[str], int]
    token_budget: int = 6000
    keep_last: int = 6  # most recent messages that are never summarised
    messages: list[dict] = field(default_factory=list)
    summary: str = ""

    def add(self, role: str, content: str, **extra) -> None:
        self.messages.append({"role": role, "content": content, **extra})

    def tokens(self) -> int:
        text = self.system_prompt + self.summary + "".join(m["content"] for m in self.messages)
        return self.count_tokens(text)

    def over_budget(self) -> bool:
        return self.tokens() > self.token_budget

    def render(self) -> list[dict]:
        """Messages to send to the model: system (+ summary) followed by the live messages."""
        system = self.system_prompt
        if self.summary:
            system += f"\n\nSummary of the conversation so far:\n{self.summary}"
        return [{"role": "system", "content": system}, *self.messages]

    def compress(self, summarize: Callable[[list[dict]], str]) -> bool:
        """Tu turno (the policy): decide *what* to summarise and *when*.

        The mechanics are here: if over budget, take everything except the last
        ``keep_last`` messages, turn it into a summary with ``summarize`` (a call to your
        model with ``SUMMARY_PROMPT``) and prepend it to the existing summary. Return True
        if a compression happened. Then think: should tool observations be truncated
        before summarising? Should the first user message always stay verbatim?
        """
        if not self.over_budget() or len(self.messages) <= self.keep_last:
            return False
        old, recent = self.messages[: -self.keep_last], self.messages[-self.keep_last :]
        new_summary = summarize(old)
        self.summary = f"{self.summary}\n{new_summary}".strip()
        self.messages = recent
        return True
