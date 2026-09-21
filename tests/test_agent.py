"""Memory mechanics and the benchmark checker are given; the loop is yours."""

from agent.evaluate import check_answer
from agent.memory import ConversationMemory


def test_memory_compresses_old_messages_when_over_budget():
    memory = ConversationMemory("sys", count_tokens=len, token_budget=60, keep_last=2)
    for i in range(6):
        memory.add("user" if i % 2 == 0 else "assistant", f"message number {i} " * 2)
    assert memory.over_budget()
    compressed = memory.compress(lambda old: f"{len(old)} old messages summarised")
    assert compressed
    assert len(memory.messages) == 2
    assert "4 old messages" in memory.summary
    assert memory.render()[0]["role"] == "system" and "Summary" in memory.render()[0]["content"]


def test_memory_does_nothing_under_budget():
    memory = ConversationMemory("sys", count_tokens=len, token_budget=10_000)
    memory.add("user", "hello")
    assert not memory.compress(lambda old: "unused")


def test_check_answer_kinds():
    assert check_answer({"check": {"type": "contains", "value": "Madrid"}}, "It is in madrid.")
    assert check_answer({"check": {"type": "number", "value": 7}}, "The name has 7 letters")
    assert check_answer({"check": {"type": "number", "value": 21.5, "tolerance": 0.5}}, "≈ 21.2 °C")
    assert not check_answer({"check": {"type": "regex", "value": r"\bF\b"}}, "no unit here")
