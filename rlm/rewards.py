"""Verifiable reward functions for reasoning training (RLVR).

These are the two rewards DeepSeek used for R1-Zero, exactly as discussed in
class: a *format* reward that checks the ``<think>...</think><answer>...</answer>``
structure, and an *accuracy* reward that compares the final answer against the
ground truth using a deterministic verifier.

Both functions follow the signature expected by ``trl.GRPOTrainer``:

    reward_fn(prompts, completions, **kwargs) -> list[float]

``kwargs`` receives every extra column of the training dataset (for example
``answer``), which is how the ground truth reaches the reward function.

Completions may arrive as plain strings or as conversational messages
(``[{"role": "assistant", "content": "..."}]``). ``_completion_text`` handles both.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

# Full structure: a single think block followed by a single answer block, nothing else.
# The tempered groups ``(?:(?!</?tag>).)*`` forbid a second opening/closing tag inside a block,
# so "<answer>4</answer><answer>5</answer>" is rejected instead of silently accepted.
FORMAT_PATTERN = re.compile(
    r"^\s*<think>(?P<think>(?:(?!</?think>).)*)</think>"
    r"\s*<answer>(?P<answer>(?:(?!</?answer>).)*)</answer>\s*$",
    re.DOTALL,
)
# Looser pattern used to *extract* an answer even when the format is imperfect.
ANSWER_PATTERN = re.compile(r"<answer>(?P<answer>.*?)</answer>", re.DOTALL)
# Fallback for math-style completions that finish with \boxed{...}.
BOXED_PATTERN = re.compile(r"\\boxed\{(?P<answer>[^{}]*)\}")
# Something that looks like a number: optional sign, digits, thousands separators, decimals.
NUMBER_PATTERN = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def _completion_text(completion: str | Sequence[dict]) -> str:
    """Return the text of a completion, whether it is a string or a list of messages."""
    if isinstance(completion, str):
        return completion
    parts = []
    for message in completion:
        if isinstance(message, dict) and message.get("role", "assistant") == "assistant":
            parts.append(str(message.get("content", "")))
    return "\n".join(parts)


def has_valid_format(text: str) -> bool:
    """True when the completion is exactly ``<think>...</think><answer>...</answer>``."""
    return FORMAT_PATTERN.match(text) is not None


def extract_answer(text: str) -> str | None:
    """Extract the final answer from a completion.

    Priority: the ``<answer>`` block, then a ``\\boxed{}`` expression. Returns ``None``
    when neither is present, so that callers can distinguish "no answer" from "wrong answer".
    """
    matches = ANSWER_PATTERN.findall(text)
    if matches:
        # The *last* block: models sometimes echo the format instructions before answering.
        return matches[-1].strip()
    boxed = BOXED_PATTERN.findall(text)
    if boxed:
        return boxed[-1].strip()
    return None


def normalize_number(text: str) -> str | None:
    """Pull the last number out of a piece of text and normalise it.

    ``"The answer is $1,234.50."`` -> ``"1234.5"``; ``"12"`` -> ``"12"``; no number -> ``None``.
    We take the *last* number because models often restate intermediate values before
    committing to a final one.
    """
    numbers = NUMBER_PATTERN.findall(text.replace("$", ""))
    if not numbers:
        return None
    raw = numbers[-1].replace(",", "")
    try:
        value = float(raw)
    except ValueError:
        return None
    if value.is_integer():
        return str(int(value))
    return f"{value:g}"


def numbers_match(predicted: str | None, expected: str | None) -> bool:
    """Compare two answers numerically after normalisation."""
    if predicted is None or expected is None:
        return False
    return normalize_number(predicted) == normalize_number(expected)


def format_reward(prompts: Sequence, completions: Sequence, **kwargs) -> list[float]:
    """1.0 if the completion respects the think/answer format, else 0.0."""
    return [1.0 if has_valid_format(_completion_text(c)) else 0.0 for c in completions]


def accuracy_reward(
    prompts: Sequence, completions: Sequence, answer: Sequence[str], **kwargs
) -> list[float]:
    """1.0 if the extracted final answer matches the ground truth numerically, else 0.0.

    ``answer`` is the dataset column holding the ground truth (one per completion).
    """
    rewards = []
    for completion, expected in zip(completions, answer, strict=True):
        predicted = extract_answer(_completion_text(completion))
        rewards.append(1.0 if numbers_match(predicted, expected) else 0.0)
    return rewards


def thinking_length(text: str) -> int:
    """Number of whitespace-separated tokens inside the ``<think>`` block (0 if absent)."""
    match = FORMAT_PATTERN.match(text)
    if match is None:
        match = re.search(r"<think>(?P<think>.*?)</think>", text, re.DOTALL)
    if match is None:
        return 0
    return len(match.group("think").split())
