"""Dataset helpers for phase 1.

Two things live here:

1. ``R1_ZERO_SYSTEM_PROMPT``: the system prompt DeepSeek used for R1-Zero, verbatim
   from the slides. It tells the model to think inside ``<think>`` and answer inside
   ``<answer>``. We use it for every reasoning dataset so that the format reward and
   the verifier always know where to look.

2. Loaders that turn a raw dataset into the *conversational* format ``GRPOTrainer``
   and ``SFTTrainer`` expect: a ``prompt`` column holding a list of chat messages and
   an ``answer`` column holding the ground truth as a string.

``load_gsm8k`` is the reference implementation (and what the smoke test uses).
``load_domain_dataset`` is the shape yours should have: read your problems from a
JSONL file with ``question`` and ``answer`` fields and return the same columns.
"""

from __future__ import annotations

from pathlib import Path

from datasets import Dataset, load_dataset

R1_ZERO_SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the "
    "Assistant solves it. The assistant first thinks about the reasoning process in the "
    "mind and then provides the user with the answer. The reasoning process and answer "
    "are enclosed within <think> </think> and <answer> </answer> tags, respectively, "
    "i.e., <think> reasoning process here </think> <answer> answer here </answer>."
)


def build_prompt(question: str, system_prompt: str = R1_ZERO_SYSTEM_PROMPT) -> list[dict]:
    """Wrap a question in the chat format used for training and inference."""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]


def gsm8k_final_answer(solution: str) -> str:
    """GSM8K stores the reference solution as text ending in ``#### <number>``."""
    return solution.split("####")[-1].strip().replace(",", "")


def load_gsm8k(split: str = "train", n_examples: int | None = None, seed: int = 0) -> Dataset:
    """Load GSM8K as (prompt, answer) pairs ready for TRL.

    ``prompt`` is a list of chat messages; ``answer`` is the final number as a string.
    Set ``n_examples`` to work with a random subset (the smoke test uses a few hundred).
    """
    dataset = load_dataset("openai/gsm8k", "main", split=split)
    if n_examples is not None:
        dataset = dataset.shuffle(seed=seed).select(range(min(n_examples, len(dataset))))
    return dataset.map(
        lambda ex: {
            "prompt": build_prompt(ex["question"]),
            "answer": gsm8k_final_answer(ex["answer"]),
        },
        remove_columns=dataset.column_names,
    )


def load_domain_dataset(path: str | Path, system_prompt: str = R1_ZERO_SYSTEM_PROMPT) -> Dataset:
    """Load a JSONL file of your own problems. Each line: {"question": ..., "answer": ...}.

    Keep any extra fields you need for your verifier (unit tests, expected SQL result,
    tolerance...): they are passed through to the reward functions as keyword arguments.
    """
    dataset = load_dataset("json", data_files=str(path), split="train")
    if "question" not in dataset.column_names or "answer" not in dataset.column_names:
        raise ValueError("The domain dataset needs at least 'question' and 'answer' fields.")
    return dataset.map(
        lambda ex: {
            "prompt": build_prompt(ex["question"], system_prompt),
            "answer": str(ex["answer"]),
        },
        remove_columns=["question"],
    )
