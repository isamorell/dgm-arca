"""Tests for the data side of rlm.train_sft (no GPU, no model downloads)."""

from __future__ import annotations

import json

import pytest

from rlm.train_sft import (
    drop_overlong,
    length_report,
    load_sft_dataset,
    load_verified_traces,
    split_by_problem,
    to_prompt_completion,
    token_length,
    warmup_steps,
)

TRACE = "<think>\nVale, voy a resolverlo paso a paso.\n</think>\n<answer>9 gotas/min</answer>"


def write_traces(path, n_problems: int = 20, per_problem: int = 2, unverified: int = 0):
    rows = []
    for problem in range(n_problems):
        for sample in range(per_problem):
            rows.append(
                {
                    "id": problem,
                    "question": f"pregunta {problem}",
                    "answer": "9",
                    "trace": TRACE,
                    "verified": True,
                    "family": "drops",
                    "n_new_tokens": 100 + sample,
                }
            )
    for extra in range(unverified):
        rows.append({"id": 1000 + extra, "question": "x", "trace": "mal", "verified": False})
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return path


class FakeTokenizer:
    """One token per whitespace-separated word of the rendered chat."""

    def apply_chat_template(self, messages, tokenize=False):
        assert tokenize is False
        return " ".join(m["content"] for m in messages)

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": text.split()}


def test_only_verified_traces_are_used(tmp_path):
    path = write_traces(tmp_path / "t.jsonl", n_problems=3, unverified=2)
    assert len(load_verified_traces(path)) == 6


def test_prompt_completion_format_matches_inference_prompt(tmp_path):
    dataset = load_sft_dataset(write_traces(tmp_path / "t.jsonl", n_problems=1))
    example = dataset[0]
    assert [m["role"] for m in example["prompt"]] == ["system", "user"]
    assert example["prompt"][1]["content"] == "pregunta 0"
    assert example["completion"] == [{"role": "assistant", "content": TRACE}]
    assert set(dataset.column_names) == {"prompt", "completion"}


class TestSplitByProblem:
    def test_no_problem_appears_on_both_sides(self, tmp_path):
        traces = load_verified_traces(write_traces(tmp_path / "t.jsonl", n_problems=40))
        train, held_out = split_by_problem(traces, 0.1, seed=0)
        assert set(train["id"]).isdisjoint(held_out["id"])
        assert len(set(held_out["id"])) == 4
        assert len(train) + len(held_out) == len(traces)

    def test_every_trace_of_a_held_out_problem_goes_with_it(self, tmp_path):
        traces = load_verified_traces(write_traces(tmp_path / "t.jsonl", n_problems=10))
        _, held_out = split_by_problem(traces, 0.2, seed=1)
        assert len(held_out) == 2 * len(set(held_out["id"]))

    def test_same_seed_same_split(self, tmp_path):
        traces = load_verified_traces(write_traces(tmp_path / "t.jsonl", n_problems=30))
        first = split_by_problem(traces, 0.1, seed=3)[1]["id"]
        assert first == split_by_problem(traces, 0.1, seed=3)[1]["id"]

    def test_zero_fraction_or_missing_id_means_no_eval_set(self, tmp_path):
        traces = load_verified_traces(write_traces(tmp_path / "t.jsonl"))
        assert split_by_problem(traces, 0.0)[1] is None
        assert split_by_problem(traces.remove_columns("id"), 0.1)[1] is None

    def test_training_set_is_never_empty(self, tmp_path):
        traces = load_verified_traces(write_traces(tmp_path / "t.jsonl", n_problems=2))
        train, held_out = split_by_problem(traces, 0.9)
        assert len(train) > 0 and len(held_out) > 0


class TestLengths:
    def test_token_length_counts_prompt_and_completion(self):
        prompt = [{"role": "user", "content": "uno dos"}]
        completion = [{"role": "assistant", "content": "tres cuatro cinco"}]
        assert token_length(FakeTokenizer(), prompt, completion) == 5

    def test_report(self):
        report = length_report(list(range(1, 101)))
        assert report == {"n": 100, "p50": 50, "p95": 95, "max": 100}
        assert length_report([]) == {"n": 0, "p50": 0, "p95": 0, "max": 0}

    def test_overlong_examples_are_dropped_not_truncated(self, tmp_path):
        rows = [
            {"question": "a", "trace": "x " * 5, "verified": True},
            {"question": "a", "trace": "x " * 500, "verified": True},
        ]
        path = tmp_path / "t.jsonl"
        path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
        dataset = to_prompt_completion(load_verified_traces(path))
        kept, report = drop_overlong(dataset, FakeTokenizer(), max_length=300)
        assert len(kept) == 1 and report["dropped"] == 1 and report["n"] == 2
        assert kept[0]["completion"][0]["content"] == "x " * 5


@pytest.mark.parametrize(
    ("n_examples", "epochs", "expected"),
    [(1500, 3.0, 14), (16, 1.0, 1), (100, 2.0, 1)],
)
def test_warmup_is_five_percent_of_the_steps(n_examples, epochs, expected):
    assert warmup_steps(n_examples, batch_size=2, grad_accum=8, epochs=epochs) == expected
