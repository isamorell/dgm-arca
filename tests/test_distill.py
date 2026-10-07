"""Tests for the pure logic of rlm.distill (no GPU, no model downloads)."""

from __future__ import annotations

import json
from argparse import Namespace

import pytest

from rlm.distill import (
    REASON_ACCEPTED,
    REASON_BAD_FORMAT,
    REASON_NO_ANSWER,
    REASON_NOT_SPANISH,
    REASON_TOO_LONG,
    REASON_TOO_SHORT,
    REASON_TRUNCATED,
    REASON_WRONG_UNIT,
    REASON_WRONG_VALUE,
    canonical_trace,
    filter_raw,
    format_summary,
    judge_generation,
    load_problems,
    problem_level,
    select_problems,
    select_traces,
    spanish_ratio,
    split_teacher_output,
    summarize,
    teacher_messages,
    to_sft_row,
)
from rlm.rewards import has_valid_format
from rlm.verifier import MedicationVerifier

SPANISH_THINKING = (
    "Primero convierto la dosis a miligramos: 0,5 g son 500 mg. Después calculo el volumen "
    "que necesito para la concentración que pide el enunciado, así que divido los miligramos "
    "entre la concentración y resto el volumen del vial. El resultado es el volumen de "
    "diluyente que hay que añadir en la bolsa."
)


def make_record(raw: str, **overrides) -> dict:
    record = {
        "id": 0,
        "sample": 0,
        "question": "¿Cuánto diluyente, en ml?",
        "answer": "95.0",
        "answer_unit": "mL",
        "answer_unit_aliases": ["ml", "cc"],
        "family": "dilution",
        "level": "2",
        "raw": raw,
        "n_new_tokens": 400,
        "finished": True,
        "teacher": "Qwen/Qwen3-4B",
    }
    record.update(overrides)
    return record


def teacher_text(thinking: str = SPANISH_THINKING, final: str = "<answer>95,0 mL</answer>") -> str:
    return f"<think>\n{thinking}\n</think>\n\n{final}"


VERIFIER = MedicationVerifier()


class TestSplitTeacherOutput:
    def test_with_and_without_opening_tag(self):
        for raw in ("<think>\nrazono\n</think>\n\nrespuesta", "razono\n</think>\n\nrespuesta"):
            assert split_teacher_output(raw) == ("razono", "respuesta")

    def test_truncated_generation(self):
        assert split_teacher_output("<think>\nrazono y razono sin acabar") == (None, None)


class TestPrefill:
    def test_prefilled_reasoning_is_part_of_the_trace(self):
        from rlm.distill import THINK_PREFILL

        raw = THINK_PREFILL + SPANISH_THINKING + "\n</think>\n\n<answer>95,0 mL</answer>"
        judged = judge_generation(make_record(raw), VERIFIER)
        assert judged["verdict"] == REASON_ACCEPTED
        assert judged["trace"].startswith("<think>\nVale, voy a resolverlo paso a paso.")


class TestTidyWhitespace:
    def test_double_spaces_are_collapsed_but_newlines_stay(self):
        from rlm.distill import tidy_whitespace

        assert tidy_whitespace("a.  b\t\tc\n\nd  e") == "a. b c\n\nd e"

    def test_trace_has_no_double_spaces(self):
        assert "  " not in canonical_trace("uno.  dos.   tres", "5 mL")


class TestCanonicalTrace:
    def test_has_the_format_the_reward_checks(self):
        trace = canonical_trace("pienso", "95,0 mL")
        assert has_valid_format(trace)
        assert trace.endswith("<answer>95,0 mL</answer>")


class TestSpanishRatio:
    def test_spanish_text(self):
        ratio, n = spanish_ratio(SPANISH_THINKING)
        assert ratio > 0.9 and n >= 5

    def test_english_text(self):
        ratio, _ = spanish_ratio(
            "First we need to convert the dose and then we divide it by the volume"
        )
        assert ratio < 0.2

    def test_no_recognised_words(self):
        assert spanish_ratio("500 mg 5 mL") == (0.0, 0)


class TestJudgeGeneration:
    def verdict(self, raw: str, **overrides) -> str:
        return judge_generation(make_record(raw, **overrides), VERIFIER)["verdict"]

    def test_accepts_correct_spanish_trace_with_decimal_comma(self):
        judged = judge_generation(make_record(teacher_text()), VERIFIER)
        assert judged["verdict"] == REASON_ACCEPTED
        assert has_valid_format(judged["trace"])
        assert judged["predicted"] == "95,0 mL"

    def test_truncated_when_think_never_closes(self):
        assert self.verdict("<think>\nrazono " * 50) == REASON_TRUNCATED

    def test_truncated_when_generation_did_not_finish(self):
        assert self.verdict(teacher_text(), finished=False) == REASON_TRUNCATED

    def test_no_answer(self):
        assert self.verdict(teacher_text(final="El resultado es 95 mL.")) == REASON_NO_ANSWER

    def test_boxed_fallback_is_canonicalised(self):
        judged = judge_generation(make_record(teacher_text(final="\\boxed{95 mL}")), VERIFIER)
        assert judged["verdict"] == REASON_ACCEPTED
        assert "<answer>95 mL</answer>" in judged["trace"]

    def test_wrong_value(self):
        assert self.verdict(teacher_text(final="<answer>90 mL</answer>")) == REASON_WRONG_VALUE

    def test_right_value_wrong_unit(self):
        assert self.verdict(teacher_text(final="<answer>95 mg</answer>")) == REASON_WRONG_UNIT

    def test_right_value_missing_unit(self):
        assert self.verdict(teacher_text(final="<answer>95</answer>")) == REASON_WRONG_UNIT

    def test_unit_alias_is_accepted(self):
        assert self.verdict(teacher_text(final="<answer>95 cc</answer>")) == REASON_ACCEPTED

    def test_stray_tag_inside_reasoning_is_bad_format(self):
        thinking = SPANISH_THINKING + " Pongo <answer>95</answer> al final."
        assert self.verdict(teacher_text(thinking=thinking)) == REASON_BAD_FORMAT

    def test_correct_but_almost_no_reasoning(self):
        assert self.verdict(teacher_text(thinking="Son 95 mL.")) == REASON_TOO_SHORT

    def test_correct_but_english(self):
        thinking = (
            "First we need to convert the dose from grams to milligrams, so the total is "
            "500 mg. Then we divide it by the concentration that the statement gives and "
            "we subtract the volume of the vial to get the diluent that is needed in the bag."
        )
        assert self.verdict(teacher_text(thinking=thinking)) == REASON_NOT_SPANISH

    def test_correct_but_too_long_for_the_student(self):
        assert self.verdict(teacher_text(), n_new_tokens=1900) == REASON_TOO_LONG

    def test_wrong_answers_are_never_reported_as_quality_problems(self):
        raw = teacher_text(thinking="Son 90.", final="<answer>90 mL</answer>")
        assert self.verdict(raw) == REASON_WRONG_VALUE


class TestSelectTraces:
    def judged(self):
        rows = []
        for sample, tokens in enumerate([900, 300, 600]):
            rows.append(
                {"id": 7, "sample": sample, "n_new_tokens": tokens, "verdict": REASON_ACCEPTED}
            )
        rows.append({"id": 7, "sample": 3, "n_new_tokens": 100, "verdict": REASON_WRONG_VALUE})
        rows.append({"id": 8, "sample": 0, "n_new_tokens": 100, "verdict": REASON_WRONG_VALUE})
        return rows

    def test_keeps_the_shortest_accepted_and_ignores_rejected(self):
        chosen = select_traces(self.judged(), keep_per_problem=2)
        assert [row["n_new_tokens"] for row in chosen] == [300, 600]

    def test_problem_without_accepted_traces_contributes_nothing(self):
        assert all(row["id"] != 8 for row in select_traces(self.judged(), 5))

    def test_random_strategy_respects_the_cap(self):
        assert len(select_traces(self.judged(), 2, strategy="random", seed=1)) == 2


class TestProblems:
    @pytest.fixture
    def problems(self):
        rows = []
        for family in ("dilution", "drops"):
            for i in range(5):
                rows.append(
                    {
                        "question": f"q{family}{i}",
                        "answer": i,
                        "family": family,
                        "branches": json.dumps({"level": "1"}),
                    }
                )
        return rows

    def test_per_family_is_balanced_and_ids_point_to_the_full_file(self, problems):
        picked = select_problems(problems, per_family=2, seed=0)
        assert len(picked) == 4
        assert sorted(p["family"] for _, p in picked) == ["dilution"] * 2 + ["drops"] * 2
        assert all(problems[pid] is problem for pid, problem in picked)

    def test_limit(self, problems):
        assert len(select_problems(problems, limit=3)) == 3

    def test_level_is_read_from_branches(self, problems):
        assert problem_level(problems[0]) == "1"
        assert problem_level({"branches": "no es json"}) == "?"

    def test_load_problems_checks_fields_and_stringifies_answer(self, tmp_path):
        good = tmp_path / "good.jsonl"
        good.write_text(json.dumps({"question": "q", "answer": 5}) + "\n", encoding="utf-8")
        assert load_problems(good)[0]["answer"] == "5"
        bad = tmp_path / "bad.jsonl"
        bad.write_text(json.dumps({"question": "q"}) + "\n", encoding="utf-8")
        with pytest.raises(ValueError):
            load_problems(bad)


def test_teacher_messages_ask_for_spanish_without_changing_the_question():
    messages = teacher_messages("¿Cuánto?")
    assert messages[0]["role"] == "system" and "<answer>" in messages[0]["content"]
    assert messages[1]["content"].startswith("¿Cuánto?") and "español" in messages[1]["content"]


def test_filter_summary_and_sft_rows_end_to_end():
    raw = [
        make_record(teacher_text(), id=0, sample=0),
        make_record(teacher_text(final="<answer>90 mL</answer>"), id=0, sample=1),
        make_record("<think>\nsin acabar", id=1, sample=0, family="drops", finished=False),
    ]
    args = Namespace(
        min_think_words=30,
        min_spanish_ratio=0.6,
        max_trace_tokens=1800,
        keep_per_problem=2,
        selection="shortest",
        seed=0,
    )
    judged, selected = filter_raw(raw, args)
    summary = summarize(judged, n_problems=2)
    assert summary["generated"] == 3 and summary["accepted"] == 1
    assert summary["reasons"][REASON_WRONG_VALUE] == 1 and summary["reasons"][REASON_TRUNCATED] == 1
    assert summary["problems_with_a_verified_trace"] == 1
    assert summary["by_family"]["drops"] == {"generated": 1, "accepted": 0}
    assert "| wrong_value | 1 |" in format_summary(summary, n_selected=len(selected))

    row = to_sft_row(selected[0])
    assert row["verified"] is True and has_valid_format(row["trace"])
    assert {"question", "trace", "verified"} <= row.keys()  # what rlm.train_sft reads
