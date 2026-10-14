"""Tests for rlm/compare_ood.py: intervals, paired test, classification and the report."""

from __future__ import annotations

import pytest

from rlm.compare_ood import (
    classify,
    compare,
    format_markdown,
    newcombe_difference,
    paired_counts,
    sign_test_p,
    summarize,
    wilson_interval,
)


def row(question, correct, family="infusion", predicted="5 mL/h", fmt=True, value_ok=None):
    """One result row as ``rlm.evaluate`` writes it (``value_ok`` feeds the fake verifier)."""
    return {
        "question": question,
        "family": family,
        "is_correct": correct,
        "predicted": predicted,
        "has_valid_format": fmt,
        "expected_value": "5",
        "_value_ok": correct if value_ok is None else value_ok,
    }


def value_matches(r):
    return r["_value_ok"]


def results(*flags_by_model):
    """{'m0': {'rows': [...]}, ...} with one row per flag."""
    return {
        f"m{i}": {"rows": [row(f"q{j}", flag) for j, flag in enumerate(flags)]}
        for i, flags in enumerate(flags_by_model)
    }


def test_wilson_interval_known_values():
    low, high = wilson_interval(0, 10)
    assert low == 0.0
    assert high == pytest.approx(3.8416 / 13.8416, abs=1e-4)
    low, high = wilson_interval(10, 10)
    assert high == 1.0
    assert low == pytest.approx(1 - 3.8416 / 13.8416, abs=1e-4)


def test_wilson_interval_contains_the_estimate_and_shrinks_with_n():
    low, high = wilson_interval(8, 10)
    assert low < 0.8 < high
    low_big, high_big = wilson_interval(80, 100)
    assert high_big - low_big < high - low


def test_wilson_interval_without_data_is_uninformative():
    assert wilson_interval(0, 0) == (0.0, 1.0)


def test_newcombe_difference_is_zero_for_equal_proportions():
    difference, low, high = newcombe_difference(80, 100, 80, 100)
    assert difference == 0
    assert low == pytest.approx(-high)
    assert low < 0 < high


def test_newcombe_difference_detects_a_large_drop():
    difference, low, high = newcombe_difference(180, 200, 40, 100)
    assert difference == pytest.approx(0.5)
    assert low > 0.3
    assert high < 0.7


@pytest.mark.parametrize(
    ("only_a", "only_b", "expected"),
    [(0, 0, 1.0), (5, 5, 1.0), (0, 6, 2 / 64), (6, 0, 2 / 64), (1, 7, 2 * 9 / 256)],
)
def test_sign_test_p(only_a, only_b, expected):
    assert sign_test_p(only_a, only_b) == pytest.approx(expected)


def test_classify_covers_the_four_kinds():
    assert classify(row("q", True), value_matches) == "ok"
    assert classify(row("q", False, predicted=None), value_matches) == "sin_answer"
    assert classify(row("q", False, value_ok=True), value_matches) == "valor_ok_unidad_mal"
    assert classify(row("q", False, value_ok=False), value_matches) == "valor_mal"


def test_summarize_counts_everything():
    rows = [
        row("q1", True, family="infusion"),
        row("q2", False, family="infusion", value_ok=False),
        row("q3", False, family="infusion_inverse", predicted=None, fmt=False),
        row("q4", True, family="infusion_inverse"),
    ]
    summary = summarize(rows, value_matches)
    assert summary["n"] == 4
    assert summary["pass@1"] == 0.5
    assert summary["valid_format"] == 0.75
    assert summary["kinds"] == {
        "ok": 2,
        "valor_ok_unidad_mal": 0,
        "valor_mal": 1,
        "sin_answer": 1,
    }
    assert summary["families"] == {
        "infusion": {"n": 2, "correct": 1},
        "infusion_inverse": {"n": 2, "correct": 1},
    }
    low, high = summary["ci"]
    assert low < 0.5 < high


def test_paired_counts_and_p_value():
    a = [row(f"q{i}", flag) for i, flag in enumerate([True, True, False, False, True])]
    b = [row(f"q{i}", flag) for i, flag in enumerate([True, False, False, True, False])]
    counts = paired_counts(a, b)
    assert counts["both"] == 1
    assert counts["only_a"] == 2
    assert counts["only_b"] == 1
    assert counts["neither"] == 1
    assert counts["p_value"] == pytest.approx(1.0)


def test_paired_counts_rejects_different_problems():
    a = [row("q1", True)]
    with pytest.raises(ValueError, match="same problems"):
        paired_counts(a, [row("other", True)])
    with pytest.raises(ValueError, match="same problems"):
        paired_counts(a, a + a)


def test_compare_reports_the_drop_and_the_pairs():
    test = results([True] * 9 + [False], [True] * 10)  # m0: 90 %, m1: 100 %
    ood = results([True] * 5 + [False] * 5, [True] * 8 + [False] * 2)  # 50 %, 80 %
    report = compare(test, ood, ["m0", "m1"], value_matches)
    assert report["models"]["m0"]["drop"] == pytest.approx(0.4)
    assert report["models"]["m1"]["drop"] == pytest.approx(0.2)
    assert len(report["pairs"]) == 1
    assert report["pairs"][0]["ood"]["only_b"] == 3
    assert report["pairs"][0]["ood"]["only_a"] == 0
    assert report["pairs"][0]["ood"]["p_value"] == pytest.approx(0.25)


def test_compare_names_the_missing_model():
    test = results([True, False])
    with pytest.raises(KeyError, match="m1"):
        compare(test, test, ["m0", "m1"], value_matches)


def test_format_markdown_has_the_tables_and_decimal_commas():
    test = results([True] * 9 + [False], [True] * 10)
    ood = results([True] * 5 + [False] * 5, [True] * 8 + [False] * 2)
    text = format_markdown(compare(test, ood, ["m0", "m1"], value_matches))
    assert "| m0 |" in text
    assert "| m1 |" in text
    assert "90,0 %" in text
    assert "50,0 %" in text
    assert "Aciertos por familia en OOD" in text
    assert "Comparación pareada" in text
    assert "Cautelas" in text


def test_format_markdown_with_one_model_has_no_pairs():
    test = results([True, False])
    text = format_markdown(compare(test, test, ["m0"], value_matches))
    assert "Comparación pareada" not in text
