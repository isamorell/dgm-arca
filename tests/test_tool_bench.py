"""Phase 2 benchmark: the 30 cases are well formed and the scoring does what it claims.

Nothing here needs a model or the network: it checks the bench file against the real tool
definitions and the metrics against hand-made model outputs.
"""

import math
from pathlib import Path

import pytest

from tool_use.evaluate import (
    answer_ok,
    arguments_match,
    build_summary,
    compare_conditions,
    load_bench,
    numbers_in,
    rescore,
    score_case,
    summarise,
    wilson_interval,
)
from tool_use.tools import default_registry, dose_calculator

BENCH = Path(__file__).resolve().parents[1] / "tool_use" / "bench" / "cases.jsonl"


@pytest.fixture(scope="module")
def cases():
    return load_bench(BENCH)


# ---- the bench file ----------------------------------------------------------------------


def test_bench_has_thirty_cases_with_unique_ids(cases):
    assert len(cases) == 30
    assert len({c["id"] for c in cases}) == 30
    assert len({c["query"] for c in cases}) == 30


def test_bench_has_cases_that_need_no_tool(cases):
    free = [c for c in cases if not c["needs_tool"]]
    assert 5 <= len(free) <= 10
    assert all(c["expected_tool"] is None and not c["expected_arguments"] for c in free)


def test_every_tool_is_covered_by_the_bench(cases):
    used = {c["expected_tool"] for c in cases if c["needs_tool"]}
    assert used == set(default_registry().tools)


def test_expected_calls_use_real_tools_and_valid_arguments(cases):
    registry = default_registry()
    for case in (c for c in cases if c["needs_tool"]):
        tool = registry.tools[case["expected_tool"]]
        tool.validate(case["expected_arguments"])  # raises if the schema does not accept it
        for key in case.get("free_text_args", []):
            assert key in case["expected_arguments"]


def test_multi_step_cases_start_with_their_expected_tool(cases):
    for case in (c for c in cases if "expected_sequence" in c):
        assert case["expected_sequence"][0] == case["expected_tool"]
        assert len(case["expected_sequence"]) >= 2


def test_expected_answers_agree_with_the_tools(cases):
    """Single-step calculation cases: the expression really gives the expected answer."""
    for case in (c for c in cases if c["expected_tool"] == "dose_calculator"):
        if "expected_sequence" in case:
            continue
        value = dose_calculator(**case["expected_arguments"])
        assert math.isclose(value, case["expected_answer"], abs_tol=0.01), case["id"]


def test_conversion_cases_agree_with_pint(cases):
    registry = default_registry()
    for case in (c for c in cases if c["expected_tool"] == "convert_units"):
        if "expected_sequence" in case:
            continue
        value = registry.tools["convert_units"](**case["expected_arguments"])
        assert math.isclose(value, case["expected_answer"], abs_tol=0.01), case["id"]


def test_the_gold_calls_score_perfectly(cases):
    for case in cases:
        gold = (
            [{"name": case["expected_tool"], "arguments": case["expected_arguments"]}]
            if case["needs_tool"]
            else []
        )
        if "expected_sequence" in case:
            gold = [{"name": n, "arguments": {}} for n in case["expected_sequence"]]
            gold[0]["arguments"] = case["expected_arguments"]
        scores = score_case(case, gold)
        assert all(scores.values()), (case["id"], scores)


# ---- argument matching -------------------------------------------------------------------


def test_expressions_are_compared_by_value():
    expected = {"expression": "15 * 70"}
    assert arguments_match("dose_calculator", {"expression": "15*70"}, expected)
    assert arguments_match("dose_calculator", {"expression": "70 * 15"}, expected)
    assert arguments_match("dose_calculator", {"expression": "(15 * 70)"}, expected)
    assert not arguments_match("dose_calculator", {"expression": "15 * 7"}, expected)
    assert not arguments_match("dose_calculator", {"expression": "quince por 70"}, expected)


def test_arguments_left_at_their_default_count_as_given():
    expected = {"active_ingredient": "furosemida"}
    explicit = {"active_ingredient": "furosemida", "max_results": 5}
    assert arguments_match("search_medication_cima", explicit, expected)
    assert not arguments_match(
        "search_medication_cima", {"active_ingredient": "furosemida", "max_results": 3}, expected
    )
    assert arguments_match(
        "dose_calculator", {"expression": "1 / 2", "decimals": 2}, {"expression": "1 / 2"}
    )
    # an expected non-default value cannot be skipped
    assert not arguments_match(
        "search_medication_cima", expected, {"active_ingredient": "amoxicilina", "max_results": 3}
    )


def test_units_are_compared_by_conversion_not_by_spelling():
    expected = {"value": 250, "from_unit": "mcg", "to_unit": "mg"}
    assert arguments_match(
        "convert_units", {"value": 250.0, "from_unit": "µg", "to_unit": "MG".lower()}, expected
    )
    assert arguments_match(
        "convert_units", {"value": "250", "from_unit": "mcg", "to_unit": "milligram"}, expected
    )
    assert not arguments_match(
        "convert_units", {"value": 250, "from_unit": "mg", "to_unit": "mcg"}, expected
    )
    assert not arguments_match(
        "convert_units", {"value": 25, "from_unit": "mcg", "to_unit": "mg"}, expected
    )
    assert arguments_match(
        "convert_units",
        {"value": 1.5, "from_unit": "l", "to_unit": "ml"},
        {"value": 1.5, "from_unit": "L", "to_unit": "mL"},
    )


def test_text_arguments_ignore_case_accents_and_spacing():
    expected = {"active_ingredient": "omeprazol"}
    assert arguments_match("search_medication_cima", {"active_ingredient": " Omeprazol "}, expected)
    assert not arguments_match(
        "search_medication_cima", {"active_ingredient": "esomeprazol"}, expected
    )


def test_extra_or_missing_arguments_do_not_match():
    expected = {"value": 1, "from_unit": "g", "to_unit": "mg"}
    assert not arguments_match("convert_units", {"value": 1, "from_unit": "g"}, expected)
    assert not arguments_match("convert_units", {**expected, "precision": 2}, expected)


def test_free_text_arguments_only_need_to_be_present():
    expected = {
        "medication": "Paracetamol",
        "concentration": "10 mg/mL",
        "dose": "1000 mg",
        "administration": "100 mL",
        "steps": "1000 mg / 10 mg/mL = 100 mL",
    }
    got = {
        **expected,
        "steps": "Dividir la dosis entre la concentración",
        "concentration": "10mg/ml",
    }
    assert arguments_match("create_preparation_sheet", got, expected, ["steps"])
    assert not arguments_match(
        "create_preparation_sheet", {**got, "steps": " "}, expected, ["steps"]
    )
    assert not arguments_match(
        "create_preparation_sheet", {**got, "dose": "500 mg"}, expected, ["steps"]
    )


# ---- tool selection ----------------------------------------------------------------------


def test_selection_for_cases_that_need_no_tool():
    case = {"needs_tool": False, "expected_tool": None}
    assert score_case(case, []) == {
        "selection_ok": True,
        "arguments_ok": True,
        "arguments_lenient_ok": True,
    }
    assert not score_case(case, [{"name": "dose_calculator", "arguments": {}}])["selection_ok"]


def test_wrong_tool_scores_zero_and_sequence_is_checked():
    case = {
        "needs_tool": True,
        "expected_tool": "convert_units",
        "expected_arguments": {"value": 1, "from_unit": "g", "to_unit": "mg"},
        "expected_sequence": ["convert_units", "dose_calculator"],
    }
    good = [
        {"name": "convert_units", "arguments": {"value": 1, "from_unit": "g", "to_unit": "mg"}},
        {"name": "dose_calculator", "arguments": {"expression": "1000 / 2"}},
    ]
    assert score_case(case, good) == {
        "selection_ok": True,
        "arguments_ok": True,
        "arguments_lenient_ok": True,
        "sequence_ok": True,
    }
    assert score_case(case, good[:1])["sequence_ok"] is False
    assert score_case(case, [good[1]])["selection_ok"] is False


# ---- final answer ------------------------------------------------------------------------


def test_numbers_are_read_in_spanish_and_english_formats():
    assert 2.5 in numbers_in("Son 2,5 mL")
    assert 2.5 in numbers_in("Son 2.5 mL")
    assert {1.05, 1050.0} <= numbers_in("1.050 mg")
    assert 1050.5 in numbers_in("1.050,5 mg")
    assert numbers_in("sin cifras") == set()


def test_answer_ok_uses_tolerance_and_ignores_numbers_from_the_query():
    case = {"query": "Dame 490 mg con 11,75 mg/mL", "expected_answer": 41.7}
    assert answer_ok(case, "Necesitas 41,70 mL.") is True
    assert answer_ok(case, "Necesitas 41,2 mL.") is False
    assert answer_ok(case, "Con 490 mg y 11,75 mg/mL salen unos 40 mL.") is False
    assert answer_ok({"query": "q", "expected_answer": 1.6667}, "1,67 mL/min") is True
    assert answer_ok({"query": "q"}, "lo que sea") is None


def test_answer_ok_rejects_small_arithmetic_slips():
    """41,75 and 41,76 are not 41,7: a 0,5 % tolerance used to accept them."""
    case = {"query": "Dame 490 mg con 11,75 mg/mL", "expected_answer": 41.7}
    assert answer_ok(case, "Salen 41,75 mL.") is False
    assert answer_ok(case, "Salen 41,76 mL.") is False
    assert answer_ok(case, "Salen 41,702 mL.") is True
    assert answer_ok({"query": "q", "expected_answer": 2.27}, "2.26796185 mg/lb") is True


def test_answer_ok_accepts_an_expected_number_that_is_in_the_query():
    case = {"query": "Son 4 mL y quiero confirmarlo", "expected_answer": 4}
    assert answer_ok(case, "Sí, son 4 mL.") is True


# ---- aggregation -------------------------------------------------------------------------


def _row(category, needs_tool, selection, arguments, calls, answer_ok_=None, cid="x"):
    return {
        "id": cid,
        "category": category,
        "needs_tool": needs_tool,
        "selection_ok": selection,
        "arguments_ok": arguments,
        "tool_calls": calls,
        "answer_ok": answer_ok_,
    }


def test_summary_separates_false_calls_missed_calls_and_wrong_tools():
    call = [{"name": "convert_units", "arguments": {}}]
    rows = [
        _row("a", True, True, True, call, True, "1"),
        _row("a", True, False, False, [], False, "2"),  # missed call
        _row("a", True, False, False, call, False, "3"),  # wrong tool
        _row("b", False, False, False, call, None, "4"),  # false call
        _row("b", False, True, True, [], None, "5"),
    ]
    s = summarise(rows)
    assert s["tool_selection_accuracy"] == pytest.approx(2 / 5)
    assert s["argument_accuracy"] == pytest.approx(2 / 5)
    assert s["false_call_rate"] == pytest.approx(1 / 2)
    assert s["missed_call_rate"] == pytest.approx(1 / 3)
    assert s["wrong_tool_rate"] == pytest.approx(1 / 3)
    assert s["answer_accuracy"] == pytest.approx(1 / 3) and s["n_answer_cases"] == 3
    assert s["by_category"]["a"]["n"] == 3


def test_compare_conditions_counts_cases_by_who_got_them_right():
    with_tools = [
        _row("a", True, True, True, [], True, "1"),
        _row("a", True, True, True, [], True, "2"),
        _row("a", True, True, True, [], False, "3"),
        _row("a", True, True, True, [], False, "4"),
        _row("a", True, True, True, [], None, "5"),
    ]
    without = [
        _row("a", True, True, True, [], False, "1"),
        _row("a", True, True, True, [], True, "2"),
        _row("a", True, True, True, [], True, "3"),
        _row("a", True, True, True, [], False, "4"),
        _row("a", True, True, True, [], None, "5"),
    ]
    result = compare_conditions(with_tools, without)
    assert (result["only_with_tools"], result["only_without_tools"]) == (1, 1)
    assert (result["both_correct"], result["both_wrong"], result["n_cases"]) == (1, 1, 4)
    assert result["correct_with_tools"] == pytest.approx(0.5)


def test_wilson_interval_is_wide_with_thirty_cases():
    low, high = wilson_interval(24, 30)
    assert low < 0.8 < 0.9 < high
    assert wilson_interval(0, 0) == (0.0, 0.0)


# ---- optional arguments and rescoring ----------------------------------------------------


def test_optional_arguments_can_be_ignored_when_the_expected_call_does_not_mention_them():
    expected = {"active_ingredient": "furosemida"}
    got = {"active_ingredient": "furosemida", "max_results": 10}
    assert not arguments_match("search_medication_cima", got, expected)
    assert arguments_match("search_medication_cima", got, expected, ignore_optional=True)
    # a wrong required argument is still wrong, and so is an optional one the case fixes
    wrong = {"active_ingredient": "ibuprofeno", "max_results": 10}
    assert not arguments_match("search_medication_cima", wrong, expected, ignore_optional=True)
    fixed = {"active_ingredient": "amoxicilina", "max_results": 3}
    assert not arguments_match(
        "search_medication_cima",
        {"active_ingredient": "amoxicilina", "max_results": 10},
        fixed,
        ignore_optional=True,
    )


def test_score_case_reports_strict_and_lenient_arguments():
    case = {
        "needs_tool": True,
        "expected_tool": "search_medication_cima",
        "expected_arguments": {"active_ingredient": "furosemida"},
    }
    calls = [
        {
            "name": "search_medication_cima",
            "arguments": {"active_ingredient": "furosemida", "max_results": 10},
        }
    ]
    scores = score_case(case, calls)
    assert scores["arguments_ok"] is False and scores["arguments_lenient_ok"] is True


def test_rescore_rewrites_the_scores_from_the_stored_behaviour():
    case = {
        "id": "c1",
        "category": "calculo",
        "query": "Dame 490 mg con 11,75 mg/mL",
        "needs_tool": False,
        "expected_tool": None,
        "expected_arguments": {},
        "expected_answer": 41.7,
    }
    data = {
        "rows": {
            "tools": [
                {
                    **case,
                    "tool_calls": [],
                    "answer": "Salen 41,75 mL.",
                    "answer_ok": True,
                    "selection_ok": False,
                    "arguments_ok": False,
                }
            ],
            "no_tools": [
                {**case, "tool_calls": [], "answer": "Salen 41,7 mL.", "answer_ok": False}
            ],
        },
        "summary": {},
    }
    result = rescore(data)
    assert result["rows"]["tools"][0]["answer_ok"] is False
    assert result["rows"]["tools"][0]["selection_ok"] is True
    assert result["rows"]["no_tools"][0]["answer_ok"] is True
    assert set(result["summary"]) == {"tools", "no_tools", "comparison"}
    assert result["summary"]["comparison"]["only_without_tools"] == 1


def test_build_summary_adds_the_lenient_argument_accuracy():
    rows = {"tools": [_row("a", True, True, True, [], None, "1")]}
    rows["tools"][0]["arguments_lenient_ok"] = True
    assert build_summary(rows)["tools"]["argument_accuracy_ignoring_optional"] == 1.0
    assert set(build_summary(rows)) == {"tools"}
