"""Tests for the verifiers. Add one class of tests per domain verifier you write."""

from rlm.verifier import ExactMatchVerifier, NumericVerifier


def test_numeric_verifier_exact():
    v = NumericVerifier()
    assert v.is_correct("42", "42")
    assert v.is_correct("$42.00", "42")
    assert not v.is_correct("41", "42")
    assert not v.is_correct(None, "42")
    assert not v.is_correct("forty-two", "42")


def test_numeric_verifier_with_tolerance():
    v = NumericVerifier(tolerance=0.05)
    assert v.is_correct("3.14", "3.1416")
    assert not v.is_correct("3.0", "3.1416")


def test_verify_extracts_from_full_completion():
    result = NumericVerifier().verify("<think>...</think><answer>18</answer>", "18")
    assert result.is_correct and result.predicted == "18"
    missing = NumericVerifier().verify("no tags at all", "18")
    assert not missing.is_correct and "no <answer>" in missing.detail


def test_exact_match_ignores_case_and_spacing():
    v = ExactMatchVerifier()
    assert v.is_correct("  Madrid ", "madrid")
    assert not v.is_correct("Barcelona", "Madrid")
