"""Tests for unit handling in rlm/rewards.py and MedicationVerifier.

The OOD split asks for ``mcg/kg/min`` (continuous infusion). Until this was supported,
``normalize_unit("mcg/kg/min")`` was ``None`` and every answer in that unit was rejected,
right or wrong. The last block pins down that the units of the original families
(``mL``, ``mL/h``, ``mg``, ``gotas/min``) behave exactly as before.
"""

from __future__ import annotations

import pytest

from rlm.rewards import extract_final_unit, normalize_unit, units_equivalent
from rlm.verifier import MedicationVerifier

MICRO_SIGN = "µ"  # what the dataset aliases use
GREEK_MU = "μ"
ALIASES = [f"{MICRO_SIGN}g/kg/min", "ug/kg/min"]  # as stored in rlm/data/test_ood.jsonl


@pytest.mark.parametrize(
    "answer",
    [
        "0.31 mcg/kg/min",
        "0,31 mcg/kg/min",
        f"0.31 {MICRO_SIGN}g/kg/min",
        f"0.31 {GREEK_MU}g/kg/min",
        "0.31 ug/kg/min",
        "0.31 mcg / kg / min",
        "0.31 MCG/KG/MIN",
        "0.31 mcg/kg/minuto",
    ],
)
def test_composite_infusion_unit_is_recognised(answer):
    assert extract_final_unit(answer) == "mcg/kg/min"
    assert units_equivalent(answer, "mcg/kg/min", ALIASES)


@pytest.mark.parametrize(
    "answer",
    [
        "0.44 mcg/min",  # per-kg is missing
        f"0.44 {MICRO_SIGN}g/min",
        "0.44 mcg",
        "0.44 mg/kg/min",
        "0.44 mL/h",
        "0.44",
    ],
)
def test_other_units_are_still_wrong_for_mcg_per_kg_per_min(answer):
    assert not units_equivalent(answer, "mcg/kg/min", ALIASES)


def test_normalize_unit_unifies_micro_spellings():
    assert normalize_unit(f"{MICRO_SIGN}g") == "mcg"
    assert normalize_unit(f"{GREEK_MU}g") == "mcg"
    assert normalize_unit(f"{MICRO_SIGN}g/kg/min") == "mcg/kg/min"
    assert normalize_unit(" mcg / kg / min ") == "mcg/kg/min"
    assert normalize_unit("unknown/unit") is None


def test_the_last_number_decides_the_unit():
    answer = "5 mg x 2 = 10 mL/h, es decir 0.31 mcg/kg/min"
    assert extract_final_unit(answer) == "mcg/kg/min"


def test_medication_verifier_accepts_value_and_composite_unit():
    verifier = MedicationVerifier()
    good = "<think>x</think><answer>0.31 mcg/kg/min</answer>"
    assert verifier.verify(good, "0.31", "mcg/kg/min", ALIASES).is_correct
    alias = f"<think>x</think><answer>0,31 {MICRO_SIGN}g/kg/min</answer>"
    assert verifier.verify(alias, "0.31", "mcg/kg/min", ALIASES).is_correct


def test_medication_verifier_still_rejects_wrong_unit_or_value():
    verifier = MedicationVerifier()
    no_kg = "<think>x</think><answer>0.31 mcg/min</answer>"
    result = verifier.verify(no_kg, "0.31", "mcg/kg/min", ALIASES)
    assert not result.is_correct and "unit" in result.detail
    wrong_value = "<think>x</think><answer>0.40 mcg/kg/min</answer>"
    assert not verifier.verify(wrong_value, "0.31", "mcg/kg/min", ALIASES).is_correct


# ---- the units of the original families must not change ---------------------------------
@pytest.mark.parametrize(
    ("answer", "unit", "aliases"),
    [
        ("95 mL", "mL", ["ml", "cc"]),
        ("95 ml", "mL", ["ml", "cc"]),
        ("95 cc", "mL", ["ml", "cc"]),  # regression: "cc" was never extracted as a unit
        ("16,7 mL/h", "mL/h", ["ml/h"]),
        ("6.9 mg", "mg", ["miligramos"]),
        ("21 gotas/min", "gotas/min", ["gotas por minuto", "gotas/minuto", "gtt/min"]),
        ("21 gotas por minuto", "gotas/min", ["gotas por minuto", "gotas/minuto", "gtt/min"]),
    ],
)
def test_original_units_are_still_accepted(answer, unit, aliases):
    assert units_equivalent(answer, unit, aliases)


@pytest.mark.parametrize(
    ("answer", "unit", "aliases"),
    [
        ("95 mg", "mL", ["ml", "cc"]),
        ("95 cc/h", "mL", ["ml", "cc"]),  # a rate is not a volume
        ("16.7 mL", "mL/h", ["ml/h"]),
        ("6.9 g", "mg", ["miligramos"]),
        ("21 mL/h", "gotas/min", ["gotas por minuto"]),
    ],
)
def test_original_units_are_still_rejected_when_wrong(answer, unit, aliases):
    assert not units_equivalent(answer, unit, aliases)
