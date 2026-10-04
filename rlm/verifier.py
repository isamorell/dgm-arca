"""Verifier interface for reinforcement learning with verifiable rewards.

A verifier answers one question deterministically: *is this answer correct for
this problem?* Everything in phase 1 hangs on it. If the verifier is sloppy, the
model will learn to exploit the sloppiness instead of learning to reason.

We ship two verifiers:

* ``NumericVerifier`` compares numbers after normalisation. It is what GSM8K needs
  and what the smoke test uses.
* ``ExactMatchVerifier`` compares normalised strings. Useful for multiple-choice
  or short factual answers.

Your domain verifier goes in this module too. Subclass ``Verifier``, implement
``is_correct`` and add a test for it in ``tests/test_verifier.py``. Common shapes:
run unit tests on generated code, execute a SQL query and compare result sets,
validate a JSON document against a schema, check that a date falls in a range.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

from rlm.rewards import extract_answer, normalize_number


@dataclass(frozen=True)
class VerificationResult:
    """What a verifier reports back. ``detail`` is free text for logging and debugging."""

    is_correct: bool
    predicted: str | None
    expected: str
    detail: str = ""


class Verifier(ABC):
    """Base class for all verifiers."""

    name: str = "verifier"

    @abstractmethod
    def is_correct(self, predicted: str | None, expected: str) -> bool:
        """Return True when ``predicted`` should be accepted as a correct answer."""

    def verify(self, completion: str, expected: str) -> VerificationResult:
        """Extract the final answer from a full completion and check it."""
        predicted = extract_answer(completion)
        ok = self.is_correct(predicted, expected)
        detail = "no <answer> block found" if predicted is None else ""
        return VerificationResult(ok, predicted, expected, detail)


class NumericVerifier(Verifier):
    """Numeric comparison with an optional absolute tolerance."""

    name = "numeric"

    def __init__(self, tolerance: float = 0.0):
        self.tolerance = tolerance

    def is_correct(self, predicted: str | None, expected: str) -> bool:
        if predicted is None:
            return False
        p, e = normalize_number(predicted), normalize_number(expected)
        if p is None or e is None:
            return False
        if self.tolerance == 0.0:
            return p == e
        return abs(float(p) - float(e)) <= self.tolerance


class ExactMatchVerifier(Verifier):
    """Case- and whitespace-insensitive string comparison."""

    name = "exact_match"

    @staticmethod
    def _normalize(text: str) -> str:
        return re.sub(r"\s+", " ", text).strip().lower()

    def is_correct(self, predicted: str | None, expected: str) -> bool:
        if predicted is None:
            return False
        return self._normalize(predicted) == self._normalize(expected)

def _unit_text(text: str) -> str:
    """Lower-case, drop spaces and write micrograms as ``mcg`` so spellings compare equal."""
    text = text.lower().replace("µ", "u").replace(" ", "")
    return text.replace("ug", "mcg")
 
 
def has_unit(text: str, unit: str, aliases: tuple[str, ...] | list[str] = ()) -> bool:
    """True when ``unit`` (or one of its ``aliases``) appears in ``text`` as a whole unit.
 
    "Whole" means it is not glued to other letters or slashes, so ``mg`` does not match
    ``mg/kg`` and ``mL`` does not match ``mL/h``. Case, spaces and ``µg``/``ug``/``mcg`` are
    normalised: ``"0,25 ml"`` has the unit ``mL`` and ``"5 µg/kg/min"`` has ``mcg/kg/min``.
    """
    haystack = _unit_text(text)
    for candidate in (unit, *aliases):
        needle = re.escape(_unit_text(candidate))
        if re.search(rf"(?<![a-z/]){needle}(?![a-z/])", haystack):
            return True
    return False
 
 
class MedicationVerifier(NumericVerifier):
    """Nursing-calculation verifier: value with a derived tolerance, and optionally the unit.
 
    The generator rounds every answer (half up) to the decimals the statement asks for and
    stores it as text (``"131.3"``, ``"100.0"``, ``"5"``). Accepting anything within half a
    unit of the last decimal of ``expected`` means "rounds to the same value", so the tolerance
    follows from the expected string itself and no extra column is needed.
 
    ``verify`` takes the asked unit (dataset column ``answer_unit``, plus
    ``answer_unit_aliases``). When it is given, the ``<answer>`` block must contain it: a
    right number in the wrong unit (mg instead of mcg, mL instead of mL/h) counts as wrong.
    A right quantity written in *another* unit (``0.75 g`` for ``750 mg``) is also wrong,
    because the statement fixes the unit to answer in.
    """
 
    name = "medication"
 
    def is_correct(self, predicted: str | None, expected: str) -> bool:
        if predicted is None:
            return False
        p, e = normalize_number(predicted), normalize_number(expected)
        if p is None or e is None:
            return False
        decimals = len(expected.strip().split(".")[1]) if "." in expected else 0
        return abs(float(p) - float(e)) <= 0.5 * 10 ** (-decimals) + 1e-9
 
    def verify(
        self,
        completion: str,
        expected: str,
        unit: str | None = None,
        unit_aliases: tuple[str, ...] | list[str] = (),
    ) -> VerificationResult:
        """Check the value and, when ``unit`` is given, the unit of the ``<answer>`` block."""
        result = super().verify(completion, expected)
        if unit is None or not result.is_correct:
            return result
        if result.predicted is not None and has_unit(result.predicted, unit, unit_aliases):
            return result
        return VerificationResult(
            False, result.predicted, expected, f"right value but missing or wrong unit ({unit})"
        )