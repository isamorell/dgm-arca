"""Verifiable problem generator for medication-administration calculations (nursing).

Same pattern as ``generate_problems.py``: ``sample_params`` -> ``solve`` -> ``render``.
``solve`` is the reference implementation *and* the verifier: the ground truth cannot be wrong.
All arithmetic is done with ``fractions.Fraction`` so results are exact and rounding is
deterministic (half up).

How a problem is built
----------------------
1. ``_sample_family`` draws the numbers of the problem (weight, dose, volume, time...) and
                      rejects draws whose answer is not plausible (see ``BOUNDS``). ``generate``
                      gives every family the same number of problems (balanced by quotas).
2. ``solve``          computes the exact answer and records which branches were taken.
3. ``render``         writes the statement in Spanish, choosing one of five styles.

Families (level = number of reasoning steps):

    pump              1    volume + time                  -> mL/h
    dilution          1-2  dose + target conc.            -> final volume (mL) or diluent (mL)
    drops             2    volume + time + drop factor    -> drops/min
    dose_day          3    weight + dose/kg/day + n       -> mg per administration
    volume            3    weight + dose/kg + vial        -> mL to administer
    infusion          4    weight + mcg/kg/min + bag      -> mL/h
    infusion_inverse  4    mL/h + bag + weight            -> mcg/kg/min

Run::

    uv run python -m rlm.generate_problems --n 800 --split train --out rlm/data/train.jsonl
    uv run python -m rlm.generate_problems --n 200 --split test  --out rlm/data/test.jsonl \
        --avoid rlm/data/train.jsonl
    uv run python -m rlm.generate_problems --n 100 --split ood   --out rlm/data/test_ood.jsonl

OOD modes (``--ood-mode``):
    family  train/test never contain continuous infusion (mcg/kg/min); ood contains only that.
    weight  train/test use weights >= 10 kg; ood uses 2-9.9 kg (neonatal/paediatric).

Answers are stored with a dot as decimal separator, while statements use a comma or a dot at
random (Spanish convention). Models may therefore answer "243,75". ``rlm.rewards.normalize_number``
must read a decimal comma correctly, otherwise "243,75" is parsed as 24375. Verify with
``NumericVerifier(tolerance=row["tolerance"])``.

Reproducibility: the same code and the same seed (``DEFAULT_SEED`` unless ``--seed`` is given)
produce the same files, on any machine and independently of ``PYTHONHASHSEED``. A single random
generator is shared by ``sample_params`` and ``render``, so editing a template changes the
following problems even with the same seed.

NOTE: the doses are arbitrary to exercise the arithmetic; they are NOT clinical guidance.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import asdict, dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any


@dataclass
class Problem:
    question: str
    answer: str
    params: dict[str, Any]
    template_id: int
    branches: dict[str, str] = field(default_factory=dict)


class ProblemGenerator(ABC):
    """Subclass this for your domain. Three methods, and the base class does the rest."""

    name: str = "generator"

    @abstractmethod
    def sample_params(self, rng: random.Random, split: str) -> dict[str, Any]:
        """One problem's data. ``split`` lets you hold a region out for the OOD set."""

    @abstractmethod
    def solve(self, params: dict[str, Any]) -> tuple[str, dict[str, str]]:
        """Reference implementation. Returns the answer and which branches were taken."""

    @abstractmethod
    def render(self, params: dict[str, Any], rng: random.Random) -> tuple[str, int]:
        """The statement in natural language. Returns the text and the template used."""

    def key(self, params: dict[str, Any]) -> str:
        """Deduplication key. Hash the *parameters*, never the text."""
        return json.dumps(params, sort_keys=True, default=str)

    def generate(self, n: int, split: str, seed: int = 0) -> list[Problem]:
        rng = random.Random(f"{seed}-{split}")
        seen: set[str] = set()
        problems: list[Problem] = []
        attempts = 0
        while len(problems) < n and attempts < 200 * n:
            attempts += 1
            params = self.sample_params(rng, split)
            fingerprint = self.key(params)
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            answer, branches = self.solve(params)
            question, template_id = self.render(params, rng)
            problems.append(Problem(question, answer, params, template_id, branches))
        if len(problems) < n:
            raise RuntimeError(
                f"only {len(problems)} unique problems after {attempts} attempts: "
                "your parameter space is too small, widen sample_params"
            )
        return problems


# ============================================================================= configuration
# Seed used by default for every split. Each split derives its own stream from it
# (``random.Random(f"{seed}-{split}")``), so train/test/ood differ but are always the same
# for a given seed. Change it only on purpose, and write the new value in EXPERIMENTS.md.
DEFAULT_SEED = 0

# --- Units -------------------------------------------------------------------------------
# All mass arithmetic is done in mg; this table is where g / mcg are converted.
MASS_TO_MG = {"g": Fraction(1000), "mg": Fraction(1), "mcg": Fraction(1, 1000)}

# Unit of the answer and number of decimals asked for, per family.
ANSWER_UNIT = {
    "pump": "mL/h", "drops": "gotas/min", "dose_day": "mg", "volume": "mL",
    "infusion": "mL/h", "infusion_inverse": "mcg/kg/min", "dilution": "mL",
}
DECIMALS = {
    "pump": 1, "drops": 0, "dose_day": 1, "volume": 2,
    "infusion": 1, "infusion_inverse": 2, "dilution": 1,
}
# Other spellings a model may legitimately use for the requested unit (for a UnitVerifier).
UNIT_ALIASES = {
    "mL/h": ["ml/h"], "gotas/min": ["gotas por minuto", "gotas/minuto", "gtt/min"],
    "mg": ["miligramos"], "mL": ["ml", "cc"], "mcg/kg/min": ["µg/kg/min", "ug/kg/min"],
}

# --- Values the parameters are drawn from (round numbers, as in real prescriptions) --------
DOSE_VALUES = {
    "g": [0.01, 0.02, 0.05, 0.1, 0.15],
    "mg": [5, 10, 15, 20, 25, 40, 50, 100],
    "mcg": [50, 100, 250, 500, 1000],
}
DROP_FACTORS = [10, 15, 20, 60]  # drops per mL (macro: 10-20, micro: 60)
# Every 15 min from 30 min to 12 h, plus 24 h (wide on purpose: see the balance note below).
TIMES_MIN = list(range(30, 721, 15)) + [1440]
STANDARD_VOLUMES = [50, 100, 250, 500, 1000]  # bag sizes; the rest are drawn in steps of 10 mL
DILUTION_DOSES = {
    "g": [0.25, 0.5, 1, 1.5, 2],
    "mg": [100, 125, 250, 300, 400, 500, 600, 750, 800, 1000, 1200, 1500, 2000],
}
TARGET_CONCENTRATIONS = [0.5, 1, 2, 2.5, 4, 5, 10, 20]  # mg/mL
RECONSTITUTION_VOLUMES = [2, 3, 5, 10, 20]  # mL
FLUIDS = ["suero fisiológico", "suero glucosado al 5 %", "Ringer lactato", "solución salina"]
DRUGS = ["amoxicilina", "ceftriaxona", "vancomicina", "paracetamol", "furosemida",
         "heparina sódica", "metronidazol", "noradrenalina", "dopamina", None]  # None = no name

# Plausible range of the *answer*, to avoid absurd problems (0.001 mL, 9000 mL/h...).
BOUNDS = {
    "pump": (5, 500), "drops": (5, 200), "dose_day": (5, 5000), "volume": (0.1, 100),
    "infusion": (0.5, 200), "infusion_inverse": (0.01, 50), "dilution": (1, 500),
}
# Difficulty level = number of reasoning steps (a dilution asking for diluent adds one).
LEVEL = {"pump": 1, "dilution": 1, "drops": 2, "dose_day": 3, "volume": 3,
         "infusion": 4, "infusion_inverse": 4}

# Parameters that only change the *wording*: two problems that differ only in these are the
# same problem, so they are excluded from the deduplication key.
STYLE_KEYS = {"decimal_comma", "time_style", "drug", "fluid", "distractor", "ml_spelling",
              "mcg_spelling"}

# --- Wording -------------------------------------------------------------------------------
SENTENCE_OPENERS = ["", "", "", "Ejercicio de prácticas. ", "Duda de cálculo. ", "Caso clínico. "]
CLAUSE_OPENERS = ["En planta, ", "Según la prescripción médica, ", "Enfermera de guardia: "]
QUESTIONS = {
    "pump": ["¿A qué velocidad, en mL/h, hay que programar la bomba?",
             "¿A cuántos mL/h se programa la bomba?", "Calcula la velocidad de infusión en mL/h.",
             "¿Qué ritmo de bomba, en mL/h, corresponde?", "Indica la velocidad en mL/h."],
    "drops": ["¿Cuántas gotas por minuto hay que regular?", "Calcula el goteo en gotas/min.",
              "¿A qué ritmo, en gotas/min, se regula el sistema?",
              "Indica el goteo en gotas por minuto."],
    "dose_day": ["¿Cuántos mg debe recibir en cada administración?",
                 "Calcula los mg de cada toma.", "¿Cuántos miligramos corresponden a cada dosis?",
                 "Indica la dosis por administración en mg."],
    "volume": ["¿Qué volumen, en mL, hay que administrar?", "¿Cuántos mL se cargan en la jeringa?",
               "Calcula el volumen a administrar en mL.",
               "Indica cuántos mL corresponden a la dosis."],
    "infusion": ["¿A qué velocidad, en mL/h, hay que programar la bomba?",
                 "Calcula la velocidad de la bomba en mL/h haciendo las conversiones necesarias.",
                 "¿Cuántos mL/h hay que poner?", "Indica la velocidad de infusión en mL/h."],
    "infusion_inverse": ["¿Cuántos mcg/kg/min está recibiendo?", "Calcula la dosis en mcg/kg/min.",
                         "¿Qué dosis, en mcg/kg/min, corresponde a esa velocidad?",
                         "Comprueba la dosis e indícala en mcg/kg/min."],
    "dilution_final": ["¿Qué volumen final, en mL, se necesita?",
                       "¿Cuál es el volumen final de la solución en mL?",
                       "Calcula el volumen final en mL."],
    "dilution_diluent": ["¿Cuántos mL de diluyente hay que añadir?",
                         "¿Cuánto diluyente adicional, en mL, necesito?",
                         "Calcula los mL de diluyente a añadir."],
}


# ============================================================================= helpers
def Q(x: Any) -> Fraction:
    """Exact ``Fraction`` from a JSON number.

    Going through ``str`` avoids float noise: ``Q(0.05)`` is exactly 1/20, whereas
    ``Fraction(0.05)`` is the binary approximation of 0.05.
    """
    return Fraction(str(x))


def round_half_up(x: Fraction, decimals: int) -> Fraction:
    """Round to ``decimals`` places with half-up rounding (2.5 -> 3, unlike ``round``)."""
    scale = 10**decimals
    return Fraction(math.floor(x * scale + Fraction(1, 2)), scale)


def fmt(x: Any, comma: bool) -> str:
    """Format a number for a statement: no trailing zeros, decimal comma or dot."""
    text = f"{float(Q(x)):.6f}".rstrip("0").rstrip(".")
    return text.replace(".", ",") if comma else text


def render_time(minutes: int, style: str) -> str:
    """Write a duration as ``"4 h"`` (style ``h``), ``"1 h 30 min"`` (``hmin``) or ``"240 min"``."""
    if style == "h":
        return f"{minutes // 60} h"
    if style == "hmin":
        return f"{minutes // 60} h {minutes % 60} min"
    return f"{minutes} min"


def rounding_sentence(decimals: int, rng: random.Random) -> str:
    """Ask for the result rounded to ``decimals`` places, with a random wording."""
    variants = {
        0: ["Redondea al entero.", "Da el resultado como número entero.",
            "Expresa el resultado redondeado a un entero."],
        1: ["Redondea a un decimal.", "Da el resultado con un decimal.",
            "Expresa el resultado con un decimal (redondeado)."],
        2: ["Redondea a dos decimales.", "Da el resultado con dos decimales.",
            "Expresa el resultado con dos decimales (redondeado)."],
    }
    return rng.choice(variants[decimals])

def describe(problems: list[Problem]) -> dict[str, Any]:
    """Branch coverage and leakage check: the two numbers I will ask you about."""
    counters: dict[str, Counter] = {}
    for problem in problems:
        for branch, value in problem.branches.items():
            counters.setdefault(branch, Counter())[value] += 1
    leaked = sum(1 for p in problems if p.answer in p.question)
    return {
        "n_problems": len(problems),
        "n_templates": len({p.template_id for p in problems}),
        "branches": {k: dict(v) for k, v in counters.items()},
        "answer_leaked_in_statement": leaked,
    }


# ============================================================================= generator
class MedicationGenerator(ProblemGenerator):
    """Nursing medication calculations: infusion rate, drops, doses, dilutions."""

    name = "medication_calc"

    def __init__(self, ood_mode: str = "family", avoid: set[str] | None = None):
        """Create the generator.

        Args:
            ood_mode: how the out-of-distribution split differs from train/test (``family`` or
                ``weight``, see the module docstring).
            avoid: deduplication keys of problems that must not be generated again (used to keep
                test disjoint from train).
        """
        if ood_mode not in {"family", "weight"}:
            raise ValueError("ood_mode must be 'family' or 'weight'")
        self.ood_mode = ood_mode
        self.avoid = avoid or set()

    # ------------------------------------------------------------------ 1. sampling parameters
    def _families(self, split: str) -> tuple[list[str], bool]:
        """Families available in ``split`` and whether weights must be low (< 10 kg)."""
        everything = list(ANSWER_UNIT)
        if self.ood_mode == "family":
            continuous = ["infusion", "infusion_inverse"]
            if split == "ood":
                return continuous, False
            return [f for f in everything if f not in continuous], False
        # ood_mode == "weight": same families, but ood uses paediatric/neonatal weights.
        if split == "ood":
            return ["dose_day", "volume", "infusion", "infusion_inverse"], True
        return everything, False

    @staticmethod
    def _weight(rng: random.Random, low: bool) -> float:
        """Patient weight in kg: neonatal when ``low``, else paediatric or adult (with .5 kg)."""
        if low:
            return round(rng.uniform(2, 9.9), 1)  # 2-9.9 kg
        # One third paediatric (10-19 kg), two thirds adult (20-110 kg).
        base = rng.choice([rng.randint(10, 19), rng.randint(20, 110), rng.randint(20, 110)])
        return base + (0.5 if rng.random() < 0.3 else 0)

    @staticmethod
    def _volume(rng: random.Random) -> int:
        """Volume in mL: a standard bag 40 % of the time, otherwise any multiple of 10 mL."""
        if rng.random() < 0.4:
            return rng.choice(STANDARD_VOLUMES)
        return rng.randrange(50, 1001, 10)

    @staticmethod
    def _dose(rng: random.Random, units: list[str]) -> tuple[float, str]:
        """Draw a dose unit and a round value that is sensible for that unit."""
        unit = rng.choice(units)
        return rng.choice(DOSE_VALUES[unit]), unit

    def _draw_family_params(
        self, rng: random.Random, family: str, low: bool
    ) -> dict[str, Any]:
        """Draw the numeric data that defines a problem of ``family`` (one branch per family)."""
        p: dict[str, Any] = {"family": family}

        if family in {"pump", "drops"}:
            # Volume to infuse and time; drops also needs the drop factor of the set.
            p["volume_ml"] = self._volume(rng)
            p["time_min"] = rng.choice(TIMES_MIN)
            # How the time is written: "4 h", "240 min" or "1 h 30 min" (forces conversions).
            if p["time_min"] % 60 == 0:
                p["time_style"] = rng.choice(["h", "min"])
            elif p["time_min"] > 60:
                p["time_style"] = rng.choice(["min", "hmin"])
            else:
                p["time_style"] = "min"
            if family == "drops":
                p["drop_factor"] = rng.choice(DROP_FACTORS)

        elif family == "dose_day":
            # Daily dose per kg (in g, mg or mcg) split into n administrations -> mg per dose.
            p["weight_kg"] = self._weight(rng, low)
            p["dose_value"], p["dose_unit"] = self._dose(rng, ["g", "mg", "mcg"])
            p["n_doses"] = rng.choice([2, 3, 4, 6, 8])

        elif family == "volume":
            # Dose per kg and a vial written as "mass in volume" -> mL to draw up.
            p["weight_kg"] = self._weight(rng, low)
            p["dose_unit"] = rng.choice(["mg", "mg", "mcg"])
            p["dose_value"] = rng.choice(
                [1, 2, 2.5, 5, 7.5, 10, 15] if p["dose_unit"] == "mg" else [5, 10, 25, 50]
            )
            p["conc_mass_unit"] = rng.choice(["mg", "mg", "g"])
            p["conc_mass"] = rng.choice(
                [50, 100, 125, 250, 500, 1000] if p["conc_mass_unit"] == "mg" else [0.5, 1, 2]
            )
            p["conc_volume_ml"] = rng.choice([1, 2, 5, 10, 20])

        elif family in {"infusion", "infusion_inverse"}:
            # A bag with a mass of drug in a volume; either the dose (mcg/kg/min) or the pump
            # rate (mL/h) is given and the other one is asked.
            p["weight_kg"] = self._weight(rng, low)
            p["bag_mass_mg"] = rng.choice([1, 2, 4, 5, 8, 10, 20, 50, 100, 200])
            p["bag_volume_ml"] = rng.choice([50, 100, 250])
            if family == "infusion":
                p["dose_mcg_kg_min"] = rng.choice([0.05, 0.1, 0.2, 0.5, 1, 2, 5, 8, 10])
            else:
                p["rate_ml_h"] = rng.choice([2, 5, 10, 12, 15, 20, 25, 30, 40, 60])

        elif family == "dilution":
            # Dose to prepare at a target concentration; optionally the powder is reconstituted
            # with a known volume and the diluent still to add is asked.
            unit = rng.choice(["g", "mg"])
            p["dose_unit"] = unit
            p["dose_value"] = rng.choice(DILUTION_DOSES[unit])
            p["target_conc_mg_ml"] = rng.choice(TARGET_CONCENTRATIONS)
            p["ask"] = rng.choice(["final", "diluent"])
            if p["ask"] == "diluent":
                p["recon_volume_ml"] = rng.choice(RECONSTITUTION_VOLUMES)
        return p

    @staticmethod
    def _draw_style_params(rng: random.Random, family: str, p: dict[str, Any]) -> None:
        """Add the parameters that only affect the wording (they are in ``STYLE_KEYS``)."""
        p["decimals"] = DECIMALS[family]                      # not random, kept with the rest
        p["decimal_comma"] = rng.random() < 0.6               # Spanish comma most of the time
        p["drug"] = rng.choice(DRUGS)                         # decorative name (or None)
        p["fluid"] = rng.choice(FLUIDS)
        p["distractor"] = rng.random() < 0.35                 # add an irrelevant fact
        p["ml_spelling"] = rng.choice(["mL", "mL", "ml"])
        p["mcg_spelling"] = rng.choice(["mcg", "mcg", "mcg", "µg"])

    def _draw(self, rng: random.Random, family: str, low: bool) -> dict[str, Any]:
        """One complete random draw: problem data first, then wording parameters."""
        p = self._draw_family_params(rng, family, low)
        self._draw_style_params(rng, family, p)
        return p

    def _sample_family(self, rng: random.Random, split: str, family: str) -> dict[str, Any]:
        """Draw parameters for ``family`` until the answer falls inside ``BOUNDS``."""
        _, low = self._families(split)
        lo, hi = BOUNDS[family]
        for _ in range(500):
            params = self._draw(rng, family, low)
            if lo <= self._value(params) <= hi:
                return params
        raise RuntimeError(f"could not draw a plausible {family} problem")

    def sample_params(self, rng: random.Random, split: str) -> dict[str, Any]:
        """Draw one problem from a random family of ``split`` (kept for the base-class API).

        ``generate`` does not use this: it asks for each family explicitly to balance them.
        """
        families, _ = self._families(split)
        return self._sample_family(rng, split, rng.choice(families))

    # ------------------------------------------------------------------ 2. deduplication
    def key(self, params: dict[str, Any]) -> str:
        """Deduplication key: the parameters *without* the wording-only ones.

        Hashing parameters (not text) is what makes train/test leakage detectable: the same
        numbers phrased differently are still the same problem.
        """
        core = {k: v for k, v in params.items() if k not in STYLE_KEYS}
        return json.dumps(core, sort_keys=True, default=str)

    @staticmethod
    def _quotas(n: int, families: list[str]) -> dict[str, int]:
        """Split ``n`` problems as evenly as possible among ``families`` (remainder to the first)."""
        base, extra = divmod(n, len(families))
        return {f: base + (1 if i < extra else 0) for i, f in enumerate(families)}

    def generate(self, n: int, split: str, seed: int = DEFAULT_SEED) -> list[Problem]:
        """Generate ``n`` unique problems for ``split``, balanced across families.

        Every family gets the same number of problems (its *quota*). Drawing the family at
        random instead would leave small parameter spaces starved: duplicates are rejected, so
        the families with more combinations end up over-represented. ``seen`` starts with
        ``self.avoid`` so a split never repeats problems of another one, and the final shuffle
        avoids a file grouped by family. The random generator is seeded with
        ``"{seed}-{split}"``, so every split has its own reproducible stream.
        """
        rng = random.Random(f"{seed}-{split}")
        families, _ = self._families(split)
        quotas = self._quotas(n, families)
        seen: set[str] = set(self.avoid)
        problems: list[Problem] = []
        for family in families:
            made, attempts = 0, 0
            while made < quotas[family]:
                attempts += 1
                if attempts > 200 * quotas[family]:
                    raise RuntimeError(
                        f"only {made} unique '{family}' problems out of {quotas[family]} "
                        "requested: widen its parameter space or lower --n"
                    )
                params = self._sample_family(rng, split, family)
                fingerprint = self.key(params)
                if fingerprint in seen:
                    continue
                seen.add(fingerprint)
                answer, branches = self.solve(params)
                question, template_id = self.render(params, rng)
                problems.append(Problem(question, answer, params, template_id, branches))
                made += 1
        rng.shuffle(problems)
        return problems

    # ------------------------------------------------------------------ 3. reference solver
    @staticmethod
    def _value(p: dict[str, Any]) -> Fraction:
        """Exact (unrounded) answer, in the unit given by ``ANSWER_UNIT[family]``."""
        f = p["family"]
        if f == "pump":
            # mL/h = volume / time, with the time converted from minutes to hours.
            return Q(p["volume_ml"]) * 60 / p["time_min"]
        if f == "drops":
            # drops/min = (volume in mL x drops per mL) / minutes.
            return Q(p["volume_ml"]) * p["drop_factor"] / p["time_min"]
        if f == "dose_day":
            # mg per administration = weight x daily dose per kg (in mg) / number of doses.
            dose_mg = Q(p["dose_value"]) * MASS_TO_MG[p["dose_unit"]]
            return Q(p["weight_kg"]) * dose_mg / p["n_doses"]
        if f == "volume":
            # mL = total dose in mg / concentration in mg/mL.
            dose_mg = Q(p["weight_kg"]) * Q(p["dose_value"]) * MASS_TO_MG[p["dose_unit"]]
            conc = Q(p["conc_mass"]) * MASS_TO_MG[p["conc_mass_unit"]] / p["conc_volume_ml"]
            return dose_mg / conc
        if f == "infusion":
            # mcg/kg/min -> mg/h (x60 min/h, /1000 mcg/mg), then divide by the bag's mg/mL.
            mg_per_h = Q(p["weight_kg"]) * Q(p["dose_mcg_kg_min"]) * 60 / 1000
            return mg_per_h / (Fraction(p["bag_mass_mg"]) / p["bag_volume_ml"])
        if f == "infusion_inverse":
            # mL/h x mg/mL = mg/h, then back to mcg/kg/min (x1000 mcg/mg, /60 min/h, /kg).
            mg_per_ml = Fraction(p["bag_mass_mg"]) / p["bag_volume_ml"]
            return Q(p["rate_ml_h"]) * mg_per_ml * 1000 / (60 * Q(p["weight_kg"]))
        if f == "dilution":
            # Final volume = dose in mg / target concentration; the diluent to add is what is
            # left after the reconstitution volume (powder displacement is ignored).
            final = Q(p["dose_value"]) * MASS_TO_MG[p["dose_unit"]] / Q(p["target_conc_mg_ml"])
            return final - p["recon_volume_ml"] if p["ask"] == "diluent" else final
        raise ValueError(f)

    def solve(self, params: dict[str, Any]) -> tuple[str, dict[str, str]]:
        """Return the rounded answer as text (dot decimal) and the branches taken.

        The branches are strings so they can be counted for the coverage table in
        ``EXPERIMENTS.md`` (``describe`` from the base class does exactly that).
        """
        family = params["family"]
        value = self._value(params)
        if value <= 0:
            raise ValueError("non-positive answer")
        decimals = params["decimals"]
        answer = f"{float(round_half_up(value, decimals)):.{decimals}f}"
        unit = params.get("dose_unit")
        branches = {
            "family": family,
            "level": str(LEVEL[family] + (1 if params.get("ask") == "diluent" else 0)),
            # Did the solver need a g/mcg -> mg conversion?
            "mass_unit_conversion": str(
                unit in {"g", "mcg"} or params.get("conc_mass_unit") == "g"
            ),
            "time_style": params.get("time_style", "-"),
            "drop_factor": str(params.get("drop_factor", "-")),
            "answer_unit": ANSWER_UNIT[family],
            "low_weight": str(params.get("weight_kg", 99) < 10),
            "decimal_comma": str(params["decimal_comma"]),
        }
        return answer, branches

    # ------------------------------------------------------------------ 4a. cohesive templates
    def _cohesive(self, params: dict[str, Any], rng: random.Random) -> tuple[str, int]:
        """Hand-written full templates, several per family. Returns (text, template index)."""
        c = params["decimal_comma"]

        def n(key: str) -> str:
            """Parameter ``key`` formatted with the statement's decimal separator."""
            return fmt(params[key], c)

        f = params["family"]
        rnd = rounding_sentence(params["decimals"], rng)
        drug = f" de {params['drug']}" if params["drug"] else ""      # "dosis de X"
        drug_obj = f" {params['drug']}" if params["drug"] else ""     # "han pautado X"
        fluid = params["fluid"]
        if f in {"pump", "drops"}:
            t = render_time(params["time_min"], params["time_style"])
            vol = n("volume_ml")

        if f == "pump":
            templates = [
                f"Hay que administrar {vol} mL de {fluid} en {t}. ¿A qué velocidad, en mL/h, "
                f"hay que programar la bomba? {rnd}",
                f"Tengo que pasar {vol} mL de {fluid} en {t}. ¿A cuántos mL/h tengo que poner "
                f"la bomba? {rnd}",
                f"Prescripción: {fluid}, {vol} mL a pasar en {t}. Calcula la velocidad de "
                f"infusión en mL/h. {rnd}",
                f"Una bolsa de {vol} mL de {fluid} debe infundirse en {t} con bomba "
                f"volumétrica. ¿Cuál es la velocidad en mL/h? {rnd}",
            ]
        elif f == "drops":
            d = params["drop_factor"]
            templates = [
                f"Se pautan {vol} mL de {fluid} en {t} con un equipo de {d} gotas/mL. ¿A cuántas "
                f"gotas por minuto hay que regular el goteo? {rnd}",
                f"Tengo que pasar {vol} mL de {fluid} en {t}. El equipo de suero es de {d} "
                f"gotas/mL. ¿Cuántas gotas/min? {rnd}",
                f"Equipo de gravedad ({d} gotas = 1 mL). Prescripción: {vol} mL de {fluid} en "
                f"{t}. Calcula el goteo en gotas/min. {rnd}",
                f"Con un sistema de {d} gotas/mL hay que infundir {vol} mL de {fluid} durante "
                f"{t}. ¿A qué ritmo, en gotas por minuto, se regula? {rnd}",
            ]
        elif f == "dose_day":
            w, dv, u, k = n("weight_kg"), n("dose_value"), params["dose_unit"], params["n_doses"]
            templates = [
                f"Un paciente pesa {w} kg y le han pautado{drug_obj} {dv} {u}/kg/día, repartida "
                f"en {k} administraciones. ¿Cuántos mg debe recibir en cada administración? {rnd}",
                f"Pauta{drug}: {dv} {u}/kg/día en {k} tomas iguales. Paciente de {w} kg. "
                f"¿Cuántos mg por toma? {rnd}",
                f"Paciente de {w} kg. Dosis diaria de {dv} {u} por kg, dividida en {k} dosis. "
                f"Calcula los mg de cada dosis. {rnd}",
            ]
        elif f == "volume":
            w, dv, u = n("weight_kg"), n("dose_value"), params["dose_unit"]
            cm, cu, cv = n("conc_mass"), params["conc_mass_unit"], n("conc_volume_ml")
            templates = [
                f"Un paciente de {w} kg necesita una dosis{drug} de {dv} {u}/kg. El vial "
                f"disponible es de {cm} {cu} en {cv} mL. ¿Qué volumen, en mL, hay que "
                f"administrar? {rnd}",
                f"Hay que administrar {dv} {u}/kg a un paciente de {w} kg. Presentación: "
                f"{cm} {cu}/{cv} mL. ¿Cuántos mL se cargan en la jeringa? {rnd}",
                f"Paciente: {w} kg. Prescripción{drug}: {dv} {u} por kg. Concentración del "
                f"vial: {cm} {cu} en {cv} mL. Calcula el volumen a administrar en mL. {rnd}",
            ]
        elif f == "infusion":
            w, dv = n("weight_kg"), n("dose_mcg_kg_min")
            m, v = n("bag_mass_mg"), n("bag_volume_ml")
            templates = [
                f"A un paciente de {w} kg le han pautado{drug_obj} {dv} mcg/kg/min. La solución "
                f"preparada contiene {m} mg en {v} mL. ¿A qué velocidad, en mL/h, hay que "
                f"programar la bomba? {rnd}",
                f"Perfusión continua{drug}: {dv} mcg/kg/min, paciente de {w} kg. Bolsa de "
                f"{m} mg en {v} mL. Calcula la velocidad de la bomba en mL/h. {rnd}",
                f"Tengo que pasar {dv} mcg/kg/min a un paciente de {w} kg con una solución "
                f"de {m} mg diluidos en {v} mL. ¿Cuántos mL/h pongo? Haz las conversiones "
                f"necesarias. {rnd}",
            ]
        elif f == "infusion_inverse":
            w, r = n("weight_kg"), n("rate_ml_h")
            m, v = n("bag_mass_mg"), n("bag_volume_ml")
            templates = [
                f"La bomba de un paciente de {w} kg va a {r} mL/h con una solución de {m} mg en "
                f"{v} mL. ¿Cuántos mcg/kg/min está recibiendo? {rnd}",
                f"Paciente de {w} kg con perfusión a {r} mL/h. La bolsa contiene {m} mg en "
                f"{v} mL. Calcula la dosis en mcg/kg/min. {rnd}",
                f"Compruebo una perfusión: {r} mL/h, bolsa de {m} mg en {v} mL, paciente de "
                f"{w} kg. ¿Qué dosis en mcg/kg/min recibe? {rnd}",
            ]
        else:  # dilution
            dv, u, tc = n("dose_value"), params["dose_unit"], n("target_conc_mg_ml")
            if params["ask"] == "final":
                templates = [
                    f"Hay que administrar {dv} {u}{drug} preparados en una solución con una "
                    f"concentración final de {tc} mg/mL. ¿Qué volumen final, en mL, se "
                    f"necesita? {rnd}",
                    f"Preparo {dv} {u}{drug} a {tc} mg/mL. ¿Cuál es el volumen final de la "
                    f"solución en mL? {rnd}",
                    f"Dosis prescrita: {dv} {u}. Concentración objetivo de la solución: {tc} "
                    f"mg/mL. Calcula el volumen final en mL. {rnd}",
                ]
            else:
                rv = n("recon_volume_ml")
                templates = [
                    f"Hay que administrar {dv} {u}{drug} en una solución final de {tc} mg/mL. "
                    f"El polvo se reconstituye con {rv} mL y ocupa un volumen despreciable. "
                    f"¿Cuántos mL de diluyente hay que añadir después para llegar al volumen "
                    f"final? {rnd}",
                    f"Preparo {dv} {u}{drug} a {tc} mg/mL. Ya he reconstituido el vial con "
                    f"{rv} mL (el desplazamiento del polvo es despreciable). ¿Cuánto diluyente "
                    f"adicional, en mL, necesito? {rnd}",
                    f"Dosis: {dv} {u}. Concentración final: {tc} mg/mL. Del volumen final ya "
                    f"tengo {rv} mL de reconstitución (sin desplazamiento). Calcula los mL de "
                    f"diluyente a añadir. {rnd}",
                ]
        template_id = rng.randrange(len(templates))
        return templates[template_id], template_id

    # ------------------------------------------------------------------ 4b. compositional engine
    def _facts(self, p: dict[str, Any]) -> tuple[list[tuple[list[str], str]], list[str]]:
        """Break a problem into facts that can be reordered and rephrased.

        Returns:
            ``(facts, questions)``. Each fact is ``(sentence_variants, label)``: the variants
            are full sentences (without final period) for the narrative style, and ``label``
            is the short "Peso: 58 kg" form for the telegraphic/list/data styles. ``questions``
            are the alternative ways to ask for this family's answer.
        """
        c = p["decimal_comma"]

        def n(key: str) -> str:
            """Parameter ``key`` formatted with the statement's decimal separator."""
            return fmt(p[key], c)

        f, drug = p["family"], p["drug"]
        of_drug = f" de {drug}" if drug else ""
        drug_obj = f" {drug}" if drug else ""
        facts: list[tuple[list[str], str]] = []

        def weight() -> None:
            """Add the patient-weight fact ('un niño' up to 30 kg)."""
            w = n("weight_kg")
            who = "un niño" if p["weight_kg"] <= 30 else "un paciente"
            facts.append(([f"El paciente pesa {w} kg", f"Se trata de {who} de {w} kg",
                           f"Paciente de {w} kg"], f"Peso: {w} kg"))

        def bag() -> None:
            """Add the fact describing the drug bag (mass in volume)."""
            m, v = n("bag_mass_mg"), n("bag_volume_ml")
            facts.append(([f"La solución preparada contiene {m} mg en {v} mL",
                           f"La bolsa lleva {m} mg diluidos en {v} mL",
                           f"Se dispone de una solución de {m} mg en {v} mL"],
                          f"Solución: {m} mg en {v} mL"))

        if f in {"pump", "drops"}:
            t = render_time(p["time_min"], p["time_style"])
            vol, fluid = n("volume_ml"), p["fluid"]
            facts.append(([f"Hay que pasar {vol} mL de {fluid}", f"El volumen prescrito es de "
                           f"{vol} mL de {fluid}", f"La bolsa es de {vol} mL de {fluid}"],
                          f"Volumen: {vol} mL ({fluid})"))
            facts.append(([f"Debe pasar en {t}", f"El tiempo de infusión es de {t}",
                           f"La duración prescrita es de {t}"], f"Tiempo: {t}"))
            if f == "drops":
                d = p["drop_factor"]
                facts.append(([f"El equipo es de {d} gotas/mL",
                               f"El sistema de gravedad tiene un factor de goteo de {d} gotas/mL",
                               f"Se usa un equipo en el que {d} gotas equivalen a 1 mL"],
                              f"Factor de goteo: {d} gotas/mL"))
        elif f == "dose_day":
            weight()
            dv, u = n("dose_value"), p["dose_unit"]
            facts.append(([f"La pauta{of_drug} es de {dv} {u}/kg/día",
                           f"Se pautan{drug_obj} {dv} {u} por kg y día"],
                          f"Pauta{of_drug}: {dv} {u}/kg/día"))
            k = p["n_doses"]
            facts.append(([f"La dosis diaria se reparte en {k} administraciones iguales",
                           f"Se administra en {k} tomas al día, todas iguales"],
                          f"Nº de administraciones: {k}"))
        elif f == "volume":
            weight()
            dv, u = n("dose_value"), p["dose_unit"]
            facts.append(([f"La dosis{of_drug} prescrita es de {dv} {u}/kg",
                           f"Se han pautado {dv} {u} por kg{of_drug}"],
                          f"Dosis{of_drug}: {dv} {u}/kg"))
            cm, cu, cv = n("conc_mass"), p["conc_mass_unit"], n("conc_volume_ml")
            facts.append(([f"El vial disponible es de {cm} {cu} en {cv} mL",
                           f"La presentación es de {cm} {cu} por cada {cv} mL",
                           f"La concentración del vial es de {cm} {cu}/{cv} mL"],
                          f"Vial: {cm} {cu}/{cv} mL"))
        elif f == "infusion":
            weight()
            dv = n("dose_mcg_kg_min")
            facts.append(([f"La pauta{of_drug} es de {dv} mcg/kg/min",
                           f"Se pautan{drug_obj} {dv} mcg/kg/min en perfusión continua"],
                          f"Dosis{of_drug}: {dv} mcg/kg/min"))
            bag()
        elif f == "infusion_inverse":
            weight()
            r = n("rate_ml_h")
            facts.append(([f"La bomba está programada a {r} mL/h",
                           f"La perfusión{of_drug} va a {r} mL/h"],
                          f"Velocidad de la bomba: {r} mL/h"))
            bag()
        else:  # dilution
            dv, u, tc = n("dose_value"), p["dose_unit"], n("target_conc_mg_ml")
            facts.append(([f"Hay que administrar {dv} {u}{of_drug}",
                           f"La dosis prescrita{of_drug} es de {dv} {u}"],
                          f"Dosis{of_drug}: {dv} {u}"))
            facts.append(([f"La concentración final de la solución debe ser de {tc} mg/mL",
                           f"Se prepara a {tc} mg/mL"], f"Concentración final: {tc} mg/mL"))
            if p["ask"] == "diluent":
                rv = n("recon_volume_ml")
                facts.append(([f"El vial ya está reconstituido con {rv} mL (el desplazamiento del "
                               f"polvo es despreciable)", f"Se reconstituye con {rv} mL y el polvo "
                               f"no desplaza volumen"],
                              f"Reconstitución: {rv} mL (sin desplazamiento)"))
        key = f if f != "dilution" else f"dilution_{p['ask']}"
        return facts, QUESTIONS[key]

    @staticmethod
    def _distractor(rng: random.Random) -> tuple[list[str], str]:
        """An irrelevant fact (bed, start time, cannula gauge...) the model must ignore.

        All numbers are drawn once so the sentence and its label agree.
        """
        bed, room = rng.randint(1, 30), rng.randint(101, 420)
        hour, minute = rng.randint(7, 22), rng.choice(["00", "15", "30", "45"])
        gauge, days = rng.choice([18, 20, 22, 24]), rng.randint(1, 12)
        return rng.choice([
            ([f"El paciente está en la cama {bed}"], f"Cama: {bed}"),
            ([f"La perfusión empieza a las {hour}:{minute} h"],
             f"Hora de inicio: {hour}:{minute}"),
            ([f"Se ha canalizado una vía periférica del calibre {gauge}G"], f"Vía: {gauge}G"),
            ([f"Lleva ingresado {days} días"], f"Días de ingreso: {days}"),
            ([f"La habitación es la {room}"], f"Habitación: {room}"),
        ])

    def render(self, params: dict[str, Any], rng: random.Random) -> tuple[str, int]:
        """Write the statement in one of five styles and return ``(text, template_id)``.

        Style weights are 3:3:2:1:1. ``template_id`` < 100 is a cohesive template; 100 to 103
        are the compositional styles: 100 narrative, 101 telegraphic, 102 list, 103 data line.
        """
        style = rng.choices(["cohesive", "narrative", "telegraphic", "list", "datos"],
                            weights=[3, 3, 2, 1, 1])[0]
        if style == "cohesive":
            text, tid = self._cohesive(params, rng)
        else:
            facts, questions = self._facts(params)
            if params["distractor"]:
                variants, label = self._distractor(rng)
                facts.append((variants, label))
            rng.shuffle(facts)  # the order of the data is random; the question goes last
            question = rng.choice(questions) + " " + rounding_sentence(params["decimals"], rng)
            # Clause openers ("En planta, ...") only fit the narrative style.
            opener = rng.choice(
                SENTENCE_OPENERS + (CLAUSE_OPENERS if style == "narrative" else [])
            )
            if style == "narrative":
                body = ". ".join(rng.choice(v) for v, _ in facts) + "."
                if opener.endswith(", "):
                    body = body[0].lower() + body[1:]
                text, tid = f"{opener}{body} {question}", 100
            elif style == "telegraphic":
                body = ". ".join(label for _, label in facts) + "."
                text, tid = f"{opener}{body} {question}", 101
            elif style == "list":
                body = "\n".join(f"- {label}" for _, label in facts)
                text, tid = f"{opener.strip()}\n{body}\n{question}".lstrip("\n"), 102
            else:
                body = "Datos: " + "; ".join(label for _, label in facts) + "."
                text, tid = f"{opener}{body} {question}", 103
        # Unit spelling variants, decided in the parameters so they are reproducible.
        if params["ml_spelling"] != "mL":
            text = text.replace("mL", params["ml_spelling"])
        if params["mcg_spelling"] != "mcg":
            text = text.replace("mcg", params["mcg_spelling"])
        return text, tid


# ============================================================================= command line
def load_avoid_keys(generator: MedicationGenerator, paths: list[str]) -> None:
    """Read the ``params_json`` of existing JSONL files and mark those problems as taken."""
    for path in paths:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            generator.avoid.add(generator.key(json.loads(json.loads(line)["params_json"])))


def write_jsonl(problems: list[Problem], out: Path, split: str) -> None:
    """Write one JSON line per problem, with the extra fields the rewards and verifier use.

    ``params`` and ``branches`` are stored as JSON *strings*: families have different keys and
    ``datasets.load_dataset`` would otherwise struggle with the heterogeneous schema.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for problem in problems:
            row = asdict(problem)
            family = row["params"]["family"]
            decimals = row["params"]["decimals"]
            row["params_json"] = json.dumps(row.pop("params"), ensure_ascii=False)
            row["branches"] = json.dumps(row["branches"], ensure_ascii=False)
            row["answer_unit"] = ANSWER_UNIT[family]
            row["family"] = family
            row["answer_unit_aliases"] = UNIT_ALIASES.get(ANSWER_UNIT[family], [])
            row["tolerance"] = 0.5 * 10 ** (-decimals)  # half a unit of the last decimal
            row["split"] = split
            row["label_source"] = "generator"
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    """Parse arguments, generate the problems, write the JSONL and print a summary."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--n", type=int, default=800)
    parser.add_argument("--split", choices=["train", "test", "ood"], default="train")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="base seed; each split derives its own stream from it")
    parser.add_argument("--ood-mode", choices=["family", "weight"], default="family")
    parser.add_argument("--out", default="rlm/data/train.jsonl")
    parser.add_argument(
        "--avoid", nargs="*", default=[], help="JSONL files whose problems must not repeat"
    )
    args = parser.parse_args()

    generator = MedicationGenerator(ood_mode=args.ood_mode)
    load_avoid_keys(generator, args.avoid)
    problems = generator.generate(args.n, args.split, args.seed)
    out = Path(args.out)
    write_jsonl(problems, out, args.split)

    # Skeleton = statement with the numbers masked: how many distinct sentence shapes we have.
    skeletons = {re.sub(r"\d+([.,]\d+)?", "#", p.question) for p in problems}
    print(f"esqueletos distintos (números enmascarados): {len(skeletons)} de {len(problems)}")
    print(json.dumps(describe(problems), indent=2, ensure_ascii=False))
    print(f"\n{len(problems)} problemas -> {out}")
    first = problems[0]
    unit = ANSWER_UNIT[first.params["family"]]
    print("Ejemplo:\n" + first.question + f"\nRespuesta: {first.answer} {unit}")


if __name__ == "__main__":
    main()
