"""Verifiable problem generator for medication-administration calculations (nursing).

Same pattern as ``generate_problems.py``: ``sample_params`` -> ``solve`` -> ``render``.
``solve`` is the reference implementation *and* the verifier. All arithmetic is done with
``fractions.Fraction`` so the ground truth is exact and rounding is deterministic (half up).

Families (level = number of reasoning steps):

    pump              1  volume + time              -> mL/h
    dilution          1-2 dose + target conc.       -> final volume (mL) or diluent (mL)
    drops             2  volume + time + drop factor -> drops/min
    dose_day          3  weight + dose/kg/day + n    -> mg per administration
    volume            3  weight + dose/kg + vial     -> mL to administer
    infusion          4  weight + mcg/kg/min + bag   -> mL/h
    infusion_inverse  4  mL/h + bag + weight         -> mcg/kg/min

Run::

    uv run python -m rlm.generate_medication --n 800 --split train --out rlm/data/train.jsonl
    uv run python -m rlm.generate_medication --n 200 --split test  --out rlm/data/test.jsonl \
        --avoid rlm/data/train.jsonl
    uv run python -m rlm.generate_medication --n 100 --split ood   --out rlm/data/test_ood.jsonl

OOD modes (``--ood-mode``):
    family  train/test never contain continuous infusion (mcg/kg/min); ood contains only that.
    weight  train/test use weights >= 10 kg; ood uses 2-9.9 kg (neonatal/paediatric).

Answers are stored with a dot as decimal separator. Statements use comma or dot at random
(Spanish convention). Verify with ``NumericVerifier(tolerance=row["tolerance"])`` after making
``normalize_number`` understand the decimal comma (see the note in the chat / EXPERIMENTS.md).

NOTE: the doses are arbitrary to exercise the arithmetic; they are NOT clinical guidance.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import random
from dataclasses import asdict
from fractions import Fraction
from pathlib import Path
from typing import Any

from rlm.generate_problems import Problem, ProblemGenerator, describe

MASS_TO_MG = {"g": Fraction(1000), "mg": Fraction(1), "mcg": Fraction(1, 1000)}
DOSE_VALUES = {
    "g": [0.01, 0.02, 0.05, 0.1, 0.15],
    "mg": [5, 10, 15, 20, 25, 40, 50, 100],
    "mcg": [50, 100, 250, 500, 1000],
}
DROP_FACTORS = [10, 15, 20, 60]
TIMES_MIN = [30, 45, 60, 90, 120, 150, 180, 240, 300, 360, 480, 720, 1440]
FLUIDS = ["suero fisiológico", "suero glucosado al 5 %", "Ringer lactato", "solución salina"]
DRUGS = ["amoxicilina", "ceftriaxona", "vancomicina", "paracetamol", "furosemida",
         "heparina sódica", "metronidazol", "noradrenalina", "dopamina", None]
ANSWER_UNIT = {
    "pump": "mL/h", "drops": "gotas/min", "dose_day": "mg", "volume": "mL",
    "infusion": "mL/h", "infusion_inverse": "mcg/kg/min", "dilution": "mL",
}
DECIMALS = {
    "pump": 1, "drops": 0, "dose_day": 1, "volume": 2,
    "infusion": 1, "infusion_inverse": 2, "dilution": 1,
}
# Plausible range of the answer, to avoid absurd problems (0.001 mL, 9000 mL/h...).
BOUNDS = {
    "pump": (5, 500), "drops": (5, 200), "dose_day": (5, 5000), "volume": (0.1, 100),
    "infusion": (0.5, 200), "infusion_inverse": (0.01, 50), "dilution": (1, 500),
}
# Spellings a model may legitimately use for the requested unit: feed these to your UnitVerifier.
UNIT_ALIASES = {
    "mL/h": ["ml/h"], "gotas/min": ["gotas por minuto", "gotas/minuto", "gtt/min"],
    "mg": ["miligramos"], "mL": ["ml", "cc"], "mcg/kg/min": ["µg/kg/min", "ug/kg/min"],
}
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
               "Calcula el volumen a administrar en mL.", "Indica cuántos mL corresponden a la dosis."],
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
LEVEL = {"pump": 1, "dilution": 1, "drops": 2, "dose_day": 3, "volume": 3,
         "infusion": 4, "infusion_inverse": 4}
STYLE_KEYS = {"decimal_comma", "time_style", "drug", "fluid", "distractor", "ml_spelling",
              "mcg_spelling"}  # not part of the dedup key


def Q(x: Any) -> Fraction:
    """Exact Fraction from a JSON number (goes through str to avoid float noise)."""
    return Fraction(str(x))


def round_half_up(x: Fraction, decimals: int) -> Fraction:
    scale = 10**decimals
    return Fraction(math.floor(x * scale + Fraction(1, 2)), scale)


def fmt(x: Any, comma: bool) -> str:
    """Human-readable number, comma or dot decimal, no trailing zeros."""
    text = f"{float(Q(x)):.6f}".rstrip("0").rstrip(".")
    return text.replace(".", ",") if comma else text


def render_time(minutes: int, style: str) -> str:
    if style == "h":
        return f"{minutes // 60} h"
    if style == "hmin":
        return f"{minutes // 60} h {minutes % 60} min"
    return f"{minutes} min"


def rounding_sentence(decimals: int, rng: random.Random) -> str:
    variants = {
        0: ["Redondea al entero.", "Da el resultado como número entero.",
            "Expresa el resultado redondeado a un entero."],
        1: ["Redondea a un decimal.", "Da el resultado con un decimal.",
            "Expresa el resultado con un decimal (redondeado)."],
        2: ["Redondea a dos decimales.", "Da el resultado con dos decimales.",
            "Expresa el resultado con dos decimales (redondeado)."],
    }
    return rng.choice(variants[decimals])


class MedicationGenerator(ProblemGenerator):
    """Nursing medication calculations: infusion rate, drops, doses, dilutions."""

    name = "medication_calc"

    def __init__(self, ood_mode: str = "family", avoid: set[str] | None = None):
        if ood_mode not in {"family", "weight"}:
            raise ValueError("ood_mode must be 'family' or 'weight'")
        self.ood_mode = ood_mode
        self.avoid = avoid or set()

    # ------------------------------------------------------------------ sampling
    def _families(self, split: str) -> tuple[list[str], bool]:
        """Families available in this split and whether weights are 'low' (<10 kg)."""
        everything = list(ANSWER_UNIT)
        if self.ood_mode == "family":
            continuous = ["infusion", "infusion_inverse"]
            if split == "ood":
                return continuous, False
            return [f for f in everything if f not in continuous], False
        if split == "ood":
            return ["dose_day", "volume", "infusion", "infusion_inverse"], True
        return everything, False

    @staticmethod
    def _weight(rng: random.Random, low: bool) -> float:
        if low:
            return round(rng.uniform(2, 9.9), 1)
        base = rng.choice([rng.randint(10, 19), rng.randint(20, 110), rng.randint(20, 110)])
        return base + (0.5 if rng.random() < 0.3 else 0)

    @staticmethod
    def _dose(rng: random.Random, units: list[str]) -> tuple[float, str]:
        unit = rng.choice(units)
        return rng.choice(DOSE_VALUES[unit]), unit

    def _draw(self, rng: random.Random, family: str, low: bool) -> dict[str, Any]:
        p: dict[str, Any] = {"family": family}
        if family in {"pump", "drops"}:
            p["volume_ml"] = rng.choice([50, 100, 250, 500, 1000, rng.randrange(50, 1001, 10)])
            p["time_min"] = rng.choice(TIMES_MIN)
            if p["time_min"] % 60 == 0:
                p["time_style"] = rng.choice(["h", "min"])
            elif p["time_min"] > 60:
                p["time_style"] = rng.choice(["min", "hmin"])
            else:
                p["time_style"] = "min"
            if family == "drops":
                p["drop_factor"] = rng.choice(DROP_FACTORS)
        elif family == "dose_day":
            p["weight_kg"] = self._weight(rng, low)
            p["dose_value"], p["dose_unit"] = self._dose(rng, ["g", "mg", "mcg"])
            p["n_doses"] = rng.choice([2, 3, 4, 6, 8])
        elif family == "volume":
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
            p["weight_kg"] = self._weight(rng, low)
            p["bag_mass_mg"] = rng.choice([1, 2, 4, 5, 8, 10, 20, 50, 100, 200])
            p["bag_volume_ml"] = rng.choice([50, 100, 250])
            if family == "infusion":
                p["dose_mcg_kg_min"] = rng.choice([0.05, 0.1, 0.2, 0.5, 1, 2, 5, 8, 10])
            else:
                p["rate_ml_h"] = rng.choice([2, 5, 10, 12, 15, 20, 25, 30, 40, 60])
        elif family == "dilution":
            unit = rng.choice(["g", "mg"])
            p["dose_unit"] = unit
            p["dose_value"] = rng.choice([0.5, 1, 2] if unit == "g" else [250, 500, 750, 1000])
            p["target_conc_mg_ml"] = rng.choice([1, 2, 5, 10, 20])
            p["ask"] = rng.choice(["final", "diluent"])
            if p["ask"] == "diluent":
                p["recon_volume_ml"] = rng.choice([2, 5, 10, 20])
        p["decimals"] = DECIMALS[family]
        p["decimal_comma"] = rng.random() < 0.6
        p["drug"] = rng.choice(DRUGS)
        p["fluid"] = rng.choice(FLUIDS)
        p["distractor"] = rng.random() < 0.35
        p["ml_spelling"] = rng.choice(["mL", "mL", "ml"])
        p["mcg_spelling"] = rng.choice(["mcg", "mcg", "mcg", "µg"])
        return p

    def sample_params(self, rng: random.Random, split: str) -> dict[str, Any]:
        families, low = self._families(split)
        family = rng.choice(families)
        for _ in range(500):
            params = self._draw(rng, family, low)
            lo, hi = BOUNDS[family]
            if lo <= self._value(params) <= hi:
                return params
        raise RuntimeError(f"could not draw a plausible {family} problem")

    def key(self, params: dict[str, Any]) -> str:
        core = {k: v for k, v in params.items() if k not in STYLE_KEYS}
        return json.dumps(core, sort_keys=True, default=str)

    def generate(self, n: int, split: str, seed: int = 0) -> list[Problem]:
        # Same loop as the base class, but pre-seeded with keys to avoid (train/test leakage).
        rng = random.Random(f"{seed}-{split}")
        seen: set[str] = set(self.avoid)
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
            raise RuntimeError(f"only {len(problems)} unique problems: widen the parameter space")
        return problems

    # ------------------------------------------------------------------ reference solver
    @staticmethod
    def _value(p: dict[str, Any]) -> Fraction:
        """Exact (unrounded) answer in the unit of ``ANSWER_UNIT[family]``."""
        f = p["family"]
        if f == "pump":
            return Q(p["volume_ml"]) * 60 / p["time_min"]
        if f == "drops":
            return Q(p["volume_ml"]) * p["drop_factor"] / p["time_min"]
        if f == "dose_day":
            dose_mg = Q(p["dose_value"]) * MASS_TO_MG[p["dose_unit"]]
            return Q(p["weight_kg"]) * dose_mg / p["n_doses"]
        if f == "volume":
            dose_mg = Q(p["weight_kg"]) * Q(p["dose_value"]) * MASS_TO_MG[p["dose_unit"]]
            conc = Q(p["conc_mass"]) * MASS_TO_MG[p["conc_mass_unit"]] / p["conc_volume_ml"]
            return dose_mg / conc
        if f == "infusion":
            mg_per_h = Q(p["weight_kg"]) * Q(p["dose_mcg_kg_min"]) * 60 / 1000
            return mg_per_h / (Fraction(p["bag_mass_mg"]) / p["bag_volume_ml"])
        if f == "infusion_inverse":
            mg_per_ml = Fraction(p["bag_mass_mg"]) / p["bag_volume_ml"]
            return Q(p["rate_ml_h"]) * mg_per_ml * 1000 / (60 * Q(p["weight_kg"]))
        if f == "dilution":
            final = Q(p["dose_value"]) * MASS_TO_MG[p["dose_unit"]] / Q(p["target_conc_mg_ml"])
            return final - p["recon_volume_ml"] if p["ask"] == "diluent" else final
        raise ValueError(f)

    def solve(self, params: dict[str, Any]) -> tuple[str, dict[str, str]]:
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
            "mass_unit_conversion": str(unit in {"g", "mcg"} or params.get("conc_mass_unit") == "g"),
            "time_style": params.get("time_style", "-"),
            "drop_factor": str(params.get("drop_factor", "-")),
            "answer_unit": ANSWER_UNIT[family],
            "low_weight": str(params.get("weight_kg", 99) < 10),
            "decimal_comma": str(params["decimal_comma"]),
        }
        return answer, branches

    # ------------------------------------------------------------------ statements
    def _cohesive(self, params: dict[str, Any], rng: random.Random) -> tuple[str, int]:
        c = params["decimal_comma"]
        n = lambda key: fmt(params[key], c)  # noqa: E731
        f = params["family"]
        rnd = rounding_sentence(params["decimals"], rng)
        drug = f" de {params['drug']}" if params["drug"] else ""
        drug_obj = f" {params['drug']}" if params["drug"] else ""
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
                f"Un paciente pesa {w} kg y le han pautado{drug_obj} {dv} {u}/kg/día, repartida en "
                f"{k} administraciones. ¿Cuántos mg debe recibir en cada administración? {rnd}",
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

    # ------------------------------------------------------------------ compositional engine
    def _facts(self, p: dict[str, Any], rng: random.Random):
        """Facts as (sentence variants, label) plus the question variants for the family."""
        c = p["decimal_comma"]
        n = lambda key: fmt(p[key], c)  # noqa: E731
        f, drug = p["family"], p["drug"]
        of_drug = f" de {drug}" if drug else ""
        drug_obj = f" {drug}" if drug else ""
        facts: list[tuple[list[str], str]] = []

        def weight():
            w = n("weight_kg")
            who = "un niño" if p["weight_kg"] <= 30 else "un paciente"
            facts.append(([f"El paciente pesa {w} kg", f"Se trata de {who} de {w} kg",
                           f"Paciente de {w} kg"], f"Peso: {w} kg"))

        def bag():
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
                           f"La perfusión{of_drug} va a {r} mL/h"], f"Velocidad de la bomba: {r} mL/h"))
            bag()
        else:  # dilution
            dv, u, tc = n("dose_value"), p["dose_unit"], n("target_conc_mg_ml")
            facts.append(([f"Hay que administrar {dv} {u}{of_drug}", f"La dosis prescrita{of_drug} "
                           f"es de {dv} {u}"], f"Dosis{of_drug}: {dv} {u}"))
            facts.append(([f"La concentración final de la solución debe ser de {tc} mg/mL",
                           f"Se prepara a {tc} mg/mL"], f"Concentración final: {tc} mg/mL"))
            if p["ask"] == "diluent":
                rv = n("recon_volume_ml")
                facts.append(([f"El vial ya está reconstituido con {rv} mL (el desplazamiento del "
                               f"polvo es despreciable)", f"Se reconstituye con {rv} mL y el polvo "
                               f"no desplaza volumen"], f"Reconstitución: {rv} mL (sin desplazamiento)"))
        key = f if f != "dilution" else f"dilution_{p['ask']}"
        return facts, QUESTIONS[key]

    @staticmethod
    def _distractor(rng: random.Random) -> tuple[list[str], str]:
        bed, room = rng.randint(1, 30), rng.randint(101, 420)
        hour, minute = rng.randint(7, 22), rng.choice(["00", "15", "30", "45"])
        gauge, days = rng.choice([18, 20, 22, 24]), rng.randint(1, 12)
        return rng.choice([
            ([f"El paciente está en la cama {bed}"], f"Cama: {bed}"),
            ([f"La perfusión empieza a las {hour}:{minute} h"], f"Hora de inicio: {hour}:{minute}"),
            ([f"Se ha canalizado una vía periférica del calibre {gauge}G"], f"Vía: {gauge}G"),
            ([f"Lleva ingresado {days} días"], f"Días de ingreso: {days}"),
            ([f"La habitación es la {room}"], f"Habitación: {room}"),
        ])

    def render(self, params: dict[str, Any], rng: random.Random) -> tuple[str, int]:
        """Pick one of five styles; ``template_id`` < 100 is a cohesive template, >= 100 a
        compositional one (100 narrative, 101 telegraphic, 102 list, 103 data line)."""
        style = rng.choices(["cohesive", "narrative", "telegraphic", "list", "datos"],
                            weights=[3, 3, 2, 1, 1])[0]
        if style == "cohesive":
            text, tid = self._cohesive(params, rng)
        else:
            facts, questions = self._facts(params, rng)
            if params["distractor"]:
                variants, label = self._distractor(rng)
                facts.append((variants, label))
            rng.shuffle(facts)
            question = rng.choice(questions) + " " + rounding_sentence(params["decimals"], rng)
            opener = rng.choice(SENTENCE_OPENERS + (CLAUSE_OPENERS if style == "narrative" else []))
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
        if params["ml_spelling"] != "mL":
            text = text.replace("mL", params["ml_spelling"])
        if params["mcg_spelling"] != "mcg":
            text = text.replace("mcg", params["mcg_spelling"])
        return text, tid


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--n", type=int, default=800)
    parser.add_argument("--split", choices=["train", "test", "ood"], default="train")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ood-mode", choices=["family", "weight"], default="family")
    parser.add_argument("--out", default="rlm/data/train.jsonl")
    parser.add_argument("--avoid", nargs="*", default=[], help="JSONL files whose problems must not repeat")
    args = parser.parse_args()

    generator = MedicationGenerator(ood_mode=args.ood_mode)
    for path in args.avoid:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            generator.avoid.add(generator.key(json.loads(json.loads(line)["params_json"])))

    problems = generator.generate(args.n, args.split, args.seed)
    out = Path(args.out)
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
            row["tolerance"] = 0.5 * 10 ** (-decimals)
            row["split"] = args.split
            row["label_source"] = "generator"
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    skeletons = {re.sub(r"\d+([.,]\d+)?", "#", p.question) for p in problems}
    print(f"esqueletos distintos (números enmascarados): {len(skeletons)} de {len(problems)}")
    print(json.dumps(describe(problems), indent=2, ensure_ascii=False))
    print(f"\n{len(problems)} problemas -> {out}")
    print("Ejemplo:\n" + problems[0].question + f"\nRespuesta: {problems[0].answer} {ANSWER_UNIT[problems[0].params['family']]}")


if __name__ == "__main__":
    main()