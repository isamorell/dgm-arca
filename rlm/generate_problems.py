"""Build a verifiable problem set programmatically. This is the pattern, with a worked example.

Nobody annotates hundreds of reasoning problems by hand. When the task of your domain is a
calculation or a decision based on published rules, you write three functions and get
thousands of problems in seconds:

    sample_params(rng, split)  -> the data of one problem (weights, dates, amounts, doses…)
    solve(params)              -> the correct answer, from a reference implementation
    render(params, rng)        -> the statement in natural language, several templates

The key point, and the reason this is not cheating: ``solve`` is at the same time the
reference implementation and the **verifier**, so the ground truth cannot be wrong.

The example below (parcel shipping cost) is deliberately *not* one of the suggested project
domains: it is here so you can read a complete generator, run it, and then write your own.
Copy the structure, throw away the rules.

    uv run python -m rlm.generate_problems --n 800 --split train --out rlm/data/train.jsonl
    uv run python -m rlm.generate_problems --n 200 --split test  --out rlm/data/test.jsonl
    uv run python -m rlm.generate_problems --n 100 --split ood   --out rlm/data/test_ood.jsonl

``train`` and ``test`` share a distribution; ``ood`` samples a weight range that never appears
in training, which is what you need for the "SFT memorises, RL generalises" experiment.
Answers are euros with two decimals, so verify with ``NumericVerifier(tolerance=0.01)``.
"""

from __future__ import annotations

import argparse
import json
import random
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import asdict, dataclass, field
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


# --------------------------------------------------------------------------- worked example

WEIGHT_TIERS = ((2.0, 4.50, 0.00), (10.0, 4.50, 0.80), (30.0, 10.90, 0.55), (1e9, 21.90, 0.35))
DISTANCE_BANDS = ((100, 0.00), (500, 3.00), (10**9, 6.50))
TIER_DISCOUNT = {"none": 0.00, "silver": 0.05, "gold": 0.12}
REMOTE_SURCHARGE = 0.25
EXPRESS_MULTIPLIER = 1.6
INSURANCE_RATE = 0.012
INSURANCE_MIN = 1.50
FREE_BASE_THRESHOLD = 60.0

TIER_LABEL = {
    "none": "sin programa de fidelidad",
    "silver": "categoría plata",
    "gold": "categoría oro",
}


class ShippingCostGenerator(ProblemGenerator):
    """Parcel shipping cost: weight tiers, distance bands, surcharges, discount and insurance.

    Enough branching that the model cannot get it right with a single multiplication, and the
    rules are arbitrary but fixed, so it has to read the statement instead of recalling facts.
    """

    name = "shipping_cost"

    @staticmethod
    def weight_charge(weight: float) -> tuple[float, str]:
        """Base charge for the weight, and the name of the tier that applied."""
        previous = 0.0
        for index, (limit, fixed, per_kg) in enumerate(WEIGHT_TIERS):
            if weight <= limit:
                return fixed + per_kg * max(weight - previous, 0.0), f"tier_{index}"
            previous = limit
        raise ValueError(f"no tier for weight {weight}")

    @staticmethod
    def distance_charge(distance_km: int) -> float:
        """Flat fee for the distance band."""
        for limit, fee in DISTANCE_BANDS:
            if distance_km <= limit:
                return fee
        raise ValueError(f"no band for distance {distance_km}")

    def sample_params(self, rng: random.Random, split: str) -> dict[str, Any]:
        if split == "ood":
            # The heavy tier (>30 kg) never appears in train or test.
            weight = round(rng.uniform(30.1, 120.0), 1)
        else:
            weight = round(rng.uniform(0.1, 29.9), 1)
        distance = rng.choice([rng.randint(5, 99), rng.randint(100, 499), rng.randint(500, 1800)])
        return {
            "weight_kg": weight,
            "distance_km": distance,
            "remote": rng.random() < 0.3,
            "express": rng.random() < 0.35,
            "customer_tier": rng.choices(["none", "silver", "gold"], weights=[5, 3, 2])[0],
            "declared_value": rng.choice([0.0, 0.0, round(rng.uniform(50, 900), 2)]),
            "order_value": round(rng.uniform(10, 140), 2),
        }

    def solve(self, params: dict[str, Any]) -> tuple[str, dict[str, str]]:
        base, weight_tier = self.weight_charge(params["weight_kg"])
        distance_fee = self.distance_charge(params["distance_km"])

        free_base = params["order_value"] >= FREE_BASE_THRESHOLD and not params["express"]
        core = (0.0 if free_base else base) + distance_fee
        if params["remote"]:
            core *= 1 + REMOTE_SURCHARGE
        if params["express"]:
            core *= EXPRESS_MULTIPLIER
        core *= 1 - TIER_DISCOUNT[params["customer_tier"]]

        insurance = 0.0
        if params["declared_value"] > 0:
            insurance = max(INSURANCE_MIN, INSURANCE_RATE * params["declared_value"])

        total = round(core + insurance, 2)
        branches = {
            "weight_tier": weight_tier,
            "distance_band": f"{distance_fee:.2f}",
            "remote": str(params["remote"]),
            "express": str(params["express"]),
            "customer_tier": params["customer_tier"],
            "insured": str(params["declared_value"] > 0),
            "free_base": str(free_base),
        }
        return f"{total:.2f}", branches

    def render(self, params: dict[str, Any], rng: random.Random) -> tuple[str, int]:
        insurance_text = (
            f" El cliente declara un valor asegurado de {params['declared_value']:.2f} €."
            if params["declared_value"] > 0
            else " El envío no lleva seguro."
        )
        speed = "envío urgente" if params["express"] else "envío estándar"
        zone = "zona remota" if params["remote"] else "zona estándar"
        tier = TIER_LABEL[params["customer_tier"]]
        weight = params["weight_kg"]
        distance = params["distance_km"]
        order = f"{params['order_value']:.2f}"
        templates = [
            (
                f"Calcula el coste total de envío de un paquete de {weight} kg que viaja "
                f"{distance} km hasta una {zone}. Es un {speed} y el cliente está {tier}. "
                f"El importe del pedido es de {order} €.{insurance_text} "
                f"Da el resultado en euros con dos decimales."
            ),
            (
                f"Un pedido de {order} € se envía a una {zone} situada a {distance} km. "
                f"El paquete pesa {weight} kg y se contrata un {speed}. El cliente está "
                f"{tier}.{insurance_text} ¿Cuánto cuesta el envío en euros?"
            ),
            (
                f"Cliente {tier}. Pedido: {order} €. Paquete: {weight} kg. "
                f"Destino: {zone}, a {distance} km. Modalidad: {speed}.{insurance_text} "
                f"Indica el coste del envío con dos decimales."
            ),
            (
                f"Necesito saber qué cobramos por enviar {weight} kg a {distance} km "
                f"({zone}) con {speed}. El cliente está {tier} y su pedido asciende a "
                f"{order} €.{insurance_text} Responde en euros, con dos decimales."
            ),
            (
                f"Un {speed} de {weight} kg recorre {distance} km hasta una {zone}."
                f"{insurance_text} El pedido vale {order} € y el cliente está {tier}. "
                f"Calcula el coste total del envío en euros, con dos decimales."
            ),
        ]
        template_id = rng.randrange(len(templates))
        return templates[template_id], template_id


RULES_TEXT = """Tarifa de envío (reglas fijas de la empresa):
- Cargo base por peso: hasta 2 kg, 4,50 €. De 2 a 10 kg, 4,50 € más 0,80 € por kg que pase de
  2. De 10 a 30 kg, 10,90 € más 0,55 € por kg que pase de 10. Más de 30 kg, 21,90 € más
  0,35 € por kg que pase de 30.
- Cargo por distancia: hasta 100 km, 0 €. De 100 a 500 km, 3,00 €. Más de 500 km, 6,50 €.
- Si el importe del pedido llega a 60 € y el envío NO es urgente, el cargo base por peso se
  elimina; el cargo por distancia se mantiene.
- Zona remota: recargo del 25 % sobre la suma de base y distancia.
- Envío urgente: multiplica por 1,6 el resultado anterior.
- Descuento de fidelidad sobre el resultado anterior: plata 5 %, oro 12 %.
- Seguro (solo si hay valor declarado): 1,2 % del valor declarado, con un mínimo de 1,50 €.
  El seguro se suma al final y no recibe descuento.
- Redondea solo el total final a dos decimales."""


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--n", type=int, default=800, help="how many unique problems")
    parser.add_argument("--split", choices=["train", "test", "ood"], default="train")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="rlm/data/train.jsonl")
    parser.add_argument(
        "--with-rules",
        action="store_true",
        help="prepend the rule sheet to every statement (easier task: nothing to memorise)",
    )
    args = parser.parse_args()

    generator = ShippingCostGenerator()
    problems = generator.generate(args.n, args.split, args.seed)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for problem in problems:
            row = asdict(problem)
            if args.with_rules:
                row["question"] = f"{RULES_TEXT}\n\n{row['question']}"
            row["split"] = args.split
            row["label_source"] = "generator"
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(json.dumps(describe(problems), indent=2, ensure_ascii=False))
    print(f"\n{len(problems)} problemas -> {out}")
    print("Ejemplo:\n" + problems[0].question + f"\nRespuesta: {problems[0].answer}")


if __name__ == "__main__":
    main()
