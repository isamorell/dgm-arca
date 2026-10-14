"""Phase 1, out-of-distribution test: how much do SFT and GRPO degrade on new problems?

``rlm.evaluate`` scores every model on ``rlm/data/test.jsonl`` (same families as train) and,
run with ``--data rlm/data/test_ood.jsonl --out reports/phase1_eval_ood.json``, on the OOD set
(a family that never appears in train). This script reads both result files and answers:
*does the pass@1 of each model drop on the OOD set, and does it drop differently for SFT than
for GRPO?* It needs no GPU and no dataset, only the two JSON files.

Run::

    uv run python -m rlm.compare_ood --models sft grpo_sin_kl

and it writes ``reports/ood_comparison.md`` (tables ready to paste in docs/EXPERIMENTS.md).

Statistics, kept deliberately simple because each set has only 100-200 problems:

* **pass@1 interval**: Wilson score interval at 95 %. It behaves well for proportions close
  to 0 or 1, where the usual ``p +- 1.96 * sqrt(p(1-p)/n)`` interval goes outside [0, 1].
* **Drop** = pass@1(test) - pass@1(OOD), so a positive number means the model got worse. Its
  95 % interval is Newcombe's, built from the two Wilson intervals (the two sets contain
  different problems, so the samples are independent). If the interval contains 0 the drop
  cannot be told apart from noise with this many problems.
* **SFT vs GRPO on the same set** are scored on the *same* problems, so the comparison is
  paired: we count the problems that only one of the two solves and run an exact two-sided
  sign test on them (McNemar's exact test). It is much more sensitive than comparing two
  separate intervals.

Caveats are printed at the end of the report, because they belong next to the numbers.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from collections.abc import Callable, Sequence
from itertools import combinations
from pathlib import Path

Row = dict
ValueMatches = Callable[[Row], bool]

# Result kinds, same names as ``rlm/analyze_eval.py`` uses for the test set.
KINDS = ("ok", "valor_ok_unidad_mal", "valor_mal", "sin_answer")


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion (95 % by default)."""
    if n == 0:
        return 0.0, 1.0
    p = successes / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return max(0.0, centre - half), min(1.0, centre + half)


def newcombe_difference(
    k1: int, n1: int, k2: int, n2: int, z: float = 1.96
) -> tuple[float, float, float]:
    """Difference ``k1/n1 - k2/n2`` of two independent proportions with Newcombe's interval."""
    p1, p2 = k1 / n1, k2 / n2
    low1, up1 = wilson_interval(k1, n1, z)
    low2, up2 = wilson_interval(k2, n2, z)
    difference = p1 - p2
    lower = difference - math.sqrt((p1 - low1) ** 2 + (up2 - p2) ** 2)
    upper = difference + math.sqrt((up1 - p1) ** 2 + (p2 - low2) ** 2)
    return difference, lower, upper


def sign_test_p(only_a: int, only_b: int) -> float:
    """Exact two-sided sign test on the problems that only one of two models solves."""
    n = only_a + only_b
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(only_a, only_b) + 1)) / 2**n
    return min(1.0, 2 * tail)


def classify(row: Row, value_matches: ValueMatches) -> str:
    """One label per example: ok / right value but wrong unit / wrong value / no answer."""
    if row["is_correct"]:
        return "ok"
    if row.get("predicted") is None:
        return "sin_answer"
    if value_matches(row):
        return "valor_ok_unidad_mal"
    return "valor_mal"


def summarize(rows: Sequence[Row], value_matches: ValueMatches) -> dict:
    """pass@1 with its interval, format rate, result kinds and per-family counts."""
    n = len(rows)
    correct = sum(bool(r["is_correct"]) for r in rows)
    kinds = Counter(classify(r, value_matches) for r in rows)
    families: dict[str, dict[str, int]] = {}
    for family in sorted({r.get("family") or "?" for r in rows}):
        subset = [r for r in rows if (r.get("family") or "?") == family]
        families[family] = {
            "n": len(subset),
            "correct": sum(bool(r["is_correct"]) for r in subset),
        }
    return {
        "n": n,
        "correct": correct,
        "pass@1": correct / n if n else 0.0,
        "ci": wilson_interval(correct, n),
        "valid_format": sum(bool(r["has_valid_format"]) for r in rows) / n if n else 0.0,
        "kinds": {kind: kinds.get(kind, 0) for kind in KINDS},
        "families": families,
    }


def paired_counts(rows_a: Sequence[Row], rows_b: Sequence[Row]) -> dict:
    """Compare two models on the same problems (``rlm.evaluate`` keeps the dataset order)."""
    if len(rows_a) != len(rows_b) or any(
        a["question"] != b["question"] for a, b in zip(rows_a, rows_b, strict=True)
    ):
        raise ValueError("the two models were not evaluated on the same problems, in order")
    both = only_a = only_b = neither = 0
    for a, b in zip(rows_a, rows_b, strict=True):
        if a["is_correct"] and b["is_correct"]:
            both += 1
        elif a["is_correct"]:
            only_a += 1
        elif b["is_correct"]:
            only_b += 1
        else:
            neither += 1
    return {
        "both": both,
        "only_a": only_a,
        "only_b": only_b,
        "neither": neither,
        "p_value": sign_test_p(only_a, only_b),
    }


def compare(test: dict, ood: dict, models: Sequence[str], value_matches: ValueMatches) -> dict:
    """Everything the report needs, from the two ``rlm.evaluate`` result dictionaries."""
    missing = [m for m in models if m not in test or m not in ood]
    if missing:
        raise KeyError(
            f"model(s) {missing} not found in both files; "
            f"test has {sorted(test)}, ood has {sorted(ood)}"
        )
    report: dict = {"models": {}, "pairs": []}
    for model in models:
        on_test = summarize(test[model]["rows"], value_matches)
        on_ood = summarize(ood[model]["rows"], value_matches)
        drop, low, high = newcombe_difference(
            on_test["correct"], on_test["n"], on_ood["correct"], on_ood["n"]
        )
        report["models"][model] = {
            "test": on_test,
            "ood": on_ood,
            "drop": drop,
            "drop_ci": (low, high),
        }
    for a, b in combinations(models, 2):
        report["pairs"].append(
            {
                "a": a,
                "b": b,
                "test": paired_counts(test[a]["rows"], test[b]["rows"]),
                "ood": paired_counts(ood[a]["rows"], ood[b]["rows"]),
            }
        )
    return report


def pct(x: float) -> str:
    return f"{100 * x:.1f} %".replace(".", ",")


def interval(low: float, high: float) -> str:
    return f"[{100 * low:.1f}; {100 * high:.1f}]".replace(".", ",")


def points(x: float) -> str:
    return f"{100 * x:+.1f}".replace(".", ",")


def family_cell(counts: dict | None) -> str:
    """``correct/n (xx,x %)`` for one family, or ``-`` when the model has no such family."""
    if not counts:
        return "-"
    return f"{counts['correct']}/{counts['n']} ({pct(counts['correct'] / counts['n'])})"


def format_markdown(report: dict) -> str:
    """Spanish tables for docs/EXPERIMENTS.md (decimal comma, as in the rest of the log)."""
    models = list(report["models"])
    lines = [
        "# Degradación en el test fuera de distribución (OOD)",
        "",
        "Caída = pass@1 en test − pass@1 en OOD (positiva: el modelo empeora con problemas de "
        "una familia nueva). Intervalos al 95 %: Wilson para cada pass@1 y Newcombe para la caída.",
        "",
        "| Modelo | Test | n | OOD | n | Caída (puntos) |",
        "|---|---|---:|---|---:|---|",
    ]
    for model in models:
        entry = report["models"][model]
        on_test, on_ood = entry["test"], entry["ood"]
        lines.append(
            f"| {model} | {pct(on_test['pass@1'])} {interval(*on_test['ci'])} | {on_test['n']} "
            f"| {pct(on_ood['pass@1'])} {interval(*on_ood['ci'])} | {on_ood['n']} "
            f"| {points(entry['drop'])} {interval(*entry['drop_ci'])} |"
        )

    lines += ["", "## Aciertos por familia en OOD", ""]
    families = sorted({f for m in models for f in report["models"][m]["ood"]["families"]})
    lines += [
        "| Familia | n | " + " | ".join(models) + " |",
        "|---|---:|" + "---:|" * len(models),
    ]
    for family in families:
        counts = [report["models"][m]["ood"]["families"].get(family) for m in models]
        n = next((c["n"] for c in counts if c), 0)
        cells = [family_cell(c) for c in counts]
        lines.append(f"| {family} | {n} | " + " | ".join(cells) + " |")

    lines += ["", "## Tipo de resultado en OOD", ""]
    lines += [
        "| Modelo | " + " | ".join(KINDS) + " | formato válido |",
        "|---|" + "---:|" * (len(KINDS) + 1),
    ]
    for model in models:
        ood = report["models"][model]["ood"]
        cells = [str(ood["kinds"][kind]) for kind in KINDS]
        lines.append(f"| {model} | " + " | ".join(cells) + f" | {pct(ood['valid_format'])} |")

    if report["pairs"]:
        lines += ["", "## Comparación pareada (mismos problemas)", ""]
        lines += [
            "| Pareja | Conjunto | Solo A acierta | Solo B acierta | Ambos | Ninguno | p |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
        for pair in report["pairs"]:
            for split in ("test", "ood"):
                c = pair[split]
                lines.append(
                    f"| A = {pair['a']}, B = {pair['b']} | {split} | {c['only_a']} | "
                    f"{c['only_b']} | {c['both']} | {c['neither']} | "
                    f"{c['p_value']:.3f} |".replace(".", ",")
                )
        lines += [
            "",
            "p: test de signos exacto bilateral sobre los problemas que solo uno de los dos "
            "resuelve; un valor por debajo de 0,05 indica una diferencia que el azar explica mal.",
        ]

    lines += [
        "",
        "## Cautelas",
        "",
        "- Cada conjunto tiene pocos problemas: un intervalo que contiene el 0 significa que la "
        "caída no se distingue del ruido.",
        "- El conjunto OOD cambia de familia y puede cambiar también de dificultad (nivel); si es "
        "así, la caída mezcla ambos efectos. Comprobadlo en `rlm/data/test_ood.jsonl`.",
        "- Es una única generación greedy por problema: no mide la variabilidad entre "
        "ejecuciones ni entre semillas de entrenamiento.",
        "",
    ]
    return "\n".join(lines)


def load_results(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--test", default="reports/phase1_eval.json", help="rlm.evaluate on test")
    parser.add_argument(
        "--ood", default="reports/phase1_eval_ood.json", help="rlm.evaluate on test_ood"
    )
    parser.add_argument(
        "--models", nargs="+", default=["sft", "grpo_sin_kl"], help="names used in --adapters"
    )
    parser.add_argument("--out", default="reports/ood_comparison.md")
    return parser


def main() -> None:
    from rlm.verifier import MedicationVerifier

    args = build_parser().parse_args()
    verifier = MedicationVerifier()

    def value_matches(row: Row) -> bool:
        return verifier.is_correct(row["predicted"], row["expected_value"])

    report = compare(load_results(args.test), load_results(args.ood), args.models, value_matches)
    text = format_markdown(report)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(text)
    print(f"report -> {out}")


if __name__ == "__main__":
    main()
