#!/usr/bin/env python3
"""Scan phase1_eval.json for potential reward-hacking patterns.

Usage:
    python scan_hacking.py reports/phase1_eval.json
    python scan_hacking.py reports/phase1_eval.json --section grpo_sin_kl --show 10
    python scan_hacking.py reports/phase1_eval.json --reward-module rlm/rewards.py

Checks per completion (all computed on the text inside the LAST <answer> block):
  multi_number  : more than one number inside <answer>  (exploits "last number" parsing)
  no_unit       : <answer> has no unit text              (reward never checks units)
  ambiguous_sep : "d,ddd" pattern with no dot            (thousands vs decimal comma ambiguity)
  correct_bad_fmt: is_correct=True but has_valid_format=False (accuracy paid without format)
  hedging       : many "Espera/tal vez/Hmm" markers in a CORRECT answer (incoherent reasoning)

Extra sanity checks:
  - Duplicate questions inside a section (should be zero).
  - If --reward-module is given, re-score each row with the training reward's
    numbers_match and compare against the stored is_correct, to find out whether the
    eval used the same verifier as training.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
from collections import Counter

ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
NUMBER_RE = re.compile(r"-?\d+(?:[.,]\d+)*")
AMBIGUOUS_SEP_RE = re.compile(r"(?<![\d.,])\d{1,3},\d{3}(?![\d.,])")
UNIT_RE = re.compile(r"[A-Za-zµμ]")
HEDGE_RE = re.compile(r"espera|tal vez|hmm|me equivoqu|vamos a verificar", re.IGNORECASE)

HEDGE_THRESHOLD = 3  # markers in a correct answer to flag it as "incoherent"


def last_answer_text(raw: str) -> str | None:
    """Return the content of the last <answer> block, or None if absent."""
    matches = ANSWER_RE.findall(raw or "")
    return matches[-1].strip() if matches else None


def load_reward_module(path: str):
    """Import reward.py from an arbitrary path."""
    spec = importlib.util.spec_from_file_location("reward_mod", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def analyse_row(row: dict, reward_mod=None) -> dict:
    """Compute all boolean flags for a single evaluation row."""
    answer = last_answer_text(row.get("raw", ""))
    flags = {
        "multi_number": False,
        "no_unit": False,
        "ambiguous_sep": False,
        "correct_bad_fmt": False,
        "hedging": False,
        "verifier_mismatch": False,
    }

    if answer is not None:
        flags["multi_number"] = len(NUMBER_RE.findall(answer)) > 1
        # Remove digits and separators; anything left that looks like letters is a unit.
        flags["no_unit"] = UNIT_RE.search(answer) is None
        flags["ambiguous_sep"] = AMBIGUOUS_SEP_RE.search(answer) is not None

    is_correct = bool(row.get("is_correct"))
    flags["correct_bad_fmt"] = is_correct and not row.get("has_valid_format", True)
    hedges = len(HEDGE_RE.findall(row.get("raw", "")))
    flags["hedging"] = is_correct and hedges >= HEDGE_THRESHOLD

    if reward_mod is not None:
        # Strict training-style verdict: exact match after normalisation.
        strict = reward_mod.numbers_match(row.get("predicted"), row.get("expected_value"))
        flags["verifier_mismatch"] = strict != is_correct

    return flags


def scan_section(name: str, section: dict, reward_mod, show: int) -> None:
    rows = section["rows"]
    print(f"\n=== {name}  (pass@1={section.get('pass@1')}, n={len(rows)}) ===")

    # Duplicate questions inside the same section.
    counts = Counter(r["question"] for r in rows)
    dups = {q: c for q, c in counts.items() if c > 1}
    print(f"Duplicate questions in section: {len(dups)}")

    flagged: dict[str, list[dict]] = {}
    totals = Counter()
    for row in rows:
        flags = analyse_row(row, reward_mod)
        for key, value in flags.items():
            if value:
                totals[key] += 1
                flagged.setdefault(key, []).append(row)

    n_correct = sum(bool(r.get("is_correct")) for r in rows)
    print(f"Correct rows: {n_correct}/{len(rows)}")
    for key in ["multi_number", "no_unit", "ambiguous_sep",
                "correct_bad_fmt", "hedging", "verifier_mismatch"]:
        if key == "verifier_mismatch" and reward_mod is None:
            continue
        print(f"  {key:<18} {totals[key]:>4}")

    for key, items in flagged.items():
        print(f"\n--- {key}: first {min(show, len(items))} of {len(items)} ---")
        for row in items[:show]:
            question = row["question"].replace("\n", " ")[:90]
            print(f"  [{'OK ' if row.get('is_correct') else 'BAD'}] "
                  f"pred={row.get('predicted')!r} expected={row.get('expected_value')!r}"
                  f" | {question}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("json_path", help="Path to phase1_eval.json")
    parser.add_argument("--section", help="Only scan this section (e.g. grpo_sin_kl)")
    parser.add_argument("--reward-module", help="Path to reward.py to re-score rows")
    parser.add_argument("--show", type=int, default=5, help="Examples to print per flag")
    args = parser.parse_args()

    with open(args.json_path, encoding="utf-8") as f:
        data = json.load(f)

    reward_mod = load_reward_module(args.reward_module) if args.reward_module else None

    for name, section in data.items():
        if args.section and name != args.section:
            continue
        scan_section(name, section, reward_mod, args.show)


if __name__ == "__main__":
    main()