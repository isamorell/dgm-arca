"""Phase 2 evaluation: does the model pick the right tool with the right arguments?

The benchmark is a JSONL file (see ``tool_use/bench/example_cases.jsonl``) where each line
holds a user query and the expected first tool call::

    {"query": "What's the temperature in Paris right now?",
     "expected_tool": "get_weather", "expected_arguments": {"location": "Paris, France"},
     "needs_tool": true}

Include cases where *no* tool is needed: a model that calls tools for everything is not
using them well. Two metrics are computed:

* tool selection accuracy: the first call names the expected tool (or there is no call when
  ``needs_tool`` is false);
* argument accuracy: among correct selections, the arguments match after normalisation.

Run::

    uv run python -m tool_use.evaluate --bench tool_use/bench/cases.jsonl \\
        --out reports/phase2_eval.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_bench(path: str | Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def normalise(value: Any) -> Any:
    """Make argument comparison tolerant to case, spacing and numeric formatting."""
    if isinstance(value, str):
        return " ".join(value.lower().split())
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, dict):
        return {k: normalise(v) for k, v in value.items()}
    return value


def score_case(case: dict[str, Any], tool_calls: list[dict[str, Any]]) -> dict[str, bool]:
    """Compare the calls the model made against the expectation for one benchmark case."""
    first = tool_calls[0] if tool_calls else None
    if not case.get("needs_tool", True):
        selection_ok = first is None
        return {"selection_ok": selection_ok, "arguments_ok": selection_ok}
    selection_ok = first is not None and first["name"] == case["expected_tool"]
    arguments_ok = selection_ok and normalise(first["arguments"]) == normalise(
        case.get("expected_arguments", {})
    )
    return {"selection_ok": selection_ok, "arguments_ok": arguments_ok}


def summarise(scores: list[dict[str, bool]]) -> dict[str, float]:
    n = max(len(scores), 1)
    return {
        "tool_selection_accuracy": sum(s["selection_ok"] for s in scores) / n,
        "argument_accuracy": sum(s["arguments_ok"] for s in scores) / n,
        "n_cases": len(scores),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--bench", default="tool_use/bench/example_cases.jsonl")
    parser.add_argument("--out", default="reports/phase2_eval.json")
    parser.add_argument("--max-turns", type=int, default=3)
    args = parser.parse_args()

    from tool_use.executor import run_tool_loop

    cases = load_bench(args.bench)
    rows = []
    for case in cases:
        response = run_tool_loop(case["query"], max_turns=args.max_turns)
        calls = [c.model_dump() for c in response.tool_calls]
        rows.append(
            {**case, "tool_calls": calls, "answer": response.answer, **score_case(case, calls)}
        )
        print(f"{'✓' if rows[-1]['selection_ok'] else '✗'} {case['query'][:70]}")

    summary = summarise(rows)
    print(json.dumps(summary, indent=2))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
