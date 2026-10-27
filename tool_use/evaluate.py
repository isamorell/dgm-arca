"""Phase 2 evaluation: does the model pick the right tool with the right arguments?

The benchmark is a JSONL file (``tool_use/bench/cases.jsonl``) where each line holds a user
query and the expected first tool call::

    {"id": "c12", "category": "calculo",
     "query": "Paciente de 70 kg con una pauta de 15 mg/kg. ¿Qué dosis le corresponde?",
     "needs_tool": true, "expected_tool": "dose_calculator",
     "expected_arguments": {"expression": "15 * 70"}, "expected_answer": 1050}

Optional fields: ``expected_answer`` (number the final answer must contain), ``expected_sequence``
(tools of a multi-step case, in order), ``free_text_args`` (arguments only checked for presence,
such as the free-text ``steps`` of a preparation sheet) and ``notes``.

Include cases where *no* tool is needed: a model that calls tools for everything is not
using them well. The metrics are:

* tool selection accuracy: the first call names the expected tool (or there is no call when
  ``needs_tool`` is false);
* argument accuracy: among correct selections, the arguments match. Matching is semantic:
  expressions are compared by value (``15*70`` equals ``70 * 15``), units are compared by
  conversion (``mcg`` equals ``µg``), numbers by value, and arguments left at the tool's default
  value count as given (``max_results=5`` equals omitting it);
* answer accuracy (final answer): the final text contains the expected number, within a small
  tolerance. It is only defined for cases with ``expected_answer``.

The same model can be run in two conditions on the same cases: with the tools, and without them
(the model answers on its own). The model comes from ``--model-id`` or, if absent, from
``ARCA_RLM_BASE_MODEL`` in your ``.env`` (see ``.env.example``), and it is always passed to the
executor explicitly, so the executor does not need the adapter variables. Tools with side
effects (the preparation sheet) are not executed unless ``--allow-side-effects`` is passed: the
benchmark only needs the first call, and this keeps it free of files. Run::

    uv run python -m tool_use.evaluate --bench tool_use/bench/cases.jsonl \\
        --out reports/phase2_eval.json
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import re
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Any

# Used only when neither --model-id nor ARCA_RLM_BASE_MODEL (.env) is set.
FALLBACK_MODEL_ID = "Qwen/Qwen3-1.7B"

NO_TOOLS_SYSTEM_PROMPT = (
    "You are an AI assistant that helps nursing students with medication calculations. "
    "Answer the question directly."
)

# Tolerance when a number in the final answer is compared with the expected one. It must accept
# the unrounded value (1.6667 for an expected 1.67) but not an arithmetic slip: a relative
# tolerance of 0.5 % accepted 41.75 and 41.76 as 41.7, and those answers are wrong.
ANSWER_REL_TOL = 1e-4
ANSWER_ABS_TOL = 0.01


def load_bench(path: str | Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


# ---------------------------------------------------------------------------- argument matching


def normalise(value: Any) -> Any:
    """Make argument comparison tolerant to case, spacing and numeric formatting."""
    if isinstance(value, str):
        return " ".join(value.lower().split())
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, dict):
        return {k: normalise(v) for k, v in value.items()}
    return value


def _compact(text: Any) -> str:
    """Lowercase, no accents, no whitespace: '10 mg/mL' and '10mg/ml' compare equal."""
    text = unicodedata.normalize("NFD", str(text))
    return "".join(c for c in text.lower() if unicodedata.category(c) != "Mn" and not c.isspace())


def _to_float(value: Any) -> float | None:
    try:
        return float(str(value).replace(",", "."))
    except ValueError:
        return None


def _same_number(a: Any, b: Any) -> bool:
    x, y = _to_float(a), _to_float(b)
    return x is not None and y is not None and math.isclose(x, y, rel_tol=1e-6, abs_tol=1e-9)


@lru_cache(maxsize=1)
def _registry():
    from tool_use.tools import default_registry

    return default_registry()


def _defaults(tool_name: str) -> dict[str, Any]:
    """Default values of the tool's optional parameters (for example ``max_results=5``)."""
    tool = _registry().tools.get(tool_name)
    if tool is None:
        return {}
    return {
        name: p.default
        for name, p in inspect.signature(tool.fn).parameters.items()
        if p.default is not inspect.Parameter.empty
    }


def _same_unit(a: Any, b: Any) -> bool:
    """Same unit by name, or equal by conversion with Pint (``mcg`` equals ``µg``)."""
    if _compact(a) == _compact(b):
        return True
    try:
        from tool_use.tools import _unit_registry

        factor = _unit_registry().Quantity(1, str(a)).to(str(b)).magnitude
        return math.isclose(float(factor), 1.0, rel_tol=1e-9)
    except Exception:  # noqa: BLE001 - unknown unit or different kinds: not the same
        return False


def _same_expression(a: Any, b: Any) -> bool:
    """Two arithmetic expressions are the same if they evaluate to the same value."""
    from tool_use.tools import dose_calculator

    try:
        x = dose_calculator(expression=str(a), decimals=6)
        y = dose_calculator(expression=str(b), decimals=6)
    except Exception:  # noqa: BLE001 - the model wrote something the calculator rejects
        return False
    return math.isclose(x, y, rel_tol=1e-6, abs_tol=1e-9)


def _same_value(key: str, got: Any, expected: Any) -> bool:
    if key == "expression":
        return _same_expression(got, expected)
    if key in ("from_unit", "to_unit"):
        return _same_unit(got, expected)
    if key in ("value", "max_results", "decimals"):
        return _same_number(got, expected)
    if isinstance(expected, str):
        return _compact(got) == _compact(expected)
    return normalise(got) == normalise(expected)


def arguments_match(
    tool_name: str,
    got: dict[str, Any],
    expected: dict[str, Any],
    free_text_args: tuple[str, ...] | list[str] = (),
    ignore_optional: bool = False,
) -> bool:
    """Do the arguments of a call match the expected ones?

    Parameters left at their default are filled in on both sides before comparing, and the
    ``free_text_args`` only need to be present and non-empty. With ``ignore_optional``, an
    optional parameter the expected call does not mention (``max_results``, ``decimals``) may
    take any value: choosing 10 results instead of 5 is not a wrong call.
    """
    defaults = _defaults(tool_name)
    if ignore_optional:
        got = {k: v for k, v in got.items() if not (k in defaults and k not in expected)}
    got_full, expected_full = {**defaults, **got}, {**defaults, **expected}
    if set(got_full) != set(expected_full):
        return False
    for key, expected_value in expected_full.items():
        if key in free_text_args:
            if not str(got_full[key]).strip():
                return False
        elif not _same_value(key, got_full[key], expected_value):
            return False
    return True


# ---------------------------------------------------------------------------- final answer


def _readings(token: str) -> set[float]:
    """Possible numeric values of a token like '1.050' (thousands or decimal) or '2,5'."""
    if "." in token and "," in token:  # '1.050,5' or '1,050.5': the last separator is decimal
        cut = max(token.rfind("."), token.rfind(","))
        return {float(re.sub(r"[.,]", "", token[:cut]) + "." + token[cut + 1 :])}
    values: set[float] = set()
    if re.fullmatch(r"\d{1,3}([.,]\d{3})+", token):  # thousands separator
        values.add(float(re.sub(r"[.,]", "", token)))
    if re.fullmatch(r"\d+[.,]\d+", token):  # decimal separator
        values.add(float(token.replace(",", ".")))
    elif token.isdigit():
        values.add(float(token))
    return values


def numbers_in(text: str) -> set[float]:
    """All the numbers a text can be read as containing (Spanish or English formatting)."""
    found: set[float] = set()
    for token in re.findall(r"\d+(?:[.,]\d+)*", text):
        found |= _readings(token)
    return found


_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)


def strip_thinking(text: str) -> str:
    """Remove the reasoning from a completion, keeping only what the user would read.

    A model trained to reason (phase 1) writes ``<think> ... </think>`` before answering and
    the intermediate numbers of that reasoning must not count as the answer. Handles a closed
    block, a block cut by the token limit (everything after ``<think>`` is dropped) and a
    closing tag without an opening one (the template may have opened it in the prompt).
    """
    text = _THINK_BLOCK.sub("", text)
    if "<think>" in text:
        text = text.split("<think>", 1)[0]
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    return text.strip()


def has_unfinished_thinking(text: str) -> bool:
    """True if the completion opened a ``<think>`` block and never closed it (ran out of tokens)."""
    return "<think>" in _THINK_BLOCK.sub("", text)


def answer_ok(case: dict[str, Any], answer: str) -> bool | None:
    """Does the final answer contain the expected number? ``None`` if the case has none.

    Numbers that already appear in the query are ignored (so repeating the data does not count),
    unless the expected answer is one of them. This is a heuristic: it can miss a correct answer
    written in words and cannot tell the right number in the wrong sentence. The reasoning
    (``<think>`` block) is not part of the answer and is removed before looking for the number.
    """
    expected = case.get("expected_answer")
    if expected is None:
        return None
    candidates = numbers_in(strip_thinking(answer))
    from_query = numbers_in(case.get("query", ""))
    if not any(math.isclose(expected, q, rel_tol=1e-9) for q in from_query):
        candidates -= from_query
    return any(
        math.isclose(expected, c, rel_tol=ANSWER_REL_TOL, abs_tol=ANSWER_ABS_TOL)
        for c in candidates
    )


# ---------------------------------------------------------------------------- scoring


def score_case(case: dict[str, Any], tool_calls: list[dict[str, Any]]) -> dict[str, bool]:
    """Compare the calls the model made against the expectation for one benchmark case."""
    first = tool_calls[0] if tool_calls else None
    if not case.get("needs_tool", True):
        selection_ok = first is None
        return {
            "selection_ok": selection_ok,
            "arguments_ok": selection_ok,
            "arguments_lenient_ok": selection_ok,
        }
    selection_ok = first is not None and first["name"] == case["expected_tool"]

    def matches(ignore_optional: bool) -> bool:
        return selection_ok and arguments_match(
            case["expected_tool"],
            first["arguments"],
            case.get("expected_arguments", {}),
            case.get("free_text_args", ()),
            ignore_optional=ignore_optional,
        )

    scores = {
        "selection_ok": selection_ok,
        "arguments_ok": matches(False),
        "arguments_lenient_ok": matches(True),
    }
    if case.get("expected_sequence"):
        scores["sequence_ok"] = [c["name"] for c in tool_calls] == case["expected_sequence"]
    return scores


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95 % Wilson interval of a proportion. With ~30 cases it is wide: report it."""
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, centre - half), min(1.0, centre + half))


def summarise_answers(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Final-answer accuracy over the cases that have an expected answer."""
    scored = [r for r in rows if r.get("answer_ok") is not None]
    out: dict[str, Any] = {"n_answer_cases": len(scored)}
    if rows and any("think>" in r.get("answer", "") for r in rows):
        # Only for models that reason: how often the thinking ate the whole token budget.
        out["unfinished_thinking_rate"] = sum(
            has_unfinished_thinking(r.get("answer", "")) for r in rows
        ) / len(rows)
    if not scored:
        return out
    ok = sum(bool(r["answer_ok"]) for r in scored)
    low, high = wilson_interval(ok, len(scored))
    out["answer_accuracy"] = ok / len(scored)
    out["answer_accuracy_ci95"] = [round(low, 3), round(high, 3)]
    return out


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate the per-case scores of the with-tools condition.

    Selection and argument accuracy divide by *all* cases, so an error in selection also
    counts against arguments. The rates of false calls (a tool called when none was needed)
    and of missed calls (a tool needed but none called) are reported on their own.
    """
    n = max(len(rows), 1)
    out: dict[str, Any] = {
        "tool_selection_accuracy": sum(r["selection_ok"] for r in rows) / n,
        "argument_accuracy": sum(r["arguments_ok"] for r in rows) / n,
        "n_cases": len(rows),
    }
    if all("arguments_lenient_ok" in r for r in rows):
        out["argument_accuracy_ignoring_optional"] = (
            sum(r["arguments_lenient_ok"] for r in rows) / n
        )
    needs = [r for r in rows if r.get("needs_tool", True)]
    free = [r for r in rows if not r.get("needs_tool", True)]
    if free:
        out["false_call_rate"] = sum(not r["selection_ok"] for r in free) / len(free)
    if needs:
        out["missed_call_rate"] = sum(not r.get("tool_calls") for r in needs) / len(needs)
        out["wrong_tool_rate"] = sum(
            bool(r.get("tool_calls")) and not r["selection_ok"] for r in needs
        ) / len(needs)
    sequences = [r for r in rows if "sequence_ok" in r]
    if sequences:
        out["sequence_accuracy"] = sum(r["sequence_ok"] for r in sequences) / len(sequences)
    out.update(summarise_answers(rows))
    categories = sorted({r["category"] for r in rows if "category" in r})
    out["by_category"] = {
        cat: {
            "n": len(group),
            "selection": sum(r["selection_ok"] for r in group) / len(group),
            "arguments": sum(r["arguments_ok"] for r in group) / len(group),
        }
        for cat in categories
        for group in [[r for r in rows if r.get("category") == cat]]
    }
    return out


def compare_conditions(
    with_tools: list[dict[str, Any]], without_tools: list[dict[str, Any]]
) -> dict[str, Any]:
    """Final-answer correctness of the same model with and without tools, case by case."""
    key = lambda r: r.get("id") or r["query"]  # noqa: E731
    no_tools = {key(r): r for r in without_tools}
    both = only_tools = only_plain = neither = 0
    only_tools_ids, only_plain_ids = [], []
    for row in with_tools:
        other = no_tools.get(key(row))
        if row.get("answer_ok") is None or other is None or other.get("answer_ok") is None:
            continue
        a, b = bool(row["answer_ok"]), bool(other["answer_ok"])
        both += a and b
        neither += not a and not b
        if a and not b:
            only_tools += 1
            only_tools_ids.append(key(row))
        if b and not a:
            only_plain += 1
            only_plain_ids.append(key(row))
    n = both + only_tools + only_plain + neither
    return {
        "n_cases": n,
        "correct_with_tools": (both + only_tools) / n if n else None,
        "correct_without_tools": (both + only_plain) / n if n else None,
        "both_correct": both,
        "only_with_tools": only_tools,
        "only_without_tools": only_plain,
        "both_wrong": neither,
        "only_with_tools_ids": only_tools_ids,
        "only_without_tools_ids": only_plain_ids,
    }


def build_summary(rows: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """Summary of a run: each condition on its own and, with both, the comparison."""
    summary: dict[str, Any] = {}
    if "tools" in rows:
        summary["tools"] = summarise(rows["tools"])
    if "no_tools" in rows:
        summary["no_tools"] = summarise_answers(rows["no_tools"])
    if "tools" in rows and "no_tools" in rows:
        summary["comparison"] = compare_conditions(rows["tools"], rows["no_tools"])
    return summary


def rescore(data: dict[str, Any]) -> dict[str, Any]:
    """Recompute every score of a saved run from its stored calls and answers (no model needed).

    Use it when the scoring changes: the raw behaviour of the model (``tool_calls`` and
    ``answer``) is kept in the file, only the derived fields are rewritten.
    """
    for row in data["rows"].get("tools", []):
        row.update(score_case(row, row["tool_calls"]))
        row["answer_ok"] = answer_ok(row, row["answer"])
    for row in data["rows"].get("no_tools", []):
        row["answer_ok"] = answer_ok(row, row["answer"])
    data["summary"] = build_summary(data["rows"])
    return data


# ---------------------------------------------------------------------------- running


def _run_with_tools(loop, case: dict[str, Any], max_turns: int) -> dict[str, Any]:
    try:
        response = loop.run(case["query"], max_turns=max_turns)
        calls = [c.model_dump() for c in response.tool_calls]
        answer, error = response.answer, None
    except Exception as exc:  # noqa: BLE001 - one broken case must not stop the benchmark
        calls, answer, error = [], "", f"{type(exc).__name__}: {exc}"
    row = {**case, "condition": "tools", "tool_calls": calls, "answer": answer}
    row.update(score_case(case, calls))
    row["answer_ok"] = answer_ok(case, answer)
    if error:
        row["error"] = error
    return row


def no_tools_messages(query: str, prompt: str = "neutral") -> list[dict[str, str]]:
    """Conversation of the no-tools condition.

    ``neutral`` is a plain instruction, the same for every model. ``phase1`` is the system
    prompt the reasoning model was trained and evaluated with (``<think>``/``<answer>``), so a
    fine-tuned adapter is not penalised in the baseline for being used off-distribution.
    """
    if prompt == "phase1":
        from rlm.data import build_prompt  # lazy: pulls in ``datasets``

        return build_prompt(query)
    return [
        {"role": "system", "content": NO_TOOLS_SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ]


def apply_adapter(loop, adapter: str) -> None:
    """Put a LoRA adapter (local folder or Hugging Face repo id) on top of the loaded base model.

    ``ToolLoop.load`` only loads the base model; this is the same ``PeftModel.from_pretrained``
    that ``rlm/inference.py`` uses for the phase-1 model. The base model the adapter was trained
    on (stored in its config) must be the one in ``loop``; a mismatch is reported, not hidden.
    """
    from peft import PeftConfig, PeftModel

    trained_on = PeftConfig.from_pretrained(adapter).base_model_name_or_path
    if trained_on and trained_on != loop.model_id:
        print(f"WARNING: adapter trained on {trained_on}, but the loaded model is {loop.model_id}")
    loop.model = PeftModel.from_pretrained(loop.model, adapter).eval()


def enable_thinking(loop) -> None:
    """Let the model write its ``<think>`` block, as it did in phase 1.

    ``ToolLoop.generate`` always renders the chat template with ``enable_thinking=False``, which
    pre-fills an empty think block. That is right for vanilla Qwen3 but off-distribution for a
    model trained to open ``<think>`` itself. The template flag is overridden here so the
    executor (not ours) stays untouched.
    """
    original = loop.tokenizer.apply_chat_template

    def apply_chat_template(*args, **kwargs):
        kwargs["enable_thinking"] = True
        return original(*args, **kwargs)

    loop.tokenizer.apply_chat_template = apply_chat_template


def _run_without_tools(loop, case: dict[str, Any], prompt: str = "neutral") -> dict[str, Any]:
    messages = no_tools_messages(case["query"], prompt)
    try:
        answer, error = loop.generate(messages).strip(), None
    except Exception as exc:  # noqa: BLE001
        answer, error = "", f"{type(exc).__name__}: {exc}"
    row = {**case, "condition": "no_tools", "tool_calls": [], "answer": answer}
    row["answer_ok"] = answer_ok(case, answer)
    if error:
        row["error"] = error
    return row


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--bench", default="tool_use/bench/cases.jsonl")
    parser.add_argument(
        "--out", default=None, help="Output JSON (default: reports/phase2_eval.json)."
    )
    parser.add_argument(
        "--rescore",
        metavar="JSON",
        help="Recompute the scores of a saved run with the current scoring, without a model.",
    )
    parser.add_argument("--max-turns", type=int, default=5)
    parser.add_argument(
        "--model-id",
        default=None,
        help="Hugging Face model id or local path (default: ARCA_RLM_BASE_MODEL from .env).",
    )
    parser.add_argument(
        "--adapter",
        default=None,
        help="LoRA adapter (local folder or Hugging Face repo id) to apply on the base model.",
    )
    parser.add_argument(
        "--thinking",
        action="store_true",
        help="Let the model reason in <think> (the phase-1 format) instead of the executor's "
        "enable_thinking=False. Use it with --adapter.",
    )
    parser.add_argument(
        "--no-tools-prompt",
        choices=["neutral", "phase1"],
        default="neutral",
        help="System prompt of the no-tools condition: a neutral one (default) or the phase-1 "
        "R1-Zero prompt the adapter was trained with.",
    )
    parser.add_argument(
        "--allow-side-effects",
        action="store_true",
        help="Really execute tools that require confirmation (writes the PDF sheets).",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=["tools", "no-tools"],
        default=["tools", "no-tools"],
        help="Run the model with the tools, without them, or both (default).",
    )
    args = parser.parse_args()

    if args.rescore:
        data = rescore(json.loads(Path(args.rescore).read_text(encoding="utf-8")))
        print(json.dumps(data["summary"], indent=2, ensure_ascii=False))
        Path(args.out or args.rescore).write_text(json.dumps(data, indent=2, ensure_ascii=False))
        return

    from dotenv import load_dotenv

    load_dotenv()  # each teammate defines their own .env (see .env.example)
    model_id = args.model_id or os.environ.get("ARCA_RLM_BASE_MODEL") or FALLBACK_MODEL_ID
    print(f"Model: {model_id}")

    from tool_use.executor import ToolLoop

    loop = ToolLoop(model_id=model_id, allow_side_effects=args.allow_side_effects)
    loop.load()
    if args.adapter:
        apply_adapter(loop, args.adapter)
        print(f"Adapter: {args.adapter}")
    if args.thinking:
        enable_thinking(loop)
    cases = load_bench(args.bench)

    rows: dict[str, list[dict[str, Any]]] = {}
    if "tools" in args.conditions:
        rows["tools"] = []
        for case in cases:
            rows["tools"].append(_run_with_tools(loop, case, args.max_turns))
            mark = "✓" if rows["tools"][-1]["selection_ok"] else "✗"
            print(f"[tools]    {mark} {case['query'][:70]}")
    if "no-tools" in args.conditions:
        rows["no_tools"] = []
        for case in cases:
            rows["no_tools"].append(_run_without_tools(loop, case, args.no_tools_prompt))
            print(f"[no-tools] {case['query'][:70]}")

    summary = build_summary(rows)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    out = Path(args.out or "reports/phase2_eval.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "model": loop.model_id,
                "adapter": args.adapter,
                "thinking": args.thinking,
                "no_tools_prompt": args.no_tools_prompt,
                "bench": args.bench,
                "summary": summary,
                "rows": rows,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
