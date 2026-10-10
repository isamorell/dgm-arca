"""Phase 1 evaluation: pass@1 of base vs SFT vs GRPO on a held-out set, plus training curves.

Run::

    uv run python -m rlm.evaluate --data rlm/data/test.jsonl \
        --adapters base=none sft=rlm/weights/sft_lora grpo=rlm/weights/final_rlm_lora

It writes ``reports/phase1_eval.json`` with per-example verdicts (so you can do the failure
analysis) and ``reports/phase1_pass1.png`` with the bar chart. ``--history`` plots the
reward curves from the JSON history that the training scripts save.
"""

from __future__ import annotations

import argparse
import gc
import inspect
import json
from pathlib import Path

from rlm.data import load_domain_dataset, load_gsm8k
from rlm.inference import VERIFIERS
from rlm.rewards import has_valid_format
from rlm.verifier import NumericVerifier, Verifier


def _question_from_prompt(prompt: list[dict]) -> str:
    """Recover the plain statement from the ``prompt`` column.
    
    ``load_domain_dataset`` stores the statement inside ``prompt`` (a system message followed
    by a user message) and drops the original ``question`` column. ``ResoningModel.generate``
    wants the bare statement because it rebuilds the chat prompt itself, so we take the content
    of the last ``user`` message.
    """
    for message in reversed(prompt):
        if message["role"] == "user":
            return message["content"]

    raise ValueError("Prompt without a user message")

def _unit_kwargs(verifier: Verifier, example: dict) -> dict:
    """Extra arguments for ``verifier.verify`` when it knows about units (like ours).
    
    ``MedicationVerifier.verify(completion, expected, unit, unit_aliases)`` checks the unit of
    the ``<answer>`` block; the asked unit travels in the dataset columns ``answer_unit`` and
    ``answer_unit_aliases`` (written by the problem generator). The plain verifiers
    (``NumericVerifier``, ``ExactMathVerifier``) have no ``unit`` parameter, so for them, or
    for rows without a unit (GSM8K), we pass nothing and they behave exactly as before.
    """
    unit = example.get("answer_unit")
    if not unit or "unit" not in inspect.signature(verifier.verify).parameters:
        return {}
    return {"unit": unit, "unit_aliases": tuple(example.get("answer_unit_aliases") or ())}

def _level(example:dict) -> str | None:
    """Difficulty level of a problem.

    The generator stores the branches taken by ``solve`` in the ``branches`` field, either as
    a dict or a JSON string (HugginFace ``datasets`` is happier with strings when keys
    vary between rows). The level lies inside it; ``None`` if it is not there.
    """
    branches = example.get("branches")
    if isinstance(branches, str):
        try:
            branches = json.loads(branches)
        except json.JSONDecodeError:
            return None
    if isinstance(branches, dict) and "level" in branches:
        return str(branches["level"])
    return None

def classify(is_correct: bool, value_correct: bool, predicted: str | None, truncated: bool) -> str:
    """Why a completion failed: the categories for the failure analysis of the report.

    * ``ok``           accepted by the verifier.
    * ``truncated``    hit ``max_new_tokens`` before closing ``</answer>``: raise the budget.
    * ``no_answer``    finished but without an ``<answer>`` block (format failure).
    * ``wrong_unit``   right number, wrong or missing unit (mg vs mcg, mL vs mL/h...).
    * ``wrong_value``  the number itself is wrong (reasoning or arithmetic error).
    """
    if is_correct:
        return "ok"
    if predicted is None:
        return "truncated" if truncated else "no_answer"
    return "wrong_unit" if value_correct else "wrong_value"

def evaluate_model(
    base_model: str, adapter: str | None, dataset, verifier: Verifier, max_new_tokens: int
) -> list[dict]:
    """Tu turno: greedy (or low-temperature) generation for every example, one verdict each.

    Return one dict per example with ``question``, ``expected``, ``raw``, ``predicted``,
    ``is_correct``, ``has_valid_format`` and ``n_tokens``. Reuse ``rlm.inference.ReasoningModel``
    instead of writing generation code again.
    """
    import torch

    from rlm.inference import ReasoningModel # template class

    rlm = ReasoningModel(base_model=base_model, adapter_path=adapter)
    rlm.load() # loads tokenizer + base model (+ LoRA adapter if given) on GPU or CPU
    # ``ReasoningModel.generate`` samples (temperature 0.6, top-p 0.95), so pass@1 is a random
    # variable. Fixing the seed makes two runs of the same model comparable.
    torch.manual_seed(0)

    rows: list[dict] = []
    for i, example in enumerate(dataset):
        question = _question_from_prompt(example["prompt"])
        expected = str(example["answer"])

        raw, n_tokens = rlm.generate(question, max_new_tokens)

        # ``verify`` extracts the <answer> block from ``raw`` (rlm.rewards.extract_answer) and
        # compares it with ``expected``. It returns a VerificationResult(is_correct, predicted,
        # expected, detail).
        unit_kwargs = _unit_kwargs(verifier, example)
        result = verifier.verify(raw, expected, **unit_kwargs)

        # Same completion judged on the value alone. The gap between ``is_correct`` and
        # ``value_correct`` counts the answers with the right number but a wrong or missing
        # unit (mg vs mcg, mL vs mL/h): the error the medication verifier exists to catch.
        value_only = verifier.verify(raw, expected) if unit_kwargs else result

        rows.append(
            {
                "question": question,
                "expected": expected,
                "unit": unit_kwargs.get("unit"),
                "family": example.get("family"),
                "level": _level(example),
                "raw": raw,
                "predicted": result.predicted,
                "is_correct": result.is_correct,
                "value_correct": value_only.is_correct,
                "detail": result.detail,  # e.g. "right value but missing or wrong unit (mg)"
                "failure": classify(
                    result.is_correct,
                    value_only.is_correct,
                    result.predicted,
                    truncated=n_tokens >= max_new_tokens,
                ),
                # Strict structure check (single think block, then a single answer block).
                "has_valid_format": has_valid_format(raw),
                "n_tokens": n_tokens,
            }
        )

        if (i+1) % 20 == 0:
            done = sum(r["is_correct"] for r in rows)
            print(f"  {i + 1}/{len(dataset)}  running pass@1 = {done / len(rows):.3f}")

    # Free the GPU before the next adapter is loaded (we evaluate base, SFT and GRPO in the
    # same process and a 0.6B model plus its cache adds up on a small card).
    del rlm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return rows

def pass_at_1(rows: list[dict]) -> float:
    return sum(r["is_correct"] for r in rows) / max(len(rows), 1)

def value_only_pass_at_1(rows: list[dict]) -> float:
    """pass@1 ignoring the unit (ours).
 
    Equal to ``pass_at_1`` when the verifier has no unit check. The difference between the two
    is the share of problems where the model got the number right and the unit wrong.
    """
    return sum(r["value_correct"] for r in rows) / max(len(rows), 1)

def failure_counts(rows: list[dict]) -> dict[str, int]:
    """How many problems fall in each failure category (ours, see ``classify``)."""
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["failure"]] = counts.get(row["failure"], 0) + 1
    return dict(sorted(counts.items(), key=lambda item: -item[1]))

def breakdown(rows: list[dict], key: str) -> dict[str, dict]:
    """pass@1 (with and without the unit) per value of ``key``, e.g. ``family`` or ``level`` (ours).
 
    This is the table that shows whether a good overall pass@1 hides a model that only solves
    the easy branch of the generator, which is the "all the weight on the easy branch" error
    that the datasets guide warns about.
    """
    groups: dict[str, list[dict]] = {}
    for row in rows:
        if row.get(key) is not None:
            groups.setdefault(str(row[key]), []).append(row)
    return {
        name: {
            "n": len(group),
            "pass@1": pass_at_1(group),
            "pass@1_value_only": value_only_pass_at_1(group),
        }
        for name, group in sorted(groups.items())
    }

def plot_pass1(results: dict, out: Path) -> None:
    """Bar chart of pass@1 (with binomial standard error) and valid-format rate per model."""
    import matplotlib
 
    matplotlib.use("Agg")  # no display on the DGX: draw straight to a file
    import matplotlib.pyplot as plt
 
    names = list(results)
    p = [results[n]["pass@1"] for n in names]
    n_rows = [len(results[n]["rows"]) for n in names]
    # Standard error of a proportion: sqrt(p(1-p)/n). With 200 problems it is about +-3 points,
    # so differences smaller than that between two models are noise.
    se = [(pi * (1 - pi) / max(n, 1)) ** 0.5 for pi, n in zip(p, n_rows, strict=True)]
    fmt = [
        sum(r["has_valid_format"] for r in results[n]["rows"]) / max(len(results[n]["rows"]), 1)
        for n in names
    ]
    x = range(len(names))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([i - 0.2 for i in x], p, 0.4, yerr=se, capsize=4, label="pass@1")
    ax.bar([i + 0.2 for i in x], fmt, 0.4, label="valid format")
    ax.set_xticks(list(x), names)
    ax.set_ylim(0, 1.05)
    ax.set_title("Phase 1: base vs SFT vs GRPO")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)

def plot_history(paths: list[str], out: Path) -> None:
    """Plot reward and completion-length curves from ``trainer_state.json`` files.
 
    Each path is ``name=file.json`` (or just a file). The file is the ``trainer_state.json``
    that the HuggingFace ``Trainer`` writes inside every ``checkpoint-XXX`` folder: it has a
    ``log_history`` list with one dict per logging step. TRL's ``GRPOTrainer`` logs there
    ``reward`` (the weighted total), ``rewards/<function_name>/mean`` (one per reward function)
    and ``completions/mean_length``. Key names depend on the TRL version, so if a curve is
    missing, open the JSON and adjust the key.
    """
    import matplotlib
 
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
 
    fig, (ax_r, ax_l) = plt.subplots(1, 2, figsize=(11, 4))
    for item in paths:
        name, _, file = item.partition("=") if "=" in item else (Path(item).stem, "", item)
        history = json.loads(Path(file).read_text())["log_history"]
        steps = [h["step"] for h in history if "reward" in h]
        ax_r.plot(steps, [h["reward"] for h in history if "reward" in h], label=f"{name} total")
        # One dashed line per individual reward (format, accuracy, domain) to see which one moves.
        for key in sorted({k for h in history for k in h if k.startswith("rewards/")}):
            pts = [(h["step"], h[key]) for h in history if key in h]
            ax_r.plot(*zip(*pts, strict=True), alpha=0.6, linestyle="--", label=key)
        length_key = "completions/mean_length"
        pts = [(h["step"], h[length_key]) for h in history if length_key in h]
        if pts:
            ax_l.plot(*zip(*pts, strict=True), label=name)
    ax_r.set(title="Reward during GRPO", xlabel="step", ylabel="mean reward")
    ax_l.set(title="Completion length", xlabel="step", ylabel="tokens")
    ax_r.legend(fontsize=7)
    ax_l.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"curves -> {out}")

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data", default="gsm8k", help="'gsm8k' (test split) or your test JSONL")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument(
        "--adapters",
        nargs="+",
        default=["base=none"],
        help="name=path pairs; use 'none' for the bare base model",
    )
    parser.add_argument("--n-examples", type=int, default=200)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--out", default="reports/phase1_eval.json")
    args = parser.parse_args()

    # Curves mode: no model is loaded, just read the training logs and draw.
    if args.history:
        out = Path(args.out).with_name("phase1_curves.png")
        out.parent.mkdir(parents=True, exist_ok=True)
        plot_history(args.history, out)
        return

    # 1. Dataset. GSM8K is the control set; anything else is our own JSONL.
    dataset = (
        load_gsm8k("test", n_examples=args.n_examples)
        if args.data == "gsm8k"
        else load_domain_dataset(args.data)
    )
    if args.n_examples and len(dataset) > args.n_examples:
        # Shuffle first: a JSONL sorted by family or level would otherwise give a biased subset.
        dataset = dataset.shuffle(seed=0).select(range(args.n_examples))

    # 2. Verifier. The registry lives in rlm/inference.py; our domain verifier must be
    #    registered there as "medication", otherwise we stop with a clear message.
    verifier_name = args.verifier or ("numeric" if args.data == "gsm8k" else "medication")
    if verifier_name not in VERIFIERS:
        raise SystemExit(
            f"verifier '{verifier_name}' is not registered; known: {sorted(VERIFIERS)}. "
            "Add it to VERIFIERS in rlm/inference.py."
        )
    verifier = (
        NumericVerifier(args.tolerance)  # only the numeric verifier takes a tolerance argument
        if verifier_name == "numeric"
        else VERIFIERS[verifier_name]()
    )

    # 3. Evaluate every ``name=path`` pair (base, SFT, GRPO...) on the same problems.    
    results = {}
    for pair in args.adapters:
        name, path = pair.split("=", 1)
        rows = evaluate_model(
            args.model,
            None if path == "none" else path,
            dataset,
            NumericVerifier(),
            args.max_new_tokens,
        )
        results[name] = {
            "pass@1": pass_at_1(rows),
            "pass@1_value_only": value_only_pass_at_1(rows),
            "by_family": breakdown(rows, "family"),
            "by_level": breakdown(rows, "level"),
            "failures": failure_counts(rows),
            "rows": rows,  # every verdict, for the failure analysis
        }
        print(
            f"{name:>8}: pass@1 = {results[name]['pass@1']:.3f} "
            f"(value only: {results[name]['pass@1_value_only']:.3f}) on {len(rows)} problems"
        )
        print(f"{'':>8}  failures: {results[name]['failures']}")
        for title, key in (("family", "by_family"), ("level", "by_level")):
            for group, stats in results[name][key].items():
                print(f"{'':>8}  {title} {group}: pass@1 = {stats['pass@1']:.3f} (n={stats['n']})")

    # 4. Save the JSON with all the verdicts and the bar chart next to it.
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"details -> {out}")


if __name__ == "__main__":
    main()
