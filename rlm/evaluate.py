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
import json
from pathlib import Path

from rlm.data import load_domain_dataset, load_gsm8k
from rlm.inference import ReasoningModel
from rlm.rewards import has_valid_format
from rlm.verifier import MedicationVerifier, NumericVerifier, Verifier

def evaluate_model(
    base_model: str, adapter: str | None, dataset, verifier: Verifier, max_new_tokens: int
) -> list[dict]:
    """Tu turno: greedy (or low-temperature) generation for every example, one verdict each.

    Return one dict per example with ``question``, ``expected``, ``raw``, ``predicted``,
    ``is_correct``, ``has_valid_format`` and ``n_tokens``. Reuse ``rlm.inference.ReasoningModel``
    instead of writing generation code again.
    """
    rlm = ReasoningModel(base_model=base_model, adapter_path=adapter)
    rlm.load() # loads tokenizer + base model (+ LoRA adapter if given) on GPU or CPU

    rows: list[dict] = []
    for ex in dataset:
        # The loaders store the question inside ``prompt`` (system + user messages)
        question = ex["prompt"][-1]["content"]
        raw, n_tokens = rlm.generate(question, max_new_tokens, greedy=True)

        if isinstance(verifier, MedicationVerifier):
            # Same call as rlm.distill: the unit asked by the problem must be in the answer
            result = verifier.verify(
                raw, ex["answer"], ex.get("answer_unit"), ex.get("answer_unit_aliases") or {}
            )
        else:
            result = verifier.verify(raw, ex["answer"])

        unit = ex.get("answer_unit")
        rows.append(
            {
                "question": question,
                "family": ex.get("family"),
                "expected": f"{ex['answer']} {unit}" if unit else ex["answer"],
                "expected_value": ex["answer"],
                "expected_unit": unit,
                "raw": raw,
                "predicted": result.predicted,
                "is_correct": result.is_correct,
                "has_valid_format": has_valid_format(raw),
                "n_tokens": n_tokens,
            }
        )

    return rows

def pass_at_1(rows: list[dict]) -> float:
    return sum(r["is_correct"] for r in rows) / max(len(rows), 1)

def plot_pass1(results: dict, out: Path) -> None:
    """Bar chart with the pass@1 of every evaluated model."""
    import matplotlib
 
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
 
    names = list(results)
    values = [results[name]["pass@1"] for name in names]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(names, values)
    ax.bar_label(bars, labels=[f"{v:.1%}" for v in values])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("pass@1")
    ax.set_title("Phase 1: pass@1")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)

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
    parser.add_argument("--history", default="reports/phase1_pass1.png")
    args = parser.parse_args()

    dataset = (
        load_gsm8k("test", n_examples=args.n_examples)
        if args.data == "gsm8k"
        else load_domain_dataset(args.data)
    )

    # GSM8K has no units and integer answers (a derived tolerance would be too loose there).
    verifier = NumericVerifier() if args.data == "gsm8k" else MedicationVerifier()

    out = Path(args.out)

    if out.exists():
        results = json.loads(out.read_text(encoding="utf-8"))
    else:
        out.parent.mkdir(parents=True, exist_ok=True)
        results = {}

    for pair in args.adapters:
        name, path = pair.split("=", 1)
        rows = evaluate_model(
            args.model,
            None if path == "none" else path,
            dataset,
            verifier,
            args.max_new_tokens,
        )
        results[name] = {"pass@1": pass_at_1(rows), "rows": rows}
        print(f"{name:>8}: pass@1 = {results[name]['pass@1']:.3f} on {len(rows)} problems")

    out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"details -> {out}")
    plot_pass1(results, Path(args.history))
    print(f"chart -> {args.history}")


if __name__ == "__main__":
    main()
