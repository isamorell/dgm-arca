"""Phase 1, step 1b: generate reasoning traces with a teacher model and keep the verified ones.

This is what Sky-T1, OpenThoughts and DeepSeek's cold start have in common: a strong
model writes solutions with visible reasoning, a verifier throws away the wrong ones,
and what survives becomes SFT data. Here the teacher is any model that can think in the
``<think>…</think><answer>…</answer>`` format (Qwen3 in thinking mode works well; a
DeepSeek-R1 distilled model too).

Run::

    uv run python -m rlm.distill --data rlm/data/train.jsonl --teacher Qwen/Qwen3-4B \
        --samples 4 --output rlm/data/sft_traces.jsonl

Output: one JSON line per generated trace with ``question``, ``answer``, ``trace``,
``verified`` and ``teacher``. Report in EXPERIMENTS.md the acceptance rate: it is your
first measurement of how hard your domain is.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rlm.data import load_domain_dataset
from rlm.verifier import NumericVerifier, Verifier


def generate_traces(
    dataset, teacher: str, samples: int, max_new_tokens: int, verifier: Verifier
) -> list[dict]:
    """Tu turno: for each problem, sample ``samples`` completions from the teacher and verify them.

    Suggested steps:

    1. Load tokenizer and model (bf16 on GPU). Batch the prompts: generation dominates the cost.
    2. For each problem, ``generate`` with ``num_return_sequences=samples``, ``do_sample=True``.
    3. Decode, run ``verifier.verify(trace, answer)``, and store every trace with its verdict.
    4. Optional but recommended: if the teacher omits the ``<answer>`` tag but ends with
       ``\\boxed{...}``, rewrite the trace into the canonical format before saving.

    Watch out for traces that are correct by luck with nonsense reasoning: a second pass
    with an LLM judge, or a minimum-length filter, is a cheap way to catch some of them.
    """
    raise NotImplementedError


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data", required=True, help="domain JSONL with question / answer")
    parser.add_argument("--teacher", default="Qwen/Qwen3-4B")
    parser.add_argument("--samples", type=int, default=4, help="traces per problem")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--output", default="rlm/data/sft_traces.jsonl")
    args = parser.parse_args()

    dataset = load_domain_dataset(args.data)
    traces = generate_traces(
        dataset, args.teacher, args.samples, args.max_new_tokens, NumericVerifier()
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in traces:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    kept = sum(1 for t in traces if t["verified"])
    print(
        f"{kept}/{len(traces)} traces verified ({100 * kept / max(len(traces), 1):.1f}%) -> {out}"
    )


if __name__ == "__main__":
    main()
