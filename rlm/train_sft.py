"""Phase 1, step 3: cold-start supervised fine-tuning on verified reasoning traces.

This is the "SFT round 1" box of the DeepSeek pipeline. The input is the JSONL produced
by ``rlm/distill.py``: problems from your domain with a teacher-generated reasoning trace
that *passed your verifier*. The output is a LoRA adapter that already speaks the
``<think>…</think><answer>…</answer>`` format before we ever run GRPO.

Run::

    uv run python -m rlm.train_sft --data rlm/data/sft_traces.jsonl --output rlm/weights/sft_lora

What is decided for you: the trainer (``trl.SFTTrainer``), LoRA, the conversational
format (``prompt`` + ``completion`` columns, so that the loss is computed only on the
assistant turn). What we decided, and why (also written in docs/EXPERIMENTS.md):

* **Base model, Qwen3-0.6B.** The student is the model that GRPO will later train, and it has
  to fit with its optimizer state, activations and the 151k-token logits in a 16 GiB MIG slice.
  The teacher (Qwen3-4B) is the same family, so the student already knows the chat template and
  the ``<think>`` token.
* **LoRA r=16 on every linear layer (alpha 32).** A cold start only has to teach a *format and
  a style* (short Spanish reasoning, then ``<answer>value unit</answer>``), not new
  knowledge, so a low-rank adapter is enough and it cannot wipe out what the model knows.
* **Learning rate 2e-4.** The usual range for LoRA is 1e-4 to 3e-4 (it is ~10x higher than
  full fine-tuning because only the adapter learns); 2e-4 with a cosine decay is the
  middle of it.
* **3 epochs, best epoch kept.** With ~1,500 traces and an effective batch of 16 there are
  ~90 optimizer steps per epoch. Two epochs would leave the format half learned; many more would
  start memorising the 800 training problems (each appears with two traces). A small
  evaluation split, held out *by problem*, is scored after every epoch and the epoch with the
  lowest evaluation loss is the adapter that gets saved.
* **Maximum length 1536 tokens.** The teacher was limited to 1,200 new tokens and the prompt
  (system prompt plus statement) takes ~250, so every example fits. Examples that do not fit are
  dropped (never truncated: a truncated trace loses its ``</think><answer>`` ending and would
  teach the model not to finish). The script prints the real length distribution so that
  the choice can be checked against the data.
* **Batch 2 x 8 accumulation steps.** Effective batch of 16 sequences; the small per-device
  batch keeps the 151k-vocabulary logits inside 16 GiB.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections.abc import Sequence
from pathlib import Path

from datasets import Dataset, load_dataset

from rlm.data import build_prompt

WARMUP_FRACTION = 0.05  # share of the optimizer steps spent ramping the learning rate up


def load_verified_traces(path: str | Path) -> Dataset:
    """Read the distilled JSONL keeping every column and only the traces marked ``verified``."""
    dataset = load_dataset("json", data_files=str(path), split="train")
    if "verified" in dataset.column_names:
        dataset = dataset.filter(lambda ex: bool(ex["verified"]))
    return dataset


def to_prompt_completion(dataset: Dataset) -> Dataset:
    """Turn distilled traces into TRL's prompt/completion conversational format.

    Expected fields: ``question`` and ``trace`` (the full ``<think>…</think><answer>…</answer>``
    text). The prompt is the same one used at inference (``rlm.data.build_prompt``), so the
    student is trained on exactly what it will see later, without the Spanish hint that the
    teacher needed.
    """
    return dataset.map(
        lambda ex: {
            "prompt": build_prompt(ex["question"]),
            "completion": [{"role": "assistant", "content": ex["trace"]}],
        },
        remove_columns=dataset.column_names,
    )


def load_sft_dataset(path: str | Path) -> Dataset:
    """Whole distilled file as prompt/completion pairs (no evaluation split)."""
    return to_prompt_completion(load_verified_traces(path))


def split_by_problem(
    dataset: Dataset, eval_fraction: float, seed: int = 0
) -> tuple[Dataset, Dataset | None]:
    """Hold out whole problems for evaluation.

    Every problem has several traces with the same statement, so splitting by row would leak:
    a trace in the evaluation set would have its twin in the training set. The split is
    therefore made on the ``id`` column written by ``rlm.distill``. Without that column, or with
    ``eval_fraction <= 0``, there is no evaluation set.
    """
    if eval_fraction <= 0 or "id" not in dataset.column_names:
        return dataset, None
    ids = sorted(set(dataset["id"]))
    if len(ids) < 2:
        return dataset, None
    random.Random(seed).shuffle(ids)
    n_eval = min(max(1, round(len(ids) * eval_fraction)), len(ids) - 1)
    eval_ids = set(ids[:n_eval])
    train = dataset.filter(lambda ex: ex["id"] not in eval_ids)
    held_out = dataset.filter(lambda ex: ex["id"] in eval_ids)
    return train, held_out


def token_length(tokenizer, prompt: list[dict], completion: list[dict]) -> int:
    """Number of tokens of one training example once the chat template is applied."""
    text = tokenizer.apply_chat_template(prompt + completion, tokenize=False)
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def length_report(lengths: Sequence[int]) -> dict:
    """Median, percentiles and maximum of the example lengths (for the experiments log)."""
    if not lengths:
        return {"n": 0, "p50": 0, "p95": 0, "max": 0}
    ordered = sorted(lengths)

    def at(q: float) -> int:
        return ordered[int(q * (len(ordered) - 1))]

    return {"n": len(ordered), "p50": at(0.5), "p95": at(0.95), "max": ordered[-1]}


def drop_overlong(dataset: Dataset, tokenizer, max_length: int) -> tuple[Dataset, dict]:
    """Remove examples longer than ``max_length`` tokens instead of letting the trainer cut them.

    Returns the filtered dataset and the length report (with the number of dropped examples).
    """
    lengths = [token_length(tokenizer, ex["prompt"], ex["completion"]) for ex in dataset]
    keep = [i for i, length in enumerate(lengths) if length <= max_length]
    report = length_report(lengths)
    report["dropped"] = len(lengths) - len(keep)
    return dataset.select(keep), report


def warmup_steps(n_examples: int, batch_size: int, grad_accum: int, epochs: float) -> int:
    """Number of warm-up optimizer steps: 5 % of the run, at least one."""
    steps_per_epoch = math.ceil(n_examples / (batch_size * grad_accum))
    return max(1, round(WARMUP_FRACTION * steps_per_epoch * epochs))


def train(args: argparse.Namespace) -> None:
    import torch
    from peft import LoraConfig
    from transformers import AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    traces = load_verified_traces(args.data)
    print(f"{len(traces)} verified traces loaded from {args.data}")
    train_raw, eval_raw = split_by_problem(traces, args.eval_fraction, args.seed)
    train_set, eval_set = to_prompt_completion(train_raw), None
    if eval_raw is not None:
        eval_set = to_prompt_completion(eval_raw)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    train_set, report = drop_overlong(train_set, tokenizer, args.max_length)
    print(f"train lengths (tokens): {report}  -> examples dropped for exceeding max_length")
    if eval_set is not None:
        eval_set, eval_report = drop_overlong(eval_set, tokenizer, args.max_length)
        print(f"eval lengths (tokens): {eval_report}")
    print(f"{len(train_set)} training examples, {len(eval_set) if eval_set else 0} for evaluation")

    peft_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=2 * args.lora_rank,
        lora_dropout=0.05,
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )
    has_eval = eval_set is not None and len(eval_set) > 0
    config = SFTConfig(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps(len(train_set), args.batch_size, args.grad_accum, args.epochs),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_length=args.max_length,
        completion_only_loss=True,  # the loss covers the assistant turn only, not the prompt
        bf16=torch.cuda.is_available(),
        gradient_checkpointing=True,
        logging_steps=5,
        eval_strategy="epoch" if has_eval else "no",
        save_strategy="epoch",
        save_total_limit=3,
        load_best_model_at_end=has_eval,
        metric_for_best_model="eval_loss" if has_eval else None,
        seed=args.seed,
        report_to="none",
        model_init_kwargs={"dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32},
    )
    trainer = SFTTrainer(
        model=args.model,
        args=config,
        train_dataset=train_set,
        eval_dataset=eval_set if has_eval else None,
        peft_config=peft_config,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(args.output)
    history = Path(args.output) / "log_history.json"
    history.write_text(json.dumps(trainer.state.log_history, indent=2), encoding="utf-8")
    print(f"adapter saved to {args.output} (training history in {history})")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--data",
        default="rlm/data/sft_traces.jsonl",
        help="JSONL with question / trace / verified (and id, for the by-problem eval split)",
    )
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--output", default="rlm/weights/sft_lora")
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=1536)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument(
        "--eval-fraction",
        type=float,
        default=0.05,
        help="share of the problems held out to pick the best epoch (0 disables)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume-from-checkpoint", default=None, help="checkpoint-XXX folder")
    return parser


def main() -> None:
    train(build_parser().parse_args())


if __name__ == "__main__":
    main()
