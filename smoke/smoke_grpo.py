"""Smoke test: a real (but tiny) GRPO run to prove the GPU, the environment and TRL work.

What it does, in order:

1. Prints the environment (device, memory, library versions).
2. Loads a small instruct model with a LoRA adapter.
3. Loads a few hundred GSM8K problems with the R1-Zero system prompt from class.
4. Trains with ``GRPOTrainer`` using two verifiable rewards: format and accuracy.
5. Prints, at every step, the mean reward of each kind, the completion length and GPU memory.
6. Saves the adapter and shows the model's answer to three held-out problems
   *before* and *after* training, side by side.

Typical use on the DGX (10-15 minutes on a single modern GPU):

    uv run arca-smoke
    uv run arca-smoke --steps 60 --num-generations 16

Without a GPU, to check that the installation itself works (about a minute on CPU):

    uv run arca-smoke --dry-run

This is a *template*, not the phase 1 deliverable. Read it, run it, and then go to
``rlm/README.md`` to see what you have to build on top of it.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

DEFAULT_MODEL = "Qwen/Qwen3-0.6B"
DRY_RUN_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"


@dataclass
class SmokeConfig:
    model: str = DEFAULT_MODEL
    steps: int = 40
    num_generations: int = 8
    max_completion_length: int = 384
    n_examples: int = 256
    learning_rate: float = 1e-5
    lora_rank: int = 16
    output_dir: str = "rlm/weights/smoke_lora"
    seed: int = 0
    dry_run: bool = False
    n_showcase: int = 3


def parse_args() -> SmokeConfig:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL, help="Hugging Face model id or local path"
    )
    parser.add_argument("--steps", type=int, default=40, help="number of GRPO optimisation steps")
    parser.add_argument("--num-generations", type=int, default=8, help="group size G in GRPO")
    parser.add_argument("--max-completion-length", type=int, default=384)
    parser.add_argument("--n-examples", type=int, default=256, help="GSM8K problems to sample")
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--output-dir", default="rlm/weights/smoke_lora")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="tiny model, 2 steps, CPU-friendly: checks the install, not the learning",
    )
    args = parser.parse_args()
    config = SmokeConfig(**{k.replace("-", "_"): v for k, v in vars(args).items()})
    if config.dry_run:
        config.model = DRY_RUN_MODEL if args.model == DEFAULT_MODEL else args.model
        config.steps = min(config.steps, 2)
        config.num_generations = 2
        config.max_completion_length = 32
        config.n_examples = 8
        config.n_showcase = 1
    return config


def pick_device_and_dtype(force_cpu: bool = False):
    """CUDA with bf16 when available; otherwise float32 on CPU.

    Apple's MPS backend is deliberately not used: generation inside the GRPO loop is
    unreliable there, and the dry run only needs to prove that the code path works.
    """
    import torch

    if torch.cuda.is_available() and not force_cpu:
        return "cuda", torch.bfloat16
    return "cpu", torch.float32


def gpu_memory_gb() -> float | None:
    import torch

    if not torch.cuda.is_available():
        return None
    return torch.cuda.max_memory_allocated() / 1e9


def print_environment(config: SmokeConfig, device: str) -> None:
    import torch
    import transformers
    import trl

    table = Table(show_header=False, box=None)
    table.add_row("model", config.model)
    table.add_row("device", device)
    if device == "cuda":
        props = torch.cuda.get_device_properties(0)
        table.add_row(
            "gpu", f"{props.name} · {props.total_memory / 1e9:.0f} GB · CUDA {torch.version.cuda}"
        )
    table.add_row(
        "torch / transformers / trl",
        f"{torch.__version__} / {transformers.__version__} / {trl.__version__}",
    )
    table.add_row(
        "steps · group size · max tokens",
        f"{config.steps} · {config.num_generations} · {config.max_completion_length}",
    )
    table.add_row(
        "examples · lr · lora rank",
        f"{config.n_examples} · {config.learning_rate} · {config.lora_rank}",
    )
    console.print(Panel(table, title="[bold]ARCA · GRPO smoke test[/bold]", border_style="cyan"))


class ProgressCallback:
    """Prints one compact line per logging step and keeps the history for later plotting."""

    def __init__(self):
        from transformers import TrainerCallback

        callback = self
        self.history: list[dict] = []

        class _Inner(TrainerCallback):
            def on_log(self, args, state, control, logs=None, **kwargs):
                if logs is None:
                    return
                row = {"step": state.global_step, **logs}
                mem = gpu_memory_gb()
                if mem is not None:
                    row["gpu_mem_gb"] = mem
                callback.history.append(row)
                callback.print_row(row)

        self.inner = _Inner()

    @staticmethod
    def _get(row: dict, *candidates: str) -> str:
        for key in candidates:
            if key in row and row[key] is not None:
                return f"{row[key]:.3f}"
        return "  -  "

    def print_row(self, row: dict) -> None:
        if "step" not in row or "loss" not in row and "reward" not in row:
            return
        console.print(
            f"step {row['step']:>4} │ "
            f"reward {self._get(row, 'reward')} │ "
            f"format {self._get(row, 'rewards/format_reward/mean')} │ "
            f"accuracy {self._get(row, 'rewards/accuracy_reward/mean')} │ "
            f"len {self._get(row, 'completions/mean_length')} │ "
            f"clipped {self._get(row, 'completions/clipped_ratio')} │ "
            f"gpu {self._get(row, 'gpu_mem_gb')} GB",
            soft_wrap=True,
        )


def build_trainer(config: SmokeConfig, dataset, device: str, dtype):
    from peft import LoraConfig
    from trl import GRPOConfig, GRPOTrainer

    from rlm.rewards import accuracy_reward, format_reward

    peft_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=2 * config.lora_rank,
        lora_dropout=0.0,
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )
    args = GRPOConfig(
        output_dir=config.output_dir,
        max_steps=config.steps,
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.num_generations,
        gradient_accumulation_steps=1,
        num_generations=config.num_generations,
        max_completion_length=config.max_completion_length,
        temperature=1.0,
        beta=0.0,  # no KL penalty, as in TRL's default and most recent practice
        epsilon=0.2,  # clipping range from the slides
        bf16=device == "cuda",
        gradient_checkpointing=device == "cuda",
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        seed=config.seed,
        log_completions=False,
        model_init_kwargs={"dtype": dtype},
        use_cpu=device == "cpu",
    )
    trainer = GRPOTrainer(
        model=config.model,
        reward_funcs=[format_reward, accuracy_reward],
        args=args,
        train_dataset=dataset,
        peft_config=peft_config,
    )
    return trainer


def generate(
    model, tokenizer, prompt_messages: list[dict], max_new_tokens: int, device: str
) -> str:
    import torch

    text = tokenizer.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)


def _verdict(format_ok: bool, correct: bool) -> str:
    return f"format {'✓' if format_ok else '✗'} · correct {'✓' if correct else '✗'}"


def showcase(trainer, config: SmokeConfig, device: str) -> list[dict]:
    """Same questions, base model vs trained adapter, side by side."""
    from rlm.data import load_gsm8k
    from rlm.rewards import extract_answer, has_valid_format, numbers_match

    test = load_gsm8k("test", n_examples=config.n_showcase, seed=config.seed)
    model, tokenizer = trainer.model, trainer.processing_class
    model.eval()
    results = []
    for example in test:
        with model.disable_adapter():
            before = generate(
                model, tokenizer, example["prompt"], config.max_completion_length, device
            )
        after = generate(model, tokenizer, example["prompt"], config.max_completion_length, device)
        row = {
            "question": example["prompt"][-1]["content"],
            "expected": example["answer"],
            "before": before,
            "after": after,
            "before_ok": numbers_match(extract_answer(before), example["answer"]),
            "after_ok": numbers_match(extract_answer(after), example["answer"]),
            "before_format": has_valid_format(before),
            "after_format": has_valid_format(after),
        }
        results.append(row)
        table = Table(
            title=f"Q: {row['question'][:110]}… (expected {row['expected']})", show_lines=True
        )
        table.add_column("base model", ratio=1)
        table.add_column("after GRPO", ratio=1)
        table.add_row(before[:900], after[:900])
        table.add_row(
            _verdict(row["before_format"], row["before_ok"]),
            _verdict(row["after_format"], row["after_ok"]),
        )
        console.print(table)
    return results


def main() -> int:
    config = parse_args()
    device, dtype = pick_device_and_dtype(force_cpu=config.dry_run)
    print_environment(config, device)
    if device != "cuda" and not config.dry_run:
        console.print(
            "[yellow]No CUDA GPU detected. This will be very slow; "
            "use --dry-run to check the install or run on the DGX.[/yellow]"
        )

    from rlm.data import load_gsm8k

    console.print("[bold]1/4[/bold] Loading GSM8K…")
    dataset = load_gsm8k("train", n_examples=config.n_examples, seed=config.seed)
    console.print(f"      {len(dataset)} problems. Example answer column: {dataset[0]['answer']!r}")

    console.print("[bold]2/4[/bold] Loading model and building the GRPO trainer…")
    trainer = build_trainer(config, dataset, device, dtype)
    # Replace the default transformers printers (tqdm bar + raw dict dump) with our one-line log.
    from transformers.trainer_callback import PrinterCallback
    from transformers.trainer_callback import ProgressCallback as HFProgressCallback

    trainer.remove_callback(PrinterCallback)
    trainer.remove_callback(HFProgressCallback)
    progress = ProgressCallback()
    trainer.add_callback(progress.inner)
    trainable, total = trainer.model.get_nb_trainable_parameters()
    share = 100 * trainable / total
    console.print(
        f"      trainable params: {trainable / 1e6:.2f}M of {total / 1e6:.0f}M ({share:.2f}%)"
    )

    console.print(
        f"[bold]3/4[/bold] Training for {config.steps} steps (group size {config.num_generations})…"
    )
    start = time.perf_counter()
    trainer.train()
    elapsed = time.perf_counter() - start
    mem = gpu_memory_gb()
    console.print(
        f"      done in {elapsed / 60:.1f} min"
        + (f" · peak GPU memory {mem:.1f} GB" if mem is not None else "")
    )

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(output_dir)
    (output_dir / "smoke_history.json").write_text(json.dumps(progress.history, indent=2))
    console.print(f"      adapter and training history saved to {output_dir}/")

    console.print("[bold]4/4[/bold] Before vs after on held-out problems…")
    results = showcase(trainer, config, device)
    (output_dir / "smoke_showcase.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False)
    )

    first = next((r for r in progress.history if "reward" in r), {})
    last = next((r for r in reversed(progress.history) if "reward" in r), {})
    summary = Table(title="Summary", show_header=True, header_style="bold")
    summary.add_column("metric")
    summary.add_column("first step")
    summary.add_column("last step")
    for label, key in [
        ("mean reward", "reward"),
        ("format reward", "rewards/format_reward/mean"),
        ("accuracy reward", "rewards/accuracy_reward/mean"),
        ("completion length", "completions/mean_length"),
    ]:
        summary.add_row(
            label, f"{first.get(key, float('nan')):.3f}", f"{last.get(key, float('nan')):.3f}"
        )
    console.print(summary)
    console.print(json.dumps(asdict(config), indent=2))
    console.print(
        Panel.fit(
            "[bold green]Smoke test finished.[/bold green] "
            "If the format reward went up, the loop works.\nNext stop: rlm/README.md",
            border_style="green",
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
