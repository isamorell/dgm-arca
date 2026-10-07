"""Phase 1, step 2: distil reasoning traces from a teacher model and keep only the verified ones.

This is what Sky-T1, OpenThoughts and DeepSeek's cold start have in common: a strong model
writes solutions with visible reasoning, a verifier throws away the wrong ones, and what
survives becomes SFT data. Here the teacher is Qwen3-4B in thinking mode and the student
(``rlm/train_sft.py``) is Qwen3-0.6B.

The pipeline has two separate stages so that the expensive one never has to be repeated:

1. **Generation** (needs the GPU). For every problem the teacher writes ``--samples``
   solutions. Every raw generation is appended to ``--raw-output`` (a JSONL under
   ``outputs/``, git-ignored) *as soon as its batch is finished*. If the DGX session dies,
   re-running the same command resumes where it stopped.
2. **Filtering** (CPU only, seconds). Each raw generation is turned into the canonical
   ``<think>…</think><answer>…</answer>`` text, judged by the ``MedicationVerifier`` (value
   *and* unit) plus a few quality filters, and given exactly one verdict (``accepted`` or a
   rejection reason). Verified traces are written to ``--output``, the small file that
   ``rlm.train_sft`` reads and that is committed to git. ``--filter-only`` runs just this
   stage, so you can change a threshold without generating anything again.

Run (pilot first, then the full run; see docs/EXPERIMENTS.md for the numbers)::

    uv run python -m rlm.distill --data rlm/data/train.jsonl --per-family 8 --samples 4 \
        --raw-output outputs/distill/raw_pilot.jsonl --output outputs/distill/pilot_traces.jsonl

    uv run python -m rlm.distill --data rlm/data/train.jsonl --samples 4 \
        --raw-output outputs/distill/raw.jsonl --output rlm/data/sft_traces.jsonl

Design decisions (the ones to defend orally):

* **Teacher sampling**: temperature 0.6, top-p 0.95, top-k 20, never greedy. These are the
  values the Qwen3 model card prescribes for thinking mode; greedy decoding makes the
  model loop and, with ``--samples`` > 1, would give the same trace four times.
* **Spanish**: the statements are in Spanish, so the teacher is asked to reason in Spanish
  and traces that are not mostly Spanish are discarded. Asking is not enough (Qwen3 keeps
  thinking in English), so its reasoning is also *started* in Spanish with a short prefill
  sentence (``THINK_PREFILL``). The student learns to reason in the language of its users.
* **Canonical format**: Qwen3 writes its reasoning, then ``</think>``, then an answer text.
  We keep only the reasoning and the content of the final ``<answer>`` block, so every
  training example has exactly the structure that the format reward of GRPO will check.
* **Unit-aware verification**: a right number in the wrong unit is rejected (``mg`` instead
  of ``mcg``, ``mL`` instead of ``mL/h``). In nursing that is a different dose.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path

from rlm.data import R1_ZERO_SYSTEM_PROMPT
from rlm.rewards import extract_answer, has_valid_format
from rlm.verifier import MedicationVerifier

# --------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------

# Sampling recommended by the Qwen3-4B model card for thinking mode (never greedy).
TEACHER_SAMPLING = {"temperature": 0.6, "top_p": 0.95, "top_k": 20}

# Added to the end of the question *only for the teacher*. The student is trained with the
# plain question and the usual system prompt (``rlm.data.build_prompt``), so after SFT it
# reasons in Spanish without needing this hint.
SPANISH_HINT = (
    "\n\nRazona paso a paso en español, escribiendo las unidades en cada paso. "
    "Al final responde con <answer>valor unidad</answer> (por ejemplo "
    "<answer>12,5 mL/h</answer>), con el redondeo que pida el enunciado."
)

# Qwen3 thinks in English even when the question and the instructions are in Spanish (seen in
# the pilot: correct answers, English reasoning). The hint above is not enough, so the
# teacher's reasoning is *started for it* in Spanish: the prompt ends with an open ``<think>``
# block and this opening sentence, and the model continues from there. The sentence is kept
# as the first words of the trace, so the student sees it too. ``--no-prefill`` disables it.
THINK_PREFILL = "<think>\nVale, voy a resolverlo paso a paso. "

# Reasons a generation can be rejected, in the order they are checked.
REASON_ACCEPTED = "accepted"
REASON_TRUNCATED = "truncated"  # hit max_new_tokens before closing </think>
REASON_NO_ANSWER = "no_answer"  # closed </think> but no <answer> / \boxed{}
REASON_BAD_FORMAT = "bad_format"  # canonical text breaks the think/answer structure
REASON_WRONG_VALUE = "wrong_value"
REASON_WRONG_UNIT = "wrong_unit"
REASON_TOO_SHORT = "too_short"  # correct, but almost no reasoning (lucky guess?)
REASON_NOT_SPANISH = "not_spanish"
REASON_TOO_LONG = "too_long"  # correct, but would not fit in the student's context
REASONS = (
    REASON_ACCEPTED,
    REASON_TRUNCATED,
    REASON_NO_ANSWER,
    REASON_BAD_FORMAT,
    REASON_WRONG_VALUE,
    REASON_WRONG_UNIT,
    REASON_TOO_SHORT,
    REASON_NOT_SPANISH,
    REASON_TOO_LONG,
)

# Small stop-word lists for the language check. Units and drug names are deliberately left
# out: they are the same in both languages and would only add noise.
SPANISH_WORDS = frozenset(
    "de la el en que los las por con para una un se es del al y o su sus como más lo "
    "este esta son pero si cada entonces tanto debe hay también donde cuántos cuánto "
    "tenemos necesitamos tiene pasar resultado dosis volumen vial paciente hora horas".split()
)
ENGLISH_WORDS = frozenset(
    "the of and to is in that for with we this so then need are be it as at on by from "
    "which each total wait let first now since because first so check should also have "
    "has will need dose volume patient hour hours result".split()
)
WORD_PATTERN = re.compile(r"[a-záéíóúüñ]+")
ANSWER_TAG_PATTERN = re.compile(r"</?answer>")


# --------------------------------------------------------------------------------------
# Problems
# --------------------------------------------------------------------------------------


def load_problems(path: str | Path) -> list[dict]:
    """Read the domain JSONL keeping every column (the verifier needs the unit fields)."""
    problems = []
    with Path(path).open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            row = json.loads(line)
            if "question" not in row or "answer" not in row:
                raise ValueError(f"{path}:{index + 1} needs 'question' and 'answer' fields")
            row["answer"] = str(row["answer"])
            problems.append(row)
    return problems


def problem_level(problem: dict) -> str:
    """Difficulty level (``"1"``/``"2"``/``"3"``) stored inside the ``branches`` JSON string."""
    branches = problem.get("branches")
    if isinstance(branches, str):
        try:
            branches = json.loads(branches)
        except json.JSONDecodeError:
            branches = None
    if isinstance(branches, dict):
        return str(branches.get("level", "?"))
    return "?"


def select_problems(
    problems: Sequence[dict], per_family: int | None = None, limit: int | None = None, seed: int = 0
) -> list[tuple[int, dict]]:
    """Return ``(id, problem)`` pairs, where ``id`` is the row index in the full file.

    ``per_family`` takes that many random problems of every family (a balanced pilot);
    ``limit`` truncates the result. The ids refer to the *full* file, so a pilot and the
    full run never confuse two different problems.
    """
    indexed = list(enumerate(problems))
    if per_family is not None:
        rng = random.Random(seed)
        by_family: dict[str, list[tuple[int, dict]]] = defaultdict(list)
        for item in indexed:
            by_family[item[1].get("family", "?")].append(item)
        indexed = []
        for family in sorted(by_family):
            group = by_family[family]
            rng.shuffle(group)
            indexed.extend(sorted(group[:per_family], key=lambda item: item[0]))
    if limit is not None:
        indexed = indexed[:limit]
    return indexed


def teacher_messages(question: str) -> list[dict]:
    """Chat messages for the teacher: the R1-Zero system prompt plus the Spanish hint."""
    return [
        {"role": "system", "content": R1_ZERO_SYSTEM_PROMPT},
        {"role": "user", "content": question + SPANISH_HINT},
    ]


# --------------------------------------------------------------------------------------
# From raw teacher text to a canonical trace
# --------------------------------------------------------------------------------------


def split_teacher_output(raw: str) -> tuple[str | None, str | None]:
    """Split Qwen3's output into ``(thinking, final_text)``.

    In thinking mode the model writes its reasoning, closes it with ``</think>`` and then
    writes the visible answer. The opening ``<think>`` may or may not be part of the
    decoded text depending on the chat template, so it is stripped if present. Returns
    ``(None, None)`` when ``</think>`` never appears (the generation was cut off).
    """
    text = raw.strip()
    if text.startswith("<think>"):
        text = text[len("<think>") :]
    if "</think>" not in text:
        return None, None
    thinking, _, final = text.partition("</think>")
    return thinking.strip(), final.strip()


def last_answer(final_text: str) -> str | None:
    """Content of the last ``<answer>`` block of the visible answer, or ``\\boxed{}`` fallback."""
    return extract_answer(final_text)


def canonical_trace(thinking: str, answer: str) -> str:
    """The exact text the student must learn to write."""
    return f"<think>\n{thinking.strip()}\n</think>\n<answer>{answer.strip()}</answer>"


def spanish_ratio(text: str) -> tuple[float, int]:
    """Fraction of Spanish among the recognised stop words of ``text``, and how many there were.

    Returns ``(ratio, n_recognised)``. With very few recognised words the ratio means nothing,
    so callers also look at the count.
    """
    words = WORD_PATTERN.findall(text.lower())
    spanish = sum(1 for word in words if word in SPANISH_WORDS)
    english = sum(1 for word in words if word in ENGLISH_WORDS)
    total = spanish + english
    return (spanish / total if total else 0.0), total


def judge_generation(
    record: dict,
    verifier: MedicationVerifier,
    *,
    min_think_words: int = 30,
    min_spanish_ratio: float = 0.6,
    max_trace_tokens: int = 1800,
) -> dict:
    """Give one raw generation its verdict. Returns the record plus the judging fields.

    Added keys: ``verdict`` (``accepted`` or a rejection reason, see ``REASONS``), ``trace``
    (canonical text, ``None`` if it could not be built), ``predicted`` and ``detail``.
    Correctness is checked *before* the quality filters, so the reasons ``too_short``,
    ``not_spanish`` and ``too_long`` only ever apply to traces that were correct.
    """
    out = dict(record)
    out.update(verdict=REASON_TRUNCATED, trace=None, predicted=None, detail="")

    thinking, final = split_teacher_output(record["raw"])
    if thinking is None or not record.get("finished", True):
        return out

    predicted = last_answer(final)
    if predicted is None:
        out["verdict"] = REASON_NO_ANSWER
        return out
    out["predicted"] = predicted

    trace = canonical_trace(thinking, predicted)
    out["trace"] = trace
    # FORMAT_PATTERN already forbids a second <think> block; a stray <answer> tag inside the
    # reasoning would pass it, but the student must never learn to write one there.
    if not has_valid_format(trace) or ANSWER_TAG_PATTERN.search(thinking):
        out["verdict"] = REASON_BAD_FORMAT
        out["detail"] = "reasoning contains a <think>/<answer> tag"
        return out

    result = verifier.verify(
        trace, record["answer"], record.get("answer_unit"), record.get("answer_unit_aliases") or ()
    )
    if not result.is_correct:
        is_unit_problem = "unit" in result.detail
        out["verdict"] = REASON_WRONG_UNIT if is_unit_problem else REASON_WRONG_VALUE
        out["detail"] = result.detail
        return out

    if len(thinking.split()) < min_think_words:
        out["verdict"] = REASON_TOO_SHORT
        return out
    ratio, recognised = spanish_ratio(thinking)
    if recognised < 5 or ratio < min_spanish_ratio:
        out["verdict"] = REASON_NOT_SPANISH
        out["detail"] = f"spanish_ratio={ratio:.2f} over {recognised} stop words"
        return out
    if record.get("n_new_tokens", 0) > max_trace_tokens:
        out["verdict"] = REASON_TOO_LONG
        out["detail"] = f"{record['n_new_tokens']} tokens > {max_trace_tokens}"
        return out

    out["verdict"] = REASON_ACCEPTED
    return out


def select_traces(
    judged: Iterable[dict], keep_per_problem: int = 2, strategy: str = "shortest", seed: int = 0
) -> list[dict]:
    """Keep at most ``keep_per_problem`` accepted traces per problem.

    Several traces per problem give the student diversity, but four copies of the same easy
    problem would over-weight it. ``shortest`` prefers concise reasoning (a 0.6B model
    copies short, clean reasoning better than rambling "wait, let me re-check" loops);
    ``random`` is the unbiased alternative.
    """
    by_problem: dict[int, list[dict]] = defaultdict(list)
    for row in judged:
        if row["verdict"] == REASON_ACCEPTED:
            by_problem[row["id"]].append(row)
    rng = random.Random(seed)
    chosen = []
    for problem_id in sorted(by_problem):
        group = by_problem[problem_id]
        if strategy == "shortest":
            group = sorted(group, key=lambda row: (row["n_new_tokens"], row["sample"]))
        else:
            rng.shuffle(group)
        chosen.extend(group[:keep_per_problem])
    return chosen


def to_sft_row(row: dict) -> dict:
    """The compact record written to ``rlm/data/sft_traces.jsonl`` (what ``train_sft`` reads)."""
    return {
        "id": row["id"],
        "question": row["question"],
        "answer": row["answer"],
        "trace": row["trace"],
        "verified": True,
        "teacher": row["teacher"],
        "family": row.get("family"),
        "level": row.get("level"),
        "n_new_tokens": row["n_new_tokens"],
    }


# --------------------------------------------------------------------------------------
# Statistics for docs/EXPERIMENTS.md
# --------------------------------------------------------------------------------------


def summarize(judged: Sequence[dict], n_problems: int | None = None) -> dict:
    """Counts needed for the report: reasons, acceptance per family/level, problem coverage."""
    reasons = Counter(row["verdict"] for row in judged)
    total = len(judged)
    accepted = reasons.get(REASON_ACCEPTED, 0)

    def rate_by(key: str) -> dict:
        totals: Counter = Counter()
        ok: Counter = Counter()
        for row in judged:
            totals[str(row.get(key))] += 1
            ok[str(row.get(key))] += row["verdict"] == REASON_ACCEPTED
        return {k: {"generated": totals[k], "accepted": ok[k]} for k in sorted(totals)}

    solved = {row["id"] for row in judged if row["verdict"] == REASON_ACCEPTED}
    seen = {row["id"] for row in judged}
    tokens = [row["n_new_tokens"] for row in judged if row["verdict"] == REASON_ACCEPTED]
    return {
        "generated": total,
        "accepted": accepted,
        "acceptance_rate": accepted / total if total else 0.0,
        "reasons": {reason: reasons.get(reason, 0) for reason in REASONS},
        "problems": n_problems if n_problems is not None else len(seen),
        "problems_with_a_verified_trace": len(solved),
        "by_family": rate_by("family"),
        "by_level": rate_by("level"),
        "mean_tokens_accepted": sum(tokens) / len(tokens) if tokens else 0.0,
    }


def format_summary(summary: dict, n_selected: int | None = None) -> str:
    """Markdown ready to paste into docs/EXPERIMENTS.md."""
    lines = [
        f"- Generaciones: {summary['generated']}; verificadas: {summary['accepted']} "
        f"({100 * summary['acceptance_rate']:.1f} %).",
        f"- Problemas con al menos una traza verificada: "
        f"{summary['problems_with_a_verified_trace']}/{summary['problems']}.",
        f"- Longitud media de las trazas aceptadas: {summary['mean_tokens_accepted']:.0f} tokens.",
    ]
    if n_selected is not None:
        lines.append(f"- Trazas que pasan al SFT tras limitar por problema: {n_selected}.")
    lines += ["", "| Motivo | Generaciones |", "|---|---|"]
    lines += [f"| {reason} | {count} |" for reason, count in summary["reasons"].items()]
    for title, key in (("Familia", "by_family"), ("Nivel", "by_level")):
        lines += ["", f"| {title} | Generadas | Verificadas | Tasa |", "|---|---|---|---|"]
        for name, counts in summary[key].items():
            rate = 100 * counts["accepted"] / max(counts["generated"], 1)
            lines.append(
                f"| {name} | {counts['generated']} | {counts['accepted']} | {rate:.1f} % |"
            )
    return "\n".join(lines)


# --------------------------------------------------------------------------------------
# JSONL helpers
# --------------------------------------------------------------------------------------


def read_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def append_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------------------
# Generation (GPU)
# --------------------------------------------------------------------------------------


def _generate_group(model, tokenizer, prompts: list[str], n: int, max_new_tokens: int):
    """Sample ``n`` completions for each prompt. Returns ``[(text, n_tokens, finished)]``.

    The result is prompt-major: the ``n`` samples of prompt 0 come first, then those of
    prompt 1, and so on (``num_return_sequences`` orders them like that). On a CUDA
    out-of-memory error the group is split (first the prompts, then the samples) and
    retried, so a long batch never kills a 24-hour run.
    """
    import torch

    output = None
    try:
        batch = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            output = model.generate(
                **batch,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                num_return_sequences=n,
                **TEACHER_SAMPLING,
            )
    except torch.cuda.OutOfMemoryError:
        pass  # handled below, outside the ``except`` block, so the failed tensors can be freed

    if output is None:
        torch.cuda.empty_cache()
        if len(prompts) > 1:
            half = len(prompts) // 2
            print(f"  out of memory: splitting {len(prompts)} prompts", flush=True)
            return _generate_group(model, tokenizer, prompts[:half], n, max_new_tokens) + (
                _generate_group(model, tokenizer, prompts[half:], n, max_new_tokens)
            )
        if n > 1:
            # One prompt, many samples: sample in two chunks (a single prompt has no
            # ordering problem, so concatenating the chunks is safe).
            print(f"  out of memory: splitting {n} samples of one prompt", flush=True)
            first = _generate_group(model, tokenizer, prompts, n // 2, max_new_tokens)
            second = _generate_group(model, tokenizer, prompts, n - n // 2, max_new_tokens)
            return first + second
        raise RuntimeError("Out of GPU memory with a single sequence; lower --max-new-tokens")

    generated = output[:, batch["input_ids"].shape[1] :]
    eos_ids = model.generation_config.eos_token_id
    eos_ids = set(eos_ids if isinstance(eos_ids, list) else [eos_ids])
    pad_id = tokenizer.pad_token_id
    results = []
    for ids in generated.tolist():
        content = [token for token in ids if token != pad_id]
        # Padding after the content means the sequence stopped early on its own; a sequence
        # that used every token is finished only if its last token is an end-of-turn token.
        finished = len(content) < len(ids) or (bool(content) and content[-1] in eos_ids)
        text = tokenizer.decode(content, skip_special_tokens=True)
        results.append((text, len(content), finished))
    return results


def generate_traces(
    problems: Sequence[tuple[int, dict]],
    teacher: str,
    samples: int,
    max_new_tokens: int,
    raw_output: str | Path,
    batch_size: int = 2,
    seed: int = 0,
    prefill: bool = True,
) -> int:
    """Generate ``samples`` teacher completions per problem and append them to ``raw_output``.

    Problems whose id is already in ``raw_output`` are skipped, which makes the function
    resumable. Each batch is written to disk right after it finishes. With ``prefill`` the
    prompt ends with ``THINK_PREFILL`` (see its comment) and the stored ``raw`` text starts
    with that prefix, so downstream code sees one complete ``<think>…`` text either way.
    Returns the number of problems generated in this call.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    done = {row["id"] for row in read_jsonl(raw_output)}
    pending = [(pid, problem) for pid, problem in problems if pid not in done]
    print(f"{len(done)} problems already in {raw_output}; {len(pending)} left to generate")
    if not pending:
        return 0

    tokenizer = AutoTokenizer.from_pretrained(teacher)
    tokenizer.padding_side = "left"  # generation continues from the right edge of the prompt
    model = AutoModelForCausalLM.from_pretrained(teacher, dtype=torch.bfloat16)
    model.to("cuda" if torch.cuda.is_available() else "cpu").eval()

    run_started = time.monotonic()
    for start in range(0, len(pending), batch_size):
        group = pending[start : start + batch_size]
        torch.manual_seed(seed + start)
        prompts = [
            tokenizer.apply_chat_template(
                teacher_messages(problem["question"]),
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
            + (THINK_PREFILL if prefill else "")
            for _, problem in group
        ]
        started = time.monotonic()
        results = _generate_group(model, tokenizer, prompts, samples, max_new_tokens)
        elapsed = time.monotonic() - started
        rows = []
        for position, (pid, problem) in enumerate(group):
            for sample in range(samples):
                text, n_tokens, finished = results[position * samples + sample]
                rows.append(
                    {
                        "id": pid,
                        "sample": sample,
                        "question": problem["question"],
                        "answer": problem["answer"],
                        "answer_unit": problem.get("answer_unit"),
                        "answer_unit_aliases": problem.get("answer_unit_aliases") or [],
                        "family": problem.get("family"),
                        "level": problem_level(problem),
                        "raw": (THINK_PREFILL if prefill else "") + text,
                        "prefill": prefill,
                        "n_new_tokens": n_tokens,
                        "finished": finished,
                        "teacher": teacher,
                    }
                )
        append_jsonl(raw_output, rows)
        done_now = start + len(group)
        total_time = time.monotonic() - run_started
        generated_tokens = sum(row["n_new_tokens"] for row in rows)
        eta_minutes = total_time / done_now * (len(pending) - done_now) / 60
        print(
            f"  [{done_now}/{len(pending)}] problems generated | batch {elapsed:.0f}s, "
            f"{generated_tokens / elapsed:.0f} tok/s (all sequences) | "
            f"{sum(not row['finished'] for row in rows)}/{len(rows)} cut at the token limit | "
            f"ETA {eta_minutes:.0f} min",
            flush=True,
        )
    return len(pending)


# --------------------------------------------------------------------------------------
# Filtering and CLI
# --------------------------------------------------------------------------------------


def filter_raw(raw_rows: Sequence[dict], args: argparse.Namespace) -> tuple[list[dict], list[dict]]:
    """Judge every raw generation and select the traces that go to the SFT file."""
    verifier = MedicationVerifier()
    judged = [
        judge_generation(
            row,
            verifier,
            min_think_words=args.min_think_words,
            min_spanish_ratio=args.min_spanish_ratio,
            max_trace_tokens=args.max_trace_tokens,
        )
        for row in raw_rows
    ]
    selected = select_traces(judged, args.keep_per_problem, args.selection, args.seed)
    return judged, selected


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data", default="rlm/data/train.jsonl", help="domain JSONL")
    parser.add_argument("--teacher", default="Qwen/Qwen3-4B")
    parser.add_argument("--samples", type=int, default=4, help="traces per problem")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=2, help="problems per generate() call")
    parser.add_argument("--per-family", type=int, default=None, help="pilot: N problems per family")
    parser.add_argument("--limit", type=int, default=None, help="use only the first N problems")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--no-prefill",
        dest="prefill",
        action="store_false",
        help="do not start the teacher's reasoning with a Spanish opening sentence",
    )
    parser.add_argument("--raw-output", default="outputs/distill/raw.jsonl")
    parser.add_argument("--output", default="rlm/data/sft_traces.jsonl")
    parser.add_argument("--summary", default=None, help="markdown report (default: next to raw)")
    parser.add_argument("--filter-only", action="store_true", help="skip generation, re-filter raw")
    parser.add_argument("--keep-per-problem", type=int, default=2)
    parser.add_argument("--selection", choices=["shortest", "random"], default="shortest")
    parser.add_argument("--min-think-words", type=int, default=30)
    parser.add_argument("--min-spanish-ratio", type=float, default=0.6)
    parser.add_argument(
        "--max-trace-tokens",
        type=int,
        default=1800,
        help="reject longer traces: the student's max_length is 2048 and the prompt takes ~200",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    problems = select_problems(load_problems(args.data), args.per_family, args.limit, args.seed)
    wanted_ids = {pid for pid, _ in problems}
    print(f"{len(problems)} problems selected from {args.data}")

    if not args.filter_only:
        generate_traces(
            problems,
            args.teacher,
            args.samples,
            args.max_new_tokens,
            args.raw_output,
            args.batch_size,
            args.seed,
            args.prefill,
        )

    raw_rows = [row for row in read_jsonl(args.raw_output) if row["id"] in wanted_ids]
    if not raw_rows:
        raise SystemExit(f"No raw generations in {args.raw_output}; run without --filter-only.")
    judged, selected = filter_raw(raw_rows, args)

    write_jsonl(args.output, (to_sft_row(row) for row in selected))
    judged_path = Path(args.raw_output).with_name(Path(args.raw_output).stem + "_judged.jsonl")
    write_jsonl(judged_path, judged)

    summary = summarize(judged, n_problems=len(problems))
    report = format_summary(summary, n_selected=len(selected))
    summary_path = Path(args.summary or Path(args.raw_output).with_suffix(".summary.md"))
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(report + "\n", encoding="utf-8")
    print(report)
    print(f"\n{len(selected)} traces -> {args.output}\njudged dump -> {judged_path}")


if __name__ == "__main__":
    main()
