import json
from collections import Counter, defaultdict
from rlm.data import load_domain_dataset
from rlm.verifier import MedicationVerifier

res = json.load(open("reports/phase1_eval.json"))
ds = load_domain_dataset("rlm/data/test.jsonl")
meta = {ex["prompt"][-1]["content"]: ex for ex in ds}
mv = MedicationVerifier()


def kind(r):
    ex = meta[r["question"]]
    if r["is_correct"]:
        return "ok"
    if r["predicted"] is None:
        return "sin_answer"
    if mv.is_correct(r["predicted"], ex["answer"]):
        return "valor_ok_unidad_mal"
    return "valor_mal"

# analisis modelo base
for m, d in res.items():
    rows = d["rows"]
    n = len(rows)
    solo_valor = sum(mv.is_correct(r["predicted"], r["expected_value"]) for r in rows) / n
    con_formato = sum(r["has_valid_format"] for r in rows) / n
    sin_answer = sum(r["predicted"] is None for r in rows) / n
    print(f"{m:<12} valor_ok={solo_valor:.3f} | formato={con_formato:.3f} | sin_answer={sin_answer:.3f} | pass@1={d['pass@1']:.3f}")

# pass@1 por familia
fams = sorted({meta[r["question"]]["family"] for r in res["sft"]["rows"]})
print(f"{'family':<14}{'n':>4}" + "".join(f"{m:>14}" for m in res))
for f in fams:
    line, n = "", 0
    for m, d in res.items():
        rows = [r for r in d["rows"] if meta[r["question"]]["family"] == f]
        n = len(rows)
        line += f"{sum(r['is_correct'] for r in rows) / n:>14.2f}"
    print(f"{f:<14}{n:>4}{line}")

# tipo de fallo por modelo
print("\nTipos de resultado:")
for m, d in res.items():
    print(m, dict(Counter(kind(r) for r in d["rows"])))

# fallos de grpo_sin_kl por familia y tipo
print("\nFallos de grpo_sin_kl:")
fail = defaultdict(Counter)
for r in res["grpo_sin_kl"]["rows"]:
    if not r["is_correct"]:
        fail[meta[r["question"]]["family"]][kind(r)] += 1
for f, c in fail.items():
    print(f, dict(c))
