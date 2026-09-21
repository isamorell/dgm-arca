# Datos de la fase 1

Aquí van vuestros problemas. Formato JSONL, una línea por problema, con al menos:

```json
{"question": "Un depósito pierde 18 litros y queda a la mitad...", "answer": "120"}
```

Podéis añadir los campos que necesite vuestro verificador (tests, tolerancia, esquema...):
llegan a las funciones de recompensa como argumentos con nombre.

Ficheros que esperamos ver:

- `train.jsonl`: problemas para SFT y GRPO.
- `test.jsonl`: problemas que el modelo no ha visto nunca, para `rlm/evaluate.py`.
- `test_ood.jsonl`: problemas de una familia que no aparece en `train`, para el experimento de
  generalización.
- `sft_traces.jsonl`: trazas del profesor ya verificadas, generadas con `rlm/distill.py`.

## ¿De dónde salen estos problemas?

**No se escriben a mano.** Hay seis estrategias, explicadas con tamaños y ejemplos en
[`docs/datasets.md`](../../docs/datasets.md). La más habitual será un generador programático,
donde la implementación de referencia es a la vez el verificador. Tenéis uno completo y
ejecutable en [`rlm/generate_problems.py`](../generate_problems.py):

```bash
uv run python -m rlm.generate_problems --n 800 --split train --out rlm/data/train.jsonl
uv run python -m rlm.generate_problems --n 200 --split test  --out rlm/data/test.jsonl
uv run python -m rlm.generate_problems --n 100 --split ood   --out rlm/data/test_ood.jsonl
```

Recordad que las **trazas de razonamiento** tampoco se escriben a mano: las genera un modelo
profesor y las filtra vuestro verificador (`rlm/distill.py`). Lo que tenéis que conseguir son
los pares (enunciado, respuesta correcta).

Los ficheros grandes no se suben al repositorio: dejad el script que los regenera. Si vuestros
problemas vienen de un generador, con el script y la semilla basta.
