# `smoke/` — La primera sesión en la DGX

Antes de que dependáis de la DGX para la fase 1, vamos a comprobar que todo funciona: que el
contenedor ve la GPU, que las librerías están bien instaladas y que un entrenamiento GRPO de
verdad, aunque sea en miniatura, arranca y aprende algo. Eso es lo que hacen los dos scripts
de esta carpeta. No son parte de la entrega; son vuestra red de seguridad.

## `check_gpu.py`

Imprime lo que el código va a ver: driver, versión de CUDA, GPUs y su memoria, versiones de
torch, transformers, trl y peft, y hace una multiplicación de matrices en la GPU para
comprobar que calcula de verdad. Devuelve 0 si hay una GPU utilizable y 1 si no.

```bash
uv run arca-check-gpu                     # entorno local
docker compose run --rm check-gpu         # dentro del contenedor
```

## `smoke_grpo.py`

Un entrenamiento GRPO real, pequeño, sobre GSM8K, con las mismas piezas que usaréis en la
fase 1: el prompt de sistema de R1-Zero, las recompensas de formato y exactitud de
`rlm/rewards.py`, LoRA y el `GRPOTrainer` de TRL. Paso a paso:

1. Muestra el entorno (dispositivo, memoria, versiones).
2. Carga `Qwen/Qwen3-0.6B` con un adaptador LoRA de rango 16.
3. Coge 256 problemas de GSM8K y los envuelve con el prompt de `<think>`/`<answer>`.
4. Entrena 40 pasos con grupos de 8 respuestas y hasta 384 tokens por respuesta.
5. Imprime en cada paso la recompensa media, la de formato, la de exactitud, la longitud de
   las respuestas, la fracción truncada y la memoria de GPU.
6. Guarda el adaptador en `rlm/weights/smoke_lora/` junto con el histórico de métricas en
   JSON (para que lo pintéis) y enseña, para tres problemas del conjunto de test, la
   respuesta del modelo base y la del modelo entrenado, lado a lado.

```bash
uv run arca-smoke                                      # 10-15 min en la GPU de 16 GB de la DGX
uv run arca-smoke --num-generations 4                  # si la memoria se queda justa
uv run arca-smoke --steps 80 --num-generations 16      # más largo, más claro (GPU grande)
uv run arca-smoke --dry-run                            # sin GPU: modelo diminuto, 2 pasos, CPU
docker compose run --rm smoke                          # en una máquina con Docker y GPU
```

## Qué deberíais ver

Al principio, la recompensa de formato estará cerca de cero: Qwen3 piensa entre `<think>` y
`</think>` por defecto, pero nadie le ha dicho que la respuesta va entre `<answer>` y
`</answer>`. En unas decenas de pasos la recompensa de formato sube claramente, a menudo
hasta cerca de 1. La de exactitud se mueve más despacio, y con 40 pasos y un modelo de 0.6B
no esperéis milagros: lo que queremos ver es que *se mueve*, y que la longitud de las
respuestas cambia a medida que el modelo aprende a cerrar la etiqueta antes de que se le
acabe el presupuesto de tokens.

Si veis eso, el entorno funciona y podéis empezar la fase 1 con tranquilidad. Con la
partición de 16 GB de la DGX, la configuración por defecto (0.6B, 8 generaciones, 384
tokens) debería caber; si la memoria se dispara, bajad `--num-generations` o
`--max-completion-length`. Si no ve la GPU, volved a `check_gpu.py` y a la sección de
problemas de [`docs/dgx.md`](../docs/dgx.md).

## Qué no es

No es la fase 1. Fijaos en lo que le falta para serlo: no hay dataset de vuestro dominio, no
hay verificador propio, no hay SFT previo, no hay tercera recompensa, no hay evaluación en
un conjunto de test, y 40 pasos no son un entrenamiento. Todo eso está descrito en
[`rlm/README.md`](../rlm/README.md). Pero leed este script con calma: es la versión mínima
de `rlm/train_grpo.py`, y entenderlo os ahorra la mitad del camino.
