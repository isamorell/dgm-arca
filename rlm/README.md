# Fase 1 — Razonamiento: de modelo de lenguaje a modelo que piensa

**Endpoint que se evalúa:** `POST /reasoning` · **Clase:** Reasoning language models

## De qué va

En clase vimos que un modelo de razonamiento no tiene nada de mágico: es un modelo entrenado
para generar primero una cadena de pensamiento entre `<think>` y `</think>` y después la
respuesta entre `<answer>` y `</answer>`. Y vimos dos formas de enseñárselo: imitando trazas
de un modelo profesor (SFT) o dejándolo explorar y premiándolo cuando acierta (RLVR con
GRPO). DeepSeek combinó ambas: un arranque en frío con pocos miles de trazas y después RL con
recompensas de formato y exactitud. En esta fase vais a reproducir ese camino, a pequeña
escala, sobre problemas de vuestro dominio.

Antes de nada, corred el smoke test (`smoke/README.md`). Es la versión mínima de lo que vais
a hacer aquí, sobre GSM8K, y entenderlo os ahorra la mitad del trabajo.

## Lo que tendréis que hacer

**1. El dataset verificable.** Unos cientos de problemas de vuestro dominio en
`rlm/data/train.jsonl` y `rlm/data/test.jsonl` (formato en `rlm/data/README.md`), y un
verificador en `rlm/verifier.py` que, dada una respuesta del modelo, diga si es correcta sin
que intervenga nadie. **Nadie anota esto a mano**: en [`docs/datasets.md`](../docs/datasets.md)
tenéis las seis formas de construirlo y cuántos problemas hacen falta de verdad, y en
[`generate_problems.py`](generate_problems.py) un generador completo y ejecutable donde la
implementación de referencia es a la vez el verificador. Os doy el numérico y el de texto exacto; el vuestro puede ejecutar
tests, comparar el resultado de una consulta, validar un esquema, lo que necesite vuestro
problema. Añadid tests en `tests/test_verifier.py` con los casos raros: es donde los
verificadores fallan y donde el modelo aprenderá a haceros trampas si le dejáis. Podéis
mezclar GSM8K o MATH como grupo de control, y os lo recomiendo para comparar, pero el
dataset del dominio es obligatorio.

**2. El arranque en frío (destilación).** Con `rlm/distill.py`, un modelo profesor (Qwen3 en
modo thinking o un destilado de R1) genera varias trazas por problema; el verificador tira
las incorrectas; lo que sobrevive es vuestro dataset de SFT. Apuntad en `EXPERIMENTS.md`
cuántas generasteis, cuántas descartasteis y por qué. Ese porcentaje es la primera medida de
lo difícil que es vuestro dominio.

**3. SFT.** Con `rlm/train_sft.py` entrenáis un adaptador LoRA sobre las trazas verificadas.
Es el mismo `SFTTrainer` de siempre; lo único nuevo es que el objetivo incluye la cadena de
razonamiento.

**4. GRPO.** Con `rlm/train_grpo.py`, partiendo del adaptador de SFT (o del modelo base, si
queréis reproducir el experimento de R1-Zero y ver qué pasa sin arranque en frío). Como
mínimo, recompensa de formato y de exactitud. Y una tercera, `domain_reward`, propia de
vuestro dominio, que tendréis que justificar: idioma de la respuesta, unidades, longitud,
cita de una fuente, respeto a un esquema. Lo que importe a vuestro usuario.

**5. GRPO a mano.** Además de usar TRL, implementáis un paso del algoritmo en
`rlm/grpo_step.py`: ventajas normalizadas por grupo, ratio de políticas, recorte y
penalización KL, tal como aparece en las diapositivas. Los tests de
`tests/test_grpo_step.py` os dicen si lo tenéis bien. No sirve para entrenar a escala; sirve
para que cuando os pregunte qué hace el clipping, me lo expliquéis con vuestro código
delante.

**6. Evaluación.** Con `rlm/evaluate.py`: pass@1 en vuestro conjunto de test para el modelo
base, tras SFT y tras GRPO. Curvas de recompensa y de longitud del razonamiento durante el
entrenamiento (el histórico se guarda en JSON). Y un análisis de al menos cinco fallos
concretos: ¿por qué se equivoca el modelo ahí?

**7. El endpoint.** `rlm/inference.py` ya sabe cargar base + adaptador y separar
pensamiento de respuesta. Solo tenéis que apuntar `ARCA_RLM_ADAPTER` a vuestro adaptador
final y registrar vuestro verificador en `VERIFIERS`.

Una nota sobre recursos: en la DGX tenéis una GPU de 16 GB y sesiones de 24 horas. Modelos de
0.6B a 1.7B con LoRA caben bien; guardad checkpoints (`--save-steps`) y reanudad con
`--resume-from-checkpoint` si la sesión se acaba. Y subid cada adaptador terminado a
Hugging Face Hub o descargadlo: la DGX no tiene copias de seguridad.

## Para ir más allá

Aquí es donde se marca la diferencia en la nota de interpretación. Algunas direcciones:

- Reproducir a pequeña escala "SFT memoriza, RL generaliza": una partición fuera de
  distribución de vuestro dominio, y comparar cómo se degradan SFT y GRPO.
- GRPO con y sin KL (`--beta`), con distintos tamaños de grupo, y explicar qué cambia en las
  curvas y en las respuestas.
- Un modelo juez como recompensa auxiliar para respuestas que no son numéricas, y medir
  cuánto coincide con vuestro verificador.
- Cazar un caso de *reward hacking*: el modelo aprende a engañar al verificador en vez de
  resolver el problema. Os pasará. Documentadlo: es de lo más instructivo de la práctica.

## Cómo se corrige

Llamaré a `POST /reasoning` con problemas de vuestro conjunto de test (y alguno que no habéis
visto) pasando `expected_answer`, y miraré:

- **Implementación (30 %):** la respuesta llega separada en `thinking` y `answer`,
  `has_valid_format` es cierto la mayoría de las veces, y `verifier.is_correct` coincide con
  lo que dice vuestro `evaluate.py`.
- **Completitud (30 %):** dataset con verificador y tests, trazas destiladas con su tasa de
  aceptación, adaptador de SFT, adaptador de GRPO con tres recompensas, `grpo_step.py` con
  los tests en verde, evaluación base/SFT/GRPO con curvas, análisis de fallos.
- **Interpretación (30 %):** las curvas explicadas; qué hizo cada recompensa; por qué elegisteis
  esos hiperparámetros; qué pasó cuando cambiasteis uno; lo que aprendisteis de los fallos; y
  en la defensa, el paso de GRPO explicado sobre vuestro código.
- **Limpieza (10 %):** scripts que corren de principio a fin con los comandos del README,
  sin huecos, con los pesos fuera del repositorio y el script que los regenera.

## Ficheros

| Fichero | Estado | Qué es |
|---|---|---|
| `rewards.py` | hecho | Recompensas de formato y exactitud, extracción de respuestas |
| `verifier.py` | hecho + tu turno | Interfaz `Verifier`, numérico y exacto; el vuestro va aquí |
| `data.py` | hecho | Prompt de sistema de R1-Zero, carga de GSM8K y de vuestro JSONL |
| `distill.py` | tu turno | Generar trazas con el profesor y filtrarlas con el verificador |
| `train_sft.py` | esqueleto guiado | SFT con LoRA sobre trazas verificadas |
| `train_grpo.py` | esqueleto guiado | GRPO con TRL; la tercera recompensa es vuestra |
| `grpo_step.py` | tu turno | Un paso de GRPO a mano, con tests |
| `evaluate.py` | tu turno | pass@1 base / SFT / GRPO y curvas |
| `inference.py` | hecho | Carga base + adaptador, genera, separa, verifica; detrás de `/reasoning` |
| `weights/` | vuestro | Adaptadores (fuera de git) |
| `data/` | vuestro | Problemas y trazas (fuera de git si son grandes) |
