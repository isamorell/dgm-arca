# EXPERIMENTS.md — cuaderno de experimentos

Copiad esta plantilla a la raíz de vuestro repositorio como `EXPERIMENTS.md` y añadid una
entrada cada vez que lancéis algo o toméis una decisión. Con fecha. No lo escribáis al final
de memoria: dentro de un mes no recordaréis por qué bajasteis la tasa de aprendizaje.

Una entrada honesta que diga "probamos X, no funcionó, creemos que por Y" vale más que diez
que digan "todo bien". Esto es lo primero que leo cuando corrijo interpretación.

---

## Formato de cada entrada

```
### AAAA-MM-DD · Fase N · Título corto

**Qué queríamos saber.** La pregunta o la hipótesis.
**Qué hicimos.** Comando exacto, modelo, datos, hiperparámetros que cambian respecto a la
entrada anterior.
**Qué pasó.** Números, curva (enlace a reports/), ejemplos.
**Qué concluimos.** Y qué hacemos a continuación.
```

---

## Ejemplo

### 2026-10-02 · Fase 1 · Primer GRPO sobre nuestro dataset

**Qué queríamos saber.** Si el modelo aprende el formato con nuestro prompt de sistema antes
de preocuparnos por la exactitud.

**Qué hicimos.** `uv run python -m rlm.train_grpo --data rlm/data/train.jsonl --steps 100
--num-generations 8 --max-completion-length 512` partiendo del adaptador de SFT. Sin KL.

**Qué pasó.** La recompensa de formato pasa de 0.31 a 0.97 en 40 pasos. La de exactitud sube
de 0.18 a 0.26 y se estanca. La longitud media baja de 480 a 210 tokens: el modelo aprende a
cerrar la etiqueta antes de quedarse sin presupuesto. Curva en `reports/grpo_run01.png`.

**Qué concluimos.** El formato está resuelto. La exactitud se estanca porque el 40 % de los
problemas del dataset tienen respuestas con unidades y el verificador numérico las ignora:
"120 litros" y "120" cuentan igual, pero "0.12 m³" cuenta como fallo. Siguiente paso:
normalizar unidades en el verificador y repetir.

---

## Entradas

### 2026-09-28 · Fase 1 · Generación de problemas
**Qué queríamos saber.** Si los valores de dosis, concentraciones y volúmenes del generador
tenían que salir de datos reales (p.ej. fichas técnicas de CIMA) o bastaba con rangos inventados.

**Qué hicimos.** Valoramos extraer posologías y presentaciones reales de CIMA. Lo descartamos 
para la fase 1: los datos de posología y dilución están en texto libre, dependen de la indicación,
la edad y la vía, y extraerlos exigiría que alguien del equipo leyese cada ficha y sacase los 
números a mano. En su lugar, muestreamos de listas de valores redondos (p.ej. dosis 5, 10, 25 mg;
viales 100mg/5mL) y descartamos los problemas cuya respuesta cae fuera de un rango plausible 
(`BOUNDS` en `generate_medication.py`).

El generador sigue el patrón de `generate_problems.py` (estrategia 1 de `datasets.md`): 
un sorteo de parámetros, una implementación de referencia (`solve`) que es a la vez el verificador, 
y varias plantillas de enunciado. Toda la aritmética se hace con `Fraction`, de modo que el resultado 
es exacto y el redondeo es siempre half-up. Se usa `Fraction` en lugar de Pint porque es exacto 
y determinista; Pint se usará en la herramienta de cálculo de la fase 2, como decía la propuesta.

Comandos (semilla por defecto, `DEFAULT_SEED = 0`):

```bash
uv run python -m rlm.generate_medication --n 800 --split train --out rlm/data/train.jsonl
uv run python -m rlm.generate_medication --n 200 --split test  --out rlm/data/test.jsonl --avoid rlm/data/train.jsonl
uv run python -m rlm.generate_medication --n 100 --split ood   --out rlm/data/test_ood.jsonl
```

**Qué pasó.** El generador produce problemas con la aritmética correcta, pero las dosis no son 
clínicamente realistas en todos los casos. Ejemplos de `train.jsonl`: metronidazol a
100 mg/kg/día y a 1000 mcg/kg/día, y ceftriaxona a 500 mcg/kg/día. La dosis habitual de cada
fármaco es muy distinta de esos valores. Un 10 % de los enunciados (78 de 800) no lleva nombre 
de fármaco.

**Qué concluimos.** Para la fase 1 el objetivo es evaluar si el modelo razona con unidades y 
conversiones, y eso no depende de que el fármaco sea real: la respuesta se verifica contra el 
cálculo, no contra un manual de farmacología. Es una limitación asumida y la dejamos escrita en el
informe: el modelo entrenado sabe calcular, no sabe qué dosis es adecuada. Las concentraciones y 
presentaciones reales entran en la fase 2 (CIMA) y la fase 3 (RAG).

---

### 2026-09-28 · Fase 1 · Diseño de las plantillas de los enunciados

**Qué queríamos saber.** Cómo conseguir enunciados lo bastante variados para que el modelo
aprenda la tarea y no la plantilla.

**Qué hicimos.** La idea inicial era usar unas 7 plantillas de enunciado y rellenarlas con
parámetros aleatorios. La descartamos porque el modelo vería solo esas 7 estructuras con los
números cambiados. En su lugar:

- separamos 7 familias de problema (velocidad de bomba, goteo, dosis por peso y día, volumen a
  administrar, infusión mcg/kg/min, su inversa y diluciones), cada una con 3-4 plantillas
  completas (26 en total);
- variamos la pregunta final (3-5 formulaciones por familia) y la petición de redondeo;
- añadimos cuatro estilos de escritura compuestos a partir de datos sueltos: narrativo,
  telegráfico, lista y línea de datos, con el orden de los datos aleatorio;
- variamos la coma o el punto decimal, las unidades (mL/ml, mcg/µg), la forma de escribir el
  tiempo (4 h, 240 min, 1 h 30 min) y añadimos datos irrelevantes (cama, hora, vía) en un 35 %
  de los problemas.

**Qué pasó.** Contamos esqueletos distintos, es decir, enunciados con los números
enmascarados: 763 de 800 en train (95 %), 193 de 200 en test y 96 de 100 en OOD. Reparto de
estilos en train: cohesivo 244, narrativo 248, telegráfico 133, lista 84, datos 91. Con 7
plantillas fijas el máximo habría sido del orden de 7 esqueletos.

**Qué concluimos.** La variedad de redacción es alta. Es variedad léxica y de estructura: las
familias siguen siendo 7, así que el modelo no ve tipos de problema nuevos, solo formas nuevas
de escribirlos.

---

### 2026-09-30 · Fase 1 · Desequilibrio entre familias en los datasets

**Qué queríamos saber.** Si los datasets quedaban equilibrados entre familias y niveles de
dificultad, y si test tiene la misma distribución que train.

**Qué hicimos.** Contamos problemas por familia y por nivel (campo `branches`) en train, test y
OOD de la primera versión del generador.

**Qué pasó.** Con sorteos equilibrados saldrían unos 160 problemas por familia en train. No fue
así:

| Primera versión | pump | drops | dose_day | volume | dilution |
|---|---|---|---|---|---|
| train (800) | 82 | 156 | 239 | 220 | 103 |
| test (200) | 12 | 34 | 74 | 69 | 11 |

El nivel 3 (dosis por peso y volumen) era el 57 % de train y el 72 % de test, y el nivel 1 solo
el 14 % de train y el 6 % de test. La causa: `pump` y `dilution` tenían un espacio de
parámetros pequeño (en `pump`, el volumen salía de 5 valores fijos 5 de cada 6 veces, así que
solo había 29 volúmenes distintos y 82 problemas únicos). La deduplicación descartaba los
repetidos y las demás familias ocupaban ese sitio. Además, test se genera con `--avoid` sobre
train y solo recibía las combinaciones que train había dejado libres. Con 12 y 11 problemas de
`pump` y `dilution` en test no se podía analizar el pass@1 por familia.

**Qué hicimos.** Dos cambios en `generate_medication.py`:

1. Espacios de parámetros más grandes: el tiempo pasa de 13 valores a uno cada 15 minutos
   entre 30 min y 12 h (más 24 h); el volumen es una bolsa estándar el 40 % de las veces y en
   el resto cualquier múltiplo de 10 mL; las diluciones tienen más dosis, más concentraciones
   objetivo y más volúmenes de reconstitución.
2. Cuotas por familia en `generate`: cada familia recibe el mismo número de problemas en vez de
   sortear la familia al azar, y el resultado se mezcla al final. Si una familia no puede
   producir su cuota de problemas únicos, el script falla con un mensaje que la nombra.

Regeneramos los tres ficheros con los mismos comandos y la misma semilla.

**Qué pasó (después).**

| Segunda versión | pump | drops | dose_day | volume | dilution |
|---|---|---|---|---|---|
| train (800) | 160 | 160 | 160 | 160 | 160 |
| test (200) | 40 | 40 | 40 | 40 | 40 |

OOD: 50 `infusion` y 50 `infusion_inverse`. Niveles 1 / 2 / 3 en train: 231 / 249 / 320 (29 %,
31 %, 40 %); en test: 50 / 70 / 80 (25 %, 35 %, 40 %). El solapamiento entre ficheros sigue en
0. Capacidad: train llega sin problema a 3.000 problemas y el límite (unos 3.200) lo marcan las
diluciones, con 653 combinaciones únicas.

Queda un desequilibrio menor dentro de `dilution`: en train hay 89 problemas de diluyente y 71
de volumen final, y en test 30 de diluyente y 10 de volumen final. Los problemas de volumen
final tienen menos combinaciones y train ya se ha llevado buena parte de ellas, así que
`--avoid` deja pocas para test.

**Qué concluimos.** Las familias ya están equilibradas y se puede analizar el pass@1 por
familia con 40 problemas cada una en test. Con solo 10 problemas de volumen final en test, el
resultado de ese subtipo de dilución no es fiable, y lo trataremos con cautela en el análisis.

(vuestras entradas, de la más antigua a la más reciente)

---

### 2026-10-07 · Fase 1 · Destilación: trazas de razonamiento con Qwen3-4B

**Qué queríamos saber.** Cuántas trazas verificadas conseguimos con un teacher de 4B en modo
thinking sobre nuestros 800 problemas de train, si razona en español, y qué familias o niveles
le cuestan más (primera medida de lo difícil que es el dominio).

**Qué hicimos.** Teacher `Qwen/Qwen3-4B` en modo thinking, muestreo con los valores de su ficha
(temperatura 0.6, top-p 0.95, top-k 20, nunca greedy), 4 trazas por problema sobre
`rlm/data/train.jsonl`, límite de 1200 tokens nuevos. Cada traza se pasa por un único juez
(`rlm/distill.py`): se reescribe a `<think>…</think><answer>…</answer>`, se verifica con
`MedicationVerifier` (valor **y** unidad) y se descarta con un motivo único. Para el SFT se
guardan como máximo las 2 trazas verificadas más cortas de cada problema.

```bash
uv run python -m rlm.distill --data rlm/data/train.jsonl --samples 4 \
  --batch-size 2 --max-new-tokens 1200 \
  --raw-output outputs/distill/raw.jsonl --output rlm/data/sft_traces.jsonl
```

Antes de la ejecución completa hubo tres decisiones tomadas con pruebas pequeñas:

1. **Idioma.** En una primera prueba (20 generaciones, 512 tokens) las 2 trazas que terminaron
   eran correctas pero razonaban en inglés, aunque el enunciado y la instrucción estaban en
   español. Pedirle español en el prompt no bastó, así que se arranca su razonamiento en español
   (el prompt acaba con `<think>` y la frase "Vale, voy a resolverlo paso a paso."). Esa frase
   queda como inicio de cada traza. Se descartan las trazas que no son mayoritariamente españolas.
2. **Piloto** (40 problemas, 160 generaciones, ~30 min): 147 verificadas (91,9 %). La longitud
   de las generaciones tenía mediana 474 tokens, percentil 95 en 954 y solo el 1 % llegaba al
   límite de 2048. Por eso se bajó el límite a 1200 en la ejecución completa: acorta los lotes
   lentos y pierde pocas trazas.
3. **Dos fallos del juez, vistos en el piloto.** Una respuesta correcta escrita en LaTeX
   (`20{,}20 mL`) se contaba como mal porque el verificador leía dos números; ahora se limpia
   antes de verificar. Y el prefill acababa en un espacio que el teacher imitaba, dejando dobles
   espacios tras cada punto; se quitó el espacio y se normalizan los espacios al construir la traza.
   De los 8 `wrong_value` del piloto, 7 eran errores de aritmética reales del teacher (por
   ejemplo 740 mL en 315 min es 140,95 → 141,0, y contestó 141,1 las tres veces): la respuesta de
   referencia del generador era la correcta.

**Qué pasó.** 3200 generaciones (800 problemas × 4):

- Verificadas: **3005 (93,9 %)**. Problemas con al menos una traza verificada: **788/800**.
  Longitud media de las trazas aceptadas: 570 tokens. Pasan al SFT **1557 trazas** (máximo 2 por
  problema).
- Motivos de rechazo (195 generaciones): `truncated` 164, `wrong_value` 25, `wrong_unit` 6;
  `no_answer`, `bad_format`, `too_short`, `not_spanish` y `too_long` 0.

| Familia | Generadas | Verificadas | Tasa |
|---|---|---|---|
| dilution | 640 | 501 | 78,3 % |
| dose_day | 640 | 629 | 98,3 % |
| drops | 640 | 625 | 97,7 % |
| pump | 640 | 621 | 97,0 % |
| volume | 640 | 629 | 98,3 % |

| Nivel | Generadas | Verificadas | Tasa |
|---|---|---|---|
| 1 | 924 | 902 | 97,6 % |
| 2 | 996 | 845 | 84,8 % |
| 3 | 1280 | 1258 | 98,3 % |

Duración de la ejecución completa en el DGX (una partición MIG de 16 GiB): unas 6 h (de 09:11 a 15:10 del 7 de octubre), en una sola ejecución sin reanudar..

**Qué concluimos.**

- El dominio es fácil para el teacher: casi todo lo que se rechaza (164 de 195, el 84 %) es
  porque no termina de razonar en 1200 tokens, no porque se equivoque (31 respuestas mal de
  3200, el 1 %). Y las truncadas están casi todas en una familia: `dilution` acumula 137 de las
  164 (el 21 % de sus 640 generaciones) y solo 2 respuestas mal; en las otras cuatro familias
  hay entre 3 y 10 truncadas cada una. Es decir, en diluciones (reconstituir, calcular el
  diluyente o el volumen final) el teacher se alarga mucho, pero cuando termina acierta. Por
  niveles, el 2 (84,8 %) es el más bajo porque es donde caen esas diluciones.
- Los errores reales del teacher están sobre todo en `pump` (11 de 25 con `wrong_value`) y
  `dose_day` (5 de las 6 con `wrong_unit`).
- Reparto de las 1557 trazas del SFT por familia: dose_day 320, volume 320, drops 319, pump 318,
  dilution 280. El desequilibrio es pequeño (280 frente a ~320), pero `dilution` es también la
  familia con más problemas sin traza verificada (12 problemas en total no tienen ninguna). Si
  el modelo de SFT falla sobre todo en `dilution` en el test, la primera opción es regenerar solo
  esos problemas con un límite mayor de tokens.
- Las 6 respuestas con el valor correcto y la unidad equivocada (`wrong_unit`) las habría dado
  por buenas un verificador solo numérico: justifican comprobar la unidad.
- Limitaciones: el juez comprueba la respuesta final, no el razonamiento, así que puede haber
  trazas con respuesta correcta y explicación pobre (hemos visto alguna que describe los pasos
  sin escribir los números intermedios). Tampoco está medido cuánto acorta las trazas el prefill:
  sin él, 18 de 20 generaciones no habían terminado en 512 tokens, y con él la mediana del piloto
  es 474, pero son pruebas con problemas y límites distintos y no se pueden comparar sin más.
- Siguiente paso: SFT con LoRA sobre `rlm/data/sft_traces.jsonl` (`feat_sft`).

---

### 2026-10-07 · Fase 1 · SFT con LoRA sobre las trazas destiladas

**Qué queríamos saber.** Si con las 1557 trazas verificadas un adaptador LoRA sobre
Qwen3-0.6B aprende el formato `<think>…</think><answer>…</answer>` y el estilo de razonamiento
en español, y cuántas épocas conviene.

**Qué hicimos.** Una sola ejecución, con valores de partida estándar y sin barrido:

```bash
uv run python -m rlm.train_sft --data rlm/data/sft_traces.jsonl --output rlm/weights/sft_lora
```

- Modelo base `Qwen/Qwen3-0.6B`: es el que entrenará GRPO después y cabe con holgura en la
  partición MIG de 16 GiB (el profesor, de la misma familia, ya usa su plantilla de chat).
- LoRA r=16, alpha 32, dropout 0.05, sobre todas las capas lineales. Un arranque en frío enseña
  un formato y un estilo, no conocimiento nuevo, así que un adaptador pequeño basta.
- Learning rate 2e-4 con decaimiento coseno y 5 % de calentamiento (rango habitual de LoRA:
  1e-4 a 3e-4). 3 épocas. Batch efectivo de 16 (2 × 8 pasos de acumulación). bf16 y
  gradient checkpointing. La pérdida se calcula solo sobre la respuesta del asistente.
- Datos: el 5 % de los problemas, elegidos por problema y no por traza (cada problema tiene
  dos trazas y así no hay fugas), se reserva para evaluación: 1479 ejemplos de entrenamiento y
  78 de evaluación. Longitudes en tokens (prompt + traza) en train: mediana 476, percentil 95 en
  752, máximo 1177. `max_length` = 1536, así que no se descartó ni cortó ningún ejemplo.
- Al final de cada época se calcula la pérdida de evaluación y se guarda el adaptador de la
  mejor época.

**Qué pasó.** 279 pasos en 27 minutos. La pérdida de entrenamiento baja de 0,81 en los primeros
pasos a ~0,32 al acabar la primera época y a ~0,205 al final (media de la ejecución: 0,286).

| Época | eval_loss | Precisión de token (eval) |
|---|---|---|
| 1 | 0,2643 | 0,9111 |
| 2 | 0,2379 | 0,9164 |
| 3 | 0,2337 | 0,9198 |

La mejor época por pérdida de evaluación es la 3, que es la que se guarda. Historial completo en
`rlm/weights/sft_lora/log_history.json`.
   
Adaptador publicado en Hugging Face: `raquelfes11/dgm-arca-sft-lora` (modelo base `Qwen/Qwen3-0.6B`).

**Qué concluimos.** La pérdida de evaluación baja en las tres épocas y no hay señal de
sobreajuste: la diferencia entre entrenamiento (~0,205) y evaluación (0,234) es de unos 0,03. De
la época 2 a la 3 la mejora es pequeña (0,004) y con solo 78 ejemplos de evaluación (39
problemas) cae dentro del ruido: lo único que podemos afirmar es que la tercera época no
empeora, no que sea mejor que la segunda. No hemos probado otros valores de learning rate, rank
ni número de épocas; los elegidos son valores de partida razonables, no óptimos.

Esta pérdida mide cuánto se parece el modelo al texto del profesor, no si sus respuestas son
correctas. La comprobación real es el pass@1 sobre `rlm/data/test.jsonl` (y `test_ood.jsonl`)
con `rlm/evaluate.py`, comparando el modelo base y el SFT. Está pendiente y es el siguiente
paso, junto con revisar a mano unos cuantos fallos.
