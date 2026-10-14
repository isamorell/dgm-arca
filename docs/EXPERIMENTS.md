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
(`BOUNDS` en `generate_problems.py`).

El generador utiliza la plantilla de `generate_problems.py` (estrategia 1 de `datasets.md`): 
un sorteo de parámetros, una implementación de referencia (`solve`) que es a la vez el verificador, 
y varias plantillas de enunciado. Toda la aritmética se hace con `Fraction`, de modo que el resultado 
es exacto y el redondeo es siempre half-up. Se usa `Fraction` en lugar de Pint porque es exacto 
y determinista; Pint se usará en la herramienta de cálculo de la fase 2, como decía la propuesta.

Comandos (semilla por defecto, `DEFAULT_SEED = 0`):

```bash
uv run python -m rlm.generate_problems --n 800 --split train --out rlm/data/train.jsonl
uv run python -m rlm.generate_problems --n 200 --split test  --out rlm/data/test.jsonl --avoid rlm/data/train.jsonl
uv run python -m rlm.generate_problems --n 100 --split ood   --out rlm/data/test_ood.jsonl
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

**Qué hicimos.** Dos cambios en `generate_problems.py`:

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

---

### 2026-10-04 · Fase 1 · Definición de la recompensa de dominio

**Qué queríamos saber.** Queríamos definir una recompensa de dominio que fuese importante para nuestra tarea de ayudar a estudiantes de enfermería

**Qué hicimos.** Decidimos que lo mejor era tener una recompensa que penalizase al modelo no utilizar las unidades que pedía el modelo. Para permitirle un poco más de flexibilidad, tenemos en cuenta que las mismas unidades pueden escribirse de distintas formas. Por ejemplo, gotas por minuto se puede abreviar en gotas/min.

También incluimos estas diferentes formas en la extracción de la unidad, que se realiza con cuidado para extraer la unidad que acompaña a la respuesta final y no otra unidad que puede que acompañe a un número distinto.

**Qué pasó.** Generamos diferentes aliases para cada unidad con el fin de poder comparar las diferentes formas en las que referenciamos una misma unidad. Estos los incluimos también en el patrón de regex para extraer las unidades de las respuestas

**Qué concluimos.** Así podremos evaluar la correcta extracción de unidades como la recompensa de dominio.

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
con `rlm/evaluate.py`, comparando el modelo base y el SFT. Estaba pendiente cuando escribimos
esta entrada; después se midió: 80,5 % en test (entrada del 2026-10-12) y 24,0 % en OOD (entrada
del 2026-10-14).

### 2026-10-11 · Fase 1 · GRPO sin beta y con beta 0.04

**Qué queríamos saber.** Queríamos saber si el modelo obtiene mejores resultados al aplicarle la penalización kl con beta 0.04 que es la que usan en el paper de DeepSeekMath (https://arxiv.org/pdf/2402.03300) o sin la penalización

**Qué hicimos.** Como entrenamos el modelo partiendo del entrenado con SFT, configuramos en el archivo .env una nueva variable con el enlace al repositorio de Hugging Face Hub donde lo guardamos en el paso antrior. Para ello, añadimos la siguiente línea al fichero:
`ARCA_SFT_REPO=raquelfes11/dgm-arca-sft-lora`

Durante la primera ejecución, vimos que la recompensa de dominio del modelo solía fallar en los ejercicios cuyas unidades eran gotas/min. Vimos que la unidad correcta a devolver era precisamente esa y así descubrimos una errata en el código del `rewards.py`. Tras corregirlo, volvimos a entrenar los dos modelos.

Para entrenar el modelo sin kl, utilizamos el siguiente comando:

`uv run python -m rlm.train_grpo --data rlm/data/train.jsonl --init-adapter rlm/weights/sft_lora`

que utiliza el modelo finetuneado como base.

Luego, entrenamos el modelo con KL con beta 0.04 para ver cómo afecta esto al entrenamietno. Para ello, ejecutamos este comando:

`uv run python -m rlm.train_grpo --data rlm/data/train.jsonl --init-adapter rlm/weights/sft_lora --beta 0.04`

**Qué pasó.** En la siguiente gráfica, se ve la evolución de las recompensas a lo largo del entrenamiento del modelo sin KL. Como se puede observar, el máximo de 0.992 se alcanza en el step 92.

![Evolución de las recompensas a lo largo del entrenamiento del modelo sin KL](../reports/rlm/final_rlm_lora_sin_kl/curves_grpo_rewards.png)

En cambio, al entrenar el modelo con KL con beta = 0.04, se puede ver como el máximo toma un valor menor (0.971) y en un step temprano, el 37. Esto puede deberse a que el modelo base entrenado con SF ya fuese muy bueno en esta tarea por lo que no hay demasiado hueco para que el entrenamiento con GRPO mejore el rendimiento del modelo.

![Evolución de las recompensas a lo largo del entrenamiento del modelo con KL (beta = 0.04)](../reports/rlm/final_rlm_lora_kl_04/curves_grpo_rewards.png)

Los adaptador se encuentran publicados en Hugging Face: 
`isamorell/dgm-arca-rlm-lora-sin-kl` (modelo entrenado con grpo y sin KL).
`isamorell/dgm-arca-rlm-lora-kl-04` (modelo entrenado con grpo y con KL (beta = 0.04)).

**Qué concluimos.** Para ambos casos, el modelo logra máximos muy altos en puntos tempranos del entrenamiento.  Esto puede deberse a que el modelo base ya lo hace muy bien. Por ello, vamos a comparar el rendimiento de los tres modelos en el conjunto de test. 

Además, en train parece que funciona mejor no utilizar KL pero esto puede deberse a que KL está actuando como un freno sobre el entrenamiento para que no se separe demasiado del modelo base. Para ver si el modelo está mejorando al incluir KL, vamos a probar el rendimiento en el conjunto de test utilizando el `evaluate.py`.

---

### 2026-10-12 · Fase 1 · Pass@1 de los 3 modelos entrenados y del modelo base

**Qué queríamos saber.** Queremos saber que modelo generaliza mejor en test para poder analizar si grpo mejora el rendimiento del modelo frente al sft, y si utilizar kl con beta 0.04 mejora frente a no usar kl.

**Qué hicimos.** Hemos calculado el Pass@1 de los modelos base, sft y grpo sin KL y con KL beta = 0.04 ejecutando el `evalute.py` con el siguiente comando:

`uv run python -m rlm.evaluate --data rlm/data/test.jsonl --adapters base=none sft=rlm/weights/sft_lora grpo_sin_kl=rlm/weights/final_rlm_lora_sin_kl grpo=rlm/weights/final_rlm_lora_04`

**Qué pasó.** La ejecución ha tardado más de dos horas ya que la inferencia se ha ido haciendo 1 a 1, sin batching.

Como podemos observar en la imagen siguiente, el modelo base obtiene un 0.0% de pass@1. Esto se debe a que sin el sft, el modelo no ha aprendido el formato y lo rompe añadiendo un bloque `<reasoning process> ... </reasoning process>`. Además, responde sin unidades lo que no cumple con la recompensa de dominio.

Una vez el modelo aprende a razonar, la puntuación sube a 80.5%. Este valor coincide para el sft y el grpo con kl con beta = 0.04. Esto probablemente se deba a que el modelo se encuentra en una zona plana de la que no puede salir debido a la penalización kl. Esto se puede deducir también del análisis anterior en el que se vio que la máxima recompensa obtenida en el entrenamiento fue en el paso 37, indicando que a partir de ese punto el modelo no es capaz de mejorar.

Por último, vemos como el modelo entrenado con grpo sin kl mejora el resultado obtenido por sft ya que es capaz de adaptarse a datos que no se han observado previamente. Este modelo obtiene un 89.5% de pass@1, un 9% más que el modelo sft y el entrenado con grpo con kl.

![Pass@1 de los modelos base, sft, grpo_sin_kl y grpo con kl y beta = 0.04](../reports/phase1_pass1.png)

**Qué concluimos.** El modelo base falla siempre debido a errores de formato y a que no devuelve las unidades de la respuesta. GRPO sin KL mejora a SFT y a GRPO con KL en test y estos dos últimos obtienen el mismo pass@1.

A continuación vamos a entrenar otro modelo con GRPO sin KL y con otro valor de tamaño de grupo.

---

### 2026-10-12 · Fase 1 · Entrenamiento modelo con GRPO sin KL y con --num-generations = 4

**Qué queríamos saber.** Queremos analizar cómo afecta el número de generaciones al entrenamiento del modelo y a su rendimiento final.

**Qué hicimos.** Comenzamos entrenando el modelo con el parámetro --num-generations = 4. Esto lo hacemos ejecutando el siguiente comando:
`uv run python -m rlm.train_grpo --data rlm/data/train.jsonl --init-adapter rlm/weights/sft_lora --num-generations 4`

Tras entrenar el modelo, sacamos sus curvas de evolución de la recompensa y la longitud de respuesta ejecutando el comando:
`uv run python -m rlm.plot_curves rlm/weights/final_rlm_lora --output reports/rlm/final_rlm_lora_sin_kl_group_4/curves_grpo`

Así podremos ver las curvas y obtener el máximo de recompensa media obtenida durante el entrenamiento.

Por último, modificamos el evaluate.py para que no se borren los resultados que ya tengo y que solo se añadan los nuevos adicionalmente. Ejecutando el siguiente comando para el nuevo modelo, obtenemos el rendimiento de este en el conjunto de test :
`uv run python -m rlm.evaluate --data rlm/data/test.jsonl --adapters grpo_group_4=rlm/weights/final_rlm_lora_sin_kl_group_4`

**Qué pasó.** Este modelo obtiene un 86% de pass@1. La diferencia principal con los otros dos modelos que hemos entrenado con GRPO es que, en este caso, hemos utilizado 4 generaciones en vez de 8 para calcular las recompensas.

Las curvas de los tres modelos siguen un patrón similar como se puede apreciar en las siguientes 3 imágenes. Las recompensas comienzan en valores por debajo del 0.5 y en torno al paso 25 ya se encuentran por encima del 0.8. La longitud media de las respuestas decae de 600 a unos 300-400 tokens cuando el modelo empieza a razonar correctamente y obtener las respuestas esperadas. El impacto de bajar el número de generaciones de 8 a 4 se puede observar en que las curvas son más ruidosas y la desviación de la recompensa dentro del grupo es un poco mayor. 

![Evolución de diferentes métricas a lo largo del entrenamiento del modelo sin KL (num-generations = 8)](../reports/rlm/final_rlm_lora_sin_kl/curves_grpo.png)

![Evolución de diferentes métricas a lo largo del entrenamiento del modelo con KL (beta = 0.04, num-generations = 8)](../reports/rlm/final_rlm_lora_kl_04/curves_grpo.png)

![Evolución de diferentes métricas a lo largo del entrenamiento del modelo sin KL (num-generations = 4)](../reports/rlm/final_rlm_lora_sin_kl_group_4/curves_grpo.png)

**Qué concluimos.** Por lo tanto, el mejor modelo continúa siendo el modelo entrenado con GRPO, sin KL y con 8 generaciones de respuestas ya que es el que obtiene un mejor pass@1 y es más estable que el modelo de 4 generaciones.

A continuación, vamos a analizar en qué falla concretamente cada modelo para entender qué ha aprendido en cada paso y qué tiene que aprender aún.

---

### 2026-10-12 · Fase 1 · Análisis de las respuestas obtenidas en el test

**Qué queríamos saber.** Queremos saber qué ha aprendido cada modelo durante cada entrenamiento y queremos comprender si hay un grupo concreto de problemas que el modelo falla para ayudarle a que aprende a resolver ese tipo concreto de ejercicios.

**Qué hicimos.** Para realizar este análisis, hemos escrito un nuevo fichero (`analyze_eval.py`) que analiza varios puntos:
1. Busca patrones en las respuestas de los modelos para determinar de forma más concreta si siguen el formato y si devuelven la respuesta correcta.
2. El porcentaje de acierto por familia de problemas
3. Tipo de respuestas devueltas por cada modelo
4. Fallos cometidos por el mejor modelo (grpo sin kl)

Para obtener este análisis hay que ejecutar el siguiente comando:
`uv run python -m rlm.analyze_eval`

**Qué pasó.** Comenzamos analizando si el modelo devuelve la respuesta correcta en el formato correcto. Como podemos ver en la siguiente tabla, el modelo base no devuelve una respuesta más de la mitad de las veces y solo cumple con el formato indicado en un 9,5% de los casos. Además, solo devuelve el valor correcto un 26,5% de las veces. Todo esto explica por qué se obtiene un pass@1 de un 0,0%.

Si comparamos los otros 4 modelos que obtienen respuestas más razonables para el problema, podemos observar como el modelo entrenado con GRPO sin KL presenta las mayores mejoras tanto en el formato como en la corrección de la respuesta.

GRPO con KL no mejora el pass@1 de SFT pero su formato es algo mejor (94,0 % frente a 92,5 %) y tiene menos respuestas ausentes (6,0 % frente a 7,5 %). Y el modelo de GRPO sin KL mejora todas las métricas frente a estos dos.

| Modelo | Valor correcto (`valor_ok`) | Formato válido | Sin respuesta (`sin_answer`) | Pass@1 |
|---|---:|---:|---:|---:|
| Base | 26,5 % | 9,5 % | 53,5 % | 0,0 % |
| SFT | 80,5 % | 92,5 % | 7,5 % | 80,5 % |
| GRPO sin KL | **89,5 %** | **97,0 %** | **3,0 %** | **89,5 %** |
| GRPO con KL | 80,5 % | 94,0 % | 6,0 % | 80,5 % |
| GRPO sin KL group 4 | 86,0 % | 94,5 % | 5,5 % | 86,0 % |

A continuación, analizamos las tasas de éxito por familia.

| Familia | n | Base | SFT | GRPO sin KL | GRPO con KL | GRPO sin KL group 4 |
|---|---:|---:|---:|---:|---:|---:|
| Dilution | 40 | 0,00 | 0,75 | **0,95** | 0,82 | 0,80 |
| Dose/day | 40 | 0,00 | 0,93 | **0,97** | 0,93 | 0.93 |
| Drops | 40 | 0,00 | 0,78 | **0,88** | 0,78 | 0,85 |
| Pump | 40 | 0,00 | 0,75 | 0,72 | 0,68 | **0,80** |
| Volume | 40 | 0,00 | 0,82 | **0,95** | 0,82 | 0,93 |

GRPO sin KL obtiene las mejores puntuaciones en 4 de las 5 familias de problemas que tenemos en el dataset de test. Para la familia de pump, obtiene un 72% de aciertos por lo que pump es el grupo con el mejor margen de mejora en el rendimiento del modelo. Esta familia es la que menores resultados presenta para todos los modelos. Esto muestra como al modelo le cuesta razonar y obtener la respueta correcta.

En la siguiente tabla podemos observar como el modelo base no devuelve nunca el valor correcto. Esto se debe a que no devuelve la respuesta la mitad de las veces y, cuando devuelve una respuesta, el valor numérico de ésta está mal o no presenta la unidad correcta.

El modelo SFT aprende a devolver siempre la unidad correcta por lo que ninguno de los modelos entrenados presenta fallos de esta casuística. También reduce los fallos debido a la falta de respuesta y de valor incorrecto. Esto sugiere que el ajuste supervisado mejora tanto la generación de respuestas completas como el cumplimiento del formato esperado.

| Tipo de resultado | Base | SFT | GRPO sin KL | GRPO con KL | GRPO sin KL group 4 |
|---|---:|---:|---:|---:|---:|
| Correcto (`ok`) | 0 | 161 | **179** | 161 | 172 |
| Valor correcto, unidad incorrecta (`valor_ok_unidad_mal`) | 53 | 0 | 0 | 0 | 0 |
| Valor incorrecto (`valor_mal`) | 40 | 24 | **15** | 27 | 17 |
| Sin respuesta (`sin_answer`) | 107 | 15 | **6** | 12 | 11 |
| **Total** | **200** | **200** | **200** | **200** | **200** |

Por último, vamos a analizar concretamente por qué falla el modelo entrenado con grpo sin KL según los grupos de familias:

| Familia | Valor incorrecto | Sin respuesta | Total de fallos |
|---|---:|---:|---:|
| Pump | 6 | 5 | **11** |
| Drops | 5 | 0 | 5 |
| Volume | 2 | 0 | 2 |
| Dose/day | 1 | 0 | 1 |
| Dilution | 1 | 1 | 2 |
| **Total** | **15** | **6** | **21** |

Aproximadamente un 25% de los fallos se deben a que el modelo no devuelve una respuesta. Esto se debe a que el razonamiento necesita más tokens de los establecidos como máximo.

De los 15 casos de respuestas incorrectas, 6 pertenecen a la familia de pump. Por lo tanto, ayudar al modelo a resolver estos problemas mejorará su rendimiento.

**Qué concluimos.** El aprendizaje supervisado mejora tanto la generación de respuestas completas como el cumplimiento del formato esperado frente al modelo base. 

De los 3 modelos entrenados, el grpo sin KL muestra las mejores tasas tanto en el formato como en la corrección de la respuesta.

A continuación, analizaremos los fallos cometidos por el modelo entrenado con grpo sin KL

---

### 2026-10-13 · Fase 1 · Fallos cometidos por el modelo entrenado con GRPO sin KL

**Qué queríamos saber.** El modelo entrenado con GRPO sin KL es el que mejor funciona (pass@1 = 0.895), y supera al entrenado con KL (`grpo`, 0.805), a `grpo_group_4` (0.860) y al `sft` (0.805). Queremos saber qué tipo de fallos comete: si son meramente aritméticos o si el modelo ha encontrado alguna forma de reward hacking.

**Qué hicimos.** Seleccionamos cinco trazas del modelo con GRPO sin KL con `is_correct = False` (en `reports/phase1_eval.json`, sección `grpo_sin_kl`). Leímos el enunciado y el razonamiento completo de cada una, comprobando si los pasos que sigue son los que daría una persona al resolver el problema.

Para cubrir también las respuestas correctas (donde se manifestaría un posible hack), ejecutamos además un filtrado automático, `rlm/scan_hacking.py`, sobre las 200 respuestas de cada modelo. Busca patrones que explotarían huecos de `rlm/rewards.py`:
- `multi_number`: más de un número dentro de `<answer>` (el verificador toma el último).
- `no_unit`: respuesta sin unidad (el reward no la comprueba).
- `ambiguous_sep`: formato `d,ddd` (el parser lo lee como separador de miles).
- `correct_bad_fmt`: respuesta correcta sin formato `<think>...</think><answer>...</answer>` válido.
- `hedging`: respuesta correcta con 3 o más marcadores de duda ("Espera", "tal vez me equivoqué"...).
- `verifier_mismatch`: discrepancia entre `is_correct` del eval y `numbers_match` de `rewards.py`.
- Además, cuenta preguntas duplicadas dentro de cada sección.

**Cómo ejecutar el script.** 

Solo el modelo GRPO sin KL, sin comparar con el reward:

```bash
python rlm/scan_hacking.py reports/phase1_eval.json --section grpo_sin_kl
```

Solo el modelo GRPO sin KL, comparando además con el verificador del entrenamiento (`rlm/rewards.py`):

```bash
python rlm/scan_hacking.py reports/phase1_eval.json --section grpo_sin_kl --reward-module rlm/rewards.py
```

Todos los modelos a la vez, para comparar:

```bash
python rlm/scan_hacking.py reports/phase1_eval.json --reward-module rlm/rewards.py
```

Opciones útiles: `--show N` imprime N ejemplos por cada flag (por defecto 5), y se puede redirigir la salida a un fichero, por ejemplo `> reports/hacking_scan.txt`.

**Qué pasó.** 4 de los 5 fallos son errores aritméticos: el modelo plantea correctamente qué operación hay que hacer, pero la ejecuta mal. Uno de ellos (caso 5) desemboca además en un bucle de dudas que agota el máximo de tokens sin llegar a dar respuesta. El fallo restante (caso 3) es un error conceptual de unidades.

A continuación se muestra cada traza completa y su análisis.

**1.**

```
{
    "question": "Datos: Tiempo: 705 min; Volumen: 490 mL (suero fisiológico). Indica la velocidad en mL/h. Da el resultado con un decimal.",
    "expected": "41.7 mL/h",
    "raw": "<think>\nVale, voy a resolverlo paso a paso. Primero, tengo que encontrar la velocidad en mL por hora. La fórmula que uso es el volumen dividido por el tiempo. El volumen es 490 mL y el tiempo es 705 minutos. Pero el resultado debe estar en horas, así que primero convierto los minutos a horas.\n\n705 minutos divididos por 60 minutos por hora. 705 / 60 = 11.75 horas. Luego, divido el volumen por el tiempo en horas: 490 mL / 11.75 h. Haciendo la división, 490 ÷ 11.75 ≈ 41.2 mL/h. El resultado ya tiene un decimal, así que ya está correcto. No necesito redondear más.\n</think>\n\n<answer>41.2 mL/h</answer>",
    "predicted": "41.2 mL/h",
    "is_correct": false,
    "has_valid_format": true,
    "n_tokens": 206,
    "family": "pump",
    "expected_unit": "mL/h",
    "expected_value": "41.7"
}
```

El modelo convierte bien 705 min a horas (11.75 h) y plantea correctamente que debe dividir volumen entre tiempo. Sin embargo, al calcular 490 / 11.75 afirma que el resultado es ≈ 41.2, cuando el valor correcto es 41.70. Es un error aritmético en una división con decimales. No intenta verificarlo.

**2.**

```
{
    "question": "En planta, se trata de un paciente de 101,5 kg. La presentación es de 0,5 g por cada 2 ml. Lleva ingresado 8 días. Se han pautado 7,5 mg por kg de amoxicilina. Calcula el volumen a administrar en ml. Da el resultado con dos decimales.",
    "expected": "3.05 mL",
    "raw": "<think>\nVale, voy a resolverlo paso a paso. Primero, el paciente pesa 101,5 kg. La dosis prescrita es 7,5 mg por kg. Entonces, necesito calcular la dosis total en mg. Multiplico 101,5 kg por 7,5 mg/kg. \n\n101,5 kg * 7,5 mg/kg = 759,75 mg. \n\nAhora, la presentación es 0,5 g por 2 ml. Debo convertir los gramos a miligramos, ya que la dosis es en mg. 0,5 g es 500 mg. Entonces, 500 mg por 2 ml. \n\nQuiero saber cuántos ml son necesarios para 759,75 mg. Hago una proporción: 500 mg / 2 ml = 759,75 mg / x ml. \n\nResolviendo, x = (759,75 mg * 2 ml) / 500 mg = 1519,5 / 500 = 3,039 ml. \n\nRedondeo a dos decimales, que sería 3,04 ml. \n\nEspera, pero tal vez me equivoqué en la conversión. Vamos a verificar. \n\nOtra forma: 0,5 g = 500 mg. 500 mg en 2 ml. Entonces, 1 mg es 2/500 = 0,004 ml. \n\nEntonces, 759,75 mg * 0,004 ml/mg = 3,039 ml. Sí, es lo mismo. \n\nEntonces, el volumen a administrar es 3,04 ml.\n</think>\n\n<answer>3,04 mL</answer>",
    "predicted": "3,04 mL",
    "is_correct": false,
    "has_valid_format": true,
    "n_tokens": 420,
    "family": "volume",
    "expected_unit": "mL",
    "expected_value": "3.05"
}
```

El razonamiento es correcto, pero el primer cálculo falla: 101,5 kg × 7,5 mg/kg da 761,25 mg y el modelo obtiene 759,75 mg. Todos los pasos posteriores (conversión de 0,5 g a 500 mg, proporción con 2 mL) son correctos, pero parten de un valor erróneo, así que el error se arrastra hasta el resultado final (3,04 mL en lugar de 3,05 mL). Cuando el modelo intenta verificar ("Espera, pero tal vez me equivoqué..."), repite exactamente el mismo producto erróneo, por lo que la verificación confirma el error en lugar de corregirlo.

**3.**

```
{
    "question": "Ejercicio de prácticas.\n- Hora de inicio: 16:00\n- Vial: 1 g/20 ml\n- Dosis de dopamina: 5 mg/kg\n- Peso: 107 kg\nIndica cuántos ml corresponden a la dosis. Expresa el resultado con dos decimales (redondeado).",
    "expected": "10.70 mL",
    "raw": "<think>\nVale, voy a resolverlo paso a paso. Primero, necesito calcular la dosis total de dopamina que se necesita. La dosis es 5 mg por kilogramo de peso, y el peso del paciente es 107 kg. Entonces, multiplico 5 mg/kg por 107 kg. \n\n5 mg/kg * 107 kg = 535 mg. \n\nAhora, tengo que convertir los miligramos a gramos, porque la concentración está dada en gramos por mililitro. 535 mg es igual a 0,535 g. \n\nLa concentración de la solución es 1 g por 20 ml. Para encontrar cuántos ml corresponden a 0,535 g, primero calculo cuántos ml hay por gramo. \n\n1 g / 20 ml = 0,05 g/ml. \n\nLuego, multiplico 0,05 g/ml por 0,535 g. \n\n0,05 g/ml * 0,535 g = 0,02675 g. \n\nAhora, convierto gramos a mililitros. 1 g = 1000 ml, así que 0,02675 g * 1000 ml/g = 26,75 ml. \n\nEl resultado es 26,75 ml, que ya tiene dos decimales. No necesito redondear porque ya está con dos decimales. Por lo tanto, la respuesta es 26,75 ml.\n</think>\n\n<answer>26,75 mL</answer>",
    "predicted": "26,75 mL",
    "is_correct": false,
    "has_valid_format": true,
    "n_tokens": 365,
    "family": "volume",
    "expected_unit": "mL",
    "expected_value": "10.70"
},
```

Calcula bien la dosis (5 mg/kg × 107 kg = 535 mg = 0,535 g) y también que 1 g / 20 mL = 0,05 g/mL. A partir de ahí falla: multiplica 0,05 g/mL × 0,535 g y trata el resultado como gramos, cuando en realidad las unidades son g²/mL. Después "convierte" esos gramos a mL con 1 g = 1000 mL, que no es una conversión válida (mezcla masa y volumen). Lo correcto era usar la concentración del vial: 0,535 g × 20 mL/g = 10,7 mL. Es un error conceptual de análisis dimensional, no aritmético.

**4.**

```
{
    "question": "Tengo que pasar 670 ml de suero glucosado al 5 % en 495 min. El equipo de suero es de 60 gotas/ml. ¿Cuántas gotas/min? Redondea al entero.",
    "expected": "81 gotas/min",
    "raw": "<think>\nVale, voy a resolverlo paso a paso. Primero, necesito calcular cuántas gotas se necesitan en total. El suero es de 670 ml al 5%, pero el problema no menciona si hay que considerar la concentración, solo el volumen. Entonces, el volumen total es 670 ml. El equipo es de 60 gotas por mililitro, así que multiplicaré 670 ml por 60 gotas/ml para obtener el número total de gotas.\n\n670 ml * 60 gotas/ml = 40200 gotas. \n\nAhora, tengo que dividir ese número total de gotas entre el tiempo total de 495 minutos para obtener las gotas por minuto. \n\n40200 gotas / 495 min ≈ 81.666... gotas/min. \n\nEl enunciado dice que redondee al entero, así que redondeo a 82 gotas por minuto. \n\nEspera, pero tal vez me equivoqué en la división. Vamos a hacerlo más precisamente. 40200 dividido entre 495. \n\nDividir 40200 entre 495: 495 * 81 = 40335, que es más de 40200. Entonces, 81 - (40335 - 40200)/495. 40335 - 40200 = 135. 135/495 = 0.2727... Así que 81 - 0.2727 ≈ 80.727, que redondea a 81. \n\nPero espera, tal vez hice un error en la multiplicación. 670 * 60 = 40200, sí. 40200 / 495: 495 * 80 = 39600. 40200 - 39600 = 600. 600 / 495 ≈ 1.222. Así que total 81.222, que redondea a 81. \n\nPero el resultado es aproximadamente 81.666, que redondea a 82. Hmm, ¿no es 81.666? Sí, 81.666... es 81.666..., que redondea a 82. Entonces, la respuesta es 82 gotas por minuto.\n</think>\n\n<answer>82 gotas/min</answer>",
    "predicted": "82 gotas/min",
    "is_correct": false,
    "has_valid_format": true,
    "n_tokens": 620,
    "family": "drops",
    "expected_unit": "gotas/min",
    "expected_value": "81"
},
```

Calcula bien las gotas totales (670 × 60 = 40200), pero al dividir 40200 / 495 dice que da ≈ 81.666, cuando el valor correcto es 81.21. Sospecha del resultado y lo reintenta: en el primer intento obtiene 80.727 (por un error intermedio, 495 × 81 = 40335 cuando son 40095) y en el segundo 81.222, que es correcto. Ambos redondean a 81. Sin embargo, en la última línea vuelve sin justificación al 81.666 inicial y redondea a 82. Se confunde y abandona la respuesta correcta que él mismo había calculado. Es un error aritmético, agravado por una autoverificación que no se usa para decidir.

**5.**

```
{
    "question": "La duración prescrita es de 165 min. La bolsa es de 370 ml de solución salina. La perfusión empieza a las 15:30 h. ¿Qué ritmo de bomba, en ml/h, corresponde? Da el resultado con un decimal.",
    "expected": "134.5 mL/h",
    "raw": "<think>\nVale, voy a resolverlo paso a paso. Primero, necesito calcular la tasa de bomba en ml por hora. La duración prescrita es 165 minutos, y la bolsa tiene 370 ml. \n\nPrimero, convierto los 165 minutos a horas, porque la bomba se pide en ml/h. 165 minutos divididos por 60 son 2.75 horas. \n\nLuego, divido la cantidad total de solución (370 ml) entre el tiempo en horas (2.75 h) para obtener la tasa. 370 dividido entre 2.75. \n\nHaciendo la división: 370 / 2.75. Calculo que 2.75 x 136 es 371, lo que me da aproximadamente 136.0 ml/h. \n\nPero para exactitud, 370 / 2.75 = 136.0 ml/h. Entonces, el ritmo de bomba es 136.0 ml/h. Redondeado a un decimal, es 136.0. \n\nEspera, pero tal vez me equivoqué en la división. Vamos a hacerlo más precisamente. 2.75 x 136 = 2.75 * 136. 2 * 136 = 272, 0.75 * 136 = 102, total 374. Entonces 370 / 2.75 = 136 - (4/2.75) ≈ 136 - 1.4545 ≈ 134.5455. Hmm, eso no es 136.0. O tal vez me equivoqué en la conversión de minutos a horas. \n\nRevisando: 165 minutos es 2.75 horas. 370 ml / 2.75 h = 134.545... ml/h. Pero el resultado debe tener un decimal. Entonces, 134.5 ml/h. Pero el enunciado dice \"con un decimal\", así que 134.5. \n\nPero tal vez me equivoqué en la división. Vamos a hacer 370 dividido entre 2.75. Multiplico numerador y denominador por 100 para eliminar decimales: 37000 / 275. Simplifico 37000 / 275. Dividir 37000 entre 275. 275 x 134 = 37100, que es un poco más que 37000. Entonces 134 - (100/275) ≈ 134 - 0.3636 ≈ 133.6364. Así que 133.6364 ml/h. Redondeado a un decimal es 133.6 ml/h. \n\nPero esto contradice con el anterior resultado. Hmm, tal vez me equivoqué en la conversión. O tal vez el resultado es 134.5 ml/h. Debo verificar. \n\nOtra forma: 2.75 h = 165 min. 370 ml / 165 min = 2.252525... ml/min. Multiplicado por 60 min/h: 2.252525 * 60 = 135.1515... ml/h. Redondeado a un decimal es 135.2 ml/h. \n\nAhora, hay que verificar si el cálculo es correcto. 370 ml / 165 min = 2.252525... ml/min. 2.252525 * 60 = 135.1515... ml/h. Sí, entonces con un decimal es 135.2 ml/h. \n\nPero el enunciado dice \"con un decimal\", así que 135.2. Pero el resultado inicial me dio 133.6, que no es correcto. Entonces, el error debe estar en la conversión. \n\nRevisando: 165 minutos es 2.75 horas. 370 / 2.75 = 134.545... ml/h. Pero 134",
    "predicted": null,
    "is_correct": false,
    "has_valid_format": false,
    "n_tokens": 1024,
    "family": "pump",
    "expected_unit": "mL/h",
    "expected_value": "134.5"
},
```

Plantea bien el método (165 min = 2.75 h; 370 / 2.75), pero la división le da resultados distintos en cada intento (136.0, 134.5, 133.6, 135.2...). Entra en un bucle de "tal vez me equivoqué en la división", recalcula sin converger y alcanza el máximo de 1024 tokens sin escribir `<answer>`. Se marca como incorrecta y sin formato válido. La raíz del fallo es también aritmética (no sabe dividir 370 / 2.75 de forma fiable).

**Resultados del filtrado automático.**

| Modelo | pass@1 | multi_number | no_unit | ambiguous_sep | correct_bad_fmt | hedging | verifier_mismatch |
|---|---|---|---|---|---|---|---|
| base | 0.000 | 0 | 93 | 0 | 0 | 0 | 47 |
| sft | 0.805 | 0 | 0 | 1 | 0 | 46 | 2 |
| grpo_sin_kl | 0.895 | 0 | 0 | 0 | 0 | 75 | 1 |
| grpo (con KL) | 0.805 | 0 | 0 | 1 | 0 | 35 | 2 |
| grpo_group_4 | 0.860 | 0 | 0 | 1 | 0 | 37 | 2 |

No hay preguntas duplicadas en ninguna sección. Nota sobre `base`: sus 93 `no_unit` y 47 `verifier_mismatch` no indican hacking; son consecuencia de que casi nunca produce el formato correcto (pass@1 = 0). El `ambiguous_sep` de `sft`, `grpo` y `grpo_group_4` es el caso `3,667 mg` (esperado 3666.7), que falla, así que no es explotación.

**Qué concluimos.** La mayoría de los fallos estudiados se deben a errores de cálculo: el modelo sabe qué operación realizar en cada momento, pero no la ejecuta correctamente, y su autoverificación no es fiable (repite el mismo error, vuelve al primer valor o no converge). El único fallo no aritmético es un error conceptual de unidades.

No encontramos evidencia de reward hacking en `grpo_sin_kl`. En los cinco fallos estudiados el modelo no obtiene reward por un atajo: son errores genuinos con reward 0. Además, el filtrado automático no detecta ninguno de los patrones que explotarían los huecos del reward: ninguna respuesta con varios números, sin unidad, con separador ambiguo o correcta sin formato válido.

Limitaciones: el filtro solo detecta los huecos que identificamos en `rewards.py`; un hack distinto no se detectaría. 

---

### 2026-10-14 · Fase 1 · Test fuera de distribución (OOD): SFT frente a GRPO sin KL

**Qué queríamos saber.** Cuánto empeoran el SFT y el mejor modelo de GRPO (sin KL, 8
generaciones) cuando los problemas son de una familia que no aparece en train, y si GRPO se
degrada menos que el SFT (la idea de "SFT memoriza, RL generaliza").

**Qué hicimos.** Evaluamos los dos modelos con `rlm.evaluate` (generación greedy, máximo 1024
tokens nuevos) sobre `rlm/data/test_ood.jsonl`: 100 problemas, 50 de `infusion` (de dosis en
mcg/kg/min a velocidad en mL/h) y 50 de `infusion_inverse` (de velocidad a dosis en
mcg/kg/min). Ninguna de las dos familias está en train ni en test, y los 100 problemas son de
nivel 4 (train y test solo llegan al nivel 3). Lo comparamos con el pass@1 en test (200
problemas) que ya teníamos en `reports/phase1_eval.json`.

```bash
uv run python -m rlm.evaluate --data rlm/data/test_ood.jsonl \
  --adapters sft=rlm/weights/sft_lora grpo_sin_kl=rlm/weights/final_rlm_lora_sin_kl \
  --out reports/phase1_eval_ood.json --history reports/phase1_pass1_ood.png
uv run python -m rlm.compare_ood --rescore --models sft grpo_sin_kl
```

`rlm/compare_ood.py` calcula el pass@1 con intervalo de Wilson al 95 %, la caída entre test y OOD
con intervalo de Newcombe, y compara los dos modelos sobre los mismos problemas con un test de
signos exacto (solo cuentan los problemas que resuelve uno de los dos).

**Un fallo del verificador que cambió el resultado.** La primera medida dio 20,0 % (SFT) y 27,0 %
(GRPO), con `infusion_inverse` en 0/50 para los dos. Al revisar los casos de "valor correcto,
unidad incorrecta" vimos que varios tenían la unidad escrita igual que la esperada
(`0.31 mcg/kg/min`). La causa: `normalize_unit("mcg/kg/min")` devolvía `None` y
`extract_final_unit` solo reconocía unidades simples, así que `units_equivalent` daba `False`
para cualquier respuesta en esa unidad, fuera correcta o no. Lo arreglamos en `rlm/rewards.py`
(17 líneas: reconoce `mcg/kg/min` con las grafías `mcg`, `µg`, `μg` y `ug`) y añadimos
`tests/test_units.py`. Antes de darlo por bueno comprobamos que no cambia nada de lo anterior:
de las 849 respuestas guardadas en test (los cinco modelos evaluados, con bloque `<answer>`),
ninguna cambia de veredicto de unidad, y las unidades de train (`mL`, `mL/h`, `mg`,
`gotas/min`, con sus alias) se normalizan igual que antes; por eso los modelos entrenados y sus
resultados en test no se ven afectados. Las respuestas OOD ya generadas se puntuaron de nuevo
con `--rescore`, sin volver a generar.

**Qué pasó.** Con el verificador corregido:

| Modelo | Test | OOD | Caída (puntos) |
|---|---|---|---|
| sft | 80,5 % [74,5; 85,4] (161/200) | 24,0 % [16,7; 33,2] (24/100) | +56,5 [45,5; 65,3] |
| grpo_sin_kl | 89,5 % [84,5; 93,0] (179/200) | 30,0 % [21,9; 39,6] (30/100) | +59,5 [48,7; 68,3] |

El repuntuado no cambia ningún veredicto en test (0 de 200 en ambos modelos) y en OOD pasan a
correctas 4 respuestas del SFT y 3 de GRPO (ninguna pasa a incorrecta): los valores antes de
corregir eran 20/100 y 27/100.

| Familia (OOD) | n | sft | grpo_sin_kl |
|---|---|---|---|
| infusion | 50 | 20/50 (40,0 %) | 27/50 (54,0 %) |
| infusion_inverse | 50 | 4/50 (8,0 %) | 3/50 (6,0 %) |

| Tipo de resultado (OOD) | ok | valor correcto, unidad mal | valor mal | sin respuesta | formato válido |
|---|---|---|---|---|---|
| sft | 24 | 3 | 33 | 40 | 60,0 % |
| grpo_sin_kl | 30 | 4 | 31 | 35 | 65,0 % |

- Las 40 y 35 respuestas "sin respuesta" llegan todas al límite de 1024 tokens: no terminan.
  En los ejemplos que leímos, el modelo se enreda ("¿Qué es el error?") y sigue razonando.
- Las 7 respuestas con el valor correcto y la unidad mal son errores reales, no del verificador:
  contestan en `µg/min` o `mcg/min`, sin el `/kg`.
- De los fallos leídos a mano (seis), el SFT convierte mal ("5 mg es 5 mL") y en
  `infusion_inverse` multiplica por el peso en vez de dividir; GRPO comete errores de
  conversión (escribe 0,1 µg/kg/min como 0,001 mg/kg/min) o se queda sin tokens.
- Comparación pareada en OOD: solo el SFT acierta 9 problemas, solo GRPO 15, ambos 15 y ninguno
  61 (p = 0,307). En test eran 6 y 24 (p = 0,001).

**Qué concluimos.**

- Los dos modelos fallan mucho en la familia nueva: caen unos 57 y 60 puntos, y los intervalos
  de las dos caídas se solapan casi por completo. No podemos decir que GRPO se degrade menos.
- GRPO queda por encima del SFT también en OOD (30,0 % frente a 24,0 %), en la misma dirección
  que en test, pero con 100 problemas la diferencia no es significativa (p = 0,307). La ventaja
  fue clara en test (p = 0,001) y aquí no se sostiene. El experimento ni apoya ni descarta que
  "RL generaliza": la evidencia es débil y ruidosa.
- No podemos separar qué causa la caída, porque el OOD cambia varias cosas a la vez: la familia,
  el sistema de unidades (mcg/kg/min no aparece en ningún problema de train, ni en las trazas
  del SFT) y el nivel (todos son de nivel 4). Además, el 35-40 % de las respuestas no termina
  en 1024 tokens, así que "no sabe" y "no acaba" se mezclan. Entre las respuestas que sí
  terminan, aciertan 24/60 (SFT) y 30/65 (GRPO); es un dato descriptivo, no una comparación
  limpia.
- Limitaciones: 100 problemas; una sola generación greedy por problema; una sola semilla de
  entrenamiento por modelo. `reports/phase1_eval_ood.json` guarda los veredictos del momento
  de generar (20/100 y 27/100) y el gráfico `reports/phase1_pass1_ood.png` también; las cifras
  corregidas son las de esta entrada y de `reports/ood_comparison.md`.
- Siguientes pasos posibles, sin hacer: un OOD más suave con `--ood-mode weight` (mismas
  familias, pesos de 2 a 9,9 kg) para ver una degradación que no sea total; repetir con más
  tokens para separar los fallos por truncado; y más problemas por conjunto.

---
