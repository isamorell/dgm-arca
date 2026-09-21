# De dónde salen los cientos de problemas de la fase 1

Esta es la pregunta que más me han hecho, y es la buena: si la fase 1 necesita un dataset
verificable de vuestro dominio, ¿quién lo escribe? La respuesta corta es que **nadie lo
escribe a mano**. La respuesta larga es este documento.

## Primero, una distinción que lo cambia todo

En la fase 1 hay dos cosas distintas y se confunden constantemente:

1. **Los problemas**: pares (enunciado, respuesta correcta). Esto es lo que hay que
   conseguir, y de dónde sale es el tema de este documento.
2. **Las trazas de razonamiento**: el `<think>…</think>` que va dentro del ejemplo de SFT.
   Esto **no lo escribís vosotros ni a mano ni nunca**. Lo genera un modelo profesor y lo
   filtra vuestro verificador. Es exactamente lo que hace `rlm/distill.py`, y es la receta de
   Sky-T1, OpenThoughts y el arranque en frío de DeepSeek-R1.

O sea: si tenéis 400 problemas con su respuesta, generáis 4 trazas por problema con el
profesor, el verificador se queda con las correctas, y de ahí salen entre 600 y 1200 ejemplos
de SFT sin haber escrito una sola cadena de razonamiento. El coste está en GPU, no en horas
de anotación.

Lo mismo con GRPO: GRPO **no necesita trazas en absoluto**. Solo necesita el enunciado y la
respuesta correcta, porque el modelo genera sus propias respuestas y el verificador las
puntúa. El dataset de GRPO es el más barato de todos.

## ¿Cuántos problemas hacen falta de verdad?

Menos de los que parece, porque no estáis entrenando un modelo desde cero: estáis ajustando
un LoRA sobre un modelo de 0.6B a 1.7B con 16 GB de GPU.

| Para qué | Cuántos problemas únicos | Por qué |
|---|---|---|
| SFT de arranque en frío | 300 a 800 | Cada uno da 2-4 trazas verificadas; 1000-2000 ejemplos de SFT bastan de sobra para un LoRA |
| GRPO | 500 a 2000 | En la configuración del repositorio, un paso ve un prompt con 8 generaciones; 300-500 pasos son 300-500 visitas a prompts, con varias épocas sobre el conjunto |
| Test | 100 a 200 | Suficiente para que pass@1 tenga un error estándar razonable; de estos, unos 50 revisados a mano |
| Test fuera de distribución | 50 a 100 | Para el experimento de generalización, con problemas de una familia que no aparece en train |

DeepSeek usó "unos pocos miles" de ejemplos de arranque en frío para un modelo de 671B
parámetros. Vosotros tenéis tres órdenes de magnitud menos de modelo. No me traigáis 50
problemas, pero tampoco os obsesionéis con llegar a 10.000.

## Las seis formas de conseguirlos

Están ordenadas de mejor a peor para esta práctica. La mayoría de los equipos acabará
combinando dos o tres.

### 1. Generador programático: el verificador **es** el generador

Es la mejor opción y la que quiero ver por defecto. Si la tarea verificable de vuestro
dominio es un cálculo o una decisión basada en reglas publicadas, escribís tres funciones:

```
sample_params(rng)   ->  los datos del problema (peso, fechas, importes, concentraciones…)
solve(params)        ->  la respuesta correcta, con una implementación de referencia
render(params, rng)  ->  el enunciado en lenguaje natural, con varias plantillas
```

Y con eso tenéis miles de problemas en segundos. Lo importante, y es lo que convence:
`solve` es a la vez la implementación de referencia y el **verificador**, así que no hay
forma de que el ground truth esté mal. Tenéis un ejemplo completo y ejecutable en
[`rlm/generate_problems.py`](../rlm/generate_problems.py):

```bash
uv run python -m rlm.generate_problems --n 800 --split train --out rlm/data/train.jsonl
uv run python -m rlm.generate_problems --n 200 --split test  --out rlm/data/test.jsonl
uv run python -m rlm.generate_problems --n 100 --split ood   --out rlm/data/test_ood.jsonl
```

Esto encaja de forma natural con casi todos los temas de
[`temas_ejemplo.md`](temas_ejemplo.md): dosis pediátricas, plazos administrativos, cuotas de
IVA, TAE y amortización, estequiometría, umbrales de ayudas sociales, métricas de campaña.
En todos ellos las reglas están publicadas y la dificultad está en aplicarlas bien.

El trabajo intelectual no es generar: es **decidir la distribución**. Qué casos límite
incluís, con qué frecuencia, cómo evitáis que el 80 % de los problemas se resuelvan con la
misma rama del código. Eso sí lo miro.

### 2. Minar datos estructurados que ya existen

Vuestro dominio probablemente ya tiene la respuesta escrita en algún sitio, solo hay que
extraerla:

- Commits que corrigen un bug: el issue es el enunciado, el test añadido es el verificador.
- Fichas técnicas de medicamentos: la tabla de posología da el par (caso, dosis).
- Series del INE o de datos abiertos: la pregunta sobre la variación tiene respuesta en la serie.
- Registros de openFDA, expedientes del BOE, resoluciones publicadas.
- Vuestros propios logs o históricos, si trabajáis en el sector (anonimizados y con permiso).

Aquí el trabajo es de extracción y limpieza, y suele ser donde se va la mitad del tiempo de
la fase 1. Compensa: son problemas reales, no sintéticos.

### 3. Benchmarks públicos del dominio

Existen, son gratis y están en Hugging Face. Usarlos está permitido y a menudo es lo
sensato, siempre que **no sean lo único**: quiero ver algo propio del tema que habéis elegido.

| Área | Datasets públicos para empezar |
|---|---|
| Matemáticas (control) | `openai/gsm8k`, `MATH`, `nvidia/OpenMathInstruct-2` |
| Código | `openai/humaneval`, `mbpp`, `livecodebench`, `princeton-nlp/SWE-bench_Verified` |
| Salud y farmacia | `bigbio/med_qa`, `pubmed_qa`, `medmcqa`, datos de `openFDA` |
| Legal | `nguha/legalbench`, `casehold`, `lex_glue` |
| Finanzas | `ChanceFocus/flare-finqa`, `tat_qa`, `convfinqa` |
| Datos tabulares y SQL | `xlangai/spider`, `BIRD`, `wikitablequestions` |
| Ciencia | `allenai/sciq`, `gpqa`, `ARC` |

Una combinación que funciona bien: 70 % de problemas propios generados o minados, 30 % de un
benchmark público como grupo de control, para poder decir "mi modelo mejora en mi dominio sin
degradarse en lo general".

### 4. Construcción inversa desde el corpus

Tenéis que montar un corpus para la fase 3 de todas formas. Aprovechadlo: cogéis un fragmento
que contiene un dato duro (una dosis, un plazo, un tipo impositivo, una constante) y pedís a
un modelo que escriba un problema **cuya respuesta sea ese dato**. La respuesta correcta se
conoce por construcción, no hay que verificarla.

Es barato y da problemas muy pegados a vuestro dominio. Dos cautelas: revisad una muestra
(el modelo a veces escribe problemas ambiguos o mal planteados), y no metáis estos problemas
en el conjunto de test si el mismo fragmento está en el corpus del RAG, porque contamináis la
evaluación de la fase 3.

### 5. Pseudoetiquetas por consenso

Cuando tenéis muchos enunciados pero ninguna respuesta: generáis k respuestas con un modelo
fuerte (o con dos modelos distintos) y os quedáis solo con los problemas donde hay acuerdo
mayoritario. Es `self-consistency` usada como etiquetador.

Funciona, pero hereda los sesgos del profesor y hace que vuestro techo sea el suyo. Si lo
usáis: que sea una minoría del conjunto, que esté marcado en el JSONL con un campo
`label_source`, y que **nunca** entre en el test. Y decidlo en el informe.

### 6. Anotación a mano: solo para el test, y poco

Anotar a mano cientos de problemas es una mala inversión de vuestro tiempo y no lo voy a
pedir. Pero revisar a mano **50 problemas del test** sí, y ahí quiero ver vuestra firma: que
alguien del equipo ha leído cada uno, ha comprobado que el enunciado es resoluble y que la
respuesta es la que dice ser. Con eso detectáis los fallos sistemáticos de vuestro generador,
que es justo para lo que sirve.

## La división del trabajo que quiero ver

```
train.jsonl      generado o minado, miles si hace falta, sin revisión manual
test.jsonl       misma distribución, 100-200, con ~50 auditados a mano
test_ood.jsonl   familia de problemas ausente de train, 50-100, para el experimento de generalización
```

El conjunto fuera de distribución no es un extra: es lo que os permite abordar el reto de
"SFT memoriza, RL generaliza" que aparece en las diapositivas y en el README de la fase 1. Si
vuestro generador tiene parámetros, dejar una región fuera del entrenamiento es gratis.

## Los cuatro errores que voy a buscar

**Fuga entre train y test.** Con un generador es facilísimo: los mismos parámetros salen dos
veces. Deduplicad por hash de los parámetros (no del texto) y comprobad la intersección entre
ficheros. El ejemplo de `generate_problems.py` lo hace.

**Cero diversidad léxica.** Si los 800 enunciados tienen la misma estructura con los números
cambiados, el modelo aprende la plantilla, no la tarea, y en la defensa se nota enseguida.
Varias plantillas, orden de los datos variable, y si podéis, una pasada de paráfrasis con un
modelo. Dadme el número de plantillas y algún ejemplo de cada una.

**Todo el peso en la rama fácil.** Contad cuántos problemas caen en cada rama de vuestro
`solve`. Si el 90 % se resuelve con una multiplicación, vuestro pass@1 del 85 % no significa
nada. Quiero ver esa tabla en `EXPERIMENTS.md`.

**La respuesta filtrada en el enunciado.** Pasa sobre todo con la construcción inversa y con
las plantillas mal escritas: el enunciado contiene el resultado. Comprobación barata: buscad
la respuesta como subcadena del enunciado y mirad cuántas veces aparece.

## Qué pido en la propuesta

En la sección de tarea verificable de [`00_propuesta.md`](00_propuesta.md), además de qué se
verifica y cómo, decidme **con cuál de las seis estrategias vais a construir el conjunto** y
cuántos problemas esperáis conseguir. Si es la 1, describid los parámetros que vais a
muestrear. Si es la 2, enseñadme la fuente y una fila de ejemplo. Si es la 3, el dataset
concreto y qué añadís de vuestra cosecha.

Es la parte de la propuesta donde más os voy a apretar, porque es la que decide si la fase 1
es viable. Y si al escribirla os dais cuenta de que no hay forma de construir el conjunto, es
mejor saberlo en la semana 1 que en la 5.
