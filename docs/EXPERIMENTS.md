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
