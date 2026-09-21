# ARCA — Agente con Razonamiento, Conocimiento y Acción

**Práctica final de Modelos Generativos Profundos · Máster en Inteligencia Artificial · ICAI · 2026-2027**

Durante el cuatrimestre hemos visto cómo un modelo de lenguaje aprende a razonar, cómo se
le enseña a usar herramientas, cómo se le conecta a una base de conocimiento y cómo todo
eso se junta en un agente que actúa en bucle. En esta práctica vais a recorrer ese camino
completo, de principio a fin, sobre un tema que elegís vosotros.

La llamamos ARCA porque cada letra es una de las piezas que vais a construir: **R**azonamiento,
**C**onocimiento, **A**cción, y el **A**gente que las une. Y porque la metáfora es
intencionada: vais a montar vuestra propia arca con lo que decidáis meter dentro.

---

> **El enunciado en PDF está en [`docs/enunciado.pdf`](docs/enunciado.pdf)** (5 páginas). Es el
> resumen de todo esto; el detalle de cada fase está en el README de su carpeta.

## La idea: vosotros elegís el tema

No os voy a dar un dominio cerrado. Cada equipo escoge un problema real que le interese y
construye un agente experto en ese problema, pasando por las cuatro fases. Un asistente
para preparar unas oposiciones, un agente que lee la normativa de un ayuntamiento y rellena
instancias, un tutor de matemáticas que corrige y explica, un copiloto para autónomos que
calcula impuestos y recuerda plazos, un entrenador de programación competitiva que ejecuta
vuestras soluciones. Lo que queráis, con una condición: que resuelva un problema a alguien
de verdad, aunque ese alguien seáis vosotros. Si al terminar podéis poner el enlace en
vuestro currículum y enseñárselo a alguien en una entrevista, habréis hecho bien el trabajo.
En [`docs/temas_ejemplo.md`](docs/temas_ejemplo.md) hay una docena de temas desarrollados, de
farmacia a marketing, con cómo se corregiría cada uno. Son para inspirar, no para elegir de
una lista.

¿Por qué lo planteo así? Porque sé que vais a programar con ayuda de agentes de código, y
no os lo voy a prohibir. Precisamente por eso, lo que vale aquí no es escribir código: es
**decidir qué construir, saber si funciona y explicar por qué**. Un agente de código os
escribe un bucle GRPO en dos minutos. Lo que no puede hacer es decidir qué tarea de vuestro
dominio tiene una recompensa verificable, qué herramientas necesita de verdad vuestro
usuario, qué documentos merecen estar en el corpus, ni interpretar por qué la recompensa se
estanca en el paso 40. Eso es lo que os pido y eso es lo que evalúo.

Tres condiciones tiene que cumplir el tema para que las fases sean posibles:

1. **Una tarea con respuesta verificable automáticamente**: un número, una fecha, un código
   que pasa tests, una consulta que devuelve un resultado, una etiqueta. Sin verificador no
   hay recompensa, y sin recompensa no hay aprendizaje por refuerzo (fase 1).
2. **Herramientas con sentido en ese dominio**: algo que consultar fuera del modelo, algo
   que calcular, algo que hacer (fase 2).
3. **Documentación real** sobre el tema que se pueda recopilar y usar (fase 3).

Al principio del cuatrimestre me diréis qué tema queréis hacer. Con esa lista formaré los
equipos y, tema por tema, os diré qué se va a evaluar exactamente en vuestro caso. Los
criterios generales son los mismos para todos; la concreción se adapta a lo que cada equipo
haya elegido. Los detalles de equipos, fechas y entregables se irán publicando aquí.

---

## Cómo está organizada

La práctica sigue el orden de las clases. Cada fase vive en su carpeta, tiene su propio
README con lo que se espera y cómo se corrige, y termina en un endpoint de la API que yo
puedo llamar. Las cuatro fases se corrigen por separado, pero todas hablan del mismo tema.

| Fase | Carpeta | Qué construís | Endpoint | Clase |
|------|---------|---------------|----------|-------|
| 0 | [`docs/00_propuesta.md`](docs/00_propuesta.md) | La propuesta de tema | — | — |
| 1 | [`rlm/`](rlm/README.md) | Un modelo que razona: SFT por destilación + GRPO con recompensas verificables | `POST /reasoning` | Reasoning language models |
| 2 | [`tool_use/`](tool_use/README.md) | Herramientas reales y el flujo de tool use escrito a mano | `POST /tools` | The augmented LLM: tool use |
| 3 | [`rag/`](rag/README.md) | Corpus, chunking, embeddings, BM25, híbrido, citas | `POST /rag` | The augmented LLM: RAG |
| 4 | [`agent/`](agent/README.md) | El bucle ReAct con todo lo anterior dentro | `POST /agent` | The augmented LLM: agents |
| — | [`api/`](api/README.md) | La API que expone las cuatro fases | `GET /health` | — |
| — | [`smoke/`](smoke/README.md) | Diagnóstico de GPU y entrenamiento de prueba para la primera sesión en la DGX | — | — |

El orden no es casual. Primero enseñamos al modelo a pensar. Luego le damos manos. Luego
memoria externa. Y al final lo ponemos a trabajar solo, usando todo lo anterior: en la fase 4
reutilizaréis el modelo de la fase 1, las herramientas de la fase 2 y el retriever de la fase 3.

En [`docs/`](docs/) tenéis la plantilla de la propuesta, los temas de ejemplo, la guía para
construir el dataset de la fase 1, la rúbrica detallada, la guía de la DGX, la de GitHub, la
plantilla del informe y la del cuaderno de experimentos.

---

## Empezar

Este repositorio es una plantilla. Pulsad **Use this template** en GitHub para crear vuestro
propio repositorio con una copia limpia (mejor que un fork: será vuestro, sin historial
ajeno, y es el que enseñaréis como portfolio). Después clonad el vuestro. Si no tenéis cuenta
de GitHub, o no sabéis cómo hacer `push` desde la DGX, está explicado paso a paso en
[`docs/github.md`](docs/github.md).

Necesitáis Python 3.11 y [`uv`](https://docs.astral.sh/uv/). Con eso:

```bash
git clone https://github.com/kendrickcetina/dgm-arca.git
cd dgm-arca
uv sync --extra train --extra rag --extra agent   # o `make setup`
uv run pytest                                     # todo verde antes de tocar nada
uv run arca-check-gpu                             # ¿ve torch la GPU?
uv run arca-smoke                                 # entrenamiento GRPO de prueba (GPU)
uv run arca-api                                   # API en http://localhost:8000/docs
```

En la DGX se trabaja exactamente así, con `uv`, desde la terminal de Code Server: no hay
SSH ni Docker dentro de las sesiones, cada uno recibe una GPU de 16 GB y las sesiones duran
como máximo 24 horas. La guía completa, incluida la prueba de la primera sesión, está en
[`docs/dgx.md`](docs/dgx.md). Leedla antes de conectaros.

Docker es para lo otro: que la API se despliegue con una orden en cualquier máquina, para la
corrección y para vuestro portfolio.

```bash
docker compose build
docker compose run --rm check-gpu
docker compose up api
```

Las dependencias están separadas en grupos para que instaléis solo lo que necesitéis:
`train` (torch, transformers, trl, peft), `rag` (sentence-transformers, chromadb, bm25),
`agent` (smolagents, gradio). Sin extras se instala lo justo para la API y los tests.

---

## Qué os doy hecho y qué es vuestro

Hecho, para que veáis código terminado desde el primer día y no solo esqueletos:

- La API con los cuatro endpoints y sus contratos fijados (`api/schemas.py`). Una fase sin
  implementar responde 501 con un mensaje claro, así que la API siempre arranca.
- Las recompensas de formato y exactitud de la fase 1, con tests (`rlm/rewards.py`).
- Los verificadores numérico y de texto exacto (`rlm/verifier.py`).
- El parser de llamadas a herramientas, el registro que convierte funciones tipadas en
  definiciones JSON, y dos herramientas de ejemplo (`tool_use/`).
- La ingesta de documentos, el chunking recursivo, el retriever BM25 y las métricas
  Recall@k y MRR (`rag/`).
- La memoria con presupuesto de tokens y el visor de trazas (`agent/`).
- El smoke test de GRPO y el diagnóstico de GPU (`smoke/`), Docker, `uv`, tests, `Makefile`.

Vuestro, marcado en el código con bloques **"Tu turno"** que explican qué hay que hacer y
por qué: el dataset verificable y el verificador de vuestro dominio, la destilación, el SFT,
el GRPO con vuestra tercera recompensa, el paso de GRPO a mano, vuestras herramientas y el
flujo de tool use, el chunking semántico, el retriever denso y el híbrido, la generación con
citas, el bucle ReAct, la política de memoria, y todas las evaluaciones. El README de cada
carpeta lo detalla.

---

## Cómo se evalúa

Los criterios son los de siempre y se aplican a cada una de las cuatro fases por separado,
de modo que cada fase pesa lo mismo en la nota final:

| Criterio | Peso | Qué miro |
|----------|------|----------|
| Implementación correcta | 30 % | El endpoint responde y hace lo que dice. Lo compruebo lanzando peticiones a vuestra API con un script de corrección. |
| Completitud | 30 % | Todo lo que el README de la fase describe como "lo que tendréis que hacer" está hecho. Cada pieza que falte resta. |
| Interpretación y explicación | 30 % | Curvas y métricas explicadas, no solo pintadas. Comparaciones, análisis de fallos, retos abordados, cuaderno de experimentos honesto y la defensa oral. |
| Limpieza y calidad | 10 % | Sin huecos ni marcadores pendientes, `uv sync` reproduce el entorno, los tests pasan, Docker levanta, README de portfolio, código en inglés con docstrings, sin secretos. |

El detalle por fase está en [`docs/rubrica.md`](docs/rubrica.md) y en el README de cada
carpeta. Cuando conozca vuestro tema os concretaré qué significa cada criterio en vuestro caso.

---

## La entrega

Cuando terminéis tendréis un sistema completo. Tratadlo como lo que es: algo que podéis enseñar.

- Repositorio público con un README de portfolio: qué hace, para quién, un GIF de la demo,
  un diagrama, la tabla de métricas de las cuatro fases y cómo levantarlo con una orden.
- `docker compose up api` arranca la API con los cuatro endpoints. Para la corrección la
  expondréis con [ngrok](https://ngrok.com/) y me pasaréis la URL.
- [`EXPERIMENTS.md`](docs/EXPERIMENTS_plantilla.md): el cuaderno fechado de qué probasteis,
  qué salió y qué decidisteis. Una entrada que diga "probamos X, no funcionó, creemos que por
  Y" vale más que diez que digan "todo bien".
- Un informe técnico de no más de diez páginas ([plantilla](docs/informe_plantilla.md)).
- Una defensa oral de unos diez minutos: tres de demo en vivo y el resto de preguntas sobre
  vuestro código y vuestros resultados.

---

## Reglas de la casa

- El texto dirigido a personas (README, informe, comentarios largos) en español. El código,
  los nombres de variables, funciones y ficheros, en inglés.
- Nada de `TODO` sueltos en la entrega. Lo que no hayáis hecho, decidlo en el informe.
- Los pesos, los índices y los datos grandes no van al repositorio. Dejad el script que los regenera.
- Ninguna clave ni token en el código. Usad `.env` (ya está en `.gitignore`).
- Usad los agentes de código todo lo que queráis, pero leed lo que os escriben. En la
  defensa el código es vuestro y las preguntas también.
- La DGX no tiene copias de seguridad. Nada que no esté también en GitHub, en Hugging Face
  Hub o en vuestro disco existe de verdad. Detalles en [`docs/dgx.md`](docs/dgx.md).

---

## Glosario rápido

- **RLM (Reasoning Language Model):** modelo entrenado para generar primero una cadena de
  razonamiento entre `<think>` y `</think>` y después la respuesta.
- **CoT (Chain of Thought):** los pasos intermedios de razonamiento. Prompting con CoT fue
  el punto de partida; SFT y RL lo convierten en un comportamiento aprendido.
- **SFT (Supervised Fine-Tuning):** ajuste con pares (pregunta, respuesta deseada). Para
  razonamiento, la respuesta incluye la traza. Cuando las trazas las genera un modelo mayor
  y se filtran con un verificador, se le llama destilación.
- **RLVR (Reinforcement Learning with Verifiable Rewards):** RL donde la recompensa la da un
  verificador determinista (¿acierta?, ¿pasa los tests?, ¿respeta el formato?).
- **GRPO (Group Relative Policy Optimization):** algoritmo de RL sin crítico. Muestrea un
  grupo de respuestas a la misma pregunta, calcula la ventaja de cada una respecto a la media
  del grupo y empuja hacia arriba las mejores. Con recorte del ratio de políticas y,
  opcionalmente, penalización KL.
- **Tool use:** el modelo genera una llamada en un formato reconocible, la aplicación la
  ejecuta y le devuelve el resultado.
- **ReAct (Reason + Act):** bucle Pensamiento → Acción → Observación hasta llegar a la respuesta.
- **RAG (Retrieval-Augmented Generation):** recuperar fragmentos relevantes de un corpus y
  meterlos en el prompt antes de generar.
- **Embeddings instruction-aware:** la consulta se embebe junto con una instrucción que
  define qué significa "relevante" para esa tarea.
- **BM25:** ranking por palabras clave (frecuencia, rareza, normalización por longitud).
  **Híbrido:** combinación ponderada de BM25 y similitud de embeddings.
