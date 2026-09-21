# Fase 2 — Acción: el modelo habla con el mundo

**Endpoint que se evalúa:** `POST /tools` · **Clase:** The augmented LLM, bloque de tool use

## De qué va

Un modelo de lenguaje consume texto y produce texto. No puede consultar el tiempo, ni
ejecutar código, ni escribir un fichero. Tool use es la forma de dárselo: la aplicación
describe al modelo qué herramientas existen, el modelo genera una llamada en un formato
reconocible, la aplicación la detecta, la ejecuta y le devuelve el resultado, y el modelo
sigue. Cinco pasos, que en clase dibujamos con `get_weather` y que aquí vais a implementar a
mano, sin frameworks, con herramientas de vuestro dominio.

## Lo que tendréis que hacer

**1. Tres herramientas reales.** En `tool_use/tools.py`. Una que consulte algo fuera del
modelo (una API HTTP de verdad, no simulada), una que calcule o ejecute (un intérprete
acotado, una función numérica, una consulta a una base de datos), y una que actúe con efecto
observable (escribir un fichero, crear un evento de calendario, enviar algo a un webhook de
pruebas). `get_weather` y `calculator` os las doy como ejemplo de cómo se escribe una: la
docstring es la descripción que lee el modelo y los tipos son el esquema. Las que tienen
efectos se registran con `requires_confirmation=True`; en la fase 4 decidiréis qué hace el
agente con eso.

**2. Definiciones precisas y argumentos validados.** El registro (`registry.py`) convierte
la función en JSON Schema y valida con Pydantic lo que el modelo genera antes de ejecutar
nada. Si la validación falla, el error vuelve al modelo como observación para que se corrija.
Esto ya está hecho; lo que os pido es que lo uséis bien y que vuestras descripciones sean lo
bastante precisas para que el modelo acierte los argumentos. Es más difícil de lo que parece.

**3. El flujo de cinco pasos.** En `tool_use/executor.py`: el prompt de sistema con las
definiciones (la plantilla es la de las diapositivas), la generación, el parser de
`<tool_call>` (os lo doy, es infraestructura), la ejecución, la reinyección del resultado
como mensaje con rol `tool`, y la respuesta final. Tiene que soportar varias llamadas
encadenadas en distintos turnos (multi-step) y varias llamadas en una misma generación
(paralelas). Sin `smolagents` ni nada parecido en esta fase: quiero ver el bucle.

**4. El banco de pruebas.** Unos treinta casos en `tool_use/bench/cases.jsonl` (formato en
`example_cases.jsonl`): para cada petición, la herramienta esperada y sus argumentos.
Incluid casos que **no** necesitan herramienta: un modelo que llama a una herramienta para
todo no las está usando bien. `tool_use/evaluate.py` calcula la precisión de selección y la
de argumentos. Y comparad el mismo modelo con y sin herramientas, como en el notebook de
clase.

## Para ir más allá

- La "opción 2" de las diapositivas: **SFT para tool use**. Generad un dataset sintético de
  llamadas a vuestras herramientas, ajustad el modelo de la fase 1 con él y medid la mejora
  en el banco de pruebas.
- Una herramienta **con estado** (SQLite) y tareas donde las acciones tienen consecuencias
  sobre llamadas posteriores.
- Una **política de seguridad**: qué herramientas piden confirmación, cómo se pide, qué pasa
  si el modelo insiste.

## Cómo se corrige

Llamaré a `POST /tools` con peticiones parecidas a las de vuestro banco de pruebas y alguna
que requiera dos herramientas encadenadas, y miraré:

- **Implementación (30 %):** el modelo elige la herramienta correcta con los argumentos
  correctos, el resultado real aparece en `tool_calls` con su latencia, y la respuesta final
  lo usa. Una petición que no necesita herramienta se responde sin llamar a ninguna.
- **Completitud (30 %):** tres herramientas de los tres tipos, definiciones desde Pydantic,
  flujo de cinco pasos con multi-step y paralelo, manejo de errores de validación, banco de
  treinta casos con métricas, comparación con/sin herramientas.
- **Interpretación (30 %):** por qué esas herramientas y no otras; qué descripciones
  funcionaron y cuáles no; en qué casos falla el modelo y por qué; qué cambió al pasar de
  single-step a multi-step.
- **Limpieza (10 %):** herramientas con docstrings claras, sin claves en el código, tests
  para las herramientas que se puedan probar sin red.

## Ficheros

| Fichero | Estado | Qué es |
|---|---|---|
| `registry.py` | hecho | `@tool`, definiciones JSON Schema, validación y ejecución con captura de errores |
| `parser.py` | hecho | Detecta `<think>` y uno o varios `<tool_call>` en la salida del modelo |
| `tools.py` | ejemplo + tu turno | `get_weather` y `calculator`; las vuestras van aquí |
| `executor.py` | tu turno | El flujo de cinco pasos; detrás de `/tools` |
| `evaluate.py` | hecho | Métricas de selección y argumentos sobre el banco de pruebas |
| `bench/` | vuestro | Casos de prueba (`example_cases.jsonl` muestra el formato) |
