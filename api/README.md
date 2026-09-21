# `api/` — La API de evaluación

Cada fase termina en un endpoint, y la corrección se hace llamándolos. Por eso los contratos
de entrada y salida están fijados en `api/schemas.py`: el script de corrección envía los
cuerpos que se definen ahí y valida las respuestas contra esos modelos. Podéis añadir campos
opcionales a las respuestas (más metadatos siempre son bienvenidos), pero no renombréis ni
quitéis los que existen.

```bash
uv run arca-api            # http://localhost:8000/docs
docker compose up api
```

| Método y ruta | Fase | Entrada | Salida |
|---|---|---|---|
| `GET /health` | — | — | Estado de cada fase: `ready`, `pending` o `error`, con detalle |
| `POST /reasoning` | 1 | `question`, `expected_answer?`, `max_new_tokens?` | `thinking`, `answer`, `raw`, `has_valid_format`, `verifier?`, `tokens_generated`, `model` |
| `POST /tools` | 2 | `query`, `max_turns?` | `answer`, `tool_calls[]` (nombre, argumentos, resultado, ok, error, latencia), `turns`, `model` |
| `POST /rag` | 3 | `question`, `top_k?`, `retriever?` (`dense`, `bm25`, `hybrid`) | `answer`, `chunks[]` (id, fuente, puntuación, texto), `citations[]`, `retriever`, `model` |
| `POST /agent` | 4 | `task`, `max_steps?`, `brain?` (`base`, `rlm`, `thinking`) | `final_answer`, `steps[]` (pensamiento, acción, observación), `n_steps`, `succeeded`, `model`, `tokens_used` |

Una fase que todavía no está implementada responde `501` con el nombre de la fase y un
mensaje. Así la API siempre arranca y podéis entregar por partes.

## Cómo se conecta cada fase

`api/backends.py` carga cada implementación de forma perezosa y la reutiliza. No hace falta
tocarlo: basta con implementar los puntos de entrada que espera.

| Fase | Punto de entrada | Configuración |
|---|---|---|
| 1 | `rlm.inference.ReasoningModel` | `ARCA_RLM_ADAPTER`, `ARCA_RLM_BASE_MODEL`, `ARCA_RLM_VERIFIER` |
| 2 | `tool_use.executor.ToolLoop` | herramientas registradas en `tool_use.tools.default_registry` |
| 3 | `rag.generate.RagPipeline` | corpus en `rag/corpus/`, índice en `rag/index/` |
| 4 | `agent.react_agent.ReActAgent` | `ARCA_THINKING_MODEL` y las de la fase 1 |

Rellenad también `ARCA_TEAM` y `ARCA_DOMAIN` en `.env`: salen en `/health` y me ayudan a
saber a quién estoy corrigiendo.

## Para la entrega

La API tiene que levantar con `docker compose up api` en una máquina limpia, con los modelos
cargándose al arrancar o en la primera petición. Lo normal es entrenar en la DGX, descargar el
adaptador (pesa poco) y servir la API desde vuestro portátil o una máquina en la nube; dentro
de la DGX no hay Docker ni puertos públicos. Para la corrección la expondréis con
[ngrok](https://ngrok.com/) (`ngrok http 8000`) y me pasaréis la URL. Antes de pasármela,
probad vosotros mismos los cuatro endpoints desde `/docs` y comprobad que `/health` los
marca como `ready`.

Los tests de `tests/test_api.py` comprueban que la API arranca con todo pendiente y que la
validación de entrada funciona. Añadid tests con vuestras implementaciones cuando las tengáis.
