# Fase 4 — El agente: todo junto en un bucle

**Endpoint que se evalúa:** `POST /agent` · **Clase:** The augmented LLM, bloque de agentes

## De qué va

Un agente LLM es un modelo con herramientas que actúa en bucle: piensa, actúa, observa, y
vuelve a pensar hasta que puede responder. Es el patrón ReAct que vimos en clase, y es donde
se juntan las tres fases anteriores: el cerebro es vuestro modelo de razonamiento, las manos
son vuestras herramientas, y la base de conocimiento es una herramienta más. Vimos también la
diferencia entre agentes que emiten JSON y agentes que escriben código, y jugamos con
`smolagents`. Aquí escribís el vuestro.

## Lo que tendréis que hacer

**1. El bucle ReAct.** En `agent/react_agent.py`: Pensamiento, Acción, Observación, hasta que
el modelo llama a `final_answer` o se agota el número de pasos. Con límite de tiempo por
herramienta y recuperación de errores: si una herramienta falla, el agente tiene que
enterarse y decidir, no morir. Reutilizad lo que ya tenéis: el parser y el registro de la
fase 2, y la memoria de `agent/memory.py`. El retriever de la fase 3 entra como la herramienta
`search_knowledge_base`.

**2. Un cerebro intercambiable.** El agente tiene que funcionar con el modelo base
(`brain="base"`), con vuestro modelo de la fase 1 (`brain="rlm"`) y con un modelo thinking
(`brain="thinking"`). Y vais a comparar los tres en el banco de tareas. Es una de las
preguntas más interesantes de la práctica: ¿ayuda de verdad haber entrenado el modelo para
razonar cuando lo pones a actuar?

**3. Memoria.** `ConversationMemory` ya cuenta tokens y sabe comprimir; lo que decidís
vosotros es la política: qué se resume, cuándo, qué se conserva siempre. Recordad el efecto
"perdido en el medio": meter todo en el contexto no es la solución.

**4. Un banco de tareas.** Una veintena en `agent/bench/tasks.jsonl` (formato en
`example_tasks.jsonl`) que requieran encadenar al menos dos herramientas, con una forma
automática de comprobar la respuesta final. `agent/evaluate.py` mide tasa de éxito, pasos y
tokens por cerebro. Y comparad vuestro agente JSON con un agente de código: aquí sí podéis
usar el `CodeAgent` de `smolagents` como contraste, y contarme si se cumple lo que decía el
artículo de CodeAct.

**5. La traza a la vista.** Cada ejecución devuelve todos los pasos, y `agent/ui.py` los
pinta como tarjetas de colores, igual que el notebook de clase. Es vuestra herramienta de
depuración y lo que enseñaréis en la demo.

## Para ir más allá

- **Multi-agente:** un planificador que delega en sub-agentes especializados de vuestro dominio.
- **Cerrar el círculo con la fase 1:** usar las trazas exitosas del agente como datos de SFT y
  medir si el agente mejora.
- **Guardarraíles:** detección de bucles (la misma llamada repetida), presupuesto de tokens
  por tarea, y qué pasa con las herramientas que piden confirmación.

## Cómo se corrige

Llamaré a `POST /agent` con tareas parecidas a las de vuestro banco, con alguna que requiera
la base de conocimiento y una herramienta, y con alguna imposible, y miraré:

- **Implementación (30 %):** el agente completa las tareas, la traza refleja lo que pasó de
  verdad (pensamiento, acción con argumentos, observación real), respeta `max_steps`, y los
  tres cerebros funcionan. Ante una tarea imposible, lo dice en vez de inventarse la respuesta.
- **Completitud (30 %):** bucle propio con `final_answer`, timeouts y recuperación de errores;
  base de conocimiento como herramienta; tres cerebros comparados; memoria con resumen; banco
  de veinte tareas con métricas; comparación JSON vs código; visor de trazas.
- **Interpretación (30 %):** qué cerebro gana y por qué; en qué tareas falla el agente y qué
  hace cuando falla; qué cambió con la memoria; qué visteis en la comparación con el agente
  de código; cómo reaccionó ante una herramienta rota.
- **Limpieza (10 %):** el agente arranca desde `docker compose up api` con las variables de
  `.env.example`, sin rutas absolutas ni secretos.

## Ficheros

| Fichero | Estado | Qué es |
|---|---|---|
| `prompts.py` | hecho | Prompt de sistema ReAct y prompt de resumen |
| `memory.py` | hecho + tu turno | Memoria con presupuesto; la política de compresión es vuestra |
| `react_agent.py` | tu turno | El bucle; `final_answer` y `search_knowledge_base`; detrás de `/agent` |
| `evaluate.py` | hecho | Éxito, pasos y tokens por cerebro sobre el banco de tareas |
| `ui.py` | hecho | Visor de trazas en Gradio contra la API |
| `bench/` | vuestro | Tareas (`example_tasks.jsonl` muestra el formato) |
