# Rúbrica

Los cuatro criterios son los mismos que en las prácticas anteriores. Se aplican a cada una
de las cuatro fases por separado, y cada fase pesa lo mismo en la nota final. La propuesta
(fase 0) no lleva nota numérica, pero sin propuesta aprobada no se corrige nada.

| Criterio | Peso |
|---|---|
| Implementación correcta | 30 % |
| Completitud | 30 % |
| Interpretación y explicación | 30 % |
| Limpieza y calidad general | 10 % |

Como cada equipo trabaja sobre un tema distinto, cuando apruebe vuestra propuesta os
devolveré por escrito la concreción de estos criterios para vuestro caso: qué cuenta como
tarea verificable, qué herramientas espero ver, qué tamaño de corpus es razonable. Lo que
sigue es la versión general.

## Implementación correcta (30 %)

El endpoint de la fase responde y hace lo que dice que hace. Lo compruebo con un script que
lanza peticiones a vuestra API, algunas parecidas a vuestros propios bancos de pruebas y
otras que no habéis visto.

- Fase 1: `thinking` y `answer` separados, formato válido en la mayoría de las respuestas,
  el veredicto del verificador coincide con la respuesta esperada tan a menudo como dice
  vuestro `evaluate.py`.
- Fase 2: herramienta correcta con argumentos correctos, resultado real en la traza, ninguna
  llamada cuando no hace falta.
- Fase 3: fragmentos relevantes, citas que existen, los tres retrievers funcionan, "no lo sé"
  cuando el corpus no tiene la respuesta.
- Fase 4: tareas completadas, traza fiel, `max_steps` respetado, tres cerebros operativos.

## Completitud (30 %)

Todo lo que el README de la fase describe como "lo que tendréis que hacer" está hecho. Cada
pieza que falte resta en proporción a su peso en la fase. Los retos de "ir más allá" no
cuentan aquí: cuentan en interpretación.

## Interpretación y explicación (30 %)

Aquí se juega la diferencia entre un trabajo correcto y uno excelente.

- Las curvas y las métricas están **explicadas**, no solo pintadas: qué pasó, por qué, qué
  esperabais y qué os sorprendió.
- Hay comparaciones y ablaciones: cambiasteis una cosa, medisteis, sacasteis una conclusión.
- Hay análisis de fallos concretos, con ejemplos.
- Los retos de "ir más allá" que hayáis abordado, y lo que aprendisteis de ellos.
- El cuaderno de experimentos es honesto y está fechado.
- **La defensa oral.** Diez minutos por equipo: tres de demo en vivo y el resto de preguntas
  sobre vuestro código y vuestros resultados. Sabéis qué hace el recorte en GRPO y podéis
  señalarlo en vuestro `grpo_step.py`. Sabéis por qué elegisteis ese λ. Sabéis qué pasó cuando
  le quitasteis el modelo de razonamiento al agente. Si el código lo escribió un agente de
  código y no sabéis explicarlo, se nota, y puntúa aquí.

## Limpieza y calidad general (10 %)

- Sin huecos sin rellenar, sin marcadores pendientes, sin código muerto.
- `uv sync` reproduce el entorno; `uv run pytest` está en verde; `docker compose up api` levanta.
- README de portfolio: qué hace, para quién, demo, arquitectura, métricas, cómo ejecutarlo.
- Código en inglés con docstrings; texto para personas en español.
- Pesos, índices y datos grandes fuera del repositorio, con el script que los regenera.
- Ningún secreto en el código.

## Lo que resta

- Un endpoint que no arranca o que no respeta el contrato de `api/schemas.py`.
- Resultados que no se pueden reproducir con los comandos del README.
- Métricas sin explicación, o explicaciones que no se corresponden con las métricas.
- Marcadores `TODO` en la entrega en vez de una frase honesta en el informe.
- Copiar el corpus, el dataset o las herramientas de otro equipo.
