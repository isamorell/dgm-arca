# Informe técnico — plantilla

Máximo diez páginas, sin contar anexos. Escribidlo para alguien que sabe de modelos de
lenguaje pero no ha visto vuestro repositorio. Las gráficas van dentro, explicadas, no en un
anexo sin comentar. Si algo no funcionó, contadlo aquí: vale más que fingir que sí.

## 1. El problema y el usuario (media página)

Qué hace el agente, para quién, y qué hace esa persona hoy sin él. Un ejemplo de principio a
fin: la pregunta del usuario, la traza del agente, la respuesta.

## 2. Arquitectura (una página)

Un diagrama de las cuatro fases y cómo se conectan: qué modelo, qué herramientas, qué
corpus, cómo entra cada cosa en el agente. Qué decisiones tomasteis y qué alternativas
descartasteis.

## 3. Fase 1: razonamiento (dos páginas)

- El dataset y el verificador: de dónde salen, cuántos problemas, qué decide el verificador y
  qué no.
- La destilación: qué profesor, cuántas trazas, tasa de aceptación.
- SFT y GRPO: hiperparámetros y por qué; las tres recompensas y por qué esa tercera.
- Resultados: pass@1 base / SFT / GRPO; curvas de recompensa y longitud; cinco fallos
  analizados.
- Qué aprendisteis del paso de GRPO a mano.

## 4. Fase 2: herramientas (una página)

Las herramientas y por qué esas. El banco de pruebas y las métricas. Con y sin herramientas.
Dónde falla el modelo.

## 5. Fase 3: conocimiento (una página y media)

El corpus. La tabla de chunking. Recall@k y MRR por retriever y por chunking. Cómo elegisteis
λ. Qué preguntas del conjunto dorado fallan y por qué. Cómo comprobáis las citas.

## 6. Fase 4: agente (una página y media)

El bucle y sus decisiones (timeouts, errores, memoria, confirmaciones). El banco de tareas.
Tasa de éxito, pasos y tokens por cerebro. Agente JSON frente a agente de código. Ejemplos
de trazas buenas y malas.

## 7. Lo que no funcionó (media página)

Sin adornos. Qué intentasteis, qué pasó, qué creéis que lo explica.

## 8. Si tuviéramos diez veces más cómputo (media página)

Qué haríais distinto. Qué experimento os gustaría correr.

## Anexos

Enlace al repositorio, URL de la API durante la corrección, comandos para reproducir cada
resultado, y el `EXPERIMENTS.md` completo.
