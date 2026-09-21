# Fase 0 — Propuesta de tema

Antes de escribir una línea de código, cada equipo me entrega una propuesta de una página.
No es burocracia: es la parte más difícil de la práctica y la que más va a determinar
vuestra nota. Un buen tema bien pensado hace que todo lo demás fluya; uno mal elegido os va a
dar problemas en cada fase.

Si no tenéis tema, en [`temas_ejemplo.md`](temas_ejemplo.md) hay una docena desarrollados con
su tarea verificable, sus herramientas y su corpus, y cómo se corregiría cada uno.

Rellenad esta plantilla, guardadla como `docs/propuesta.md` en vuestro repositorio y
enviadme el enlace. Hasta que os diga que está aprobada, no empecéis la fase 1: os ahorraréis
trabajo tirado. Cuando la apruebe, os devolveré por escrito qué se va a evaluar exactamente en
vuestro caso.

---

## Equipo

Nombres y correo de contacto.

## El tema en una frase

Qué hace el agente y para quién. Si no cabe en una frase, todavía no está claro.

## El usuario y su problema

¿Quién es la persona que usaría esto? ¿Qué hace hoy sin vuestro agente? ¿Qué le cuesta:
tiempo, dinero, errores? Si el usuario sois vosotros mismos, decidlo: es perfectamente válido.

## Diez preguntas o tareas reales

De la más fácil a la más difícil. Escribidlas tal como las diría el usuario, no como las
diría un ingeniero. Esto os obliga a pensar en concreto y a mí me sirve para calibrar.

1.
2.
3.
4.
5.
6.
7.
8.
9.
10.

## La tarea verificable (fase 1)

¿Qué tipo de problema de vuestro dominio tiene una respuesta que se puede comprobar
automáticamente, sin que nadie la lea? Un número, una fecha, un código que pasa tests, una
consulta que devuelve un resultado, una etiqueta. Describid:

- El tipo de problema y dos ejemplos con su respuesta.
- Cómo lo verificaríais con código: comparar números, ejecutar tests, comparar conjuntos...
- **Con cuál de las seis estrategias de [`datasets.md`](datasets.md) vais a construir el
  conjunto**, y cuántos problemas esperáis conseguir. Si es un generador, qué parámetros
  muestreáis; si es minería de datos, la fuente y una fila de ejemplo; si es un benchmark
  público, cuál y qué añadís de vuestra cosecha. Esta es la parte donde más os voy a apretar.

Dedicadle tiempo a esta sección. Si no encontráis respuesta, el tema no sirve tal como está
y hay que darle una vuelta. Venid a hablarlo.

## Las herramientas (fase 2)

Al menos tres, una de cada tipo:

- **Consulta fuera del modelo:** qué API o fuente externa real (con enlace a su documentación).
- **Cálculo o ejecución:** qué se calcula o ejecuta y por qué el modelo no debería hacerlo de cabeza.
- **Acción con efecto observable:** qué hace y cómo se comprueba que lo ha hecho.

## El corpus (fase 3)

- De dónde salen los documentos y cuántos hay (aproximadamente, en documentos y en páginas).
- En qué formato están.
- Con qué licencia, o por qué podéis usarlos.
- Dos ejemplos de pregunta que solo se pueden responder leyendo el corpus.

## Qué puede salir mal

Los dos o tres riesgos que veis (la API tiene límite de peticiones, el corpus está en
imágenes, el verificador es ambiguo...) y qué haríais en cada caso.

## Por qué este tema

Dos o tres líneas honestas. "Porque me interesa" es una razón perfectamente buena.
