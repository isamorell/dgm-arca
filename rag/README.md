# Fase 3 — Conocimiento: el modelo lee lo que no sabe

**Endpoint que se evalúa:** `POST /rag` · **Clase:** The augmented LLM, bloque de RAG

## De qué va

El conocimiento de un modelo se congela en su fecha de corte, y meterle documentos enteros en
el contexto no funciona: se pierde en el medio. RAG es la alternativa: recopilar documentos,
trocearlos, convertir los trozos en vectores, y en el momento de la pregunta recuperar los
pocos fragmentos relevantes y ponerlos en el prompt. En clase vimos cada paso: estrategias de
chunking, cómo se entrenan los embeddings instruction-aware de Qwen3, recuperación semántica,
BM25 e híbrida. Aquí lo montáis entero sobre los documentos de vuestro dominio, y lo medís.

## Lo que tendréis que hacer

**1. Un corpus real.** Documentos de verdad de vuestro dominio en `rag/corpus/`, en cantidad
suficiente para que recuperar tenga sentido: como referencia, decenas de documentos o varios
cientos de fragmentos. `rag/ingest.py` lee PDF, Markdown, texto y HTML. Si el corpus no cabe
en git o no es vuestro para redistribuirlo, dejad el script que lo descarga y decid de dónde
sale y con qué licencia.

**2. Dos formas de trocear, comparadas.** El chunking recursivo os lo doy hecho en
`rag/chunking.py`. El semántico, basado en embeddings de frase y un umbral de distancia, es
vuestro: el algoritmo está en la docstring y en las diapositivas. Comparadlos con números:
cuántos fragmentos salen, de qué longitud, y cuál recupera mejor sobre vuestro conjunto
dorado. Pintad las distancias entre frases consecutivas de un documento con el umbral, como
en clase: es la mejor forma de justificar el umbral que elijáis.

**3. Embeddings con instrucción.** `rag/embed.py` usa Qwen3-Embedding y embebe la consulta
con una instrucción de tarea, tal como vimos que se entrena. Cambiad la instrucción y mirad
qué pasa con la recuperación: es una palanca que pocos tocan.

**4. Tres retrievers.** En `rag/retriever.py`: BM25 (hecho), denso con Chroma o NumPy
(vuestro, persistido en `rag/index/` para no reembeber el corpus cada vez) e híbrido
(vuestro): `s = λ·s_bm25 + (1-λ)·s_denso` con las puntuaciones reescaladas a [0, 1]. El λ se
elige con datos, no a ojo.

**5. El conjunto dorado.** Este es el trabajo de verdad de la fase. Medio centenar de
preguntas en `rag/gold/gold.jsonl`, cada una con el identificador del fragmento que la
responde. Las escribís vosotros, leyendo el corpus. Con eso, `rag/evaluate.py` calcula
Recall@k y MRR para los tres retrievers y para las dos estrategias de chunking. Anotad las
preguntas donde dos fragmentos son igual de válidos: os dirán mucho de vuestro chunking.

**6. Respuestas con citas.** `rag/generate.py`: el prompt aumentado (la plantilla de clase,
con identificadores de fragmento), la generación y la extracción de citas. Cada afirmación
tiene que apoyarse en un fragmento citado, y hay una comprobación automática de que las citas
existen. Es la forma más sencilla de empezar a controlar las alucinaciones.

## Para ir más allá

- Un **reranker** (Qwen3-Reranker) sobre los primeros veinte resultados y el salto en MRR.
- **Reescritura de la consulta** con vuestro modelo de razonamiento de la fase 1 antes de
  recuperar (HyDE, multi-query), medida en el conjunto dorado.
- **Fidelidad** de las respuestas con un modelo juez, y un análisis de dónde alucina.
- El reto más bonito, que conecta con la pérdida contrastiva que vimos: **ajustar el modelo
  de embeddings** con pares pregunta-pasaje de vuestro dominio y comprobar si Recall@k mejora.

## Cómo se corrige

Llamaré a `POST /rag` con preguntas de vuestro conjunto dorado, con otras que yo redacte
leyendo vuestro corpus, y con alguna que el corpus no puede responder, y miraré:

- **Implementación (30 %):** los fragmentos devueltos son relevantes, las puntuaciones tienen
  sentido, los tres valores de `retriever` funcionan, las citas de la respuesta existen entre
  los fragmentos devueltos, y cuando el corpus no tiene la respuesta el modelo lo dice.
- **Completitud (30 %):** corpus con ingesta reproducible, dos chunkings comparados,
  embeddings con instrucción, tres retrievers, conjunto dorado de cincuenta preguntas con
  Recall@k y MRR, barrido de λ, generación con citas y comprobación automática.
- **Interpretación (30 %):** la tabla de chunking explicada; por qué gana el retriever que
  gana; cómo elegisteis λ; qué preguntas fallan y por qué; qué pasó al cambiar la instrucción.
- **Limpieza (10 %):** índice fuera de git y regenerable, script de ingesta que corre solo,
  licencias del corpus documentadas.

## Ficheros

| Fichero | Estado | Qué es |
|---|---|---|
| `ingest.py` | hecho | Lee el corpus (PDF, MD, TXT, HTML) como `Document` con id y fuente |
| `chunking.py` | hecho + tu turno | Recursivo hecho; semántico vuestro; estadísticas para la tabla |
| `embed.py` | hecho | Qwen3-Embedding con instrucción para consultas |
| `retriever.py` | hecho + tu turno | BM25 hecho; denso e híbrido vuestros |
| `generate.py` | tu turno | Prompt aumentado, generación, citas; detrás de `/rag` |
| `evaluate.py` | hecho + tu turno | Recall@k y MRR hechos; montar y comparar retrievers vuestro |
| `corpus/`, `gold/`, `index/` | vuestros | Documentos, conjunto dorado, índice persistido |
