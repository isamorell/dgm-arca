# Corpus para el sistema RAG

## 1. Descripción

El código para construir el corpus utilizado por el sistema RAG de la práctica se encuentra en el fichero `rag/download_corpus.py`. Este código descargará y guardará el corpus en la carpeta rag/corpus; además de guardar la información original sin procesamiento en la carpeta rag/raw.

El corpus se obtiene a partir de las fichas técnicas de medicamentos publicadas en **CIMA (Centro de Información Online de Medicamentos), de la Agencia Española de Medicamentos y Productos Sanitarios (AEMPS)**, mediante su API REST.

El objetivo es proporcionar al sistema RAG información farmacológica estructurada y trazable que pueda recuperarse para responder consultas relacionadas con medicamentos, en el contexto de la formación de estudiantes de Enfermería.

## 2. Fuente de datos y criterios de selección

Los documentos se descargan de la API REST de CIMA:

- **Fuente:** [CIMA — AEMPS](https://cima.aemps.es/cima/publico/home.html).
- **Documentación de la API:** [CIMA REST API](https://cima.aemps.es/cima/resources/docs/CIMA_REST_API.pdf).
- **Tipo de documento:** ficha técnica del medicamento.
- **Formato del corpus:** Markdown (`.md`).

La generación del corpus parte de una lista predefinida de principios activos en `rag/download_corpus.py`. Para cada principio activo, el script busca medicamentos que cumplan los siguientes criterios:

- Que estén autorizados y comercializados, según los filtros utilizados en la consulta a CIMA.
- Que dispongan de una ficha técnica con secciones disponibles.
- Que tengan una vía de administración compatible con los criterios definidos en el código, que incluyen las vías intravenosa, intramuscular, subcutánea y parenteral.
- Que cumplan, cuando corresponda, los requisitos adicionales para combinaciones de principios activos.

Cuando hay varios candidatos válidos, se priorizan los medicamentos con vía intravenosa y, después, el menor número de registro CIMA.

La lista de principios activos abarca diferentes grupos farmacológicos, entre ellos:

- Antibióticos y antiinfecciosos.
- Analgésicos y antiinflamatorios.
- Vasopresores y medicamentos cardiovasculares.
- Electrolitos.
- Anticoagulantes.
- Sedantes y anestésicos.

La lista concreta y los criterios de selección están definidos en `rag/download_corpus.py`.

## 3. Contenido de las fichas técnicas

Para cada medicamento seleccionado se conservan las secciones especificadas en la constante `SECTIONS` del script:

| Sección | Contenido |
|---|---|
| 2 | Composición cualitativa y cuantitativa |
| 4.2 | Posología y forma de administración |
| 6.2 | Incompatibilidades |
| 6.3 | Periodo de validez |
| 6.4 | Precauciones especiales de conservación |
| 6.5 | Naturaleza y contenido del envase |
| 6.6 | Precauciones especiales de eliminación y otras manipulaciones |

El script convierte el contenido HTML de estas secciones en texto plano, conservando la estructura básica necesaria para facilitar su lectura y posterior procesamiento.

Cada documento Markdown incluye:

- Nombre comercial del medicamento.
- Principio activo utilizado en la búsqueda.
- Número de registro CIMA.
- Laboratorio titular.
- Forma farmacéutica.
- Vías de administración.
- Enlace a la ficha técnica original.
- Fecha de última actualización de la ficha, cuando está disponible.
- Fecha de descarga.
- Contenido de las secciones seleccionadas.

## 4. Generación y actualización del corpus

Para generar el corpus por primera vez, ejecutamos el siguiente comando:

`uv run python -m rag.download_corpus`

Si el corpus ya ha sido generado y lo que queremos es volver a descargar los medicamentos registrados previamente, debemos ejecutar el siguiente comando:

`uv run python -m rag.download_corpus --from-manifest`

En este modo se utilizan los números de registro CIMA almacenados en `rag/corpus_manifest.csv`, en lugar de realizar una nueva selección de medicamentos por principio activo.

Esto permite mantener la selección de medicamentos entre ejecuciones, aunque **no garantiza recuperar exactamente las mismas versiones históricas de las fichas técnicas**, ya que CIMA puede actualizar sus documentos.

Para probar el proceso con los primeros k principios activos, podemos ejecutar el siguiente comando tomando, en este caso, k=5:

`uv run python -m rag.download_corpus --limit 5`

Esta opción es útil para comprobar la conectividad con la API y validar el proceso de generación antes de construir el corpus completo.

## 5. Consideraciones legales y uso responsable

Los documentos proceden de las fichas técnicas publicadas por CIMA (AEMPS). Antes de redistribuir el corpus o publicar sus contenidos, debe comprobarse el aviso legal vigente de la AEMPS y respetarse las condiciones aplicables a la reproducción de sus documentos.

- [Aviso legal de la AEMPS](https://www.aemps.gob.es/avisoLegal/)

Según lo establecido en la fecha de creación del corpus (28/10/2026), estamos autorizadas a reproducir total o parcialmente el contenido de la web, citando debidamente que la información procede de las fichas técnicas publicadas por CIMA. La fecha de actualización de la ficha y de descarga se pueden encontrar al comienzo de cada documento del corpus.

El corpus está destinado a fines académicos y al desarrollo de un sistema RAG experimental.