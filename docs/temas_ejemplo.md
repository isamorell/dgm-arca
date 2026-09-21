# Temas de ejemplo: qué se puede hacer y cómo se corregiría

El tema lo elegís vosotros. Esta lista no es un menú cerrado: es para que veáis qué nivel de
ambición espero, cómo se traduce un tema en las cuatro fases, y cómo concreto la rúbrica
cuando sé de qué va vuestro proyecto. Si os inspira alguno, adelante, pero cambiadlo hasta
que sea vuestro. Si tenéis otro, mejor: traedlo a la propuesta.

Cada tema indica de dónde salen los problemas de la fase 1, porque es la pregunta que más me
hacéis. Ninguno se resuelve anotando a mano: las seis estrategias están en
[`datasets.md`](datasets.md).

Todos comparten la misma estructura, porque es la que tiene que tener vuestra propuesta:
un usuario con un problema, una **tarea verificable** para la fase 1, **tres herramientas**
para la fase 2, un **corpus** para la fase 3, y **tareas encadenadas** para la fase 4. Y
después, la parte que os interesa: qué miraría yo exactamente en ese tema y qué requisitos
añadiría a los generales. Los requisitos generales están en el README y en la rúbrica; los
que veis aquí son los que dependen del tema.

Una nota sobre dificultad. Ninguno de estos temas es fácil, y no debería serlo: para eso
tenéis cuatro meses y una GPU. Lo que marca la diferencia entre un tema difícil y uno
imposible es que la tarea verificable exista y el corpus sea accesible. Comprobad eso antes
de enamoraros de una idea.

---

## Farmacia y salud

### Asistente de interacciones y posología para farmacia comunitaria

**Usuario.** El farmacéutico que atiende el mostrador y tiene que responder en treinta
segundos si dos medicamentos se pueden tomar juntos o cuánto ibuprofeno le toca a un niño de
14 kilos.

**Tarea verificable (fase 1).** Cálculo de dosis pediátrica y ajuste renal a partir de peso,
edad y aclaramiento de creatinina, con respuesta numérica y unidades. Verificador numérico
con tolerancia y comprobación de unidades. Los problemas se generan a partir de las fichas
técnicas (posología por kg) con un generador propio, así que podéis tener miles.

**Herramientas (fase 2).** Consulta a la API de CIMA de la AEMPS (fichas técnicas y
prospectos reales); calculadora de dosis y de aclaramiento (Cockcroft-Gault); generador de
una hoja de consejo al paciente en PDF con las pautas.

**Corpus (fase 3).** Fichas técnicas de CIMA de unos 200 principios activos frecuentes, guías
de interacciones de la propia AEMPS y del Ministerio.

**Agente (fase 4).** "Paciente de 78 años con insuficiencia renal moderada, toma warfarina,
le han recetado claritromicina: ¿hay problema y qué dosis?" Requiere buscar en el corpus,
consultar CIMA, calcular y redactar la hoja.

**Cómo lo corregiría.** En la fase 1 miraría sobre todo el verificador de unidades (mg/kg/día
frente a mg/dosis es el error clásico) y que el conjunto de test tenga casos límite:
neonatos, obesidad, diálisis. En la fase 3 exigiría que el conjunto dorado incluya preguntas
sobre interacciones que solo están en la ficha técnica y no en el conocimiento general del
modelo. En la fase 4, que ante una interacción grave el agente lo diga primero y con la
fuente citada.

**Requisitos propios del tema.** Aviso claro en cada respuesta de que no sustituye el juicio
clínico. Nada de datos reales de pacientes: los casos son sintéticos. La tercera recompensa
de GRPO tiene que penalizar respuestas sin unidades o sin fuente.

**Riesgos.** Las fichas técnicas son PDF largos con tablas; el chunking va a ser vuestro
primer dolor de cabeza. Tratadlo como parte del trabajo, no como un obstáculo.

### Copiloto de farmacovigilancia: de la notificación al informe

**Usuario.** El técnico de farmacovigilancia que recibe notificaciones de sospechas de
reacciones adversas en texto libre y tiene que codificarlas y evaluar la causalidad.

**Tarea verificable.** Codificación MedDRA de la reacción descrita (etiqueta exacta contra un
diccionario acotado) y evaluación de causalidad con el algoritmo de Naranjo (puntuación
entera). Verificador de coincidencia exacta y numérico.

**Herramientas.** Consulta a la base pública de FAERS (openFDA) por fármaco y reacción;
calculadora de la puntuación de Naranjo a partir de respuestas estructuradas; generador del
formulario de notificación en XML o JSON con el esquema E2B simplificado.

**Corpus.** Guías de buenas prácticas de farmacovigilancia de la EMA, fichas técnicas de los
fármacos del dataset, boletines de seguridad de la AEMPS.

**Agente.** Recibe una notificación en texto libre y produce la codificación, la evaluación
de causalidad con sus criterios, la comparación con la frecuencia en FAERS y el borrador del
informe.

**Cómo lo corregiría.** Aquí la fase 1 es la difícil: la codificación exacta es una tarea de
clasificación con miles de etiquetas y el modelo pequeño va a sufrir. Valoraría mucho que
acotéis bien el diccionario y que analicéis en qué se confunde. En la fase 2, que la
herramienta de FAERS gestione bien los límites de la API y los sinónimos de fármacos.

**Requisitos propios.** Casos sintéticos o del propio openFDA, nunca de vuestra empresa.
Terminología MedDRA acotada a un subconjunto que podáis distribuir.

---

## Legal y administración

### Asistente de plazos y recursos administrativos

**Usuario.** Un ciudadano o una pequeña gestoría que recibe una notificación de la
administración y necesita saber qué puede hacer y hasta cuándo.

**Tarea verificable.** Cálculo de plazos administrativos: fecha límite para recurrir,
alegaciones o pagos a partir de la fecha de notificación, el tipo de acto y el calendario de
días hábiles (Ley 39/2015, con agosto inhábil en la vía judicial, festivos autonómicos, el
efecto de las notificaciones electrónicas). Respuesta: una fecha. Verificador de igualdad
de fechas. Los problemas se generan combinando tipos de acto, fechas y comunidades.

**Herramientas.** Consulta al BOE por identificador o materia (API de datos abiertos del
BOE); calculadora de plazos con calendario oficial de días inhábiles; generador de un
escrito de recurso o alegaciones en formato DOCX a partir de una plantilla.

**Corpus.** Ley 39/2015, Ley 40/2015, ley de la jurisdicción contencioso-administrativa,
ordenanzas de un ayuntamiento concreto, y las FAQ de la sede electrónica correspondiente.

**Agente.** "Me han notificado una multa de tráfico el 3 de julio por correo, vivo en Bilbao:
¿qué puedo hacer, hasta cuándo, y prepárame el escrito." Buscar la norma, calcular la fecha,
redactar.

**Cómo lo corregiría.** En la fase 1 el interés está en los casos difíciles del calendario:
notificación en viernes por la tarde, festivo local, agosto, notificación electrónica no
abierta. El conjunto de test los tiene que tener. En la fase 3 el corpus es texto legal muy
estructurado: quiero ver que aprovecháis esa estructura en el chunking (artículos, apartados)
y que el conjunto dorado exige recuperar el artículo exacto. En la fase 4, que el agente cite
el artículo y no se invente la norma.

**Requisitos propios.** Cada respuesta indica que es orientativa y remite a un profesional.
Las fechas se calculan con la herramienta, nunca de cabeza: la tercera recompensa de GRPO
puede premiar precisamente que el razonamiento use el calendario correcto.

### Revisor de contratos de alquiler frente a la ley de vivienda

**Usuario.** Un inquilino que recibe un contrato y quiere saber qué cláusulas son abusivas o
nulas antes de firmar.

**Tarea verificable.** Dado un contrato sintético con cláusulas etiquetadas, clasificar cada
cláusula como válida, nula o dudosa según la LAU y la ley de vivienda. Verificador de
coincidencia exacta de etiquetas por cláusula, con un dataset generado por vosotros
combinando cláusulas tipo.

**Herramientas.** Consulta de índices oficiales de referencia de precios de alquiler (API del
Ministerio); calculadora de actualización de renta con el índice que corresponda; generador
de un informe de revisión en Markdown o PDF con las cláusulas marcadas.

**Corpus.** LAU, ley de vivienda, sentencias resumidas sobre cláusulas abusivas, guías de
consumo de las comunidades autónomas.

**Cómo lo corregiría.** Valoraría la calidad del dataset de cláusulas: cuántas plantillas,
cuánta variedad léxica, cómo evitáis que el modelo memorice la redacción. Un modelo que
clasifica bien solo las cláusulas que ha visto redactadas igual no ha aprendido nada, y eso
se ve con una partición fuera de distribución.

---

## Finanzas

### Copiloto fiscal para autónomos

**Usuario.** Un autónomo que hace su propia contabilidad y tiene que presentar el IVA
trimestral y los pagos fraccionados del IRPF sin equivocarse.

**Tarea verificable.** Cálculo de cuotas: IVA del modelo 303, pago fraccionado del modelo 130,
retenciones, con facturas y gastos sintéticos como entrada. Respuesta numérica exacta al
céntimo. El verificador es aritmética, pero el dataset puede tener toda la casuística que
queráis: prorrata, bienes de inversión, operaciones intracomunitarias, gastos parcialmente
deducibles.

**Herramientas.** Consulta del calendario fiscal de la AEAT y de los tipos vigentes;
calculadora de liquidaciones; generador de un borrador del modelo en el formato de
importación de la AEAT o en PDF.

**Corpus.** Manuales prácticos de IVA e IRPF de la AEAT, consultas vinculantes de la
Dirección General de Tributos sobre autónomos, ley del IVA.

**Agente.** "Estas son mis facturas del trimestre en CSV, dime qué me toca pagar, si tengo
algo raro y prepárame el borrador." Leer el fichero, buscar la norma dudosa, calcular,
generar.

**Cómo lo corregiría.** En la fase 1, que el conjunto de test tenga casos donde el modelo
tiene que decidir la deducibilidad antes de calcular, no solo sumar. En la fase 2, la
herramienta que lee el CSV tiene que validar el esquema y devolver errores útiles cuando la
factura está mal. En la fase 4, valoraría que el agente detecte inconsistencias (una factura
sin NIF, un gasto duplicado) y las pregunte en vez de tragárselas.

**Requisitos propios.** Todo dato es sintético. Aviso de que no es asesoramiento fiscal. La
tercera recompensa puede penalizar respuestas que den un importe sin desglose.

### Analista de riesgo de crédito para préstamos al consumo (educativo)

**Usuario.** Un estudiante de finanzas o un analista junior que quiere entender cómo se evalúa
una solicitud de préstamo y qué condiciones son razonables.

**Tarea verificable.** Cálculo de TAE, cuota de amortización francesa, ratio de
endeudamiento y coste total con comisiones, a partir de la descripción de un préstamo.
Verificador numérico con tolerancia. Miles de problemas generables.

**Herramientas.** Consulta de tipos oficiales del Banco Central Europeo y del euríbor
(API del BCE); simulador de amortización que devuelve el cuadro completo; exportador del
cuadro a CSV o generador de un gráfico.

**Corpus.** Guías del Banco de España y de la CNMV para consumidores, ley de contratos de
crédito al consumo, memorias de condiciones de varias entidades.

**Cómo lo corregiría.** La fase 1 es fácil de verificar y difícil de aprender: la TAE es una
ecuación implícita y el modelo tiene que aprender a delegar en la herramienta o a aproximar
bien. Quiero ver esa tensión analizada. En la fase 4, que el agente compare dos ofertas y
explique cuál es mejor con números, no con adjetivos.

**Requisitos propios.** Ningún consejo de inversión personalizado; el agente explica, no
recomienda comprar nada. Datos sintéticos.

---

## Desarrollo de software

### Agente de triaje y reproducción de bugs para un proyecto open source

**Usuario.** El mantenedor de una librería con cien issues abiertas que necesita saber cuáles
son reproducibles, cuáles duplicadas y por dónde empezar.

**Tarea verificable.** Dado un issue y un fragmento del código, escribir un test que
reproduzca el fallo. Verificador: el test falla en la versión con el bug y pasa en la versión
corregida (ejecución en sandbox). Podéis construir el dataset a partir del historial de un
proyecto real: cada commit de corrección con su issue asociado es un problema.

**Herramientas.** API de GitHub (issues, commits, diffs); ejecutor de tests en un
contenedor o proceso aislado con límite de tiempo; creador de comentarios o etiquetas en el
issue (contra un repositorio de pruebas vuestro).

**Corpus.** Documentación del proyecto, su guía de contribución, los issues cerrados con su
resolución.

**Agente.** "Triángula el issue 4521: ¿es duplicado, es reproducible, en qué fichero está el
problema?" Buscar issues parecidos en el corpus, leer el código, escribir y ejecutar el test,
comentar.

**Cómo lo corregiría.** Este tema tiene el mejor verificador posible (tests que se ejecutan)
y el mayor riesgo de *reward hacking*: el modelo aprenderá a escribir tests que fallan por
cualquier motivo. Quiero ver cómo lo detectáis y cómo diseñáis la recompensa para evitarlo.
En la fase 2, la seguridad del ejecutor es parte de la nota: qué puede y qué no puede hacer
el código que ejecuta el agente.

**Requisitos propios.** El sandbox tiene límite de tiempo, de memoria y sin red. La
herramienta que escribe en GitHub apunta a un repositorio de pruebas y pide confirmación.

### Asistente de migración de código entre versiones de una librería

**Usuario.** Un equipo que tiene que migrar código de una versión mayor a otra de una
librería con cambios incompatibles (pandas, pydantic, transformers, la que elijáis).

**Tarea verificable.** Dado un fragmento que usa la API antigua, producir el equivalente en
la API nueva. Verificador: los tests del fragmento pasan con la versión nueva instalada. El
dataset se construye a partir de la guía de migración oficial y de commits reales de
proyectos que migraron.

**Herramientas.** Consulta de la documentación de la librería por símbolo (API o índice
propio); ejecutor de tests en dos entornos virtuales (versión vieja y nueva); generador de
un parche en formato diff.

**Corpus.** Guías de migración, changelogs, documentación de ambas versiones, discusiones
de issues sobre la migración.

**Cómo lo corregiría.** Que el conjunto de test incluya cambios de comportamiento silenciosos
(misma API, distinta semántica), que son los que de verdad duelen. Que la fase 3 demuestre
que el retriever distingue entre la documentación de las dos versiones, que se parecen mucho:
es un caso perfecto para embeddings con instrucción.

---

## Social y sector público

### Navegador de ayudas y prestaciones sociales

**Usuario.** Una trabajadora social o una persona en situación vulnerable que necesita saber a
qué ayudas tiene derecho y cómo pedirlas.

**Tarea verificable.** Elegibilidad y cuantía: dada una situación familiar y económica
sintética, determinar si cumple los requisitos del ingreso mínimo vital, de una ayuda
autonómica o del bono social, y cuánto le corresponde. Verificador de etiqueta (sí/no por
ayuda) y numérico (cuantía). Los criterios están publicados y son reglas, así que el dataset
se genera con un motor de reglas que vosotros escribís, y ese motor es a la vez el verificador.

**Herramientas.** Consulta al catálogo de ayudas de la administración (API de datos abiertos
o índice propio); calculadora de renta garantizada y umbrales por unidad de convivencia;
generador de la lista de documentos a presentar y del borrador de solicitud.

**Corpus.** Normativa del ingreso mínimo vital, decretos de rentas autonómicas, guías de la
Seguridad Social y de servicios sociales municipales, en lenguaje llano donde exista.

**Agente.** "Somos tres, cobro 900 euros, mi hijo tiene una discapacidad del 40 %, vivimos de
alquiler en Sevilla: ¿qué puedo pedir y qué necesito?" Buscar, calcular, listar documentos,
preparar el borrador.

**Cómo lo corregiría.** Aquí el lenguaje importa tanto como el cálculo: el usuario no es un
experto. En la fase 1 la tercera recompensa puede premiar respuestas en lenguaje llano y con
el siguiente paso claro. En la fase 4 valoraría que el agente pregunte lo que le falta antes
de responder, y que nunca diga "no tienes derecho" sin citar el requisito que falla.

**Requisitos propios.** Cero datos personales reales. Cada respuesta remite a servicios
sociales. El motor de reglas se entrega y se puede auditar.

### Verificador de afirmaciones en debates públicos locales

**Usuario.** Un periodista local o una asociación vecinal que quiere comprobar cifras que se
dicen en plenos municipales.

**Tarea verificable.** Dada una afirmación con una cifra ("el paro ha bajado un 12 % este
año") y acceso a los datos, determinar si es verdadera, falsa o imprecisa, y calcular la cifra
correcta. Verificador de etiqueta y numérico contra datos del INE o del ayuntamiento.

**Herramientas.** API del INE o de datos abiertos municipales; calculadora de variaciones,
tasas y comparaciones; generador de una tarjeta de verificación con la fuente y el cálculo.

**Corpus.** Actas de plenos, presupuestos municipales, notas de prensa, boletines
estadísticos.

**Cómo lo corregiría.** El interés está en la fase 2: que las herramientas de datos devuelvan
exactamente la serie que hace falta (periodo, ámbito, definición) y que el agente no compare
peras con manzanas. Y en la honestidad: cuando los datos no permiten verificar, el agente lo
dice.

---

## Marketing y negocio

### Analista de campañas con generación de informes verificables

**Usuario.** El responsable de marketing de una pyme que tiene los datos de campañas en hojas
de cálculo y quiere respuestas, no dashboards.

**Tarea verificable.** Métricas de campaña a partir de datos tabulares sintéticos: CAC, ROAS,
tasa de conversión por canal, test A/B con su significación estadística. Respuesta numérica
o etiqueta ("la variante B es significativamente mejor al 95 %"). Verificador numérico y
exacto. Datos y problemas generables sin límite.

**Herramientas.** Lector y consultor de datos tabulares (SQL sobre DuckDB o pandas
restringido); calculadora estadística (intervalos, tests de proporciones); generador de un
informe con gráficos en PDF o HTML.

**Corpus.** Documentación de las plataformas publicitarias sobre definición de métricas,
guías de buenas prácticas de experimentación, el histórico de informes de la propia empresa
(sintético).

**Agente.** "Dime qué canal deja de merecer la pena este trimestre y prepárame el informe
para dirección." Consultar datos, calcular, contrastar con la definición de la métrica,
redactar con gráficos.

**Cómo lo corregiría.** El peligro de este tema es que el modelo haga estadística de cabeza.
La tercera recompensa de GRPO y la evaluación de la fase 4 tienen que castigar cualquier cifra
que no venga de una herramienta. Valoraría un banco de tareas con trampas: datos con valores
faltantes, una campaña con muy pocas muestras, una métrica definida de dos formas distintas
en el corpus.

**Requisitos propios.** Los datos son sintéticos y se generan con un script. Cada cifra del
informe es trazable a una llamada a herramienta.

### Redactor de fichas de producto con verificación contra el catálogo

**Usuario.** Una tienda online con miles de productos y fichas incompletas o inconsistentes.

**Tarea verificable.** Extracción estructurada: dado el texto libre de un proveedor,
producir la ficha en JSON con un esquema estricto (dimensiones en unidades normalizadas,
materiales de una lista cerrada, compatibilidades). Verificador de esquema y de igualdad
campo a campo. Se puede construir un dataset grande a partir de catálogos públicos.

**Herramientas.** Consulta al catálogo existente por SKU o similitud; conversor de unidades
y normalizador de atributos; publicador de la ficha en una tienda de pruebas (API de una
plataforma de comercio electrónico en modo sandbox).

**Corpus.** Manual de estilo de la tienda, normativa de etiquetado del sector elegido,
fichas de productos ya publicadas.

**Cómo lo corregiría.** La fase 1 es un caso bonito de recompensa de formato llevada al
extremo: el esquema JSON es la recompensa. Quiero ver cómo evoluciona la tasa de JSON válido
durante GRPO y qué campos se resisten. En la fase 4, que el agente no publique nada que
contradiga al catálogo sin avisar.

---

## Ciencia y educación

### Tutor de química que balancea, calcula y consulta seguridad

**Usuario.** Un estudiante de primero de grado o de bachillerato que se atasca con
estequiometría y quiere entender, no copiar.

**Tarea verificable.** Balanceo de ecuaciones y cálculos estequiométricos con respuesta
numérica y unidades; verificador numérico con tolerancia y comprobación de conservación de
átomos. Problemas generables por miles a partir de plantillas de reacciones.

**Herramientas.** Consulta a PubChem por nombre o fórmula; calculadora de masas molares y
estequiometría con RDKit o una implementación propia; generador de hojas de ejercicios con
solución en PDF.

**Corpus.** Fichas de seguridad, un libro de texto de licencia abierta, problemas resueltos.

**Cómo lo corregiría.** Aquí es donde más sentido tiene el reto de "SFT memoriza, RL
generaliza": entrenad con unas familias de reacciones y evaluad con otras. En la fase 4
valoraría un agente que da pistas antes que la solución, y que consulta la ficha de seguridad
cuando el estudiante pregunta por un reactivo peligroso.

---

## Cómo se convierte un tema en requisitos concretos

Cuando apruebe vuestra propuesta os devolveré una página con esto mismo para vuestro tema.
Para que sepáis qué esperar, así funciona:

**Los mínimos de tamaño se ajustan a lo que el tema permite.** Si vuestros problemas se
generan con un script, os pediré miles, no cientos. Si se anotan a mano, unos cientos bien
hechos. Si el corpus son cinco leyes, pediré que el conjunto dorado sea más grande y más
fino, porque el corpus es pequeño y estructurado. Si son mil fichas técnicas en PDF, pediré
menos preguntas pero un análisis serio del chunking.

**La tercera recompensa la define el tema.** Unidades en farmacia y química. Fuente citada
en legal. Desglose en finanzas. Tests que pasan por la razón correcta en desarrollo. Lenguaje
llano en social. Cifras trazables en marketing. Os la propondré, y podéis discutirla.

**Los casos difíciles del conjunto de test los fijamos juntos.** Cada tema tiene sus
trampas clásicas y quiero que estén en el test: el festivo local, la prorrata de IVA, el
cambio semántico silencioso, la campaña con veinte muestras. Traedme vuestra lista y yo
añadiré alguna.

**Las herramientas con efectos apuntan a entornos de prueba.** Nada que escriba en un sistema
real: repositorios de pruebas, tiendas en modo sandbox, webhooks propios, ficheros locales.

**La seguridad es parte de la nota cuando el tema lo exige.** Salud, legal, finanzas y social
llevan aviso obligatorio y remisión a un profesional en cada respuesta. Desarrollo lleva
sandbox con límites. Cualquier tema con datos de personas usa datos sintéticos, sin excepción.

**La fase 4 se evalúa con tareas del tema, no genéricas.** Prepararé cinco tareas que no
habréis visto, del estilo de las de vuestro banco pero distintas, y una imposible. La
imposible no es una trampa: quiero ver qué hace vuestro agente cuando no puede.

Si después de leer esto tenéis un tema en la cabeza, id a la plantilla de la propuesta
([`00_propuesta.md`](00_propuesta.md)) y empezad por la sección de la tarea verificable. Si
esa sección se escribe sola, tenéis tema.
