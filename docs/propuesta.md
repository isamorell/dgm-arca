# Fase 0 — Propuesta de tema

## Equipo

Raquel Fernández Esquinas - 202209790@alu.comillas.edu

Teresa X. Garvía Gallego - tgarviagallego@alu.comillas.edu

Isabel V. Morell Maudes - 202207975@alu.comillas.edu


## El tema en una frase

Un asistente educativo que resuelve y explica paso a paso cálculos de administración de medicación, como velocidad de infusión, goteo, dosis por peso y diluciones, ayudando a estudiantes de Enfermería a practicar y comprobar sus cálculos.

## El usuario y su problema

La persona que utilizaría este agente sería un estudiante de Enfermería que necesita practicar cálculos de administración de medicación, resolver dudas sobre el procedimiento o comprobar si sus propios cálculos son correctos. El agente también le permitiría generar nuevos ejercicios similares a los que ya haya realizado y obtener una explicación paso a paso de su resolución.

Sin este agente, el estudiante debe limitarse a los ejercicios proporcionados por el profesor, que pueden no incluir soluciones, o dedicar tiempo a buscar y preparar ejercicios adicionales. Además, cuando comete un error en el planteamiento o en las conversiones de unidades, puede no detectarlo hasta que recibe ayuda del profesor o compara su resultado con una solución.

El agente busca, por tanto, reducir el tiempo necesario para encontrar ejercicios y comprobar soluciones, y ayudar a detectar errores en los cálculos y en su planteamiento, sin sustituir la supervisión del profesor.


## Diez preguntas o tareas reales

1. Tengo que pasar 500 mL de suero en 4 horas. ¿A qué velocidad tengo que poner la bomba en mL/h? 
2. Tengo que pasar 750 mL en 5 horas con un equipo de 20 gotas/mL. ¿A cuántas gotas por minuto tengo que regular el goteo 
3. Un paciente pesa 60 kg y le han pautado una dosis de 0,05 g/kg/día, repartida en 4 administraciones. ¿Cuántos mg debe recibir en cada administración? 
4. Un paciente pesa 75 kg y necesita una dosis de 2,5 mg/kg. El medicamento está disponible en una concentración de 500 mg/10 mL. ¿Qué cantidad total de medicamento necesita el paciente y qué volumen hay que administrar? 
5. Tengo que administrar 750 mg de paracetamol. Busca una presentación adecuada en CIMA y calcula qué volumen tengo que administrar si la concentración disponible está expresada en mg/mL 
6. Tengo que administrar 1 g de ceftriaxona a un paciente. Busca una presentación adecuada y calcula qué volumen tengo que preparar para administrar la dosis prescrita. Expresa el resultado final en mL y comprueba todas las conversiones de unidades. 
7. Tengo que administrar 0,5 g de vancomicina y preparar una solución con una concentración final de 5 mg/mL. ¿Qué volumen final necesito? ¿Qué diluyente necesito y cuánta cantidad tengo que añadir? Explícame las conversiones de unidades.» 
8. A un paciente de 65 kg le han pautado 5 mcg/kg/min de noradrenalina. La solución que tengo preparada contiene 4 mg en 50 mL. ¿A qué velocidad en mL/h tengo que programar la bomba? Haz todas las conversiones necesarias. 
9. Tengo que administrar amoxicilina a un paciente de 25 kg. La dosis prescrita es de 40 mg/kg/día repartida en 3 administraciones. Busca una presentación adecuada y calcula cuántos gramos de principio activo necesita al día, cuántos mg corresponden a cada administración y qué volumen debo administrar en cada una. 
20. Un paciente de 70 kg necesita recibir un medicamento a una dosis de 8 mcg/kg/min. Tengo disponible una presentación del medicamento y necesito preparar una perfusión intravenosa. Averigua la información necesaria sobre el medicamento, calcula la cantidad que necesita el paciente, determina cómo preparar la perfusión y a qué velocidad debo administrarla. Explica todos los pasos, realiza las conversiones de unidades necesarias, comprueba el resultado y prepara una hoja de preparación. 

## La tarea verificable (fase 1)

- El tipo de problema y dos ejemplos con su respuesta.

En nuestro dominio, los problemas de cálculo de medicación y cambios de unidades tienen una respuesta numérica y una unidad que se pueden comprobar automáticamente. El verificador puede realizar el mismo cálculo mediante una implementación de referencia y comparar el resultado obtenido por el modelo con el resultado correcto comprobando también que las unidades sean correctas. 

Por ejemplo:

**Problema:** Tengo que pasar 500 mL de suero en 4 horas. ¿A qué velocidad tengo que poner la bomba en mL/h? 

**Respuesta:** 500 mL / 4h = 125 mL/h

El verificador comprueba que el resultado sea 125 mL/h y que la unidad final sea correcta. Para ello, toma los valores y unidades del problema y realiza el cálculo mediante una implementación de referencia. 

**Problema:** Un paciente pesa 60 kg y le han pautado una dosis de 0,05 g/kg/día, repartida en 4 administraciones. ¿Cuántos mg debe recibir en cada administración? 

**Respuesta:**

60 kg * 0,05 g/(kg * dia) = 3 g/dia
3 g/dia * 1000 mg/g = 3000 mg/dia
3000 mg/dia / 4 administraciones = 750 mg/administracion

El verificador comprobaría tanto el valor numérico como la unidad final, permitiendo detectar errores como confundir gramos con miligramos. 


- Cómo lo verificaríais con código: comparar números, ejecutar tests, comparar conjuntos...

Para verificarlo programaremos un código para que parsee los parámetros del problema y calcule la respuesta correcta independientemente del modelo. La respuesta generada por el modelo se normalizará a unidades estándar y se comparará con la respuesta de referencia. Se comprobarán: el valor numérico, las unidades de la respuesta y las conversiones entre unidades.

- **Con cuál de las seis estrategias de [`datasets.md`](datasets.md) vais a construir el
  conjunto**, y cuántos problemas esperáis conseguir. Si es un generador, qué parámetros
  muestreáis; si es minería de datos, la fuente y una fila de ejemplo; si es un benchmark
  público, cuál y qué añadís de vuestra cosecha. Esta es la parte donde más os voy a apretar.

Los cientos de problemas de entrenamiento se generarán programáticamente, en lugar de anotarlos manualmente. Para cada problema se generarán aleatoriamente los parámetros relevantes, como peso del paciente, dosis, concentración, volumen, tiempo y unidades. 


Para ello haremos una función que muestree parámetros, calcule la respuesta correcta, genere enunciados con distintas unidades y estructuras gramaticales y guarde el problema con su respuesta.

El mismo cálculo utilizado para generar la respuesta correcta servirá como verificador de referencia. Esto permitirá generar cientos de problemas diferentes de forma automática y controlar su dificultad.

Los problemas se dividirán en distintos niveles, desde cálculos de un solo paso hasta problemas de varios pasos que requieran conversiones entre unidades. Se reservará además un conjunto de problemas de test que no se utilizará durante el entrenamiento.

**Posibles casos límites:** diferenciar entre coma o punto decimal (en España usamos la coma), redondeo, mg contra mcg.


## Las herramientas (fase 2)

- **Consulta fuera del modelo:** qué API o fuente externa real (con enlace a su documentación).

Vamos a utilizar dos APIs:

1. CIMA Rest API (https://cima.aemps.es/cima/resources/docs/CIMA_REST_API.pdf) para consultar información de medicamentos, como principios activos, presentaciones, concentraciones, formas farmacéuticas y vías de administración. La API permite consultar medicamentos y presentaciones mediante peticiones REST y devuelve la información en JSON. 

2. PubChem PUG REST (https://pubchem.ncbi.nlm.nih.gov/docs/pug-rest) para obtener información química de los principios activos, como la fórmula molecular y la masa molecular. La API permite consultar estas propiedades directamente a partir del nombre o identificador del compuesto. 

Estas APIs permitirán que el agente consulte información externa y actualizada en lugar de depender únicamente de los conocimientos del modelo. Por ejemplo, ante un ejercicio que incluya un medicamento concreto, el agente podrá consultar en CIMA una presentación y concentración disponibles antes de realizar los cálculos. 


- **Cálculo o ejecución:** qué se calcula o ejecuta y por qué el modelo no debería hacerlo de cabeza.

Utilizaremos Python con la librería Pint para realizar los cálculos numéricos y las conversiones de unidades. 

El modelo no debería realizar estos cálculos directamente de cabeza, ya que queremos que los resultados numéricos sean precisos, reproducibles y verificables. Además, el uso de unidades permite detectar errores de planteamiento, como confundir gramos con miligramos o minutos con horas. El agente delegará las operaciones en esta herramienta y utilizará el resultado para generar la explicación al estudiante. 

- **Acción con efecto observable:** qué hace y cómo se comprueba que lo ha hecho.

El agente podrá generar una hoja de preparación en PDF a partir de los resultados obtenidos. Esta hoja incluirá, por ejemplo, el medicamento utilizado, la concentración, la dosis calculada, el volumen o velocidad de administración, las unidades y los pasos principales del cálculo.

El efecto será observable porque se generará un fichero PDF que el usuario podrá abrir y revisar. Además, podremos comprobar automáticamente que el fichero se ha creado correctamente y que contiene los campos y resultados esperados.


## El corpus (fase 3)

- De dónde salen los documentos y cuántos hay (aproximadamente, en documentos y en páginas).

El corpus sale de las fichas técnicas de CIMA (AEMPS), descargadas por script contra la API (docSegmentado), no copiadas a mano, así que es reproducible. Nos quedamos con una selección de unos 45-50 medicamentos de administración parenteral habituales en cálculos de infusión y dilución (antibióticos IV, analgésicos IV, vasopresores, electrolitos, anticoagulantes, sedantes), y de cada uno solo las secciones relevantes para el dominio: 4.2 (posología y forma de administración), 6.2 (incompatibilidades), 6.3 (periodo de validez), 6.4 (precauciones especiales de conservación) y 6.6 (precauciones especiales de eliminación y otras manipulaciones). Son entre 4 y 5 secciones por medicamento; si se contasen como fichas completas equivaldría a unas 150-200 páginas, aunque al quedarnos solo con esas secciones el texto real es bastante más corto.

- En qué formato están.

La API devuelve cada sección como HTML dentro de un JSON (lo hemos comprobado: párrafos y spans con el texto). Los limpiamos a texto plano antes de indexarlos, y guardamos también el JSON crudo de cada sección como respaldo, por si hace falta volver a él.

- Con qué licencia, o por qué podéis usarlos.

Son documentos públicos de un organismo público (la AEMPS), de acceso libre y sin registro. No hemos encontrado que CIMA publique una licencia explícita tipo Creative Commons; en principio les aplicaría la Ley 37/2007 de reutilización de la información del sector público (modificada por la Ley 18/2015), pero esto hay que confirmarlo mirando el aviso legal de la AEMPS antes de darlo por definitivo en el informe, en vez de asumirlo sin más.

- Dos ejemplos de pregunta que solo se pueden responder leyendo el corpus.

1. ¿En cuántos mL hay que reconstituir un vial de amoxicilina/ácido clavulánico Sandoz 1000 mg/200 mg antes de prepararlo para perfusión, y cuánto tiempo puede pasar como máximo entre la reconstitución y la administración?
2. ¿Es compatible [medicamento X] con suero glucosado al 5 %, o solo puede diluirse en suero fisiológico?

## Qué puede salir mal

Una de las limitaciones es que los datos de dilución en CIMA están en texto libre, no en un campo numérico, lo hemos comprobado con un ejemplo real. Por lo que construir el generador de problemas y el conjunto dorado a partir de medicamentos reales va a suponer que alguien del grupo se lea los datos y saque los números a mano.

Por otro lado, PubChem puede fallar con la librería HTTP equivocada, ya que nos ha pasado con algunas pruebas que hemos realizado. Nos salió error 503 de forma sistemática con HTTP/1.1 pero con HTTP/2 funcionaba bien siempre. Cuando lo probemos en el DGX veremos si el comportamiento cambia o no. Además, es una API que solo permite 5 peticiones al segundo como máximo, sino te puede llegar a bloquear, por lo que ya veremos como solucionamos este hecho (si evitando esta API o usándola en menor medida).

## Por qué este tema

Nos interesa bastante el tema de medicina y farmacología, y este tema nos permite trabajar con datos reales (CIMA, PubChem) y con matemáticas concretas (conversión de unidades, concentración, velocidad de infusión). Además, es un problema que reconocemos de verdad, comprobamos si un cálculo de este tipo está bien hecho, sin depender de que un profesor lo revise, que creemos que puede ser bastante útil para cualquier estudiante de enfermería o similar.
