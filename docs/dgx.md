# Trabajar en la DGX

La DGX del departamento es una máquina compartida con GPUs NVIDIA. Este curso el acceso ha
cambiado respecto al anterior, así que leed esto entero antes de la primera sesión, aunque ya
la hayáis usado. Lo esencial: **no hay SSH ni VPN**, todo va por el navegador; **cada uno
recibe una GPU de 16 GB** durante un máximo de 24 horas; y **no hay copias de seguridad**.

## Cómo se accede

Todo el acceso es vía web, a través del portal **Open OnDemand** de Comillas:
[https://hpc.comillas.edu](https://hpc.comillas.edu). La guía oficial del portal, con
capturas, está en [hpc.comillas.edu/public/docs/guia-hpc-comillas.html](https://hpc.comillas.edu/public/docs/guia-hpc-comillas.html)
Resumido:

1. Entráis en el portal con vuestra cuenta de Comillas (Microsoft, la misma del correo).
2. En **Interactive Apps** elegís **code-server → DGX** (o **Jupyter → DGX**) y rellenáis el
   formulario: instancias de GPU (1 = 16 GiB), duración de la sesión y QoS (viene fijado según
   vuestro perfil, se deja como está). Pulsáis **Launch**.
3. SLURM pone la sesión en cola (**Queued**, tarjeta azul) y en unos segundos pasa a
   **Running** (tarjeta verde) con el botón para abrirla. No recarguéis ni volváis a pulsar Launch.
4. Trabajáis desde **code-server** o desde **JupyterLab**, en una pestaña nueva.
5. Al terminar, volvéis a **My Interactive Sessions** y pulsáis **Delete**: cerrar la pestaña
   no libera la GPU, borrar la sesión sí.

code-server es Visual Studio Code en el navegador (en realidad Code-OSS, con extensiones de
Open VSX), con terminal integrada. Es lo que os recomiendo para esta práctica: vais a trabajar
con un repositorio, no con un notebook. Jupyter sirve igual si preferís celdas; el flujo de
abajo está pensado para la terminal de code-server.

**Vuestra carpeta de trabajo es `/home/<usuario>/clusters/dgx`.** Es el único sitio que se
conserva de una sesión a la siguiente. Cualquier otra cosa que cuelgue de vuestro `/home`
fuera de esa carpeta no está garantizado que sobreviva. Clonad el repositorio ahí, y usad el
script `smoke/dgx_env.sh` (abajo) para que también las cachés de `uv` y de Hugging Face vivan
ahí; si no, volveréis a descargar varios gigas en cada sesión.

Las imágenes base traen las librerías del curso pasado. Lo que necesitemos de más lo
instalamos en nuestro espacio con `uv`, sin pedir permiso a nadie. Si algo os parece de uso
general, se puede pedir al equipo del clúster que lo incorpore a la imagen.

## La GPU: una partición de 16 GB

Las GPU físicas de la DGX (H200) están particionadas con MIG en siete instancias de 16 GiB
cada una. De forma estándar cada sesión recibe **una instancia de 16 GB** que, desde vuestro
código, aparece como una GPU normal (`cuda:0`). SLURM se
encarga de asignarla: no toquéis `CUDA_VISIBLE_DEVICES`, ya viene puesta.

Con 16 GB se hace toda la práctica si elegís bien los tamaños: los modelos de 0.6B a 1.7B
parámetros con LoRA y bf16 caben de sobra para SFT y GRPO, y Qwen3-Embedding-0.6B para el
RAG también. Los scripts del repositorio tienen valores por defecto pensados para ese
presupuesto. Si os quedáis sin memoria, antes de pedir más GPU bajad `--num-generations`,
`--max-completion-length` o el tamaño del modelo; en `EXPERIMENTS.md` anotad qué configuración
cabe y cuál no.

Si de verdad necesitáis más de 16 GB, varias GPU o un proceso de más de 24 horas, se puede
pedir al equipo del clúster, pero hay que planificarlo con antelación: algunos cambios de
particionado requieren reiniciar la máquina. Hablad conmigo primero.

## Las sesiones duran 24 horas como máximo

Cualquier entrenamiento tiene que poder morir y reanudarse. Los scripts de la fase 1 guardan
checkpoints (`--save-steps`) y admiten `--resume-from-checkpoint` para continuar desde el
último. Usadlo. Y no lancéis nada largo un viernes a última hora sin haber comprobado antes
que el checkpoint se escribe donde creéis.

## Almacenamiento: la DGX no es un disco

Es un entorno de cálculo, no un sitio donde guardar cosas. El flujo es: subir datos, calcular,
descargar resultados. Y dos avisos que hay que tomarse en serio:

- **No hay copias de seguridad** de las carpetas de usuario.
- El almacenamiento está en **RAID 0**: rápido, pero sin tolerancia a fallos. Si un disco
  falla, se pierde todo.

Por tanto: no dejéis en la DGX nada que no tengáis también en otro sitio. El código va a
vuestro repositorio de GitHub (haced `git push` a menudo; cómo configurarlo desde la DGX está
en [`docs/github.md`](github.md)). Los adaptadores LoRA pesan poco:
subidlos a Hugging Face Hub (`huggingface-cli upload`) o descargadlos al terminar cada
entrenamiento. Los datasets y el corpus, en el repositorio o con un script que los regenere.
La caché de modelos de Hugging Face (con `dgx_env.sh`, en `clusters/dgx/.hf_cache`) se puede
borrar y regenerar sin problema, pero ocupa: limpiadla cuando terminéis una fase. Más adelante habrá límites de
espacio por usuario; hasta entonces, sed responsables.

## La prueba de la primera sesión, paso a paso

Desde una sesión de code-server con GPU, en la terminal (que ya se abre dentro de
`clusters/dgx`):

1. **Clonad vuestro repositorio** (o `git pull` si ya lo tenéis). Para la prueba de la
   primera sesión vale el de la plantilla; para trabajar, el vuestro, creado con
   *Use this template* (ver [`docs/github.md`](github.md), que también explica cómo hacer
   `push` desde la DGX):

   ```bash
   cd ~/clusters/dgx
   git clone https://github.com/kendrickcetina/dgm-arca.git
   cd dgm-arca
   ```

2. **Preparad el entorno de la sesión.** Esto hay que hacerlo al principio de **cada**
   sesión: coloca las cachés dentro de vuestra carpeta de trabajo e instala `uv` ahí si falta.

   ```bash
   source smoke/dgx_env.sh
   ```

3. **Instalad el entorno.** `uv` crea `.venv` dentro del proyecto con Python 3.11 y todas
   las librerías, incluido torch con CUDA 12.8 para Linux. La primera vez descarga varios
   gigas; después es instantáneo:

   ```bash
   uv sync --extra train
   ```

   (Añadid `--extra rag --extra agent` cuando lleguéis a esas fases, o `make setup` para todo.)

4. **Comprobad que torch ve la GPU:**

   ```bash
   uv run arca-check-gpu
   ```

   Tenéis que ver una GPU de unos 16 GB en la tabla, `bf16 supported: True` y un número de
   TFLOP/s razonable. Si dice que no ve CUDA, la sesión no tiene GPU asignada: mirad los
   recursos que pedisteis al lanzarla en el portal.

5. **Lanzad el smoke test:**

   ```bash
   uv run arca-smoke
   ```

   En otra terminal, `watch -n 2 nvidia-smi` para ver la memoria y la utilización subir. En
   10-15 minutos veréis una línea por paso con las recompensas y, al final, la comparación
   antes/después. Lo que tiene que pasar está explicado en [`smoke/README.md`](../smoke/README.md).
   Si la memoria se queda justa, `uv run arca-smoke --num-generations 4`.

6. **Comprobad que la API arranca:**

   ```bash
   uv run arca-api &
   curl -s localhost:8000/health | python3 -m json.tool
   kill %1
   ```

   Todas las fases saldrán como `pending`. Es lo esperado: todavía no habéis hecho nada.

7. **Pasad los tests:**

   ```bash
   uv run pytest
   ```

Si los siete pasos funcionan, el entorno está listo y podéis empezar la fase 1.

## Entrenamientos largos dentro de una sesión

Si cerráis la pestaña del navegador, la sesión de SLURM sigue viva hasta su límite, pero la
terminal de Code Server puede perder el proceso. Lanzad los entrenamientos con `nohup` o
dentro de `tmux` si está disponible, y redirigid la salida a un fichero:

```bash
nohup uv run python -m rlm.train_grpo --data rlm/data/train.jsonl --steps 500 \
    --save-steps 50 > runs/grpo_run01.log 2>&1 &
tail -f runs/grpo_run01.log
```

Cuando la sesión termine, abrid otra y continuad con
`--resume-from-checkpoint rlm/weights/final_rlm_lora/checkpoint-XXX`.

## Y Docker, ¿qué?

Dentro de las sesiones de la DGX no tenéis Docker: el entorno lo gestiona SLURM y no hay
permisos para ejecutar contenedores. Ahí trabajáis con `uv` directamente, como en los pasos
de arriba. El `Dockerfile` y el `docker-compose.yml` del repositorio sirven para lo otro que
os pido: que la API se pueda desplegar con una orden en cualquier máquina con Docker, para la
corrección y para vuestro portfolio. Lo normal será entrenar en la DGX, descargar el
adaptador, y servir la API desde vuestro portátil o una máquina en la nube con
`docker compose up api` y ngrok. Si el equipo del clúster habilita contenedores (Apptainer o
similar) más adelante, os avisaré.

## Cuando algo falla

**`check_gpu.py` no ve CUDA.** La sesión no tiene GPU asignada, o pedisteis recursos sin
GPU al lanzarla en el portal. Comprobad con `nvidia-smi` en la terminal: si no lista ninguna
GPU, cerrad la sesión y lanzad otra con GPU.

**`CUDA out of memory`.** Tenéis 16 GB. Bajad `--num-generations`, `--max-completion-length`
o el tamaño del modelo. El gradient checkpointing ya está activado en los scripts.

**La sesión murió a las 24 horas a mitad de entrenamiento.** Para eso están los checkpoints.
Reanudad con `--resume-from-checkpoint`. Si no guardasteis ninguno, la lección está aprendida.

**Descargas lentas o límite de peticiones a Hugging Face.** Poned vuestro `HF_TOKEN` en
`.env` (es gratis).

**Se ha llenado el disco.** Borrad de `clusters/dgx/.hf_cache/hub` los modelos que ya no uséis
y los checkpoints intermedios que ya tengáis replicados. Recordad que nada de la DGX tiene copia.

**Código del curso pasado que no funciona.** Han cambiado el sistema, el firmware, los
drivers y el entorno de software. Si vuestro código estaba preparado para varias GPU, ahora
tiene que trabajar con la que SLURM le asigne.

**`uv` o los modelos han desaparecido al abrir una sesión nueva.** Estaban fuera de
`clusters/dgx`. Haced `source smoke/dgx_env.sh` al principio de cada sesión y volverán a
instalarse en el sitio correcto.

**Cualquier otra cosa.** Copiad el error completo, el comando exacto y la salida de
`check_gpu.py`, y traedlo a clase o al canal de la asignatura. Para incidencias del propio
clúster (acceso, portal, cuotas) el contacto es gestion_cluster@comillas.edu.
