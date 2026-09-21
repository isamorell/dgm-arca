#!/usr/bin/env bash
# Entorno para trabajar en la DGX de Comillas. Ejecutad al principio de CADA sesión:
#
#     source smoke/dgx_env.sh
#
# Qué hace: sitúa las cachés de uv y de Hugging Face dentro de vuestra carpeta de trabajo
# (/home/<usuario>/clusters/dgx), que es la única que se conserva entre sesiones, e instala
# uv ahí mismo si no está. Sin esto, uv y los modelos descargados pueden desaparecer con la
# sesión y volveréis a descargar varios gigas cada vez.

WORK_DIR="${ARCA_WORK_DIR:-$HOME/clusters/dgx}"

if [ ! -d "$WORK_DIR" ]; then
    echo "No encuentro $WORK_DIR. ¿Estás en la DGX? Si tu carpeta de trabajo es otra,"
    echo "exporta ARCA_WORK_DIR antes de hacer source de este script."
    return 1 2>/dev/null || exit 1
fi

export UV_INSTALL_DIR="$WORK_DIR/.uv/bin"
export UV_CACHE_DIR="$WORK_DIR/.uv/cache"
export UV_PYTHON_INSTALL_DIR="$WORK_DIR/.uv/python"
export HF_HOME="$WORK_DIR/.hf_cache"
# La configuración global de git (identidad, credenciales) también dentro del espacio persistente.
export GIT_CONFIG_GLOBAL="$WORK_DIR/.gitconfig"
touch "$GIT_CONFIG_GLOBAL"
export TOKENIZERS_PARALLELISM=false
mkdir -p "$UV_INSTALL_DIR" "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR" "$HF_HOME"
export PATH="$UV_INSTALL_DIR:$PATH"

if ! command -v uv >/dev/null 2>&1; then
    echo "Instalando uv en $UV_INSTALL_DIR ..."
    curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$UV_INSTALL_DIR" UV_NO_MODIFY_PATH=1 sh
fi

if [ -f "$WORK_DIR/dgm-arca/.env" ]; then
    set -a; . "$WORK_DIR/dgm-arca/.env"; set +a
fi

echo "uv:        $(uv --version 2>/dev/null || echo 'no disponible')"
echo "HF_HOME:   $HF_HOME"
echo "git:       $(git config --global user.name 2>/dev/null || echo 'sin identidad; ver docs/github.md')"
echo "GPU:       $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'ninguna visible')"
