# ARCA · atajos. Escribe `make` sin argumentos para ver la lista.

.DEFAULT_GOAL := help
UV ?= uv

help: ## Muestra esta ayuda
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

setup: ## Instala todo en .venv con uv (fase 1 + RAG + agente + dev)
	$(UV) sync --extra train --extra rag --extra agent

setup-min: ## Instala solo lo necesario para la API y los tests
	$(UV) sync

lock: ## Regenera uv.lock tras cambiar pyproject.toml
	$(UV) lock

test: ## Ejecuta los tests
	$(UV) run pytest

lint: ## Comprueba estilo con ruff
	$(UV) run ruff check .
	$(UV) run ruff format --check .

format: ## Formatea el código con ruff
	$(UV) run ruff format .
	$(UV) run ruff check --fix .

check-gpu: ## Diagnóstico de GPU y librerías
	$(UV) run arca-check-gpu

smoke: ## Entrenamiento GRPO de prueba en GPU (10-15 min)
	$(UV) run arca-smoke

smoke-dry: ## Prueba de instalación en CPU con un modelo diminuto (~1 min)
	$(UV) run arca-smoke --dry-run

api: ## Levanta la API en local con recarga automática
	$(UV) run uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload

enunciado: ## Recompila docs/enunciado.pdf a partir del .tex
	cd docs && pdflatex -interaction=nonstopmode enunciado.tex >/dev/null
	cd docs && pdflatex -interaction=nonstopmode enunciado.tex >/dev/null
	cd docs && rm -f enunciado.aux enunciado.log enunciado.out
	@echo "docs/enunciado.pdf actualizado"

build: ## Construye la imagen Docker
	docker compose build

docker-check-gpu: ## Diagnóstico de GPU dentro del contenedor
	docker compose run --rm check-gpu

docker-smoke: ## Smoke test dentro del contenedor
	docker compose run --rm smoke

docker-api: ## API dentro del contenedor
	docker compose up api

shell: ## Shell interactiva con GPU dentro del contenedor
	docker compose run --rm train

.PHONY: help setup setup-min lock test lint format check-gpu smoke smoke-dry api enunciado build docker-check-gpu docker-smoke docker-api shell
