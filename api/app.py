"""ARCA evaluation API.

One endpoint per phase, all on the same domain. Start it with::

    uv run arca-api                # or: make api
    docker compose up api

and open http://localhost:8000/docs for the interactive documentation.

The grader calls these endpoints with the bodies defined in ``api/schemas.py``. A phase
that is not implemented yet answers 501 with a clear message, so the API is always up
and you can hand it in incrementally.
"""

from __future__ import annotations

import logging
import time

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse

from api.backends import Backends, env
from api.schemas import (
    AgentRequest,
    AgentResponse,
    HealthResponse,
    RagRequest,
    RagResponse,
    ReasoningRequest,
    ReasoningResponse,
    ToolsRequest,
    ToolsResponse,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("arca.api")

app = FastAPI(
    title="ARCA · Agente con Razonamiento, Conocimiento y Acción",
    description=(
        "API de evaluación de la práctica final de Modelos Generativos Profundos (MIA, ICAI). "
        "Un endpoint por fase: /reasoning, /tools, /rag y /agent."
    ),
    version="0.1.0",
)
backends = Backends()


def _run(phase: str, loader, call):
    """Load the phase backend and run the call; a missing phase becomes HTTP 501."""
    try:
        backend = loader()
        return call(backend)
    except NotImplementedError as exc:
        raise _not_ready(phase, exc) from exc
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001 - report loading/runtime failures with context
        log.exception("phase %s failed", phase)
        raise HTTPException(status_code=500, detail={"phase": phase, "message": str(exc)}) from exc


def _not_ready(phase: str, exc: Exception) -> HTTPException:
    return HTTPException(
        status_code=501,
        detail={
            "phase": phase,
            "message": str(exc) or f"Phase '{phase}' is not implemented yet.",
            "hint": "See the README of the corresponding folder for what is expected.",
        },
    )


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health():
    """Which phases are ready. The grader calls this first."""
    return HealthResponse(
        project="dgm-arca",
        team=env("ARCA_TEAM"),
        domain=env("ARCA_DOMAIN"),
        phases=backends.status(),
    )


@app.post("/reasoning", response_model=ReasoningResponse, tags=["fase 1"])
def reasoning(request: ReasoningRequest):
    """Phase 1: the model thinks, then answers. Verifier verdict when ground truth is given."""
    started = time.perf_counter()
    response = _run(
        "reasoning",
        backends.reasoning,
        lambda m: m.answer(request.question, request.expected_answer, request.max_new_tokens),
    )
    log.info("/reasoning %.1fs", time.perf_counter() - started)
    return response


@app.post("/tools", response_model=ToolsResponse, tags=["fase 2"])
def tools(request: ToolsRequest):
    """Phase 2: the model decides which tools to call; the application executes them."""
    return _run("tools", backends.tools, lambda loop: loop.run(request.query, request.max_turns))


@app.post("/rag", response_model=RagResponse, tags=["fase 3"])
def rag(request: RagRequest):
    """Phase 3: retrieve from your corpus, answer with citations."""
    return _run(
        "rag",
        backends.rag,
        lambda p: p.answer(request.question, top_k=request.top_k, retriever=request.retriever),
    )


@app.post("/agent", response_model=AgentResponse, tags=["fase 4"])
def agent(request: AgentRequest):
    """Phase 4: the full ReAct loop with every tool, including the knowledge base."""
    return _run(
        "agent",
        backends.agent,
        lambda a: a.run(request.task, max_steps=request.max_steps, brain=request.brain),
    )


def main() -> None:
    """Entry point for ``uv run arca-api``."""
    uvicorn.run("api.app:app", host="0.0.0.0", port=int(env("ARCA_API_PORT", "8000")))


if __name__ == "__main__":
    main()
