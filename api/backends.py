"""Glue between the API and the four phases.

Each phase exposes one entry point that ``api/app.py`` calls. While a phase is not
implemented, its entry point raises ``NotImplementedError`` and the API answers with
HTTP 501 and a message telling the grader which phase is missing. ``/health`` reports
the same information.

You do not need to touch this file unless you change the entry points' names. The
places to implement are:

* phase 1: ``rlm.inference.ReasoningModel``
* phase 2: ``tool_use.executor.ToolLoop``
* phase 3: ``rag.generate.RagPipeline``
* phase 4: ``agent.react_agent.ReActAgent``
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from typing import Any

from api.schemas import PhaseStatus

log = logging.getLogger("arca.api")

PHASES = ("reasoning", "tools", "rag", "agent")


class Backends:
    """Lazy holder for the phase implementations, so the API starts even if a phase is missing."""

    def __init__(self) -> None:
        self._cache: dict[str, Any] = {}
        self._errors: dict[str, str] = {}

    # -- loaders --------------------------------------------------------------

    def _load(self, phase: str, factory: Callable[[], Any]) -> Any:
        if phase in self._cache:
            return self._cache[phase]
        try:
            self._cache[phase] = factory()
            self._errors.pop(phase, None)
        except NotImplementedError as exc:
            self._errors[phase] = str(exc) or f"Phase '{phase}' is not implemented yet."
            raise
        except Exception as exc:  # noqa: BLE001 - surface any loading problem to /health
            self._errors[phase] = f"{type(exc).__name__}: {exc}"
            log.exception("Failed to load phase %s", phase)
            raise
        return self._cache[phase]

    def reasoning(self):
        def factory():
            from rlm.inference import ReasoningModel

            model = ReasoningModel.from_env()
            model.load()
            return model

        return self._load("reasoning", factory)

    def tools(self):
        def factory():
            from tool_use.executor import build_tool_loop

            return build_tool_loop()

        return self._load("tools", factory)

    def rag(self):
        def factory():
            from rag.generate import build_pipeline

            return build_pipeline()

        return self._load("rag", factory)

    def agent(self):
        def factory():
            from agent.react_agent import ReActAgent

            return ReActAgent.from_env()

        return self._load("agent", factory)

    # -- health ---------------------------------------------------------------

    def status(self) -> dict[str, PhaseStatus]:
        """Try to load every phase (cheaply) and report what is ready."""
        report: dict[str, PhaseStatus] = {}
        for phase in PHASES:
            if phase in self._cache:
                report[phase] = PhaseStatus(status="ready")
                continue
            try:
                getattr(self, phase)()
                report[phase] = PhaseStatus(status="ready")
            except NotImplementedError as exc:
                report[phase] = PhaseStatus(status="pending", detail=str(exc))
            except Exception as exc:  # noqa: BLE001
                report[phase] = PhaseStatus(status="error", detail=f"{type(exc).__name__}: {exc}")
        return report


def env(name: str, default: str = "") -> str:
    """Small helper so every module reads configuration the same way."""
    return os.environ.get(name, default)
