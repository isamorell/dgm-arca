"""Tool creation and tool definition, the two halves of "setting up the tools".

In class we split tool setup in two steps: write the function, then describe it to the
model (name, description, arguments). This module does both from a single Python
function with type hints and a docstring::

    @tool
    def get_weather(location: str) -> float:
        \"\"\"Current temperature in °C for a location such as 'Madrid, Spain'.\"\"\"
        ...

``get_weather.definition`` is the JSON Schema the model sees; ``registry.call(...)``
validates the arguments the model produced (with Pydantic) and executes the function,
timing it and capturing errors so that the loop can hand them back to the model as an
observation instead of crashing.
"""

from __future__ import annotations

import inspect
import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ValidationError, create_model

from api.schemas import ToolCallRecord


@dataclass
class Tool:
    name: str
    description: str
    fn: Callable[..., Any]
    args_model: type[BaseModel]
    requires_confirmation: bool = False

    @property
    def definition(self) -> dict[str, Any]:
        """Tool definition in the JSON format used by Qwen and most chat templates."""
        schema = self.args_model.model_json_schema()
        schema.pop("title", None)
        return {
            "type": "function",
            "function": {"name": self.name, "description": self.description, "parameters": schema},
        }

    def validate(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Raise ``pydantic.ValidationError`` when the model's arguments do not fit the schema."""
        return self.args_model(**arguments).model_dump()

    def __call__(self, **arguments: Any) -> Any:
        return self.fn(**self.validate(arguments))


def _args_model_from_signature(fn: Callable[..., Any]) -> type[BaseModel]:
    """Build a Pydantic model whose fields mirror the function parameters."""
    fields: dict[str, Any] = {}
    for name, param in inspect.signature(fn).parameters.items():
        annotation = param.annotation if param.annotation is not inspect.Parameter.empty else Any
        default = ... if param.default is inspect.Parameter.empty else param.default
        fields[name] = (annotation, default)
    return create_model(f"{fn.__name__}_args", **fields)


def tool(fn: Callable[..., Any] | None = None, *, requires_confirmation: bool = False):
    """Decorator that turns a typed, documented function into a ``Tool``."""

    def wrap(func: Callable[..., Any]) -> Tool:
        if not func.__doc__:
            raise ValueError(f"tool {func.__name__} needs a docstring: it is what the model reads")
        return Tool(
            name=func.__name__,
            description=inspect.cleandoc(func.__doc__),
            fn=func,
            args_model=_args_model_from_signature(func),
            requires_confirmation=requires_confirmation,
        )

    return wrap(fn) if fn is not None else wrap


@dataclass
class ToolRegistry:
    """The set of tools the application hosts and the model may call."""

    tools: dict[str, Tool] = field(default_factory=dict)

    def register(self, *tools: Tool) -> ToolRegistry:
        for t in tools:
            self.tools[t.name] = t
        return self

    def definitions(self) -> list[dict[str, Any]]:
        return [t.definition for t in self.tools.values()]

    def render_for_prompt(self) -> str:
        """Definitions as text, ready to be pasted into a system prompt."""
        return "\n".join(json.dumps(d, ensure_ascii=False) for d in self.definitions())

    def call(self, name: str, arguments: dict[str, Any]) -> ToolCallRecord:
        """Execute a tool call the model produced. Never raises: errors become observations."""
        started = time.perf_counter()
        if name not in self.tools:
            return ToolCallRecord(
                name=name,
                arguments=arguments,
                ok=False,
                error=f"unknown tool '{name}'. Available: {sorted(self.tools)}",
                latency_ms=0.0,
            )
        try:
            result = self.tools[name](**arguments)
            return ToolCallRecord(
                name=name,
                arguments=arguments,
                result=result,
                ok=True,
                latency_ms=1000 * (time.perf_counter() - started),
            )
        except ValidationError as exc:
            error = f"invalid arguments: {exc.errors(include_url=False)}"
        except Exception as exc:  # noqa: BLE001 - the model must see what went wrong
            error = f"{type(exc).__name__}: {exc}"
        return ToolCallRecord(
            name=name,
            arguments=arguments,
            ok=False,
            error=error,
            latency_ms=1000 * (time.perf_counter() - started),
        )
