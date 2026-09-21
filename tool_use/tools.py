"""The tools your model can call. One is given as a worked example; the rest are yours.

``get_weather`` is the exact tool from the slides, using Open-Meteo (no API key). Study
how the docstring becomes the description and the type hints become the schema, then
write your own. Your domain needs at least three: one that queries the outside world
(a real HTTP API), one that computes or executes, and one that acts with an observable
effect. Ideas that have worked well: a currency or unit converter, a SQLite lookup, a
sandboxed Python evaluator, a calendar-event writer (``.ics`` file), a webhook notifier.

A tool that has side effects should be registered with ``requires_confirmation=True``;
in phase 4 you decide what the agent does with that flag.
"""

from __future__ import annotations

import requests

from tool_use.registry import ToolRegistry, tool

OPEN_METEO_GEOCODING = "https://geocoding-api.open-meteo.com/v1/search"
OPEN_METEO_FORECAST = "https://api.open-meteo.com/v1/forecast"


@tool
def get_weather(location: str) -> float:
    """Return the current air temperature in degrees Celsius for a location.

    Args:
        location: City and country, for example 'Madrid, Spain'.
    """
    geo = requests.get(OPEN_METEO_GEOCODING, params={"name": location, "count": 1}, timeout=15)
    geo.raise_for_status()
    results = geo.json().get("results") or []
    if not results:
        raise ValueError(f"no coordinates found for {location!r}")
    lat, lon = results[0]["latitude"], results[0]["longitude"]
    forecast = requests.get(
        OPEN_METEO_FORECAST,
        params={
            "latitude": lat,
            "longitude": lon,
            "current_weather": True,
            "temperature_unit": "celsius",
        },
        timeout=15,
    )
    forecast.raise_for_status()
    return float(forecast.json()["current_weather"]["temperature"])


@tool
def calculator(expression: str) -> float:
    """Evaluate an arithmetic expression such as '(12.5 * 3) / 2 - 1'.

    Only numbers, parentheses and the operators + - * / ** % are allowed.

    Args:
        expression: The arithmetic expression to evaluate.
    """
    import ast
    import operator

    allowed = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.Mod: operator.mod,
        ast.USub: operator.neg,
    }

    def evaluate(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in allowed:
            return allowed[type(node.op)](evaluate(node.left), evaluate(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in allowed:
            return allowed[type(node.op)](evaluate(node.operand))
        raise ValueError("unsupported expression")

    return float(evaluate(ast.parse(expression, mode="eval").body))


def default_registry() -> ToolRegistry:
    """The registry the executor and the agent use. Register your tools here."""
    registry = ToolRegistry().register(get_weather, calculator)
    # Tu turno: registry.register(my_domain_api, my_domain_compute, my_domain_action)
    return registry
