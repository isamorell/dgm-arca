"""Tool definitions come from type hints and docstrings; calls are validated before running."""

from tool_use.registry import ToolRegistry, tool
from tool_use.tools import calculator


@tool
def add(a: int, b: int = 1) -> int:
    """Add two integers.

    Args:
        a: First number.
        b: Second number, defaults to 1.
    """
    return a + b


def test_definition_has_name_description_and_schema():
    definition = add.definition["function"]
    assert definition["name"] == "add"
    assert definition["description"].startswith("Add two integers")
    params = definition["parameters"]
    assert params["properties"]["a"]["type"] == "integer"
    assert params["required"] == ["a"]


def test_registry_executes_valid_calls_and_times_them():
    registry = ToolRegistry().register(add)
    record = registry.call("add", {"a": 2, "b": 3})
    assert record.ok and record.result == 5 and record.latency_ms >= 0


def test_registry_reports_invalid_arguments_and_unknown_tools():
    registry = ToolRegistry().register(add)
    bad = registry.call("add", {"a": "two"})
    assert not bad.ok and "invalid arguments" in bad.error
    unknown = registry.call("multiply", {})
    assert not unknown.ok and "unknown tool" in unknown.error


def test_calculator_is_safe_and_correct():
    assert calculator(expression="(12.5 * 3) / 2 - 1") == 17.75
    record = (
        ToolRegistry().register(calculator).call("calculator", {"expression": "__import__('os')"})
    )
    assert not record.ok
