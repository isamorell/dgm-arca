"""Tool definitions come from type hints and docstrings; calls are validated before running."""

import pytest

from tool_use import tools
from tool_use.registry import ToolRegistry, tool
from tool_use.tools import (
    convert_units,
    create_preparation_sheet,
    dose_calculator,
    search_medication_cima,
)


@tool
def add(a: int, b: int = 1) -> int:
    """Add two integers.

    Args:
        a: First number.
        b: Second number, defaults to 1.
    """
    return a + b


# ---- generic registry behaviour (the original tests) -----------------------------------


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


def test_dose_calculator_is_safe_and_correct():
    assert dose_calculator(expression="(12.5 * 3) / 2 - 1") == 17.75
    record = (
        ToolRegistry()
        .register(dose_calculator)
        .call("dose_calculator", {"expression": "__import__('os')"})
    )
    assert not record.ok


# ---- definitions of the domain tools ---------------------------------------------------


def _definition(t):
    return t.definition["function"]


@pytest.mark.parametrize(
    ("t", "name", "required"),
    [
        (dose_calculator, "dose_calculator", ["expression"]),
        (convert_units, "convert_units", ["value", "from_unit", "to_unit"]),
        (search_medication_cima, "search_medication_cima", ["active_ingredient"]),
        (
            create_preparation_sheet,
            "create_preparation_sheet",
            ["medication", "concentration", "dose", "administration", "steps"],
        ),
    ],
)
def test_domain_definitions_have_name_description_and_required_args(t, name, required):
    definition = _definition(t)
    assert definition["name"] == name
    assert len(definition["description"]) > 40
    assert definition["parameters"]["required"] == required


def test_domain_definition_types_and_defaults():
    props = _definition(dose_calculator)["parameters"]["properties"]
    assert props["decimals"]["type"] == "integer"
    assert _definition(convert_units)["parameters"]["properties"]["value"]["type"] == "number"
    assert _definition(search_medication_cima)["parameters"]["properties"]["max_results"]["type"] == "integer"


# ---- executing domain tools through the registry ---------------------------------------


@pytest.fixture
def registry():
    return ToolRegistry().register(
        dose_calculator, convert_units, search_medication_cima, create_preparation_sheet
    )


def test_only_the_action_tool_requires_confirmation():
    assert create_preparation_sheet.requires_confirmation is True
    for t in (dose_calculator, convert_units, search_medication_cima):
        assert t.requires_confirmation is False


def test_dose_calculator_through_registry_uses_default_decimals(registry):
    record = registry.call("dose_calculator", {"expression": "(15 * 70) / 250"})
    assert record.ok and record.result == 4.2 and record.latency_ms >= 0
    record = registry.call("dose_calculator", {"expression": "1 / 3", "decimals": 3})
    assert record.ok and record.result == 0.333


def test_convert_units_through_registry(registry):
    record = registry.call("convert_units", {"value": 250, "from_unit": "mcg", "to_unit": "mg"})
    assert record.ok and record.result == pytest.approx(0.25)


def test_domain_tools_report_invalid_arguments(registry):
    missing = registry.call("dose_calculator", {})
    assert not missing.ok and "invalid arguments" in missing.error
    wrong_type = registry.call(
        "convert_units", {"value": "ten", "from_unit": "mg", "to_unit": "g"}
    )
    assert not wrong_type.ok and "invalid arguments" in wrong_type.error


def test_domain_tools_return_runtime_errors_as_observations(registry):
    zero = registry.call("dose_calculator", {"expression": "1 / 0"})
    assert not zero.ok and "zero" in zero.error.lower()
    kinds = registry.call("convert_units", {"value": 10, "from_unit": "mg", "to_unit": "mL"})
    assert not kinds.ok and kinds.error


def test_preparation_sheet_through_registry_writes_a_pdf(registry, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    record = registry.call(
        "create_preparation_sheet",
        {
            "medication": "Paracetamol",
            "concentration": "10 mg/mL",
            "dose": "1050 mg",
            "administration": "105 mL",
            "steps": "15 mg/kg x 70 kg = 1050 mg\n1050 mg / 10 mg/mL = 105 mL",
        },
    )
    assert record.ok
    with open(record.result, "rb") as f:
        assert f.read().startswith(b"%PDF")


def test_cima_through_registry_with_mocked_http(registry, monkeypatch):
    class FakeResponse:
        status_code = 200
        content = b"x"

        def raise_for_status(self):
            pass

        def json(self):
            return {
                "resultados": [
                    {
                        "nregistro": "12345",
                        "nombre": "PARACETAMOL KABI 10 mg/ml",
                        "dosis": "10 mg/ml",
                        "formaFarmaceutica": {"nombre": "SOLUCION PARA PERFUSION"},
                        "viasAdministracion": [{"nombre": "VIA INTRAVENOSA"}],
                    }
                ]
            }

    monkeypatch.setattr(tools.requests, "get", lambda *a, **k: FakeResponse())
    record = registry.call("search_medication_cima", {"active_ingredient": "paracetamol"})
    assert record.ok and "10 mg/ml" in record.result