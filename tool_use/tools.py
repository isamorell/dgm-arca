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

import re
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from xml.sax.saxutils import escape

import requests

from tool_use.registry import ToolRegistry, tool

CIMA_API = "https://cima.aemps.es/cima/rest"

# Where create_preparation_sheet writes its PDF files
SHEETS_DIR = Path("tool_use/outputs/sheets")

@tool
def search_medication_cima(active_ingredient: str, max_results: int = 5) -> str:
    """Search the Spanish medicines agency (CIMA, AEMPS) for marketed presentations of a drug.
 
    Returns, for each medicine, its name, the amount of active ingredient per unit (dose,
    for example '40 mg' per tablet or '20 mg' per ampoule), pharmaceutical form and routes
    of administration. For injectables and oral solutions the concentration is usually
    written in the name (for example '20mg/2ml'), so read it from there. It does not
    calculate anything: do the arithmetic with dose_calculator or convert_units.
 
    Args:
        active_ingredient: Active ingredient name in Spanish, without dose or form,
            for example 'paracetamol', 'ibuprofeno' or 'furosemida'.
        max_results: Maximum number of medicines to return, from 1 to 10.
    """
    max_results = max(1, min(max_results, 10))
    response = requests.get(
        f"{CIMA_API}/medicamentos",
        params={"practiv1": active_ingredient.strip(), "comerc": 1, "pagina": 1},
        timeout=15,
    )
    response.raise_for_status()
    results = response.json().get("resultados") if response.content else None
    if not results:
        raise ValueError(f"No marketed medicines found for active ingredient {active_ingredient!r}")
    
    lines = []
    for med in results[:max_results]:
        form = (med.get("formaFarmaceutica") or {}).get("nombre", "n/a")
        routes = ", ".join(r["nombre"] for r in med.get("viasAdministracion") or []) or "n/a"
        lines.append(
            f"- {med['nombre']} | dose: {med.get('dosis', 'n/a')} | form: {form}"
            f" | routes: {routes} | registration: {med['nregistro']}"
        )

    return "\n".join(lines)

@lru_cache(maxsize=1)
def _unit_registry():
    """Pint registry with the abbreviations nurses use (mcg)."""
    import pint

    ureg = pint.UnitRegistry()
    ureg.define("mcg = 1e-6 * gram")
    return ureg

@tool
def convert_units(value: float, from_unit: str, to_unit: str) -> float:
    """Convert a quantity between two units of the same kind, such as mass, volume, time or rate.
 
    Examples: 0.5 g to mg, 250 mcg to mg, 1.5 L to mL, 2 h to min, 5 mg/kg to mg/lb,
    100 mL/h to mL/min. It cannot convert between different kinds (for example mg to mL):
    that needs the concentration, so use dose_calculator for it.
 
    Args:
        value: The numeric quantity to convert.
        from_unit: Unit of the value, for example 'mg', 'mcg', 'g', 'mL', 'L', 'kg', 'h', 'mg/kg/h'.
        to_unit: Unit to convert to, written in the same style as from_unit.
    """
    import pint

    ureg = _unit_registry()
    try:
        return float(ureg.Quantity(value, from_unit).to(to_unit).magnitude)
    except pint.errors.DimensionalityError as exc:
        raise ValueError(f"cannot convert {from_unit!r} to {to_unit!r}: different kinds of unit") from exc

@tool
def dose_calculator(expression: str, decimals: int = 2) -> float:
    """Compute exactly any multiplication or division of doses, weights, volumes or rates.
 
    Always use this tool instead of calculating in your head when a step multiplies or
    divides numbers, for example '(15 * 70) / 250' for a 15 mg/kg dose in a 70 kg patient
    with a 250 mg/mL stock, or '500 / (8 * 60)' for mL/min. Units must already be
    consistent (convert them first with convert_units). The arithmetic is exact (no
    floating-point error) and the result is rounded half up to the given decimals.
    Only numbers, parentheses and the operators + - * / ** (integer exponents) are allowed.
 
    Args:
        expression: Arithmetic expression with plain numbers, without units,
            for example '(0.5 * 12) / 2.5'. Use a dot as the decimal separator.
        decimals: Number of decimal places of the result, from 0 to 6.
    """
    import ast
    import math
    from fractions import Fraction
 
    if not 0 <= decimals <= 6:
        raise ValueError("decimals must be between 0 and 6")
 
    def evaluate(node) -> Fraction:
        if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
            return Fraction(str(node.value))  # str() keeps '0.1' exact
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub | ast.UAdd):
            value = evaluate(node.operand)
            return -value if isinstance(node.op, ast.USub) else value
        if isinstance(node, ast.BinOp):
            left, right = evaluate(node.left), evaluate(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                if right == 0:
                    raise ValueError("division by zero")
                return left / right
            if isinstance(node.op, ast.Pow) and right.denominator == 1 and abs(right) <= 10:
                return left ** int(right)
        raise ValueError("unsupported expression: use only numbers, parentheses and + - * / **")
 
    result = evaluate(ast.parse(expression, mode="eval").body)
    scaled = abs(result) * 10**decimals
    rounded = Fraction(math.floor(scaled + Fraction(1, 2)), 10**decimals)
    return float(-rounded if result < 0 else rounded)

@tool(requires_confirmation=True)
def create_preparation_sheet(
    medication: str, concentration: str, dose: str, administration: str, steps: str
) -> str:
    """Create a medication preparation sheet as a PDF file and return its file path.
 
    This writes a file, so use it only when the user explicitly asks for a preparation
    sheet, and only after the calculation is finished. Never include patient names.
 
    Args:
        medication: Medication name, for example 'Paracetamol'.
        concentration: Concentration or strength of the stock used, with unit, for example '10 mg/mL'.
        dose: Calculated dose, with unit, for example '1050 mg'.
        administration: Volume or rate to administer, with unit, for example '105 mL' or '21 mL/h'.
        steps: Main calculation steps, one per line, for example '15 mg/kg x 70 kg = 1050 mg'.
    """
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
 
    step_lines = [line.strip() for line in steps.splitlines() if line.strip()]
    if not step_lines:
        raise ValueError("steps must contain at least one calculation step")
 
    SHEETS_DIR.mkdir(parents=True, exist_ok=True)
    now = datetime.now()
    slug = re.sub(r"[^a-z0-9]+", "-", medication.lower()).strip("-") or "medication"
    path = SHEETS_DIR / f"{now.strftime('%Y%m%d-%H%M%S')}-{slug}.pdf"
 
    styles = getSampleStyleSheet()
    cell = styles["BodyText"]
    fields = [
        ("Medication", medication),
        ("Concentration", concentration),
        ("Calculated dose", dose),
        ("Volume / rate to administer", administration),
    ]
    table = Table(
        [[Paragraph(f"<b>{escape(k)}</b>", cell), Paragraph(escape(v), cell)] for k, v in fields],
        colWidths=[170, 300],
    )
    table.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.5, colors.grey)]))
 
    story = [
        Paragraph("Medication preparation sheet", styles["Title"]),
        Paragraph(now.strftime("%Y-%m-%d %H:%M"), styles["Normal"]),
        Spacer(1, 12),
        table,
        Spacer(1, 12),
        Paragraph("Calculation steps", styles["Heading2"]),
    ]
    story += [Paragraph(f"{i}. {escape(line)}", styles["Normal"]) for i, line in enumerate(step_lines, 1)]
    SimpleDocTemplate(str(path), pagesize=A4, title="Medication preparation sheet").build(story)
    return str(path)

def default_registry() -> ToolRegistry:
    """The registry the executor and the agent use. Register your tools here."""
    # Tu turno: registry.register(my_domain_api, my_domain_compute, my_domain_action)
    registry = ToolRegistry().register(search_medication_cima, convert_units, dose_calculator, create_preparation_sheet)
    return registry
