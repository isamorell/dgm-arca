import pytest

from api.schemas import ToolCallRecord
from tool_use.executor import ToolLoop, build_system_prompt
from tool_use.parser import format_tool_result, parse_tool_calls
from tool_use.registry import ToolRegistry, tool

WRITTEN: list[str] = []


@tool
def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b


@tool
def boom(x: int) -> int:
    """Always fails."""
    raise RuntimeError("kaboom")


@tool(requires_confirmation=True)
def write_note(text: str) -> str:
    """Write a note (side effect)."""
    WRITTEN.append(text)
    return "saved"


def call(name, **arguments):
    import json

    return f'<tool_call>\n{json.dumps({"name": name, "arguments": arguments})}\n</tool_call>'


def make_loop(monkeypatch, outputs, allow_side_effects=True):
    """ToolLoop whose generate() returns scripted outputs and records what it received."""
    registry = ToolRegistry().register(add, boom, write_note)
    loop = ToolLoop(registry=registry, allow_side_effects=allow_side_effects)
    it = iter(outputs)
    loop.seen = []  # snapshot of the conversation at each generate() call

    def fake_generate(messages):
        loop.seen.append(list(messages))
        return next(it)

    monkeypatch.setattr(loop, "generate", fake_generate)
    return loop


@pytest.fixture(autouse=True)
def _clear_written():
    WRITTEN.clear()


# ----------------------------------------------------------------- happy paths
def test_answer_without_tools(monkeypatch):
    r = make_loop(monkeypatch, ["Hola, ¿en qué te ayudo?"]).run("hola")
    assert r.answer == "Hola, ¿en qué te ayudo?"
    assert r.turns == 1 and r.tool_calls == []


def test_tool_then_answer(monkeypatch):
    loop = make_loop(monkeypatch, [call("add", a=2, b=3), "El resultado es 5."])
    r = loop.run("suma 2 y 3")
    assert r.answer == "El resultado es 5." and r.turns == 2
    assert r.tool_calls[0].ok and r.tool_calls[0].result == 5
    # the result went back to the model as a 'tool' message
    last = loop.seen[1][-1]
    assert last["role"] == "tool" and last["name"] == "add" and "5" in last["content"]


def test_parallel_calls_in_one_turn(monkeypatch):
    two = call("add", a=1, b=1) + "\n" + call("add", a=2, b=2)
    r = make_loop(monkeypatch, [two, "Listo."]).run("dos sumas")
    assert r.turns == 2
    assert [c.result for c in r.tool_calls] == [2, 4]


def test_multi_step(monkeypatch):
    r = make_loop(
        monkeypatch, [call("add", a=1, b=2), call("add", a=3, b=4), "Fin."]
    ).run("encadena")
    assert r.turns == 3 and [c.result for c in r.tool_calls] == [3, 7]


# --------------------------------------------------------------- error handling
def test_unknown_tool(monkeypatch):
    r = make_loop(monkeypatch, [call("nope"), "ok"]).run("x")
    assert not r.tool_calls[0].ok and "unknown tool" in r.tool_calls[0].error


def test_invalid_arguments(monkeypatch):
    r = make_loop(monkeypatch, [call("add", a="texto", b=1), "ok"]).run("x")
    assert not r.tool_calls[0].ok and "invalid arguments" in r.tool_calls[0].error


def test_tool_exception_becomes_observation(monkeypatch):
    r = make_loop(monkeypatch, [call("boom", x=1), "ok"]).run("x")
    assert not r.tool_calls[0].ok and "kaboom" in r.tool_calls[0].error


def test_max_turns_reached(monkeypatch):
    r = make_loop(monkeypatch, [call("add", a=1, b=1)] * 3).run("x", max_turns=3)
    assert r.turns == 3 and len(r.tool_calls) == 3
    assert "No pude completar" in r.answer


# ------------------------------------------------------------- requires_confirmation
def test_side_effect_tool_blocked(monkeypatch):
    loop = make_loop(monkeypatch, [call("write_note", text="hola"), "Confirma."],
                     allow_side_effects=False)
    r = loop.run("guarda")
    rec = r.tool_calls[0]
    assert not rec.ok and rec.requires_confirmation and "NOT executed" in rec.error
    assert WRITTEN == []  # the function never ran


def test_side_effect_tool_allowed_and_flagged(monkeypatch):
    r = make_loop(monkeypatch, [call("write_note", text="hola"), "Hecho."],
                  allow_side_effects=True).run("guarda")
    rec = r.tool_calls[0]
    assert rec.ok and rec.requires_confirmation and WRITTEN == ["hola"]


def test_normal_tool_not_flagged(monkeypatch):
    r = make_loop(monkeypatch, [call("add", a=1, b=1), "ok"]).run("x")
    assert r.tool_calls[0].requires_confirmation is False


# ------------------------------------------------------------------ prompt, parser
def test_system_prompt_lists_tools():
    registry = ToolRegistry().register(add, boom)
    prompt = build_system_prompt(registry)
    assert '"name": "add"' in prompt and '"name": "boom"' in prompt


def test_first_message_is_system_then_user(monkeypatch):
    loop = make_loop(monkeypatch, ["ok"])
    loop.run("mi pregunta")
    system, user = loop.seen[0]
    assert system["role"] == "system" and user == {"role": "user", "content": "mi pregunta"}


def test_parser_ignores_think_and_keeps_text():
    out = "<think>pienso</think>Voy a calcular.\n" + call("add", a=1, b=2)
    parsed = parse_tool_calls(out)
    assert parsed.thinking == "pienso" and parsed.text == "Voy a calcular."
    assert parsed.tool_calls[0].name == "add" and parsed.has_tool_calls


def test_parser_accepts_string_arguments():
    out = '<tool_call>{"name": "add", "arguments": "{\\"a\\": 1, \\"b\\": 2}"}</tool_call>'
    assert parse_tool_calls(out).tool_calls[0].arguments == {"a": 1, "b": 2}


def test_format_tool_result():
    assert format_tool_result("add", "texto") == "texto"
    assert format_tool_result("add", {"x": 1}) == '{"x": 1}'


def test_record_type(monkeypatch):
    r = make_loop(monkeypatch, [call("add", a=1, b=1), "ok"]).run("x")
    assert isinstance(r.tool_calls[0], ToolCallRecord)

# ------------------------------------------- the adapter is only needed to load the model


def test_building_the_loop_needs_neither_env_vars_nor_the_adapter(monkeypatch):
    monkeypatch.delenv("ARCA_RLM_ADAPTER", raising=False)
    monkeypatch.delenv("ARCA_RLM_BASE_MODEL", raising=False)
    ToolLoop(registry=ToolRegistry().register(add))
    monkeypatch.setenv("ARCA_RLM_ADAPTER", "does/not/exist")
    ToolLoop(registry=ToolRegistry().register(add))


def test_load_fails_clearly_if_the_adapter_is_missing_and_there_is_no_repo(monkeypatch):
    monkeypatch.setenv("ARCA_RLM_ADAPTER", "does/not/exist")
    monkeypatch.delenv("ARCA_RLM_ADAPTER_REPO", raising=False)
    loop = ToolLoop(registry=ToolRegistry().register(add))
    with pytest.raises(FileNotFoundError, match="ARCA_RLM_ADAPTER_REPO"):
        loop.load()


def test_load_downloads_the_adapter_from_the_hub_if_it_is_missing(monkeypatch, tmp_path):
    import sys
    from types import SimpleNamespace

    from tool_use import executor

    downloads = []
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(snapshot_download=lambda repo, local_dir: downloads.append((repo, local_dir))),
    )
    fake = SimpleNamespace(from_pretrained=lambda *a, **k: SimpleNamespace(eval=lambda: "model"))
    monkeypatch.setattr(executor, "AutoTokenizer", fake)
    monkeypatch.setattr(executor, "AutoModelForCausalLM", fake)
    adapter = str(tmp_path / "adapter")
    monkeypatch.setenv("ARCA_RLM_ADAPTER", adapter)
    monkeypatch.setenv("ARCA_RLM_ADAPTER_REPO", "user/repo")
    ToolLoop(registry=ToolRegistry().register(add)).load()
    assert downloads == [("user/repo", adapter)]


def test_load_does_not_check_the_adapter_if_the_model_is_given(monkeypatch):
    from types import SimpleNamespace

    from tool_use import executor

    fake = SimpleNamespace(from_pretrained=lambda *a, **k: SimpleNamespace(eval=lambda: "model"))
    monkeypatch.setattr(executor, "AutoTokenizer", fake)
    monkeypatch.setattr(executor, "AutoModelForCausalLM", fake)
    monkeypatch.setenv("ARCA_RLM_ADAPTER", "does/not/exist")  # would raise if it were checked
    monkeypatch.delenv("ARCA_RLM_ADAPTER_REPO", raising=False)
    ToolLoop(registry=ToolRegistry().register(add), model_id="Qwen/Qwen3-0.6B").load()


def test_max_turns_reached_works_without_a_model_id(monkeypatch):
    monkeypatch.delenv("ARCA_RLM_BASE_MODEL", raising=False)
    r = make_loop(monkeypatch, [call("add", a=1, b=1)] * 2).run("x", max_turns=2)
    assert r.model == ""


def test_response_reports_the_model_that_answered(monkeypatch):
    monkeypatch.setenv("ARCA_RLM_BASE_MODEL", "Qwen/Qwen3-0.6B")
    r = make_loop(monkeypatch, [call("add", a=1, b=1), "Son 2."]).run("x")
    assert r.model == "Qwen/Qwen3-0.6B"
    monkeypatch.delenv("ARCA_RLM_BASE_MODEL")
    assert make_loop(monkeypatch, ["Hola."]).run("x").model == ""


def test_system_prompt_asks_to_keep_calling_tools_until_the_request_is_solved():
    prompt = build_system_prompt(ToolRegistry().register(add))
    assert "call the next tool" in prompt
    assert "Answer only when every part of the request is done" in prompt
