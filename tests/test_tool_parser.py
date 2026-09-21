"""The parser is infrastructure we give you; these tests pin down its behaviour."""

from tool_use.parser import parse_tool_calls

QWEN_OUTPUT = """<think>
The user wants the weather. I should call get_weather.
</think>

<tool_call>
{"name": "get_weather", "arguments": {"location": "Paris, France"}}
</tool_call>"""


def test_parses_think_and_single_call():
    parsed = parse_tool_calls(QWEN_OUTPUT)
    assert "call get_weather" in parsed.thinking
    assert parsed.has_tool_calls
    assert parsed.tool_calls[0].name == "get_weather"
    assert parsed.tool_calls[0].arguments == {"location": "Paris, France"}
    assert parsed.text == ""


def test_parses_parallel_calls_in_order():
    output = (
        '<tool_call>{"name": "a", "arguments": {"x": 1}}</tool_call>\n'
        '<tool_call>{"name": "b", "arguments": {"y": 2}}</tool_call>'
    )
    names = [c.name for c in parse_tool_calls(output).tool_calls]
    assert names == ["a", "b"]


def test_malformed_json_is_reported_not_dropped():
    parsed = parse_tool_calls('<tool_call>{"name": "a", "arguments": {oops}}</tool_call>')
    assert not parsed.has_tool_calls
    assert parsed.tool_calls[0].error and "JSON" in parsed.tool_calls[0].error


def test_plain_answer_has_no_calls():
    parsed = parse_tool_calls("The temperature in Paris is 18 °C.")
    assert not parsed.has_tool_calls
    assert parsed.text.startswith("The temperature")


def test_arguments_given_as_string_are_decoded():
    output = '<tool_call>{"name": "a", "arguments": "{\\"x\\": 1}"}</tool_call>'
    assert parse_tool_calls(output).tool_calls[0].arguments == {"x": 1}
