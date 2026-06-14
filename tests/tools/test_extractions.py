"""Tests for all 3 extract_tools_from_content_* methods in cecli.helpers.responses."""

import json

from cecli.helpers.responses import (
    _parse_bracket_arguments,
    extract_tools_from_content_json,
    extract_tools_from_content_xml,
    extract_tools_from_pseudo_json,
)

# =============================================================================
# extract_tools_from_content_json
# =============================================================================


def test_json_single_tool_call_with_arguments_key():
    """Standard tool call with 'arguments' key should be extracted."""
    content = '{"name": "ls", "arguments": {"path": "."}}'
    result = extract_tools_from_content_json(content)
    assert result is not None
    assert len(result) == 1
    assert result[0].function.name == "ls"
    assert json.loads(result[0].function.arguments) == {"path": "."}


def test_json_single_tool_call_with_parameters_key():
    """Tool call with 'parameters' key should be extracted."""
    content = '{"name": "read_file", "parameters": {"file_path": "/tmp/test.txt"}}'
    result = extract_tools_from_content_json(content)
    assert result is not None
    assert len(result) == 1
    assert result[0].function.name == "read_file"
    assert json.loads(result[0].function.arguments) == {"file_path": "/tmp/test.txt"}


def test_json_single_tool_call_with_params_key():
    """Tool call with 'params' key should be extracted."""
    content = '{"name": "search", "params": {"query": "hello"}}'
    result = extract_tools_from_content_json(content)
    assert result is not None
    assert len(result) == 1
    assert result[0].function.name == "search"
    assert json.loads(result[0].function.arguments) == {"query": "hello"}


def test_json_array_of_tool_calls():
    """A JSON array of tool call objects should all be extracted."""
    content = (
        "["
        '{"name": "ls", "arguments": {"path": "."}},'
        '{"name": "grep", "arguments": {"pattern": "test"}}'
        "]"
    )
    result = extract_tools_from_content_json(content)
    assert result is not None
    assert len(result) == 2
    assert result[0].function.name == "ls"
    assert result[1].function.name == "grep"


def test_json_empty_content_returns_none():
    """Empty content should return None."""
    assert extract_tools_from_content_json("") is None


def test_json_no_braces_returns_none():
    """Content without braces or brackets should return None."""
    assert extract_tools_from_content_json("plain text") is None


def test_json_missing_required_keys_returns_none():
    """JSON without 'name' and arg keys should not be extracted."""
    content = '{"foo": "bar", "baz": 42}'
    assert extract_tools_from_content_json(content) is None


def test_json_malformed_json_returns_none():
    """Malformed JSON should return None."""
    content = '{"name": "ls", "arguments": }'
    assert extract_tools_from_content_json(content) is None


def test_json_with_string_arguments():
    """Tool call where arguments is a string (not dict/list) should work."""
    content = '{"name": "echo", "arguments": "hello world"}'
    result = extract_tools_from_content_json(content)
    assert result is not None
    assert len(result) == 1
    assert result[0].function.name == "echo"
    # String arguments get serialised as a JSON string
    assert result[0].function.arguments == "hello world"


def test_json_tool_with_nested_arguments():
    """Tool call with deeply nested arguments should work."""
    content = (
        '{"name": "ReadRange", "arguments": {'
        '"show": [{"file_path": "test.py", "start_text": "hello"}]'
        "}}"
    )
    result = extract_tools_from_content_json(content)
    assert result is not None
    assert len(result) == 1
    assert result[0].function.name == "ReadRange"
    args = json.loads(result[0].function.arguments)
    assert args["show"][0]["file_path"] == "test.py"


def test_json_multiple_tool_calls_in_list_first_not_valid():
    """Array where first item is not a valid tool call should skip it."""
    content = (
        "[" '{"irrelevant": true},' '{"name": "actual_tool", "arguments": {"key": "value"}}' "]"
    )
    result = extract_tools_from_content_json(content)
    assert result is not None
    assert len(result) == 1
    assert result[0].function.name == "actual_tool"


def test_json_all_invalid_array_items_returns_none():
    """Array where ALL items are invalid tool calls should return None."""
    content = "[" '{"irrelevant": true},' '{"also_invalid": [1, 2, 3]}' "]"
    result = extract_tools_from_content_json(content)
    assert result is None


# =============================================================================
# extract_tools_from_content_xml
# =============================================================================


def test_xml_single_tool_call():
    """Basic XML-style tool call should be extracted."""
    content = (
        "<function=UpdateTodoList>"
        "<parameter=tasks>"
        '[{"task": "Update task list", "done": false}]'
        "</parameter>"
        "</function>"
    )
    result = extract_tools_from_content_xml(content)
    assert result is not None
    assert len(result) == 1
    assert result[0].function.name == "UpdateTodoList"
    args = json.loads(result[0].function.arguments)
    assert "tasks" in args
    assert args["tasks"][0]["task"] == "Update task list"


def test_xml_multiple_parameters():
    """Tool call with multiple parameters should work."""
    content = (
        "<function=ReadRange>"
        "<parameter=file_path>"
        '"test.py"'
        "</parameter>"
        "<parameter=start_text>"
        '"hello"'
        "</parameter>"
        "</function>"
    )
    result = extract_tools_from_content_xml(content)
    assert result is not None
    assert len(result) == 1
    assert result[0].function.name == "ReadRange"
    args = json.loads(result[0].function.arguments)
    assert args["file_path"] == "test.py"
    assert args["start_text"] == "hello"


def test_xml_multiple_tool_calls():
    """Multiple XML tool calls in content should all be extracted."""
    content = (
        "Some text "
        "<function=ToolA>"
        "<parameter=arg1>"
        '"val1"'
        "</parameter>"
        "</function>"
        " more text "
        "<function=ToolB>"
        "<parameter=count>42</parameter>"
        "</function>"
    )
    result = extract_tools_from_content_xml(content)
    assert result is not None
    assert len(result) == 2
    assert result[0].function.name == "ToolA"
    assert result[1].function.name == "ToolB"
    args_b = json.loads(result[1].function.arguments)
    assert args_b["count"] == 42


def test_xml_empty_content_returns_none():
    """Empty content should return None."""
    assert extract_tools_from_content_xml("") is None


def test_xml_no_function_tags_returns_none():
    """Content without <function=...> tags should return None."""
    assert extract_tools_from_content_xml("plain text") is None


def test_xml_parameter_with_array_value():
    """Parameter with a JSON array value should parse correctly."""
    content = "<function=BulkTool>" "<parameter=items>" "[1, 2, 3, 4]" "</parameter>" "</function>"
    result = extract_tools_from_content_xml(content)
    assert result is not None
    assert len(result) == 1
    args = json.loads(result[0].function.arguments)
    assert args["items"] == [1, 2, 3, 4]


def test_xml_parameter_with_string_fallback():
    """Non-JSON parameter value should fall back to raw string."""
    content = (
        "<function=SimpleTool>"
        "<parameter=note>"
        "just some plain text"
        "</parameter>"
        "</function>"
    )
    result = extract_tools_from_content_xml(content)
    assert result is not None
    assert len(result) == 1
    args = json.loads(result[0].function.arguments)
    assert args["note"] == "just some plain text"


def test_xml_nested_in_text():
    """XML tool call embedded in surrounding text should be extracted."""
    content = (
        "I will use the UpdateTodoList tool.\n"
        "<function=UpdateTodoList>"
        "<parameter=tasks>"
        '[{"task": "test", "done": false}]'
        "</parameter>"
        "</function>"
        "\nThat should update the list."
    )
    result = extract_tools_from_content_xml(content)
    assert result is not None
    assert len(result) == 1
    assert result[0].function.name == "UpdateTodoList"


# =============================================================================
# extract_tools_from_pseudo_json
# =============================================================================


def test_pseudo_single_tool_with_array_arg():
    """Bracket format with a JSON array argument should be extracted."""
    content = '[Local--ReadRange(show=[{"file_path": "test.py", ' '"start_text": "def foo"}])]'
    result = extract_tools_from_pseudo_json(content)
    assert result is not None
    assert len(result) == 1
    assert result[0].function.name == "Local--ReadRange"
    args = json.loads(result[0].function.arguments)
    assert args["show"][0]["file_path"] == "test.py"


def test_pseudo_multiple_args_with_different_types():
    """Multiple args with boolean, string, and array values."""
    content = (
        '[Local--ReadRange(show=[{"file_path": "test.py", '
        '"start_text": "class A"}], verbose=true, mode="strict")]'
    )
    result = extract_tools_from_pseudo_json(content)
    assert result is not None
    assert len(result) == 1
    assert result[0].function.name == "Local--ReadRange"
    args = json.loads(result[0].function.arguments)
    assert args["verbose"] is True
    assert args["mode"] == "strict"
    assert len(args["show"]) == 1


def test_pseudo_multiple_tool_blocks():
    """Multiple bracket tool blocks in content should all be extracted."""
    content = 'First [ToolA(arg1="val1")] and ' "then [ToolB(count=42, flag=true)]"
    result = extract_tools_from_pseudo_json(content)
    assert result is not None
    assert len(result) == 2
    assert result[0].function.name == "ToolA"
    assert result[1].function.name == "ToolB"
    args_b = json.loads(result[1].function.arguments)
    assert args_b["count"] == 42
    assert args_b["flag"] is True


def test_pseudo_empty_content_returns_none():
    """Empty content should return None."""
    assert extract_tools_from_pseudo_json("") is None


def test_pseudo_no_brackets_returns_none():
    """Content without brackets should return None."""
    assert extract_tools_from_pseudo_json("plain text") is None


def test_pseudo_nested_parentheses():
    """Values with nested parentheses should be handled correctly."""
    content = '[DeepNest(calc="((1+2)*3)", name="test")]'
    result = extract_tools_from_pseudo_json(content)
    assert result is not None
    assert len(result) == 1
    assert result[0].function.name == "DeepNest"
    args = json.loads(result[0].function.arguments)
    assert args["calc"] == "((1+2)*3)"
    assert args["name"] == "test"


def test_pseudo_incomplete_bracket_no_match():
    """Missing closing bracket should not be extracted."""
    content = '[ToolA(arg1="val1"'
    assert extract_tools_from_pseudo_json(content) is None


def test_pseudo_missing_closing_paren():
    """Missing closing parenthesis should skip the block."""
    content = '[ToolA(arg1="val1") more text'
    assert extract_tools_from_pseudo_json(content) is None


def test_pseudo_tool_in_surrounding_text():
    """Bracket tool call embedded in text should be extracted."""
    content = (
        "I will use the Local--ReadRange tool:\n"
        '[Local--ReadRange(show=[{"file_path": "test.py"}])]'
        "\nThat should read the file."
    )
    result = extract_tools_from_pseudo_json(content)
    assert result is not None
    assert len(result) == 1
    assert result[0].function.name == "Local--ReadRange"


def test_pseudo_numeric_and_null_values():
    """Numeric and null values should parse correctly."""
    content = "[MathTool(x=42, y=3.14, optional=null)]"
    result = extract_tools_from_pseudo_json(content)
    assert result is not None
    assert len(result) == 1
    args = json.loads(result[0].function.arguments)
    assert args["x"] == 42
    assert args["y"] == 3.14
    assert args["optional"] is None


# =============================================================================
# _parse_bracket_arguments (helper)
# =============================================================================


def test_parse_bracket_arguments_single():
    """Single argument should parse correctly."""
    payload = 'show=[{"file_path": "test.py"}]'
    result = _parse_bracket_arguments(payload)
    assert result["show"] == [{"file_path": "test.py"}]


def test_parse_bracket_arguments_multiple():
    """Multiple arguments of mixed types should parse correctly."""
    payload = 'show=[{"file_path": "test.py"}], verbose=true, mode="strict"'
    result = _parse_bracket_arguments(payload)
    assert len(result) == 3
    assert result["verbose"] is True
    assert result["mode"] == "strict"
    assert result["show"] == [{"file_path": "test.py"}]


def test_parse_bracket_arguments_empty():
    """Empty payload should return empty dict."""
    assert _parse_bracket_arguments("") == {}


def test_parse_bracket_arguments_numbers_and_null():
    """Numbers and null values should be parsed correctly."""
    payload = "x=42, y=3.14, z=null"
    result = _parse_bracket_arguments(payload)
    assert result["x"] == 42
    assert result["y"] == 3.14
    assert result["z"] is None


def test_parse_bracket_arguments_escaped_quotes():
    """Values with escaped double quotes inside strings should parse correctly."""
    payload = 'arg="value with \\"escaped quote\\" inside"'
    result = _parse_bracket_arguments(payload)
    assert result["arg"] == 'value with "escaped quote" inside'


def test_parse_bracket_arguments_escaped_backslash():
    """Values with escaped backslashes inside strings should parse correctly."""
    payload = 'path="C:\\\\Users\\\\test"'
    result = _parse_bracket_arguments(payload)
    assert result["path"] == "C:\\Users\\test"


def test_parse_bracket_arguments_single_quoted_string():
    """Single-quoted string values retain their quotes (JSON doesn't parse single quotes)."""
    payload = "name='hello world'"
    result = _parse_bracket_arguments(payload)
    assert result["name"] == "'hello world'"
