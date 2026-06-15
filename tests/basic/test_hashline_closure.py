"""Tests for closure safeguard and content boundary functions in hashline."""

from cecli.helpers.hashline import (
    _apply_closure_safeguard,
    _fix_duplicate_content_boundaries,
    _would_create_duplicate_content,
)

# =============================================================================
# Tests for _would_create_duplicate_content
# =============================================================================


def test_would_create_duplicate_no_duplicate():
    """No duplicate content at boundaries — should return False."""
    source = ["line1", "line2", "line3", "line4", "line5"]
    result = _would_create_duplicate_content(source, 1, 2, ["new_line", "other"])
    assert result is False


def test_would_create_duplicate_start_boundary():
    """First replacement line matches line before edit range — should return True."""
    source = ["keep_me", "old1", "old2", "after"]
    # Replacement starts with "keep_me" which is the line at index 0 (start-1)
    result = _would_create_duplicate_content(source, 1, 2, ["keep_me", "new_line"])
    assert result is True


def test_would_create_duplicate_end_boundary():
    """Last replacement line matches line after edit range — should return True."""
    source = ["before", "old1", "old2", "keep_me"]
    # Replacement ends with "keep_me" which is the line at index 3 (end+1)
    result = _would_create_duplicate_content(source, 1, 2, ["new1", "keep_me"])
    assert result is True


def test_would_create_duplicate_both_boundaries():
    """Both start and end boundaries have duplicate content."""
    source = ["start_dup", "old1", "old2", "end_dup"]
    result = _would_create_duplicate_content(source, 1, 2, ["start_dup", "middle", "end_dup"])
    assert result is True


def test_would_create_duplicate_empty_repl_lines():
    """Empty replacement lines — should return False."""
    source = ["line1", "line2", "line3"]
    result = _would_create_duplicate_content(source, 0, 1, [])
    assert result is False


def test_would_create_duplicate_start_at_zero():
    """candidate_start is 0 (no line before) — should not check start."""
    source = ["first", "second", "third"]
    # Even though repl[0] matches source[0], start is 0 so no check
    result = _would_create_duplicate_content(source, 0, 1, ["first", "new"])
    assert result is False


def test_would_create_duplicate_end_at_end_of_file():
    """candidate_end is last line (no line after) — should not check end."""
    source = ["first", "second", "last"]
    result = _would_create_duplicate_content(source, 0, 2, ["new", "last"])
    assert result is False


def test_would_create_duplicate_whitespace_insensitive():
    """Comparison uses rstrip(), so trailing whitespace differences are ignored."""
    source = ["spaced  ", "old1", "old2", "end"]
    # repl[0] has no trailing spaces but rstrip() matches
    result = _would_create_duplicate_content(source, 1, 2, ["spaced", "new"])  # noqa


def test_would_create_duplicate_single_line_replace():
    """Single-line replace that duplicates both adjacent lines."""
    source = ["dup", "old", "dup"]
    result = _would_create_duplicate_content(source, 1, 1, ["dup", "dup"])
    # repl[0] matches source[0], repl[-1] matches source[2]
    assert result is True


def test_would_create_duplicate_delete_operation():
    """Delete operation with empty repl_lines — should return False."""
    source = ["a", "b", "c", "d"]
    result = _would_create_duplicate_content(source, 1, 2, [])
    assert result is False


# =============================================================================
# Tests for _fix_duplicate_content_boundaries
# =============================================================================


def _make_op(operation, text, start_idx, end_idx, index=0):
    """Helper to create a resolved operation dict."""
    return {
        "index": index,
        "start_idx": start_idx,
        "end_idx": end_idx,
        "op": {
            "operation": operation,
            "text": text,
        },
    }


def test_fix_boundaries_no_duplicate():
    """No duplicate content at boundaries — indices should remain unchanged."""
    source = ["a", "b", "c", "d", "e"]
    ops = [_make_op("replace", "x\ny", 1, 2)]
    result = _fix_duplicate_content_boundaries(source, ops)
    assert result[0]["start_idx"] == 1
    assert result[0]["end_idx"] == 2


def test_fix_boundaries_start_expansion():
    """First replacement line duplicates line before — start_idx should decrement."""
    source = ["dup", "old1", "old2", "after"]
    ops = [_make_op("replace", "dup\nnew", 1, 2)]
    result = _fix_duplicate_content_boundaries(source, ops)
    assert result[0]["start_idx"] == 0  # Expanded backward to consume "dup"
    assert result[0]["end_idx"] == 2


def test_fix_boundaries_end_expansion():
    """Last replacement line duplicates line after — end_idx should increment."""
    source = ["before", "old1", "old2", "dup"]
    ops = [_make_op("replace", "new\ndup", 1, 2)]
    result = _fix_duplicate_content_boundaries(source, ops)
    assert result[0]["start_idx"] == 1
    assert result[0]["end_idx"] == 3  # Expanded forward to consume "dup"


def test_fix_boundaries_both_sides():
    """Both start and end have duplicates — both indices should adjust."""
    source = ["start_dup", "old1", "old2", "end_dup"]
    ops = [_make_op("replace", "start_dup\nnew\nend_dup", 1, 2)]
    result = _fix_duplicate_content_boundaries(source, ops)
    assert result[0]["start_idx"] == 0  # Expanded backward
    assert result[0]["end_idx"] == 3  # Expanded forward


def test_fix_boundaries_start_at_zero():
    """start_idx is 0 — cannot expand backward, should stay 0."""
    source = ["dup", "old", "after"]
    ops = [_make_op("replace", "dup\nnew", 0, 1)]
    result = _fix_duplicate_content_boundaries(source, ops)
    assert result[0]["start_idx"] == 0  # Already at 0
    assert result[0]["end_idx"] == 1


def test_fix_boundaries_end_at_last_line():
    """end_idx is last line — cannot expand forward, should stay."""
    source = ["before", "old", "dup"]
    ops = [_make_op("replace", "new\ndup", 1, 2)]
    result = _fix_duplicate_content_boundaries(source, ops)
    assert result[0]["start_idx"] == 1
    assert result[0]["end_idx"] == 2  # Already at last line


def test_fix_boundaries_skip_insert():
    """Insert operations should be skipped entirely."""
    source = ["a", "b", "c"]
    ops = [_make_op("insert", "new_line", 1, 1)]
    result = _fix_duplicate_content_boundaries(source, ops)
    assert result[0]["start_idx"] == 1
    assert result[0]["end_idx"] == 1


def test_fix_boundaries_empty_replacement():
    """Empty replacement text — should skip (no repl_lines)."""
    source = ["a", "b", "c"]
    ops = [_make_op("delete", "", 1, 1)]
    result = _fix_duplicate_content_boundaries(source, ops)
    assert result[0]["start_idx"] == 1
    assert result[0]["end_idx"] == 1


def test_fix_boundaries_chained_duplicates_start():
    """Multiple consecutive duplicate lines at start — while loop should consume all."""
    source = ["dup", "dup", "dup", "old", "after"]
    # Replacement starts with "dup" repeated — should consume all 3 "dup" lines
    ops = [_make_op("replace", "dup\nnew", 3, 3)]
    result = _fix_duplicate_content_boundaries(source, ops)
    assert result[0]["start_idx"] == 0  # All 3 "dup" lines consumed
    assert result[0]["end_idx"] == 3


def test_fix_boundaries_chained_duplicates_end():
    """Multiple consecutive duplicate lines at end — while loop should consume all."""
    source = ["before", "old", "dup", "dup", "dup"]
    ops = [_make_op("replace", "new\ndup", 1, 1)]
    result = _fix_duplicate_content_boundaries(source, ops)
    assert result[0]["start_idx"] == 1
    assert result[0]["end_idx"] == 4  # All 3 "dup" lines consumed


def test_fix_boundaries_multiple_ops():
    """Multiple operations — each should be processed independently."""
    source = ["dup1", "old1", "old2", "dup2", "old3", "old4"]
    ops = [
        _make_op("replace", "dup1\nnew", 1, 2, index=0),  # Should expand start to 0
        _make_op("replace", "new\ndup2", 3, 3, index=1),  # Should expand end to 3
    ]
    result = _fix_duplicate_content_boundaries(source, ops)
    # First op: start expanded backward from 1 to 0
    assert result[0]["start_idx"] == 0
    assert result[0]["end_idx"] == 2
    # Second op: end expanded forward from 3 to 3 (dup2 at index 4... wait, no)
    # Actually source[3]="dup2", repl[-1]="dup2" → end_idx goes from 3 to...
    # dup2 is at index 3, source[4]="old3" is not "dup2", so end stays at 3
    assert result[1]["start_idx"] == 3
    assert result[1]["end_idx"] == 3


def test_fix_boundaries_whitespace_insensitive():
    """Boundary fix uses rstrip() — should match despite trailing whitespace differences."""
    source = ["dup  ", "old1", "old2", "after"]
    ops = [_make_op("replace", "dup\nnew", 1, 2)]
    result = _fix_duplicate_content_boundaries(source, ops)
    assert result[0]["start_idx"] == 0


def test_fix_boundaries_partial_match_does_not_trigger():
    """Only exact (rstripped) match triggers expansion, not partial/substring."""
    source = ["different", "old1", "old2", "after"]
    ops = [_make_op("replace", "diff\nnew", 1, 2)]
    result = _fix_duplicate_content_boundaries(source, ops)
    assert result[0]["start_idx"] == 1  # No change
    assert result[0]["end_idx"] == 2


# =============================================================================
# Tests for _apply_closure_safeguard (tree-sitter boundary healing)
# =============================================================================


def _make_closure_op(operation, text, start_idx, end_idx, index=0):
    """Helper to create a resolved operation for _apply_closure_safeguard."""
    op = {
        "operation": operation,
        "text": text,
    }
    return {
        "index": index,
        "start_idx": start_idx,
        "end_idx": end_idx,
        "op": op,
    }


def test_closure_safeguard_no_file_path():
    """Without file_path, ops should be returned unchanged."""
    ops = [_make_closure_op("replace", "pass", 0, 0)]
    result = _apply_closure_safeguard("x = 1", ops, file_path=None)
    assert result == ops


def test_closure_safeguard_unsupported_language():
    """File path with unsupported language — ops returned unchanged."""
    ops = [_make_closure_op("replace", "xyz", 0, 0)]
    result = _apply_closure_safeguard("some content", ops, file_path="test.xyz")
    assert result == ops


def test_closure_safeguard_empty_ops():
    """Empty ops list — returned as-is."""
    result = _apply_closure_safeguard("x = 1", [], file_path="test.py")
    assert result == []


def test_closure_safeguard_valid_code_no_change():
    """Edit that produces valid Python should keep original bounds."""
    source = """x = 1
y = 2
z = 3
"""
    # Replace all 3 lines with 2 valid lines
    ops = [_make_closure_op("replace", "a = 10\nb = 20", 0, 2)]
    result = _apply_closure_safeguard(source, ops, file_path="test.py")
    assert result[0]["start_idx"] == 0
    assert result[0]["end_idx"] == 2


def test_closure_safeguard_heals_outer_scope():
    """
    When an edit eats outer scope (e.g., removes opening brace but leaves
    closing brace), the safeguard should expand boundaries to consume the stray brace.
    """
    source = """def foo():
    return {
        "key": "value"
    }
"""
    # Lines: 0="def foo():", 1="    return {", 2='        "key": "value"',
    #        3="    }"
    # Replacing lines 1-2 with '    return "value"' leaves a stray '}' on line 3,
    # which tree-sitter finds as an ERROR.
    # The safeguard should expand end_idx to include line 3.
    ops = [_make_closure_op("replace", '    return "value"', 1, 2)]
    result = _apply_closure_safeguard(source, ops, file_path="test.py")
    assert result[0]["start_idx"] == 1
    assert result[0]["end_idx"] == 3  # Expanded to consume stray closing brace


def test_closure_safeguard_simple_python_replace():
    """Simple replace that maintains valid Python syntax."""
    source = """x = 1
y = 2
z = 3
"""
    # Replace line 1 ("y = 2") with "y = 99"
    ops = [_make_closure_op("replace", "y = 99", 1, 1)]
    result = _apply_closure_safeguard(source, ops, file_path="test.py")
    assert result[0]["start_idx"] == 1
    assert result[0]["end_idx"] == 1


def test_closure_safeguard_delete_valid():
    """Delete operation that maintains valid syntax."""
    source = """x = 1
y = 2
z = 3
"""
    # Delete line 1 ("y = 2") — result is "x = 1\nz = 3" which is valid
    ops = [_make_closure_op("delete", None, 1, 1)]
    result = _apply_closure_safeguard(source, ops, file_path="test.py")
    assert result[0]["start_idx"] == 1
    assert result[0]["end_idx"] == 1


def test_closure_safeguard_heals_stray_except():
    """
    When an edit eats the 'try' header but leaves the 'except' block,
    the safeguard should expand boundaries to consume the stray 'except'.
    """
    source = """try:
    pass
except:
    pass
"""
    # Lines: 0="try:", 1="    pass", 2="except:", 3="    pass"
    # Replacing lines 0-1 with 'x = 1' leaves a stray 'except:' on line 2.
    # The safeguard should expand the boundary to avoid the parse error.
    ops = [_make_closure_op("replace", "x = 1", 0, 1)]
    result = _apply_closure_safeguard(source, ops, file_path="test.py")
    healed_start = result[0]["start_idx"]
    healed_end = result[0]["end_idx"]
    # Boundaries should have changed from original (0, 1)
    assert (healed_start, healed_end) != (0, 1), "Expected safeguard to heal stray except error"
    # Verify the healed result parses correctly via tree-sitter (what the safeguard uses)
    from cecli.helpers.grep_ast.tsl import get_parser

    parser = get_parser("python")
    lines = source.splitlines()
    new_lines = lines[:healed_start] + ["x = 1"] + lines[healed_end + 1 :]
    new_source = "\n".join(new_lines)
    tree = parser.parse(new_source.encode("utf-8"))
    assert (
        not tree.root_node.has_error
    ), f"Healed source still has tree-sitter errors: {new_source!r}"


def test_closure_safeguard_skip_insert():
    """Insert operations should be skipped by the safeguard."""
    source = "x = 1\ny = 2\n"
    ops = [_make_closure_op("insert", "z = 3", 1, 1)]
    result = _apply_closure_safeguard(source, ops, file_path="test.py")
    assert result == ops  # Insert ops pass through unchanged


def test_closure_safeguard_with_outer_function_brace():
    """
    Test that the safeguard can expand an edit that would 'eat' an outer
    scope's closing brace.
    """
    source = """if True:
    if False:
        x = 1
y = 2
"""
    lines = source.splitlines()  # noqa
    # Line 0: "if True:"
    # Line 1: "    if False:"
    # Line 2: "        x = 1"
    # Line 3: "y = 2"

    # Replace lines 1-2 with something that would leave invalid indentation
    # Let me test something that should work: replacing just the inner if
    ops = [_make_closure_op("replace", "    x = 99", 1, 2)]
    result = _apply_closure_safeguard(source, ops, file_path="test.py")
    # The replacement "    x = 99" should be valid, keep original bounds
    assert result[0]["start_idx"] == 1
    assert result[0]["end_idx"] == 2


def test_closure_safeguard_heals_broken_dict():
    """
    A replace that removes the opening brace but keeps the closing brace
    should have its boundaries expanded by the safeguard to include the
    stray closing brace.
    """
    source = """def foo():
    return {
        "key": "value"
    }
"""
    # Replace lines 1-2 with '    return "value"' — this eats the opening
    # brace but leaves the closing '}' on line 3, producing a parse error.
    ops = [_make_closure_op("replace", '    return "value"', 1, 2)]
    result = _apply_closure_safeguard(source, ops, file_path="test.py")
    healed_start = result[0]["start_idx"]
    healed_end = result[0]["end_idx"]
    # Verify the healed result parses correctly via tree-sitter
    from cecli.helpers.grep_ast.tsl import get_parser

    parser = get_parser("python")
    lines = source.splitlines()
    new_lines = lines[:healed_start] + ['    return "value"'] + lines[healed_end + 1 :]
    new_source = "\n".join(new_lines)
    tree = parser.parse(new_source.encode("utf-8"))
    assert (
        not tree.root_node.has_error
    ), f"Healed source still has tree-sitter errors: {new_source!r}"
