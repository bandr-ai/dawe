import pytest

from dawe import bfcl
from dawe.harness import Harness, available, parse_calls, render, target, to_bfcl

ITEM = bfcl.Item(
    id="t0",
    category="simple",
    question="Area of a triangle with base 10 and height 5?",
    functions=[
        {
            "name": "geo.triangle_area",
            "description": "Area of a triangle.",
            "parameters": {
                "type": "dict",
                "properties": {
                    "base_len": {"type": "integer", "description": "Base."},
                    "height": {"type": "integer", "description": "Height."},
                    "unit": {"type": "string", "description": "Unit."},
                },
                "required": ["base_len", "height"],
            },
        }
    ],
    ground_truth=[{"geo.triangle_area": {"base_len": [10], "height": [5], "unit": ["units", ""]}}],
)


def call(**args):
    return [{"name": "geo.triangle_area", "arguments": args}]


def test_checker_accepts_and_rejects():
    assert bfcl.check(call(base_len=10, height=5), ITEM)
    assert bfcl.check(call(base_len=10.0, height=5, unit="Units"), ITEM)  # int/float and case tolerant
    assert not bfcl.check(call(base_len=11, height=5), ITEM)  # wrong value
    assert not bfcl.check(call(base_len=10), ITEM)  # missing required
    assert not bfcl.check(call(base_len=10, height=5, color="red"), ITEM)  # unknown argument
    assert not bfcl.check([{"name": "triangle_area", "arguments": {"base_len": 10, "height": 5}}], ITEM)
    assert not bfcl.check(call(base_len=True, height=5), ITEM)  # bool is not int
    assert not bfcl.check(None, ITEM)
    assert not bfcl.check(call(base_len=10, height=5) * 2, ITEM)  # too many calls


@pytest.mark.parametrize("name", available())
def test_roundtrip_each_harness(name):
    h = Harness.load(name)
    _, names = render(h, ITEM)
    calls = parse_calls(h, target(h, ITEM))
    assert bfcl.check(to_bfcl(calls, names), ITEM)


def test_old_format_fails_under_new_harness():
    """A v1-style reply must not score under v4 (XML) or v2 (renamed tools)."""
    v1_reply = target(Harness.load("v1"), ITEM)
    v4 = Harness.load("v4_format")
    assert parse_calls(v4, v1_reply) is None
    v2 = Harness.load("v2_rename")
    _, names = render(v2, ITEM)
    assert not bfcl.check(to_bfcl(parse_calls(v2, v1_reply), names), ITEM)


@pytest.mark.parametrize("junk", ["", "sure!", "{not json", "[f(1)]", '<tool_call name="x">'])
def test_garbage_is_a_format_failure(junk):
    for name in available():
        h = Harness.load(name)
        if junk == "[f(1)]" and h.call_format == "pypos":
            continue  # valid positional syntax there; it fails grading instead
        assert parse_calls(h, junk) is None, name


def test_thinking_is_ignored():
    h = Harness.load("v1")
    assert parse_calls(h, "<think>maybe {\"x\": 1}</think>" + target(h, ITEM)) is not None


def test_xml_accepts_unquoted_strings():
    h = Harness.load("v4_format")
    text = '<tool_call name="geo.triangle_area">\n<arg name="base_len">10</arg>\n<arg name="height">5</arg>\n<arg name="unit">units</arg>\n</tool_call>'
    _, names = render(h, ITEM)
    assert bfcl.check(to_bfcl(parse_calls(h, text), names), ITEM)


def test_new_formats_reject_v1_replies():
    """A v1 JSON reply must not score under the new output formats."""
    v1_reply = target(Harness.load("v1"), ITEM)
    for name in ("v8_native", "v9_pycall", "v10_positional", "v11_plan"):
        h = Harness.load(name)
        _, names = render(h, ITEM)
        calls = parse_calls(h, v1_reply)
        assert calls is None or not bfcl.check(to_bfcl(calls, names), ITEM), name


def test_native_types_follow_schema():
    h = Harness.load("v8_native")
    _, names = render(h, ITEM)
    text = "<tool_call>\n<function=geo.triangle_area>\n<parameter=base_len>\n10\n</parameter>\n<parameter=height>\n5\n</parameter>\n</function>\n</tool_call>"
    assert bfcl.check(to_bfcl(parse_calls(h, text), names), ITEM)


def test_positional_maps_by_declared_order():
    h = Harness.load("v10_positional")
    _, names = render(h, ITEM)
    assert bfcl.check(to_bfcl(parse_calls(h, "[geo.triangle_area(10, 5)]"), names), ITEM)
    assert not bfcl.check(to_bfcl(parse_calls(h, "[geo.triangle_area(5, 10)]"), names), ITEM)
