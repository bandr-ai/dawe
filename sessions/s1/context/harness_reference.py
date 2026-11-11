"""A harness is everything around the model that is not the task: tool naming,
how tools are described, the call format, and the system prompt.

Each harness version is a TOML file in harnesses/. Changing one field there is a
"harness change". Items are always scored in BFCL's own names, so every rename
here must be reversible.
"""

from __future__ import annotations

import ast
import json
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .bfcl import Call, Item, gold_calls

HARNESS_DIR = Path(__file__).resolve().parent.parent / "harnesses"


@dataclass(frozen=True)
class Harness:
    name: str
    description: str = ""
    # naming
    tool_prefix: str = ""
    tool_dots: str = "."  # replacement for "." in tool names
    tool_case: str = "keep"  # keep | camel
    arg_case: str = "keep"  # keep | camel
    # how tools are shown
    schema_style: str = "json"  # json | signature
    tools_in: str = "system"  # system | user
    # how the model must answer
    call_format: str = "json"  # json | xml | pycall
    system_prompt: str = "You are a helpful assistant that can call functions.\n\n{tools}\n\n{format}"
    extra: dict = field(default_factory=dict)

    @classmethod
    def load(cls, name_or_path: str | Path) -> "Harness":
        p = Path(name_or_path)
        if not p.suffix:
            p = HARNESS_DIR / f"{name_or_path}.toml"
        data = tomllib.loads(p.read_text())
        base = data.pop("extends", None)
        if base:
            merged = cls.load(base).__dict__ | data
            merged["name"] = data.get("name", p.stem)
            return cls(**merged)
        data.setdefault("name", p.stem)
        return cls(**data)


def available() -> list[str]:
    return sorted(p.stem for p in HARNESS_DIR.glob("*.toml"))


# --- renaming -------------------------------------------------------------------


def _camel(s: str) -> str:
    parts = re.split(r"_+", s)
    return parts[0] + "".join(p[:1].upper() + p[1:] for p in parts[1:]) if parts else s


def tool_name(h: Harness, orig: str) -> str:
    n = orig.replace(".", h.tool_dots)
    if h.tool_case == "camel":
        n = h.tool_dots.join(_camel(seg) for seg in n.split(h.tool_dots)) if h.tool_dots else _camel(n)
    return h.tool_prefix + n


def arg_name(h: Harness, orig: str) -> str:
    return _camel(orig) if h.arg_case == "camel" else orig


@dataclass
class Names:
    """Per-item maps between BFCL names and harness names (both directions)."""

    tools: dict[str, str]  # bfcl -> harness
    args: dict[str, dict[str, str]]  # bfcl tool -> {bfcl arg -> harness arg}

    def tool_back(self, n: str) -> str | None:
        return {v: k for k, v in self.tools.items()}.get(n)

    def arg_back(self, bfcl_tool: str, n: str) -> str | None:
        return {v: k for k, v in self.args.get(bfcl_tool, {}).items()}.get(n)


def names_for(h: Harness, item: Item) -> Names:
    tools, args = {}, {}
    for fn in item.functions:
        tools[fn["name"]] = tool_name(h, fn["name"])
        props = fn["parameters"].get("properties", {})
        args[fn["name"]] = {a: arg_name(h, a) for a in props}
        if len(set(args[fn["name"]].values())) != len(props):
            raise ValueError(f"{item.id}: arg rename collision in {fn['name']}")
    if len(set(tools.values())) != len(tools):
        raise ValueError(f"{item.id}: tool rename collision")
    return Names(tools, args)


# --- rendering tools --------------------------------------------------------------


def _rename_schema(fn: dict, names: Names) -> dict:
    p = fn["parameters"]
    amap = names.args[fn["name"]]
    return {
        "name": names.tools[fn["name"]],
        "description": fn.get("description", ""),
        "parameters": {
            "type": p.get("type", "dict"),
            "properties": {amap[k]: v for k, v in p.get("properties", {}).items()},
            "required": [amap[k] for k in p.get("required", [])],
        },
    }


def _signature(fn: dict) -> str:
    p = fn["parameters"]
    req = set(p.get("required", []))
    parts, docs = [], []
    for a, spec in p.get("properties", {}).items():
        t = spec.get("type", "any")
        parts.append(f"{a}: {t}" if a in req else f"{a}?: {t}")
        line = f"    {a}: {spec.get('description', '')}".rstrip()
        if "enum" in spec:
            line += f" One of {spec['enum']}."
        if "default" in spec:
            line += f" Default {spec['default']!r}."
        docs.append(line)
    head = f"{fn['name']}({', '.join(parts)})"
    return "\n".join([head, f"    {fn['description']}".rstrip(), *docs])


def render_tools(h: Harness, item: Item, names: Names) -> str:
    schemas = [_rename_schema(fn, names) for fn in item.functions]
    if h.schema_style == "json":
        return "Available functions (JSON schema):\n" + json.dumps(schemas, indent=2)
    if h.schema_style == "signature":
        return "Tools:\n\n" + "\n\n".join(_signature(s) for s in schemas)
    raise ValueError(f"unknown schema_style {h.schema_style}")


FORMAT_HELP = {
    "json": 'Reply with only a JSON object: {"tool_calls": [{"name": "<function>", "arguments": {...}}]}. '
    "Put several calls in the list when needed.",
    "xml": 'Reply with only tool calls, one per block:\n<tool_call name="<function>">\n<arg name="<argument>">'
    "<JSON value></arg>\n</tool_call>",
    "pycall": "Reply with only a Python list of calls, e.g. [func_a(x=1, y='b'), func_b(z=[1, 2])].",
}


def render(h: Harness, item: Item) -> tuple[list[dict], Names]:
    """Chat messages for this item under this harness."""
    names = names_for(h, item)
    tools = render_tools(h, item, names)
    fmt = FORMAT_HELP[h.call_format]
    if h.tools_in == "system":
        system = h.system_prompt.format(tools=tools, format=fmt)
        user = item.question
    elif h.tools_in == "user":
        system = h.system_prompt.format(tools="", format=fmt).strip()
        user = f"{tools}\n\nRequest: {item.question}"
    else:
        raise ValueError(f"unknown tools_in {h.tools_in}")
    return [{"role": "system", "content": system}, {"role": "user", "content": user}], names


# --- call formats -------------------------------------------------------------------


def format_calls(h: Harness, calls: list[Call]) -> str:
    """Write calls (already in harness names) in this harness's format."""
    if h.call_format == "json":
        return json.dumps({"tool_calls": calls}, ensure_ascii=False)
    if h.call_format == "xml":
        blocks = []
        for c in calls:
            args = "\n".join(
                f'<arg name="{k}">{json.dumps(v, ensure_ascii=False)}</arg>' for k, v in c["arguments"].items()
            )
            blocks.append(f'<tool_call name="{c["name"]}">\n{args}\n</tool_call>'.replace("\n\n", "\n"))
        return "\n".join(blocks)
    if h.call_format == "pycall":
        inner = ", ".join(
            f"{c['name']}({', '.join(f'{k}={v!r}' for k, v in c['arguments'].items())})" for c in calls
        )
        return f"[{inner}]"
    raise ValueError(f"unknown call_format {h.call_format}")


def _xml_value(v: str) -> Any:
    """XML args are JSON values, but plain text (unquoted strings) is accepted too."""
    try:
        return json.loads(v)
    except json.JSONDecodeError:
        return v.strip()


_THINK = re.compile(r"<think>.*?</think>", re.S)


def parse_calls(h: Harness, text: str) -> list[Call] | None:
    """Read calls (in harness names) from model output. None means the format was wrong."""
    text = _THINK.sub("", text).strip()
    text = re.sub(r"^```\w*\n?|\n?```$", "", text).strip()
    try:
        if h.call_format == "json":
            obj = json.loads(text[text.index("{") : text.rindex("}") + 1])
            calls = obj["tool_calls"]
            if not all(isinstance(c.get("name"), str) and isinstance(c.get("arguments"), dict) for c in calls):
                return None
            return [{"name": c["name"], "arguments": c["arguments"]} for c in calls]
        if h.call_format == "xml":
            calls = []
            for name, body in re.findall(r'<tool_call name="([^"]+)">(.*?)</tool_call>', text, re.S):
                args = {k: _xml_value(v) for k, v in re.findall(r'<arg name="([^"]+)">(.*?)</arg>', body, re.S)}
                calls.append({"name": name, "arguments": args})
            return calls or None
        if h.call_format == "pycall":
            tree = ast.parse(text[text.index("[") : text.rindex("]") + 1], mode="eval").body
            if not isinstance(tree, ast.List):
                return None
            return [
                {"name": ast.unparse(c.func), "arguments": {kw.arg: ast.literal_eval(kw.value) for kw in c.keywords}}
                for c in tree.elts
                if isinstance(c, ast.Call) and not c.args
            ] or None
    except (ValueError, KeyError, TypeError, SyntaxError, json.JSONDecodeError, AttributeError):
        return None
    raise ValueError(f"unknown call_format {h.call_format}")


# --- name mapping for calls ---------------------------------------------------------


def to_harness(calls: list[Call], names: Names) -> list[Call]:
    return [
        {
            "name": names.tools[c["name"]],
            "arguments": {names.args[c["name"]][k]: v for k, v in c["arguments"].items()},
        }
        for c in calls
    ]


def to_bfcl(calls: list[Call], names: Names) -> list[Call]:
    """Map harness-named calls back to BFCL names.

    A name this harness doesn't offer becomes "<unknown:...>", so it always fails the
    check, even when it happens to equal the BFCL name (e.g. an old v1 name under v2).
    """
    out = []
    for c in calls:
        tool = names.tool_back(c["name"]) or f"<unknown:{c['name']}>"
        args: dict[str, Any] = {}
        for k, v in c["arguments"].items():
            args[names.arg_back(tool, k) or f"<unknown:{k}>"] = v
        out.append({"name": tool, "arguments": args})
    return out


def target(h: Harness, item: Item) -> str:
    """The correct assistant reply under this harness (used for SFT and round-trip tests)."""
    names = names_for(h, item)
    return format_calls(h, to_harness(gold_calls(item), names))
