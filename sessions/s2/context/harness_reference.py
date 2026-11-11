"""A harness is everything around the model that is not the task: tool naming,
how tools are described, the call format, and the system prompt.

Each harness version is a TOML file in harnesses/. Changing one field there is a
"harness change". Items are always scored in BFCL's own names, so every rename
here must be reversible.
"""

from __future__ import annotations

import ast
import hashlib
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
    tool_case: str = "keep"  # keep | camel | opaque (per-item IDs like t_3f9a)
    arg_case: str = "keep"  # keep | camel | opaque (a1, a2, ... in schema order)
    # how tools are shown
    schema_style: str = "json"  # json | signature
    tools_in: str = "system"  # system | user
    # how the model must answer
    call_format: str = "json"  # json | xml | pycall | pypos (positional args) | qwen (Qwen3.5 native)
    require_plan: bool = False  # reply must start with <plan>...</plan>
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
    types: dict[str, dict[str, str]] = field(default_factory=dict)  # bfcl tool -> {bfcl arg -> schema type}

    def tool_back(self, n: str) -> str | None:
        return {v: k for k, v in self.tools.items()}.get(n)

    def arg_back(self, bfcl_tool: str, n: str) -> str | None:
        return {v: k for k, v in self.args.get(bfcl_tool, {}).items()}.get(n)


def _opaque_tool(item: Item, orig: str) -> str:
    """A meaningless ID that differs per item, so it can't be memorized."""
    return "t_" + hashlib.sha256(f"{item.id}:{orig}".encode()).hexdigest()[:4]


def names_for(h: Harness, item: Item) -> Names:
    tools, args, types = {}, {}, {}
    for fn in item.functions:
        types[fn["name"]] = {a: str(spec.get("type", "")) for a, spec in fn["parameters"].get("properties", {}).items()}
        tools[fn["name"]] = _opaque_tool(item, fn["name"]) if h.tool_case == "opaque" else tool_name(h, fn["name"])
        props = fn["parameters"].get("properties", {})
        if h.arg_case == "opaque":
            args[fn["name"]] = {a: f"a{i + 1}" for i, a in enumerate(props)}
        else:
            args[fn["name"]] = {a: arg_name(h, a) for a in props}
        if len(set(args[fn["name"]].values())) != len(props):
            raise ValueError(f"{item.id}: arg rename collision in {fn['name']}")
    if len(set(tools.values())) != len(tools):
        raise ValueError(f"{item.id}: tool rename collision")
    return Names(tools, args, types)


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
    if h.schema_style == "qwen":  # exactly what Qwen3.5's chat template writes for tools=[...]
        lines = "".join("\n" + json.dumps({"type": "function", "function": s}, ensure_ascii=False) for s in schemas)
        return "# Tools\n\nYou have access to the following functions:\n\n<tools>" + lines + "\n</tools>"
    raise ValueError(f"unknown schema_style {h.schema_style}")


FORMAT_HELP = {
    "json": 'Reply with only a JSON object: {"tool_calls": [{"name": "<function>", "arguments": {...}}]}. '
    "Put several calls in the list when needed.",
    "xml": 'Reply with only tool calls, one per block:\n<tool_call name="<function>">\n<arg name="<argument>">'
    "<JSON value></arg>\n</tool_call>",
    "pycall": "Reply with only a Python list of calls, e.g. [func_a(x=1, y='b'), func_b(z=[1, 2])].",
    "pypos": "Reply with only a Python list of calls. Pass arguments by position, in the order the function "
    "declares them, e.g. [func_a(1, 'b'), func_b([1, 2])]. Use name=value only for an argument that comes after one you skip.",
    "qwen": "If you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n"
    "<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n"
    "<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n"
    "</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified "
    "format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n"
    "- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural "
    "language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question "
    "like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>",
}
PLAN_HELP = ("\n\nBefore the calls, write exactly one line <plan>...</plan> saying which functions you will call "
             "and why. Replies without a plan are rejected.")


def render(h: Harness, item: Item) -> tuple[list[dict], Names]:
    """Chat messages for this item under this harness."""
    names = names_for(h, item)
    tools = render_tools(h, item, names)
    fmt = FORMAT_HELP[h.call_format] + (PLAN_HELP if h.require_plan else "")
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


def _qwen_value(v: Any) -> str:
    """How Qwen3.5's template writes a parameter value (Jinja `string` of a Python value)."""
    if isinstance(v, (dict, list)):
        return json.dumps(v, ensure_ascii=False)
    if isinstance(v, bool):
        return "True" if v else "False"
    if v is None:
        return "None"
    return str(v)


def format_calls(h: Harness, calls: list[Call], order: dict[str, list[str]] | None = None) -> str:
    """Write calls (already in harness names) in this harness's format.
    `order` (tool -> declared argument order) is needed for positional formats."""
    if h.call_format == "qwen":
        blocks = []
        for c in calls:
            params = "".join(f"<parameter={k}>\n{_qwen_value(v)}\n</parameter>\n" for k, v in c["arguments"].items())
            blocks.append(f"<tool_call>\n<function={c['name']}>\n{params}</function>\n</tool_call>")
        return "\n".join(blocks)
    if h.call_format == "pypos":
        out = []
        for c in calls:
            declared = (order or {}).get(c["name"], [])
            pos, kw, gap = [], [], False
            for k in declared:
                if k in c["arguments"] and not gap:
                    pos.append(repr(c["arguments"][k]))
                elif k in c["arguments"]:
                    kw.append(f"{k}={c['arguments'][k]!r}")
                else:
                    gap = True
            kw += [f"{k}={v!r}" for k, v in c["arguments"].items() if k not in declared]
            out.append(f"{c['name']}({', '.join(pos + kw)})")
        return "[" + ", ".join(out) + "]"
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


class RawText(str):
    """A value written as plain text (Qwen native format). to_bfcl converts it using the schema type,
    the way vLLM's parser for this format does."""


def _qwen_parse_value(v: str) -> Any:
    return RawText(v.strip("\n"))


def _coerce(v: RawText, typ: str) -> Any:
    t = typ.lower()
    if t == "string":
        return str(v)
    if t in ("boolean", "bool"):
        return {"true": True, "false": False}.get(v.strip().lower(), str(v))
    if t in ("integer", "int"):
        try:
            return int(v)
        except ValueError:
            pass
    if t in ("number", "float"):
        try:
            return float(v)
        except ValueError:
            pass
    if v.strip() == "None":
        return None
    if v.strip() in ("True", "False"):
        return v.strip() == "True"
    try:
        return json.loads(v)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(v)
        except (ValueError, SyntaxError):
            return str(v)


_PLAN = re.compile(r"^\s*<plan>.*?</plan>", re.S)


def parse_calls(h: Harness, text: str) -> list[Call] | None:
    """Read calls (in harness names) from model output. None means the format was wrong.
    Positional arguments come back as "__pos0", "__pos1", ...; to_bfcl resolves them."""
    text = _THINK.sub("", text).strip()
    if h.require_plan:
        if not _PLAN.match(text):
            return None
        text = _PLAN.sub("", text, count=1).strip()
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
        if h.call_format == "qwen":
            calls = []
            for name, body in re.findall(r"<tool_call>\s*<function=([^>\n]+)>(.*?)</function>\s*</tool_call>", text, re.S):
                args = {k: _qwen_parse_value(v) for k, v in re.findall(r"<parameter=([^>\n]+)>(.*?)</parameter>", body, re.S)}
                calls.append({"name": name.strip(), "arguments": args})
            return calls or None
        if h.call_format == "pypos":
            tree = ast.parse(text[text.index("[") : text.rindex("]") + 1], mode="eval").body
            if not isinstance(tree, ast.List):
                return None
            calls = []
            for c in tree.elts:
                if not isinstance(c, ast.Call):
                    return None
                args = {f"__pos{i}": ast.literal_eval(a) for i, a in enumerate(c.args)}
                args.update({kw.arg: ast.literal_eval(kw.value) for kw in c.keywords})
                calls.append({"name": ast.unparse(c.func), "arguments": args})
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
        declared = list(names.args.get(tool, {}))  # BFCL arg names in declared order
        args: dict[str, Any] = {}
        for k, v in c["arguments"].items():
            if k.startswith("__pos"):
                i = int(k[5:])
                key = declared[i] if i < len(declared) else f"<unknown:{k}>"
            else:
                key = names.arg_back(tool, k) or f"<unknown:{k}>"
            if isinstance(v, RawText):
                v = _coerce(v, names.types.get(tool, {}).get(key, ""))
            if key in args:  # same argument given twice (by position and by name)
                key = f"<dup:{key}>"
            args[key] = v
        out.append({"name": tool, "arguments": args})
    return out


def target(h: Harness, item: Item) -> str:
    """The correct assistant reply under this harness (used for SFT and round-trip tests)."""
    names = names_for(h, item)
    calls = to_harness(gold_calls(item), names)
    order = {names.tools[t]: list(a.values()) for t, a in names.args.items()}
    reply = format_calls(h, calls, order)
    if h.require_plan:
        tools = ", ".join(c["name"] for c in calls)
        reply = f"<plan>Call {tools}.</plan>\n" + reply
    return reply
