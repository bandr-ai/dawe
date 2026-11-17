"""Anonymous usage telemetry for DAWE milestone commands. Stdlib only.

One ``command_finished`` event per milestone command, sent to PostHog's capture API.
The embedded value is a PostHog project write key (``phc_``); a personal key (``phx_``)
is rejected. ``DAWE_POSTHOG_API_KEY`` or ``POSTHOG_PROJECT_TOKEN`` replaces it, and an
empty value turns capture off.

Never includes paths, model or checkpoint names, patch names, data, prompts, or error
text. Property names and string values are allow-listed below; anything else is dropped.
"""

from __future__ import annotations

import json
import os
import platform
import sys
import tempfile
import time
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Callable

from . import __version__

# Commands that produce an event. Read/dev commands (inspect, fetch, render, score, ...) do not.
MILESTONES = frozenset(
    {
        "skill install",
        "workbench init",
        "workbench freeze",
        "patch create",
        "patch from-dense",
        "patch apply",
        "patch restore",
        "patch verify",
    }
)

SCHEMA_VERSION = 1
# Bump when NOTICE changes so existing installs see the new text once.
NOTICE_VERSION = 1

NOTICE = (
    "dawe sends anonymous usage for skill install, workbench init/freeze, and patch "
    "create/from-dense/apply/restore/verify: version, OS, command, duration, outcome, flags, "
    "and bucketed counts. No paths, model or patch names, data, or prompts. By default the "
    "analytics service also derives your approximate location from your IP and keeps a "
    "profile for your install ID.\n"
    "Please keep it on: it is the only way we learn whether DAWE is being used and which steps "
    "work or fail, and it decides what we improve next.\n"
    "You can turn off location with `dawe telemetry location off`, or everything with "
    "`dawe telemetry off` or DAWE_TELEMETRY_DISABLED=1. It is also off when CI or DO_NOT_TRACK=1 "
    "is set. `dawe telemetry status` shows the current state."
)

# Event name and library tag are DAWE-specific so they never mix with other tools' events
# (for example mint's plain `command_finished`) in a shared PostHog project.
EVENT = "dawe_command_finished"
LIB = "dawe"

_DEFAULT_HOST = "https://us.i.posthog.com"
# Allow time for a fresh HTTPS connection; telemetry stays best-effort.
_TIMEOUT = 2.0
# Public project write key (safe to embed; it can only ingest events).
_EMBEDDED_KEY = "phc_CvhAGavirL5hbBfgmGzacMUV7v8w4gHpi749P7KJ4XnG"
_FALSE_CI = frozenset({"", "0", "false", "no"})

_BUCKETS = frozenset({"0", "1", "2-9", "10-99", "100-999", "1000+"})
_MAGNITUDES = frozenset({"0", *(f"1e{k}" for k in range(13)), "1e13+"})
_ENUM_FIELDS = {
    "checked": frozenset({"full", "fast"}),
    "link": frozenset({"copy", "hardlink", "symlink"}),
    "expect": frozenset({"source", "result"}),
    "exit": frozenset({"ok", "usage", "error", "interrupted"}),
    "agent": frozenset({"claude_code", "codex", "cursor", "human", "unknown"}),
    "error_type": frozenset({"usage", "operation", "interrupted", "internal"}),
}
_BOOL_FIELDS = frozenset({"budget_set", "verified", "tty"})
_COUNT_FIELDS = frozenset(
    {"max_tries", "dev_examples", "eval_examples", "repair", "preserve", "tensors"}
)
_MAGNITUDE_FIELDS = frozenset({"changed_weights", "file_bytes"})
_ALLOWED = {
    "skill install": (),
    "workbench init": ("budget_set", "max_tries"),
    "workbench freeze": ("dev_examples", "eval_examples", "repair", "preserve"),
    "patch create": ("tensors", "changed_weights", "file_bytes"),
    "patch from-dense": ("tensors", "changed_weights", "file_bytes"),
    "patch apply": ("tensors", "changed_weights", "checked", "link"),
    "patch restore": ("tensors", "changed_weights", "checked", "link"),
    "patch verify": ("expect", "checked", "verified"),
}


# --- value helpers (used by callers so only coarse values ever reach `record`) ---------------


def bucket(n: int) -> str:
    n = 0 if n < 0 else int(n)
    if n <= 1:
        return str(n)
    if n < 10:
        return "2-9"
    if n < 100:
        return "10-99"
    if n < 1000:
        return "100-999"
    return "1000+"


def magnitude(n: int) -> str:
    """Order of magnitude for large quantities (weights changed, file bytes)."""
    n = 0 if n < 0 else int(n)
    if n == 0:
        return "0"
    digits = len(str(n))
    return f"1e{digits - 1}" if digits <= 13 else "1e13+"


def os_family() -> str:
    plat = sys.platform
    if plat.startswith("linux"):
        return "linux"
    if plat == "darwin":
        return "darwin"
    if plat in ("win32", "cygwin"):
        return "windows"
    return "other"


def detect_agent() -> str:
    """Closed-set guess at who is driving the CLI. Reads only the names of env vars."""
    env = os.environ
    if env.get("CLAUDECODE"):
        return "claude_code"
    if any(k.startswith("CODEX_") for k in env):
        return "codex"
    if env.get("CURSOR_AGENT") or env.get("CURSOR_TRACE_ID"):
        return "cursor"
    return "human" if sys.stderr.isatty() else "unknown"


# --- per-command properties --------------------------------------------------------------------

_pending: dict[str, Any] = {}


def record(**props: Any) -> None:
    """Attach outcome properties to the milestone command currently running."""
    _pending.update(props)


# --- config ------------------------------------------------------------------------------------


def config_path() -> Path | None:
    try:
        base = os.environ.get("XDG_CONFIG_HOME")
        if base is None or not base.strip():
            base = str(Path.home() / ".config")
        return Path(base) / "dawe" / "telemetry.json"
    except (RuntimeError, OSError):
        return None


def _read(path: Path) -> tuple[dict, str]:
    """Return ``(state, kind)``; kind is ``missing``, ``ok``, or ``unreadable``."""
    state = {"anonymous_id": None, "enabled": True, "location": True, "notice_version": None}
    if not path.exists():
        return state, "missing"
    try:
        data = json.loads(path.read_text())
    except (OSError, UnicodeError, json.JSONDecodeError):
        return state, "unreadable"
    if not isinstance(data, dict):
        return state, "unreadable"
    for key in ("enabled", "location"):
        if key in data and not isinstance(data[key], bool):
            return state, "unreadable"
        state[key] = data.get(key, True)
    anon = data.get("anonymous_id")
    if isinstance(anon, str) and anon.strip():
        state["anonymous_id"] = anon.strip()
    if data.get("notice_version") == NOTICE_VERSION:
        state["notice_version"] = NOTICE_VERSION
    return state, "ok"


def _save(path: Path, state: dict) -> None:
    if not state.get("anonymous_id"):
        state["anonymous_id"] = str(uuid.uuid4())
    body = {k: v for k, v in state.items() if v is not None}
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".telemetry-")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(json.dumps(body, indent=2, sort_keys=True) + "\n")
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def project_key() -> str:
    """Env override, then the embedded project write key. Empty means off."""
    for var in ("DAWE_POSTHOG_API_KEY", "POSTHOG_PROJECT_TOKEN"):
        if var in os.environ:
            return os.environ[var].strip()
    return _EMBEDDED_KEY.strip()


def _under_pytest() -> bool:
    return bool(os.environ.get("PYTEST_CURRENT_TEST"))


def _early_block() -> str | None:
    """Gate steps that must not touch the config file. ``None`` means keep going."""
    key = project_key()
    if not key:
        return "no API key"
    if not key.startswith("phc_"):
        return "API key is not a project key"
    ci = os.environ.get("CI")
    if ci is not None and ci.strip().lower() not in _FALSE_CI:
        return "CI"
    if os.environ.get("DO_NOT_TRACK") == "1":
        return "DO_NOT_TRACK"
    if os.environ.get("DAWE_TELEMETRY_DISABLED") == "1":
        return "DAWE_TELEMETRY_DISABLED"
    if _under_pytest():
        return "pytest"
    return None


def gate() -> tuple[bool, str]:
    early = _early_block()
    if early:
        return False, early
    path = config_path()
    if path is None:
        return False, "config unreadable"
    state, kind = _read(path)
    if kind == "unreadable":
        return False, "config unreadable"
    if state["enabled"] is False:
        return False, "persisted off"
    return True, "on"


def status() -> dict[str, Any]:
    effective, reason = gate()
    path = config_path()
    persisted, location, notice = "default", True, False
    if path is not None:
        state, kind = _read(path)
        if kind == "ok":
            persisted = "on" if state["enabled"] else "off"
            location = state["location"]
            notice = state["notice_version"] == NOTICE_VERSION
    return {
        "effective": effective,
        "reason": reason,
        "persisted": persisted,
        "location": location,
        "notice_shown": notice,
        "config": "" if path is None else str(path),
    }


def _update(**changes: bool) -> None:
    """Persist a setting. Creates the install ID if there is none; nothing is sent while off."""
    path = config_path()
    if path is None:
        return
    state, kind = _read(path)
    if kind == "unreadable":
        state = {"anonymous_id": None, "enabled": True, "location": True, "notice_version": None}
    state.update(changes)
    _save(path, state)


def set_enabled(enabled: bool) -> None:
    _update(enabled=enabled)


def set_location(location: bool) -> None:
    _update(location=location)


# --- delivery ----------------------------------------------------------------------------------


def origin() -> str:
    raw = os.environ.get("DAWE_POSTHOG_HOST", "").strip() or _DEFAULT_HOST
    return raw.rstrip("/")


def post(url: str, body: bytes, timeout: float) -> int:
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", "User-Agent": f"dawe/{__version__}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        resp.read()
        return resp.status


def _properties(
    command: str, exit_class: str, duration_ms: int, extra: dict[str, Any], location: bool
) -> dict[str, Any]:
    props: dict[str, Any] = {
        "product": "dawe",
        "$lib": LIB,
        "$lib_version": __version__,
        "dawe_version": __version__,
        "python_version": platform.python_version(),
        "os": os_family(),
        "command": command,
        "duration_ms": duration_ms,
        "exit": exit_class,
        "schema_version": SCHEMA_VERSION,
    }
    if location:
        # Default: PostHog keeps a person profile for the install ID and enriches it with
        # GeoIP derived from the connection IP. Profile fields answer "who is still on which version".
        props["$process_person_profile"] = True
        props["$set"] = {"dawe_version": __version__, "dawe_os": os_family()}
        props["$set_once"] = {"dawe_first_version": __version__, "dawe_first_os": os_family()}
    else:
        props["$process_person_profile"] = False
        props["$geoip_disable"] = True
    for key in (*_ALLOWED.get(command, ()), "tty", "agent", "error_type"):
        value = extra.get(key)
        if key in _BOOL_FIELDS and isinstance(value, bool):
            props[key] = value
        elif isinstance(value, str):
            if key in _ENUM_FIELDS and value in _ENUM_FIELDS[key]:
                props[key] = value
            elif key in _COUNT_FIELDS and value in _BUCKETS:
                props[key] = value
            elif key in _MAGNITUDE_FIELDS and value in _MAGNITUDES:
                props[key] = value
    return props


def _prepare() -> tuple[str, bool] | None:
    """Return ``(install id, location on)`` when the send gate is open, else None.

    The notice goes to stderr on the first captured command, terminal or not: DAWE is
    mostly driven by an agent, whose tool output carries stderr back to the user.
    """
    ok, _reason = gate()
    path = config_path()
    if not ok or path is None:
        return None
    state, _kind = _read(path)
    dirty = False
    if not state["anonymous_id"]:
        state["anonymous_id"] = str(uuid.uuid4())
        dirty = True
    if state["notice_version"] != NOTICE_VERSION:
        print(NOTICE, file=sys.stderr, flush=True)
        state["notice_version"] = NOTICE_VERSION
        dirty = True
    if dirty:
        _save(path, state)
    return state["anonymous_id"], bool(state["location"])


def _debug(msg: str) -> None:
    """``DAWE_TELEMETRY_DEBUG=1`` prints delivery results to stderr (never event contents)."""
    if os.environ.get("DAWE_TELEMETRY_DEBUG") == "1":
        print(f"dawe telemetry: {msg}", file=sys.stderr, flush=True)


def emit(
    command: str, exit_class: str, duration_ms: int, extra: dict[str, Any] | None = None
) -> None:
    """Best-effort: one synchronous attempt, no retry, never raises."""
    try:
        if command not in MILESTONES:
            return
        prepared = _prepare()
        if not prepared:
            _debug(f"not sent: capture is off ({gate()[1]})")
            return
        anon, location = prepared
        payload = {
            "api_key": project_key(),
            "event": EVENT,
            "distinct_id": anon,
            "properties": _properties(
                command, exit_class, max(0, int(duration_ms)), extra or {}, location
            ),
        }
        status = post(f"{origin()}/i/v0/e/", json.dumps(payload).encode(), _TIMEOUT)
        _debug(f"sent {EVENT} command={command!r} distinct_id={anon} -> HTTP {status}")
    except Exception as exc:
        _debug(f"not sent: {type(exc).__name__}: {exc}")


def command_name(argv: list[str]) -> str | None:
    """``dawe patch apply ...`` -> ``patch apply`` if it is a milestone, else None."""
    if len(argv) >= 2:
        name = f"{argv[0]} {argv[1]}"
        if name in MILESTONES:
            return name
    return None


def _exit_class(code: int) -> str:
    return "ok" if code == 0 else "usage" if code == 2 else "error"


def run(command: str | None, fn: Callable[[list[str]], int], argv: list[str]) -> int:
    """Run ``fn(argv)`` and emit one event if ``command`` is a milestone.

    Argparse errors and ``--help`` raise SystemExit and are not captured.
    """
    if command is None:
        return fn(argv)
    _pending.clear()
    started = time.monotonic()
    exit_class: str | None = None
    error_type: str | None = None
    try:
        code = fn(argv)
        exit_class = _exit_class(code)
        if code != 0:
            error_type = "usage" if code == 2 else "operation"
        return code
    except KeyboardInterrupt:
        exit_class = error_type = "interrupted"
        raise
    except SystemExit:
        raise
    except Exception:
        exit_class, error_type = "error", "internal"
        raise
    finally:
        if exit_class:
            extra = dict(_pending)
            extra["tty"] = sys.stderr.isatty()
            extra["agent"] = detect_agent()
            if error_type:
                extra["error_type"] = error_type
            emit(command, exit_class, int((time.monotonic() - started) * 1000), extra)
        _pending.clear()


# --- `dawe telemetry` --------------------------------------------------------------------------


def main(argv: list[str]) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="dawe telemetry", description="anonymous usage telemetry")
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("status", help="show whether telemetry is on")
    sub.add_parser("on", help="turn anonymous usage telemetry on")
    sub.add_parser("off", help="turn anonymous usage telemetry off")
    loc = sub.add_parser("location", help="IP-derived location and install profile (default: on)")
    loc.add_argument("state", choices=("on", "off"))
    a = ap.parse_args(argv)
    try:
        if a.command in ("on", "off"):
            set_enabled(a.command == "on")
        elif a.command == "location":
            set_location(a.state == "on")
    except OSError as exc:
        print(f"could not save telemetry settings: {exc}", file=sys.stderr)
        return 1
    info = status()
    print(f"effective: {'on' if info['effective'] else 'off'} ({info['reason']})")
    print(f"persisted: {info['persisted']}")
    print(f"location (IP-derived): {'on' if info['location'] else 'off'}")
    print(f"notice shown: {'yes' if info['notice_shown'] else 'no'}")
    print(f"config: {info['config']}")
    return 0
