"""Telemetry: gating, payload contents, privacy, and that it can never break a command."""

from __future__ import annotations

import json

import pytest

from dawe import telemetry
from dawe.cli import main

from test_patch import edit_rows, make_checkpoint  # noqa: F401  (fixtures below reuse these)


@pytest.fixture
def sent(tmp_path, monkeypatch):
    """Open the gate (pytest/CI guards off, private config dir) and capture what would be posted."""
    # pytest re-sets PYTEST_CURRENT_TEST for each phase, so open the guard itself.
    monkeypatch.setattr(telemetry, "_under_pytest", lambda: False)
    for var in ("CI", "DO_NOT_TRACK", "DAWE_TELEMETRY_DISABLED",
                "DAWE_POSTHOG_API_KEY", "POSTHOG_PROJECT_TOKEN", "DAWE_POSTHOG_HOST"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    events: list[dict] = []
    monkeypatch.setattr(telemetry, "post", lambda url, body, timeout: events.append({"url": url, **json.loads(body)}))
    return events


def init_args(tmp_path, name="run", *extra):
    src = tmp_path / "private-customer-data"
    src.mkdir(exist_ok=True)
    return ["workbench", "init", "--source", str(src), "--model", "acme/secret-model", "--out", str(tmp_path / name), *extra]


def test_off_under_pytest_by_default(tmp_path, monkeypatch):
    # CI systems set CI, which is checked before pytest; clear the other guards so this isolates pytest.
    for var in ("CI", "DO_NOT_TRACK", "DAWE_TELEMETRY_DISABLED", "DAWE_POSTHOG_API_KEY", "POSTHOG_PROJECT_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    assert telemetry.gate() == (False, "pytest")


@pytest.mark.parametrize("var,value,reason", [
    ("CI", "true", "CI"),
    ("DO_NOT_TRACK", "1", "DO_NOT_TRACK"),
    ("DAWE_TELEMETRY_DISABLED", "1", "DAWE_TELEMETRY_DISABLED"),
    ("DAWE_POSTHOG_API_KEY", "", "no API key"),
    ("DAWE_POSTHOG_API_KEY", "phx_personal", "API key is not a project key"),
])
def test_gate_closes(sent, monkeypatch, var, value, reason):
    monkeypatch.setenv(var, value)
    assert telemetry.gate() == (False, reason)


def test_false_ci_values_do_not_close_gate(sent, monkeypatch):
    monkeypatch.setenv("CI", "false")
    assert telemetry.gate() == (True, "on")


def test_unreadable_config_disables_capture(sent, tmp_path, capsys):
    cfg = telemetry.config_path()
    cfg.parent.mkdir(parents=True)
    cfg.write_text("{not json")
    assert main(init_args(tmp_path)) == 0
    assert sent == [] and "dawe sends anonymous usage" not in capsys.readouterr().err


def test_init_event_has_only_allowlisted_coarse_properties(sent, tmp_path, capsys):
    assert main(init_args(tmp_path, "run", "--budget-usd", "50")) == 0
    [event] = sent
    assert event["event"] == "dawe_command_finished"  # never the bare name other tools use
    assert event["url"] == "https://us.i.posthog.com/i/v0/e/"
    assert event["api_key"].startswith("phc_")
    props = event["properties"]
    assert props["command"] == "workbench init"
    assert props["exit"] == "ok" and props["product"] == "dawe" and props["schema_version"] == 1
    assert props["budget_set"] is True and props["max_tries"] == "2-9"
    assert props["$process_person_profile"] is True and "$geoip_disable" not in props
    assert props["$lib"] == "dawe" and props["$set"]["dawe_version"] == props["dawe_version"]
    assert props["$set_once"]["dawe_first_version"] == props["dawe_version"]
    assert props["agent"] in {"claude_code", "codex", "cursor", "human", "unknown"}
    wire = json.dumps(event)
    for private in ("private-customer-data", "secret-model", str(tmp_path)):
        assert private not in wire


def test_notice_printed_once_and_install_id_stable(sent, tmp_path, capsys):
    main(init_args(tmp_path, "a"))
    first = capsys.readouterr().err
    main(init_args(tmp_path, "b"))
    second = capsys.readouterr().err
    assert "dawe telemetry off" in first and "dawe telemetry off" not in second
    assert "Please keep it on" in first
    assert sent[0]["distinct_id"] == sent[1]["distinct_id"]


def test_location_off_disables_profile_and_geoip(sent, tmp_path):
    assert main(["telemetry", "location", "off"]) == 0
    assert main(init_args(tmp_path)) == 0
    props = sent[0]["properties"]
    assert props["$process_person_profile"] is False and props["$geoip_disable"] is True
    assert "$set" not in props and "$set_once" not in props


def test_telemetry_off_sends_nothing_but_still_has_an_install_id(sent, tmp_path, capsys):
    assert main(["telemetry", "off"]) == 0
    assert main(init_args(tmp_path)) == 0
    assert sent == []
    assert json.loads(telemetry.config_path().read_text())["anonymous_id"]
    assert main(["telemetry", "status"]) == 0
    assert "effective: off (persisted off)" in capsys.readouterr().out
    assert main(["telemetry", "on"]) == 0
    assert main(init_args(tmp_path, "again")) == 0
    assert len(sent) == 1


def test_failures_report_error_type_without_text(sent, tmp_path):
    assert main(["workbench", "freeze", str(tmp_path / "missing.jsonl"),
                 "--development-dir", str(tmp_path / "d"), "--evaluation-dir", str(tmp_path / "e")]) == 2
    props = sent[0]["properties"]
    assert (props["exit"], props["error_type"]) == ("usage", "usage")
    assert "missing.jsonl" not in json.dumps(sent[0])


def test_argparse_errors_and_help_are_not_captured(sent):
    with pytest.raises(SystemExit):
        main(["patch", "apply", "--help"])
    with pytest.raises(SystemExit):
        main(["workbench", "init"])
    assert sent == []


def test_non_milestone_commands_are_not_captured(sent, tmp_path):
    assert main(["skill", "path"]) == 0
    assert sent == []


def test_delivery_failure_never_breaks_the_command(sent, tmp_path, monkeypatch):
    def boom(*a):
        raise OSError("offline")

    monkeypatch.setattr(telemetry, "post", boom)
    assert main(init_args(tmp_path)) == 0
    assert (tmp_path / "run" / "run.json").exists()


def test_unexpected_exception_is_reported_then_propagates(sent, monkeypatch):
    def broken(argv):
        raise RuntimeError("secret detail")

    with pytest.raises(RuntimeError):
        telemetry.run("patch apply", broken, [])
    props = sent[0]["properties"]
    assert (props["exit"], props["error_type"]) == ("error", "internal")
    assert "secret detail" not in json.dumps(sent[0])


def test_interrupt_is_reported_then_propagates(sent):
    def interrupted(argv):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        telemetry.run("patch apply", interrupted, [])
    assert sent[0]["properties"]["exit"] == "interrupted"


def test_unknown_names_and_values_are_dropped(sent):
    def command(argv):
        telemetry.record(path="/secret", tensors="12345", changed_weights="lots", checked="/etc/passwd", link="copy")
        return 0

    telemetry.run("patch apply", command, [])
    props = sent[0]["properties"]
    assert props["link"] == "copy"
    for dropped in ("path", "tensors", "changed_weights", "checked"):
        assert dropped not in props


def test_pending_properties_do_not_leak_between_commands(sent):
    telemetry.run("patch verify", lambda argv: telemetry.record(verified=True) or 0, [])
    telemetry.run("patch verify", lambda argv: 0, [])
    assert "verified" in sent[0]["properties"] and "verified" not in sent[1]["properties"]


def test_buckets_and_magnitudes():
    assert [telemetry.bucket(n) for n in (-1, 0, 1, 2, 9, 10, 99, 100, 999, 1000)] == [
        "0", "0", "1", "2-9", "2-9", "10-99", "10-99", "100-999", "100-999", "1000+"]
    assert [telemetry.magnitude(n) for n in (0, 1, 9, 10, 4_700_000, 10**13)] == ["0", "1e0", "1e0", "1e1", "1e6", "1e13+"]


def test_patch_lifecycle_events(sent, tmp_path):
    src = make_checkpoint(tmp_path / "src")
    edited = edit_rows(src, tmp_path / "edited", {3: 1, 7: 2})
    gp = tmp_path / "fix.gpatch"
    assert main(["patch", "create", "--source", str(src), "--edited", str(edited), "-o", str(gp)]) == 0
    assert main(["patch", "apply", str(gp), "--source", str(src), "--out", str(tmp_path / "patched"), "--link", "hardlink"]) == 0
    assert main(["patch", "verify", str(gp), str(src), "--expect", "result"]) == 1
    assert main(["patch", "restore", str(gp), "--patched", str(tmp_path / "patched"), "--out", str(tmp_path / "back"), "--fast"]) == 0
    create, apply_, verify, restore = (e["properties"] for e in sent)
    assert create["command"] == "patch create" and create["tensors"] == "1" and create["changed_weights"] == "1e1"
    assert apply_["checked"] == "full" and apply_["link"] == "hardlink"
    assert (verify["exit"], verify["error_type"], verify["verified"], verify["expect"]) == ("error", "operation", False, "result")
    assert restore["checked"] == "fast"
    wire = json.dumps(sent)
    assert str(tmp_path) not in wire and "fix.gpatch" not in wire


@pytest.mark.parametrize("env,expected", [({"CLAUDECODE": "1"}, "claude_code"), ({"CODEX_HOME": "/x"}, "codex"),
                                          ({"CURSOR_AGENT": "1"}, "cursor")])
def test_agent_detection_is_a_closed_set(sent, monkeypatch, env, expected):
    for k in ("CLAUDECODE", "CURSOR_AGENT", "CURSOR_TRACE_ID"):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    telemetry.run("skill install", lambda argv: 0, [])
    assert sent[0]["properties"]["agent"] == expected


def test_debug_reports_delivery_without_event_contents(sent, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("DAWE_TELEMETRY_DEBUG", "1")
    monkeypatch.setattr(telemetry, "post", lambda *a: 200)
    assert main(init_args(tmp_path)) == 0
    err = capsys.readouterr().err
    assert "HTTP 200" in err and "workbench init" in err and "secret-model" not in err
    monkeypatch.setenv("DAWE_TELEMETRY_DISABLED", "1")
    main(init_args(tmp_path, "two"))
    assert "capture is off (DAWE_TELEMETRY_DISABLED)" in capsys.readouterr().err
