# DAWE usage telemetry

DAWE sends one anonymous `dawe_command_finished` event to PostHog for each milestone command:
`skill install`, `workbench init`, `workbench freeze`, `patch create`, `patch from-dense`,
`patch apply`, `patch restore`, and `patch verify`. It is product-usage counting, not
execution tracing or error logging.

**Please keep it on.** It is the only way the maintainers learn whether DAWE is used and where
runs fail. If you do want out, disable everything with `dawe telemetry off` or `DAWE_TELEMETRY_DISABLED=1`. Turn off only the
IP-derived location and install profile with `dawe telemetry location off`. Telemetry is also off
when `DO_NOT_TRACK=1`, `CI` is set, or under pytest. `dawe telemetry status` shows the effective
setting. A malformed config file disables capture. Because DAWE is usually driven by an agent, the
first captured command prints the notice to stderr whether or not a terminal is attached; the agent
should relay it to you.

## Event contract (`schema_version` 1)

The event name, `product=dawe`, and `$lib=dawe` keep DAWE separate from other tools' events
(for example mint's `command_finished`) in a shared PostHog project. Person profile fields
(`dawe_version`, `dawe_os`, `dawe_first_version`) are set only while location/profile is on.
Every event carries `dawe_version`, `python_version`, OS family, `command`,
`duration_ms`, `exit` (`ok`, `usage`, `error`, `interrupted`), `tty` (was stderr a terminal), `agent` (`claude_code`, `codex`, `cursor`,
`human`, `unknown`: guessed from environment variable names only), and `schema_version`. A non-zero exit adds `error_type`
(`usage`, `operation`, `interrupted`, `internal`), a closed category that never contains exception text.

| Command | Additional properties |
| --- | --- |
| `skill install` | none |
| `workbench init` | `budget_set`, `max_tries` (bucket) |
| `workbench freeze` | `dev_examples`, `eval_examples`, `repair`, `preserve` (buckets) |
| `patch create`, `patch from-dense` | `tensors` (bucket), `changed_weights`, `file_bytes` (order of magnitude) |
| `patch apply`, `patch restore` | `tensors` (bucket), `changed_weights` (magnitude), `checked` (`full`/`fast`), `link` |
| `patch verify` | `expect`, `checked`, `verified` |

Counts use buckets `0`, `1`, `2-9`, `10-99`, `100-999`, `1000+`. Large quantities use orders of
magnitude (`1e6` means 1,000,000 to 9,999,999). A missing property means unknown or not applicable.
`patch verify` returning "not that checkpoint" is a normal outcome: it exits 1 with `verified=false`.

## Privacy

Property names and string values are allow-listed in `dawe/telemetry.py`; anything else is dropped.
Capture excludes paths, model and checkpoint names, patch names and descriptions, tensor names,
fingerprints, data, prompts, scores, and exception text or stack traces. Data-dependent
commands (`inspect`, `fetch`, `report`, `score`, ...) are never captured. Nothing is sent from the
GPU lab jobs (`gpu/`, `scripts/lab.py`).

A random UUID is created on the first captured event and stored in `~/.config/dawe/telemetry.json`
(or `$XDG_CONFIG_HOME/dawe/`). It is separate from other tools' IDs, identifies a config directory
and not a person, and is created with the config file, even if you turn telemetry off first (nothing is sent while off). By default PostHog keeps a
person profile for that ID and derives approximate location from the connection IP;
`dawe telemetry location off` sends `$process_person_profile: false` and `$geoip_disable: true`
instead. The service still receives the connection IP either way. Treat IP and location as personal
data and check your consent obligations (for example GDPR) before relying on it.

## Delivery limits

One synchronous attempt with a 2-second socket timeout, no retry, queue, or background process.
Offline events are discarded, so counts undercount. Argparse errors, `--help`, and a killed process
emit nothing. The write key embedded in `dawe/telemetry.py` is a public ingest-only `phc_` key.
Override it with `DAWE_POSTHOG_API_KEY` (or `POSTHOG_PROJECT_TOKEN`), and the host with
`DAWE_POSTHOG_HOST`; an empty key turns capture off.

## Checking that it works

`DAWE_TELEMETRY_DEBUG=1` prints one stderr line per attempt, such as
`dawe telemetry: sent dawe_command_finished command='patch apply' distinct_id=... -> HTTP 200`,
or why nothing was sent. In PostHog, open **Activity** (live events) and filter on the event
`dawe_command_finished`, or on `product = dawe`; search by the `distinct_id` printed above (also
stored in `~/.config/dawe/telemetry.json`). For trends, create an insight on that event and break
down by `command`, `exit`, `agent`, or `dawe_version`.

## PostHog references

[Capture API](https://posthog.com/docs/api/capture), anonymous-event controls, and
[product analytics best practices](https://posthog.com/docs/product-analytics/best-practices)
(static event names, outcome properties, schema versioning). The stdlib sender avoids a new
dependency and a shutdown lifecycle for one event per invocation.
