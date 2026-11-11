"""The agent's tool. Talks to the deployed Modal app "geodesic-lab".

  lab status                    budget used so far
  lab run analysis.py           run your analysis code on the GPU (model + ctx loaded); prints come back
  lab patch edit.py NAME        run edit(model, ctx) and save the weight change as patch NAME
  lab try NAME                  score patch NAME on all train items under v1 and v4_format (1 try)
  lab submit NAME [note.md]     pick your final patch (only one; you can resubmit until you stop)

Session name comes from $LAB_SESSION or the file .lab_session in the current folder.
"""

import json
import os
import sys
import time
from pathlib import Path

import modal

APP = "geodesic-lab"


def session() -> str:
    s = os.environ.get("LAB_SESSION") or (Path(".lab_session").read_text().strip() if Path(".lab_session").exists() else "")
    if not s:
        sys.exit("no session: set LAB_SESSION or write .lab_session")
    return s


def show(res: dict) -> None:
    if res.get("stdout"):
        print(res["stdout"].rstrip())
    if res.get("error"):
        print("ERROR:\n" + res["error"].rstrip())
    rest = {k: v for k, v in res.items() if k not in ("stdout", "error")}
    if rest:
        print(json.dumps(rest, indent=1))


def main(argv: list[str]) -> int:
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    s, cmd = session(), argv[0]
    t0 = time.time()
    lab = modal.Cls.from_name(APP, "Lab")()
    if cmd == "status":
        vol = modal.Volume.from_name("geodesic-runs")
        try:
            led = json.loads(b"".join(vol.read_file(f"agent/{s}/ledger.json")))
        except Exception:
            led = {"tries": 0, "runs": 0, "usd": 0}
        print(json.dumps({k: led[k] for k in ("tries", "runs", "usd")} | {"limits": {"tries": 25, "runs": 120, "usd": 5}}))
        return 0
    if cmd == "run":
        show(lab.run.remote(s, Path(argv[1]).read_text()))
    elif cmd == "patch":
        show(lab.make_patch.remote(s, argv[2], Path(argv[1]).read_text()))
    elif cmd == "try":
        res = modal.Function.from_name(APP, "try_patch").remote(s, argv[1])
        Path("tries").mkdir(exist_ok=True)
        Path(f"tries/{argv[1]}.json").write_text(json.dumps(res, indent=1))
        print(json.dumps({k: v for k, v in res.items() if k != "fail_examples"}, indent=1))
        print(f"({len(res['fail_examples'])} failure examples in tries/{argv[1]}.json)")
    elif cmd == "submit":
        note = Path(argv[2]).read_text() if len(argv) > 2 else ""
        Path("SUBMITTED.json").write_text(json.dumps({"session": s, "patch": argv[1], "note": note,
                                                      "time": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=1))
        print("submitted", argv[1])
    else:
        print(__doc__)
        return 2
    print(f"[{time.time() - t0:.0f}s]", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
