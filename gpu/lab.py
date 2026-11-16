"""DAWE lab: the GPU side of the editing agent's tools.

The agent (Codex, GPT-6 Astra) never trains. It reads maps and weights, runs its own
analysis code, writes an edit, and spends a fixed budget of evaluations on TRAINING items.
Test items are never loaded here. Budgets live on the Modal volume, so the agent can't reset them.

  modal deploy gpu/lab.py                          once, after any change to this file
  modal run gpu/lab.py::setup --session s1         clean map + baseline for a new session

Pieces
  Lab (class, warm SFT model on an H100)
    run(session, code)         agent analysis code; prints come back
    make_patch(session, name, code)   agent's edit(model, ctx) -> a sparse .gpatch on the volume
  try_patch(session, name)     vLLM: score a patch on all train items under v1 and v4 (costs 1 try)

What is and is not enforced (read docs/LAB.md before relying on it):
  - Budgets and the "no optimizer / no generate / no test data" rules are checked by regex and by a ledger
    file on the volume. The agent's code runs in-process with the volume mounted writable, so these are
    guard rails for a cooperative agent, not a sandbox against an adversarial one.
  - Test items are never loaded into the Lab, but the full BFCL files are in the image
    (/root/data/bfcl) and the split is a public hash, so withholding is by instruction, not by isolation.
"""

from __future__ import annotations

import contextlib
import io
import json
import math
import re
import time
import traceback
from pathlib import Path

import modal

from dawe import pins

ROOT = Path(__file__).resolve().parent.parent
PRICE = {"H100": 0.001097, "cpu_core": 0.0000131, "mem_gib": 0.00000222}
CPU, MEM_GIB = 2.0, 16
RATE = PRICE["H100"] + CPU * PRICE["cpu_core"] + MEM_GIB * PRICE["mem_gib"]  # ≈ $4.29/hr

BASE_RUN = "sft_v1"
TARGET, KEEP = "v4_format", "v1"
LIMITS = {"tries": 25, "runs": 120, "usd": 5.0}
WARM_SECS = 60  # keep the analysis GPU warm briefly between agent calls (idle time is billed)

app = modal.App("geodesic-lab")
runs = modal.Volume.from_name("geodesic-runs", create_if_missing=True)
hf_cache = modal.Volume.from_name("geodesic-hf-cache", create_if_missing=True)
VOLS = {"/runs": runs, "/root/.cache/huggingface": hf_cache}


def _with_code(img: modal.Image) -> modal.Image:
    return (
        img.add_local_python_source("dawe")
        .add_local_dir(ROOT / "harnesses", "/root/harnesses")
        .add_local_dir(ROOT / "data" / "bfcl", "/root/data/bfcl")
    )


lab_image = _with_code(
    modal.Image.debian_slim(python_version=pins.PYTHON_IMAGE)
    .uv_pip_install(
        *pins.pip_pins({k: v for k, v in pins.TRAIN_PACKAGES.items() if k != "peft"}),
        "numpy", "hf_transfer",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)
vllm_image = _with_code(
    modal.Image.debian_slim(python_version=pins.PYTHON_IMAGE)
    .uv_pip_install(*pins.pip_pins(pins.VLLM_PACKAGES), "hf_transfer")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "VLLM_USE_FLASHINFER_SAMPLER": "0"})
)
COMMON = dict(gpu="H100", cpu=CPU, memory=MEM_GIB * 1024, volumes=VOLS)


# --- session ledger (on the volume) -------------------------------------------------------


def _sdir(session: str) -> Path:
    if not re.fullmatch(r"[a-z0-9_-]{1,40}", session):
        raise ValueError("bad session name")
    return Path(f"/runs/agent/{session}")


def _ledger(session: str) -> dict:
    p = _sdir(session) / "ledger.json"
    return json.loads(p.read_text()) if p.exists() else {"tries": 0, "runs": 0, "usd": 0.0, "log": []}


def _charge(session: str, kind: str, secs: float, note: str = "") -> dict:
    runs.reload()
    led = _ledger(session)
    if kind in ("tries", "runs"):
        led[kind] += 1
    usd = round(secs * RATE, 4)
    led["usd"] = round(led["usd"] + usd, 4)
    led["log"].append({"t": time.strftime("%H:%M:%S"), "kind": kind, "secs": round(secs), "usd": usd, "note": note})
    _sdir(session).mkdir(parents=True, exist_ok=True)
    (_sdir(session) / "ledger.json").write_text(json.dumps(led, indent=1))
    runs.commit()
    return led


def _config(session: str) -> dict:
    """Which harness this session must fix and which it must keep. Written by `setup`."""
    p = _sdir(session) / "config.json"
    return json.loads(p.read_text()) if p.exists() else {"target": TARGET, "keep": KEEP}


def _check(session: str, kind: str) -> None:
    runs.reload()
    led = _ledger(session)
    if kind in LIMITS and led[kind] >= LIMITS[kind]:
        raise RuntimeError(f"budget used up: {kind} {led[kind]}/{LIMITS[kind]}")
    if led["usd"] >= LIMITS["usd"]:
        raise RuntimeError(f"GPU money budget used up: ${led['usd']:.2f}/${LIMITS['usd']}")


BANNED = [r"torch\.optim", r"\bAdam", r"\bSGD\b", r"\.generate\(", r"lr_scheduler", r"bfcl_test", r"data/eval"]


def _vet(code: str) -> None:
    for pat in BANNED:
        if re.search(pat, code):
            raise ValueError(f"not allowed in agent code: /{pat}/ (no optimizers, no generation, no test data)")


# --- decision points ------------------------------------------------------------------------


def points_for(reply: str, fmt: str) -> list[tuple[int, str]]:
    """Decision points in a gold reply, as (char offset, kind). The decision is the token right after the offset.
    kinds: plan_start (reply must open with <plan>), after_plan (calls must follow the plan),
           continue (another call follows), stop (end of turn)."""
    pts: list[tuple[int, str]] = []
    body, shift = reply, 0
    if reply.startswith("<plan>"):
        end = reply.index("</plan>") + len("</plan>")
        pts += [(0, "plan_start"), (end, "after_plan")]
        rest = reply[end:]
        shift = end + (len(rest) - len(rest.lstrip()))
        body = reply[shift:]
    pts += [(shift + e, "continue" if c else "stop") for e, c in decision_points(body, fmt)]
    return pts


def decision_points(reply: str, fmt: str) -> list[tuple[int, bool]]:
    """Char offsets right after each tool call in a gold reply, and whether another call follows."""
    ends = []
    if fmt == "xml":
        i = 0
        while (j := reply.find("</tool_call>", i)) != -1:
            ends.append(j + len("</tool_call>"))
            i = ends[-1]
    else:
        i = 0
        for c in json.loads(reply)["tool_calls"]:
            cj = json.dumps(c, ensure_ascii=False)
            j = reply.find(cj, i)
            if j == -1:
                return []
            ends.append(j + len(cj))
            i = ends[-1]
    return [(e, k < len(ends) - 1) for k, e in enumerate(ends)]


# --- the analysis / patching GPU --------------------------------------------------------------


class Ctx:
    """What agent code gets as `ctx`. See BRIEF.md in the agent workspace for the full list."""

    def __init__(self, lab: "Lab"):
        self._lab = lab
        self.tok = lab.tok
        self.items = lab.items  # train items only: list of dicts
        self.harnesses = list(lab.harnesses)  # [keep, target] for this session
        self.P = lab.P
        self.n_layers = lab.n_layers

    def orig(self, name: str):
        """The unedited SFT weight (GPU tensor; don't modify)."""
        return self._lab.orig[name]

    def example(self, item_id: str, h: str) -> dict:
        return self._lab.example(item_id, h)

    def forward(self, ex: dict, hidden: bool = False):
        """One forward pass on an example (no grad). Returns logits [T, V] and, optionally, hidden states
        as a tensor [n_layers+1, T, d] (index 0 = embeddings, L+1 = output of layer L)."""
        import torch

        with torch.no_grad():
            out = self._lab.model(input_ids=ex["ids"][None].cuda(), output_hidden_states=hidden)
        logits = out.logits[0]
        if not hidden:
            return logits, None
        return logits, torch.stack([h[0] for h in out.hidden_states])

    def point_scores(self, item_ids: list[str], h: str, top: int = 3) -> list[dict]:
        """At every decision point (right after a tool call in the gold reply): p(gold next token),
        whether greedy picks it, and the top tokens."""
        import torch

        rows = []
        for iid in item_ids:
            ex = self.example(iid, h)
            if not ex["points"]:
                continue
            logits, _ = self.forward(ex)
            lp = torch.log_softmax(logits.float(), -1)
            for d, kind in ex["points"]:
                gold = ex["ids"][d + 1].item()
                tv, ti = lp[d].topk(top)
                rows.append({
                    "id": iid, "kind": kind, "p_gold": round(lp[d, gold].exp().item(), 4),
                    "greedy_ok": int(ti[0].item() == gold), "gold": self.tok.decode([gold]),
                    "top": [(self.tok.decode([t]), round(math.exp(v), 3)) for v, t in zip(tv.tolist(), ti.tolist())],
                })
        return rows

    def maps(self) -> list[str]:
        return sorted(p.name for p in Path(f"/runs/{BASE_RUN}/maps").iterdir())

    def load_map(self, name: str):
        """grad_*.safetensors -> dict of tensors; states_*.pt -> {'continue': [n, layers+1, d], 'stop': ...}."""
        p = Path(f"/runs/{BASE_RUN}/maps") / name
        if p.suffix == ".safetensors":
            from safetensors.torch import load_file

            return load_file(str(p))
        import torch

        return torch.load(p)


def _no_generate(*a, **k):
    raise RuntimeError("generation is not available here; use `lab try` (budgeted)")


@app.cls(image=lab_image, timeout=20 * 60, scaledown_window=WARM_SECS, **COMMON)
class Lab:
    @modal.enter()
    def load(self):
        import torch
        from transformers import AutoModelForImageTextToText, AutoTokenizer

        from dawe import bfcl

        t0 = time.time()
        runs.reload()
        path = f"/runs/{BASE_RUN}/merged"
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForImageTextToText.from_pretrained(path, dtype=torch.bfloat16).cuda().eval()
        self.model.generate = _no_generate
        self.P = {n: p for n, p in self.model.named_parameters() if "language_model" in n}
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.orig = {n: p.detach().clone() for n, p in self.P.items()}
        self.n_layers = self.model.config.text_config.num_hidden_layers
        self.items = [
            {"id": it.id, "category": it.category, "question": it.question, "n_calls": len(it.ground_truth)}
            for it in bfcl.load_all(Path("/root/data/bfcl")) if it.split == "train"
        ]
        self._items = {it.id: it for it in bfcl.load_all(Path("/root/data/bfcl")) if it.split == "train"}
        self._cache: dict = {}
        self.harnesses = [KEEP, TARGET]
        self._unbilled = time.time() - t0  # charged to the first call on this container
        print(f"[lab] loaded {BASE_RUN} in {time.time() - t0:.0f}s; {len(self.items)} train items")

    def example(self, item_id: str, h: str) -> dict:
        import torch

        from dawe.harness import Harness, render, target

        key = (item_id, h)
        if key in self._cache:
            return self._cache[key]
        it = self._items[item_id]  # KeyError for anything not in the train split
        hz = Harness.load(h)
        msgs, _ = render(hz, it)
        reply = target(hz, it)
        prompt = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        enc = self.tok(prompt + reply + "<|im_end|>", add_special_tokens=False, return_offsets_mapping=True)
        offs = enc["offset_mapping"]
        start = next(i for i, (s, e) in enumerate(offs) if e > len(prompt))
        pts = []
        for ce, kind in points_for(reply, hz.call_format):
            d = max(i for i, (s, e) in enumerate(offs) if e <= len(prompt) + ce)
            if d + 1 < len(offs):
                pts.append((d, kind))
        ex = {"id": item_id, "harness": h, "text": prompt + reply + "<|im_end|>", "reply": reply,
              "ids": torch.tensor(enc["input_ids"]), "reply_start": start, "points": pts}
        self._cache[key] = ex
        return ex

    def _take_unbilled(self) -> float:
        s, self._unbilled = self._unbilled, 0.0
        return s

    def _restore(self) -> list[str]:
        """Put back any weight the agent code changed; return the names that differed."""
        import torch

        changed = []
        with torch.no_grad():
            for n, p in self.P.items():
                if not torch.equal(p, self.orig[n]):
                    changed.append(n)
                    p.copy_(self.orig[n])
        return changed

    def _exec(self, code: str, call_edit: bool):
        import torch

        _vet(code)
        ns = {"model": self.model, "ctx": Ctx(self), "torch": torch, "__name__": "__agent__"}
        buf = io.StringIO()
        err = None
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            try:
                exec(compile(code, "<agent>", "exec"), ns)
                if call_edit:
                    if "edit" not in ns:
                        raise ValueError("edit file must define edit(model, ctx)")
                    with torch.no_grad():
                        ns["edit"](self.model, ns["ctx"])
            except Exception:
                err = traceback.format_exc(limit=6)
        for p in self.model.parameters():
            p.requires_grad_(False)
            p.grad = None
        out = buf.getvalue()
        if len(out) > 30000:
            out = out[:15000] + "\n...[output cut]...\n" + out[-15000:]
        return out, err

    @modal.method()
    def run(self, session: str, code: str) -> dict:
        _check(session, "runs")
        cfg = _config(session)
        self.harnesses = [cfg["keep"], cfg["target"]]
        t0 = time.time()
        out, err = self._exec(code, call_edit=False)
        touched = self._restore()
        led = _charge(session, "runs", time.time() - t0 + self._take_unbilled())
        return {"stdout": out, "error": err, "weights_touched_and_restored": touched, "ledger": _brief(led)}

    @modal.method()
    def make_patch(self, session: str, name: str, code: str) -> dict:
        """Run the agent's edit(model, ctx) and save what changed as a sparse, checkpoint-bound .gpatch."""
        import torch

        from dawe import patch as gp

        _check(session, "runs")
        cfg = _config(session)
        self.harnesses = [cfg["keep"], cfg["target"]]
        if not re.fullmatch(r"[a-z0-9_-]{1,40}", name):
            raise ValueError("bad patch name")
        t0 = time.time()
        out, err = self._exec(code, call_edit=True)
        edits, stats = {}, []
        if err is None:
            with torch.no_grad():
                for n, p in self.P.items():
                    o = self.orig[n]
                    if torch.equal(p, o):
                        continue
                    if p.dtype != torch.bfloat16:
                        raise RuntimeError(f"{n}: patches are saved for bf16 tensors, got {p.dtype}")
                    pi, oi = p.reshape(-1).view(torch.int16), o.reshape(-1).view(torch.int16)  # bitwise, so NaN/-0 are exact
                    idx = (pi != oi).nonzero().squeeze(1)
                    edits[n] = (idx.cpu().tolist(), pi[idx].cpu().numpy().tobytes())
                    diff = p.float() - o.float()
                    stats.append({"tensor": n, "changed": len(idx), "of": o.numel(),
                                  "rel_norm": round(diff.norm().item() / (o.float().norm().item() + 1e-12), 6)})
        self._restore()
        res = {"stdout": out, "error": err}
        if err is None:
            if not edits:
                res["error"] = "edit() changed no weights"
            else:
                d = _sdir(session) / "patches"
                d.mkdir(parents=True, exist_ok=True)
                try:
                    patch = gp.create_from_edits(
                        f"/runs/{BASE_RUN}/merged", edits, d / f"{name}{gp.SUFFIX}", name=f"{session}/{name}",
                        description=f"DAWE session {session}: fix {cfg['target']}, keep {cfg['keep']}",
                        lineage=pins.lineage_sft_v1(), ident=_source_identity())
                except gp.PatchError as e:
                    res["error"] = f"could not save patch: {e}"
                else:
                    meta = {"name": name, "changed_weights": patch.changed(), "file_bytes": patch.size_bytes,
                            "patch_id": patch.patch_id, "tensors": stats, "code": code,
                            "created": time.strftime("%Y-%m-%d %H:%M:%S")}
                    (d / f"{name}.json").write_text(json.dumps(meta, indent=1))
                    runs.commit()
                    res["patch"] = {k: v for k, v in meta.items() if k != "code"}
        led = _charge(session, "runs", time.time() - t0 + self._take_unbilled(), f"patch {name}")
        res["ledger"] = _brief(led)
        return res


def _source_identity():
    """Content fingerprint of the SFT checkpoint, hashed once and cached next to it on the volume."""
    from dawe import patch as gp

    root = Path(f"/runs/{BASE_RUN}/merged")
    cache = root.parent / "fingerprint.json"
    stamp = {f.name: [f.stat().st_size, f.stat().st_mtime_ns] for f in sorted(root.glob("*.safetensors"))}
    if cache.exists():
        c = json.loads(cache.read_text())
        if c["stamp"] == stamp:
            return gp.Identity(c["layout_sha256"], c["fingerprint"], c["n_tensors"], c["config"], c["tokenizer"], c["hashes"])
    ident = gp.identify(root)
    cache.write_text(json.dumps({"stamp": stamp, "layout_sha256": ident.layout_sha256, "fingerprint": ident.fingerprint,
                                 "n_tensors": ident.n_tensors, "config": ident.config, "tokenizer": ident.tokenizer,
                                 "hashes": ident.hashes}))
    runs.commit()
    return ident


def _brief(led: dict) -> dict:
    return {k: f"{led[k]}/{LIMITS[k]}" for k in ("tries", "runs")} | {"gpu_usd": f"{led['usd']:.2f}/{LIMITS['usd']}"}


# --- scoring a patch with vLLM -------------------------------------------------------------------


def _patched_dir(session: str | None, name: str | None) -> str:
    """The SFT model with a patch applied, as a model folder vLLM can load (unchanged files are symlinks).

    Uses the same apply code as the public `dawe patch apply`, in --fast mode: the lab made the patch from
    this very checkpoint, so re-hashing 8 GB on every try would only cost GPU minutes.
    Old sessions saved whole tensors as <name>.safetensors; those are converted to the sparse format on first use.
    """
    import shutil

    from dawe import patch as gp

    src = Path(f"/runs/{BASE_RUN}/merged")
    if not name:
        return str(src)
    pdir = _sdir(session) / "patches"
    gpatch = pdir / f"{name}{gp.SUFFIX}"
    if not gpatch.exists():
        gp.create_from_dense(src, pdir / f"{name}.safetensors", gpatch, name=f"{session}/{name}",
                             lineage=pins.lineage_sft_v1(), ident=_source_identity())
        runs.commit()
    dst = Path(f"/tmp/patched_{session}_{name}")
    shutil.rmtree(dst, ignore_errors=True)
    gp.apply_patch(gpatch, src, dst, full=False, link="symlink")
    return str(dst)


def _gen_and_score(model_dir: str, items, harnesses: list[str], max_tokens: int = 512):
    from vllm import LLM, SamplingParams

    from dawe.datasets import eval_rows, score
    from dawe.harness import Harness

    rows = eval_rows(items, [Harness.load(h) for h in harnesses])
    llm = LLM(model=model_dir, max_model_len=8192, gpu_memory_utilization=0.85, enforce_eager=True,
              limit_mm_per_prompt={"image": 0, "video": 0}, seed=0)
    outs = llm.chat([r["messages"] for r in rows], SamplingParams(temperature=0.0, max_tokens=max_tokens),
                    chat_template_kwargs={"enable_thinking": False})
    preds = [{"id": r["id"], "harness": r["harness"], "text": o.outputs[0].text} for r, o in zip(rows, outs)]
    return preds, score(preds, items)


@app.function(image=vllm_image, timeout=20 * 60, scaledown_window=2, **COMMON)
def try_patch(session: str, name: str | None, free: bool = False) -> dict:
    """Score a patch on ALL train items under v1 and v4. Costs one try unless free (baseline only)."""
    from dawe import bfcl
    from dawe.harness import Harness, parse_calls, render, to_bfcl

    if not free:
        _check(session, "tries")
    t0 = time.time()
    runs.reload()
    items = [it for it in bfcl.load_all(Path("/root/data/bfcl")) if it.split == "train"]
    cfg = _config(session)
    keep, tgt = cfg["keep"], cfg["target"]
    preds, sc = _gen_and_score(_patched_dir(session, name), items, [keep, tgt])
    # a few failures under the target harness, so the agent sees what goes wrong
    by_id = {it.id: it for it in items}
    fails = []
    hz = Harness.load(tgt)
    for p in preds:
        if p["harness"] != tgt:
            continue
        it = by_id[p["id"]]
        _, names = render(hz, it)
        calls = parse_calls(hz, p["text"])
        if calls is None or not bfcl.check(to_bfcl(calls, names), it):
            fails.append({"id": it.id, "category": it.category, "n_gold_calls": len(it.ground_truth),
                          "got_calls": None if calls is None else len(calls), "text": p["text"][:600]})
    res = {"name": name or "baseline", "scores": sc, "n_fail_target": len(fails), "fail_examples": fails[:12]}
    d = _sdir(session) / "tries"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{name or 'baseline'}.json").write_text(json.dumps(res | {"preds": preds}))
    runs.commit()
    led = _charge(session, "free" if free else "tries", time.time() - t0, f"try {name or 'baseline'}")
    res["ledger"] = _brief(led)
    return res


@app.function(image=vllm_image, timeout=20 * 60, scaledown_window=2, **COMMON)
def final_eval(session: str, name: str | None, rows: list[dict]) -> dict:
    """Run by us, not the agent: the hidden test prompts with the patch applied. Returns raw preds."""
    from vllm import LLM, SamplingParams

    t0 = time.time()
    runs.reload()
    llm = LLM(model=_patched_dir(session, name), max_model_len=8192, gpu_memory_utilization=0.85, enforce_eager=True,
              limit_mm_per_prompt={"image": 0, "video": 0}, seed=0)
    outs = llm.chat([r["messages"] for r in rows], SamplingParams(temperature=0.0, max_tokens=512),
                    chat_template_kwargs={"enable_thinking": False})
    preds = [{"id": r["id"], "harness": r["harness"], "text": o.outputs[0].text} for r, o in zip(rows, outs)]
    secs = time.time() - t0
    # also keep them on the volume, so a dropped local connection can't lose the results
    d = _sdir(session) / "final"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{name}.jsonl").write_text("".join(json.dumps(p) + "\n" for p in preds))
    runs.commit()
    return {"preds": preds, "cost": {"job": f"final {session}/{name}", "seconds": round(secs), "usd_est": round(secs * RATE, 3)}}


# --- clean map: gradient of the 'continue' decision only ------------------------------------------


@app.function(image=lab_image, timeout=30 * 60, scaledown_window=2, **COMMON)
def decision_map(harnesses: list[str] = [KEEP, TARGET], kinds: list[str] = ["continue"], tag: str = "",
                 multi_only: bool = True) -> dict:
    """Gradient of -log p(gold token) at decision points of the given kinds only (no quoting/style noise).
    Also saves hidden states at those points. Files: /runs/sft_v1/maps/grad_dp_<tag><h>.safetensors and
    states_dp_<tag><h>.pt ({kind: [n, layers+1, d]})."""
    import torch
    from safetensors.torch import save_file
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    from dawe import bfcl
    from dawe.harness import Harness, render, target

    t0 = time.time()
    runs.reload()
    path = f"/runs/{BASE_RUN}/merged"
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForImageTextToText.from_pretrained(path, dtype=torch.bfloat16).cuda()
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    P = {n: p for n, p in model.named_parameters() if "language_model" in n and p.dim() == 2}
    for n, p in model.named_parameters():
        p.requires_grad_(n in P)
    items = [it for it in bfcl.load_all(Path("/root/data/bfcl"))
             if it.split == "train" and (len(it.ground_truth) > 1 or not multi_only)]
    out = {"n_items": len(items)}
    for h in harnesses:
        hz = Harness.load(h)
        model.zero_grad(set_to_none=True)
        n_pts, nll = 0, 0.0
        per_kind: dict = {}
        states: dict = {}
        for it in items:
            msgs, _ = render(hz, it)
            reply = target(hz, it)
            prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            enc = tok(prompt + reply + "<|im_end|>", add_special_tokens=False, return_offsets_mapping=True)
            offs = enc["offset_mapping"]
            pts, pk = [], []
            for ce, kind in points_for(reply, hz.call_format):
                if kind in kinds:
                    d = max(i for i, (s, e) in enumerate(offs) if e <= len(prompt) + ce)
                    pts.append(d)
                    pk.append(kind)
            if not pts:
                continue
            ids = torch.tensor([enc["input_ids"]]).cuda()
            res = model(input_ids=ids, output_hidden_states=True)
            lp = torch.log_softmax(res.logits[0][pts].float(), -1)
            gold = ids[0, [d + 1 for d in pts]]
            each = -lp[torch.arange(len(pts)), gold]
            each.sum().backward()
            with torch.no_grad():
                for j, (d, kind) in enumerate(zip(pts, pk)):
                    per_kind.setdefault(kind, []).append(each[j].item())
                    states.setdefault(kind, []).append(torch.stack([hs[0, d].float() for hs in res.hidden_states]).half().cpu())
            n_pts += len(pts)
            nll += each.sum().item()
        grads = {n: (p.grad / max(1, n_pts)).detach().to(torch.bfloat16).cpu().contiguous()
                 for n, p in P.items() if p.grad is not None}
        save_file(grads, f"/runs/{BASE_RUN}/maps/grad_dp_{tag}{h}.safetensors")
        torch.save({k: torch.stack(v) for k, v in states.items()}, f"/runs/{BASE_RUN}/maps/states_dp_{tag}{h}.pt")
        norms = {n: g.float().norm().item() for n, g in grads.items()}
        out[h] = {"n_points": n_pts, "nll": nll / max(1, n_pts), "grad_norm": norms,
                  "nll_by_kind": {k: sum(v) / len(v) for k, v in per_kind.items()},
                  "n_by_kind": {k: len(v) for k, v in per_kind.items()}}
    runs.commit()
    secs = time.time() - t0
    out["cost"] = {"job": "decision_map", "seconds": round(secs), "usd_est": round(secs * RATE, 3)}
    return out


# --- local entry points (run by us) --------------------------------------------------------------


def _log_cost(c: dict) -> None:
    with (ROOT / "runs" / "costs.jsonl").open("a") as f:
        f.write(json.dumps(c) + "\n")


@app.local_entrypoint()
def setup(session: str = "s1", target: str = TARGET, keep: str = KEEP, kinds: str = "continue",
          tag: str = "", skip_map: bool = False, multi_only: bool = True):
    """Session config + decision-point map + the unpatched baseline on train items for this session."""
    write_config.remote(session, {"target": target, "keep": keep})
    jobs = {"base": try_patch.spawn(session, None, free=True)}
    if not skip_map:
        jobs["map"] = decision_map.spawn([keep, target], kinds.split(","), tag, multi_only)
    if "map" in jobs:
        m = jobs["map"].get()
        _log_cost(m["cost"])
        (ROOT / "runs" / f"map_dp{('_' + tag.rstrip('_')) if tag else ''}.json").write_text(json.dumps(m))
        print("saved map", m["cost"], {h: (m[h]["n_by_kind"], m[h]["nll_by_kind"]) for h in (keep, target)})
    b = jobs["base"].get()
    out = ROOT / "runs" / "agent" / session
    out.mkdir(parents=True, exist_ok=True)
    (out / "baseline_train.json").write_text(json.dumps(b, indent=1))
    led = b.pop("ledger")
    print(json.dumps({h: {k: v for k, v in s.items()} for h, s in b["scores"].items()}, indent=1), led)


@app.function(image=lab_image, timeout=120, cpu=0.25, memory=512, volumes=VOLS)
def write_config(session: str, cfg: dict) -> None:
    _sdir(session).mkdir(parents=True, exist_ok=True)
    (_sdir(session) / "config.json").write_text(json.dumps(cfg))
    runs.commit()


@app.local_entrypoint()
def final(session: str, name: str, eval_file: str = "data/eval/bfcl_test.jsonl", suffix: str = ""):
    """Hidden test set, with the patch applied. Writes runs/preds_dawe_<session>_<name><suffix>.jsonl."""
    rows = [json.loads(l) for l in (ROOT / eval_file).read_text().splitlines() if l.strip()]
    r = final_eval.remote(session, name, rows)
    _log_cost(r["cost"])
    out = ROOT / "runs" / f"preds_dawe_{session}_{name}{suffix}.jsonl"
    out.write_text("".join(json.dumps(p) + "\n" for p in r["preds"]))
    print("saved", out, r["cost"])


@app.function(image=lab_image, timeout=30 * 60, cpu=2, memory=8 * 1024, volumes=VOLS)
def export_patch(session: str, name: str) -> bytes:
    """The sparse .gpatch for a session patch, converting an old whole-tensor patch if needed. CPU only.

    The old saver recorded `changed_weights` next to each patch; the conversion must reproduce that count exactly,
    otherwise it raises (the stored tensors and the checkpoint disagree about what was edited).
    """
    from dawe import patch as gp

    runs.reload()
    pdir = _sdir(session) / "patches"
    gpatch = pdir / f"{name}{gp.SUFFIX}"
    if not gpatch.exists():
        gp.create_from_dense(f"/runs/{BASE_RUN}/merged", pdir / f"{name}.safetensors", gpatch, name=f"{session}/{name}",
                             lineage=pins.lineage_sft_v1(), ident=_source_identity())
        runs.commit()
    meta = json.loads((pdir / f"{name}.json").read_text())
    got = gp.load_patch(gpatch).changed()
    if got != meta["changed_weights"]:
        raise RuntimeError(f"{session}/{name}: sparse patch changes {got} weights, the lab recorded {meta['changed_weights']}")
    return gpatch.read_bytes()


@app.local_entrypoint()
def export(session: str, name: str, out: str = "release/patches"):
    """Download a session patch as a sparse .gpatch: modal run gpu/lab.py::export --session s1 --name eos12"""
    data = export_patch.remote(session, name)
    dest = ROOT / out / f"{session}_{name}.gpatch"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    print("saved", dest, f"({len(data):,} bytes)")


@app.function(image=lab_image, cpu=1, memory=2048, timeout=600)
def freeze_lab() -> str:
    import importlib.metadata as md
    import platform

    pkgs = sorted({f"{d.metadata['Name']}=={d.version}" for d in md.distributions()}, key=str.lower)
    return f"# python {platform.python_version()} on {platform.platform()}\n" + "\n".join(pkgs) + "\n"


@app.local_entrypoint()
def freeze():
    """modal run gpu/lab.py::freeze  ->  repro/freeze_lab.txt (the lab's analysis image; vLLM image: see gpu/app.py::freeze)."""
    (ROOT / "repro").mkdir(exist_ok=True)
    (ROOT / "repro" / "freeze_lab.txt").write_text(freeze_lab.remote())
    print("saved", ROOT / "repro" / "freeze_lab.txt")
