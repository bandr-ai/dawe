"""Modal jobs: LoRA SFT and vLLM generation for Qwen3.5-4B.

Every job prints its wall time and an estimated cost from Modal's list prices.
Resources are kept small on purpose: 1 H100, 2 CPU cores, 32 GiB RAM.

  modal run gpu/app.py::smoke          ~10 min: 10 training steps + 20 generations
  modal run gpu/app.py::sft            full LoRA SFT on data/sft/v1_bfcl_train.jsonl
  modal run gpu/app.py::gen --model base|<run name>
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import modal

MODEL = "Qwen/Qwen3.5-4B"
REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"

# Modal list prices (modal.com/pricing, checked 2026-11-11), USD per second.
PRICE = {"H100": 0.001097, "cpu_core": 0.0000131, "mem_gib": 0.00000222}
GPU, CPU, MEM_GIB = "H100", 2.0, 16
RATE = PRICE["H100"] + CPU * PRICE["cpu_core"] + MEM_GIB * PRICE["mem_gib"]  # ≈ $4.29/hr

app = modal.App("geodesic")
runs = modal.Volume.from_name("geodesic-runs", create_if_missing=True)
hf_cache = modal.Volume.from_name("geodesic-hf-cache", create_if_missing=True)
VOLS = {"/runs": runs, "/root/.cache/huggingface": hf_cache}

train_image = modal.Image.debian_slim(python_version="3.12").uv_pip_install(
    "torch==2.13.0",
    "transformers==5.20.2",
    "peft==0.21.2",
    "accelerate==1.15.0",
    "flash-linear-attention==0.5.2",
    "hf_transfer",
).env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})

# The FlashInfer sampler compiles a kernel at start-up and needs nvcc, which this slim image
# doesn't have. We decode greedily, so vLLM's own sampler is fine.
vllm_image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("vllm==0.31.0", "hf_transfer").env(
    {"HF_HUB_ENABLE_HF_TRANSFER": "1", "VLLM_USE_FLASHINFER_SAMPLER": "0"}
)

COMMON = dict(gpu=GPU, cpu=CPU, memory=MEM_GIB * 1024, volumes=VOLS, scaledown_window=2)


def _cost(t0: float, label: str) -> dict:
    secs = time.time() - t0
    out = {"job": label, "seconds": round(secs), "usd_est": round(secs * RATE, 3)}
    print(f"[cost] {label}: {secs / 60:.1f} min ≈ ${out['usd_est']:.2f}")
    return out


# --- training ----------------------------------------------------------------------


@app.function(image=train_image, timeout=45 * 60, **COMMON)
def train(rows: list[dict], run: str, epochs: float = 3, lr: float = 1e-4, rank: int = 32, max_steps: int = -1) -> dict:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    t0 = time.time()
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    model = AutoModelForImageTextToText.from_pretrained(MODEL, revision=REVISION, dtype=torch.bfloat16).cuda()

    # LoRA on every Linear layer of the language model (not vision, not lm_head).
    targets = [
        n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear) and "language_model" in n and "lm_head" not in n
    ]
    kinds = sorted({n.rsplit(".", 1)[-1] for n in targets})
    print(f"LoRA targets: {len(targets)} layers, kinds {kinds}")
    model = get_peft_model(model, LoraConfig(r=rank, lora_alpha=2 * rank, lora_dropout=0.0, target_modules=targets))
    model.print_trainable_parameters()

    # Prompt = chat template with thinking off; loss only on the reply.
    examples = []
    for r in rows:
        prompt = tok.apply_chat_template(r["messages"][:-1], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        c_ids = tok(r["messages"][-1]["content"] + "<|im_end|>", add_special_tokens=False)["input_ids"]
        examples.append((p_ids + c_ids, [-100] * len(p_ids) + c_ids))
    lens = sorted(len(x) for x, _ in examples)
    print(f"{len(examples)} examples, tokens median {lens[len(lens) // 2]}, max {lens[-1]}, total {sum(lens)}")

    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.0)
    batch, accum = 4, 4  # effective batch 16
    steps_per_epoch = max(1, len(examples) // (batch * accum))
    total = max_steps if max_steps > 0 else int(steps_per_epoch * epochs)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 10) * max(0.0, 1 - s / total))
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    g = torch.Generator().manual_seed(0)
    order: list[int] = []
    model.train()
    log = []
    for step in range(total):
        loss_sum = 0.0
        for _ in range(accum):
            if len(order) < batch:
                order += torch.randperm(len(examples), generator=g).tolist()
            idx, order = order[:batch], order[batch:]
            b = [examples[i] for i in idx]
            L = max(len(x) for x, _ in b)
            ids = torch.tensor([x + [pad] * (L - len(x)) for x, _ in b]).cuda()
            lab = torch.tensor([y + [-100] * (L - len(y)) for _, y in b]).cuda()
            att = torch.tensor([[1] * len(x) + [0] * (L - len(x)) for x, _ in b]).cuda()
            loss = model(input_ids=ids, attention_mask=att, labels=lab).loss / accum
            loss.backward()
            loss_sum += loss.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(), sched.step(), opt.zero_grad()
        log.append(round(loss_sum, 4))
        if step % 5 == 0 or step == total - 1:
            print(f"step {step + 1}/{total} loss {loss_sum:.4f}")

    out = Path(f"/runs/{run}")
    model.save_pretrained(out / "adapter")
    merged = model.merge_and_unload()
    merged.save_pretrained(out / "merged", safe_serialization=True)
    tok.save_pretrained(out / "merged")
    # vLLM needs the processor files that came with the base model.
    from huggingface_hub import hf_hub_download

    for f in ("preprocessor_config.json", "video_preprocessor_config.json", "chat_template.jinja"):
        (out / "merged" / f).write_bytes(Path(hf_hub_download(MODEL, f, revision=REVISION)).read_bytes())
    info = {"run": run, "steps": total, "loss": log, "lr": lr, "rank": rank, "epochs": epochs, "n": len(rows)}
    (out / "train.json").write_text(json.dumps(info, indent=2))
    runs.commit()
    info["cost"] = _cost(t0, f"train {run}")
    return info


# --- generation --------------------------------------------------------------------


@app.function(image=vllm_image, timeout=30 * 60, **COMMON)
def generate(rows: list[dict], model: str = "base", max_tokens: int = 512) -> dict:
    from vllm import LLM, SamplingParams

    t0 = time.time()
    if model == "base":
        path, rev = MODEL, REVISION
    else:
        runs.reload()
        path, rev = f"/runs/{model}/merged", None
    llm = LLM(
        model=path,
        revision=rev,
        max_model_len=8192,
        gpu_memory_utilization=0.85,
        limit_mm_per_prompt={"image": 0, "video": 0},
        seed=0,
    )
    params = SamplingParams(temperature=0.0, max_tokens=max_tokens)
    outs = llm.chat([r["messages"] for r in rows], params, chat_template_kwargs={"enable_thinking": False})
    preds = [{"id": r["id"], "harness": r["harness"], "text": o.outputs[0].text} for r, o in zip(rows, outs)]
    return {"preds": preds, "cost": _cost(t0, f"generate {model} x{len(rows)}")}


# --- entry points (run locally, call the GPU functions) --------------------------------

ROOT = Path(__file__).resolve().parent.parent


def _read(p: str) -> list[dict]:
    return [json.loads(line) for line in (ROOT / p).read_text().splitlines() if line.strip()]


def _save(name: str, result: dict) -> Path:
    out = ROOT / "runs" / f"{name}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(p) + "\n" for p in result["preds"]))
    with (ROOT / "runs" / "costs.jsonl").open("a") as f:
        f.write(json.dumps(result["cost"]) + "\n")
    return out


@app.local_entrypoint()
def smoke():
    """Tiny end to end check: 10 training steps, then 20 generations from base and from the smoke model."""
    sft = _read("data/sft/v1_bfcl_train.jsonl")[:64]
    info = train.remote(sft, "smoke", max_steps=10)
    print("loss", info["loss"])
    with (ROOT / "runs" / "costs.jsonl").open("a") as f:
        f.write(json.dumps(info["cost"]) + "\n")
    rows = _read("data/eval/bfcl_test.jsonl")
    ev = [r for r in rows if r["harness"] == "v1"][:10] + [r for r in rows if r["harness"] == "v4_format"][:10]
    print("saved", _save("smoke_base", generate.remote(ev, "base")))
    print("saved", _save("smoke_sft", generate.remote(ev, "smoke")))


@app.local_entrypoint()
def smoke_gen():
    """Generation half of the smoke test (reuses the saved smoke model)."""
    rows = _read("data/eval/bfcl_test.jsonl")
    ev = [r for r in rows if r["harness"] == "v1"][:10] + [r for r in rows if r["harness"] == "v4_format"][:10]
    print("saved", _save("smoke_base", generate.remote(ev, "base")))
    print("saved", _save("smoke_sft", generate.remote(ev, "smoke")))


@app.local_entrypoint()
def sft(run: str = "sft_v1", epochs: float = 3):
    info = train.remote(_read("data/sft/v1_bfcl_train.jsonl"), run, epochs=epochs)
    with (ROOT / "runs" / "costs.jsonl").open("a") as f:
        f.write(json.dumps(info["cost"]) + "\n")
    print(json.dumps({k: v for k, v in info.items() if k != "loss"}, indent=2), "final loss", info["loss"][-5:])


@app.local_entrypoint()
def gen(model: str = "base", eval_file: str = "data/eval/bfcl_test.jsonl", append: bool = False):
    """append=True adds new harness rows to an existing preds file (for a newly added harness)."""
    res = generate.remote(_read(eval_file), model)
    if append:
        old = ROOT / "runs" / f"preds_{model}.jsonl"
        res["preds"] = [json.loads(l) for l in old.read_text().splitlines() if l.strip()] + res["preds"]
    print("saved", _save(f"preds_{model}", res))


@app.local_entrypoint()
def full(run: str = "sft_v1", epochs: float = 3):
    """Step 2: SFT on v1, then base and trained answers on every test prompt (run in parallel)."""
    info = train.remote(_read("data/sft/v1_bfcl_train.jsonl"), run, epochs=epochs)
    with (ROOT / "runs" / "costs.jsonl").open("a") as f:
        f.write(json.dumps(info["cost"]) + "\n")
    print("loss first/last", info["loss"][:3], info["loss"][-3:])
    rows = _read("data/eval/bfcl_test.jsonl")
    calls = {"base": generate.spawn(rows, "base"), run: generate.spawn(rows, run)}
    for name, c in calls.items():
        print("saved", _save(f"preds_{name}", c.get()))


@app.local_entrypoint()
def ceiling(harnesses: str = "v4_format,v6_all", epochs: float = 3, eval_files: str = "data/eval/bfcl_test.jsonl"):
    """Step 3: train base Qwen directly on each new harness (the best case), then score on every test prompt."""
    names = harnesses.split(",")
    jobs = {h: train.spawn(_read(f"data/sft/{h}_bfcl_train.jsonl"), f"ceiling_{h}", epochs=epochs) for h in names}
    rows = [r for f in eval_files.split(",") for r in _read(f)]
    gens = {}
    for h, j in jobs.items():
        info = j.get()
        with (ROOT / "runs" / "costs.jsonl").open("a") as f:
            f.write(json.dumps(info["cost"]) + "\n")
        print(h, "loss first/last", info["loss"][:2], info["loss"][-2:])
        gens[h] = generate.spawn(rows, f"ceiling_{h}")
    for h, g in gens.items():
        print("saved", _save(f"preds_ceiling_{h}", g.get()))


# --- step 4: data -> weight map ---------------------------------------------------------


def _decision_points(reply: str, fmt: str, calls_json: list[str] | None = None) -> list[tuple[int, bool]]:
    """Character offsets right after each tool call in a reply, and whether another call should follow."""
    ends = []
    if fmt == "xml":
        i = 0
        while (j := reply.find("</tool_call>", i)) != -1:
            ends.append(j + len("</tool_call>"))
            i = ends[-1]
    else:  # json: find each serialized call object in order
        i = 0
        for cj in calls_json or []:
            j = reply.find(cj, i)
            if j == -1:
                return []
            ends.append(j + len(cj))
            i = ends[-1]
    return [(e, k < len(ends) - 1) for k, e in enumerate(ends)]


@app.function(image=train_image, timeout=40 * 60, **COMMON)
def weight_map(pairs: list[dict], a: str, b: str, run: str = "sft_v1") -> dict:
    """For one model and paired traces (harness a vs b):
    - gradient of the SFT loss on a-replies and on b-replies, per module (saved to the volume)
    - hidden states at 'another call should follow' vs 'stop here' points, per layer
    - how surprised the model is to continue after a non-final call, under a and under b
    """
    import torch
    from safetensors.torch import save_file
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    t0 = time.time()
    runs.reload()
    path = f"/runs/{run}/merged"
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForImageTextToText.from_pretrained(path, dtype=torch.bfloat16).cuda()
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    lm_params = {n: p for n, p in model.named_parameters() if "language_model" in n and p.dim() == 2}
    for n, p in model.named_parameters():
        p.requires_grad_(n in lm_params)

    def encode(msgs):
        prompt = tok.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        reply = msgs[-1]["content"]
        enc = tok(prompt + reply + "<|im_end|>", add_special_tokens=False, return_offsets_mapping=True)
        return prompt, reply, enc

    out: dict = {"a": a, "b": b, "run": run, "n_pairs": len(pairs)}
    for h in (a, b):
        fmt = "xml" if "<tool_call" in pairs[0][h][-1]["content"] else "json"
        model.zero_grad(set_to_none=True)
        states = {True: [], False: []}  # continue? -> list of [layers, hidden]
        nll = {True: [], False: []}
        for pr in pairs:
            msgs = pr[h]
            prompt, reply, enc = encode(msgs)
            ids = torch.tensor([enc["input_ids"]]).cuda()
            offs = enc["offset_mapping"]
            start = next(i for i, (s, e) in enumerate(offs) if e > len(prompt))
            labels = ids.clone()
            labels[0, :start] = -100
            res = model(input_ids=ids, labels=labels, output_hidden_states=True)
            (res.loss).backward()
            calls_json = None
            if fmt == "json":
                calls_json = [json.dumps(c, ensure_ascii=False) for c in json.loads(reply)["tool_calls"]]
            with torch.no_grad():
                logp = torch.log_softmax(res.logits[0].float(), -1)
                for ce, cont in _decision_points(reply, fmt, calls_json):
                    abs_end = len(prompt) + ce
                    d = max(i for i, (s, e) in enumerate(offs) if e <= abs_end)
                    if d + 1 >= ids.shape[1]:
                        continue
                    states[cont].append(torch.stack([hs[0, d].float() for hs in res.hidden_states]).cpu())
                    nll[cont].append(-logp[d, ids[0, d + 1]].item())
        grads = {n: p.grad.detach().to(torch.bfloat16).cpu().contiguous() for n, p in lm_params.items() if p.grad is not None}
        Path(f"/runs/{run}/maps").mkdir(parents=True, exist_ok=True)
        save_file(grads, f"/runs/{run}/maps/grad_{h}.safetensors")
        out[h] = {
            "grad_norm": {n: g.float().norm().item() for n, g in grads.items()},
            "numel": {n: g.numel() for n, g in grads.items()},
            "n_continue": len(states[True]),
            "n_stop": len(states[False]),
            "nll_continue": sum(nll[True]) / max(1, len(nll[True])),
            "nll_stop": sum(nll[False]) / max(1, len(nll[False])),
        }
        # per layer: how well a single direction separates continue vs stop (fit on half, test on other half)
        if states[True] and states[False]:
            C, S = torch.stack(states[True]), torch.stack(states[False])  # [n, layers, d]
            torch.save({"continue": C.half(), "stop": S.half()}, f"/runs/{run}/maps/states_{h}.pt")
            sep = []
            for L in range(C.shape[1]):
                c, s = C[:, L], S[:, L]
                c1, c2, s1, s2 = c[::2], c[1::2], s[::2], s[1::2]
                direction = c1.mean(0) - s1.mean(0)
                direction = direction / (direction.norm() + 1e-8)
                thr = ((c1 @ direction).mean() + (s1 @ direction).mean()) / 2
                acc = (((c2 @ direction) > thr).float().sum() + ((s2 @ direction) <= thr).float().sum()) / (len(c2) + len(s2))
                pc, ps = c @ direction, s @ direction
                d_eff = (pc.mean() - ps.mean()) / (((pc.var() + ps.var()) / 2).sqrt() + 1e-8)
                sep.append({"layer": L, "heldout_acc": round(acc.item(), 3), "cohens_d": round(d_eff.item(), 2),
                            "dir": (C[:, L].mean(0) - S[:, L].mean(0)).tolist()})
            out[h]["separation"] = sep
    runs.commit()
    out["cost"] = _cost(t0, f"map {run} {a} vs {b}")
    return out


@app.local_entrypoint()
def map_(pairs: str = "v1__v4_format,v1__v6_all", run: str = "sft_v1"):
    """Step 4. Writes runs/map_<pair>.json locally."""
    jobs = {}
    for p in pairs.split(","):
        a, b = p.split("__")
        jobs[p] = weight_map.spawn(_read(f"data/pairs/{p}.jsonl"), a, b, run)
    for p, j in jobs.items():
        res = j.get()
        with (ROOT / "runs" / "costs.jsonl").open("a") as f:
            f.write(json.dumps(res["cost"]) + "\n")
        (ROOT / "runs" / f"map_{p}.json").write_text(json.dumps(res))
        print("saved", ROOT / "runs" / f"map_{p}.json")
