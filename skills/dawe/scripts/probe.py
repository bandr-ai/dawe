"""Collect development decision states with a local Transformers model; never trains or generates."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def decision_indices(offsets, prompt: str, reply: str, rendered: str, decisions: list[dict]) -> list[tuple[int, str]]:
    """Return target token indices; reject chat-template and mid-token misalignment."""
    if not rendered.startswith(prompt + reply):
        raise ValueError("Chat template does not align prompt + assistant target; use a model-specific adapter.")
    indices = []
    for point in decisions:
        prefix = point["prefix"]
        if not reply.startswith(prefix):
            raise ValueError("Decision prefix does not match the assistant target.")
        boundary = len(prompt) + len(prefix)
        hits = [i for i, (start, end) in enumerate(offsets) if start == boundary and end > start]
        if len(hits) != 1 or hits[0] == 0:
            raise ValueError("Decision is not an unambiguous token boundary; inspect tokenization or use an adapter.")
        indices.append((hits[0], point["kind"]))
    return indices


def collect(args) -> None:
    # Heavy imports only for an actual authorized model job.
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

    from dawe.patch import identify
    from dawe.workbench import digest, encoded, read_examples, write_new_directory

    root = args.development.resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    raw = (root / "examples.jsonl").read_bytes()
    if manifest.get("split") != "development" or manifest.get("sha256") != digest(raw):
        raise ValueError("Expected unchanged frozen development examples and their manifest.")
    rows = read_examples(root / "examples.jsonl")
    if len(rows) != manifest.get("count") or {r["purpose"] for r in rows} != {"repair", "preserve"}:
        raise ValueError("Development manifest counts or purposes do not match examples.")
    if any(r.get("split") == "evaluation" for r in rows):
        raise ValueError("Evaluation rows are not accepted by the state collector.")
    if not all(row.get("decisions") for row in rows):
        raise ValueError("Every example needs at least one inspected decision prefix before mapping.")
    count = sum(len(r["decisions"]) for r in rows)
    if count > args.max_points:
        raise ValueError(f"{count} points exceeds --max-points {args.max_points}; select fewer development examples.")
    if args.out.exists():
        raise ValueError("Output already exists.")
    options = json.loads(args.template_options)
    if not isinstance(options, dict) or set(options) & {"tokenize", "add_generation_prompt"}:
        raise ValueError("Template options must be an object without tokenize/add_generation_prompt.")
    model_path = Path(args.model).expanduser()
    if not model_path.exists() and not args.revision:
        raise ValueError("A remote model needs --revision pinned to its exact commit.")
    if not model_path.exists() and (len(args.revision) != 40 or any(c not in "0123456789abcdef" for c in args.revision)):
        raise ValueError("Use a full 40-character model commit for --revision.")
    model_id = str(model_path.resolve()) if model_path.exists() else args.model
    kwargs = {"revision": args.revision} if args.revision else {}
    tok = AutoTokenizer.from_pretrained(model_id, **kwargs)
    if not tok.is_fast:
        raise ValueError("The bundled collector needs a fast tokenizer for offset checks.")
    # Validate every token boundary before loading weights or spending forward passes.
    examples = []
    for row in rows:
        prompt = tok.apply_chat_template(row["messages"][:-1], tokenize=False, add_generation_prompt=True, **options)
        rendered = tok.apply_chat_template(row["messages"], tokenize=False, add_generation_prompt=False, **options)
        tokens = tok(rendered, add_special_tokens=False, return_offsets_mapping=True)
        ids = tokens["input_ids"]
        if len(ids) > args.max_length:
            raise ValueError(f"Example {row['id']} exceeds --max-length; no targets have been truncated.")
        points = decision_indices(tokens["offset_mapping"], prompt, row["messages"][-1]["content"], rendered, row["decisions"])
        examples.append((row, ids, points))
    identity = identify(model_path.resolve()).fingerprint if model_path.is_dir() else None
    cfg = AutoConfig.from_pretrained(model_id, **kwargs)
    loader = AutoModelForImageTextToText if getattr(cfg, "text_config", None) is not None else AutoModelForCausalLM
    dtype = torch.float32 if args.device == "cpu" else torch.bfloat16
    model = loader.from_pretrained(model_id, torch_dtype=dtype, **kwargs).to(args.device).eval()
    states, metadata = [], []
    with torch.inference_mode():
        for row, ids, points in examples:
            inputs = torch.tensor([ids], device=args.device)
            output = model(input_ids=inputs, output_hidden_states=True, use_cache=False)
            if output.hidden_states is None:
                raise ValueError("Model did not expose text hidden states; use a model-specific adapter.")
            for index, kind in points:
                logits = output.logits[0, index - 1].float()
                gold, greedy = ids[index], int(logits.argmax())
                states.append(torch.stack([h[0, index - 1].detach().float().cpu() for h in output.hidden_states]))
                metadata.append({"id": row["id"], "group_id": row["group_id"], "source_ref": row["source_ref"],
                                 "purpose": row["purpose"], "kind": kind, "target_index": index,
                                 "gold_token_id": gold, "greedy_token_id": greedy,
                                 "gold_token": tok.decode([gold]), "greedy_token": tok.decode([greedy]),
                                 "gold_nll": float(-torch.log_softmax(logits, -1)[gold])})
            del output
    summary = {"schema_version": 1, "model": model_id, "revision": args.revision, "device": args.device,
               "checkpoint_fingerprint": identity, "dtype": str(dtype), "template_options": options,
               "development_sha256": manifest["sha256"], "n_points": len(metadata),
               "n_examples": len(rows), "by_kind": {kind: sum(p["kind"] == kind for p in metadata)
                                                       for kind in sorted({p["kind"] for p in metadata})},
               "torch_version": torch.__version__}

    def write(p):
        torch.save({"states": torch.stack(states)}, p / "states.pt")
        (p / "points.json").write_bytes(encoded(metadata))
        (p / "summary.json").write_bytes(encoded(summary))

    write_new_directory(args.out, write)
    print(json.dumps(summary, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--revision")
    p.add_argument("--development", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cuda")
    p.add_argument("--max-length", type=int, default=4096)
    p.add_argument("--max-points", type=int, default=256)
    p.add_argument("--template-options", default="{}")
    args = p.parse_args()
    if args.max_length < 1 or args.max_points < 1:
        p.error("max-length and max-points must be positive")
    try:
        collect(args)
    except (ValueError, OSError, ImportError) as exc:
        p.exit(2, f"{exc}\n")


if __name__ == "__main__":
    main()
