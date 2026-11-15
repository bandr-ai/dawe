"""Everything a reproduction has to match, in one place.

The Modal jobs in gpu/ import these constants, and docs/REPRODUCE.md quotes them.
Change a pin here, not in the job files.
"""

from __future__ import annotations

# --- base model -------------------------------------------------------------------------------
BASE_MODEL = "Qwen/Qwen3.5-4B"
BASE_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"  # Hugging Face commit, 2026-03-02

# --- dataset ----------------------------------------------------------------------------------
BFCL_REPO = "gorilla-llm/Berkeley-Function-Calling-Leaderboard"
# Hugging Face commit the data files below were fetched from (last modified 2026-04-29, before our runs).
BFCL_REVISION = "61fc0608cfd831fcfbbaa676ebdfef0ed963eeda"
BFCL_SHA256 = {
    "BFCL_v3_simple.json": "fbc37b2ad252bf9af985582e0e07b456173fe627d957491472ea9cef5fb83158",
    "BFCL_v3_multiple.json": "aef168155ebd74b7ac2401198b201343bc7d16d7a3d7e0d4e6d8ee82c6969b2a",
    "BFCL_v3_parallel.json": "fad375d68295776efb63e5aa331dd884f8635e7bb50e05908113c74f14281c3c",
    "possible_answer/BFCL_v3_simple.json": "2911a2bc00df82c4f999ffa64fedb0164cb88e96212ffa5972087eb91fd496ee",
    "possible_answer/BFCL_v3_multiple.json": "244e00ce9395df948bcafc7bee64e8f9c87ef70887587d83cae45b13699f3047",
    "possible_answer/BFCL_v3_parallel.json": "3af54bf04e5b7640b03a9b6a82440f52474feb24b398f94435fd2b155c7727de",
}

# --- baseline recipe (LoRA SFT on harness v1) -------------------------------------------------
SFT = {
    "harness": "v1",
    "epochs": 3,
    "lr": 1e-4,
    "lora_rank": 32,
    "lora_alpha": 64,
    "lora_dropout": 0.0,
    "lora_targets": "every nn.Linear in the language model (not vision tower, not lm_head)",
    "batch": 4,
    "grad_accum": 4,
    "warmup_steps": 10,
    "schedule": "linear decay to 0",
    "weight_decay": 0.0,
    "grad_clip": 1.0,
    "seed": 0,
    "precision": "bf16",
    "thinking": "off (enable_thinking=False)",
    "loss": "reply tokens + <|im_end|> only",
}

# --- generation ------------------------------------------------------------------------------
GENERATION = {"temperature": 0.0, "max_tokens": 512, "max_model_len": 8192, "seed": 0, "thinking": "off"}

# --- software (top-level pins; transitive versions are resolved when Modal builds the image) -----
PYTHON_IMAGE = "3.12"
TRAIN_PACKAGES = {
    "torch": "2.13.0",
    "transformers": "5.20.2",
    "peft": "0.21.2",
    "accelerate": "1.15.0",
    "flash-linear-attention": "0.5.2",
}
VLLM_PACKAGES = {"vllm": "0.31.0"}
UNPINNED_HELPERS = ("hf_transfer", "numpy")  # numpy only in the lab image


def pip_pins(packages: dict[str, str]) -> list[str]:
    return [f"{name}=={version}" for name, version in packages.items()]


def lineage_sft_v1() -> dict:
    """How the checkpoint the released patches apply to (`sft_v1/merged`) was made."""
    return {
        "checkpoint": "sft_v1/merged",
        "base_model": {"repo": BASE_MODEL, "revision": BASE_REVISION},
        "method": "LoRA SFT on BFCL train split rendered with harness v1, adapter merged into bf16 weights",
        "sft": SFT,
        "train_split": "splits/bfcl_v3_split.json (train ids)",
        "dataset": {"repo": BFCL_REPO, "revision": BFCL_REVISION},
        "software": {**TRAIN_PACKAGES, "image": f"modal debian_slim python {PYTHON_IMAGE}"},
        "note": "LoRA does not touch embed_tokens, so the patched rows are identical to the base model's; "
                "the fingerprint covers every tensor to tell the fine-tune from the base.",
    }
