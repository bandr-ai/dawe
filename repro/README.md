# repro/

Machine-readable records of the environment.

| File | What |
|---|---|
| `freeze_train.txt` | every package (with transitive versions) in the SFT / map image (`gpu/app.py::train_image`) |
| `freeze_vllm.txt` | the same for the generation image (`gpu/app.py::vllm_image`, `gpu/lab.py::vllm_image`) |
| `freeze_lab.txt` | the same for the lab's analysis image (`gpu/lab.py::lab_image`) |

Recorded 2026-11-16 by `modal run gpu/app.py::freeze` and `modal run gpu/lab.py::freeze` (CPU containers).

**Caveats.** These are the versions the images resolve to *when built on that date*. The original runs (2026-11-11 and 11-12)
used images built then; Modal caches images, so they are probably the same, but nothing recorded at the time proves it. Note that
the generation image resolves `transformers==5.17.0` (pulled in by vLLM 0.31.0), not the 5.20.2 used for training, so generation and
training used different `transformers` versions.

Top-level pins live in [`geodesic/pins.py`](../geodesic/pins.py); the checkpoint lineage is in
[`release/lineage_sft_v1.json`](../release/lineage_sft_v1.json).
