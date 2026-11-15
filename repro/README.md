# repro/

Machine-readable records of the environment.

| File | What | Status |
|---|---|---|
| `freeze_train.txt`, `freeze_vllm.txt`, `freeze_lab.txt` | every package (with transitive versions) in the three Modal images | **not yet recorded.** Generate with `modal run gpu/app.py::freeze` and `modal run gpu/lab.py::freeze`, then commit |

Top-level pins live in [`geodesic/pins.py`](../geodesic/pins.py); the checkpoint lineage is in
[`release/lineage_sft_v1.json`](../release/lineage_sft_v1.json).
