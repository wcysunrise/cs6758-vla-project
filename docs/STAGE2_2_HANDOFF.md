# Stage 2.2 handoff

## Local deliverables

Branch: `stage2-memory`. Stage 1 and Stage 2.1 remain unchanged.

- `memory/embeddings.py`: frozen ResNet-18 and feature/state transformations.
- `memory/build_memory.py`: validates original 40/10 split and builds training memory.
- `memory/retrieve.py`: exact cosine Top-1/Top-5 and raw LIBERO observation adapter.
- `memory/demo_memory.npz`: actual full CPU-built memory, **excluded from Git**.
- `memory/metadata.json`: encoder, preprocessing, source/split provenance and NPZ hash.
- `memory/heldout_retrieval.json`: full real held-out results, **excluded from Git**.
- `docs/STAGE2_2_LOCAL_VALIDATION.json`: measured validation and test evidence.
- `docs/STAGE2_2_RETRIEVAL_EXAMPLES.json`: compact real-query examples.
- `notebooks/stage2_2_build_memory.ipynb`: prepared fresh-cloud workflow.

The existing `data/stage2_1_verified/` provides both HDF5 files and split/metadata
JSON files. It is excluded from Git. Do not assume cloning downloads this data.
The memory has 6,193 training rows; evaluation queries 1,565 frames from 10
held-out trajectories. See the [memory report](STAGE2_2_MEMORY_REPORT.md) for
schema, exact transformations, success caveats and metric interpretation.

## Verified locally / unverified remotely

**28 tests passed, no skips.** Frozen pretrained CPU inference built all 6,193
memory rows and evaluated all 1,565 held-out frames in both modes. All 15,650
returned neighbor action/identity records matched memory. No held-out leakage.
Prior-stage files and extracted HDF5 hashes are unchanged. The complete notebook
passes schema and code-cell syntax checks.

CUDA/V100 execution, fresh cloud execution, GCS transfer, live simulator adapter
behavior and simulator-verified demonstration success remain **UNVERIFIED**.

## Reproduce locally (CPU)

```bash
.venv/bin/python -m pip install -r requirements-stage2_2.txt
# Use fresh outputs; defaults already contain the validated local memory.
.venv/bin/python -m memory.build_memory --data-dir data/stage2_1_verified --output-dir data/stage2_2_new/memory --device cpu
.venv/bin/python -m memory.retrieve --data-dir data/stage2_1_verified --memory-dir data/stage2_2_new/memory --output data/stage2_2_new/heldout_retrieval.json --device cpu
.venv/bin/python -m unittest discover -s tests -v
```

`--batch-size` controls encoding memory. `--camera eye_in_hand_rgb` creates a
wrist-only memory; default uses the main camera. Set retrieval
`--visual-weight 1 --state-weight 1` as needed. Statistics always come from train.
Unit tests use genuine pretrained weights (first download about 45 MB), tiny
synthetic fixtures for exact-ranking edge cases, and real artifacts for integration.
Set `STAGE2_DATA_DIR` and `STAGE2_MEMORY_DIR` when testing non-default locations.
A missing real artifact causes an explicitly labeled skip, not a passing real test.

Python API for a raw, native OpenGL LIBERO observation:

```python
from memory.retrieve import Retriever
visual = Retriever('memory', mode='visual', device='cpu')
combined = Retriever('memory', mode='visual_state', visual_weight=1,
                     state_weight=1, device='cpu')
# raw_obs is the unmodified LIBERO observation dictionary, not policy input.
top5 = combined.query_observation(raw_obs, k=5, task_id=0)
top1 = top5[0]
```

The only supported task is the recorded Stage 1 task; querying another task ID
fails. The adapter uses main/wrist image keys according to memory metadata.
It returns actions as data only and never executes them. For a Stage 2.1 frame,
use `query_image(native_rgb, state8, k=5)` without a second action shift.

## Run on Colab Enterprise V100

1. Publish this local Stage 2.2 commit or upload a checkout ZIP. Open
   `notebooks/stage2_2_build_memory.ipynb` in a fresh Linux Python 3.10–3.12 runtime.
   Set `REPO_DIR` to the checkout and `SOURCE_REF` to the published commit SHA.
   The task commits locally; publishing is a separate action.
2. Attach the V100 16 GB and leave `DEVICE='cuda'`. The notebook installs isolated
   torch 2.7.1 / torchvision 0.22.1 cu126, checks architecture `sm_70` on capability
   7.0, and performs real CUDA matmul before building. CPU fallback is explicit.
3. Choose input: upload the Stage 2.1 directory and set `LOCAL_DATA_DIR`, upload its
   archive and set `LOCAL_STAGE2_1_ZIP`, or set `GCS_STAGE2_1_ZIP`. The file browser
   is the upload fallback. With no input, the notebook downloads and extracts the
   pinned public native file. With existing data, it validates and reuses it.
4. Set `GCS_OUTPUT_URI='gs://YOUR_BUCKET/cs6758/stage2_2'` when possible. Use runtime
   ADC/IAM permissions configured externally; never embed credentials. Allow
   several GB for dependencies, weights, raw/extracted data and exports.
5. Run all cells. Memory and both-mode held-out diagnostics go under
   `~/stage2_2_outputs/<UTC_RUN_ID>/memory/`. Hardware, tests, build/retrieval logs,
   package versions and source hashes live in the run directory.
6. The notebook exports `~/stage2_2_outputs/<UTC_RUN_ID>.zip`. With GCS configured,
   it uploads to `<GCS_OUTPUT_URI>/<UTC_RUN_ID>.zip`, without overwriting and with
   checksum validation. On unavailable GCS, the explicit browser download/file
   browser fallback remains available. Confirm completion before stopping runtime.

The archive includes both Stage 2.1 HDF5 files, split metadata, the memory NPZ,
metadata, retrieval results and run evidence. `export_locations.json` locates the
image-reference data after extraction. Return this ZIP (or its local extracted
path) for subsequent inspection; do not send credentials. Local code cannot
verify cloud execution or persistence before those results exist.

## Before Stage 2.3

- Run and inspect the cloud notebook; check its actual GPU tests, provenance,
  exports and retrieval examples. Revalidate NPZ/source hashes after transfer.
- Validate raw simulator image orientation/resolution and quaternion conversion
  against the stored demonstrations; test live observation retrieval.
- If verified-success memory is required, replay demonstrations and record actual
  task predicates. Current success status is expert provenance, not replay proof.
- Decide the action comparison space: expert actions are unnormalized relative
  OSC values, while internal policy predictions use checkpoint normalization.
- Specify candidate/chunk scoring, horizon, weights and paired evaluation protocol
  before intervention. Current memory stores single-step actions, not candidate
  policy chunks; any future chunks must stay within one training trajectory.

Stage 2.3, policy intervention, SmolVLA weight changes, GPU execution and simulator
success verification have not been performed by this local implementation.
