# Stage 2.1 handoff

## Delivered

Branch: `stage2-memory`. Local code and real native-data extraction completed.
Stage 1 remains unchanged. Cloud runtime execution is **UNVERIFIED**.

Task: `libero_object`, ID 0, order 0,
`pick_up_the_alphabet_soup_and_place_it_in_the_basket`.
Source: `yifengzhu-hf/LIBERO-datasets`, immutable revision
`f13aa24a3da8c43c7225569f28c562979fa0e35a`.

Read [the data report](STAGE2_1_DATA_REPORT.md) for the complete source/action
comparison, provenance and alignment caveats. Native images are 128×128 RGB;
state is 8D EE position + axis-angle + gripper; actions are 7D relative OSC inputs.
No success labels are fabricated: 50 official expert trajectories are included
under a provenance assumption, with simulator success verification pending.

## Actual local outputs

All paths are relative to the repository. Large files are ignored by Git.

- Source: `data/raw/pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo.hdf5`.
- Final validated output: `data/stage2_1_verified/train_demonstrations.hdf5`.
- Final validated output: `data/stage2_1_verified/eval_demonstrations.hdf5`.
- Exact membership: `data/stage2_1_verified/split_manifest.json`.
- Provenance, schema and statistics: `data/stage2_1_verified/dataset_metadata.json`.
- Extraction log: `data/extraction_verified.log`.
- Earlier validation run: `data/stage2_1/` (superseded by the verified directory).
- Small committed evidence: `docs/STAGE2_1_LOCAL_VALIDATION.json`.

Source: 50 trajectories, 7,808 steps, min/mean/max 136/156.16/196.
Seed 6758; 40 train trajectories / 6,193 aligned pairs;
10 evaluation trajectories / 1,565 pairs. Zero rejected, zero overlap.
Each pair uses source observation i and action i+1 because native observations
are recorded after actions. Original action values and source data are preserved.

## Local reproduction (Mac or Linux CPU)

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-stage2.txt
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m memory.extract_demonstrations --download --output data/new_run
.venv/bin/python -c 'from memory.dataset_utils import validate_outputs; print(validate_outputs("data/new_run"))'
```

For offline operation replace `--download` with `--source /path/to/native.hdf5`.
Output directories must be new. The default checksum gate prevents silently
applying the post-action shift to regenerated datasets. The pipeline requires
no Torch, GPU, model weights, credentials or LIBERO simulator.

## Colab Enterprise / Google Cloud

1. Publish this local branch/commit to GitHub, or upload a ZIP of this checkout
   and unzip it in the runtime. A local commit is not available to cloud clones
   until pushed. This task commits locally; it does not publish the branch.
2. Open `notebooks/stage2_1_prepare_data.ipynb` in Colab Enterprise. Use a fresh
   Linux Python 3.10–3.12 runtime with the V100 attached. Allow several GB for
   the Torch environment, raw dataset, extracted data and export ZIP.
3. Set `REPO_DIR` to the checkout/upload location. For cloning, set `SOURCE_REF`
   to the published commit SHA (`git rev-parse stage2-memory`).
4. Prefer setting `GCS_OUTPUT_URI` to `gs://YOUR_BUCKET/cs6758/stage2_1`.
   Use the runtime identity/ADC with bucket permissions; never paste credentials.
5. Run all cells. Dependencies install in `.venv-stage2`, preserving system Torch.
   The preflight checks CUDA, device capability, native `sm_70` support for a V100,
   and an actual FP32 CUDA matrix multiplication. It stops on failures.
6. Inspect tests, metadata and validation. The notebook downloads the pinned file,
   extracts it, checks output hashes and all arrays, and prints statistics.
7. Export cell writes a ZIP of outputs, logs and source hashes, uploads with
   checksum validation and no overwrite to GCS, or offers an explicit download.
   If a browser download/link fails, download the printed archive path through
   the notebook file browser. Confirm persistence before shutting down.

The raw file remains in `LOCAL_OUTPUT_ROOT/raw`; the archive contains its revision
and checksum for redownload, not a duplicate 744 MiB copy. Each cloud run goes
under `LOCAL_OUTPUT_ROOT/<UTC run ID>/data`, with both partitions and JSON files.

## Stage 2.2 contract / pending verification

Read HDF5 with `h5py.File(path, 'r')['trajectories']`; each group is one trajectory,
and a row's identity is `(group.attrs['trajectory_id'], group['timestep'][i])`.
Slice `state`, either RGB camera and `actions` with that same index. Load only
training groups when fitting embeddings/statistics; preserve these partitions.
Never build chunks across groups or include held-out trajectories in memory.

Explicitly resolve image resolution/orientation and policy normalization before
comparing actions or observations. Do not apply the temporal shift a second time.
Do not treat terminal reward as a success label. If verified-success memory is
required, replay episodes and record actual task predicates first. Native
simulator replay, V100 preflight, full fresh-cloud execution, cloud upload/download
and compatibility at the rendered/policy level remain **UNVERIFIED**.
No Stage 2.2 implementation or weight modification is included.
