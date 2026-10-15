# Stage 1 setup and reproduction

## Validation boundary

The existing Colab Enterprise runtime was accessed on 2026-10-14. It runs Linux
with Python 3.12.13, one V100-SXM2-16GB and driver 580.178.04. A fresh CUDA matrix
multiplication passed on the repaired system PyTorch 2.11.0+cu126; its architecture
list includes sm_70. System torchvision is 0.26.0+cu126.

LeRobot 0.5.1 declares torch >=2.7,<2.11 and torchvision <0.26. Therefore the
project uses an isolated venv with torch 2.7.1+cu126 and torchvision 0.22.1+cu126;
these wheels were installed successfully without modifying the system runtime.
The venv CUDA test and pip check passed. LIBERO task initialization, reset,
7-dimensional action acceptance and both 256x256 EGL camera renders passed.
Frozen-model rollout measurements are recorded in results/STATUS.md and the raw CSV.

## Validated versions

| Component | Validated pin | Evidence / constraint |
| --- | --- | --- |
| Python | 3.12.13 | actual cloud runtime; LeRobot 0.5.1 requires Python >=3.12 |
| LeRobot | 0.5.1 | release source commit `1396b9fab7aecddd10006c33c47a487ffdcb54b4` |
| LIBERO | hf-libero 0.1.4 | HF maintained distribution, imported as `libero`; use no second LIBERO install |
| PyTorch / torchvision | 2.7.1 / 0.22.1 | matched official CUDA 12.6 wheels; meets LeRobot's torch >=2.7 |
| CUDA wheel runtime | 12.6 | `cu126`, not the system `nvcc` version; actual driver must pass preflight |
| torchcodec | 0.3.0 | paired with torch 2.7; not used for recording our videos |
| Transformers | 5.3.0 | exact LeRobot v0.5.1 extra pin |
| MuJoCo | 3.3.2 | within HF LIBERO 0.1.4's >=3.0,<3.9 range |
| robosuite / robomimic / BDDL | 1.4.0 / 0.2.0 / 1.0.1 | HF LIBERO dependencies |
| NumPy | 2.2.6 | within LeRobot >=2.0,<2.3 |
| Policy | `6721902bc4d61e50a3bfdb11dfb4cb626f05d102` | full checkpoint commit; includes processor stats |

These direct pins passed `pip check`, imports, real CUDA operations, EGL rendering,
strict checkpoint loading and the real smoke rollout on the recorded V100 runtime.
`results/installed_versions.txt` and the run manifests preserve resolved package
versions. Other machines still need the same preflight and smoke checks.

V100 requires care: use CUDA 12.6 wheels that retain Volta support and FP32
inference. Do not use BF16 or require FlashAttention-2. The script uses `PreTrainedConfig.from_pretrained` to resolve the saved `type` discriminator,
then constructs
the VLM architecture without independently loading base VLM weights, then
strictly loads the full SmolVLA checkpoint and freezes every parameter.

## Cloud prerequisites and authorization

Reuse the existing `smolvla-libero` template and any running authorized runtime.
Before starting compute, verify billing is active, remaining credits and expiry,
the template's region, global GPU quota, regional V100 quota, CPU/disk quotas,
and GPU availability. Required APIs: `aiplatform.googleapis.com` and
`compute.googleapis.com`; inspect the enabled-services list, not just the
existence of a notebook. The user authorized Stage 1 debugging/evaluation on the existing runtime. Do not
create duplicate resources or change billing settings.

Desired template: n1-standard-4, one NVIDIA V100, idle shutdown 30 minutes.
Template configuration and runtime status must be read from the Console.
No cloud identifiers or exported cloud settings belong in this public repository.

Reference estimate, Iowa/us-central1 on-demand Colab pricing checked 2026-10-14:
GPU $2.976/hour + N1 4 vCPU/15 GiB approximately $0.228/hour + disks. Assuming
100 GiB SSD boot and 100 GiB standard data disks, approximately **$3.24/hour**
before taxes/network/discounts/credits. Thirty minutes is approximately $1.62,
one hour $3.24. This is not an invoice or measured experiment cost. Disk charges
continue when stopped; idle shutdown is not a total spending cap.

## Install on an already approved Linux GPU runtime

Run in a terminal or a notebook `%%bash` cell. Do not run installation in the
system Python; a separate venv avoids the notebook's existing package conflicts.

```bash
git clone https://github.com/wcysunrise/cs6758-vla-project.git
cd cs6758-vla-project
python3.12 -m venv --without-pip .venv
python3.12 -m pip --python .venv/bin/python install pip
source .venv/bin/activate
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
python -m pip check
```

Linux system libraries must include NVIDIA EGL and GLVND, FFmpeg shared libraries
for torchcodec, and standard GL/X11 runtime libraries required by robosuite.
Inspect what the runtime image provides before installing anything with apt.
Typical Ubuntu prerequisites if missing:

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg libegl1 libgl1 libgles2 libglvnd0 libglib2.0-0 libosmesa6
```

The NVIDIA driver comes from the GPU runtime image. Do not replace its driver
or CUDA toolkit blindly. `nvidia-smi` and a real torch CUDA operation must work.

## Configuration and checkpoint download

```bash
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export TOKENIZERS_PARALLELISM=false
export MPLBACKEND=Agg
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python scripts/run_baseline.py --check-config
python scripts/run_baseline.py --preflight
python scripts/check_simulator.py
python -m pip freeze > results/installed_versions.txt
```

`baseline.yaml` uses JSON syntax, valid YAML 1.2. Keep that syntax when editing.
The script resolves and downloads the pinned checkpoint (about 1.22 GB), its
processor normalization files, backbone config/tokenizer files (no base-model
weight download), and LIBERO assets. The renamed backbone ID
`HuggingFaceTB/SmolVLM2-500M-Video-Instruct` is the current redirect target of the
checkpoint's `HuggingFaceTB/SmolVLM2-500M-Instruct` reference.

Backbone revision `7b375e1b73b11138ff12fe22c8f2822d8fe03467` and assets revision
`0b3ea86be5fe169d0fd036ae63d1070ec09e90f6` were verified from their Hub commit pages
and are pinned in the script. They are recorded in `hub_revisions.json`;
the evaluation reuses the smoke run's exact revisions.
Copy this lock to reproduce that run. `lerobot/libero-assets` is a **dataset** repository,
not a model repository; the script downloads it with `repo_type="dataset"`.
The authorized runtime must verify Hub access. If a token is
required, use the normal external HF credential cache or environment variable,
never a config committed to git. No untrusted Hub Python code is executed.

An isolated ignored `.libero/config.yaml` is generated before importing LIBERO,
preventing interactive dataset-path prompts. BDDL and fixed initial-state files
come from hf-libero; their hashes enter the manifest. Simulation assets use the
pinned Hub snapshot. No demonstration dataset download is required for inference.

## Task and rollout protocol

Suite `libero_object`, task ID 0 (zero-based), task order index 0. Exact name:
`pick_up_the_alphabet_soup_and_place_it_in_the_basket`. This is a single-object
pick-and-place task with a clear environment success predicate and a short
280-step horizon. It exercises perception, grasping, transport and release,
without the multiple subtasks of LIBERO-10. It was selected before obtaining
results; it is not evidence that this is the easiest or highest-scoring task.

Main and wrist RGB are 256x256. LeRobot's `preprocess_observation` converts
uint8 HWC pixels to float32 BCHW; its `LiberoProcessorStep` flips both image axes
and converts xyzw end-effector quaternions to axis-angle. State is 8-dimensional:
position(3), axis-angle(3), gripper positions(2). The saved checkpoint processors
normalize observations, tokenize the exact instruction, and unnormalize actions.
Actions are 7-dimensional relative OSC commands. No extra gripper inversion,
normalization, or clipping is added. Controller defaults and finite action shape
are checked; success is `env.check_success()`, not a guessed distance threshold.

`chunk_size=50`, `n_action_steps=1` and 10 flow steps preserve the checkpoint defaults.
Each call predicts 50 actions, executes only the first, discards 49 and queries
again at the next simulator step. Control rate is 20 Hz simulated time; actual
wall-clock inference may be slower. Inference timing includes the checkpoint
preprocessor (normalization/tokenization/device transfer), policy selection,
action unnormalization and CUDA synchronization. It excludes LIBERO image/state
conversion, physics/rendering and the final action transfer to CPU. The first
call may include cold-start cost.
Each episode resets the policy's action queue, simulator seed, random generators,
and fixed initial state. Smoke uses state 5/seed 1005; evaluation uses states
0–4/seeds 1000–1004. Ten no-op steps let objects settle. Already-successful initial
states are rejected. The state file and task name are validated to prevent
silently evaluating the wrong task.

```bash
python scripts/run_baseline.py --smoke
python scripts/run_baseline.py
```

The smoke must complete without execution error and save a decodable MP4; task
failure is still a valid smoke. Its record and video checksum gate evaluation.
To use 10 evaluation episodes, change the config first and repeat smoke. The
smoke then uses state 10/seed 1010 and evaluation uses states 0–9/seeds 1000–1009.
Never silently combine trials from different configs.

The script cannot start or stop a cloud VM. Reuse the existing runtime and
stop it through Runtime management after exporting results.

## Artifacts and reproducibility

`results/smoke/` stores smoke CSV, MP4, JSONL, manifest, revisions and marker.
`results/` stores evaluation CSV, `episode_001.mp4` through `episode_005.mp4`,
per-episode action JSONL, `evaluation.log`, `preflight.json`, `manifest.json`,
`hub_revisions.json`, and `summary.json`. The CSV records termination reason, checkpoint revision, configuration hash,
CUDA-synchronized inference call count/time and runtime including
reset, settling, inference, video encoding/validation; it excludes model loading.
Videos contain the initial frame and every action frame at the simulator's
20 Hz control rate. A partial video remains `.partial.mp4` if execution aborts;
it is not a completed episode and no row is added for it.

The wrapper log records environment checks, episodes and exceptions. For raw
upstream stdout/stderr, preserve an additional console capture:

```bash
set -o pipefail
python -u scripts/run_baseline.py --smoke 2>&1 | tee results/smoke_console.log
python -u scripts/run_baseline.py 2>&1 | tee results/evaluation_console.log
```

The script refuses to overwrite an existing CSV with episode rows. Use a fresh
`--output-dir results/runs/NAME` and `--smoke-record` pointing to that run's
`smoke/smoke.json` for subsequent trials. Completed rows survive interruptions;
infrastructure errors are never counted as unsuccessful robot episodes.
Fixed seeds do not guarantee bitwise determinism across GPUs, drivers or MuJoCo
versions. Five episodes have substantial uncertainty and cannot support broad
claims or comparisons to full-suite published success rates.

## Offline checks

```bash
python scripts/run_baseline.py --check-config
python -m unittest discover -s tests -v
```

These check protocol validation, disjoint smoke/eval states, duplicate-result
protection and the smoke artifact integrity gate. They do not exercise the model,
EGL renderer, controller, or physics. The local macOS host has no NVIDIA GPU and
could not reach PyPI; all reported robot episodes ran on the existing Linux V100
runtime. Local DNS failures are separate from cloud execution results. GitHub
APIs were used for source inspection. The explicitly authorized GitHub delivery
uses the existing browser account because the connector has no write permission.

## Export, cleanup and git

Before stopping/deleting compute, copy all results including videos, traces,
version manifests and raw console logs to durable storage. Verify hashes there.
Stop the existing runtime after export; preserve the notebook, template and
project. No runtime deletion is part of Stage 1.
Persistent disks remain billable while stopped. Verify the Console's final
status rather than assuming an idle timeout has triggered.

Commit only reviewed code/docs and small real results. `.gitignore` excludes
weights, credentials, local HF/LIBERO cache and bulk videos, but allows the
recorded formal and smoke MP4s used in this baseline. Review console logs for
credentials before adding them. No placeholder MP4 should be committed.

## Known issues and targeted fixes

- Use the official `PreTrainedConfig.from_pretrained` factory for the saved
  checkpoint's `type` discriminator. Direct `SmolVLAConfig.from_pretrained`
  failed before any robot action; the factory resolves the correct config class.
- Colab exports an inline Matplotlib backend that the isolated venv does not
  contain. The evaluator sets `MPLBACKEND=Agg` before robotics imports.
- If normal venv creation fails during `ensurepip`, use the documented
  `--without-pip` and existing Python's `pip --python` bootstrap.
- Warnings about missing robosuite private macros or `OpenGL_accelerate` did not
  prevent the recorded EGL rollout. Diagnose an actual traceback before changing
  packages. Preserve unsuccessful startup logs separately from episode metrics.
- Delivered `results/` already contains completed episodes. Reproduce into a
  fresh directory as shown in README.md; do not delete genuine results to rerun.

## Primary references inspected

- [LeRobot v0.5.1 dependency metadata](https://github.com/huggingface/lerobot/blob/v0.5.1/pyproject.toml)
- [LeRobot LIBERO wrapper](https://github.com/huggingface/lerobot/blob/v0.5.1/src/lerobot/envs/libero.py)
- [LIBERO observation processor](https://github.com/huggingface/lerobot/blob/v0.5.1/src/lerobot/processor/env_processor.py)
- [Official policy evaluation loop](https://github.com/huggingface/lerobot/blob/v0.5.1/src/lerobot/scripts/lerobot_eval.py)
- [HF LIBERO metadata](https://github.com/huggingface/LIBERO/blob/main/pyproject.toml)
- [LIBERO task ordering](https://github.com/huggingface/LIBERO/blob/main/libero/libero/benchmark/libero_suite_task_map.py)
- [Checkpoint configuration](https://huggingface.co/HuggingFaceVLA/smolvla_libero/blob/6721902bc4d61e50a3bfdb11dfb4cb626f05d102/config.json)
- [PyTorch official wheel combinations](https://pytorch.org/get-started/previous-versions/)
- [Colab Enterprise pricing](https://cloud.google.com/colab/pricing)
- [Runtime/disk billing behavior](https://docs.cloud.google.com/colab/docs/runtimes)
